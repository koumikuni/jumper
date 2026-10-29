"""The board's firmware: a balance loop that knows nothing about the rider.

Runs as a `step` event, once per policy step (50 Hz), and writes the two wheels'
velocity targets; MuJoCo's velocity actuators close the inner loop at the physics
rate. That split is the prototype's (`lab/segway_proto.py`), which was tuned with
exactly this 20 ms update.

    a      = kp * pitch + kd * pitch_rate + kv * speed
    v_cmd += a * dt                       (clipped to +/- v_max)
    yaw    = -k_steer * deck_roll         (clipped to +/- yaw_max)
    wheel  = (v_cmd -/+ yaw * track / 2) / r

**`kv` is positive, and that is the counter-intuitive part.** Steady state needs
`a = 0`, so `kp * pitch = -kv * speed`: to hold a forward speed the board must sit
nose-up, which is what keeps the rider's forward lean balanced over the axle.
Without it a lean has no speed to settle at: of the prototype's 100 gain sets,
the 14 that stayed upright at a 5 ms step all had `kv` of 2 or more.

The only inputs are the board's own: chassis attitude and rate (an IMU), wheel
speed (the motor encoders) and deck roll (an encoder on the roll hinge). Nothing
about the robot is read -- except in idle mode, below.

## Idle mode, for getting on

**The balance loop cannot hold an empty board.** Measured: a board with nobody on
it, given a one-tenth-second press on its front edge, drives off and keeps
accelerating -- 8.7 m and 1.34 m/s ten seconds later, pitched 15 degrees. The loop
is tuned for a rider's mass over the axle; without one, `kv`'s positive feedback
has nothing to balance against.

So with `idle_mode` on (jumper.segway_board) the board starts **parked**: wheels
held at zero speed, as a real board does before anyone stands on it. It engages
the loop once `engage_feet` of the robot's feet are on the deck *and* the deck is
within `engage_pitch` of level, and parks again when fewer than `release_feet`
remain. The foot count stands in for the load cells under a real board's foot
pads; it is the one thing in this file that looks at the robot, and only in this
mode. jumper.segway leaves it off, so its policy replays exactly as trained.

Parked is not rigid: the wheels are held relative to the chassis, so the board
still rocks on its axle like a seesaw when a foot lands on one end.

## Kneeling

A board built with a `deck_pitch` hinge (`vehicle.DESIGNS["kneel"]`) kneels while
parked: the deck tips forward to `kneel_deg` so its front edge rests on the
ground as a ramp. Once `engage_feet` feet are on it, the deck is brought back
level -- lifting the robot with it -- and the balance loop engages only when the
hinge is back within `KNEEL_LEVEL` of zero. Riding, the hinge is held at zero.

## The ramp flap

A board with a `ramp_flap` (`vehicle.SegwayDesign.flap_len`) holds it down at
`flap_down` while parked and swings it up to `flap_up` when the loop engages.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from ..vehicle import DESIGN, FIRMWARE
from . import state

#: A foot counts as on the deck within this band of its top face, in metres.
#: Measured on the robot standing on the deck: the six foot origins sit -3.5 to
#: +1.2 mm from it.
FOOT_ON_DECK_Z = (-0.015, 0.03)

#: The kneel hinge counts as level within this, in radians (about 3 degrees).
KNEEL_LEVEL = 0.05


#: The deck outline the feet are counted against; a board task with another
#: design sets it (`segway_board/env_cfg.py`). Rectangles in the deck's frame.
DECK_RECTS = DESIGN.deck_rects


def feet_on_deck_count(env: Any, name: str = state.ENTITY) -> torch.Tensor:
    """How many of the six feet are on the deck. Shape (N,), float.

    On means over one of the deck's plates (`DECK_RECTS`) and within
    `FOOT_ON_DECK_Z` of its top. A ramp's top is measured as level: a foot on
    the low end of one is not yet on.
    """
    local = state.feet_on_deck(env, name=name)
    z = local[..., 2] - DESIGN.deck_t / 2
    level = (z > FOOT_ON_DECK_Z[0]) & (z < FOOT_ON_DECK_Z[1])
    over = torch.zeros_like(level)
    for r in DECK_RECTS:
        over |= ((local[..., 0] > r[0] - 0.01) & (local[..., 0] < r[1] + 0.01)
                 & (local[..., 1] > r[2] - 0.01) & (local[..., 1] < r[3] + 0.01))
    return (level & over).float().sum(-1)


class firmware:
    """Event term, mode `step`. Holds `v_cmd`, the integrated speed command."""

    def __init__(self, cfg, env) -> None:
        self._env = env
        self.v_cmd = torch.zeros(env.num_envs, device=env.device)
        entity = env.scene[cfg.params.get("entity_name", state.ENTITY)]
        self._wheel_ids = torch.tensor(
            [entity.joint_names.index("wheel_l"), entity.joint_names.index("wheel_r")],
            device=env.device,
        )
        self._entity = entity
        self.idle_mode = bool(cfg.params.get("idle_mode", False))
        self.engaged = torch.full((env.num_envs,), not self.idle_mode, device=env.device)
        names = entity.joint_names
        self._kneel_id = (torch.tensor([names.index("deck_pitch")], device=env.device)
                          if "deck_pitch" in names else None)
        self._flap_id = (torch.tensor([names.index("ramp_flap")], device=env.device)
                         if "ramp_flap" in names else None)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.v_cmd[env_ids] = 0.0
        self.engaged[env_ids] = not self.idle_mode
        zero = torch.zeros_like(self.v_cmd[env_ids]).unsqueeze(-1).expand(-1, 2)
        self._entity.set_joint_velocity_target(zero, joint_ids=self._wheel_ids, env_ids=env_ids)

    def __call__(self, env: Any, env_ids, entity_name: str = state.ENTITY,
                 idle_mode: bool = False, engage_feet: int = 4, release_feet: int = 2,
                 engage_pitch: float = 0.17, kneel_deg: float = 0.0,
                 flap_down: float = 0.0, flap_up: float = 0.0) -> None:
        del env_ids  # a step event always covers every environment
        c = FIRMWARE
        pitch = state.pitch(env, entity_name)
        if idle_mode:
            feet = feet_on_deck_count(env, entity_name)
            engage = (feet >= engage_feet) & (pitch.abs() < engage_pitch)
            if self._kneel_id is not None:
                hinge = self._entity.data.joint_pos[:, self._kneel_id[0]]
                engage = engage & (hinge.abs() < KNEEL_LEVEL)
                kneeling = ~self.engaged & (feet < engage_feet)
                target = kneeling.float() * math.radians(kneel_deg)
                self._entity.set_joint_position_target(target.unsqueeze(-1), joint_ids=self._kneel_id)
            self.engaged = (self.engaged | engage) & (feet >= release_feet)
        if self._flap_id is not None:
            # The ramp flap: down while parked, stowed once the loop runs.
            target = torch.where(self.engaged, torch.full_like(pitch, flap_up),
                                 torch.full_like(pitch, flap_down))
            self._entity.set_joint_position_target(target.unsqueeze(-1), joint_ids=self._flap_id)
        acc = (
            c.kp * pitch
            + c.kd * state.pitch_rate(env, entity_name)
            + c.kv * state.speed(env, entity_name)
        )
        self.v_cmd = (self.v_cmd + acc * env.step_dt).clamp(-c.v_max, c.v_max)
        yaw = (-c.k_steer * state.deck_roll(env, entity_name)).clamp(-c.yaw_max, c.yaw_max)
        on = self.engaged.float()
        self.v_cmd = self.v_cmd * on
        yaw = yaw * on
        half = DESIGN.track / 2
        target = torch.stack(
            [(self.v_cmd - yaw * half), (self.v_cmd + yaw * half)], dim=-1
        ) / DESIGN.wheel_r
        self._entity.set_joint_velocity_target(target, joint_ids=self._wheel_ids)


def term(env: Any) -> firmware:
    return env.event_manager.get_term_cfg("segway_firmware").func


def v_cmd(env: Any) -> torch.Tensor:
    """The firmware's current speed command, for the critic and the replay line."""
    return term(env).v_cmd


def engaged(env: Any) -> torch.Tensor:
    """Whether the balance loop is running (always, outside idle mode). Shape (N,)."""
    return term(env).engaged
