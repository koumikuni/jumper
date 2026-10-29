"""Environment config for jumper.segway_board.

Built like `segway/env_cfg.py`: `velocity_env_cfg` with the walking command
taken out, and the board installed by the task. Two differences that matter:

- **The board starts parked** (`firmware` with `idle_mode`): an empty board
  under the balance loop drives itself away from the first foot that touches it,
  so getting on would be chasing it.
- **The actor is told where the board is**: the boarded spot and the deck's
  heading and tilt in its own frame. A robot that had to find the board by
  touch would be a different task; this is what its camera or dToF would give it.

`upright` and `fell_over` are kept, against gravity, because the robot spends
most of an episode on the ground; on the deck the board is parked and level, or
balancing it with the deck near level, so gravity is still the right reference.
"""

from __future__ import annotations

import math
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from ..common.constants import GAIT_JOINTS
from ..common.mdp.curriculum import STD_ANG_RATIO, STD_LIN_RATIO
from ..common.velocity_env import _ARM_JOINTS, _LEG_JOINTS, velocity_env_cfg
from ..segway.mdp import firmware as segway_firmware
from ..segway.mdp import rewards as segway_rewards
from ..segway.mdp import state as segway_state
from ..segway.mdp import terminations as segway_terms
from . import mdp

_WALKING_REWARDS = (
    "track_linear_velocity",
    "track_angular_velocity",
    "pose",
    "air_time",
    "foot_clearance",
    "foot_swing_height",
    "foot_slip",
    "soft_landing",
    "standing_sway",
    "body_tilt_rate",
)


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    from ..segway.vehicle import segway_entity

    design_name, design = mdp.active_design()
    print(f"[segway_board] board design: {design_name}")
    segway_firmware.DECK_RECTS = design.deck_rects

    cfg = velocity_env_cfg(
        asset=asset,
        play=play,
        teleop=False,
        gamepad=False,
        operator_command=False,
        standing_sway_weight=0.0,
        # As in `segway/env_cfg.py`: required by the skeleton, and except
        # `action_rate_weight` configuring terms removed below.
        action_rate_weight=-0.1,
        pose_weight=1.0,
        foot_target_height=0.025,
        command_lin_ceiling=0.8,
        command_ang_ceiling=1.0,
        command_lin_std_ratio=STD_LIN_RATIO,
        command_ang_std_ratio=STD_ANG_RATIO,
    )

    cfg.scene.entities[segway_state.ENTITY] = segway_entity(design)
    # The skeleton's 128 is a robot on flat ground. Measured: the low-floor
    # board, with the robot clambering over a deck 4.5 cm off the ground,
    # needed 145 contacts in the first iterations of training and overflowed.
    cfg.sim.nconmax = 192
    # And its constraint rows: 586 measured at the same point, against the
    # skeleton's 512.
    cfg.sim.njmax = 768
    cfg.events["segway_firmware"] = EventTermCfg(
        func=segway_firmware.firmware, mode="step",
        params={"idle_mode": True, "kneel_deg": design.kneel_deg,
                "flap_down": design.flap_down_rad, "flap_up": design.flap_up_rad},
    )

    cfg.commands.pop("twist", None)
    cfg.curriculum = {}
    for name in _WALKING_REWARDS:
        cfg.rewards.pop(name, None)
    for group in cfg.observations.values():
        group.terms.pop("command", None)

    cfg.events.pop("reset_base", None)
    cfg.events["reset_beside_the_board"] = EventTermCfg(
        func=mdp.reset_beside_the_board, mode="reset",
        # A replay always starts on the ground: getting on is what it is for.
        params={"on_deck_share": 0.0} if play else {},
    )
    if "push_robot" in cfg.events:
        ranges = cfg.events["push_robot"].params["velocity_range"]
        for key, (lo, hi) in list(ranges.items()):
            ranges[key] = (lo / 2.0, hi / 2.0)

    # ── What it is paid for ──────────────────────────────────────────────
    # A ladder: be near (1.0), get feet up (up to 2.0), be on and square
    # (up to 4.0 more). Each rung is worth more than the one below, so the
    # policy is never paid more for hovering beside the board than for trying.
    cfg.rewards["approach"] = RewardTermCfg(func=mdp.approach, weight=1.0, params={"std": 0.35})
    cfg.rewards["feet_on_the_deck"] = RewardTermCfg(func=mdp.feet_on_the_deck, weight=2.0, params={})
    cfg.rewards["standing_on_the_board"] = RewardTermCfg(
        func=mdp.standing_on_the_board, weight=4.0, params={}
    )
    cfg.rewards["board_pushed"] = RewardTermCfg(func=mdp.board_pushed, weight=-0.5, params={})
    # **No charge for rocking the board.** The first run had one, at -0.5, and
    # it is what taught the policy to leave the board alone: rocking is what any
    # attempt on a parked seesaw does, and the climb that would have paid for it
    # was never found. The rocking is the board's fault, and it is still
    # measured -- `face.BoardingMood` gets angry about it -- just not charged.
    cfg.rewards["under_the_deck"] = RewardTermCfg(func=mdp.under_the_deck, weight=-1.0, params={})
    cfg.rewards["pose"] = RewardTermCfg(
        func=segway_rewards.posture,
        weight=0.1,
        params={
            # Wider than riding: climbing a 14 cm step is big joint travel.
            "std": {_ARM_JOINTS: 0.8, _LEG_JOINTS: 0.5},
            "asset_cfg": SceneEntityCfg("robot", joint_names=GAIT_JOINTS),
        },
    )

    # ── When it is over ──────────────────────────────────────────────────
    # 50 degrees, not the riding task's 40: a parked board rocked onto one end
    # rests at about 32 degrees with its deck's edge on the ground, and that is
    # a board being climbed, not one that has gone over.
    cfg.terminations["board_over"] = TerminationTermCfg(
        func=segway_terms.board_over, params={"limit_angle": math.radians(50.0)}
    )
    cfg.terminations["wandered_off"] = TerminationTermCfg(
        func=mdp.wandered_off, params={"limit": 1.5}
    )

    # ── What it sees ─────────────────────────────────────────────────────
    for group in cfg.observations.values():
        for name, func, kind in (
            ("board_in_body", mdp.board_in_body, "vec3"),
            ("deck_heading_in_body", mdp.deck_heading_in_body, "vec3"),
            ("deck_up", mdp.deck_up, "vec3"),
        ):
            group.terms[name] = ObservationTermCfg(
                func=func, history_length=1, params={"mirror_kind": kind}
            )
    for name, func in (("feet_on_deck", mdp.feet_on_deck), ("board_engaged", mdp.board_engaged)):
        cfg.observations["critic"].terms[name] = ObservationTermCfg(
            func=func, history_length=1, params={"mirror_kind": "scalar"}
        )

    if play:
        from ..segway.face import add_face_screen

        robot = cfg.scene.entities["robot"]
        build_robot = robot.spec_fn
        robot.spec_fn = lambda: add_face_screen(build_robot())
    else:
        cfg.episode_length_s = 12.0
    cfg.viewer.distance = 2.0
    cfg.viewer.elevation = -20.0
    return cfg
