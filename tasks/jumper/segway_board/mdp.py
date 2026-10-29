"""Reset, rewards, observations and terminations for getting on the board.

All geometry comes from `segway/mdp/state.py`, so the deck this task scores the
feet against is the one the firmware counts them on.
"""

from __future__ import annotations

import math
import os
from typing import Any

import torch

from ..common.constants import STAND_Z
from ..segway.mdp import firmware, state
from ..segway.vehicle import DESIGN, DESIGNS

#: Which board to get onto: a key of `vehicle.DESIGNS`. Read from the
#: environment rather than a flag so train, play and the recorder agree without
#: each growing an argument; the run's log line says which it was.
DESIGN_ENV = "CRAB_BOARD_DESIGN"


def active_design():
    name = os.environ.get(DESIGN_ENV, "baseline").strip() or "baseline"
    if name not in DESIGNS:
        raise ValueError(f"{DESIGN_ENV}={name!r}; expected one of {sorted(DESIGNS)}")
    return name, DESIGNS[name]

# ── Where the robot should end up ────────────────────────────────────────


def _target_w(env: Any) -> torch.Tensor:
    """Where the robot's base stands when boarded, world frame, (N, 3)."""
    board = env.scene[state.ENTITY]
    _, deck = state._rows(env, state.ENTITY)
    deck_pos = board.data.body_link_pos_w[:, deck]
    axes = state.deck_axes(env)
    ox, _, _ = DESIGN.robot_offset
    return deck_pos + axes[:, :, 0] * ox


def distance_to_target(env: Any) -> torch.Tensor:
    """Horizontal distance from the robot's base to its boarded spot, (N,)."""
    delta = env.scene["robot"].data.root_link_pos_w[:, :2] - _target_w(env)[:, :2]
    return delta.norm(dim=-1)


def boarded(env: Any, max_tilt: float = 0.35) -> torch.Tensor:
    """All six feet on the deck and the body square to it, (N,) bool."""
    feet = firmware.feet_on_deck_count(env)
    tilt = torch.arccos(state.deck_up_in_body(env)[:, 2].clamp(-1.0, 1.0))
    return (feet >= 6) & (tilt < max_tilt)


# ── Reset ────────────────────────────────────────────────────────────────


def reset_beside_the_board(
    env: Any,
    env_ids: torch.Tensor | None,
    distance_range: tuple[float, float] = (0.30, 0.55),
    lateral: float = 0.08,
    heading_noise: float = 0.25,
    on_deck_share: float = 0.2,
) -> None:
    """Park the board at the origin, stand the robot on the ground facing it.

    **From the design's own approach side** (`SegwayDesign.approach`): the front
    for a board that kneels forward, the back for a comb that opens backwards. The board's yaw is
    random, so "in front" is never a direction in the world.

    `on_deck_share` of episodes start with the robot already standing on the
    deck instead. The first overnight run never once got a foot on, so it never
    saw what being on is worth; starting some episodes there teaches the critic
    that before the policy can climb, and teaches the policy to stay.
    """
    from mjlab.envs.mdp.events import resolve_env_ids

    env_ids = resolve_env_ids(env, env_ids)
    board = env.scene[state.ENTITY]
    robot = env.scene["robot"]
    device = env.device
    n = len(env_ids)
    origin = env.scene.env_origins[env_ids]

    yaw = (torch.rand(n, device=device) * 2 - 1) * math.pi
    root = torch.zeros((n, 13), device=device)
    root[:, 0:3] = origin + torch.tensor([0.0, 0.0, DESIGN.wheel_r], device=device)
    root[:, 3] = torch.cos(yaw / 2)
    root[:, 6] = torch.sin(yaw / 2)
    board.write_root_state_to_sim(root, env_ids=env_ids)
    _, design = active_design()
    on_deck = torch.rand(n, device=device) < on_deck_share
    q = torch.zeros((n, len(board.joint_names)), device=device)
    if "deck_pitch" in board.joint_names:
        # Parked boards start kneeling, unless the robot starts on them.
        q[:, board.joint_names.index("deck_pitch")] = (~on_deck).float() * math.radians(design.kneel_deg)
    if "ramp_flap" in board.joint_names:
        # Down for getting on, stowed if the robot starts on the deck.
        q[:, board.joint_names.index("ramp_flap")] = torch.where(
            on_deck, torch.full_like(yaw, design.flap_up_rad), torch.full_like(yaw, design.flap_down_rad))
    board.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)

    side = torch.full((n,), float(design.approach), device=device)
    # Distances were chosen against the 45 cm plate, whose ends are 22 cm from
    # the axle; a deck reaching further on the approach side pushes the start
    # back by the difference, so no start overlaps it.
    reach = max(design.approach * x for r in design.deck_rects for x in r[:2])
    extra = max(0.0, reach - DESIGN.deck_x / 2)
    lo, hi = distance_range[0] + extra, distance_range[1] + extra
    dist = lo + (hi - lo) * torch.rand(n, device=device)
    off = (torch.rand(n, device=device) * 2 - 1) * lateral
    cos, sin = torch.cos(yaw), torch.sin(yaw)
    fx, fy = side * dist, off
    pos = root[:, 0:3].clone()
    pos[:, 0] += cos * fx - sin * fy
    pos[:, 1] += sin * fx + cos * fy
    pos[:, 2] = origin[:, 2] + STAND_Z
    face = yaw + torch.where(side > 0, torch.full_like(yaw, math.pi), torch.zeros_like(yaw))
    face = face + (torch.rand(n, device=device) * 2 - 1) * heading_noise
    # On-deck starts: on the boarded spot, facing along the board.
    ox, _, oz = design.robot_offset
    pos[on_deck, 0] = root[on_deck, 0] + cos[on_deck] * ox
    pos[on_deck, 1] = root[on_deck, 1] + sin[on_deck] * ox
    pos[on_deck, 2] = root[on_deck, 2] + oz
    face = torch.where(on_deck, yaw, face)
    # Behind the board, facing it, is facing the way it will ride.
    robot_root = torch.zeros((n, 13), device=device)
    robot_root[:, 0:3] = pos
    robot_root[:, 3] = torch.cos(face / 2)
    robot_root[:, 6] = torch.sin(face / 2)
    robot.write_root_state_to_sim(robot_root, env_ids=env_ids)


# ── Rewards ──────────────────────────────────────────────────────────────


def under_the_deck(env: Any) -> torch.Tensor:
    """1 while the robot's base is under the deck's outline and below its top.

    **The first overnight run's answer to a 14 cm step**: crouch and push its
    head under the deck, which is as close to the boarded spot as being on it,
    measured horizontally, and nothing like boarding.
    """
    local = state.robot_on_deck(env)
    over = (local[:, 0].abs() < DESIGN.deck_x / 2 + 0.05) & (local[:, 1].abs() < DESIGN.deck_y / 2)
    return (over & (local[:, 2] < 0.0)).float()


def approach(env: Any, std: float) -> torch.Tensor:
    """`exp(-d^2 / std^2)` on the distance to the boarded spot -- and nothing
    from under the deck, so hiding there is not a way to be close."""
    near = torch.exp(-distance_to_target(env).pow(2) / std**2)
    return near * (1.0 - under_the_deck(env))


def feet_on_the_deck(env: Any) -> torch.Tensor:
    """The share of the six feet on the deck, 0..1."""
    return firmware.feet_on_deck_count(env) / 6.0


def standing_on_the_board(env: Any) -> torch.Tensor:
    """1 while boarded, scaled by how centred the robot is on the deck."""
    local = state.robot_on_deck(env)
    off = (local[:, 0] - DESIGN.robot_offset[0]).pow(2) + local[:, 1].pow(2)
    return boarded(env).float() * (0.5 + 0.5 * torch.exp(-off / 0.08**2))


def board_pushed(env: Any) -> torch.Tensor:
    """The board's horizontal speed squared: getting on should not shove it away."""
    return env.scene[state.ENTITY].data.root_link_lin_vel_w[:, :2].pow(2).sum(-1)


def board_rocked(env: Any) -> torch.Tensor:
    """The board's tilt squared: the seesaw a foot on one end makes."""
    return state.tilt(env).pow(2)


# ── Observations (actor and critic) ──────────────────────────────────────


def board_in_body(env: Any, mirror_kind: str | None = None) -> torch.Tensor:
    """The boarded spot, in the robot's own frame: where to go. What a camera or
    the dToF would tell a real robot about a board in front of it."""
    from mjlab.utils.lab_api.math import matrix_from_quat

    robot = env.scene["robot"]
    body = matrix_from_quat(robot.data.root_link_quat_w)
    delta = _target_w(env) - robot.data.root_link_pos_w
    return torch.einsum("nij,nj->ni", body.transpose(1, 2), delta)


def deck_heading_in_body(env: Any, mirror_kind: str | None = None) -> torch.Tensor:
    """The deck's long axis in the robot's frame: which way the board lies."""
    from mjlab.utils.lab_api.math import matrix_from_quat

    body = matrix_from_quat(env.scene["robot"].data.root_link_quat_w)
    return torch.einsum("nij,nj->ni", body.transpose(1, 2), state.deck_axes(env)[:, :, 0])


def deck_up(env: Any, mirror_kind: str | None = None) -> torch.Tensor:
    return state.deck_up_in_body(env)


def feet_on_deck(env: Any, mirror_kind: str | None = None) -> torch.Tensor:
    return (firmware.feet_on_deck_count(env) / 6.0).unsqueeze(-1)


def board_engaged(env: Any, mirror_kind: str | None = None) -> torch.Tensor:
    return firmware.engaged(env).float().unsqueeze(-1)


# ── Terminations ─────────────────────────────────────────────────────────


def wandered_off(env: Any, limit: float) -> torch.Tensor:
    return distance_to_target(env) > limit
