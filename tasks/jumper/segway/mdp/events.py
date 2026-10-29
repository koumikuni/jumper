"""Starting an episode: the board at rest, the robot standing on its deck.

Replaces `reset_base` for the same reason the swing's reset does: mjlab's version
scatters the robot half a metre and a full turn, which is a robot beside the
board rather than on it. Board and robot are placed together, from the
environment origin and the design constants -- not read back from `data`, which
still describes the previous episode at this point.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from ..vehicle import DESIGNS
from . import state


def reset_on_the_board(
    env: Any,
    env_ids: torch.Tensor | None,
    yaw_range: tuple[float, float] = (-math.pi, math.pi),
    design: str = "baseline",
    robot_name: str = "robot",
    entity_name: str = state.ENTITY,
) -> None:
    from mjlab.envs.mdp.events import resolve_env_ids

    env_ids = resolve_env_ids(env, env_ids)
    DESIGN = DESIGNS[design]
    board = env.scene[entity_name]
    robot = env.scene[robot_name]
    device = env.device
    n = len(env_ids)

    low, high = yaw_range
    yaw = low + (high - low) * torch.rand(n, device=device)
    cos, sin = torch.cos(yaw), torch.sin(yaw)
    quat = torch.stack(
        [torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)],
        dim=-1,
    )

    origin = env.scene.env_origins[env_ids]
    root = torch.zeros((n, 13), device=device)
    root[:, 0:3] = origin + torch.tensor([0.0, 0.0, DESIGN.wheel_r], device=device)
    root[:, 3:7] = quat
    board.write_root_state_to_sim(root, env_ids=env_ids)
    q = torch.zeros((n, len(board.joint_names)), device=device)
    if "ramp_flap" in board.joint_names:
        q[:, board.joint_names.index("ramp_flap")] = DESIGN.flap_up_rad  # riding: stowed
    board.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)

    ox, oy, oz = DESIGN.robot_offset
    pos = root[:, 0:3].clone()
    pos[:, 0] += cos * ox - sin * oy
    pos[:, 1] += sin * ox + cos * oy
    pos[:, 2] += oz
    robot_root = torch.zeros((n, 13), device=device)
    robot_root[:, 0:3] = pos
    robot_root[:, 3:7] = quat
    robot.write_root_state_to_sim(robot_root, env_ids=env_ids)
