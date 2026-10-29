"""Where the board is, what it is doing, and where the robot is on it.

One place for the geometry every other module needs, as `swing/mdp/state.py` is
for the swing: the reward, the observation, the termination and the firmware all
read the same pitch and the same deck frame, so they cannot drift into measuring
slightly different boards.

The robot is scored against the **deck**, not gravity. The board cruises with its
deck tilted back about 11 degrees at 1 m/s (`vehicle.py`), so a robot standing
square on it is 11 degrees off vertical while riding perfectly.
"""

from __future__ import annotations

from typing import Any

import torch

from ..vehicle import DESIGN

ENTITY = "segway"


def _rows(env: Any, name: str) -> tuple[int, int]:
    names = env.scene[name].body_names
    return names.index("chassis"), names.index("deck")


def chassis_axes(env: Any, name: str = ENTITY) -> torch.Tensor:
    """The chassis rotation matrix, world frame. Shape (N, 3, 3), columns x/y/z."""
    from mjlab.utils.lab_api.math import matrix_from_quat

    return matrix_from_quat(env.scene[name].data.root_link_quat_w)


def deck_axes(env: Any, name: str = ENTITY) -> torch.Tensor:
    from mjlab.utils.lab_api.math import matrix_from_quat

    _, deck = _rows(env, name)
    return matrix_from_quat(env.scene[name].data.body_link_quat_w[:, deck])


def pitch(env: Any, name: str = ENTITY) -> torch.Tensor:
    """Chassis pitch in radians, **nose down positive**. Shape (N,)."""
    x_axis = chassis_axes(env, name)[:, :, 0]
    return torch.atan2(-x_axis[:, 2], torch.hypot(x_axis[:, 0], x_axis[:, 1]))


def pitch_rate(env: Any, name: str = ENTITY) -> torch.Tensor:
    """Rate about the axle, chassis frame; nose down positive like `pitch`."""
    return env.scene[name].data.root_link_ang_vel_b[:, 1]


def tilt(env: Any, name: str = ENTITY) -> torch.Tensor:
    """How far the chassis' up-axis is from vertical, any direction. Shape (N,)."""
    return torch.arccos(chassis_axes(env, name)[:, 2, 2].clamp(-1.0, 1.0))


def wheel_speeds(env: Any, name: str = ENTITY) -> torch.Tensor:
    """(left, right) wheel rates in rad/s, relative to the chassis. Shape (N, 2)."""
    entity = env.scene[name]
    names = entity.joint_names
    ids = [names.index("wheel_l"), names.index("wheel_r")]
    return entity.data.joint_vel[:, ids]


def speed(env: Any, name: str = ENTITY) -> torch.Tensor:
    """Forward speed as the board measures it: mean wheel rate times radius."""
    return wheel_speeds(env, name).mean(-1) * DESIGN.wheel_r


def ground_speed(env: Any, name: str = ENTITY) -> torch.Tensor:
    """Forward speed over the ground, along the chassis' heading. Shape (N,).

    What the rewards track, rather than `speed`: a wheel spinning on the spot
    reads as speed to the board and is not travel.
    """
    axes = chassis_axes(env, name)
    heading = axes[:, :2, 0]
    heading = heading / heading.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    vel = env.scene[name].data.root_link_lin_vel_w[:, :2]
    return (vel * heading).sum(-1)


def yaw_rate(env: Any, name: str = ENTITY) -> torch.Tensor:
    """World-frame yaw rate of the chassis. Shape (N,)."""
    return env.scene[name].data.root_link_ang_vel_w[:, 2]


def deck_roll(env: Any, name: str = ENTITY) -> torch.Tensor:
    entity = env.scene[name]
    return entity.data.joint_pos[:, entity.joint_names.index("deck_roll")]


def robot_on_deck(env: Any, robot_name: str = "robot", name: str = ENTITY) -> torch.Tensor:
    """The robot's base relative to the deck's top-face centre, deck coordinates.

    Shape (N, 3): `x` fore-aft, `y` across, `z` above the top face.
    """
    _, deck = _rows(env, name)
    delta = env.scene[robot_name].data.root_link_pos_w - env.scene[name].data.body_link_pos_w[
        :, deck
    ]
    local = torch.einsum("nij,nj->ni", deck_axes(env, name).transpose(1, 2), delta)
    return local - torch.tensor([0.0, 0.0, DESIGN.deck_t / 2], device=local.device)


def deck_up_in_body(env: Any, robot_name: str = "robot", name: str = ENTITY) -> torch.Tensor:
    """The deck's up-axis in the robot's own frame. `(0, 0, 1)` is square on it."""
    from mjlab.utils.lab_api.math import matrix_from_quat

    body = matrix_from_quat(env.scene[robot_name].data.root_link_quat_w)
    return torch.einsum("nij,nj->ni", body.transpose(1, 2), deck_axes(env, name)[:, :, 2])


def body_rate_on_deck(env: Any, robot_name: str = "robot", name: str = ENTITY) -> torch.Tensor:
    """The robot's angular velocity with the deck's taken out, robot frame."""
    from mjlab.utils.lab_api.math import matrix_from_quat

    _, deck = _rows(env, name)
    robot = env.scene[robot_name]
    deck_w = env.scene[name].data.body_link_ang_vel_w[:, deck]
    body = matrix_from_quat(robot.data.root_link_quat_w)
    deck_b = torch.einsum("nij,nj->ni", body.transpose(1, 2), deck_w)
    return robot.data.root_link_ang_vel_b - deck_b


def feet_on_deck(env: Any, robot_name: str = "robot", name: str = ENTITY) -> torch.Tensor:
    """Each foot's position in deck coordinates. Shape (N, 6, 3)."""
    from ...common.constants import FEET

    robot = env.scene[robot_name]
    _, deck = _rows(env, name)
    ids = [robot.body_names.index(f) for f in FEET]
    delta = robot.data.body_link_pos_w[:, ids] - env.scene[name].data.body_link_pos_w[
        :, deck
    ].unsqueeze(1)
    return torch.einsum("nij,nkj->nki", deck_axes(env, name).transpose(1, 2), delta)


def feet_slip_on_deck(env: Any, robot_name: str = "robot", name: str = ENTITY) -> torch.Tensor:
    """Each foot's velocity relative to the deck material under it, deck frame.

    Shape (N, 6, 3). The same `v_deck + omega x r` subtraction as the swing's
    `feet_slip_on_the_deck`: a foot glued to a board doing 1 m/s is not slipping.
    """
    from ...common.constants import FEET

    robot = env.scene[robot_name]
    entity = env.scene[name]
    _, deck = _rows(env, name)
    ids = [robot.body_names.index(f) for f in FEET]
    arm = robot.data.body_link_pos_w[:, ids] - entity.data.body_link_pos_w[:, deck].unsqueeze(1)
    deck_at_foot = entity.data.body_link_lin_vel_w[:, deck].unsqueeze(1) + torch.cross(
        entity.data.body_link_ang_vel_w[:, deck].unsqueeze(1).expand_as(arm), arm, dim=-1
    )
    relative = robot.data.body_link_lin_vel_w[:, ids] - deck_at_foot
    return torch.einsum("nij,nkj->nki", deck_axes(env, name).transpose(1, 2), relative)
