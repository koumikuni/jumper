"""What the rider is paid for: going where it is asked, on the board, square to it.

The objective is the **board's** motion against the ride command, not the robot's
-- the robot is carried, and a robot that walked forward along a stationary deck
would otherwise read as riding.
"""

from __future__ import annotations

from typing import Any

import torch

from . import state

#: The swing's fixed-width `pose`, reused: same formula, same key, same reason --
#: there is no gait here for `variable_posture` to switch its width on.
from ...swing.mdp.rewards import posture  # noqa: F401


def track_ride_speed(env: Any, std: float, command_name: str = "ride") -> torch.Tensor:
    """`exp(-(v - v*)^2 / std^2)` on the board's ground speed along its heading."""
    target = env.command_manager.get_command(command_name)[:, 0]
    return torch.exp(-(state.ground_speed(env) - target).pow(2) / std**2)


def track_ride_yaw(env: Any, std: float, command_name: str = "ride") -> torch.Tensor:
    """`exp(-(w - w*)^2 / std^2)` on the board's world yaw rate."""
    target = env.command_manager.get_command(command_name)[:, 2]
    return torch.exp(-(state.yaw_rate(env) - target).pow(2) / std**2)


def deck_upright(env: Any, deadband: float, std: float) -> torch.Tensor:
    """Square to the deck within `deadband` is full marks; Gaussian beyond.

    As the swing's: leaning is how a rider moves its centre of mass, so the band
    leaves the lean free and the falloff is a rail, not a posture demand.
    """
    tilt = torch.arccos(state.deck_up_in_body(env)[:, 2].clamp(-1.0, 1.0))
    return torch.exp(-((tilt - deadband).clamp(min=0.0) / std) ** 2)


def centred_on_the_deck(env: Any, std: float) -> torch.Tensor:
    """Stay near where the reset stood the robot, in deck coordinates.

    Measured from the stance offset rather than the deck centre, for the swing's
    reason: it is the feet that must be centred, and the base sits behind them.
    """
    from ..vehicle import DESIGN

    local = state.robot_on_deck(env)
    fore_aft = local[:, 0] - DESIGN.robot_offset[0]
    return torch.exp(-(fore_aft.pow(2) + local[:, 1].pow(2)) / std**2)


def feet_off_the_deck(env: Any) -> torch.Tensor:
    """One count per foot outside the deck's outline."""
    from ..vehicle import DESIGN

    local = state.feet_on_deck(env)
    outside = (local[..., 0].abs() > DESIGN.deck_x / 2) | (local[..., 1].abs() > DESIGN.deck_y / 2)
    return outside.float().sum(-1)


def feet_slip_on_the_deck(env: Any) -> torch.Tensor:
    """Tangential foot travel across the deck, summed over the six feet."""
    return state.feet_slip_on_deck(env)[..., :2].pow(2).sum(-1).sum(-1)


def body_tilt_rate_on_the_deck(env: Any, deadband: float) -> torch.Tensor:
    """Roll and pitch rate relative to the deck, above a deadband."""
    rate = state.body_rate_on_deck(env)
    return (rate[:, 0].pow(2) + rate[:, 1].pow(2) - deadband).clamp(min=0.0)


def board_pitch(env: Any) -> torch.Tensor:
    """The board's pitch squared.

    The firmware rides nose-up to hold a speed, so some pitch is the vehicle doing
    its job. What this charges is the large pitch on the way to a nosedive or a
    tail-strike, gently enough that it does not argue with the speed command.
    """
    return state.pitch(env).pow(2)
