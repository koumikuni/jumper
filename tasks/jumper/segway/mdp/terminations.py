"""When the ride is over: off the deck, over on the deck, or the board itself over."""

from __future__ import annotations

from typing import Any

import torch

from ..vehicle import DESIGN
from . import state


def off_the_deck(env: Any, margin: float, drop: float) -> torch.Tensor:
    """The base has left the deck sideways, or dropped below its top face."""
    local = state.robot_on_deck(env)
    outside = (local[:, 0].abs() > DESIGN.deck_x / 2 + margin) | (
        local[:, 1].abs() > DESIGN.deck_y / 2 + margin
    )
    return outside | (local[:, 2] < -drop)


def tipped_on_the_deck(env: Any, limit_angle: float) -> torch.Tensor:
    """The robot has gone over relative to the deck, not to gravity."""
    tilt = state.deck_up_in_body(env)[:, 2].clamp(-1.0, 1.0)
    return torch.arccos(tilt) > limit_angle


def board_over(env: Any, limit_angle: float) -> torch.Tensor:
    """The board has nosedived, tail-struck or rolled past `limit_angle`.

    Needed beside the two above for the swing's reason: a robot glued square to a
    board lying on its nose reads as perfectly upright and perfectly centred.
    """
    return state.tilt(env) > limit_angle
