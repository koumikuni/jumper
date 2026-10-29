"""What the critic is told about the board. The actor is told none of it.

The actor rides on proprioception and the ride command: the board's pitch and
turn already reach it through its own IMU, because it stands on the deck.
These terms are the privileged half of an asymmetric critic.

`mirror_kind` is accepted and discarded by every function, the convention
`swing/mdp/observations.py` explains: `symmetry.py` reads it off `params`, and
`params` are handed to the function as keywords too.
"""

from __future__ import annotations

from typing import Any

import torch

from . import firmware, state


def ride_command(env: Any, command_name: str = "ride", mirror_kind: str | None = None) -> torch.Tensor:
    """(v*, 0, w*) -- the same layout as a twist, so it mirrors as one."""
    return env.command_manager.get_command(command_name)


def board_gravity(env: Any, mirror_kind: str | None = None) -> torch.Tensor:
    """Gravity in the chassis frame: the board's pitch and roll in one vector."""
    return torch.einsum("nij,j->ni", state.chassis_axes(env).transpose(1, 2),
                        torch.tensor([0.0, 0.0, -1.0], device=env.device))


def board_lin_vel(env: Any, mirror_kind: str | None = None) -> torch.Tensor:
    return env.scene[state.ENTITY].data.root_link_lin_vel_b


def board_ang_vel(env: Any, mirror_kind: str | None = None) -> torch.Tensor:
    return env.scene[state.ENTITY].data.root_link_ang_vel_b


def board_speed_command(env: Any, mirror_kind: str | None = None) -> torch.Tensor:
    """The firmware's integrated speed command, (N, 1)."""
    return firmware.v_cmd(env).unsqueeze(-1)


def deck_up(env: Any, mirror_kind: str | None = None) -> torch.Tensor:
    return state.deck_up_in_body(env)


def deck_offset(env: Any, mirror_kind: str | None = None) -> torch.Tensor:
    return state.robot_on_deck(env)
