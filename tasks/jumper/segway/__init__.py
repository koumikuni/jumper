"""jumper.segway -- ride a two-wheeled self-balancing board by leaning.

The robot stands on the deck of a segway-style board (`vehicle.py`) and is given
a ride command: a forward speed and a yaw rate. It has no handlebar and no
throttle. The board's own firmware balances it and drives under its tilt, so
leaning forward goes, leaning back stops, and pressing one side of the sprung
deck down turns. The robot is paid for the **board's** motion against the
command, and has to learn to steer a vehicle with nothing but where it puts its
weight.

This directory holds what belongs to this task: its name (below), the vehicle and
its firmware, the environment config, the hyper-parameters and the MDP terms only
it uses.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.segway",
    assets=JUMPER_ASSETS,
    description="jumper hexapod rides a self-balancing two-wheeled board, steering by leaning",
    tags=("jumper", "segway", "vehicle", "balance", "prop"),
)
