"""jumper.segway_board -- get onto the segway board, alone, and stand on it.

The robot starts on the ground, in front of or behind the parked board
(`segway/vehicle.py`), facing it. It is paid for getting its six feet onto the
14 cm-high deck and standing there square and centred until the board's
firmware engages and balances it.

Nothing about the board is made easier for this. Parked, the board rocks on its
axle like a seesaw when a foot lands on one end, and once four feet are on and
the deck is level the balance loop takes over with the robot wherever it happens
to be standing. How hard that is, is the measurement: it is the board as designed,
before any change made to help a rider get on.

The board, its firmware and the shared geometry are `jumper.segway`'s; this
directory holds only the reset, the rewards and the observations of getting on.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.segway_board",
    assets=JUMPER_ASSETS,
    description="jumper hexapod climbs onto a parked self-balancing board and stands on it",
    tags=("jumper", "segway", "vehicle", "climb", "prop"),
)
