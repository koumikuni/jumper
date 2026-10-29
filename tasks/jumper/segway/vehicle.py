"""The segway board: a two-wheeled, self-balancing deck the robot rides by leaning.

**The robot drives it with its body and nothing else.** There is no handlebar,
no throttle and no observation of the vehicle in the actor. The board carries its
own firmware (`mdp/firmware.py`) which reads only what a real board can sense
about itself -- its pitch, its pitch rate, its wheel speed and how far its deck
has rolled -- and turns those into wheel speeds:

    fore-aft   the classic inverted-pendulum loop. Lean forward and the board
               tilts, the firmware accelerates under the tilt, and the speed
               settles where the lean is balanced. Lean back to stop.
    turning    the deck sits on a **sprung roll hinge** along the direction of
               travel. Press one side down and the deck rolls; the firmware turns
               the roll into a yaw rate by driving the wheels differentially.

So both controls are physical. The roll hinge is a real mechanism rather than a
load-cell estimate of where the rider is standing, which is what makes "how stiff
should the deck be" a vehicle design question the improvement videos can answer.

## Prototyped before it was a task

Everything here was first tuned in `lab/segway_proto.py`, outside the repository,
with the robot stood in for by a 1.8 kg box on two servoed slides. A 100-point
gain scan over (kp, kd, kv) kept 14 sets upright at the task's 5 ms physics step
(16 at 2 ms) through a push, a 4 cm lean and a lean-and-turn -- every one of them
with `kv` of 2 or more; `FIRMWARE` is the one picked from them. Measured there, on
the box rider, at 2 ms:

    rider centred, 6 N push      pitch stays within 0.14 deg, speed returns to 0
    rider 4 cm forward           settles at 0.97 m/s -- about twice the tripod's
                                 walking ceiling
    rider 4 cm fwd + 5 cm side   1.5 rad/s of yaw, deck on its 12 deg roll stop

Two behaviours from that prototype are worth knowing before reading a replay:

- **Cruising tilts the deck back.** At 1 m/s the board rides about 11 degrees
  nose-up: the speed loop needs a standing tilt to hold the speed against the
  rider's forward lean. A rider standing square to the deck is therefore leaning
  back relative to gravity at speed.
- **Saturating the speed limit nosedives.** With `v_max` reached the firmware
  can no longer accelerate under the tilt, and the board pitches forward and
  goes over -- the failure real electric boards are known for. The termination
  catches it; it is a property of the vehicle, not a bug in the task.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from ..common.constants import STAND_Z

#: The robot's foot rectangle at HOME, from `scenes/swing.py`: the deck is sized
#: by the same rule as the swing's plank, the stance plus a margin on every side.
STANCE_X = 0.1995 - (-0.1460)
STANCE_Y = 0.2050 - (-0.2047)
STANCE_CENTRE_X = (0.1995 + (-0.1460)) / 2
FOOT_MARGIN = 0.05


@dataclass(frozen=True)
class SegwayDesign:
    """Every number that makes this board this board.

    The improvement loop changes these and nothing else, so a design is one
    object that can be printed beside a run's results.
    """

    wheel_r: float = 0.09
    #: Wheel centre to wheel centre. Wide enough that the wheels clear the deck
    #: and the robot's hips.
    track: float = 0.68
    wheel_w: float = 0.04
    deck_x: float = STANCE_X + 2 * FOOT_MARGIN
    deck_y: float = STANCE_Y + 2 * FOOT_MARGIN
    deck_t: float = 0.02
    #: Axle to the underside of the deck.
    deck_above_axle: float = 0.03
    #: The battery hangs under the axle: lower is steadier and costs ground
    #: clearance.
    battery_below_axle: float = 0.04
    m_chassis: float = 0.4
    m_battery: float = 0.6
    m_deck: float = 0.4
    m_wheel: float = 0.25
    #: N m / rad. At 6.0 a 5 cm sideways shift of the 1.8 kg rider puts the deck
    #: on its stop, which the prototype showed is softer than it should be.
    roll_stiffness: float = 6.0
    roll_damping: float = 0.15
    roll_limit_deg: float = 12.0
    #: Velocity actuator gain, N m per rad/s of wheel speed error.
    wheel_kv: float = 2.0
    wheel_torque: float = 4.0
    #: Hub-motor rotor inertia reflected to the wheel.
    wheel_armature: float = 0.002
    wheel_friction: float = 1.0
    #: Kneeling: while parked, a hinge between chassis and deck tips the deck
    #: forward this far, so its front edge rests on the ground as a ramp. 0 means
    #: no hinge at all -- the deck is fixed in pitch, as the riding task trained.
    kneel_deg: float = 0.0
    kneel_kp: float = 40.0
    kneel_kd: float = 2.0
    kneel_torque: float = 8.0
    #: A ramp flap hinged on the deck's back edge, this long; 0 for none. Parked
    #: it is lowered until its tip rests `flap_lip` above the ground; once the
    #: balance loop engages it swings up to `flap_stow_deg`, clear of the road --
    #: a ramp that is only there while getting on (2026-10-23: a fixed ramp
    #: reaching 5 mm off the ground could not drive over anything).
    flap_len: float = 0.0
    flap_lip: float = 0.005
    flap_stow_deg: float = 80.0
    flap_kp: float = 30.0
    flap_kd: float = 1.0
    flap_torque: float = 6.0
    #: Reflected inertia of the flap's gearmotor. **Not optional at high kp**:
    #: the plate alone is about 0.008 kg m^2, so at kp 400 it rings at 220 rad/s,
    #: past what the 5 ms step integrates -- the locked flap shook and walked
    #: the parked board up to 47 cm in 3 s with nobody near it (2026-10-23).
    flap_armature: float = 0.0
    #: A solid axle joining the wheels across the board. False is two hub
    #: motors on stub axles: the wheels hang off the chassis and nothing spans
    #: the deck -- which a deck dropped below the axle needs, or the axle bar
    #: crosses the deck at the robot's chest (measured: lowfloor got two feet on
    #: in 96% of tries and never four, stopped by the bar).
    through_axle: bool = True
    #: The deck's outline as rectangles (x0, x1, y0, y1) in the deck's frame, x
    #: towards the front, where a rider gets on. None is the full deck_x by
    #: deck_y plate. A comb -- a spine at the back and tines reaching forward --
    #: lets the robot walk in with its legs in the slots (`comb_deck`).
    deck_shape: tuple[tuple[float, float, float, float], ...] | None = None
    #: Batteries in the wheel hubs instead of a pack under the axle, for a deck
    #: too low to leave room under it. Their mass goes to the wheels.
    battery_in_hubs: bool = False
    #: Which end a rider gets on from: +1 the front, -1 the back. A kneeling
    #: deck kneels forward, so it is boarded from the front; a comb opens to the
    #: back, so the robot walks in facing the way it will ride.
    approach: int = 1

    @property
    def deck_rects(self) -> tuple[tuple[float, float, float, float], ...]:
        if self.deck_shape is not None:
            return self.deck_shape
        hx, hy = self.deck_x / 2, self.deck_y / 2
        return ((-hx, hx, -hy, hy),)

    @property
    def flap_down_rad(self) -> float:
        """The flap's hinge angle with the underside of its tip `flap_lip` above
        the ground: negative, tip down.

        **The plate's thickness counts.** The hinge is on the plate's top face,
        so its underside hangs `deck_t` lower; the first version left it out and
        put the tip's underside 1.5 cm into the ground. Soft, the hinge gave and
        the board moved 2 - 4 mm on its own in 3 s; locked (kp 400) it shoved
        the parked board up to 28 cm with nobody near it (measured 2026-10-23,
        `lab/probe_flap.py`), which spoilt the first locked-flap comparison.
        """
        if self.flap_len <= 0:
            return 0.0
        rise = self.deck_top_z - self.deck_t - self.flap_lip
        return -math.asin(min(1.0, rise / self.flap_len))

    @property
    def flap_up_rad(self) -> float:
        return math.radians(self.flap_stow_deg)

    @property
    def deck_top_z(self) -> float:
        """Height of the deck's top face above the ground at rest."""
        return self.wheel_r + self.deck_above_axle + self.deck_t

    @property
    def robot_offset(self) -> tuple[float, float, float]:
        """Where the robot's base stands, in the chassis frame at rest.

        Shifted back by the stance centre so that the **feet** are centred on
        the deck rather than the base, as on the swing's plank.
        """
        return (-STANCE_CENTRE_X, 0.0, self.deck_top_z - self.wheel_r + STAND_Z)


@dataclass(frozen=True)
class Firmware:
    """The board's own balance loop. See `mdp/firmware.py`."""

    kp: float = 20.0  # m/s^2 of commanded acceleration per rad of pitch
    kd: float = 4.0  # per rad/s of pitch rate
    kv: float = 4.0  # per m/s of speed
    k_steer: float = 8.0  # rad/s of yaw per rad of deck roll
    v_max: float = 1.5
    yaw_max: float = 2.0


DESIGN = SegwayDesign()
FIRMWARE = Firmware()


def comb_deck(center_w: float, outer_in: float, outer_w: float = 0.08,
              length: float = 0.58, spine: float = 0.06, center_drop: float = 0.0) -> tuple:
    """A comb-shaped deck: a spine across the front, tines reaching back.

    The robot walks in from the back with its legs in the slots and its belly
    over the tines, then steps its feet sideways onto them. The numbers it has
    to clear, measured on the robot standing (`lab/probe_body.py`):

        belly (base_link) underside   8.9 cm up, |y| < 9 cm
        hips and thighs underside     6.4 cm up, |y| 4 - 14 cm
        middle legs                   |y| 15 - 21 cm, ground to 12 cm
        rear legs                     |y| 11 - 18 cm
        claws, out in front           x up to +21 cm, |y| 2 - 15 cm, low
        wheels' inner faces           |y| 32 cm

    So the tines' tops have to be under 6.4 cm wherever a hip passes over them,
    and the outer tines fit between 21 and 30 cm. The spine sits ahead of the
    claws: from x = length/2 - spine forwards.

    `center_w` is the width of the tine under the belly (0 for none: a コ),
    `outer_in` and `outer_w` place the two outer tines. `center_drop` makes the
    centre tine a ramp, its back end that far below the deck's top: the robot
    walks its belly onto the low end and pushes itself up it with its back legs
    (the user's idea, 2026-10-22). Kept narrower than the hips, which start at
    |y| 4 cm, so only the belly rides it.
    """
    hx = length / 2
    front = hx - spine
    outer = outer_in + outer_w
    rects = [(front, hx, -outer, outer)]
    if center_w > 0:
        rects.append((-hx, front, -center_w / 2, center_w / 2, center_drop))
    rects += [(-hx, front, outer_in, outer), (-hx, front, -outer, -outer_in)]
    return tuple(rects)


def ramp_deck(top: float, lanes: tuple, back: float = 0.45, flat_from: float = 0.19,
              spine_from: float = 0.16, front: float = 0.29, half_width: float = 0.30,
              lip: float = 0.005) -> tuple:
    """Lanes that ramp up from just off the ground at the back to a flat deck.

    Each lane in `lanes` is (y0, y1). It is a ramp from x = -back to
    x = -flat_from, then flat to the spine, which spans the deck at the front from
    `spine_from` and carries the claws. The robot walks straight up: no foot has
    to move sideways or step up, which is what the comb decks asked for and the
    robot never learnt (2026-10-22: it walked into every comb and stopped there).

    The flat part starts behind the rear feet (at x -16 cm on the boarded robot),
    so every foot ends on a level plate and the boarding test is unchanged. The
    ramp's low end leaves the plate `lip` above the ground so it never digs in.
    """
    drop = top - DESIGN.deck_t - lip
    rects = [(spine_from, front, -half_width, half_width)]
    for y0, y1 in lanes:
        rects.append((-back, -flat_from, y0, y1, drop))
        rects.append((-flat_from, spine_from, y0, y1))
    return tuple(rects)


def _flap(top: float, flap_len: float, lock: bool = False) -> SegwayDesign:
    """A plain plate deck at `top` with a stowing ramp flap behind it.

    `lock` holds the lowered flap stiffly enough to prop the board. Measured
    without it (2026-10-23, flap_h9_l22 / h9_l30): the robot's weight on the
    flap bent the 30 N m/rad hinge, the board pitched into a "へ" about the flap's
    tip, never came level enough to engage, and was shoved 26 - 36 cm before
    every attempt; 0% boarded. The fixed ramp of `ramp_full`, which props the
    board through its own tip, boarded 100%.
    """
    extra = dict(flap_kp=400.0, flap_kd=10.0, flap_torque=40.0, flap_armature=0.06) if lock else {}
    return replace(DESIGN, deck_above_axle=top - DESIGN.deck_t - DESIGN.wheel_r,
                   through_axle=False, battery_in_hubs=True, approach=-1, flap_len=flap_len,
                   **extra)


def _low_comb(top: float = 0.05, shape: tuple | None = None, **comb) -> SegwayDesign:
    """A comb deck at `top` metres, on hub motors with the batteries in the hubs."""
    shape = shape if shape is not None else comb_deck(**comb)
    xs = [v for r in shape for v in r[:2]]
    ys = [v for r in shape for v in r[2:4]]
    return replace(
        DESIGN, deck_shape=shape, deck_x=max(xs) - min(xs), deck_y=max(ys) - min(ys),
        deck_above_axle=top - DESIGN.deck_t - DESIGN.wheel_r, through_axle=False,
        battery_in_hubs=True, approach=-1,
    )

#: The boards the boarding comparison runs on (`segway_board/env_cfg.py`).
#:
#:   baseline   the board as the riding task trained on: 14 cm step to the deck
#:   kneel      the same board, kneeling forward 30 degrees while parked. At 30
#:              the front edge of the deck sits about 9 mm off the ground
#:   lowfloor   deck dropped below the axle: a 6.5 cm step. The battery moves
#:              under the deck to make room, 5 mm above the ground
DESIGNS = {
    "baseline": DESIGN,
    "kneel": replace(DESIGN, kneel_deg=30.0),
    "lowfloor": replace(DESIGN, deck_above_axle=-0.045, battery_below_axle=0.065),
    "lowfloor_hub": replace(DESIGN, deck_above_axle=-0.045, battery_below_axle=0.065,
                            through_axle=False),
    # Comb decks (`comb_deck`), 5 cm high, boarded from the back.
    "comb_U": _low_comb(center_w=0.0, outer_in=0.22),
    "comb_E": _low_comb(center_w=0.10, outer_in=0.22),
    "comb_E_wide": _low_comb(center_w=0.16, outer_in=0.22),
    "flat_low": _low_comb(center_w=0.44, outer_in=0.22),
    # The belly ramp: the centre tine runs from 1.5 cm off the ground at the back
    # up to the deck. With the belly carrying the robot up, the deck can be
    # higher than the hips would otherwise allow.
    "comb_E_ramp": _low_comb(top=0.07, center_w=0.08, outer_in=0.22, center_drop=0.045),
    "comb_E_ramp_low": _low_comb(top=0.05, center_w=0.08, outer_in=0.22, center_drop=0.025),
    # Step 4: ramp lanes under the feet's own tracks (`ramp_deck`). The claws
    # (|y| 2 - 15 cm) take the centre lane, the middle and rear feet (|y| 11 -
    # 21 cm) the outer ones; the 3 cm slots between them carry nothing.
    "rails": _low_comb(top=0.07, shape=ramp_deck(0.07, ((-0.105, 0.105), (0.135, 0.225), (-0.225, -0.135)))),
    # The control for `rails`: the same ramp at full width, no slots.
    "ramp_full": _low_comb(top=0.07, shape=ramp_deck(0.07, ((-0.225, 0.225),))),
    # Step 5: the 45 cm plate on hub motors, with a ramp flap at the back that
    # stows when riding. Deck height against flap length: a higher deck clears
    # the road at the 11 degree cruise tilt, a longer flap keeps the ramp gentle.
    "flap_h7_l22": _flap(top=0.07, flap_len=0.22),
    "flap_h9_l22": _flap(top=0.09, flap_len=0.22),
    "flap_h9_l30": _flap(top=0.09, flap_len=0.30),
    "flap_h11_l40": _flap(top=0.11, flap_len=0.40),
    "flap_h9_l40_lock": _flap(top=0.09, flap_len=0.40, lock=True),
    "flap_h9_l30_lock": _flap(top=0.09, flap_len=0.30, lock=True),
    "flap_h11_l40_lock": _flap(top=0.11, flap_len=0.40, lock=True),
}

# Step 7: back to what boarded -- the plain low plate on hub motors -- with the
# batteries moved into the hubs and the deck shortened to 42 cm, so it clears
# the road at the cruise tilt (`lab/deck_screen.py`: 6.5 cm step 0.4 cm at
# 12 degrees, 9 cm step 2.9 cm). One design per step height, for a curriculum
# that raises the step and carries the policy up from each height to the next.
LOWCELL_HEIGHTS_CM = (6.5, 8, 9, 10, 11, 12, 13, 14)
for _h in LOWCELL_HEIGHTS_CM:
    DESIGNS[f"lowcell_h{_h:g}"] = replace(
        DESIGN, through_axle=False, battery_in_hubs=True, deck_x=0.42,
        deck_above_axle=_h / 100 - DESIGN.deck_t - DESIGN.wheel_r,
    )

#: Teal, so the red crab reads against its board.
_DECK_RGBA = "0.17 0.71 0.69 1"
_TYRE_RGBA = "0.12 0.12 0.13 1"
_HUB_RGBA = "0.95 0.62 0.15 1"
_FRAME_RGBA = "0.55 0.56 0.60 1"
_BATTERY_RGBA = "0.18 0.36 0.75 1"


def _xml(d: SegwayDesign) -> str:
    half_track = d.track / 2
    wheel_mass = d.m_wheel + (d.m_battery / 2 if d.battery_in_hubs else 0.0)

    def wheel(side: str, y: float) -> str:
        return f"""
      <body name="wheel_{side}" pos="0 {y} 0">
        <joint name="wheel_{side}" type="hinge" axis="0 1 0" damping="0.002"/>
        <geom name="tyre_{side}" type="cylinder" size="{d.wheel_r} {d.wheel_w / 2}"
              euler="90 0 0" mass="{wheel_mass}" condim="4"
              friction="{d.wheel_friction} 0.01 0.001" rgba="{_TYRE_RGBA}"/>
        <geom name="hub_{side}" type="box" size="{d.wheel_r * 0.7} {d.wheel_w / 2 + 0.002} 0.012"
              mass="0" contype="0" conaffinity="0" rgba="{_HUB_RGBA}"/>
      </body>"""

    deck_z = d.deck_above_axle + d.deck_t / 2
    if d.through_axle:
        axle = (f'<geom name="axle" type="capsule" fromto="0 {-half_track + d.wheel_w} 0 0 '
                f'{half_track - d.wheel_w} 0" size="0.014" mass="{d.m_chassis}" rgba="{_FRAME_RGBA}"/>')
    else:
        # Two stubs from the deck's sides to the hubs, the chassis mass split
        # between them; nothing above the deck between the wheels.
        stub = min(d.deck_y / 2, half_track - d.wheel_w - 0.03)
        axle = "\n      ".join(
            f'<geom name="stub_{s}" type="capsule" fromto="0 {sign * stub} 0 0 '
            f'{sign * (half_track - d.wheel_w)} 0" size="0.014" mass="{d.m_chassis / 2}" '
            f'rgba="{_FRAME_RGBA}"/>' for s, sign in (("l", 1), ("r", -1)))
    battery = "" if d.battery_in_hubs else (
        f'<geom name="battery" type="box" pos="0 0 {-d.battery_below_axle}" size="0.10 0.14 0.02" '
        f'mass="{d.m_battery}" rgba="{_BATTERY_RGBA}"/>')
    rects = d.deck_rects
    area = sum((r[1] - r[0]) * (r[3] - r[2]) for r in rects)

    def plate(i, r):
        # A fifth number is a ramp: how far the back end (x0) drops below the
        # deck's top, the front end (x1) staying level with it.
        x0, x1, y0, y1 = r[:4]
        drop = r[4] if len(r) > 4 else 0.0
        run = x1 - x0
        half = math.hypot(run, drop) / 2
        pitch = -math.degrees(math.atan2(drop, run))
        return (f'<geom name="{"deck" if i == 0 else f"deck_{i}"}" type="box" '
                f'pos="{(x0 + x1) / 2} {(y0 + y1) / 2} {-drop / 2}" euler="0 {pitch} 0" '
                f'size="{half} {(y1 - y0) / 2} {d.deck_t / 2}" '
                f'mass="{d.m_deck * run * (y1 - y0) / area}" condim="3" rgba="{_DECK_RGBA}"/>')

    deck_geoms = "\n        ".join(plate(i, r) for i, r in enumerate(rects))
    flap = ""
    if d.flap_len > 0:
        # Hinged on the deck's top back edge; the plate hangs behind the hinge,
        # so a positive angle lifts its tip and a negative one lowers it.
        back = min(r[0] for r in rects)
        flap = (f'<body name="flap" pos="{back} 0 {d.deck_t / 2}">'
                f'<joint name="ramp_flap" type="hinge" axis="0 1 0" damping="0.05" '
                f'range="{math.degrees(d.flap_down_rad) - 10} {d.flap_stow_deg + 5}" limited="true"/>'
                f'<geom name="flap" type="box" pos="{-d.flap_len / 2} 0 {-d.deck_t / 2}" '
                f'size="{d.flap_len / 2} {d.deck_y / 2} {d.deck_t / 2}" mass="0.15" condim="3" '
                f'rgba="{_DECK_RGBA}"/></body>')
    kneel = (f'<joint name="deck_pitch" type="hinge" axis="0 1 0" damping="0.2" '
             f'range="-5 {d.kneel_deg + 5}" limited="true"/>') if d.kneel_deg > 0 else ""
    return f"""
<mujoco model="segway">
  <worldbody>
    <body name="chassis" pos="0 0 {d.wheel_r}">
      <freejoint name="segway_root"/>
      {axle}
      {battery}
      {wheel("l", half_track)}
      {wheel("r", -half_track)}
      <body name="deck" pos="0 0 {deck_z}">
        {kneel}<joint name="deck_roll" type="hinge" axis="1 0 0" stiffness="{d.roll_stiffness}"
               damping="{d.roll_damping}" range="{-d.roll_limit_deg} {d.roll_limit_deg}"
               limited="true"/>
        {deck_geoms}
        {flap}
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def segway_spec(design: SegwayDesign = DESIGN) -> Any:
    import mujoco

    return mujoco.MjSpec.from_string(_xml(design))


def segway_entity(design: SegwayDesign = DESIGN) -> Any:
    """The board as an mjlab entity: floating base, two velocity-driven wheels.

    **Floating base, unlike the swing.** The swing had to stay fixed-base so its
    frame could be welded per environment; a board has nothing welded to the
    world, so it takes the one freejoint an entity is allowed, and none of the
    swing's mocap workarounds apply.
    """
    from mjlab.actuator import BuiltinPositionActuatorCfg, BuiltinVelocityActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

    actuators = [
        BuiltinVelocityActuatorCfg(
            target_names_expr=("wheel_l", "wheel_r"),
            damping=design.wheel_kv,
            effort_limit=design.wheel_torque,
            armature=design.wheel_armature,
        ),
    ]
    if design.flap_len > 0:
        actuators.append(BuiltinPositionActuatorCfg(
            target_names_expr=("ramp_flap",),
            stiffness=design.flap_kp,
            damping=design.flap_kd,
            effort_limit=design.flap_torque,
            armature=design.flap_armature or None,
        ))
    if design.kneel_deg > 0:
        actuators.append(BuiltinPositionActuatorCfg(
            target_names_expr=("deck_pitch",),
            stiffness=design.kneel_kp,
            damping=design.kneel_kd,
            effort_limit=design.kneel_torque,
        ))
    return EntityCfg(
        spec_fn=lambda: segway_spec(design),
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, design.wheel_r)),
        articulation=EntityArticulationInfoCfg(actuators=tuple(actuators)),
    )
