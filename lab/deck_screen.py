"""Step 2 of the deck-shape search: a geometric screen, no learning.

For each board design, the crab in its standing pose is slid from behind the
board (or in front, per `approach`) to the boarded spot, and at every 1 cm the
contacts between its collision geoms and the board are listed by body part. A
shape that lets the robot walk in touches only with parts it can lift (the
claws) or with its belly on a ramp; one that blocks it hits legs or hips.

At the boarded spot it also measures, for each foot, how far it would have to
move sideways to stand on a plate (0 if it is already over one), and an
engineering note on the shape: the longest unsupported tine and the narrowest.

    jumper/.venv/Scripts/python.exe lab/deck_screen.py baseline lowfloor_hub comb_E comb_E_ramp ...
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

# The framework checkout: `jumper/` beside `lab/` while developing, or the
# repository itself when `lab/` is published inside it.
_ROOT = Path(__file__).resolve().parents[1]
JUMPER = _ROOT / "jumper" if (_ROOT / "jumper" / "tasks").is_dir() else _ROOT
sys.path.insert(0, str(JUMPER))
from tasks.jumper.common.constants import FEET  # noqa: E402
from tasks.jumper.segway import vehicle  # noqa: E402

POSE = json.loads((Path(__file__).parent / "out" / "stand_pose.json").read_text(encoding="utf-8"))


def part(name: str) -> str:
    """Group collision geoms into the body parts a reader thinks in."""
    n = name.replace("_meshcol", "")
    for key, label in (("palm", "claw"), ("finger", "claw"), ("forearm", "arm"), ("upper_arm", "arm"),
                       ("shoulder", "arm"), ("calf", "leg"), ("foot", "foot"), ("thigh", "hip"),
                       ("hip", "hip"), ("base", "belly"), ("shell", "shell"), ("display", "shell")):
        if key in n:
            return label
    return n


def screen(name: str) -> dict:
    d = vehicle.DESIGNS[name]
    spec = mujoco.MjSpec.from_file(str(JUMPER / "assets" / "jumper" / "jumper.xml"))
    board = mujoco.MjSpec.from_string(vehicle._xml(d))
    # The chassis body already sits at wheel_r in the board XML.
    frame = spec.worldbody.add_frame(pos=[0, 0, 0])
    frame.attach_body(board.body("chassis"), "board_", "")
    m = spec.compile()
    data = mujoco.MjData(m)
    for jn, q in POSE.items():
        j = m.joint(jn)
        data.qpos[m.jnt_qposadr[j.id]:m.jnt_qposadr[j.id] + len(q)] = q
    root = m.joint("floating_base").qposadr[0] if hasattr(m.joint("floating_base"), "qposadr") else 0
    stand_z = POSE["floating_base"][2]
    board_geoms = {g for g in range(m.ngeom)
                   if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("board_")}
    ox = vehicle.DESIGN.robot_offset[0]
    side = d.approach
    start = side * (d.deck_x / 2 + 0.25)
    def surface(x: float, y: float = 0.18) -> float:
        """Height of the deck's top under (x, y), 0 off it: the robot walks up ramps."""
        top = 0.0
        for r in d.deck_rects:
            if r[0] <= x <= r[1] and r[2] <= y <= r[3]:
                drop = r[4] if len(r) > 4 else 0.0
                top = max(top, d.deck_top_z - drop * (r[1] - x) / (r[1] - r[0]))
        return top

    hits_by_x = []
    for x in np.arange(start, ox, -side * 0.01):
        # Only ramps lift the robot; stepping up onto a level plate is not walking in.
        lift = surface(x) if any(len(r) > 4 and r[4] > 0 and r[3] > 0.1 for r in d.deck_rects) else 0.0
        data.qpos[root:root + 3] = [x, 0.0, stand_z + lift]
        yaw = 0.0 if side < 0 else math.pi
        data.qpos[root + 3:root + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        mujoco.mj_forward(m, data)
        parts = Counter()
        for c in data.contact[:data.ncon]:
            a, b = int(c.geom1), int(c.geom2)
            if (a in board_geoms) == (b in board_geoms):
                continue
            robot_g = b if a in board_geoms else a
            if c.dist < -0.002:
                parts[part(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, robot_g) or "")] += 1
        hits_by_x.append((round(float(x), 2), dict(parts)))
    blocked = Counter()
    for _, parts in hits_by_x:
        for p in parts:
            blocked[p] += 1
    # Feet at the boarded spot, in the deck frame (flat deck at rest).
    steps = []
    for f in FEET:
        bid = m.body(f).id
        fx, fy = data.xpos[bid][0], data.xpos[bid][1]
        best = min(math.hypot(max(r[0] - fx, 0, fx - r[1]), max(r[2] - fy, 0, fy - r[3]))
                   for r in d.deck_rects)
        steps.append(round(best, 3))
    tines = [r for r in d.deck_rects if (r[1] - r[0]) > (r[3] - r[2])]
    return {
        "design": name,
        "deck_top_cm": round(d.deck_top_z * 100, 1),
        "walk_in_cm_blocked_by_part": dict(blocked),
        "feet_step_to_plate_cm": [round(s * 100, 1) for s in steps],
        "longest_tine_cm": round(max((r[1] - r[0] for r in tines), default=0) * 100, 1),
        "narrowest_tine_cm": round(min((r[3] - r[2] for r in tines), default=0) * 100, 1),
        "plates": len(d.deck_rects),
    }


if __name__ == "__main__":
    for n in sys.argv[1:]:
        print(json.dumps(screen(n), ensure_ascii=False))


def ride_clearance(name: str, tilts=(-12.0, 0.0, 12.0)) -> dict:
    """Lowest point of the board, wheels aside, with the flap stowed, pitched
    about the axle by each of `tilts` degrees: what scrapes on a bump at the
    cruise tilt (about 11 degrees nose-up at 1 m/s)."""
    d = vehicle.DESIGNS[name]
    m = mujoco.MjModel.from_xml_string(vehicle._xml(d))
    data = mujoco.MjData(m)
    if d.flap_len > 0:
        data.qpos[m.jnt_qposadr[m.joint("ramp_flap").id]] = d.flap_up_rad
    out = {}
    for deg in tilts:
        a = math.radians(deg)  # positive: nose up
        data.qpos[0:3] = [0, 0, d.wheel_r]
        data.qpos[3:7] = [math.cos(-a / 2), 0, math.sin(-a / 2), 0]
        mujoco.mj_forward(m, data)
        low = 9.0
        for g in range(m.ngeom):
            gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if gname.startswith(("tyre", "hub")):
                continue
            R, c, sz = data.geom_xmat[g].reshape(3, 3), data.geom_xpos[g], m.geom_size[g]
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                for sx in (-1, 1):
                    for sy in (-1, 1):
                        for sz_ in (-1, 1):
                            low = min(low, float((c + R @ (sz * [sx, sy, sz_]))[2]))
            elif m.geom_type[g] in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
                # Along the geom's own z: the two end centres, less the radius.
                for e in (-1, 1):
                    low = min(low, float((c + R[:, 2] * sz[1] * e)[2] - sz[0]))
            else:
                low = min(low, float(c[2] - m.geom_rbound[g]))
        out[f"{deg:+.0f}deg"] = round(low * 100, 1)
    return {"design": name, "lowest_point_cm": out,
            "length_parked_cm": round((d.deck_x + d.flap_len * math.cos(d.flap_down_rad)) * 100, 1)}


