"""Export the board for modelling in Rhino 6: drive parts, the deck, and the
closed curves around where the crab actually touched it.

Reads `lab/out/contacts_<design>.json` (`lab/contact_map.py`) and writes a
version-6 .3dm in millimetres. Frame: origin on the ground under the axle,
X forward, Y left, Z up -- the board at rest.

Keep regions are built on a 5 mm grid of contact counts on the deck's top:

    touched    every cell with at least `--min-hits` contacts
    frequent   the fewest cells holding `--share` of all contacts

each closed by one cell of dilation, then traced into closed polylines at the
deck's top height. Grid noise is dropped: islands under `--min-island` cm^2
and holes under `--min-hole` cm^2 (holes are filled); the holes that remain go
on their own layer.

    lab/.venv/Scripts/python.exe lab/export_rhino.py --design lowcell_h14 \
        --contacts lab/out/contacts_lowcell_h14.json --out lab/out/crab_board_lowcell_h14.3dm
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import rhino3dm as r3

ROOT = Path(__file__).resolve().parents[1]
# `jumper/` beside `lab/` while developing, or the repository itself when published.
JUMPER = ROOT / "jumper" if (ROOT / "jumper" / "tasks").is_dir() else ROOT
CELL = 0.005
MM = 1000.0


# ── Keep regions ─────────────────────────────────────────────────────────


def grid_counts(points, x_range, y_range):
    nx = int(math.ceil((x_range[1] - x_range[0]) / CELL))
    ny = int(math.ceil((y_range[1] - y_range[0]) / CELL))
    g = np.zeros((nx, ny), dtype=np.int64)
    for p in points:
        i = int((p[0] - x_range[0]) / CELL)
        j = int((p[1] - y_range[0]) / CELL)
        if 0 <= i < nx and 0 <= j < ny:
            g[i, j] += 1
    return g


def dilate(mask):
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def frequent_mask(counts, share):
    flat = np.sort(counts.ravel())[::-1]
    total = flat.sum()
    if total == 0:
        return counts > 0
    k = int(np.searchsorted(np.cumsum(flat), share * total)) + 1
    cutoff = flat[min(k, len(flat)) - 1]
    return counts >= max(cutoff, 1)


def trace_loops(mask):
    """Closed loops along the boundary of the filled cells, in cell corners.

    Each filled cell contributes its four edges, counter-clockwise; an edge shared
    with a filled neighbour cancels against the neighbour's opposite edge, so what
    survives is the boundary, outer loops counter-clockwise and holes clockwise.
    """
    edges = {}
    nx, ny = mask.shape
    for i in range(nx):
        for j in range(ny):
            if not mask[i, j]:
                continue
            for a, b in (((i, j), (i + 1, j)), ((i + 1, j), (i + 1, j + 1)),
                         ((i + 1, j + 1), (i, j + 1)), ((i, j + 1), (i, j))):
                if (b, a) in edges:
                    del edges[(b, a)]
                else:
                    edges[(a, b)] = True
    nxt = {}
    for a, b in edges:
        nxt.setdefault(a, []).append(b)
    loops = []
    while nxt:
        start = next(iter(nxt))
        loop, cur = [start], start
        while True:
            b = nxt[cur].pop()
            if not nxt[cur]:
                del nxt[cur]
            if b == start:
                break
            loop.append(b)
            cur = b
        loops.append(loop)
    return loops


def simplify(loop):
    """Drop corners that lie on a straight run."""
    out = []
    n = len(loop)
    for k in range(n):
        a, b, c = loop[k - 1], loop[k], loop[(k + 1) % n]
        if (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) != 0:
            out.append(b)
    return out


def signed_area(loop):
    return 0.5 * sum(loop[k - 1][0] * loop[k][1] - loop[k][0] * loop[k - 1][1]
                     for k in range(len(loop)))


def regions(points, x_range, y_range, min_hits, share, min_island=1.5, min_hole=4.0):
    counts = grid_counts(points, x_range, y_range)
    out = {}
    cm2 = 1e-4
    for name, mask in (("touched", counts >= min_hits), ("frequent", frequent_mask(counts, share))):
        loops = [simplify(l) for l in trace_loops(dilate(mask))]
        loops = [[(x_range[0] + i * CELL, y_range[0] + j * CELL) for i, j in l]
                 for l in loops if len(l) >= 3]
        keep = []
        for l in loops:
            area = signed_area(l)
            if area > 0 and area >= min_island * cm2:
                keep.append(l)
            elif area < 0 and -area >= min_hole * cm2:
                keep.append(l)
        out[name] = keep
    return out, counts


# ── Rhino geometry ───────────────────────────────────────────────────────


def layer(model, name, rgb):
    lay = r3.Layer()
    lay.Name = name
    lay.Color = (*rgb, 255)
    return model.Layers.Add(lay)


def attrs(layer_index, name=""):
    a = r3.ObjectAttributes()
    a.LayerIndex = layer_index
    a.Name = name
    return a


def polyline(model, pts_m, z_m, layer_index, name):
    pl = r3.Polyline()
    for x, y in pts_m:
        pl.Add(x * MM, y * MM, z_m * MM)
    pl.Add(pts_m[0][0] * MM, pts_m[0][1] * MM, z_m * MM)
    model.Objects.AddCurve(pl.ToPolylineCurve(), attrs(layer_index, name))


def mesh_from(verts_m, faces, layer_index, name, model):
    me = r3.Mesh()
    for v in verts_m:
        me.Vertices.Add(float(v[0] * MM), float(v[1] * MM), float(v[2] * MM))
    for f in faces:
        me.Faces.AddFace(int(f[0]), int(f[1]), int(f[2]))
    me.Normals.ComputeNormals()
    me.Compact()
    model.Objects.AddMesh(me, attrs(layer_index, name))


def cylinder_y(center, radius, width, segments=48):
    """A closed cylinder whose axis is the Y axis through `center`, metres."""
    cx, cy, cz = center
    verts, faces = [], []
    for s in (-1, 1):
        for k in range(segments):
            a = 2 * math.pi * k / segments
            verts.append((cx + radius * math.cos(a), cy + s * width / 2, cz + radius * math.sin(a)))
    verts += [(cx, cy - width / 2, cz), (cx, cy + width / 2, cz)]
    c0, c1 = 2 * segments, 2 * segments + 1
    for k in range(segments):
        a, b = k, (k + 1) % segments
        faces += [(a, b, segments + b), (a, segments + b, segments + a)]
        faces += [(c0, b, a), (c1, segments + a, segments + b)]
    return verts, faces


def box(x0, x1, y0, y1, z0, z1):
    v = [(x, y, z) for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)]
    f = [(0, 2, 3), (0, 3, 1), (4, 5, 7), (4, 7, 6), (0, 1, 5), (0, 5, 4),
         (2, 6, 7), (2, 7, 3), (0, 4, 6), (0, 6, 2), (1, 3, 7), (1, 7, 5)]
    return v, f


def line(model, a, b, layer_index, name):
    model.Objects.AddLine(r3.Point3d(*(c * MM for c in a)), r3.Point3d(*(c * MM for c in b)),
                          attrs(layer_index, name))


def dot(model, text, p, layer_index):
    model.Objects.AddTextDot(text, r3.Point3d(*(c * MM for c in p)), attrs(layer_index, text))


def reference_crab(model, layer_index, design):
    """The robot's visual meshes in its standing pose on the boarded spot."""
    import mujoco

    pose = json.loads((ROOT / "lab" / "out" / "stand_pose.json").read_text(encoding="utf-8"))
    m = mujoco.MjModel.from_xml_path(str(JUMPER / "assets" / "jumper" / "jumper.xml"))
    d = mujoco.MjData(m)
    for jn, q in pose.items():
        j = m.joint(jn)
        d.qpos[m.jnt_qposadr[j.id]:m.jnt_qposadr[j.id] + len(q)] = q
    ox, _, oz = design.robot_offset
    d.qpos[0:3] = [ox, 0.0, design.wheel_r + oz]
    d.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    for g in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or "visual" not in name:
            continue
        k = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]]
        f = m.mesh_face[m.mesh_faceadr[k]:m.mesh_faceadr[k] + m.mesh_facenum[k]]
        w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        mesh_from(w, f, layer_index, name.replace("_visual", ""), model)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--contacts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-hits", type=int, default=2)
    ap.add_argument("--share", type=float, default=0.9)
    ap.add_argument("--min-island", type=float, default=1.5, help="cm^2")
    ap.add_argument("--min-hole", type=float, default=4.0, help="cm^2")
    a = ap.parse_args()

    sys.path.insert(0, str(JUMPER))
    from tasks.jumper.segway import vehicle

    d = vehicle.DESIGNS[a.design]
    data = json.loads(a.contacts.read_text(encoding="utf-8"))
    hx, hy = d.deck_x / 2, d.deck_y / 2
    top = d.deck_top_z
    x_range, y_range = (-hx - 0.02, hx + 0.02), (-hy - 0.02, hy + 0.02)

    model = r3.File3dm()
    model.Settings.ModelUnitSystem = r3.UnitSystem.Millimeters
    L = {
        "wheels": layer(model, "01_Wheels", (40, 40, 45)),
        "drive": layer(model, "02_Drive", (240, 150, 40)),
        "deck": layer(model, "03_Deck_current", (43, 181, 176)),
        "board": layer(model, "10_Keep_boarding", (230, 80, 60)),
        "ride": layer(model, "11_Keep_riding", (60, 110, 230)),
        "union": layer(model, "12_Keep_union", (200, 30, 160)),
        "freq": layer(model, "13_Keep_frequent_90pct", (120, 30, 120)),
        "holes": layer(model, "14_Keep_holes", (150, 150, 150)),
        "notes": layer(model, "80_Notes", (90, 90, 90)),
        "crab": layer(model, "90_Reference_crab", (220, 50, 40)),
    }

    # Drive: wheels, hub motors (batteries inside), stub axles, axle and roll axes.
    half_track = d.track / 2
    for side, sign in (("L", 1), ("R", -1)):
        v, f = cylinder_y((0, sign * half_track, d.wheel_r), d.wheel_r, d.wheel_w)
        mesh_from(v, f, L["wheels"], f"wheel_{side}  D{d.wheel_r * 2 * MM:.0f} x W{d.wheel_w * MM:.0f}", model)
        v, f = cylinder_y((0, sign * half_track, d.wheel_r), d.wheel_r * 0.55, d.wheel_w + 0.004)
        mesh_from(v, f, L["drive"], f"hub_motor_{side} (batteries in hub)", model)
        stub0 = min(d.deck_y / 2, half_track - d.wheel_w - 0.03)
        y0, y1 = sign * stub0, sign * (half_track - d.wheel_w / 2)
        v, f = cylinder_y((0, (y0 + y1) / 2, d.wheel_r), 0.014, abs(y1 - y0), 24)
        mesh_from(v, f, L["drive"], f"stub_axle_{side}", model)
    line(model, (0, -half_track - 0.05, d.wheel_r), (0, half_track + 0.05, d.wheel_r), L["drive"], "axle_axis")
    deck_center_z = d.wheel_r + d.deck_above_axle + d.deck_t / 2
    line(model, (-hx - 0.05, 0, deck_center_z), (hx + 0.05, 0, deck_center_z), L["drive"],
         f"deck_roll_hinge_axis (+/-{d.roll_limit_deg:g} deg, {d.roll_stiffness:g} N m/rad)")

    # The current deck.
    v, f = box(-hx, hx, -hy, hy, top - d.deck_t, top)
    mesh_from(v, f, L["deck"], "deck_current", model)
    polyline(model, [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)], top, L["deck"], "deck_outline")

    # Keep regions.
    summary = {}
    for phase, lay in (("boarding", L["board"]), ("riding", L["ride"])):
        reg, _ = regions(data[phase], x_range, y_range, a.min_hits, a.share,
                         a.min_island, a.min_hole)
        for kind, loops in reg.items():
            for n, loop in enumerate(loops):
                target = lay if kind == "touched" else L["freq"]
                if signed_area(loop) < 0:
                    target = L["holes"]
                polyline(model, loop, top, target, f"{phase}_{kind}_{n}")
        summary[phase] = {k: len(v) for k, v in reg.items()}
    union, counts = regions(data["boarding"] + data["riding"], x_range, y_range, a.min_hits, a.share,
                            a.min_island, a.min_hole)
    for n, loop in enumerate(union["touched"]):
        polyline(model, loop, top, L["union"] if signed_area(loop) > 0 else L["holes"], f"keep_union_{n}")
    for n, loop in enumerate(union["frequent"]):
        polyline(model, loop, top, L["freq"] if signed_area(loop) > 0 else L["holes"],
                 f"keep_union_frequent_{n}")
    summary["union"] = {k: len(v) for k, v in union.items()}

    # Notes.
    dot(model, f"deck top Z{top * MM:.0f}", (hx, hy, top), L["notes"])
    dot(model, f"track {d.track * MM:.0f} (wheel centres)", (0, half_track, d.wheel_r * 2 + 0.03), L["notes"])
    dot(model, "front +X", (hx + 0.08, 0, top), L["notes"])
    dot(model, f"ground clearance at 12 deg pitch must stay > 0 (now {d.deck_top_z * MM:.0f} step)",
        (0, -half_track, 0.0), L["notes"])

    reference_crab(model, L["crab"], d)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    model.Write(str(a.out), 6)
    print("wrote", a.out, summary)


if __name__ == "__main__":
    main()
