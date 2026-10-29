"""Blender: the crab's errand through town, from a recorded simulation.

Builds a small town from `lab/city/route.json`, rebuilds the robot and its board
from the geoms in `scene.npz` (written by `lab/ride_record.py` with CRAB_EXPORT),
and moves them every frame to exactly where the simulation put them. Nothing is
animated by hand: the motion is the physics, Blender only draws it.

    blender -b --factory-startup -P lab/blender/build_city.py -- \
        --run lab/out/city_run --route lab/city/route.json --out lab/out/city_frames \
        [--frames 1-120] [--samples 32] [--save lab/out/city.blend]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

FONT = "C:/Windows/Fonts/BIZ-UDGothicB.ttc"


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--route", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", default=None)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--save", type=Path, default=None)
    ap.add_argument("--shots", type=Path, default=None, help="JSON list of [start_s, camera, side]")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--film-end", type=int, default=None,
                    help="last frame of the film; sets when the closing shot starts")
    return ap.parse_args(argv)


# ── Materials ────────────────────────────────────────────────────────────

_mats: dict = {}


def mat(name, rgb, rough=0.6, metal=0.0, emit=None, emit_strength=0.0):
    key = (name, tuple(round(c, 3) for c in rgb), rough, metal)
    if key in _mats:
        return _mats[key]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    if emit is not None:
        bsdf.inputs["Emission Color"].default_value = (*emit, 1.0)
        bsdf.inputs["Emission Strength"].default_value = emit_strength
    _mats[key] = m
    return m


def srgb(hexstr):
    h = hexstr.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return tuple(x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c)


# ── Primitives ───────────────────────────────────────────────────────────


def box(name, x0, x1, y0, y1, z0, z1, material, bevel=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))
    o = bpy.context.active_object
    o.name = name
    o.scale = (x1 - x0, y1 - y0, z1 - z0)
    bpy.ops.object.transform_apply(scale=True)
    o.data.materials.append(material)
    if bevel > 0:
        b = o.modifiers.new("bevel", "BEVEL")
        b.width, b.segments = bevel, 2
    return o


def cyl(name, x, y, z0, z1, r, material, verts=24):
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=r, depth=z1 - z0,
                                        location=(x, y, (z0 + z1) / 2))
    o = bpy.context.active_object
    o.name = name
    o.data.materials.append(material)
    bpy.ops.object.shade_smooth()
    return o


def ball(name, x, y, z, r, material):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=r, location=(x, y, z))
    o = bpy.context.active_object
    o.name = name
    o.data.materials.append(material)
    bpy.ops.object.shade_smooth()
    return o


def text(name, body, x, y, z, size, material, rot=(math.pi / 2, 0, 0), align="CENTER"):
    curve = bpy.data.curves.new(name, "FONT")
    curve.body = body
    try:
        curve.font = bpy.data.fonts.load(FONT, check_existing=True)
    except RuntimeError:
        pass
    curve.size = size
    curve.align_x = align
    curve.extrude = 0.01
    o = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(o)
    o.location = (x, y, z)
    o.rotation_euler = rot
    o.data.materials.append(material)
    return o


# ── The town ─────────────────────────────────────────────────────────────


def build_town(route):
    c = route["city"]
    sw = c["sidewalk_width"] / 2
    asphalt = mat("asphalt", srgb("#3b3f45"), 0.9)
    paving = mat("paving", srgb("#c9c2b6"), 0.85)
    curb = mat("curb", srgb("#9a958d"), 0.8)
    grass = mat("grass", srgb("#7fa25a"), 0.95)
    paint = mat("paint", srgb("#f2f2ee"), 0.6)

    box("ground", -40, 60, -40, 50, -0.06, -0.02, grass)
    # Main street along x, south of the first sidewalk; side street along x
    # between the corner and the shopping street.
    box("road_main", -40, 60, -sw - 6.0, -sw, -0.03, -0.005, asphalt)
    box("road_side", -40, 60, 8.0, 12.0, -0.03, -0.005, asphalt)
    # Sidewalks the crab rides on, all at z = 0 like the physics plane.
    box("walk_home", -6, 14.0, -sw, sw, -0.03, 0.0, paving)
    box("walk_corner", 11.0, 14.0, -sw, 8.0, -0.03, 0.0, paving)
    box("walk_far", 11.0, 14.0, 12.0, 13.0, -0.03, 0.0, paving)
    box("walk_shops", -6, 40, 13.0, 16.0, -0.03, 0.0, paving)
    # Curbs where a sidewalk meets a road, except at the crossing.
    box("curb_main", -6, 14.0, -sw - 0.12, -sw, 0.0, 0.06, curb)
    for x0, x1 in ((-6, 11.0), (14.0, 40)):
        box("curb_side_s", x0, x1, 7.88, 8.0, 0.0, 0.06, curb)
        box("curb_side_n", x0, x1, 12.0, 12.12, 0.0, 0.06, curb)
    # Zebra crossing.
    cw = c["crosswalk"]
    for k in range(7):
        x = cw["x"] - 1.2 + k * 0.4
        box(f"zebra{k}", x - 0.12, x + 0.12, cw["y0"], cw["y1"], -0.005, 0.0, paint)
    # Lane marking on the main road.
    for k in range(-12, 20):
        box(f"lane{k}", k * 3.0, k * 3.0 + 1.5, -sw - 3.06, -sw - 2.94, -0.005, 0.0, paint)

    _houses()
    _corner_blocks()
    _shops(c["shop"])
    _street_furniture()
    return _traffic_light(cw)


def _window_row(prefix, x0, x1, y, z, h, n, glass, frame, facing=-1):
    w = (x1 - x0) / n
    for k in range(n):
        cx = x0 + (k + 0.5) * w
        box(f"{prefix}_frame{k}", cx - w * 0.32, cx + w * 0.32, y + facing * 0.02, y + facing * 0.06,
            z, z + h, frame)
        box(f"{prefix}_glass{k}", cx - w * 0.27, cx + w * 0.27, y + facing * 0.06, y + facing * 0.08,
            z + h * 0.08, z + h * 0.92, glass)


def _houses():
    glass = mat("glass", srgb("#9cc3d8"), 0.1, 0.0, emit=srgb("#fff3d6"), emit_strength=0.4)
    frame = mat("frame", srgb("#f5f1ea"), 0.6)
    roof = mat("roof", srgb("#7a4a3a"), 0.7)
    colors = ["#e9d8b8", "#d7e3e8", "#f0c9a8", "#e2e8cf"]
    # Three houses, all west of the corner sidewalk (x 11 - 14).
    for k, (x0, x1) in enumerate(((-6.0, 1.0), (2.0, 6.2), (6.8, 10.6))):
        wall = mat(f"house{k}", srgb(colors[k % 4]), 0.8)
        box(f"house{k}", x0, x1, 3.0, 9.0, 0.0, 5.5, wall)
        box(f"house{k}_roof", x0 - 0.2, x1 + 0.2, 2.8, 9.2, 5.5, 5.8, roof)
        _window_row(f"house{k}_w", x0 + 0.4, x1 - 0.4, 3.0, 3.2, 1.4, 2, glass, frame)
    # The crab's own house: a door at the start of the route.
    door = mat("door", srgb("#c0392b"), 0.5)
    box("home_door", -2.3, -0.7, 2.94, 3.0, 0.0, 2.2, door)
    box("home_step", -2.6, -0.4, 1.6, 3.0, 0.0, 0.0, mat("step", srgb("#b8b2a8"), 0.8))
    text("home_sign", "カニのおうち", -1.5, 2.93, 2.5, 0.35, mat("signink", srgb("#2c3e50"), 0.5))
    # Low hedge between sidewalk and houses.
    hedge = mat("hedge", srgb("#4e7d3a"), 0.9)
    for x0, x1 in ((-6.0, -2.8), (-0.2, 11.0)):
        box("hedge", x0, x1, 1.8, 2.6, 0.0, 0.7, hedge, bevel=0.15)


def _corner_blocks():
    brick = mat("brick", srgb("#b5654b"), 0.85)
    glass = mat("glass2", srgb("#8fb3c9"), 0.1, emit=srgb("#fff0d0"), emit_strength=0.3)
    frame = mat("frame2", srgb("#efe8de"), 0.6)
    box("corner_west", -6.0, 10.6, 3.0, 7.6, 0.0, 0.0, brick)  # footprint only, houses stand here
    box("corner_east", 14.6, 30.0, 1.8, 7.6, 0.0, 8.0, brick)
    _window_row("corner_east_w1", 15.0, 29.6, 1.8, 2.0, 1.6, 6, glass, frame)
    _window_row("corner_east_w2", 15.0, 29.6, 1.8, 4.8, 1.6, 6, glass, frame)
    # A little park on the far side of the side street.
    grass = mat("parkgrass", srgb("#86ad5e"), 0.95)
    box("park", -6.0, 10.6, 12.3, 12.9, 0.0, 0.02, grass)


def _shops(shop):
    awning_colors = ["#e74c3c", "#2e86c1", "#27ae60", "#f39c12", "#8e44ad"]
    wall_colors = ["#f4e1c1", "#e8eef2", "#f2d7d5", "#e9f0dc", "#f6e7cf"]
    glass = mat("shopglass", srgb("#a9cce3"), 0.05, emit=srgb("#fff6e0"), emit_strength=0.8)
    names = ["パン", "花", "本", "八百屋"]
    xs = [-6.0, 1.0, 8.0, 15.0, 22.0]
    for k, x0 in enumerate(xs):
        x1 = x0 + 6.6
        wall = mat(f"shopwall{k}", srgb(wall_colors[k]), 0.8)
        box(f"shop{k}", x0, x1, 16.6, 23.0, 0.0, 6.0 + (k % 2) * 1.5, wall)
        box(f"shop{k}_window", x0 + 0.5, x1 - 0.5, 16.55, 16.6, 0.4, 2.6, glass)
        aw = mat(f"awning{k}", srgb(awning_colors[k]), 0.7)
        o = box(f"shop{k}_awning", x0 + 0.2, x1 - 0.2, 15.9, 16.6, 2.9, 3.0, aw)
        o.rotation_euler = (math.radians(-12), 0, 0)
        label = names[k] if k < len(names) else "おみせ"
        text(f"shop{k}_sign", label, (x0 + x1) / 2, 16.54, 3.4, 0.6,
             mat("signink2", srgb("#3d3d3d"), 0.5))
    # The destination.
    x0 = shop["x"] - 3.0
    wall = mat("kaniwall", srgb("#fdf2e9"), 0.7)
    box("kani_shop", x0, x0 + 7.0, 16.6, 23.0, 0.0, 7.0, wall)
    box("kani_window", x0 + 0.4, x0 + 2.6, 16.55, 16.6, 0.4, 2.6, glass)
    box("kani_door", x0 + 2.9, x0 + 4.4, 16.55, 16.6, 0.0, 2.4, mat("kanidoor", srgb("#a04000"), 0.5))
    red = mat("kanired", srgb("#e74c3c"), 0.6, emit=srgb("#ff5a3c"), emit_strength=0.6)
    box("kani_board", x0 + 0.6, x0 + 6.4, 16.45, 16.55, 3.3, 4.5, red)
    text("kani_sign", shop["name"], x0 + 3.5, 16.43, 3.62, 0.75, mat("white", srgb("#ffffff"), 0.4,
         emit=srgb("#ffffff"), emit_strength=0.5))
    stripes = mat("kaniawning", srgb("#ffffff"), 0.7)
    o = box("kani_awning", x0 + 0.2, x0 + 6.8, 15.8, 16.6, 2.9, 3.0, stripes)
    o.rotation_euler = (math.radians(-12), 0, 0)


def _street_furniture():
    pole = mat("pole", srgb("#5d6d7e"), 0.4, 0.6)
    lamp = mat("lamp", srgb("#fff7e0"), 0.2, emit=srgb("#fff2cc"), emit_strength=3.0)
    trunk = mat("trunk", srgb("#6e4b32"), 0.9)
    leaves = [mat(f"leaves{k}", srgb(c), 0.9) for k, c in enumerate(("#5c8a3c", "#6b9b44", "#4f7d33"))]
    for k, x in enumerate((-4.0, 4.0, 18.0, 26.0, 34.0)):
        cyl(f"lamp_pole{k}", x, -1.35, 0, 3.6, 0.05, pole, 12)
        ball(f"lamp{k}", x, -1.35, 3.7, 0.18, lamp)
    for k, x in enumerate((2.0, 6.0, 20.0, 30.0)):
        cyl(f"tree_trunk{k}", x, 2.2 if x < 11 else 1.2, 0, 2.2, 0.12, trunk, 10)
        ball(f"tree{k}", x, 2.2 if x < 11 else 1.2, 2.8, 1.1, leaves[k % 3])
    for k, x in enumerate((-4.0, 2.5, 9.0)):
        cyl(f"parktree_t{k}", x, 12.6, 0, 1.8, 0.1, trunk, 10)
        ball(f"parktree{k}", x, 12.6, 2.4, 0.9, leaves[(k + 1) % 3])
    # Planters along the shopping street.
    pot = mat("pot", srgb("#a1887f"), 0.8)
    for k, x in enumerate((3.0, 10.0, 17.0, 24.0, 33.0)):
        box(f"planter{k}", x - 0.4, x + 0.4, 15.7, 16.3, 0.0, 0.45, pot)
        ball(f"flowers{k}", x, 16.0, 0.55, 0.35, mat(f"flower{k}", srgb(("#f1948a", "#f7dc6f", "#bb8fce")[k % 3]), 0.8))


def _traffic_light(cw):
    pole = mat("tl_pole", srgb("#4a4a4a"), 0.5, 0.5)
    x, y = cw["x"] + 1.7, cw["y0"] - 0.3
    cyl("tl_pole", x, y, 0, 2.6, 0.05, pole, 12)
    box("tl_box", x - 0.18, x + 0.18, y - 0.12, y + 0.12, 2.3, 3.0, mat("tl_case", srgb("#2b2b2b"), 0.5))
    red = mat("tl_red", srgb("#400000"), 0.3, emit=srgb("#ff2a1a"), emit_strength=0.0)
    green = mat("tl_green", srgb("#003000"), 0.3, emit=srgb("#2aff6a"), emit_strength=0.0)
    # In front of the housing's face (x - 0.18), not sunk in it.
    ball("tl_red_lamp", x - 0.20, y, 2.82, 0.10, red).scale = (0.5, 1, 1)
    ball("tl_green_lamp", x - 0.20, y, 2.48, 0.10, green).scale = (0.5, 1, 1)
    return red, green


# ── The crab and its board ───────────────────────────────────────────────


def build_rider(run: Path):
    z = np.load(run / "scene.npz")
    names, types, sizes, rgba, meshid = z["names"], z["type"], z["size"], z["rgba"], z["meshid"]
    nframes = z["pos"].shape[0]
    objs = []
    face_img = bpy.data.images.load(str(run / "face" / "0001.png"))
    face_img.source = "SEQUENCE"
    mesh_cache = {}
    for i, name in enumerate(names):
        name = str(name)
        # Collision hulls and the ground plane are in the export too; the town
        # draws its own ground.
        if "meshcol" in name or name.endswith("terrain"):
            continue
        t = int(types[i])
        col = tuple(float(c) for c in rgba[i][:3])
        if name.endswith("face_screen"):
            m = bpy.data.materials.new("face")
            m.use_nodes = True
            nt = m.node_tree
            bsdf = nt.nodes.get("Principled BSDF")
            tex = nt.nodes.new("ShaderNodeTexImage")
            tex.image = face_img
            tex.image_user.frame_duration = nframes
            tex.image_user.frame_start = 1
            tex.image_user.use_auto_refresh = True
            tex.interpolation = "Closest"
            nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
            nt.links.new(tex.outputs["Color"], bsdf.inputs["Emission Color"])
            bsdf.inputs["Emission Strength"].default_value = 2.0
            material = m
        elif name.endswith("deck"):
            material = mat("deck", srgb("#2bb5b0"), 0.35)
        else:
            # The robot's red reads as orange under daylight; keep it saturated.
            material = mat(f"geom_{i}", tuple(c ** 2.2 for c in col), 0.45)
        if t == 7:  # mesh
            k = int(meshid[i])
            if k not in mesh_cache:
                v, f = z[f"mesh{k}_v"], z[f"mesh{k}_f"]
                if name.endswith("face_screen"):
                    f = f[:2]  # the other two are the same quad wound the other way
                me = bpy.data.meshes.new(f"mesh{k}")
                me.from_pydata(v.tolist(), [], f.tolist())
                if name.endswith("face_screen"):
                    uv = me.uv_layers.new()
                    # Same corner order as `face.add_face_screen`'s texcoords.
                    corners = {0: (0, 0), 1: (1, 0), 2: (1, 1), 3: (0, 1)}
                    for loop in me.loops:
                        uv.data[loop.index].uv = corners.get(loop.vertex_index, (0, 0))
                else:
                    for p in me.polygons:
                        p.use_smooth = True
                me.update()
                mesh_cache[k] = me
            o = bpy.data.objects.new(name, mesh_cache[k].copy() if name.endswith("face_screen")
                                     else mesh_cache[k])
            bpy.context.collection.objects.link(o)
            if not name.endswith("face_screen"):
                o.modifiers.new("smooth", "WEIGHTED_NORMAL")
        elif t == 6:  # box
            bpy.ops.mesh.primitive_cube_add(size=2)
            o = bpy.context.active_object
            o.scale = tuple(float(s) for s in sizes[i])
            bpy.ops.object.transform_apply(scale=True)
        elif t == 5:  # cylinder: radius, half-height
            bpy.ops.mesh.primitive_cylinder_add(vertices=40, radius=float(sizes[i][0]),
                                                depth=2 * float(sizes[i][1]))
            o = bpy.context.active_object
            bpy.ops.object.shade_smooth()
        elif t == 3:  # capsule
            bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=float(sizes[i][0]),
                                                depth=2 * float(sizes[i][1]))
            o = bpy.context.active_object
        elif t == 2:  # sphere
            bpy.ops.mesh.primitive_uv_sphere_add(radius=float(sizes[i][0]))
            o = bpy.context.active_object
            bpy.ops.object.shade_smooth()
        else:
            continue
        o.name = name
        if o.data.materials:
            o.data.materials[0] = material
        else:
            o.data.materials.append(material)
        o.rotation_mode = "QUATERNION"
        objs.append((i, o))
    return z, objs


def deck_track(z):
    """The deck's position and its heading from its own x axis, per frame.

    Heading from velocity flips when the board creeps backwards after arriving,
    and the camera would swing round to the crab's back.
    """
    names = [str(n) for n in z["names"]]
    i = next(k for k, n in enumerate(names) if n.endswith("deck"))
    x_axis = z["mat"][:, i].reshape(-1, 3, 3)[:, :, 0]
    return z["pos"][:, i, :], np.unwrap(np.arctan2(x_axis[:, 1], x_axis[:, 0]))


# ── Cameras ──────────────────────────────────────────────────────────────


def smooth(a, k):
    out = np.copy(a)
    for i in range(1, len(a)):
        out[i] = out[i - 1] + k * (a[i] - out[i - 1])
    return out


def make_cameras(track, yaw):
    cams = {}
    for name, lens in (("lead", 32), ("chase", 35), ("wide", 24), ("close", 45), ("crossing", 35), ("arrival", 24)):
        cd = bpy.data.cameras.new(name)
        cd.lens = lens
        cd.clip_start = 0.02
        o = bpy.data.objects.new(name, cd)
        bpy.context.collection.objects.link(o)
        cams[name] = o
    target = smooth(track, 0.08)
    heading = smooth(yaw[:, None], 0.04)[:, 0]
    return cams, target, heading


def look_at(obj, eye, at):
    obj.location = eye
    d = Vector(at) - Vector(eye)
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = d.to_track_quat("-Z", "Y")


#: The cut list, in seconds: (start, camera, side). "lead" rides ahead of the crab
#: looking back at its face; "side" picks which side it rides on. `--shots` loads
#: another list as JSON, timed from that run's own events. The default is timed
#: from the first recorded ride (corner 15.6 s, crossing stop 30.5 s, across
#: 45.4 s, arrival 66.7 s).
SHOTS = [
    (0.0, "wide", 0), (4.0, "lead", 1), (13.5, "chase", 0), (21.0, "lead", -1),
    (28.5, "crossing", 0), (44.5, "lead", -1), (60.0, "close", 0),
]


def shot_for(frame, fps):
    t = frame / fps
    current = SHOTS[0]
    for row in SHOTS:
        if t >= row[0]:
            current = row
    return current


# ── Lighting and render settings ─────────────────────────────────────────


def setup_world(a):
    scene = bpy.context.scene
    for engine in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
        try:
            scene.render.engine = engine
            break
        except TypeError:
            continue
    try:
        scene.eevee.taa_render_samples = a.samples
    except AttributeError:
        pass
    for attr, val in (("use_shadows", True), ("use_raytracing", True)):
        if hasattr(scene.eevee, attr):
            setattr(scene.eevee, attr, val)
    scene.render.resolution_x, scene.render.resolution_y = a.width, a.height
    # JPEG, not PNG: 1740 frames at 1080p are about 5 GB as PNG and 0.7 GB as JPEG.
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = 92
    try:
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        pass
    world = bpy.data.worlds.new("sky")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes.get("Background")
    sky = nt.nodes.new("ShaderNodeTexSky")
    for kind in ("MULTIPLE_SCATTERING", "NISHITA", "HOSEK_WILKIE"):
        try:
            sky.sky_type = kind
            break
        except TypeError:
            continue
    if hasattr(sky, "sun_elevation"):
        sky.sun_elevation = math.radians(28)
        sky.sun_rotation = math.radians(140)
    nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 0.22
    scene.view_settings.exposure = -0.9
    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = 3.0
    sun.angle = math.radians(2.0)
    so = bpy.data.objects.new("sun", sun)
    bpy.context.collection.objects.link(so)
    so.rotation_euler = (math.radians(55), 0, math.radians(140 + 180))


def main():
    a = args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    route = json.loads(a.route.read_text(encoding="utf-8"))
    if a.shots:
        SHOTS[:] = [tuple(r) for r in json.loads(a.shots.read_text(encoding="utf-8"))]
    red, green = build_town(route)
    z, objs = build_rider(a.run)
    setup_world(a)
    fps = int(z["fps"])
    n = z["pos"].shape[0]
    scene = bpy.context.scene
    scene.render.fps = fps
    scene.frame_start, scene.frame_end = 1, n
    if a.frames:
        lo, hi = (int(v) for v in a.frames.split("-"))
        scene.frame_start, scene.frame_end = lo, hi
    if a.step > 1:
        scene.frame_step = a.step

    track, yaw = deck_track(z)
    cams, target, heading = make_cameras(track, yaw)
    speed = np.r_[0.0, np.linalg.norm(np.diff(track[:, :2], axis=0), axis=1) * fps]
    cw = route["city"]["crosswalk"]
    near_cross = np.hypot(track[:, 0] - cw["x"], track[:, 1] - (cw["y0"] - 1.0)) < 1.5
    waiting = near_cross & (smooth(speed[:, None], 0.2)[:, 0] < 0.08)
    # The light is red until the crab has waited, then green.
    first_wait = int(np.argmax(waiting)) if waiting.any() else n
    go_frame = first_wait + int(1.5 * fps)

    pos, mats = z["pos"], z["mat"]

    def apply(frame_index):
        f = max(0, min(n - 1, frame_index - 1))
        for i, o in objs:
            o.location = pos[f, i].tolist()
            o.rotation_quaternion = Matrix(mats[f, i].reshape(3, 3).tolist()).to_quaternion()
        tx, ty, tz = target[f]
        h = heading[f]
        _, name, side = shot_for(f, fps)
        if name == "lead":
            ang = h + side * 0.55
            look_at(cams["lead"], (tx + 1.7 * math.cos(ang), ty + 1.7 * math.sin(ang), tz + 0.45),
                    (tx, ty, tz + 0.16))
        elif name == "chase":
            look_at(cams["chase"], (tx - 1.9 * math.cos(h - 0.45), ty - 1.9 * math.sin(h - 0.45), tz + 0.8),
                    (tx + 0.6 * math.cos(h), ty + 0.6 * math.sin(h), tz + 0.1))
        elif name == "wide":
            look_at(cams["wide"], (tx + 5.0 * math.cos(h - 0.9), ty + 5.0 * math.sin(h - 0.9), 2.4),
                    (tx - 0.5, ty + 1.2, 1.0))
        elif name == "close":
            look_at(cams["close"], (tx + 0.95 * math.cos(h + 0.35), ty + 0.95 * math.sin(h + 0.35), tz + 0.3),
                    (tx, ty, tz + 0.17))
        elif name == "crossing":
            # Standing in the side street east of the crossing, clear of the
            # corner building, looking back along the zebra at the light.
            # It pans with the crab from where it stands, so the light, the zebra
            # and the crab stay in one picture while it waits and crosses.
            # From the far kerb, low, looking back across the road: the crab
            # waits and crosses towards the camera, face on, the light above it.
            look_at(cams["crossing"], (cw["x"] - 1.5, cw["y1"] + 0.6, 0.6),
                    (cw["x"] + 0.8, cw["y0"] - 0.4, 1.25))
        elif name == "arrival":
            # Wide on the shop front, the sign in the frame, the crab arriving.
            sx = route["city"]["shop"]["x"]
            look_at(cams["arrival"], (sx - 4.8, 10.2, 1.4), (sx - 1.0, 15.2, 1.9))
        scene.camera = cams[name]
        on_red = f < go_frame
        red.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = 6.0 if on_red else 0.0
        green.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = 0.0 if on_red else 6.0

    def handler(scn, *_):
        apply(scn.frame_current)

    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_pre.append(handler)
    apply(scene.frame_start)

    if a.save:
        bpy.ops.wm.save_as_mainfile(filepath=str(a.save.resolve()))
    a.out.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(a.out.resolve()) + "/"
    bpy.ops.render.render(animation=True)


main()
