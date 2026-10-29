"""The crab's face: a screen on the display module that shows how the ride is going.

**Replay only, and nothing reads it.** The screen is a visual quad added to the
robot's spec when `env_cfg(play=True)` builds it; it has no mass and no contact,
and no observation, reward or termination looks at it, so a policy trained
without it replays identically with it.

**The expression is judged from what the robot can feel, plus the command.**

    tilt        the body's tilt from vertical, from `projected_gravity` (the IMU)
    wobble      the body's roll and pitch rate, from the gyro
    pitch       the board's pitch -- large means a nosedive or a tail-strike
    error       how far the board's speed and turn are from the ride command,
                each against the width its tracking reward uses

and the mood follows, roughly from bad to good:

    panic     あわわ    tilting or wobbling hard, or the board pitching over
    angry     おこ      (getting on only) the board got away, rocked, or the robot
                        slipped off it -- see `BoardingMood`
    worried   こまり    well off the command
    focus     しんけん  a little off the command
    excited   キラキラ  on the command and going fast
    happy     ニコニコ  on the command
    calm      まったり  asked to stand still, and standing still

Panic shows at once; every other change has to hold for `HOLD_S` first, so the
face does not flicker at the boundary between two moods.

Nothing here can be drawn by `play.py` on its own: the screen is a texture, and
MuJoCo only shows a changed texture once it is uploaded to the renderer's GL
context. `lab/ride_record.py` does that for each recorded frame.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

#: Texture size. The display's lit area is 107 x 34 mm, so 3.15 : 1.
TEX_W, TEX_H = 252, 80
TEXTURE = "face_tex"

#: The quad, in `display_module_link`'s frame. The module spans x -14..10.5 mm,
#: y +/-53.6 mm, z -18..15.6 mm there and faces +x; the quad sits 0.3 mm proud of
#: its front face so it never z-fights with the mesh.
_FRONT_X = 0.0108
_HALF_Y, _Z_LO, _Z_HI = 0.050, -0.016, 0.0145

HOLD_S = 0.4
MOODS = ("calm", "happy", "excited", "focus", "worried", "panic", "angry")
LABELS = {
    "calm": "まったり", "happy": "ニコニコ", "excited": "キラキラ",
    "focus": "しんけん", "worried": "こまり", "panic": "あわわ", "angry": "おこ",
}
#: What the crab says when it is angry, by what made it so.
COMPLAINTS = {
    "fled": "にげないでよ〜！",
    "rocked": "グラグラするよ！",
    "slipped": "ひとりじゃ のりにくいよ！",
}


def add_face_screen(spec: Any) -> Any:
    """Add the screen quad, its texture and material to the robot's spec."""
    import mujoco

    tex = spec.add_texture(name=TEXTURE, type=mujoco.mjtTexture.mjTEXTURE_2D,
                           width=TEX_W, height=TEX_H, nchannel=3)
    tex.data = np.zeros(TEX_W * TEX_H * 3, dtype=np.uint8).tobytes()
    mat = spec.add_material(name="face_mat", emission=1.0, specular=0.2)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB.value] = TEXTURE
    # Seen from the front the robot's right (-y) is on the right of the picture,
    # so u runs towards -y; v runs up.
    hy, cz = _HALF_Y, (_Z_LO + _Z_HI) / 2
    hz = (_Z_HI - _Z_LO) / 2
    mesh = spec.add_mesh(name="face_quad")
    mesh.uservert = [0, hy, -hz, 0, -hy, -hz, 0, -hy, hz, 0, hy, hz]
    # Both windings, so the quad is drawn whichever way MuJoCo decides it faces.
    mesh.userface = [0, 1, 2, 0, 2, 3, 0, 2, 1, 0, 3, 2]
    mesh.usertexcoord = [0, 0, 1, 0, 1, 1, 0, 1]
    mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    body = spec.body("display_module_link")
    body.add_geom(name="face_screen", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="face_quad",
                  material="face_mat", pos=[_FRONT_X, 0.0, cz], mass=0.0,
                  contype=0, conaffinity=0, group=2)
    return spec


# ── Judging the mood ─────────────────────────────────────────────────────


class Mood:
    """Turns the ride's state into a mood, with a hold so the face stays put."""

    def __init__(self) -> None:
        self.mood = "calm"
        self._pending, self._since = "calm", 0.0
        self._err = 0.0

    def update(self, env: Any, index: int, t: float) -> str:
        import torch

        from .mdp import state

        with torch.inference_mode():
            robot = env.scene["robot"]
            g = robot.data.projected_gravity_b[index]
            tilt = math.degrees(math.acos(float(max(-1.0, min(1.0, -g[2])))))
            w = robot.data.root_link_ang_vel_b[index]
            wobble = math.hypot(float(w[0]), float(w[1]))
            pitch = abs(math.degrees(float(state.pitch(env)[index])))
            cmd = env.command_manager.get_command("ride")[index]
            v_err = abs(float(state.ground_speed(env)[index]) - float(cmd[0]))
            w_err = abs(float(state.yaw_rate(env)[index]) - float(cmd[2]))
            moving = abs(float(cmd[0])) > 0.05 or abs(float(cmd[2])) > 0.1
            fast = abs(float(cmd[0])) >= 0.5

        # The widths of `track_ride_speed` and `track_ride_yaw` (env_cfg.py).
        err = v_err / 0.25 + w_err / 0.4
        self._err += 0.15 * (err - self._err)

        if tilt > 25.0 or wobble > 3.0 or pitch > 22.0:
            want = "panic"
        elif self._err > 2.0:
            want = "worried"
        elif self._err > 0.9:
            want = "focus"
        elif not moving:
            want = "calm"
        else:
            want = "excited" if fast else "happy"

        if want == "panic":
            self.mood, self._pending, self._since = want, want, t
        elif want != self._pending:
            self._pending, self._since = want, t
        elif want != self.mood and t - self._since >= HOLD_S:
            self.mood = want
        return self.mood


class BoardingMood:
    """The mood while getting on the parked board (jumper.segway_board).

    **Angry is earned, not scripted**: it fires on one of three things the
    simulation actually did, and `reason` says which --

        fled      the board moved off at more than FLEE_SPEED while the robot
                  was not yet on it: a foot pushed it away
        rocked    the board tipped past ROCK_DEG while being climbed: the seesaw
        slipped   at least two feet had been on the deck and now none are

    -- and it stays for ANGRY_S, because a crab that has just watched its board
    roll away does not cheer up the next frame.
    """

    FLEE_SPEED = 0.10
    ROCK_DEG = 12.0
    ANGRY_S = 1.8

    def __init__(self) -> None:
        self.mood, self.reason = "focus", None
        self._angry_until = -1.0
        self._best_feet = 0.0
        self._pending, self._since = "focus", 0.0

    def update(self, env: Any, index: int, t: float) -> str:
        import torch

        from ..segway_board import mdp as board_mdp
        from .mdp import firmware, state

        with torch.inference_mode():
            robot = env.scene["robot"]
            g = robot.data.projected_gravity_b[index]
            tilt = math.degrees(math.acos(float(max(-1.0, min(1.0, -g[2])))))
            feet = float(firmware.feet_on_deck_count(env)[index])
            on = bool(firmware.engaged(env)[index])
            board_v = float(env.scene[state.ENTITY].data.root_link_lin_vel_w[index, :2].norm())
            board_tilt = math.degrees(float(state.tilt(env)[index]))
            done = bool(board_mdp.boarded(env)[index])

        reason = None
        if not done and board_v > self.FLEE_SPEED:
            reason = "fled"
        elif not done and board_tilt > self.ROCK_DEG:
            reason = "rocked"
        elif self._best_feet >= 2 and feet == 0:
            reason = "slipped"
        self._best_feet = 0.0 if feet == 0 and reason == "slipped" else max(self._best_feet, feet)
        if reason is not None:
            self._angry_until, self.reason = t + self.ANGRY_S, reason

        if tilt > 35.0:
            want = "panic"
        elif t < self._angry_until:
            want = "angry"
        elif done and on:
            want = "excited"
        elif done:
            want = "happy"
        elif feet >= 1:
            want = "worried" if board_tilt > 6.0 else "focus"
        else:
            want = "focus"

        if want in ("panic", "angry"):
            self.mood, self._pending, self._since = want, want, t
        elif want != self._pending:
            self._pending, self._since = want, t
        elif want != self.mood and t - self._since >= HOLD_S:
            self.mood = want
        if self.mood != "angry":
            self.reason = None
        return self.mood


# ── Drawing the face ─────────────────────────────────────────────────────

_BG = (6, 10, 16)
_EYE = (80, 230, 255)
_CHEEK = (255, 110, 150)
_STAR = (255, 215, 90)
_SWEAT = (120, 180, 255)
_ANGER = (255, 70, 60)


def draw(mood: str, t: float) -> np.ndarray:
    """The face for `mood` at time `t`, (TEX_H, TEX_W, 3) uint8, top row first."""
    from PIL import Image, ImageDraw

    s = 4  # supersample, then shrink, for smooth edges
    W, H = TEX_W * s, TEX_H * s
    img = Image.new("RGB", (W, H), _BG)
    d = ImageDraw.Draw(img)
    lx, rx, cy = W * 0.30, W * 0.70, H * 0.48
    ew, eh = W * 0.085, H * 0.30
    blink = (t % 3.2) < 0.14

    def oval(x, y, w, h, fill=_EYE):
        d.ellipse((x - w, y - h, x + w, y + h), fill=fill)

    def arc_eye(x, y, up=True):
        box = (x - ew * 1.3, y - eh * 0.9, x + ew * 1.3, y + eh * 1.1)
        d.arc(box, 200 if up else 20, 340 if up else 160, fill=_EYE, width=int(H * 0.09))

    def star(x, y, r):
        pts = []
        for k in range(10):
            a = -math.pi / 2 + k * math.pi / 5
            rr = r if k % 2 == 0 else r * 0.45
            pts.append((x + rr * math.cos(a), y + rr * math.sin(a)))
        d.polygon(pts, fill=_STAR)

    def cheeks():
        for x in (lx - ew * 1.6, rx + ew * 1.6):
            d.ellipse((x - ew * 0.9, H * 0.68, x + ew * 0.9, H * 0.82), fill=_CHEEK)

    if mood == "calm":
        for x in (lx, rx):
            if blink:
                d.rounded_rectangle((x - ew, cy - H * 0.03, x + ew, cy + H * 0.03), H * 0.03, fill=_EYE)
            else:
                oval(x, cy, ew, eh)
    elif mood == "happy":
        arc_eye(lx, cy)
        arc_eye(rx, cy)
        cheeks()
    elif mood == "excited":
        r = eh * (1.25 + 0.12 * math.sin(t * 9.0))
        star(lx, cy, r)
        star(rx, cy, r)
        cheeks()
    elif mood == "focus":
        for x, sgn in ((lx, 1), (rx, -1)):
            oval(x, cy + eh * 0.25, ew, eh * 0.45)
            d.line((x - ew * 1.2, cy - eh * 0.9 - sgn * eh * 0.25, x + ew * 1.2,
                    cy - eh * 0.9 + sgn * eh * 0.25), fill=_EYE, width=int(H * 0.06))
    elif mood == "worried":
        # Inner ends raised: the opposite slant to "angry".
        for x, sgn in ((lx, 1), (rx, -1)):
            oval(x, cy + eh * 0.1, ew * 0.85, eh * 0.8)
            d.line((x - ew * 1.2, cy - eh * 1.05 + sgn * eh * 0.35, x + ew * 1.2,
                    cy - eh * 1.05 - sgn * eh * 0.35), fill=_EYE, width=int(H * 0.06))
        mx, my = W * 0.5, H * 0.80
        d.line([(mx - W * 0.05 + k * W * 0.0125, my + (H * 0.03 if k % 2 else -H * 0.03))
                for k in range(9)], fill=_EYE, width=int(H * 0.04))
    elif mood == "panic":
        spin = t * 12.0
        for x in (lx, rx):
            for k, rr in enumerate((1.0, 0.68, 0.36)):
                r = eh * 1.1 * rr
                start = math.degrees(spin) + k * 60
                d.arc((x - r, cy - r, x + r, cy + r), start, start + 300,
                      fill=_EYE, width=int(H * 0.05))
        sx, sy = W * 0.88, H * (0.25 + 0.06 * math.sin(t * 10))
        d.polygon([(sx, sy - H * 0.12), (sx - W * 0.025, sy + H * 0.02), (sx + W * 0.025, sy + H * 0.02)],
                  fill=_SWEAT)
        d.ellipse((sx - W * 0.025, sy - H * 0.04, sx + W * 0.025, sy + H * 0.10), fill=_SWEAT)
    elif mood == "angry":
        shake = math.sin(t * 40.0) * W * 0.004
        for x, sgn in ((lx + shake, 1), (rx + shake, -1)):
            # Half-closed glaring eyes under brows that dive towards the nose.
            d.pieslice((x - ew, cy - eh * 0.6, x + ew, cy + eh * 0.9), 0, 180, fill=_ANGER)
            d.line((x - sgn * ew * 1.3, cy - eh * 0.95, x + sgn * ew * 1.1, cy - eh * 0.35),
                   fill=_ANGER, width=int(H * 0.08))
        # The anger mark in the corner, four bent strokes.
        vx, vy, r = W * 0.88, H * 0.26, H * 0.15
        # Each stroke is the quarter of a circle centred off the mark's corner,
        # bowed towards the mark's centre -- the cross-shaped vein of manga.
        for dx, dy, start in ((-1, -1, 0), (1, -1, 90), (1, 1, 180), (-1, 1, 270)):
            cx, cy2 = vx + dx * r, vy + dy * r
            d.arc((cx - r, cy2 - r, cx + r, cy2 + r), start + 12, start + 78,
                  fill=_ANGER, width=int(H * 0.04))
        d.line((W * 0.44, H * 0.83, W * 0.5, H * 0.78, W * 0.56, H * 0.83), fill=_ANGER,
               width=int(H * 0.05))
    return np.asarray(img.resize((TEX_W, TEX_H), Image.LANCZOS), dtype=np.uint8)
