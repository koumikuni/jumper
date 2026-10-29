"""Cut the Round 1 film from the recorded clips, following the storyboard doc.

Every clip is a recording of the simulation (MuJoCo replays and the Blender town,
which draws recorded poses); this only cuts, scales, captions and scores them.

    lab/.venv/Scripts/python.exe lab/make_video.py --city lab/out/city_v2.mp4 \
        --city-marks 4,16,30,46,62 --url github.com/<you>/jumper [--music] --out lab/out/crab_segway_60s.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
import bgm  # noqa: E402

OUT = Path(__file__).parent / "out"
W, H, FPS = 1920, 1080, 30
BOLD = "C:/Windows/Fonts/BIZ-UDGothicB.ttc"
REG = "C:/Windows/Fonts/BIZ-UDGothicR.ttc"
# Labels and cards are English only: a Latin face reads better there.
LATIN_BOLD = "C:/Windows/Fonts/segoeuib.ttf"
LATIN = "C:/Windows/Fonts/segoeui.ttf"
INK = (24, 28, 36)
ACCENT = (255, 196, 92)
TEAL = (43, 181, 176)
RED = (226, 66, 52)


def font(size, bold=True, latin=False):
    if latin:
        return ImageFont.truetype(LATIN_BOLD if bold else LATIN, size)
    return ImageFont.truetype(BOLD if bold else REG, size)


# ── Sources ──────────────────────────────────────────────────────────────


class Clip:
    """One recorded file, read frame by frame on demand and scaled to fill 1920x1080.

    Frames are fetched by index, not loaded whole: the learning-progress film
    alone is 100 s of 960x540, several gigabytes decoded.
    """

    _readers: dict = {}

    def __init__(self, path: Path):
        self.path = path
        if path not in Clip._readers:
            reader = imageio.get_reader(path)
            Clip._readers[path] = (reader, reader.get_meta_data().get("fps", 30), reader.count_frames())
        self.reader, self.fps, self.count = Clip._readers[path]

    def take(self, t0: float, t1: float, speed: float = 1.0):
        """Frames from t0 to t1 seconds of the source, at `speed` x."""
        n = int(round((t1 - t0) / speed * FPS))
        for k in range(n):
            src = t0 + k * speed / FPS
            i = min(self.count - 1, int(src * self.fps))
            yield fit(self.reader.get_data(i))


def fit(frame: np.ndarray) -> Image.Image:
    img = Image.fromarray(frame[..., :3])
    s = max(W / img.width, H / img.height)
    img = img.resize((int(img.width * s + 0.5), int(img.height * s + 0.5)), Image.LANCZOS)
    x, y = (img.width - W) // 2, (img.height - H) // 2
    return img.crop((x, y, x + W, y + H))


# ── Captions and cards ───────────────────────────────────────────────────


def caption(img: Image.Image, text: str, alpha: float = 1.0, size: int = 64, pos: str = "bottom") -> Image.Image:
    """A caption -- `pos` "bottom", "top" (the crab low in the frame) or "mid"
    (the crab low and a sign up top): white type on a dark rounded band."""
    if not text or alpha <= 0:
        return img
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    f = font(size)
    box = d.textbbox((0, 0), text, font=f)
    tw, th = box[2] - box[0], box[3] - box[1]
    x = (W - tw) // 2
    y = {"top": 120, "mid": int(H * 0.56)}.get(pos, H - 140 - th)
    a = int(255 * alpha)
    d.rounded_rectangle((x - 40, y - 26, x + tw + 40, y + th + 30), 28, fill=(16, 20, 28, int(170 * alpha)))
    d.text((x, y - box[1]), text, font=f, fill=(255, 255, 255, a))
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


def bubble(img: Image.Image, text: str, anchor=(1180, 300), alpha: float = 1.0) -> Image.Image:
    """A speech bubble, the crab's own words."""
    if alpha <= 0:
        return img
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    f = font(72)
    box = d.textbbox((0, 0), text, font=f)
    tw, th = box[2] - box[0], box[3] - box[1]
    x, y = anchor
    a = int(255 * alpha)
    d.rounded_rectangle((x - 40, y - 34, x + tw + 40, y + th + 40), 46, fill=(255, 255, 255, a),
                        outline=(*RED, a), width=6)
    d.polygon([(x + 30, y + th + 36), (x - 30, y + th + 120), (x + 110, y + th + 38)],
              fill=(255, 255, 255, a), outline=(*RED, a))
    d.text((x, y - box[1]), text, font=f, fill=(*RED, a))
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


def corner(img: Image.Image, text: str, color=ACCENT) -> Image.Image:
    """A small label top-left: what this cut is measuring."""
    if not text:
        return img
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    f = font(44, latin=True)
    box = d.textbbox((0, 0), text, font=f)
    d.rounded_rectangle((40, 40, 40 + box[2] - box[0] + 56, 40 + box[3] - box[1] + 40), 22,
                        fill=(16, 20, 28, 175))
    d.text((68, 60 - box[1]), text, font=f, fill=(*color, 255))
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


def card(seconds: float, draw) -> list:
    frames = []
    n = int(seconds * FPS)
    for k in range(n):
        img = Image.new("RGB", (W, H), (20, 24, 32))
        draw(ImageDraw.Draw(img), k / FPS, img)
        frames.append(img)
    return frames


def title_card(seconds=3.0):
    def draw(d, t, img):
        a = min(1.0, t / 0.4)
        c = tuple(int(v * a) for v in (255, 255, 255))
        d.text((W / 2, H / 2 - 50), "Jumper rides a segway", font=font(120, latin=True), fill=c, anchor="mm")
        d.text((W / 2, H / 2 + 80), "learned by itself, in physics simulation", font=font(48, False, latin=True),
               fill=tuple(int(v * a) for v in ACCENT), anchor="mm")
    return card(seconds, draw)


def steps_card(seconds: float, rows: list):
    """Step height against how often the crab got on, bar by bar."""
    def draw(d, t, img):
        d.text((W / 2, 120), "step height  →  got on by itself", font=font(72, latin=True), fill=(255, 255, 255), anchor="mm")
        x0, base, bw, gap = 260, 880, 120, 60
        shown = int(min(len(rows), t / (seconds * 0.7) * len(rows) + 1))
        for i, (h, share) in enumerate(rows[:shown]):
            x = x0 + i * (bw + gap)
            top = base - h * 38
            d.rounded_rectangle((x, top, x + bw, base), 14, fill=TEAL)
            d.text((x + bw / 2, base + 50), f"{h:g}cm", font=font(40, latin=True), fill=(255, 255, 255), anchor="mm")
            d.text((x + bw / 2, top - 40), f"{share:.0%}", font=font(38, latin=True), fill=ACCENT, anchor="mm")
        d.text((W / 2, base + 120), "share of 128 starts from the ground", font=font(34, False, latin=True),
               fill=(200, 205, 215), anchor="mm")
    return card(seconds, draw)


def end_card(seconds: float, url: str):
    def draw(d, t, img):
        d.text((W / 2, 330), "policies + board designs", font=font(84, latin=True), fill=(255, 255, 255), anchor="mm")
        d.text((W / 2, 450), "all open source", font=font(96, latin=True), fill=ACCENT, anchor="mm")
        d.text((W / 2, 620), url, font=font(52, False, latin=True), fill=(220, 225, 235), anchor="mm")
        d.text((W / 2, 760), "#Jumper  #CrabRobot  #OpenSourceRobot", font=font(56, latin=True), fill=TEAL, anchor="mm")
        d.text((W / 2, 900), "every motion is recorded physics simulation — no hand animation",
               font=font(34, False, latin=True), fill=(170, 175, 185), anchor="mm")
    return card(seconds, draw)


# ── The cut ──────────────────────────────────────────────────────────────


def build(city: Path, marks: list, url: str):
    """The storyboard, cut by cut: (frames, caption, corner label, bubble, sound cue)."""
    prog = Clip(OUT / "progress_v2.mp4")
    seg = lambda k, a, b, speed=1.0: prog.take(2.0 + k * 14.0 + a, 2.0 + k * 14.0 + b, speed)  # noqa: E731
    cityc = Clip(city)
    lead = Clip(OUT / "city_lead_21s.mp4")
    start, corner_t, cross_t, shops_t, arrive_t = marks
    evals = []
    for h in (6.5, 8, 9, 10, 11, 12, 13, 14):
        p = OUT / f"eval_lowcell_h{h:g}.json"
        if p.exists():
            evals.append((h, json.loads(p.read_text(encoding="utf-8"))["on_and_riding_at_end"]))

    cuts = [
        # (frames, caption, corner label, bubble, cue at start)
        (cityc.take(start + 6.0, start + 10.0), "カニが、セグウェイに乗ってる。", "", None, "pop"),
        (title_card(3.0), "", "", None, "sparkle"),
        (seg(0, 2.0, 6.0), "最初は… あわわ", "", None, "boing"),
        (list(seg(2, 4.0, 6.0)) + list(seg(6, 6.0, 8.0)), "3000回 練習したら… 乗れた！", "", None, "sparkle"),
        (Clip(OUT / "board_100.mp4").take(2.0, 7.0), "でも、ひとりで乗るのは…", "getting on: training", ("グラグラするよ！", 1.2), "puff"),
        (Clip(OUT / "board_9999_side.mp4").take(1.0, 5.0), "一晩 練習したら… 隠れちゃった", "10,000 iterations", None, "boing"),
        (Clip(OUT / "board_v2_lowfloor.mp4").take(1.0, 3.5), "ボードのほうを 直していく", "low floor, through axle → 0%", None, "pop"),
        (Clip(OUT / "board_v2_kneel.mp4").take(1.0, 3.5), "ボードのほうを 直していく", "kneeling deck → 0%", None, "pop"),
        (Clip(OUT / "board_v3_lowfloor_hub.mp4").take(0.0, 3.0), "棒をなくしたら… 乗れた！", "hub motors, no axle → 99%", None, "sparkle"),
        (steps_card(2.5, evals), "", "", None, "pop"),
        (Clip(OUT / "board_v7_h14.mp4").take(0.0, 1.75, speed=0.5), "14cmも、ひょいっ。", "step 14 cm → 99%", None, "sparkle"),
        (cityc.take(start, start + 4.0), "じゃあ、おつかいへ。", "", None, "pop"),
        # The crossing in three shots: waiting face on, the light going green
        # (the town's crossing camera), crossing face on. `lead` is a second
        # render of the same recorded ride from the camera ahead of the crab.
        (list(lead.take(cross_t - 1.0 - LEAD_T0, cross_t + 1.0 - LEAD_T0))
         + list(cityc.take(cross_t + 1.3, cross_t + 3.3))
         + list(lead.take(cross_t + 5.0 - LEAD_T0, cross_t + 7.0 - LEAD_T0)),
         "信号、まってます", "", None, "pop", "top"),
        # Arriving: wide on the shop's sign with the caption up top, clear of the
        # crab low in the frame; then the close-up.
        (list(cityc.take(arrive_t - 4.0, arrive_t - 0.5)) + list(cityc.take(arrive_t + 2.5, arrive_t + 4.0)),
         "とうちゃく！", "", None, "sparkle", "mid"),
        (end_card(6.0, url), "", "", None, None),
    ]
    return cuts


def render(cuts, write):
    """Caption every cut's frames and hand them to `write`, one at a time.
    Returns the film's length in seconds and its sound cues as (second, kind)."""
    cues, t = [], 0.0
    for clip, cap, label, bub, cue, *rest in cuts:
        pos = rest[0] if rest else "bottom"
        clip = list(clip)
        n = len(clip)
        if cue:
            cues.append((t, cue))
        if bub:
            cues.append((t + bub[1], "puff"))
        for k, img in enumerate(clip):
            local = k / FPS
            fade = min(1.0, local / 0.25, (n - k) / FPS / 0.25)
            img = corner(img, label)
            img = caption(img, cap, alpha=fade, pos=pos)
            if bub and local >= bub[1]:
                img = bubble(img, bub[0], alpha=min(1.0, (local - bub[1]) / 0.2))
            write(np.asarray(img))
        t += n / FPS
    return t, cues


#: The 30 s version, as indices into `build`'s list (storyboard #1, #3, #4, #5,
#: #8 graph + 14 cm, #11, #12).
SHORT = (0, 2, 3, 4, 10, 13, 14)

#: Where `city_lead_21s.mp4` starts in the ride: it is frames 630 - 900.
LEAD_T0 = 629 / 30


def score(seconds: float, cues: list) -> np.ndarray:
    music = bgm.build(seconds)
    for at, kind in cues:
        bgm.add(music, bgm.sfx(kind) * 0.9, at)
    return music / max(1e-6, np.abs(music).max()) * 0.8


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", type=Path, required=True)
    ap.add_argument("--city-marks", required=True,
                    help="start,corner,crosswalk,shops,arrival in seconds of the city film")
    ap.add_argument("--url", default="github.com/koumikuni/jumper")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--music", action="store_true",
                    help="add the generated soundtrack and cues (lab/bgm.py); silent otherwise")
    ap.add_argument("--short-only", action="store_true",
                    help="the 30 s cut: storyboard cuts 1, 3-4, 5, 8, 11, 12")
    a = ap.parse_args()

    marks = [float(v) for v in a.city_marks.split(",")]
    cuts = build(a.city, marks, a.url)
    if a.short_only:
        cuts = [cuts[i] for i in SHORT]
    silent = a.out.with_suffix(".silent.mp4") if a.music else a.out
    w = imageio.get_writer(silent, fps=FPS, codec="libx264", quality=8, macro_block_size=8,
                           ffmpeg_params=["-pix_fmt", "yuv420p"])
    seconds, cues = render(cuts, w.append_data)
    w.close()
    if a.music:
        wav = a.out.with_suffix(".wav")
        bgm.write_wav(wav, score(seconds, cues))
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(silent), "-i", str(wav), "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "192k", "-shortest", str(a.out)], check=True)
    print(f"wrote {a.out} ({seconds:.1f} s, {len(cues)} sound cues)")


if __name__ == "__main__":
    main()
