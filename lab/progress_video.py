"""Video 1: how the crab learned to ride. One clip per checkpoint, captioned, joined.

Every checkpoint rides the same course (`ride_script.py`), so the only thing
that changes between clips is how much the policy has learned.

Usage (from jumper01/, with the jumper venv's python for play and this venv for
the editing):
    lab/.venv/Scripts/python.exe lab/progress_video.py --run <logs/.../run dir> \
        --iters 0 25 100 300 1000 --out lab/out/progress.mp4
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
import ride_script  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# `jumper/` beside `lab/` while developing, or the repository itself when published.
JUMPER = ROOT / "jumper" if (ROOT / "jumper" / "tasks").is_dir() else ROOT
JUMPER_PY = JUMPER / ".venv" / "Scripts" / "python.exe"
FONT_BOLD = "C:/Windows/Fonts/BIZ-UDGothicB.ttc"
FONT = "C:/Windows/Fonts/BIZ-UDGothicR.ttc"
# The on-screen labels are English only: a Latin face reads better there.
LATIN_BOLD = "C:/Windows/Fonts/segoeuib.ttf"
LATIN = "C:/Windows/Fonts/segoeui.ttf"
FPS = 30
W, H = 960, 540


def record(run: Path, it: int, clip: Path, task: str) -> None:
    if clip.exists():
        return
    cmd = [
        str(JUMPER_PY), str(Path(__file__).parent / "ride_record.py"), "--task", task,
        "--backend", "native", "--device", "cpu", "--headless",
        "--checkpoint", str(run / f"model_{it}.pt"),
        "--steps", str(int(ride_script.DURATION * 50)),
        "--command-script", str(Path(__file__).parent / "ride_script.py"),
        "--scene", "studio",
        "--video", str(clip), "--video-fps", str(FPS),
        "--video-width", str(W), "--video-height", str(H),
        "--video-distance", "1.1", "--video-azimuth", "150", "--video-elevation", "-16",
    ]
    env = {"PYTHONUTF8": "1", **__import__("os").environ}
    done = subprocess.run(cmd, cwd=JUMPER, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if not clip.exists():
        sys.stderr.write((done.stdout or "")[-3000:] + (done.stderr or "")[-3000:])
        raise SystemExit(f"recording iteration {it} failed")


# On-screen labels stay in English or plain numbers; spoken lines are subtitles.
MOOD_LABELS = {"calm": "calm", "happy": "happy", "excited": "excited",
               "focus": "focused", "worried": "worried", "panic": "panic"}


def caption(frame: np.ndarray, it: int, t: float, note: str, mood: str | None) -> np.ndarray:
    img = Image.fromarray(frame).convert("RGBA")
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    big = ImageFont.truetype(LATIN_BOLD, 40)
    small = ImageFont.truetype(LATIN, 24)
    d.rounded_rectangle((20, 20, 380, 120), 14, fill=(15, 18, 24, 170))
    d.text((40, 30), f"iteration {it:,}", font=big, fill=(255, 255, 255, 255))
    d.text((40, 82), note, font=small, fill=(255, 205, 120, 255))
    d.text((W - 150, H - 46), f"{t:4.1f} s", font=small, fill=(255, 255, 255, 220))
    if mood:
        d.rounded_rectangle((W - 250, 20, W - 20, 70), 12, fill=(15, 18, 24, 170))
        d.text((W - 230, 30), f"mood: {MOOD_LABELS.get(mood, mood)}", font=small,
               fill=(120, 230, 255, 255))
    return np.array(Image.alpha_composite(img, layer).convert("RGB"))


def title_card(text: str, sub: str, seconds: float = 2.0) -> list[np.ndarray]:
    img = Image.new("RGB", (W, H), (18, 22, 30))
    d = ImageDraw.Draw(img)
    big = ImageFont.truetype(LATIN_BOLD, 56)
    small = ImageFont.truetype(LATIN, 28)
    d.text((W / 2, H / 2 - 30), text, font=big, fill=(255, 255, 255), anchor="mm")
    d.text((W / 2, H / 2 + 40), sub, font=small, fill=(255, 190, 110), anchor="mm")
    return [np.array(img)] * int(seconds * FPS)


def iteration_stats(log: Path) -> dict[int, dict]:
    """Per-iteration numbers from the training log, keyed by iteration."""
    text = log.read_text(encoding="utf-8", errors="ignore")
    out, cur = {}, None
    for line in text.splitlines():
        m = re.search(r"Learning iteration (\d+)/", line)
        if m:
            cur = int(m.group(1))
            out[cur] = {}
            continue
        if cur is None:
            continue
        m = re.match(r"\s*([A-Za-z_/ ]+?):\s+(-?[\d.]+)\s*$", line)
        if m:
            out[cur][m.group(1).strip()] = float(m.group(2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--iters", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--task", default="jumper.segway")
    ap.add_argument("--thumbs", type=Path, default=None,
                    help="also write one small JPEG per checkpoint here, and an index JSON")
    a = ap.parse_args()
    a.run = a.run.resolve()

    a.out = a.out.resolve()
    work = a.out.parent / "clips"
    work.mkdir(parents=True, exist_ok=True)
    frames = title_card("Jumper rides a segway", "the same course at each stage of training")
    thumbs = []
    for it in a.iters:
        clip = work / f"iter_{it:05d}.mp4"
        record(a.run, it, clip, a.task)
        raw = [f for f in imageio.get_reader(clip)]
        moods_file = Path(str(clip) + ".moods.json")
        moods = json.loads(moods_file.read_text(encoding="utf-8")) if moods_file.exists() else []
        for k, f in enumerate(raw):
            t = k / FPS
            mood = moods[min(k, len(moods) - 1)]["mood"] if moods else None
            frames.append(caption(f[:H, :W], it, t, ride_script.phase(t)[2], mood))
        if a.thumbs is not None:
            a.thumbs.mkdir(parents=True, exist_ok=True)
            pick = raw[min(len(raw) - 1, int(7.5 * FPS))]
            path = a.thumbs / f"iter_{it:05d}.jpg"
            Image.fromarray(pick[:H, :W]).resize((480, 270)).save(path, quality=72)
            thumbs.append({"iter": it, "file": path.name})
    imageio.mimsave(a.out, frames, fps=FPS, macro_block_size=8)
    if a.thumbs is not None:
        (a.thumbs / "index.json").write_text(json.dumps(thumbs), encoding="utf-8")
    print("wrote", a.out, f"{len(frames) / FPS:.1f} s")


if __name__ == "__main__":
    main()
