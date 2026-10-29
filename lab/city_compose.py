"""Join Blender's frames into the city film, with captions from the ride itself.

The scene names come from `lab/city/route.json` (the label of the route point the
crab most recently passed) and the mood from the run's `moods.json`, so every
caption says what the simulation was actually doing at that frame.

    lab/.venv/Scripts/python.exe lab/city_compose.py --frames lab/out/city_frames \
        --run lab/out/city_run --route lab/city/route.json --out lab/out/city.mp4
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_BOLD = "C:/Windows/Fonts/BIZ-UDGothicB.ttc"
FONT = "C:/Windows/Fonts/BIZ-UDGothicR.ttc"
MOOD_LABELS = {"calm": "まったり", "happy": "ニコニコ", "excited": "キラキラ",
               "focus": "しんけん", "worried": "こまり", "panic": "あわわ"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--route", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    z = np.load(a.run / "scene.npz")
    fps = int(z["fps"])
    names = [str(n) for n in z["names"]]
    deck = z["pos"][:, next(k for k, n in enumerate(names) if n.endswith("deck")), :2]
    moods = json.loads((a.run / "moods.json").read_text(encoding="utf-8"))
    route = json.loads(a.route.read_text(encoding="utf-8"))
    labelled = [(p["x"], p["y"], p["label"]) for p in route["points"] if p.get("label")]

    files = sorted(list(a.frames.glob("*.png")) + list(a.frames.glob("*.jpg")))
    big = ImageFont.truetype(FONT_BOLD, 44)
    mid = ImageFont.truetype(FONT_BOLD, 30)
    small = ImageFont.truetype(FONT, 24)
    writer = imageio.get_writer(a.out, fps=fps, macro_block_size=8, quality=8)
    scene_label, shown_since = labelled[0][2], 0
    for k, path in enumerate(files):
        f = int(path.stem) - 1
        img = Image.open(path).convert("RGBA")
        W, H = img.size
        for x, y, label in labelled:
            if math.hypot(deck[f, 0] - x, deck[f, 1] - y) < 0.8 and label != scene_label:
                scene_label, shown_since = label, f
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        # Title for the first 3.5 s.
        t = f / fps
        if t < 3.5:
            alpha = int(255 * min(1.0, (3.5 - t) / 0.6, t / 0.4 if t > 0 else 0))
            d.rounded_rectangle((W / 2 - 330, 60, W / 2 + 330, 190), 24, fill=(20, 24, 32, int(alpha * 0.7)))
            d.text((W / 2, 105), route["name"], font=big, fill=(255, 255, 255, alpha), anchor="mm")
            d.text((W / 2, 158), "カニちゃん、セグウェイでおつかいへ", font=small,
                   fill=(255, 205, 130, alpha), anchor="mm")
        # The scene, for 3 s after it begins.
        since = (f - shown_since) / fps
        if t >= 3.5 and since < 3.0:
            alpha = int(255 * min(1.0, since / 0.3, (3.0 - since) / 0.5))
            d.rounded_rectangle((40, H - 120, 40 + 30 * len(scene_label) + 60, H - 50), 18,
                                fill=(20, 24, 32, int(alpha * 0.7)))
            d.text((70, H - 85), scene_label, font=mid, fill=(255, 255, 255, alpha), anchor="lm")
        mood = moods[min(f, len(moods) - 1)]
        d.rounded_rectangle((W - 270, 30, W - 30, 82), 14, fill=(20, 24, 32, 150))
        d.text((W - 250, 56), f"気分：{MOOD_LABELS.get(mood, mood)}", font=small,
               fill=(120, 230, 255, 255), anchor="lm")
        writer.append_data(np.asarray(Image.alpha_composite(img, layer).convert("RGB")))
    writer.close()
    print("wrote", a.out, f"{len(files) / fps:.1f} s")


if __name__ == "__main__":
    main()
