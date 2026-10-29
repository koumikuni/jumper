"""Top-view picture of an exported .3dm's keep regions, to check before opening Rhino."""
import sys
import rhino3dm as r3
from PIL import Image, ImageDraw, ImageFont

m = r3.File3dm.Read(sys.argv[1])
names = {i: l.Name for i, l in enumerate(m.Layers)}
S, W, H = 0.6, 1400, 560
img = Image.new("RGB", (W, H), (250, 250, 252)); d = ImageDraw.Draw(img)
F = ImageFont.truetype("C:/Windows/Fonts/BIZ-UDGothicB.ttc", 20)
f = ImageFont.truetype("C:/Windows/Fonts/BIZ-UDGothicR.ttc", 15)
titles = ["乗り込みで触れた範囲", "操縦中に触れた範囲", "合わせた範囲（残すべき部分）"]
for k, t in enumerate(titles):
    d.text((k * 466 + 20, 12), t, font=F, fill=(30, 30, 40))
d.text((20, H - 30), "上が前（+X）。細線=触れた範囲、太線=接触の90%が収まる範囲、灰=穴", font=f, fill=(90, 90, 100))
col = {"10_Keep_boarding": (230, 80, 60), "11_Keep_riding": (60, 110, 230), "12_Keep_union": (200, 30, 160),
       "13_Keep_frequent_90pct": (60, 0, 60), "14_Keep_holes": (150, 150, 150), "03_Deck_current": (43, 181, 176)}
for o in m.Objects:
    ln, nm, g = names[o.Attributes.LayerIndex], o.Attributes.Name, o.Geometry
    if not isinstance(g, r3.PolylineCurve):
        continue
    if ln == "03_Deck_current":
        panels = [0, 1, 2]
    else:
        panels = [0] if nm.startswith("boarding") else [1] if nm.startswith("riding") else [2] if nm.startswith("keep") else []
    pts = [g.Point(i) for i in range(g.PointCount)]
    for k in panels:
        cx, cy = k * 466 + 233, 290
        d.line([(cx - p.Y * S, cy - p.X * S) for p in pts], fill=col.get(ln, (0, 0, 0)),
               width=3 if ln == "13_Keep_frequent_90pct" else 2)
img.save(sys.argv[2])
