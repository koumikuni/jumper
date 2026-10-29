"""Write the dashboard's data files from the training log and the checkpoint thumbnails.

    lab/.venv/Scripts/python.exe lab/dash_data.py --log lab/train_segway_v1.log \
        --thumbs lab/out/thumbs --out lab/out/dash

Writes training.json (one row per iteration), snapshots.json (one row per
thumbnail, the image inline as a data URL) and milestones.json (the project plan,
from MILESTONES below -- edit it there).
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from progress_video import iteration_stats  # noqa: E402

CONTROL_HZ = 50
EPISODE_S = 20.0
#: Reward weights in tasks/jumper/segway/env_cfg.py, to turn a logged episode
#: reward back into the per-step tracking term, 0..1.
TRACK_WEIGHTS = {"speed": 3.0, "yaw": 1.5}
FALLS = ("Episode_Termination/board_over", "Episode_Termination/off_the_deck",
         "Episode_Termination/tipped_on_the_deck")

MILESTONES = [
    # (id, phase, item, status, date)
    ("m01", "準備", "ルールと機体の調査", "完了", "2026-10-21"),
    ("m02", "準備", "学習環境（GPU）の構築", "完了", "2026-10-21"),
    ("m03", "乗り物A セグウェイ", "試作機で倒立制御を調整", "完了", "2026-10-21"),
    ("m04", "乗り物A セグウェイ", "学習タスク jumper.segway を実装", "完了", "2026-10-21"),
    ("m05", "乗り物A セグウェイ", "本番の学習（3000回）", "完了", "2026-10-21"),
    ("m06", "乗り物A セグウェイ", "動画①「学習の様子」の制作", "進行中", "2026-10-22"),
    ("m06b", "乗り物A セグウェイ", "カニちゃんの表情（気分の判定と顔の表示）", "完了", "2026-10-21"),
    ("m07", "乗り物A セグウェイ", "見た目（スキン・シーン）の仕上げ", "予定", "2026-10-25"),
    ("m08", "第1ラウンド", "動画①を X / YouTube に投稿", "予定", "2026-10-30"),
    ("m09", "乗り物B・C", "キックボード型・一輪型を実装", "予定", "2026-11-10"),
    ("m10", "乗り物B・C", "動画②「インタビュー」の制作", "予定", "2026-11-15"),
    ("m11", "改善", "乗り物の設計パラメータを改良し再学習", "予定", "2026-11-25"),
    ("m12", "改善", "動画③「乗り物の改善」とコード公開", "予定", "2026-11-30"),
]


def tracking(s: dict, axis: str) -> float | None:
    ridden = s["Mean episode length"] / CONTROL_HZ / EPISODE_S
    value = s.get(f"Episode_Reward/track_ride_{axis}")
    if value is None or ridden <= 0:
        return None
    return round(min(1.0, value / TRACK_WEIGHTS[axis] / ridden), 4)


def training_rows(log: Path) -> list[dict]:
    rows = []
    for it, s in sorted(iteration_stats(log).items()):
        if "Mean episode length" not in s:
            continue
        ends = sum(s.get(k, 0.0) for k in FALLS) + s.get("Episode_Termination/time_out", 0.0)
        falls = sum(s.get(k, 0.0) for k in FALLS)
        rows.append({
            "iter": it,
            "reward": round(s.get("Mean reward", 0.0), 3),
            "ride_seconds": round(s["Mean episode length"] / CONTROL_HZ, 2),
            # `Episode_Reward/x` is the episode sum over the *maximum* episode
            # length, so dividing by the weight and the share of the episode
            # actually ridden gives the mean of the 0..1 tracking term.
            "speed_tracking": tracking(s, "speed"),
            "yaw_tracking": tracking(s, "yaw"),
            "fall_share": round(falls / ends, 4) if ends > 0 else None,
        })
    return rows


def snapshot_rows(thumbs: Path, training: list[dict]) -> list[dict]:
    by_iter = {r["iter"]: r for r in training}
    rows = []
    for item in json.loads((thumbs / "index.json").read_text(encoding="utf-8")):
        data = base64.b64encode((thumbs / item["file"]).read_bytes()).decode()
        near = by_iter.get(item["iter"]) or by_iter.get(max(item["iter"] - 1, 0), {})
        rows.append({
            "iter": item["iter"],
            "ride_seconds": near.get("ride_seconds"),
            "image": f"data:image/jpeg;base64,{data}",
        })
    return rows


MOOD_ORDER = ("panic", "worried", "focus", "calm", "happy", "excited")
MOOD_LABELS = {"calm": "まったり", "happy": "ニコニコ", "excited": "キラキラ",
               "focus": "しんけん", "worried": "こまり", "panic": "あわわ"}


def mood_rows(clips: Path) -> list[dict]:
    """Share of recorded frames in each mood, per checkpoint clip."""
    rows = []
    for f in sorted(clips.glob("iter_*.mp4.moods.json")):
        it = int(f.name.split("_")[1].split(".")[0])
        moods = [r["mood"] for r in json.loads(f.read_text(encoding="utf-8"))]
        for k, mood in enumerate(MOOD_ORDER):
            rows.append({"iter": it, "order": k, "mood": MOOD_LABELS[mood],
                         "share": round(moods.count(mood) / len(moods), 4)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--thumbs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--clips", type=Path, default=None)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    training = training_rows(a.log)
    (a.out / "training.json").write_text(json.dumps(training), encoding="utf-8")
    (a.out / "snapshots.json").write_text(
        json.dumps(snapshot_rows(a.thumbs, training)), encoding="utf-8")
    if a.clips is not None:
        (a.out / "moods.json").write_text(json.dumps(mood_rows(a.clips), ensure_ascii=False),
                                          encoding="utf-8")
    keys = ("id", "phase", "item", "status", "date")
    (a.out / "milestones.json").write_text(
        json.dumps([dict(zip(keys, m)) for m in MILESTONES], ensure_ascii=False),
        encoding="utf-8")
    print(f"{len(training)} iterations, last {training[-1] if training else None}")


if __name__ == "__main__":
    main()
