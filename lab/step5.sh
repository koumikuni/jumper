#!/bin/sh
# Shape step 5 (stowing ramp flaps) and riding test, end to end.
cd "$(dirname "$0")/../jumper"
L=logs/jumper/jumper.segway_board
for D in flap_h9_l22 flap_h9_l30 flap_h11_l40; do
  CRAB_BOARD_DESIGN=$D PYTHONUTF8=1 ./.venv/Scripts/python.exe scripts/train.py \
    --task jumper.segway_board --backend warp --device cuda:0 --num_envs 2048 \
    --max-iterations 1500 --headless --no-tensorboard > ../lab/train_board_v5_$D.log 2>&1
  run=$(ls -d $L/*/ | sort | tail -1); ck=$(ls $run/model_*.pt | sort -V | tail -1)
  echo "$D $run" >> ../lab/out/step5_runs.txt
  CRAB_BOARD_DESIGN=$D PYTHONUTF8=1 ./.venv/Scripts/python.exe ../lab/eval_board.py \
    --checkpoint $ck --out ../lab/out/eval_$D.json > /dev/null 2>&1
done
BEST=$(../lab/.venv/Scripts/python.exe - <<'PY'
import json
rows = [json.load(open(f"../lab/out/eval_{d}.json", encoding="utf-8")) for d in ("flap_h9_l22", "flap_h9_l30", "flap_h11_l40")]
top = max(r["on_and_riding_at_end"] for r in rows)
close = [r for r in rows if r["on_and_riding_at_end"] >= top - 0.02]
print(min(close, key=lambda r: r["board_pushed_before_on_m"])["design"])
PY
)
echo "best $BEST" >> ../lab/out/step5_runs.txt
CRAB_TERRAIN=road CRAB_BOARD_DESIGN=$BEST PYTHONUTF8=1 ./.venv/Scripts/python.exe scripts/train.py \
  --task jumper.segway --backend warp --device cuda:0 --num_envs 2048 \
  --max-iterations 3000 --headless --no-tensorboard > ../lab/train_ride_road_$BEST.log 2>&1
echo "ride $(ls -d logs/jumper/jumper.segway/*/ | sort | tail -1)" >> ../lab/out/step5_runs.txt
