#!/bin/sh
# Shape step 6: locked ramp flaps, fine-tuned from the full-ramp policy; then,
# only if one of them boards, the riding test on the road terrain.
cd "$(dirname "$0")/../jumper"
L=logs/jumper/jumper.segway_board
FROM=$L/2026-10-22_23-21-26/model_1499.pt   # ramp_full: 100% boarded
LOG=../lab/out/step6_runs.txt
free_gb() { powershell -c "[int]((Get-PSDrive C).Free/1GB)"; }
DESIGNS="flap_h9_l40_lock flap_h9_l30_lock flap_h11_l40_lock"
for D in $DESIGNS; do
  [ "$(free_gb)" -ge 3 ] || { echo "stopped: disk below 3 GB" >> $LOG; exit 1; }
  CRAB_BOARD_DESIGN=$D PYTHONUTF8=1 ./.venv/Scripts/python.exe scripts/train.py \
    --task jumper.segway_board --backend warp --device cuda:0 --num_envs 2048 \
    --max-iterations 600 --resume --checkpoint $FROM --headless --no-tensorboard \
    > ../lab/train_board_v6_$D.log 2>&1
  run=$(ls -d $L/*/ | sort | tail -1); ck=$(ls $run/model_*.pt | sort -V | tail -1)
  echo "$D $run $(basename $ck)" >> $LOG
  CRAB_BOARD_DESIGN=$D PYTHONUTF8=1 ./.venv/Scripts/python.exe ../lab/eval_board.py \
    --checkpoint $ck --out ../lab/out/eval_$D.json > /dev/null 2>&1
done
BEST=$(../lab/.venv/Scripts/python.exe - $DESIGNS <<'PY'
import json, os, sys
rows = []
for d in sys.argv[1:]:
    path = f"../lab/out/eval_{d}.json"
    if os.path.exists(path):
        rows.append(json.load(open(path, encoding="utf-8")))
rows = [r for r in rows if r["on_and_riding_at_end"] >= 0.5]
if rows:
    top = max(r["on_and_riding_at_end"] for r in rows)
    close = [r for r in rows if r["on_and_riding_at_end"] >= top - 0.02]
    print(min(close, key=lambda r: r["board_pushed_before_on_m"])["design"])
PY
)
[ -n "$BEST" ] || { echo "no flap design boarded 50%: riding test not started" >> $LOG; exit 0; }
echo "best $BEST" >> $LOG
[ "$(free_gb)" -ge 3 ] || { echo "stopped: disk below 3 GB" >> $LOG; exit 1; }
CRAB_TERRAIN=road CRAB_BOARD_DESIGN=$BEST PYTHONUTF8=1 ./.venv/Scripts/python.exe scripts/train.py \
  --task jumper.segway --backend warp --device cuda:0 --num_envs 2048 \
  --max-iterations 3000 --headless --no-tensorboard > ../lab/train_ride_road_$BEST.log 2>&1
echo "ride $BEST $(ls -d logs/jumper/jumper.segway/*/ | sort | tail -1)" >> $LOG
