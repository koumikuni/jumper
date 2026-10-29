#!/bin/sh
# Step 7, overnight: raise the step of the low hub-motor board a little at a
# time, each height fine-tuned from the one below, then ride the tallest one
# that still boards on the road terrain.
cd "$(dirname "$0")/../jumper"
L=logs/jumper/jumper.segway_board
LOG=../lab/out/step7_runs.txt
FROM=$L/2026-10-22_16-20-23/model_1499.pt   # lowfloor_hub: 99% boarded
free_gb() { powershell -c "[int]((Get-PSDrive C).Free/1GB)"; }
for H in 6.5 8 9 10 11 12 13 14; do
  D=lowcell_h$H
  IT=700; [ "$H" = "6.5" ] && IT=400
  [ "$(free_gb)" -ge 2 ] || { echo "stopped at $D: disk below 2 GB" >> $LOG; break; }
  CRAB_BOARD_DESIGN=$D PYTHONUTF8=1 ./.venv/Scripts/python.exe scripts/train.py \
    --task jumper.segway_board --backend warp --device cuda:0 --num_envs 2048 \
    --max-iterations $IT --resume --checkpoint $FROM --headless --no-tensorboard \
    > ../lab/train_board_v7_$D.log 2>&1
  run=$(ls -d $L/*/ | sort | tail -1); ck=$(ls $run/model_*.pt | sort -V | tail -1)
  CRAB_BOARD_DESIGN=$D PYTHONUTF8=1 ./.venv/Scripts/python.exe ../lab/eval_board.py \
    --checkpoint $ck --out ../lab/out/eval_$D.json > /dev/null 2>&1
  echo "$D $ck" >> $LOG
  FROM=$ck
done
BEST=$(../lab/.venv/Scripts/python.exe - <<'PY'
import json, os
best = None
for h in ("6.5", "8", "9", "10", "11", "12", "13", "14"):
    path = f"../lab/out/eval_lowcell_h{h}.json"
    if os.path.exists(path) and json.load(open(path, encoding="utf-8"))["on_and_riding_at_end"] >= 0.9:
        best = f"lowcell_h{h}"
print(best or "")
PY
)
[ -n "$BEST" ] || { echo "no height boarded 90%: riding test not started" >> $LOG; exit 0; }
echo "best $BEST" >> $LOG
[ "$(free_gb)" -ge 2 ] || { echo "stopped before riding: disk below 2 GB" >> $LOG; exit 1; }
CRAB_TERRAIN=road CRAB_BOARD_DESIGN=$BEST PYTHONUTF8=1 ./.venv/Scripts/python.exe scripts/train.py \
  --task jumper.segway --backend warp --device cuda:0 --num_envs 2048 \
  --max-iterations 3000 --headless --no-tensorboard > ../lab/train_ride_road_$BEST.log 2>&1
run=$(ls -d logs/jumper/jumper.segway/*/ | sort | tail -1); ck=$(ls $run/model_*.pt | sort -V | tail -1)
echo "ride $BEST $ck" >> $LOG
CRAB_BOARD_DESIGN=$BEST PYTHONUTF8=1 ./.venv/Scripts/python.exe ../lab/eval_ride.py \
  --checkpoint $ck --out ../lab/out/ride_$BEST.json > /dev/null 2>&1
# The reference: the first riding policy, trained on flat ground, on the original board.
CRAB_BOARD_DESIGN=baseline PYTHONUTF8=1 ./.venv/Scripts/python.exe ../lab/eval_ride.py \
  --checkpoint logs/jumper/jumper.segway/2026-10-21_15-58-26/model_2999.pt \
  --out ../lab/out/ride_baseline_flatpolicy.json > /dev/null 2>&1
echo "done" >> $LOG
