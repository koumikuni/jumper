#!/bin/sh
# Step 3 of the deck-shape search: wait for the running board training, then
# train and evaluate each comb deck in turn.
cd "$(dirname "$0")/../jumper"
while powershell -c "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*segway_board*' } | Select-Object -First 1" | grep -q python; do sleep 30; done
L=logs/jumper/jumper.segway_board
eval_latest() {  # design
  run=$(ls -d $L/*/ | sort | tail -1)
  ck=$(ls $run/model_*.pt | sort -V | tail -1)
  CRAB_BOARD_DESIGN=$1 PYTHONUTF8=1 ./.venv/Scripts/python.exe ../lab/eval_board.py --checkpoint $ck --out ../lab/out/eval_$1.json > /dev/null 2>&1
  echo "$1 $run"; cat ../lab/out/eval_$1.json
}
[ -n "$SKIP_BASE" ] || eval_latest lowfloor_hub
for D in ${DESIGNS:-comb_U comb_E_ramp comb_E}; do
  CRAB_BOARD_DESIGN=$D PYTHONUTF8=1 ./.venv/Scripts/python.exe scripts/train.py \
    --task jumper.segway_board --backend warp --device cuda:0 --num_envs 2048 \
    --max-iterations ${ITERS:-1000} --headless --no-tensorboard > ../lab/train_board_v3_$D.log 2>&1
  eval_latest $D
done
