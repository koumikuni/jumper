#!/bin/sh
# Train the boarding task on each board design, one after another.
cd "$(dirname "$0")/../jumper"
for D in kneel lowfloor baseline; do
  CRAB_BOARD_DESIGN=$D PYTHONUTF8=1 ./.venv/Scripts/python.exe scripts/train.py \
    --task jumper.segway_board --backend warp --device cuda:0 --num_envs 2048 \
    --max-iterations 1500 --headless --no-tensorboard > ../lab/train_board_v2_$D.log 2>&1
  echo "$D exit $?"
done
