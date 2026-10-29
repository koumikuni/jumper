"""Probe: a parked flap board with the robot kept well away -- does it move by itself?"""
import importlib, math, os, sys, torch
sys.argv = ["x"]
from mjrl.backend.select import use_backend
use_backend(importlib.import_module("mjrl.backend.resolve").resolve(backend="native", device="cpu", num_envs=4))
from mjlab.envs import ManagerBasedRlEnv
from tasks import load_env_cfg
from tasks.jumper.segway.mdp import state
cfg = load_env_cfg("jumper.segway_board", play=True); cfg.scene.num_envs = 4
cfg.events["reset_beside_the_board"].params.update(distance_range=(1.0, 1.0), on_deck_share=0.0)
cfg.terminations.pop("wandered_off", None)
env = ManagerBasedRlEnv(cfg=cfg, device="cpu"); env.reset()
b = env.scene[state.ENTITY]; start = b.data.root_link_pos_w[:, :2].clone()
for k in range(150):
    env.step(torch.zeros(4, env.action_manager.total_action_dim))
moved = (b.data.root_link_pos_w[:, :2] - start).norm(dim=-1)
print(os.environ["CRAB_BOARD_DESIGN"], "board moved by itself in 3 s (m):", [round(float(m), 3) for m in moved],
      "tilt deg", [round(math.degrees(float(t)), 1) for t in state.tilt(env)])
