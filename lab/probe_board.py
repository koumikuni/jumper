"""Probe each board design: kneel geometry, deck height, on-deck starts engaging."""
import importlib, math, os, sys, torch
sys.argv = ["x"]
from mjrl.backend.select import use_backend
R = importlib.import_module("mjrl.backend.resolve")
use_backend(R.resolve(backend="native", device="cpu", num_envs=16))
from mjlab.envs import ManagerBasedRlEnv
from tasks import load_env_cfg
from tasks.jumper.segway.mdp import firmware, state
from tasks.jumper.segway.vehicle import DESIGNS
name = os.environ["CRAB_BOARD_DESIGN"]
cfg = load_env_cfg("jumper.segway_board", play=False); cfg.scene.num_envs = 16
for k in ("push_robot",): cfg.events.pop(k, None)
cfg.events["reset_beside_the_board"].params["on_deck_share"] = 0.5
env = ManagerBasedRlEnv(cfg=cfg, device="cpu"); env.reset()
board = env.scene[state.ENTITY]
for i in range(60):
    env.step(torch.zeros(16, env.action_manager.total_action_dim))
d = DESIGNS[name]
feet = firmware.feet_on_deck_count(env)
eng = firmware.engaged(env)
axes = state.deck_axes(env); _, deck = state._rows(env, state.ENTITY)
front = board.data.body_link_pos_w[:, deck] + axes[:, :, 0] * d.deck_x / 2 - axes[:, :, 2] * d.deck_t / 2
print(name, "deck top at rest %.3f m" % d.deck_top_z)
print(" feet on deck per env", feet.int().tolist())
print(" engaged", eng.int().tolist())
print(" front edge z (parked envs)", [round(float(z), 3) for z, e in zip(front[:, 2], eng) if not e])
print(" board tilt deg", [round(math.degrees(float(t)), 1) for t in state.tilt(env)])
