"""Probe: where the six feet sit relative to the deck's top face when standing on it."""
import importlib, sys, torch
sys.argv = ["x"]
from mjrl.backend.select import use_backend
R = importlib.import_module("mjrl.backend.resolve")
use_backend(R.resolve(backend="native", device="cpu", num_envs=2))
from mjlab.envs import ManagerBasedRlEnv
from tasks import load_env_cfg
from tasks.jumper.segway.mdp import state
from tasks.jumper.segway.vehicle import DESIGN
cfg = load_env_cfg("jumper.segway", play=True); cfg.scene.num_envs = 2
env = ManagerBasedRlEnv(cfg=cfg, device="cpu"); env.reset()
for _ in range(25): env.step(torch.zeros(2, env.action_manager.total_action_dim))
f = state.feet_on_deck(env)[0]
f[:, 2] -= DESIGN.deck_t / 2
print(f.numpy().round(4))
robot = env.scene["robot"]
print("base above deck top", float(state.robot_on_deck(env)[0, 2]))
