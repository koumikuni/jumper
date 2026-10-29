"""Probe: passive robot (zero action) on the segway -- deck roll and robot COM offset."""
import sys, torch
sys.argv = ["x"]
from mjrl.backend.select import use_backend
import mjrl.backend.resolve as _r
from tasks import load_env_cfg
import importlib
R = importlib.import_module("mjrl.backend.resolve")
res = R.resolve(backend="native", device="cpu", num_envs=8)
use_backend(res)
from mjlab.envs import ManagerBasedRlEnv
from tasks.jumper.segway.mdp import state
cfg = load_env_cfg("jumper.segway", play=True)
cfg.scene.num_envs = 8
for k in ("push_robot", "foot_friction", "encoder_bias", "base_com"):
    cfg.events.pop(k, None)
env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
env.reset()
act = torch.zeros(8, env.action_manager.total_action_dim)
robot = env.scene["robot"]
for i in range(150):
    env.step(act)
    if i % 25 == 0:
        m = env.sim.mj_model if hasattr(env.sim, "mj_model") else None
        com = (robot.data.body_com_pos_w * 0).sum() if False else None
        off = state.robot_on_deck(env)
        print(f"step {i:3d} roll {float(state.deck_roll(env)[0])*57.3:+6.2f} deg  "
              f"pitch {float(state.pitch(env)[0])*57.3:+6.2f}  base y on deck {float(off[0,1])*1000:+6.1f} mm  "
              f"yaw {float(state.yaw_rate(env)[0]):+5.2f}")
