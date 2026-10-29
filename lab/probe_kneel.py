"""Probe the kneel board under the trained policy: chassis tilt, hinge, feet, engaged."""
import importlib, math, os, sys
from dataclasses import asdict
import torch
os.environ["CRAB_BOARD_DESIGN"] = "kneel"
sys.argv = ["x"]
from mjrl.backend.select import use_backend
res = importlib.import_module("mjrl.backend.resolve").resolve(backend="native", device="cpu", num_envs=1)
use_backend(res)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
import tasks
from tasks.jumper.segway.mdp import firmware, state
cfg = tasks.load_env_cfg("jumper.segway_board", play=True); cfg.scene.num_envs = 1
cfg.events["reset_beside_the_board"].params["on_deck_share"] = 0.0
ag = tasks.load_agent_cfg("jumper.segway_board")
env = ManagerBasedRlEnv(cfg=cfg, device="cpu"); w = RslRlVecEnvWrapper(env, clip_actions=ag.clip_actions)
r = MjlabOnPolicyRunner(w, asdict(ag), device="cpu"); r.load(sys.argv[0] if False else os.environ["CK"], load_cfg={"actor": True}, strict=True, map_location="cpu")
pol = r.get_inference_policy(device="cpu"); obs = w.get_observations()
b = env.scene[state.ENTITY]; hj = b.joint_names.index("deck_pitch")
with torch.inference_mode():
    for k in range(600):
        obs, *_ = w.step(pol(obs))
        if k % 50 == 0:
            print(f"t {k*0.02:5.1f}  chassis pitch {math.degrees(float(state.pitch(env)[0])):+6.1f}  hinge {math.degrees(float(b.data.joint_pos[0, hj])):5.1f}"
                  f"  feet {int(firmware.feet_on_deck_count(env)[0])}  engaged {bool(firmware.engaged(env)[0])}  board v {float(b.data.root_link_lin_vel_w[0,:2].norm()):.2f}")
