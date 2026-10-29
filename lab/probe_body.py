"""Probe: the crab's belly clearance and where its legs pass, standing on flat ground."""
import importlib, sys, torch, numpy as np
from pathlib import Path
from pathlib import Path
sys.argv = ["x"]
from mjrl.backend.select import use_backend
use_backend(importlib.import_module("mjrl.backend.resolve").resolve(backend="native", device="cpu", num_envs=1))
from mjlab.envs import ManagerBasedRlEnv
from tasks import load_env_cfg
cfg = load_env_cfg("jumper.tripod", play=True); cfg.scene.num_envs = 1
env = ManagerBasedRlEnv(cfg=cfg, device="cpu"); env.reset()
for _ in range(40): env.step(torch.zeros(1, env.action_manager.total_action_dim))
m = env.sim.mj_model
import mujoco
d = mujoco.MjData(m); d.qpos[:] = env.sim.data.qpos[0].cpu().numpy(); mujoco.mj_forward(m, d)
robot = env.scene["robot"]
base = d.xpos[m.body("robot/base_link").id]; R = d.xmat[m.body("robot/base_link").id].reshape(3, 3)
print("base z", base[2].round(4))
rows = []
for g in range(m.ngeom):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    if not name.startswith("robot/") or "meshcol" not in name:
        continue
    mid = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
    lb = (w - base) @ R
    rows.append((name.split("/")[1].replace("_meshcol", ""), lb[:, 0].min(), lb[:, 0].max(), lb[:, 1].min(), lb[:, 1].max(), w[:, 2].min(), w[:, 2].max()))
for r in sorted(rows, key=lambda r: r[5]):
    print("%-28s x %+.3f..%+.3f  y %+.3f..%+.3f  z %.3f..%.3f" % r)

# Saved for lab/deck_screen.py: the standing pose, by joint name.
import json
pose = {}
for j in range(m.njnt):
    name = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "").replace("robot/", "")
    a, n = m.jnt_qposadr[j], {0: 7, 1: 4}.get(int(m.jnt_type[j]), 1)
    pose[name] = d.qpos[a:a + n].tolist()
Path("../lab/out/stand_pose.json").write_text(json.dumps(pose), encoding="utf-8")
print("saved", len(pose))
