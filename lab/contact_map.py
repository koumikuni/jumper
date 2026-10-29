"""Where the crab touches the board: contact points on the deck, boarding and riding.

Runs a trained policy on the native (CPU) backend and, every control step, copies
each world's qpos into a scratch MjData of the physics model, runs the collision
pass and keeps every contact between a robot geom and a board geom. Points are
stored in the **board frame at rest** -- origin on the ground under the axle,
x forward, y left, z up, metres -- by mapping through the deck's own pose, so a
point is where on the deck it was even while the deck rolls or the board pitches.

    boarding  jumper.segway_board, from the ground, until 2 s after boarding
    riding    jumper.segway on the road terrain, under random ride commands

    CRAB_BOARD_DESIGN=lowcell_h14 jumper/.venv/Scripts/python.exe lab/contact_map.py \
        --board-ckpt <...> --ride-ckpt <...> --out lab/out/contacts_lowcell_h14.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

# The framework checkout: `jumper/` beside `lab/` while developing, or the
# repository itself when `lab/` is published inside it.
_ROOT = Path(__file__).resolve().parents[1]
JUMPER = _ROOT / "jumper" if (_ROOT / "jumper" / "tasks").is_dir() else _ROOT
sys.path.insert(0, str(JUMPER))


def collect(task: str, ckpt: Path, num_envs: int, seconds: float, phase_fn=None) -> list:
    import mujoco
    import torch

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner

    import tasks
    from tasks.jumper.segway import vehicle

    cfg = tasks.load_env_cfg(task, play=True)
    cfg.scene.num_envs = num_envs
    agent = tasks.load_agent_cfg(task)
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent), device="cpu")
    runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location="cpu")
    policy = runner.get_inference_policy(device="cpu")

    m = env.sim.mj_model
    d = mujoco.MjData(m)
    name = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    robot = {g for g in range(m.ngeom) if name(g).startswith("robot/")}
    board = {g for g in range(m.ngeom) if name(g).startswith("segway/")
             and not name(g).split("/")[-1].startswith(("tyre", "hub"))}
    deck = m.body("segway/deck").id
    design = vehicle.DESIGNS[os.environ.get("CRAB_BOARD_DESIGN", "baseline")]
    deck_rest = np.array([0.0, 0.0, design.wheel_r + design.deck_above_axle + design.deck_t / 2])

    points = []
    obs = wrapped.get_observations()
    qpos = env.sim.data.qpos
    origins = env.scene.env_origins.cpu().numpy()
    done_since = np.full(num_envs, -1.0)
    with torch.inference_mode():
        for k in range(int(seconds / env.step_dt)):
            obs, _, dones, _ = wrapped.step(policy(obs))
            t = (k + 1) * env.step_dt
            phase = phase_fn(env) if phase_fn else None
            q = qpos.cpu().numpy() if hasattr(qpos, "cpu") else np.asarray(qpos)
            for w in range(num_envs):
                if phase is not None:
                    if phase[w] and done_since[w] < 0:
                        done_since[w] = t
                    if done_since[w] >= 0 and t - done_since[w] > 2.0:
                        continue
                d.qpos[:] = q[w]
                mujoco.mj_forward(m, d)
                R, x = d.xmat[deck].reshape(3, 3), d.xpos[deck]
                for c in d.contact[:d.ncon]:
                    a, b = int(c.geom1), int(c.geom2)
                    if not ((a in robot and b in board) or (b in robot and a in board)):
                        continue
                    rg = a if a in robot else b
                    local = R.T @ (c.pos - x) + deck_rest
                    part = name(rg).split("/")[-1].replace("_meshcol", "")
                    points.append([round(float(v), 4) for v in local] + [part, round(t, 2), w])
            if dones.any():
                done_since[dones.cpu().numpy().astype(bool)] = -1.0
    return points


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board-ckpt", type=Path, required=True)
    ap.add_argument("--ride-ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num_envs", type=int, default=24)
    a = ap.parse_args()

    from mjrl.backend.select import use_backend
    use_backend(importlib.import_module("mjrl.backend.resolve").resolve(
        backend="native", device="cpu", num_envs=a.num_envs))
    from tasks.jumper.segway_board import mdp as board_mdp

    boarding = collect("jumper.segway_board", a.board_ckpt, a.num_envs, 6.0,
                       phase_fn=lambda env: board_mdp.boarded(env).cpu().numpy())
    os.environ["CRAB_TERRAIN"] = "road"
    riding = collect("jumper.segway", a.ride_ckpt, a.num_envs, 20.0)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "design": os.environ.get("CRAB_BOARD_DESIGN"),
        "frame": "board at rest: origin on the ground under the axle, x forward, y left, z up, metres",
        "columns": ["x", "y", "z", "robot_part", "t", "env"],
        "boarding": boarding, "riding": riding,
    }), encoding="utf-8")
    print(f"boarding {len(boarding)} contacts, riding {len(riding)} contacts -> {a.out}")


if __name__ == "__main__":
    main()
