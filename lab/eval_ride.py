"""Evaluate riding on the road terrain: falls, tracking, and how often the board scrapes.

Rides `--seconds` under the task's own random ride commands (replay settings:
no observation noise, no pushes) on `CRAB_TERRAIN=road`, with a contact sensor
between the board's underside parts -- deck plates, flap, stubs, axle, battery
-- and the terrain. The wheels are left out: they are meant to touch.

    CRAB_BOARD_DESIGN=flap_h9_l40_lock jumper/.venv/Scripts/python.exe lab/eval_ride.py \
        --checkpoint <run>/model_N.pt --out lab/out/ride_<design>.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

# The framework checkout: `jumper/` beside `lab/` while developing, or the
# repository itself when `lab/` is published inside it.
_ROOT = Path(__file__).resolve().parents[1]
JUMPER = _ROOT / "jumper" if (_ROOT / "jumper" / "tasks").is_dir() else _ROOT
sys.path.insert(0, str(JUMPER))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num_envs", type=int, default=128)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--backend", default="warp")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    os.environ.setdefault("CRAB_TERRAIN", "road")

    import mujoco
    import torch

    from mjrl.backend.select import use_backend
    res = importlib.import_module("mjrl.backend.resolve").resolve(
        backend=a.backend, device=a.device, num_envs=a.num_envs)
    use_backend(res)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner
    from mjlab.sensor import ContactMatch, ContactSensorCfg

    import tasks
    from tasks.jumper.segway import vehicle
    from tasks.jumper.segway.mdp import state

    design_name = os.environ.get("CRAB_BOARD_DESIGN", "baseline")
    xml = mujoco.MjModel.from_xml_string(vehicle._xml(vehicle.DESIGNS[design_name]))
    under = [mujoco.mj_id2name(xml, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(xml.ngeom)]
    under = [n for n in under if n and not n.startswith(("tyre", "hub"))]

    cfg = tasks.load_env_cfg("jumper.segway", play=True)
    cfg.scene.num_envs = a.num_envs
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (ContactSensorCfg(
        name="board_scrape",
        primary=ContactMatch(mode="geom", pattern=tuple(under), entity=state.ENTITY),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",), reduce="netforce", num_slots=1,
        history_length=cfg.decimation,
    ),)
    agent_cfg = tasks.load_agent_cfg("jumper.segway")
    env = ManagerBasedRlEnv(cfg=cfg, device=res.device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), device=res.device)
    runner.load(str(a.checkpoint), load_cfg={"actor": True}, strict=True, map_location=res.device)
    policy = runner.get_inference_policy(device=res.device)

    n, steps = a.num_envs, int(a.seconds / env.step_dt)
    fell = torch.zeros(n, dtype=torch.bool)
    scraped_steps = torch.zeros(n)
    v_err = torch.zeros(n)
    w_err = torch.zeros(n)
    counted = torch.zeros(n)
    obs = wrapped.get_observations()
    sensor = env.scene["board_scrape"]
    with torch.inference_mode():
        for _ in range(steps):
            obs, _, dones, _ = wrapped.step(policy(obs))
            live = (~fell).float()
            found = sensor.data.found.reshape(n, -1).cpu()
            scraped_steps += (found > 0).any(-1).float() * live
            cmd = env.command_manager.get_command("ride").cpu()
            v_err += (state.ground_speed(env).cpu() - cmd[:, 0]).abs() * live
            w_err += (state.yaw_rate(env).cpu() - cmd[:, 2]).abs() * live
            counted += live
            fell |= dones.cpu().bool()
    rows = counted.clamp(min=1)
    summary = {
        "design": design_name,
        "checkpoint": str(a.checkpoint),
        "terrain": os.environ.get("CRAB_TERRAIN"),
        "episodes": n,
        "seconds": a.seconds,
        "fell_share": round(float(fell.float().mean()), 3),
        "mean_speed_error_mps": round(float((v_err / rows).mean()), 3),
        "mean_yaw_error_radps": round(float((w_err / rows).mean()), 3),
        "scrape_share_of_time": round(float((scraped_steps / rows).mean()), 4),
        "episodes_with_a_scrape": round(float((scraped_steps > 0).float().mean()), 3),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
