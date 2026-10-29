"""Evaluate a boarding policy from the ground only: does the crab get on, alone?

Every episode starts on the ground in front of the parked board (the training
reset's on-deck starts are switched off), with replay settings -- no observation
noise, no pushes -- and runs for `--seconds`. Per episode it records whether and
when the robot was first boarded, whether it is boarded with the board's balance
loop engaged at the end, and how many times each of `face.BoardingMood`'s three
complaints fired (rising edges, so one long shove is one complaint).

    CRAB_BOARD_DESIGN=kneel jumper/.venv/Scripts/python.exe lab/eval_board.py \
        --checkpoint <run>/model_1499.pt --out lab/out/eval_kneel.json
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
    ap.add_argument("--seconds", type=float, default=12.0)
    a = ap.parse_args()

    import torch

    from mjrl.backend.select import use_backend
    res = importlib.import_module("mjrl.backend.resolve").resolve(
        backend="warp", device="cuda:0", num_envs=a.num_envs)
    use_backend(res)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner

    import tasks
    from tasks.jumper.segway.face import BoardingMood
    from tasks.jumper.segway.mdp import firmware, state
    from tasks.jumper.segway_board import mdp

    cfg = tasks.load_env_cfg("jumper.segway_board", play=True)
    cfg.scene.num_envs = a.num_envs
    cfg.events["reset_beside_the_board"].params["on_deck_share"] = 0.0
    agent_cfg = tasks.load_agent_cfg("jumper.segway_board")
    env = ManagerBasedRlEnv(cfg=cfg, device=res.device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), device=res.device)
    runner.load(str(a.checkpoint), load_cfg={"actor": True}, strict=True, map_location=res.device)
    policy = runner.get_inference_policy(device=res.device)

    n, steps, dt = a.num_envs, int(a.seconds / env.step_dt), env.step_dt
    first_boarded = torch.full((n,), float("nan"))
    ended = torch.zeros(n, dtype=torch.bool)  # an episode that terminated is over
    end_reason = [""] * n
    best_feet = torch.zeros(n)
    max_feet = torch.zeros(n)
    # How far the board was shoved before the robot was on it, metres.
    pushed = torch.zeros(n)
    prev = {k: torch.zeros(n, dtype=torch.bool) for k in ("fled", "rocked", "slipped")}
    counts = {k: torch.zeros(n) for k in prev}
    obs = wrapped.get_observations()
    with torch.inference_mode():
        for k in range(steps):
            obs, _, dones, _ = wrapped.step(policy(obs))
            t = (k + 1) * dt
            on = mdp.boarded(env).cpu()
            feet = firmware.feet_on_deck_count(env).cpu()
            board_v = env.scene[state.ENTITY].data.root_link_lin_vel_w[:, :2].norm(dim=-1).cpu()
            tilt = torch.rad2deg(state.tilt(env)).cpu()
            now = {
                "fled": ~on & (board_v > BoardingMood.FLEE_SPEED),
                "rocked": ~on & (tilt > BoardingMood.ROCK_DEG),
                "slipped": (best_feet >= 2) & (feet == 0),
            }
            live = ~ended
            for key, flag in now.items():
                counts[key] += (flag & ~prev[key] & live).float()
                prev[key] = flag
            max_feet = torch.where(live, torch.maximum(max_feet, feet), max_feet)
            pushed += (board_v * dt) * (live & torch.isnan(first_boarded)).float()
            best_feet = torch.where(feet == 0, torch.zeros_like(best_feet), torch.maximum(best_feet, feet))
            newly = on & live & torch.isnan(first_boarded)
            first_boarded[newly] = t
            d = dones.cpu().bool()
            for i in torch.nonzero(d & live).flatten().tolist():
                end_reason[i] = "terminated"
            ended |= d
        final_on = (mdp.boarded(env).cpu() & firmware.engaged(env).cpu() & ~ended)

    design = os.environ.get(mdp.DESIGN_ENV, "baseline")
    boarded_t = first_boarded[~torch.isnan(first_boarded)]
    summary = {
        "design": design,
        "checkpoint": str(a.checkpoint),
        "episodes": n,
        "seconds": a.seconds,
        "boarded_ever": round(float((~torch.isnan(first_boarded)).float().mean()), 3),
        "on_and_riding_at_end": round(float(final_on.float().mean()), 3),
        "median_time_to_board_s": round(float(boarded_t.median()), 2) if len(boarded_t) else None,
        "terminated": round(float(ended.float().mean()), 3),
        "board_pushed_before_on_m": round(float(pushed.mean()), 3),
        "max_feet_on_deck_share": {f">={k}": round(float((max_feet >= k).float().mean()), 3)
                                   for k in (1, 2, 4, 6)},
        "complaints_per_episode": {k: round(float(v.mean()), 2) for k, v in counts.items()},
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
