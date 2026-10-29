"""Hyper-parameters for jumper.segway.

Started from the swing's, which is the other task where the robot stands on a
moving prop; the rollout is shorter because the thing to learn here -- lean, and
the board answers within a second -- is faster than a swing's cycle.
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    return jumper_ppo_baseline(
        experiment_name="jumper.segway",
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        init_std=0.3,
        entropy_coef=0.005,
        learning_rate=1.0e-3,
        desired_kl=0.01,
        gamma=0.99,
        lam=0.95,
        num_learning_epochs=5,
        num_mini_batches=4,
        num_steps_per_env=24,
        max_iterations=5_000,
        # 25 for the first run, which the learning-progress video was cut from;
        # 100 since, after checkpoints filled the disk on 2026-10-23.
        save_interval=100,
    )


def play_status():
    """The line `scripts/play.py` prints while replaying: the board, not the robot."""

    def status(entity, index: int, env) -> str:
        import torch

        from .mdp import firmware, state

        unwrapped = getattr(env, "unwrapped", env)
        with torch.inference_mode():
            cmd = unwrapped.command_manager.get_command("ride")[index]
            v = float(state.ground_speed(unwrapped)[index])
            w = float(state.yaw_rate(unwrapped)[index])
            pitch = float(state.pitch(unwrapped)[index])
            roll = float(state.deck_roll(unwrapped)[index])
            v_cmd = float(firmware.v_cmd(unwrapped)[index])
        deg = 57.29578
        return (
            f"ride {float(cmd[0]):+4.2f} m/s {float(cmd[2]):+4.2f} rad/s   "
            f"board {v:+4.2f} m/s {w:+4.2f} rad/s   "
            f"pitch {pitch * deg:+5.1f}   deck roll {roll * deg:+5.1f}   fw v {v_cmd:+4.2f}"
        )

    return status
