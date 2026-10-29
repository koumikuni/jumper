"""Hyper-parameters for jumper.segway_board.

The riding task's, with a longer rollout -- getting on is a sequence (walk up,
step up, shuffle to the middle, wait for the board) a few seconds long -- and a
longer run, sized to train overnight: about 8 hours at 2.7 s an iteration.
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    return jumper_ppo_baseline(
        experiment_name="jumper.segway_board",
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        init_std=0.4,
        entropy_coef=0.008,
        learning_rate=1.0e-3,
        desired_kl=0.01,
        gamma=0.99,
        lam=0.95,
        num_learning_epochs=5,
        num_mini_batches=4,
        num_steps_per_env=32,
        max_iterations=10_000,
        save_interval=100,
    )


def play_status():
    """The replay line: how many feet are on, and whether the board has engaged."""

    def status(entity, index: int, env) -> str:
        import torch

        from ..segway.mdp import firmware, state
        from . import mdp

        u = getattr(env, "unwrapped", env)
        with torch.inference_mode():
            feet = int(firmware.feet_on_deck_count(u)[index])
            on = bool(firmware.engaged(u)[index])
            dist = float(mdp.distance_to_target(u)[index])
            tilt = float(state.tilt(u)[index]) * 57.29578
            v = float(u.scene[state.ENTITY].data.root_link_lin_vel_w[index, :2].norm())
        return (f"feet on deck {feet}/6   board {'ON ' if on else 'parked'}   "
                f"to spot {dist:4.2f} m   board tilt {tilt:4.1f} deg  speed {v:4.2f} m/s")

    return status
