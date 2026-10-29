"""Environment config for jumper.segway.

Built the swing's way: start from `velocity_env_cfg`, which is "this robot in
this simulator", and subtract the walking. What is taken out and why:

    commands      twist                  replaced by `ride`, which has no lateral
                                         axis -- a board cannot go sideways
    observations  command                read `twist`; replaced by `ride_command`
    rewards       track_*, pose, the     the robot's own velocity and gait; the
                  foot-gait terms,       objective here is the *board's* motion,
                  standing_sway, upright and `upright` measures against gravity
                  body_tilt_rate         while the deck cruises nose-up
    terminations  fell_over              against gravity, likewise
    curriculum    terrain, command       both read `twist`
    events        reset_base             would put the robot beside the board

The board and its firmware are installed by the task, not by `--scene`, for the
swing's reason: a task that needs its prop must not train happily without it.
"""

from __future__ import annotations

import math
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from ..common.constants import GAIT_JOINTS
from ..common.mdp.curriculum import STD_ANG_RATIO, STD_LIN_RATIO
from ..common.velocity_env import _ARM_JOINTS, _LEG_JOINTS, velocity_env_cfg
from .mdp import events as segway_events
from .mdp import firmware as segway_firmware
from .mdp import observations as segway_obs
from .mdp import rewards as segway_rewards
from .mdp import state as segway_state
from .mdp import terminations as segway_terms

_WALKING_REWARDS = (
    "track_linear_velocity",
    "track_angular_velocity",
    "pose",
    "air_time",
    "foot_clearance",
    "foot_swing_height",
    "foot_slip",
    "soft_landing",
    "standing_sway",
    "upright",
    "body_tilt_rate",
)

#: The ride command's ranges. Forward up to 1.0 m/s -- the prototype's box rider
#: reached 0.97 m/s on a 4 cm lean -- and a little reverse, because stopping is
#: leaning back and a policy that can only lean forward cannot stop.
RIDE_SPEED_RANGE = (-0.3, 1.0)
RIDE_YAW_RANGE = (-1.0, 1.0)

#: The board's tilt at which the episode ends. Cruising at 1 m/s rides about 11
#: degrees nose-up (`vehicle.py`), so this leaves room for that and for braking.
BOARD_OVER_ANGLE = math.radians(40.0)

IMU_HISTORY = 10

#: `CRAB_TERRAIN=road`: what a board is ridden over, not what a hexapod walks
#: over. `scenes/rough.py` is 40% stairs, which no two-wheeled board climbs, so
#: this is the road a board meets: flat, small bumps, and slopes up to about 10
#: degrees (a gradient of 0.18), up and down. No curriculum: every tile is
#: drawn at a random difficulty, and an episode is ridden on whichever it lands.
TERRAIN_ENV = "CRAB_TERRAIN"


def road_terrain():
    from mjlab.terrains.config import box_random_grid, hf_pyramid_slope, hf_pyramid_slope_inv, random_rough
    from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

    return TerrainGeneratorCfg(
        curriculum=False,
        size=(6.0, 6.0),
        num_rows=4,
        num_cols=8,
        border_width=2.0,
        sub_terrains={
            "flat": box_random_grid(proportion=0.30, grid_width=1.0, grid_height_range=(0.0, 0.0),
                                    merge_similar_heights=False),
            "bumps": random_rough(proportion=0.35, noise_range=(0.0, 0.02), noise_step=0.005),
            "slope_up": hf_pyramid_slope(proportion=0.20, slope_range=(0.0, 0.18), platform_width=1.5),
            "slope_down": hf_pyramid_slope_inv(proportion=0.15, slope_range=(0.0, 0.18), platform_width=1.5),
        },
    )


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

    import os

    from .vehicle import DESIGNS, segway_entity

    # The board to ride: `CRAB_BOARD_DESIGN`, as for jumper.segway_board, so a
    # design that won the boarding comparison can be ridden by the same name.
    design_name = os.environ.get("CRAB_BOARD_DESIGN", "baseline").strip() or "baseline"
    DESIGN = DESIGNS[design_name]
    print(f"[segway] board design: {design_name}")
    segway_firmware.DECK_RECTS = DESIGN.deck_rects

    cfg = velocity_env_cfg(
        asset=asset,
        play=play,
        teleop=False,
        gamepad=False,
        operator_command=False,
        standing_sway_weight=0.0,
        # Required by the skeleton and, except `action_rate_weight`, configuring
        # terms that are removed below -- passed as the shared constants for the
        # reason `swing/env_cfg.py` gives.
        action_rate_weight=-0.1,
        pose_weight=1.0,
        foot_target_height=0.025,
        command_lin_ceiling=0.8,
        command_ang_ceiling=1.0,
        command_lin_std_ratio=STD_LIN_RATIO,
        command_ang_std_ratio=STD_ANG_RATIO,
    )

    # ── The board ────────────────────────────────────────────────────────
    cfg.scene.entities[segway_state.ENTITY] = segway_entity(DESIGN)
    # Measured on `--scene rough` with the flap board: 157 contacts at the first
    # step, against the skeleton's 128. The same headroom as jumper.segway_board.
    cfg.sim.nconmax = 192
    cfg.sim.njmax = 768
    cfg.events["segway_firmware"] = EventTermCfg(
        func=segway_firmware.firmware, mode="step",
        params={"flap_down": DESIGN.flap_down_rad, "flap_up": DESIGN.flap_up_rad},
    )

    # ── Take the walking out ─────────────────────────────────────────────
    cfg.commands.pop("twist", None)
    cfg.curriculum = {}
    for name in _WALKING_REWARDS:
        cfg.rewards.pop(name, None)
    for group in cfg.observations.values():
        group.terms.pop("command", None)
    cfg.terminations.pop("fell_over", None)

    # ── The ride command ─────────────────────────────────────────────────
    cfg.commands["ride"] = UniformVelocityCommandCfg(
        entity_name="robot",
        resampling_time_range=(4.0, 8.0),
        rel_standing_envs=0.15,
        rel_heading_envs=0.0,
        heading_command=False,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=RIDE_SPEED_RANGE, lin_vel_y=(0.0, 0.0), ang_vel_z=RIDE_YAW_RANGE
        ),
    )

    # ── The robot starts on the deck ─────────────────────────────────────
    robot = cfg.scene.entities["robot"]
    ox, oy, oz = DESIGN.robot_offset
    robot.init_state.pos = (ox, oy, DESIGN.wheel_r + oz)
    cfg.events.pop("reset_base", None)
    cfg.events["reset_on_the_board"] = EventTermCfg(
        func=segway_events.reset_on_the_board,
        mode="reset",
        # Replay starts facing +x, so a scripted route means the same place
        # every time (`lab/route_drive.py`).
        params={"yaw_range": (0.0, 0.0), "design": design_name} if play else {"design": design_name},
    )
    # Pushes arrive through a deck the size of the stance; halved, as the swing's.
    if "push_robot" in cfg.events:
        ranges = cfg.events["push_robot"].params["velocity_range"]
        for key, (lo, hi) in list(ranges.items()):
            ranges[key] = (lo / 2.0, hi / 2.0)

    # ── What it is paid for ──────────────────────────────────────────────
    # Standing square and centred on a level board collects 1.5 + 1.0 before the
    # command pays anything, so staying on is always worth more than falling off
    # -- the swing's first run showed what a negative reward rate does.
    cfg.rewards["track_ride_speed"] = RewardTermCfg(
        func=segway_rewards.track_ride_speed, weight=3.0, params={"std": 0.25}
    )
    cfg.rewards["track_ride_yaw"] = RewardTermCfg(
        func=segway_rewards.track_ride_yaw, weight=1.5, params={"std": 0.4}
    )
    cfg.rewards["deck_upright"] = RewardTermCfg(
        func=segway_rewards.deck_upright, weight=1.5, params={"deadband": 0.26, "std": 0.35}
    )
    cfg.rewards["centred_on_the_deck"] = RewardTermCfg(
        func=segway_rewards.centred_on_the_deck, weight=1.0, params={"std": 0.12}
    )
    cfg.rewards["pose"] = RewardTermCfg(
        func=segway_rewards.posture,
        weight=0.125,
        params={
            "std": {_ARM_JOINTS: 0.60, _LEG_JOINTS: 0.30},
            "asset_cfg": SceneEntityCfg("robot", joint_names=GAIT_JOINTS),
        },
    )
    cfg.rewards["body_tilt_rate_on_the_deck"] = RewardTermCfg(
        func=segway_rewards.body_tilt_rate_on_the_deck, weight=-0.05, params={"deadband": 0.5}
    )
    cfg.rewards["feet_slip_on_the_deck"] = RewardTermCfg(
        func=segway_rewards.feet_slip_on_the_deck, weight=-0.2, params={}
    )
    cfg.rewards["feet_off_the_deck"] = RewardTermCfg(
        func=segway_rewards.feet_off_the_deck, weight=-0.2, params={}
    )
    cfg.rewards["board_pitch"] = RewardTermCfg(
        func=segway_rewards.board_pitch, weight=-1.0, params={}
    )

    # ── When it is over ──────────────────────────────────────────────────
    cfg.terminations["off_the_deck"] = TerminationTermCfg(
        func=segway_terms.off_the_deck, params={"margin": 0.10, "drop": 0.05}
    )
    cfg.terminations["tipped_on_the_deck"] = TerminationTermCfg(
        func=segway_terms.tipped_on_the_deck, params={"limit_angle": math.radians(50.0)}
    )
    cfg.terminations["board_over"] = TerminationTermCfg(
        func=segway_terms.board_over, params={"limit_angle": BOARD_OVER_ANGLE}
    )

    # ── What it sees ─────────────────────────────────────────────────────
    # The actor: proprioception plus the command. The board's pitch reaches it
    # through its own IMU, so the IMU gets a history, as on the swing.
    for group in cfg.observations.values():
        group.terms["ride_command"] = ObservationTermCfg(
            func=segway_obs.ride_command, history_length=1, params={"mirror_kind": "twist"}
        )
        for name in ("projected_gravity", "base_ang_vel"):
            if name in group.terms:
                group.terms[name].history_length = IMU_HISTORY
    for name, func, kind in (
        ("board_gravity", segway_obs.board_gravity, "vec3"),
        ("board_lin_vel", segway_obs.board_lin_vel, "vec3"),
        ("board_ang_vel", segway_obs.board_ang_vel, "pseudo3"),
        ("board_speed_command", segway_obs.board_speed_command, "scalar"),
        ("deck_up", segway_obs.deck_up, "vec3"),
        ("deck_offset", segway_obs.deck_offset, "vec3"),
    ):
        cfg.observations["critic"].terms[name] = ObservationTermCfg(
            func=func, history_length=1, params={"mirror_kind": kind}
        )

    # The face screen, replay only (`face.py`): visual, massless, read by nothing.
    if play:
        from .face import add_face_screen

        build_robot = robot.spec_fn
        robot.spec_fn = lambda: add_face_screen(build_robot())

    if os.environ.get(TERRAIN_ENV, "").strip() == "road":
        cfg.scene.terrain.terrain_type = "generator"
        cfg.scene.terrain.terrain_generator = road_terrain()
        print("[segway] terrain: road (flat, bumps, slopes up to ~10 deg)")

    if not play:
        cfg.episode_length_s = 20.0
    cfg.viewer.distance = 2.4
    cfg.viewer.elevation = -15.0
    return cfg
