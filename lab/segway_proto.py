"""Segway-board prototype for the Jumper crab, before it becomes an mjlab task.

The crab is stood in for by a 1.8 kg box on two position-servoed slides, so the
vehicle and its balance controller can be tuned without any learning at all. The
question this file answers: is there a controller that (1) holds the board level
with the rider centred, (2) turns a fore-aft shift of the rider's mass into a
steady speed, and (3) turns a sideways shift into a turn -- all through the
rider's body and nothing else.

Usage:
    python segway_proto.py scan            # gain scan, prints the best rows
    python segway_proto.py demo [--video]  # scripted rider: lean fwd, turn, stop
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace

import mujoco
import numpy as np


@dataclass(frozen=True)
class Vehicle:
    wheel_r: float = 0.09
    track: float = 0.68  # wheel centre to wheel centre
    deck_x: float = 0.46
    deck_y: float = 0.52
    deck_above_axle: float = 0.04
    battery_below_axle: float = 0.04
    m_chassis: float = 0.4
    m_battery: float = 0.6
    m_deck: float = 0.4
    m_wheel: float = 0.25
    roll_stiffness: float = 6.0  # N m / rad, deck on its roll hinge
    roll_damping: float = 0.15
    roll_limit_deg: float = 12.0
    wheel_kv: float = 2.0  # velocity actuator gain, N m / (rad/s)
    wheel_torque: float = 4.0
    wheel_friction: float = 1.0


@dataclass(frozen=True)
class Controller:
    kp: float = 20.0  # m/s^2 per rad of pitch
    kd: float = 4.0  # m/s^2 per rad/s
    kv: float = 4.0  # m/s^2 per m/s of speed
    k_steer: float = 8.0  # rad/s of yaw per rad of deck roll
    v_max: float = 1.5
    yaw_max: float = 2.0


RIDER_MASS = 1.8
RIDER_HALF = (0.12, 0.13, 0.04)
RIDER_COM_ABOVE_DECK = 0.10
CTRL_DT = 0.02
PHYS_DT = 0.005


def build_xml(v: Vehicle) -> str:
    deck_t = 0.01
    deck_z = v.deck_above_axle
    rider_z = deck_z + deck_t + RIDER_COM_ABOVE_DECK
    wy = v.track / 2
    wheel = lambda side, y: f"""
      <body name="wheel_{side}" pos="0 {y} 0">
        <joint name="wheel_{side}" type="hinge" axis="0 1 0" damping="0.002"/>
        <geom type="cylinder" size="{v.wheel_r} 0.02" euler="90 0 0" mass="{v.m_wheel}"
              friction="{v.wheel_friction} 0.01 0.001" rgba="0.15 0.15 0.15 1"/>
        <geom type="box" size="{v.wheel_r*0.7} 0.021 0.01" mass="0" contype="0" conaffinity="0"
              rgba="0.9 0.5 0.1 1"/>
      </body>"""
    return f"""
<mujoco model="segway_proto">
  <option timestep="{PHYS_DT}" integrator="implicitfast"/>
  <visual><global offwidth="960" offheight="540"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.82 0.85 0.88" rgb2="0.72 0.75 0.78"/>
    <material name="grid" texture="grid" texrepeat="8 8"/>
  </asset>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="40 40 0.1" material="grid"/>
    <body name="chassis" pos="0 0 {v.wheel_r}">
      <freejoint/>
      <geom type="capsule" fromto="0 {-wy} 0 0 {wy} 0" size="0.012" mass="{v.m_chassis}"
            rgba="0.5 0.5 0.55 1"/>
      <geom type="box" pos="0 0 {-v.battery_below_axle}" size="0.10 0.14 0.02"
            mass="{v.m_battery}" rgba="0.2 0.4 0.8 1"/>
      {wheel("l", wy)}
      {wheel("r", -wy)}
      <body name="deck" pos="0 0 {deck_z}">
        <joint name="deck_roll" type="hinge" axis="1 0 0" stiffness="{v.roll_stiffness}"
               damping="{v.roll_damping}" range="{-v.roll_limit_deg} {v.roll_limit_deg}"
               limited="true"/>
        <geom type="box" size="{v.deck_x/2} {v.deck_y/2} {deck_t}" mass="{v.m_deck}"
              rgba="0.85 0.35 0.25 1"/>
        <body name="rider" pos="0 0 {rider_z - deck_z}">
          <joint name="rx" type="slide" axis="1 0 0" range="-0.08 0.08" limited="true"
                 damping="5"/>
          <joint name="ry" type="slide" axis="0 1 0" range="-0.08 0.08" limited="true"
                 damping="5"/>
          <geom type="box" size="{RIDER_HALF[0]} {RIDER_HALF[1]} {RIDER_HALF[2]}"
                mass="{RIDER_MASS}" rgba="0.95 0.45 0.2 1" contype="0" conaffinity="0"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="rx" joint="rx" kp="300"/>
    <position name="ry" joint="ry" kp="300"/>
    <velocity name="wheel_l" joint="wheel_l" kv="{v.wheel_kv}"
              forcerange="{-v.wheel_torque} {v.wheel_torque}" forcelimited="true"/>
    <velocity name="wheel_r" joint="wheel_r" kv="{v.wheel_kv}"
              forcerange="{-v.wheel_torque} {v.wheel_torque}" forcelimited="true"/>
  </actuator>
</mujoco>
"""


class Segway:
    """The vehicle's own firmware: reads only what the board can sense about itself."""

    def __init__(self, model: mujoco.MjModel, veh: Vehicle, ctl: Controller):
        self.m, self.veh, self.ctl = model, veh, ctl
        self.chassis = model.body("chassis").id
        self.roll_q = model.jnt_qposadr[model.joint("deck_roll").id]
        self.wl = model.jnt_dofadr[model.joint("wheel_l").id]
        self.wr = model.jnt_dofadr[model.joint("wheel_r").id]
        self.v_cmd = 0.0

    def reset(self):
        self.v_cmd = 0.0

    def sense(self, d: mujoco.MjData):
        R = d.xmat[self.chassis].reshape(3, 3)
        x_axis = R[:, 0]
        pitch = math.atan2(-x_axis[2], math.hypot(x_axis[0], x_axis[1]))  # nose down +
        # Body-frame angular velocity of the free joint is qvel[3:6].
        pitch_rate = d.qvel[4]
        r = self.veh.wheel_r
        speed = 0.5 * (d.qvel[self.wl] + d.qvel[self.wr]) * r
        roll = d.qpos[self.roll_q]
        return pitch, pitch_rate, speed, roll

    def step(self, d: mujoco.MjData):
        c, v = self.ctl, self.veh
        pitch, pitch_rate, speed, roll = self.sense(d)
        acc = c.kp * pitch + c.kd * pitch_rate + c.kv * speed
        self.v_cmd = float(np.clip(self.v_cmd + acc * CTRL_DT, -c.v_max, c.v_max))
        # Deck rolled +x (left side up) means the rider pushed the right side down.
        yaw = float(np.clip(-c.k_steer * roll, -c.yaw_max, c.yaw_max))
        half = v.track / 2
        d.ctrl[2] = (self.v_cmd - yaw * half) / v.wheel_r
        d.ctrl[3] = (self.v_cmd + yaw * half) / v.wheel_r
        return pitch, speed, roll, yaw


def yaw_of(d, body):
    R = d.xmat[body].reshape(3, 3)
    return math.atan2(R[1, 0], R[0, 0])


def run(veh: Vehicle, ctl: Controller, script, seconds: float, frames=None, push=None):
    m = mujoco.MjModel.from_xml_string(build_xml(veh))
    d = mujoco.MjData(m)
    seg = Segway(m, veh, ctl)
    n = int(seconds / CTRL_DT)
    log = []
    renderer = cam = None
    if frames is not None:
        renderer = mujoco.Renderer(m, 540, 960)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = seg.chassis
        cam.distance, cam.elevation, cam.azimuth = 2.2, -18, 135
    for k in range(n):
        t = k * CTRL_DT
        d.ctrl[0], d.ctrl[1] = script(t)
        pitch, speed, roll, yaw = seg.step(d)
        if push and abs(t - push[0]) < CTRL_DT / 2:
            d.xfrc_applied[seg.chassis, 0] = push[1]
        for _ in range(int(CTRL_DT / PHYS_DT)):
            mujoco.mj_step(m, d)
        d.xfrc_applied[:] = 0
        log.append((t, pitch, speed, roll, yaw_of(d, seg.chassis), *d.xpos[seg.chassis]))
        if renderer is not None and k % 2 == 0:
            renderer.update_scene(d, cam)
            frames.append(renderer.render())
        if abs(pitch) > math.radians(45) or not np.isfinite(d.qpos).all():
            break
    return np.array(log)


def lean(x=0.0, y=0.0, t0=0.0):
    return lambda t: (x, y) if t >= t0 else (0.0, 0.0)


def score(veh, ctl):
    """Lower is better. inf means it fell."""
    total = 0.0
    hold = run(veh, ctl, lean(), 6.0, push=(1.0, 6.0))
    if len(hold) < 300:
        return math.inf, {}
    total += np.abs(hold[:, 1]).mean() * 10 + np.abs(hold[-50:, 2]).mean()
    fwd = run(veh, ctl, lean(0.04, t0=0.5), 8.0)
    if len(fwd) < 400:
        return math.inf, {}
    v_end = fwd[-100:, 2].mean()
    total += np.abs(fwd[-100:, 2]).std() * 5 + max(0.0, 0.3 - v_end) * 10
    turn = run(veh, ctl, lean(0.03, 0.05, t0=0.5), 8.0)
    if len(turn) < 400:
        return math.inf, {}
    yaw_rate = np.diff(np.unwrap(turn[-100:, 4])).mean() / CTRL_DT
    return total, {"v_lean4cm": v_end, "yaw_rate": yaw_rate,
                   "pitch_hold_deg": math.degrees(np.abs(hold[:, 1]).max())}


def scan():
    veh = Vehicle()
    rows = []
    for kp in (12, 20, 30, 45, 60):
        for kd in (1, 2, 4, 6):
            for kv in (0.0, 0.5, 1.0, 2.0, 4.0):
                ctl = Controller(kp=kp, kd=kd, kv=kv)
                s, info = score(veh, ctl)
                rows.append((s, kp, kd, kv, info))
    rows.sort(key=lambda r: r[0])
    ok = sum(math.isfinite(r[0]) for r in rows)
    print(f"{ok}/{len(rows)} gain sets stayed up")
    for s, kp, kd, kv, info in rows[:12]:
        print(f"score {s:6.3f}  kp {kp:3}  kd {kd}  kv {kv:3}  {info}")


def demo(video: str | None):
    veh, ctl = Vehicle(), Controller()

    def script(t):
        if t < 1.0:
            return 0.0, 0.0
        if t < 5.0:
            return 0.04, 0.0  # lean forward: go
        if t < 9.0:
            return 0.04, 0.05  # lean forward-right: carve
        if t < 13.0:
            return -0.02, 0.0  # lean back: stop
        return 0.0, 0.0

    frames = [] if video else None
    log = run(veh, ctl, script, 15.0, frames=frames)
    for row in log[::50]:
        t, p, v, r, yaw, x, y, z = row
        print(f"t {t:5.2f}  pitch {math.degrees(p):6.2f}  v {v:5.2f}  "
              f"roll {math.degrees(r):6.2f}  yaw {math.degrees(yaw):7.1f}  x {x:6.2f} y {y:6.2f}")
    print("fell" if len(log) < 15.0 / CTRL_DT else "stayed up for the whole run")
    if video:
        import imageio.v2 as imageio
        imageio.mimsave(video, frames, fps=int(1 / (2 * CTRL_DT)))
        print("wrote", video)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("scan", "demo"))
    ap.add_argument("--video", default=None)
    a = ap.parse_args()
    scan() if a.mode == "scan" else demo(a.video)
