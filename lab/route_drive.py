"""Drive the ride command along a route, so the crab travels somewhere on its own.

A command script for `lab/ride_record.py` (not plain play.py: it needs the
environment, which `ride_record.py` hands it through `script.bind`). The route is
`lab/city/route.json`: points, a target speed for the leg leaving each point,
and optional stops.

Pure pursuit on the board's own pose: aim at the point `LOOKAHEAD` metres ahead
along the route, and turn at the rate that puts the board on a circle through
it, `w = 2 v sin(alpha) / Ld`. Speed eases between legs and down to a stop, so the
command never asks for a step the rider would have to throw itself at.

    python lab/ride_record.py ... --command-script lab/route_drive.py
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

ROUTE = Path(os.environ.get("CRAB_ROUTE", Path(__file__).parent / "city" / "route.json"))
LOOKAHEAD = 1.2
YAW_MAX = 1.0
ACCEL = 0.4  # m/s^2 the speed command may change by
ARRIVE = 0.35  # metres from a stop point that counts as there

_env = None
_state = {"leg": 0, "v": 0.0, "stop_until": None, "done": False, "last_t": 0.0, "log": []}
_route = json.loads(ROUTE.read_text(encoding="utf-8"))
_pts = [(p["x"], p["y"]) for p in _route["points"]]


def bind(env) -> None:
    global _env
    _env = env


def _pose():
    import torch

    from tasks.jumper.segway.mdp import state

    with torch.inference_mode():
        board = _env.scene[state.ENTITY]
        pos = board.data.root_link_pos_w[0] - _env.scene.env_origins[0]
        axes = state.chassis_axes(_env)[0]
        yaw = math.atan2(float(axes[1, 0]), float(axes[0, 0]))
    return float(pos[0]), float(pos[1]), yaw


def _project(x, y, leg):
    """Closest point on legs `leg..`, as (leg index, fraction along it)."""
    best = (leg, 0.0, float("inf"))
    for i in range(leg, min(leg + 2, len(_pts) - 1)):
        (ax, ay), (bx, by) = _pts[i], _pts[i + 1]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        u = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / L2))
        d = math.hypot(ax + u * dx - x, ay + u * dy - y)
        if d < best[2]:
            best = (i, u, d)
    return best[0], best[1]


def _ahead(leg, u, dist):
    """The point `dist` metres further along the route from (leg, u)."""
    while leg < len(_pts) - 1:
        (ax, ay), (bx, by) = _pts[leg], _pts[leg + 1]
        L = math.hypot(bx - ax, by - ay)
        left = (1.0 - u) * L
        if dist <= left:
            f = u + dist / L
            return ax + f * (bx - ax), ay + f * (by - ay)
        dist -= left
        leg, u = leg + 1, 0.0
    return _pts[-1]


def _remaining_to_stop(leg, u):
    """Metres to the next point that stops (or the end), and that point's index."""
    dist = 0.0
    for i in range(leg, len(_pts) - 1):
        (ax, ay), (bx, by) = _pts[i], _pts[i + 1]
        L = math.hypot(bx - ax, by - ay)
        dist += (1.0 - u) * L if i == leg else L
        nxt = _route["points"][i + 1]
        if nxt.get("stop") or i + 1 == len(_pts) - 1:
            return dist, i + 1
    return 0.0, len(_pts) - 1


def script(t: float):
    s = _state
    dt = max(0.0, t - s["last_t"])
    s["last_t"] = t
    if _env is None or s["done"]:
        return {"ride": [0.0, 0.0, 0.0]}
    x, y, yaw = _pose()
    leg, u = _project(x, y, s["leg"])
    s["leg"] = leg

    if s["stop_until"] is not None:
        if t < s["stop_until"]:
            s["v"] = 0.0
            return {"ride": [0.0, 0.0, 0.0]}
        s["stop_until"] = None

    want = _route["points"][leg]["speed"] or 0.4
    gap, stop_i = _remaining_to_stop(leg, u)
    # Ease down to arrive at the stop: v^2 = 2 a d.
    want = min(want, math.sqrt(max(0.0, 2 * ACCEL * max(0.0, gap - ARRIVE * 0.5))))
    if gap < ARRIVE:
        point = _route["points"][stop_i]
        if stop_i == len(_pts) - 1:
            s["done"] = True
        s["stop_until"] = t + point.get("stop", 0.0)
        s["leg"] = min(stop_i, len(_pts) - 2)
        s["log"].append({"t": round(t, 2), "event": point.get("label", "")})
        s["v"] = 0.0
        return {"ride": [0.0, 0.0, 0.0]}
    step = ACCEL * dt
    s["v"] = max(s["v"] - step, min(s["v"] + step, want))

    tx, ty = _ahead(leg, u, LOOKAHEAD)
    alpha = math.atan2(ty - y, tx - x) - yaw
    alpha = math.atan2(math.sin(alpha), math.cos(alpha))
    v = s["v"]
    w = 2.0 * max(v, 0.25) * math.sin(alpha) / LOOKAHEAD
    return {"ride": [v, 0.0, max(-YAW_MAX, min(YAW_MAX, w))]}


script.bind = bind
script.state = _state
