"""The same short course for every checkpoint, so clips compare like for like.

Used by `play.py --command-script lab/ride_script.py`. Times are seconds since the
replay started; values are the `ride` command (forward m/s, 0, yaw rad/s).
"""

COURSE = (
    (0.0, (0.0, 0.0, 0.0), "stand still"),
    (2.0, (0.6, 0.0, 0.0), "forward 0.6 m/s"),
    (6.0, (0.6, 0.0, -0.8), "turn right"),
    (9.0, (0.3, 0.0, 0.8), "turn left"),
    (12.0, (0.0, 0.0, 0.0), "stop"),
)
DURATION = 14.0


def phase(t: float):
    current = COURSE[0]
    for row in COURSE:
        if t >= row[0]:
            current = row
    return current


def script(t: float):
    return {"ride": list(phase(t)[1])}
