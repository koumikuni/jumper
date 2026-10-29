"""An original, generated BGM track: no samples, no licence to clear.

A light pentatonic tune over a bass line and soft drums, built from sine and
triangle waves with simple envelopes. Sections follow the film: a bright
opening, a quieter middle while the crab struggles, and a lift for the errand.

    lab/.venv/Scripts/python.exe lab/bgm.py --seconds 60 --out lab/out/bgm.wav
"""

from __future__ import annotations

import argparse
import wave

import numpy as np

SR = 44100
BPM = 112
BEAT = 60.0 / BPM

# C major pentatonic, MIDI numbers.
MELODY_A = [72, 76, 79, 76, 81, 79, 76, 74, 72, 74, 76, 79, 76, 74, 72, None]
MELODY_B = [69, 72, 74, 72, 69, None, 67, 69, 72, None, 74, 72, 69, 67, 69, None]
MELODY_C = [76, 79, 81, 84, 81, 79, 76, 79, 81, 79, 76, 74, 76, 79, 84, None]
BASS = [48, 48, 55, 55, 45, 45, 52, 52, 53, 53, 48, 48, 55, 55, 43, 43]


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def tone(freq, dur, kind="sine", attack=0.01, release=0.12, vol=0.3):
    t = np.arange(int(dur * SR)) / SR
    if kind == "sine":
        w = np.sin(2 * np.pi * freq * t)
    else:  # triangle
        w = 2 * np.abs(2 * ((freq * t) % 1) - 1) - 1
    env = np.minimum(1.0, t / attack) * np.exp(-t / max(release, 1e-3))
    return w * env * vol


def kick(vol=0.5):
    t = np.arange(int(0.18 * SR)) / SR
    f = 120 * np.exp(-t * 25) + 45
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 18) * vol


def hat(vol=0.08, rng=np.random.default_rng(3)):
    n = int(0.05 * SR)
    return rng.standard_normal(n) * np.exp(-np.arange(n) / SR * 80) * vol


def add(buf, sig, at):
    i = int(at * SR)
    j = min(len(buf), i + len(sig))
    if i < len(buf):
        buf[i:j] += sig[: j - i]


def build(seconds: float, quiet=(15.0, 38.0)) -> np.ndarray:
    buf = np.zeros(int(seconds * SR) + SR)
    step = BEAT / 2  # eighth notes
    n = int(seconds / step)
    for k in range(n):
        t = k * step
        section = "B" if quiet[0] <= t < quiet[1] else ("C" if t >= quiet[1] else "A")
        mel = {"A": MELODY_A, "B": MELODY_B, "C": MELODY_C}[section]
        note = mel[k % len(mel)]
        loud = 0.55 if section == "B" else 1.0
        if note is not None:
            add(buf, tone(hz(note), step * 1.6, "sine", release=0.18, vol=0.18 * loud), t)
            add(buf, tone(hz(note + 12), step * 0.9, "sine", release=0.06, vol=0.04 * loud), t)
        if k % 2 == 0:
            b = BASS[(k // 2) % len(BASS)]
            add(buf, tone(hz(b), BEAT * 0.95, "triangle", release=0.25, vol=0.16 * loud), t)
        if section != "B" and k % 4 == 0:
            add(buf, kick(0.35), t)
        if section != "B" and k % 2 == 1:
            add(buf, hat(), t)
        if section == "B" and k % 8 == 0:
            add(buf, kick(0.18), t)
    buf = buf[: int(seconds * SR)]
    fade = int(1.5 * SR)
    buf[-fade:] *= np.linspace(1, 0, fade)
    buf[: int(0.3 * SR)] *= np.linspace(0, 1, int(0.3 * SR))
    return buf / max(1e-6, np.abs(buf).max()) * 0.6


def sfx(kind: str) -> np.ndarray:
    """Short cues for the edit: pop (caption), boing (fall), sparkle (success), puff (angry)."""
    if kind == "pop":
        out = tone(880, 0.12, release=0.04, vol=0.35)
        add(out, tone(1320, 0.08, release=0.03, vol=0.2), 0.0)
        return out
    if kind == "boing":
        t = np.arange(int(0.5 * SR)) / SR
        f = 220 + 120 * np.sin(2 * np.pi * 9 * t) * np.exp(-t * 4)
        return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 5) * 0.4
    if kind == "sparkle":
        out = np.zeros(int(0.6 * SR))
        for i, m in enumerate((84, 88, 91, 96)):
            add(out, tone(hz(m), 0.3, release=0.12, vol=0.22), i * 0.07)
        return out
    if kind == "puff":
        t = np.arange(int(0.3 * SR)) / SR
        f = 300 - 500 * t
        return np.sign(np.sin(2 * np.pi * np.cumsum(f) / SR)) * np.exp(-t * 10) * 0.12
    raise ValueError(kind)


def write_wav(path, x):
    x16 = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(x16.tobytes())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    write_wav(a.out, build(a.seconds))
    print("wrote", a.out)
