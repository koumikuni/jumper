"""`scripts/play.py`, with the crab's face drawn on its screen in the recorded video.

Takes exactly play.py's arguments. Before play starts, `ReplayVideo` is wrapped so
that before every recorded frame the mood is judged (`tasks/jumper/segway/face.py`)
and the face texture is uploaded to the renderer's GL context. Also writes
`<video>.moods.json`: the mood at every recorded frame, for captions and the
interview video.

Two more things, for the city film:

- A command script that defines `script.bind(env)` is handed the environment
  before its first call, so it can steer on the board's real pose
  (`lab/route_drive.py`). play.py's own scripts only see the time.
- With `CRAB_EXPORT=<dir>`, every recorded frame's visual geom poses and face are
  written there for Blender (`lab/blender/build_city.py`): `scene.npz` with the
  meshes and per-frame poses, and `face/0001.png ...`.

    jumper/.venv/Scripts/python.exe lab/ride_record.py --task jumper.segway ... --video x.mp4
"""

from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path

import numpy as np

# The framework checkout: `jumper/` beside `lab/` while developing, or the
# repository itself when `lab/` is published inside it.
_ROOT = Path(__file__).resolve().parents[1]
JUMPER = _ROOT / "jumper" if (_ROOT / "jumper" / "tasks").is_dir() else _ROOT


class _Exporter:
    """Collects visual geom poses per frame from the replay renderer's model."""

    def __init__(self, out: Path, model) -> None:
        import mujoco

        self.out = out
        (out / "face").mkdir(parents=True, exist_ok=True)
        keep = []
        for g in range(model.ngeom):
            if model.geom_group[g] > 2 or model.geom_rgba[g, 3] == 0:
                continue
            body = model.geom_bodyid[g]
            if body == 0:  # the floor and anything else welded to the world
                continue
            keep.append(g)
        self.ids = np.array(keep)
        self.names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
                      for g in keep]
        self.meshes = {}
        for g in keep:
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                m = int(model.geom_dataid[g])
                if m not in self.meshes:
                    va, vn = model.mesh_vertadr[m], model.mesh_vertnum[m]
                    fa, fn = model.mesh_faceadr[m], model.mesh_facenum[m]
                    self.meshes[m] = (model.mesh_vert[va:va + vn].copy(),
                                      model.mesh_face[fa:fa + fn].copy())
        self.model = model
        self.pos, self.mat, self.t, self.moods = [], [], [], []

    def frame(self, data, t: float, mood: str, face_rgb: np.ndarray, count: int) -> None:
        from PIL import Image

        for _ in range(count):
            self.pos.append(data.geom_xpos[self.ids].astype(np.float32))
            self.mat.append(data.geom_xmat[self.ids].astype(np.float32))
            self.t.append(t)
            self.moods.append(mood)
            Image.fromarray(face_rgb).save(self.out / "face" / f"{len(self.t):04d}.png")

    def write(self, fps: int) -> None:
        m = self.model
        meshes = sorted(self.meshes)
        np.savez_compressed(
            self.out / "scene.npz",
            fps=fps, t=np.array(self.t), names=np.array(self.names),
            type=m.geom_type[self.ids], size=m.geom_size[self.ids], rgba=m.geom_rgba[self.ids],
            meshid=np.array([meshes.index(int(m.geom_dataid[g])) if int(m.geom_dataid[g]) in self.meshes
                             else -1 for g in self.ids]),
            pos=np.stack(self.pos), mat=np.stack(self.mat),
            **{f"mesh{i}_v": self.meshes[k][0] for i, k in enumerate(meshes)},
            **{f"mesh{i}_f": self.meshes[k][1] for i, k in enumerate(meshes)},
        )
        (self.out / "moods.json").write_text(json.dumps(self.moods), encoding="utf-8")


def install() -> None:
    import mujoco

    from mjrl import replay
    from mjrl.viewer import video as video_mod
    from tasks.jumper.segway import face

    original_scripts = replay.script_commands

    def script_commands(env, script):
        if hasattr(script, "bind"):
            script.bind(env)
        return original_scripts(env, script)

    replay.script_commands = script_commands

    cls = video_mod.ReplayVideo
    original_write, original_close = cls._write_frames, cls.close
    export_dir = os.environ.get("CRAB_EXPORT")

    def _write_frames(self, count):
        r = self._renderer
        if not hasattr(self, "_face"):
            model = r._model
            tex = next(i for i in range(model.ntex)
                       if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TEXTURE, i) or "")
                       .endswith(face.TEXTURE))
            # Riding has a command to judge against; getting on has none, and
            # its moods come from what the board does (`face.BoardingMood`).
            riding = "ride" in self._env.command_manager.active_terms
            self._face = (face.Mood() if riding else face.BoardingMood(), tex, [])
            self._export = _Exporter(Path(export_dir), model) if export_dir else None
        mood, tex, log = self._face
        t = self._steps * self._dt
        name = mood.update(self._env, self._env_index, t)
        log.append({"t": round(t, 3), "mood": name, "reason": getattr(mood, "reason", None)})
        model = r._model
        adr, size = model.tex_adr[tex], face.TEX_W * face.TEX_H * 3
        rgb = face.draw(name, t)
        # MuJoCo's texture rows run bottom-up.
        model.tex_data[adr:adr + size] = rgb[::-1].reshape(-1)
        gl = r.renderer
        gl._gl_context.make_current()
        mujoco.mjr_uploadTexture(model, gl._mjr_context, tex)
        result = original_write(self, count)
        if self._export is not None:
            self._export.frame(r._data, t, name, rgb, count)
        return result

    def close(self):
        if hasattr(self, "_face") and self.path is not None:
            Path(str(self.path) + ".moods.json").write_text(
                json.dumps(self._face[2], ensure_ascii=False), encoding="utf-8")
            if self._export is not None and self._export.t:
                self._export.write(self.fps)
        return original_close(self)

    cls._write_frames = _write_frames
    cls.close = close


if __name__ == "__main__":
    sys.path.insert(0, str(JUMPER))
    sys.path.insert(0, str(JUMPER / "scripts"))
    sys.path.insert(0, str(Path(__file__).parent))
    install()
    sys.argv = [str(JUMPER / "scripts" / "play.py"), *sys.argv[1:]]
    runpy.run_path(sys.argv[0], run_name="__main__")
