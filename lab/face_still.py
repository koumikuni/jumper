"""Still render of the robot with the face screen, to place and orient the quad."""
import sys
import mujoco, numpy as np, imageio.v2 as io
sys.path.insert(0, ".")
from tasks.jumper.segway import face
spec = mujoco.MjSpec.from_file("assets/jumper/jumper.xml")
face.add_face_screen(spec)
spec.texture(face.TEXTURE).data = face.draw(sys.argv[1] if len(sys.argv) > 1 else "happy", 0.0)[::-1].tobytes()
spec.worldbody.add_light(pos=[0.6, 0, 0.8], dir=[-0.6, 0, -0.8])
m = spec.compile(); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 360, 640)
cam = mujoco.MjvCamera(); cam.lookat[:] = d.xpos[m.body("base_link").id]; cam.distance = 0.45
out = []
for az, el in ((180, -15), (180, -45), (150, -30)):
    cam.azimuth, cam.elevation = az, el
    r.update_scene(d, cam); out.append(r.render())
io.imwrite("../lab/out/face_still.png", np.concatenate(out, 1))
g = m.geom("face_screen"); print("face geom pos", d.geom_xpos[g.id].round(4))
