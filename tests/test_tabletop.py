"""The live arm and physical objects must actually rest on a tabletop."""
import unittest
import mujoco
import numpy as np
from perception.collect import DEFAULT_SCENE


class TabletopTests(unittest.TestCase):
    def test_free_objects_settle_on_table_and_home_restores_their_poses(self):
        model = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE))
        data = mujoco.MjData(model)
        top = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'tabletop')
        self.assertGreaterEqual(top, 0, 'The default scene needs a table')
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        surface = data.geom_xpos[top, 2]+model.geom_size[top, 2]
        objects = [(f'table_object_{i}', height) for i, height in enumerate((.023, .035, .042))]
        for name, height in objects:
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            self.assertGreaterEqual(body, 0)
            self.assertGreater(data.xpos[body, 2], surface, 'Home must retain free-object positions')
        mujoco.mj_step(model, data, nstep=1000)
        mujoco.mj_forward(model, data)
        for name, height in objects:
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            self.assertAlmostEqual(data.xpos[body, 2]-height, surface, delta=.005)
            joint = model.body_jntadr[body]
            velocity = model.jnt_dofadr[joint]
            self.assertLess(np.linalg.norm(data.qvel[velocity:velocity+3]), .005)  # m/s
            self.assertLess(np.linalg.norm(data.qvel[velocity+3:velocity+6]), .03)  # rad/s
        self.assertTrue(np.isfinite(data.qpos).all())
        self.assertFalse(any(w.number for w in data.warning))
        self.assertEqual(model.nu, 7)
        mount = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'arm_mount')
        self.assertAlmostEqual(data.xpos[mount, 2], surface, places=6)


if __name__ == '__main__':
    unittest.main()
