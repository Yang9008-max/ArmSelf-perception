"""Check causal separation and collector observation/evaluation boundaries."""
import tempfile
from pathlib import Path
import unittest
import mujoco
import numpy as np

from perception.collect import DEFAULT_SCENE, collect_dataset
from perception.environment import create_environment_model, create_static_scene_model, EnvironmentController


class EnvironmentTests(unittest.TestCase):
    def test_live_props_stay_fixed_while_robot_joints_move(self):
        model = create_static_scene_model(DEFAULT_SCENE)
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        props = [model.body(f'table_object_{i}').id for i in range(3)]
        positions, rotations = data.xpos[props].copy(), data.xquat[props].copy()
        robot = data.qpos.copy()
        data.ctrl[0] += .3
        mujoco.mj_step(model, data, nstep=500)
        mujoco.mj_forward(model, data)
        np.testing.assert_array_equal(data.xpos[props], positions)
        np.testing.assert_array_equal(data.xquat[props], rotations)
        self.assertGreater(np.linalg.norm(data.qpos - robot), .1)
        self.assertEqual(model.nq, 8)

    def test_shadow_option_changes_rgb_without_changing_robot_or_body_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for enabled in (False, True):
                collect_dataset(root/str(enabled), train=0, val=0, test=1,
                                environment='static', seed=29, shadows=enabled)
            with np.load(root/'False/observations/test/test_00000.npz') as clear, np.load(
                    root/'True/observations/test/test_00000.npz') as shadow:
                np.testing.assert_array_equal(clear['q'], shadow['q'])
                np.testing.assert_array_equal(clear['dq'], shadow['dq'])
                self.assertGreater(np.count_nonzero(np.max(np.abs(
                    clear['rgb'].astype(int)-shadow['rgb'].astype(int)), axis=2) > 10), 50)
            with np.load(root/'False/evaluation/test/test_00000.npz') as clear, np.load(
                    root/'True/evaluation/test/test_00000.npz') as shadow:
                np.testing.assert_array_equal(clear['body_mask'], shadow['body_mask'])

    def test_generated_objects_stay_on_table_and_old_floor_scene_still_loads(self):
        model, _ = create_environment_model(DEFAULT_SCENE)
        self.assertEqual(model.nq, 8, 'Generated scene must replace the three free objects')
        controller = EnvironmentController(model)
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        controller.configure(data, 'moving', np.random.default_rng(11))
        controller.begin_probe(np.random.default_rng(12))
        for elapsed in (.35, 5., 20.):
            data.time = controller.source_time+elapsed
            controller.advance(data)
            positions = data.mocap_pos[controller.mocap_ids]
            np.testing.assert_allclose(positions[:, 2]-controller.support_heights, controller.table_surface)
            self.assertTrue(np.all(np.abs(positions[:, :2]-controller.table_center)
                                   < controller.table_half))
            controller.advance_bounded(data)
            np.testing.assert_allclose(data.mocap_pos[controller.mocap_ids, 2]-controller.support_heights,
                                        controller.table_surface)
        archived = Path(DEFAULT_SCENE).parent/'scenes/floor-v1.xml'
        previous, _ = create_environment_model(archived)
        self.assertIsNone(EnvironmentController(previous).table_surface)

    def test_prop_render_motion_does_not_change_physical_robot_state(self):
        model, _ = create_environment_model(DEFAULT_SCENE)
        controller = EnvironmentController(model)
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        controller.configure(data, 'moving', np.random.default_rng(4))
        controller.begin_probe(np.random.default_rng(5))
        positions, velocities = data.qpos.copy(), data.qvel.copy()
        origins = data.mocap_pos.copy()
        data.time += .35
        controller.advance(data)
        np.testing.assert_array_equal(data.qpos, positions)
        np.testing.assert_array_equal(data.qvel, velocities)
        self.assertGreater(np.linalg.norm(data.mocap_pos-origins), .05)
        props = model.geom_group == 2
        self.assertTrue(np.all(model.geom_contype[props] == 0))
        self.assertTrue(np.all(model.geom_conaffinity[props] == 0))

    def test_collector_keeps_masks_out_of_observations_and_measures_hold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'dataset'
            collect_dataset(root, train=0, val=0, test=1, environment='moving', seed=23)
            with np.load(root/'observations/test/test_00000.npz') as record:
                self.assertFalse(any('mask' in key for key in record.files))
                self.assertEqual(record['null_rgb'].shape, (2, 144, 192, 3))
                self.assertLess(np.abs(record['null_dq']).max(), 1e-5)
                self.assertGreater(np.abs(record['null_rgb'].astype(float)-record['rgb']).sum(), 100)
            with np.load(root/'evaluation/test/test_00000.npz') as record:
                self.assertIn('null_body_mask', record.files)
                self.assertIn('unoccluded_body_mask', record.files)


if __name__ == '__main__':
    unittest.main()
