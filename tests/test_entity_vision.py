import unittest
import tempfile
from pathlib import Path
import mujoco
import numpy as np
from embodiment.vision import triangulate, project


class EntityVisionTests(unittest.TestCase):
    def test_real_eye_pair_has_parallel_axes_and_metric_baseline(self):
        from embodiment.collect import stereo_scene
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'pair.xml'
            calibration = stereo_scene('sim/scenes/robot-eye-v6.xml', path)
            model = mujoco.MjModel.from_xml_path(str(path))
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            a, b = model.camera('fixed').id, model.camera('entity_right').id
            np.testing.assert_allclose(data.cam_xmat[a], data.cam_xmat[b], atol=1e-8)
            np.testing.assert_allclose(data.cam_xpos[b]-data.cam_xpos[a],
                                       calibration['baseline_m']*data.cam_xmat[a].reshape(3, 3)[:, 0], atol=1e-8)

    def test_metric_stereo_roundtrip_and_invalid_disparity(self):
        calibration = dict(focal_px=400., baseline_m=.06, cx=319.5, cy=239.5)
        xyz = np.array([[.1, -.05, .8], [-.2, .1, 1.4]], dtype=np.float32)
        left = project(xyz, calibration)
        right = left.copy()
        right[:, 0] -= calibration['focal_px']*.06/xyz[:, 2]
        observed, valid = triangulate(left, right, calibration)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(observed, xyz, atol=2e-6)
        _, bad = triangulate(left, left, calibration)
        self.assertFalse(bad.any())
        right[:, 1] += 4
        _, bad = triangulate(left, right, calibration)
        self.assertFalse(bad.any())
