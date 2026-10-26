import unittest

import numpy as np

from embodiment.recovery import VisualBodyBank, VisualFeedbackCorrection, farthest_poses


class LinearBody:
    def __init__(self):
        self.jacobian = np.random.default_rng(23).normal(0., .07, (12, 3, 7))

    def __call__(self, q):
        return np.einsum('bq,pcq->bpc', np.atleast_2d(q), self.jacobian)+np.array([0, 0, 1.])


class RecoveryTests(unittest.TestCase):
    def test_visual_residual_can_identify_unknown_encoder_offset(self):
        body = LinearBody()
        correction = VisualFeedbackCorrection(body, np.ones(7))
        q = np.array([.12, -.08, .02, .03, -.01, .01, .02])
        bias = np.array([.04, -.03, .02, -.02, .01, .02, -.01])
        observed = body(q)[0]
        before = np.linalg.norm(correction.predict(q+bias)-observed)
        for time in range(8):
            result = correction.update(q+bias, observed, np.ones(12, bool), time)
        self.assertTrue(result['updated'])
        self.assertLess(np.linalg.norm(correction.predict(q+bias)-observed), before*.01)
        np.testing.assert_allclose(correction.offset, bias, atol=.0001)

    def test_missing_or_grossly_wrong_vision_cannot_change_feedback_state(self):
        body = LinearBody()
        correction = VisualFeedbackCorrection(body, np.ones(7))
        for observed, valid in [(body(np.zeros(7))[0], np.zeros(12, bool)),
                                (body(np.zeros(7))[0]+.5, np.ones(12, bool))]:
            result = correction.update(np.zeros(7), observed, valid, 1.)
            self.assertFalse(result['updated'])
            np.testing.assert_array_equal(correction.offset, np.zeros(7))
        self.assertIsNone(correction.last_support)

    def test_blank_cameras_do_not_turn_a_spatial_prior_into_visual_evidence(self):
        gray = np.random.default_rng(11).integers(0, 256, (1, 120, 160), dtype=np.uint8)
        uv = np.array([[[50., 50.], [80., 60.], [110., 70.]]], np.float32)
        calibration = dict(focal_px=100., baseline_m=.06, cx=79.5, cy=59.5)
        bank = VisualBodyBank(dict(gray=gray, q=np.zeros((1, 7)), uv=uv,
            valid=np.ones((1, 3), bool), scale=np.ones(7)), calibration)
        prediction = np.c_[(uv[0]-[79.5, 59.5])/100., np.ones(3)]
        blank = np.zeros((120, 160, 3), np.uint8)
        result = bank.observe(blank, blank, np.ones((120, 160), bool), np.zeros(7), prediction)
        self.assertFalse(result['valid'].any())
        self.assertTrue(np.all(result['reference'] == -1))

    def test_pose_memory_is_bounded_and_training_indices_are_unique(self):
        q = np.random.default_rng(7).normal(size=(100, 7))
        indices = farthest_poses(q, np.ones(7), budget=16)
        self.assertEqual(len(indices), 16)
        self.assertEqual(len(set(indices)), 16)
        self.assertTrue(np.all((indices >= 0) & (indices < len(q))))


if __name__ == '__main__':
    unittest.main()
