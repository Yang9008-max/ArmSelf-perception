"""Stationary-feedback controls use image evidence, not object identities."""
import unittest
import numpy as np

from perception.null_evidence import stationary_control_evidence
from tests.test_flow import moving_patch_probes


class NullEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.rgb, probes, _, self.scales = moving_patch_probes()
        self.controls = probes[[0, 2]]
        self.dq = np.zeros((2, 7), np.float32)
        self.labels = np.full(self.rgb.shape[:2], -1, np.int8)

    def test_moving_patch_is_negative_but_unchanged_region_is_not(self):
        labels, evidence, _ = stationary_control_evidence(
            self.rgb, self.controls, self.dq, self.scales, self.labels)
        # Two independent photometric checks deliberately leave ambiguous texture
        # pixels unknown; the textured moving interior should supply a majority.
        self.assertGreater(np.mean(labels[42:58, 59:77] == 0), .7)
        self.assertFalse(evidence[5:17, 5:19].any())

    def test_measured_arm_movement_disables_control(self):
        self.dq[0, 0] = .1
        labels, evidence, diagnostic = stationary_control_evidence(
            self.rgb, self.controls, self.dq, self.scales, self.labels)
        np.testing.assert_array_equal(labels, self.labels)
        self.assertFalse(evidence.any())
        self.assertFalse(diagnostic['feedback_stationary'])

    def test_positive_conflict_becomes_unknown(self):
        self.labels[:] = 1
        labels, evidence, diagnostic = stationary_control_evidence(
            self.rgb, self.controls, self.dq, self.scales, self.labels)
        self.assertGreater(evidence.sum(), 100)
        self.assertTrue(np.all(labels[evidence] == -1))
        self.assertEqual(diagnostic['positive_conflicts'], int(evidence.sum()))

    def test_one_changing_control_is_insufficient(self):
        self.controls[1] = self.rgb
        labels, evidence, _ = stationary_control_evidence(
            self.rgb, self.controls, self.dq, self.scales, self.labels)
        self.assertFalse(evidence.any())
        np.testing.assert_array_equal(labels, self.labels)


if __name__ == '__main__':
    unittest.main()
