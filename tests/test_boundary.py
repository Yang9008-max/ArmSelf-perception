"""Boundary refinement must use only RGB and predicted probabilities."""
import unittest

import numpy as np

from perception.boundary import GRABCUT_CONFIG, prediction_mask


class BoundaryTests(unittest.TestCase):
    def test_uncertain_border_follows_rgb_and_inputs_remain_unchanged(self):
        rgb = np.full((64, 80, 3), [180, 190, 200], np.uint8)
        rgb[18:46, 25:55] = [20, 40, 70]
        probability = np.full((64, 80), .01, np.float32)
        probability[14:50, 21:59] = .65
        probability[22:42, 29:51] = .99
        original = rgb.copy(), probability.copy()
        truth = np.zeros((64, 80), bool)
        truth[18:46, 25:55] = True
        refined = prediction_mask(rgb, probability, GRABCUT_CONFIG)
        np.testing.assert_array_equal(refined, truth)
        np.testing.assert_array_equal(rgb, original[0])
        np.testing.assert_array_equal(probability, original[1])
        self.assertGreater(np.count_nonzero((probability >= .5) != truth), 200)

    def test_absent_class_preserves_prediction_and_old_checkpoints_use_raw_mask(self):
        rgb = np.zeros((32, 40, 3), np.uint8)
        for value in (.01, .99):
            probability = np.full(rgb.shape[:2], value)
            np.testing.assert_array_equal(prediction_mask(rgb, probability, GRABCUT_CONFIG),
                                          probability >= .5)
        probability = np.random.default_rng(4).uniform(size=rgb.shape[:2])
        np.testing.assert_array_equal(prediction_mask(rgb, probability), probability >= .5)

    def test_nonfinite_probabilities_and_unsupported_metadata_are_rejected(self):
        rgb = np.zeros((32, 40, 3), np.uint8)
        for value in (np.nan, np.inf, -1, 2):
            with self.assertRaises(ValueError):
                prediction_mask(rgb, np.full(rgb.shape[:2], value), GRABCUT_CONFIG)
        with self.assertRaises(ValueError):
            prediction_mask(rgb, np.zeros(rgb.shape[:2]), {'method': 'unknown'})
