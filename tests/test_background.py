"""Tests of temporal RGB evidence, without simulator labels or geometry."""

import unittest

import numpy as np

from perception.background import extend_background_labels, fit_background_reference


def explored_frames():
    frames = np.full((6, 40, 64, 3), [170, 190, 210], dtype=np.uint8)
    for index in range(len(frames)):
        # Each synthetic foreground position is observed in only one frame.
        x = 5 + index * 8
        frames[index, 20:28, x:x + 5] = [70, 55, 40]
    return frames


class BackgroundTests(unittest.TestCase):
    def test_uniform_sky_receives_seeds_from_repeated_observations(self):
        frames = explored_frames()
        reference = fit_background_reference(frames)
        source = frames[0]
        probes = np.repeat(source[None], 14, axis=0)
        labels = np.full(source.shape[:2], -1, np.float32)
        extended = extend_background_labels(source, probes, labels, reference)
        self.assertTrue(np.all(extended[:16] == 0))
        self.assertTrue(np.all(reference["support"][:16] == 1))
        self.assertEqual(reference["median"].dtype, np.uint8)
        self.assertEqual(reference["support"].dtype, np.float32)
        self.assertTrue(np.all(labels == -1))

    def test_moving_patch_is_not_overwritten_and_unknown_patch_stays_unknown(self):
        frames = explored_frames()
        reference = fit_background_reference(frames)
        source = frames[0]
        probes = np.repeat(source[None], 14, axis=0)
        labels = np.full(source.shape[:2], -1, np.float32)
        labels[22:25, 6:8] = 1
        labels[10, 20] = 1
        labels[2:4, 2:4] = 0
        extended = extend_background_labels(source, probes, labels, reference)
        self.assertTrue(np.all(extended[22:25, 6:8] == 1))
        self.assertEqual(extended[21, 5], -1)
        self.assertTrue(np.all(extended[2:4, 2:4] == 0))
        self.assertEqual(extended[10, 19], -1)  # one-pixel positive exclusion

    def test_one_changed_probe_prevents_negative_seed(self):
        frames = explored_frames()
        reference = fit_background_reference(frames)
        source = frames[0]
        probes = np.repeat(source[None], 14, axis=0)
        probes[9, 6:9, 12:16] = [30, 45, 60]
        labels = np.full(source.shape[:2], -1, np.float32)
        extended = extend_background_labels(source, probes, labels, reference)
        self.assertTrue(np.all(extended[6:9, 12:16] == -1))
        self.assertEqual(extended[5, 12], 0)

    def test_low_reference_support_prevents_negative_seed(self):
        frames = explored_frames()
        frames[:, 2:6, 2:6] = np.arange(6, dtype=np.uint8)[:, None, None, None] * 30
        reference = fit_background_reference(frames)
        source = reference["median"].copy()
        probes = np.repeat(source[None], 14, axis=0)
        labels = np.full(source.shape[:2], -1, np.float32)
        extended = extend_background_labels(source, probes, labels, reference)
        self.assertTrue(np.all(extended[2:6, 2:6] == -1))
        self.assertTrue(np.all(reference["support"][2:6, 2:6] < 0.6))

    def test_reference_remains_frozen_on_new_observations(self):
        reference = fit_background_reference(explored_frames())
        old_median, old_support = reference["median"].copy(), reference["support"].copy()
        source = old_median.copy()
        source[5:10, 40:45] = [0, 0, 0]
        probes = np.repeat(source[None], 14, axis=0)
        labels = np.full(source.shape[:2], -1, np.float32)
        extended = extend_background_labels(source, probes, labels, reference)
        self.assertTrue(np.all(extended[5:10, 40:45] == -1))
        np.testing.assert_array_equal(reference["median"], old_median)
        np.testing.assert_array_equal(reference["support"], old_support)


if __name__ == "__main__":
    unittest.main()
