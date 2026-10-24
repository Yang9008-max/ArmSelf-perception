"""Image-edge refinement must use action seeds, not a robot color prior."""
import unittest
import numpy as np

from perception.refine import refine_partial_labels


class RefinementTests(unittest.TestCase):
    def scene(self):
        rgb = np.full((48, 64, 3), [170, 190, 210], np.uint8)
        rgb[14:35, 20:44] = [65, 45, 30]
        labels = np.zeros((48, 64), np.float32)
        labels[8:41, 14:50] = -1
        labels[22:27, 29:34] = 1
        truth = np.zeros((48, 64), bool)
        truth[14:35, 20:44] = True
        return rgb, labels, truth

    def test_sparse_seeds_follow_image_edge_instead_of_unknown_halo(self):
        rgb, labels, expected = self.scene()
        refined = refine_partial_labels(rgb, labels)
        predicted = refined == 1
        iou = (predicted & expected).sum() / (predicted | expected).sum()
        self.assertGreater(iou, .95)
        self.assertEqual(refined[10, 18], 0)

    def test_hard_negative_is_preserved_even_with_foreground_appearance(self):
        rgb, labels, _ = self.scene()
        labels[17:20, 23:26] = 0
        refined = refine_partial_labels(rgb, labels)
        self.assertTrue(np.all(refined[17:20, 23:26] == 0))
        self.assertTrue(np.all(refined[labels == 1] == 1))

    def test_without_action_positive_evidence_no_object_is_invented(self):
        rgb, labels, _ = self.scene()
        labels[labels == 1] = -1
        np.testing.assert_array_equal(refine_partial_labels(rgb, labels), labels)

    def test_deterministic_and_inputs_not_modified(self):
        rgb, labels, _ = self.scene()
        before_rgb, before_labels = rgb.copy(), labels.copy()
        first = refine_partial_labels(rgb, labels)
        second = refine_partial_labels(rgb, labels)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(rgb, before_rgb)
        np.testing.assert_array_equal(labels, before_labels)


if __name__ == '__main__':
    unittest.main()
