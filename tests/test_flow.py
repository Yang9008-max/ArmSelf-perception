"""Observation-only flow tests; no simulator or robot body labels are used."""

import unittest

import cv2
import numpy as np

from perception.flow import estimate_flow, probe_supervision


def textured_rgb(seed=3, height=96, width=128):
    random = np.random.default_rng(seed)
    gray = random.integers(25, 231, (height, width), dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (3, 3), 0.65)
    return np.repeat(gray[..., None], 3, axis=2)


def moving_patch_probes():
    background = textured_rgb()
    # The interior of this unchanging flat region has no evidence of motion.
    background[:24, :28] = 120
    patch = textured_rgb(seed=8, height=40, width=44)
    x, y = 46, 30

    def render(dx=0, dy=0):
        result = background.copy()
        result[y + dy:y + dy + 40, x + dx:x + dx + 44] = patch
        return result

    directions = [(2, 0), (0, 2), (2, 1), (-1, 2), (1, -2), (2, -1), (-2, -1)]
    probes, increments = [], []
    for axis, (dx, dy) in enumerate(directions):
        for sign in (1, -1):
            probes.append(render(sign * dx, sign * dy))
            delta = np.zeros(7, dtype=np.float32)
            delta[axis] = sign
            increments.append(delta)
    return render(), np.stack(probes), np.stack(increments), np.ones(7, np.float32)


class FlowTests(unittest.TestCase):
    def test_known_translation_has_correct_direction_and_magnitude(self):
        source = textured_rgb()
        transform = np.array([[1, 0, 2], [0, 1, -1]], np.float32)
        destination = cv2.warpAffine(source, transform, (source.shape[1], source.shape[0]),
                                     borderMode=cv2.BORDER_REFLECT)
        flow, valid = estimate_flow(source, destination)
        interior = flow[16:-16, 16:-16]
        np.testing.assert_allclose(np.median(interior, axis=(0, 1)), [2, -1], atol=0.15)
        self.assertGreater(valid[16:-16, 16:-16].mean(), 0.95)
        self.assertEqual(flow.dtype, np.float32)
        self.assertEqual(valid.dtype, np.bool_)

    def test_signed_probes_produce_positive_negative_and_unknown(self):
        result = probe_supervision(*moving_patch_probes())
        labels = result["labels"]
        self.assertTrue(result["diagnostics"]["excitation_ok"])
        self.assertGreater(np.mean(labels[42:58, 59:77] == 1), 0.85)
        self.assertGreater(np.mean(labels[32:70, 8:24] == 0), 0.85)
        self.assertTrue(np.all(labels[5:17, 5:19] == -1))
        self.assertTrue(np.all(result["confidence"][labels < 0] == 0))
        self.assertEqual(result["teacher_j"].shape, (7, 2, 96, 128))
        np.testing.assert_allclose(
            np.median(result["teacher_j"][0, :, 42:58, 59:77], axis=(1, 2)),
            [2, 0], atol=0.20)
        self.assertLess(result["diagnostics"]["heldout_negative_moving_error_px"],
                        result["diagnostics"]["heldout_negative_zero_error_px"])

    def test_wrong_feedback_breaks_heldout_prediction(self):
        rgb, probes, increments, scales = moving_patch_probes()
        correct = probe_supervision(rgb, probes, increments, scales)
        # Keep signed and full-rank feedback but associate negative observations
        # with the wrong joint. This preserves excitation while breaking causality.
        wrong = increments.copy()
        wrong[1::2] = np.roll(wrong[1::2], shift=2, axis=1) + np.diag([-0.10] * 7)
        incorrect = probe_supervision(rgb, probes, wrong, scales)
        self.assertTrue(incorrect["diagnostics"]["excitation_ok"])
        self.assertGreater(incorrect["diagnostics"]["heldout_negative_moving_error_px"],
                           correct["diagnostics"]["heldout_negative_moving_error_px"] * 3)
        self.assertLess(np.mean(incorrect["labels"][42:58, 59:77] == 1), 0.20)

    def test_insufficient_excitation_remains_unknown(self):
        rgb, probes, increments, scales = moving_patch_probes()
        increments[:, 3] = 0
        result = probe_supervision(rgb, probes, increments, scales)
        self.assertFalse(result["diagnostics"]["excitation_ok"])
        self.assertTrue(np.all(result["labels"] == -1))
        self.assertTrue(np.isfinite(result["teacher_j"]).all())

    def test_uniform_static_pixels_are_not_background_labels(self):
        source = np.full((32, 40, 3), 100, np.uint8)
        probes = np.repeat(source[None], 14, axis=0)
        increments = np.repeat(np.eye(7, dtype=np.float32), 2, axis=0)
        increments[1::2] *= -1
        result = probe_supervision(source, probes, increments, np.ones(7))
        self.assertTrue(np.all(result["labels"] == -1))

    def test_interpolated_motion_in_unchanged_background_is_not_positive(self):
        background = np.full((96, 128, 3), 120, np.uint8)
        patch = textured_rgb(seed=19, height=32, width=20)

        def render(dx=0, dy=0):
            image = background.copy()
            image[32 + dy:64 + dy, 48 + dx:68 + dx] = patch
            return image

        directions = [(2, 0), (0, 2), (2, 1), (-1, 2), (1, -2), (2, -1), (-2, -1)]
        probes = np.stack([render(sign * dx, sign * dy)
                           for dx, dy in directions for sign in (1, -1)])
        increments = np.repeat(np.eye(7, dtype=np.float32), 2, axis=0)
        increments[1::2] *= -1
        result = probe_supervision(render(), probes, increments, np.ones(7))
        # This strip is outside every translated patch and never changes, but
        # dense flow interpolates adjacent patch motion into it.
        strip = np.s_[36:60, 71:78]
        apparent_motion = np.linalg.norm(result["flow"], axis=-1).max(axis=0)[strip]
        self.assertGreater(np.sum(apparent_motion > 0.35), 15)
        self.assertTrue(np.all(result["labels"][strip] == -1))
        self.assertGreater(np.mean(result["labels"][42:54, 54:62] == 1), 0.75)
        self.assertGreater(result["diagnostics"]["uncorroborated_motion_fraction"], 0)


if __name__ == "__main__":
    unittest.main()
