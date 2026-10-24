"""Verify that live inference uses only RGB and measured robot feedback."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from perception.model import BodyPerception
from perception.live import LiveBodyPredictor


class LiveInferenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.checkpoint = Path(self.directory.name)/'model.pt'
        config = {'channels': 7, 'width': 4, 'use_feedback': True}
        model = BodyPerception(**config)
        with torch.no_grad():
            model.body_head.weight.zero_()
            model.body_head.bias.fill_(-2)
        self.center = np.arange(7, dtype=np.float32)
        self.scale = np.arange(1, 8, dtype=np.float32)
        torch.save({'model_config': config, 'state_dict': model.state_dict(),
                    'preprocessing': {'q_center': self.center.tolist(),
                                      'q_scale': self.scale.tolist()}}, self.checkpoint)

    def test_measured_feedback_and_rgb_reach_network_without_truth_inputs(self):
        predictor = LiveBodyPredictor(self.checkpoint)
        captured = []
        predictor.network.register_forward_pre_hook(
            lambda module, inputs: captured.append(tuple(value.clone() for value in inputs)))
        rgb = np.full((48, 64, 3), [20, 80, 160], dtype=np.uint8)
        original = rgb.copy()
        measured = self.center+self.scale*np.array([.1, -.2, .3, -.4, .5, -.6, .7])
        result = predictor.predict(rgb, measured)
        inputs, normalized = captured[0]
        self.assertEqual(tuple(inputs.shape), (1, 3, 144, 192))
        np.testing.assert_allclose(inputs[0, :, 50, 50].numpy(), [20/255, 80/255, 160/255], atol=1e-6)
        np.testing.assert_allclose(normalized[0].numpy(), [.1, -.2, .3, -.4, .5, -.6, .7], atol=1e-6)
        self.assertFalse(result['mask'].any())
        np.testing.assert_array_equal(result['overlay'], rgb)
        np.testing.assert_array_equal(rgb, original)
        self.assertEqual(result['probability'].shape, (144, 192))
        self.assertGreater(result['inference_ms'], 0)
        self.assertEqual(len(predictor.metadata['checkpoint_sha256']), 64)

    def test_positive_prediction_is_separate_from_unchanged_camera(self):
        predictor = LiveBodyPredictor(self.checkpoint)
        with torch.no_grad():
            predictor.network.body_head.bias.fill_(2)
        rgb = np.zeros((48, 64, 3), dtype=np.uint8)
        result = predictor.predict(rgb, self.center)
        self.assertTrue(result['mask'].all())
        self.assertEqual(result['mask'].shape, rgb.shape[:2])
        self.assertTrue(np.all(result['overlay'][..., 1] == 120))
        self.assertFalse(rgb.any())

    def test_invalid_or_environment_sized_feedback_cannot_enter_model(self):
        predictor = LiveBodyPredictor(self.checkpoint)
        rgb = np.zeros((48, 64, 3), dtype=np.uint8)
        for feedback in (np.zeros(29), np.full(7, np.nan), np.full(7, np.inf)):
            with self.subTest(shape=feedback.shape):
                with self.assertRaises(ValueError):
                    predictor.predict(rgb, feedback)
        with self.assertRaises(ValueError):
            predictor.predict(rgb.astype(float), self.center)

    def test_viewer_provenance_follows_checkpoint_instead_of_old_version_label(self):
        saved = torch.load(self.checkpoint, weights_only=True)
        saved['deployment'] = {'model': 'V6', 'training_scene': '机器人眼位桌面',
                               'scene_sha256': 'a'*64, 'live_scene_sha256': 'b'*64}
        torch.save(saved, self.checkpoint)
        metadata = LiveBodyPredictor(self.checkpoint).metadata
        self.assertEqual(metadata['model'], 'V6')
        self.assertEqual(metadata['training_scene'], '机器人眼位桌面')
        self.assertEqual(metadata['training_scene_sha256'], 'a'*64)


if __name__ == '__main__':
    unittest.main()
