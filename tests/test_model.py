import tempfile
import unittest
from pathlib import Path
import json

import numpy as np
import torch

from perception.model import BodyPerception, predict_flow, partial_segmentation_loss, stationary_negative_loss


class ModelTests(unittest.TestCase):
    def test_stationary_negative_loss_only_updates_supported_pixels(self):
        logits = torch.zeros(1, 1, 3, 4, requires_grad=True)
        evidence = torch.zeros(1, 3, 4, dtype=torch.bool)
        evidence[0, 1, 2] = True
        stationary_negative_loss(logits, evidence).backward()
        self.assertGreater(logits.grad[0, 0, 1, 2], 0)
        self.assertEqual(int((logits.grad != 0).sum()), 1)
        self.assertEqual(float(stationary_negative_loss(logits, torch.zeros_like(evidence)).detach()), 0.)

    def test_feedback_contraction_and_zero_increment(self):
        jacobian = torch.zeros(1, 2, 2, 3, 4)
        jacobian[:, 0, 0] = 2
        jacobian[:, 1, 1] = -3
        prediction = predict_flow(jacobian, torch.tensor([[[.5, 2.], [0., 0.]]]))
        self.assertTrue(torch.all(prediction[0, 0, 0] == 1))
        self.assertTrue(torch.all(prediction[0, 0, 1] == -6))
        self.assertTrue(torch.all(prediction[0, 1] == 0))

    def test_unknown_pixels_have_no_classification_gradient(self):
        logits = torch.zeros(1, 1, 1, 3, requires_grad=True)
        loss = partial_segmentation_loss(logits, torch.tensor([[[1., -1., 0.]]]))
        loss.backward()
        self.assertLess(logits.grad[0, 0, 0, 0], 0)
        self.assertEqual(logits.grad[0, 0, 0, 1], 0)
        self.assertGreater(logits.grad[0, 0, 0, 2], 0)

    def test_network_accepts_non_square_current_images(self):
        model = BodyPerception(channels=7, width=4)
        result = model(torch.rand(2, 3, 32, 48), torch.zeros(2, 7))
        self.assertEqual(tuple(result['jacobian'].shape), (2, 7, 2, 32, 48))
        self.assertEqual(tuple(result['logits'].shape), (2, 1, 32, 48))

    def test_loader_never_needs_evaluation_directory(self):
        from perception.data import ObservationDataset
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'observations/train').mkdir(parents=True)
            (root / 'cache/train').mkdir(parents=True)
            (root / 'metadata.json').write_text(json.dumps({'q_center': [0]*7, 'q_scale': [1]*7}))
            np.savez(root / 'observations/train/a.npz', rgb=np.zeros((8,8,3), dtype=np.uint8), q=np.zeros(7))
            np.savez(root / 'cache/train/a.npz', flow=np.zeros((14,8,8,2)), valid=np.ones((14,8,8),bool),
                     dq_scaled=np.zeros((14,7)), labels=np.full((8,8),-1.))
            dataset = ObservationDataset(root, root/'cache', 'train')
            self.assertEqual(set(dataset[0]), {'rgb','q','flow','valid','dq','labels'})
            self.assertFalse((root/'evaluation').exists())


if __name__ == '__main__':
    unittest.main()
