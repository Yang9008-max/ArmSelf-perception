"""A segmentation update must not overwrite the learned physical response."""
import tempfile
import unittest
from pathlib import Path

import torch

from perception.model import BodyPerception
from perception.compose import compose_checkpoints


class CompositionTests(unittest.TestCase):
    def test_body_support_only_removes_candidates_and_stays_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            body, reference = self.checkpoints(root)
            compose_checkpoints(root/'body.pt', root/'motion.pt', root/'supported.pt',
                                support_checkpoint=root/'motion.pt')
            saved = torch.load(root/'supported.pt', weights_only=True)
            model = BodyPerception(**saved['model_config'])
            model.load_state_dict(saved['state_dict'])
            rgb, q = torch.rand(1, 3, 32, 48), torch.rand(1, 7)
            predicted = model(rgb, q)['logits']
            expected = torch.minimum(body(rgb, q)['logits'], reference(rgb, q)['logits'])
            torch.testing.assert_close(predicted, expected, rtol=0, atol=0)
            self.assertTrue(torch.all((predicted >= 0) <= (reference(rgb, q)['logits'] >= 0)))
            predicted.mean().backward()
            self.assertTrue(all(p.grad is None for p in model.support_branch.parameters()))

    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)

    def test_body_gradient_and_updates_cannot_change_frozen_motion(self):
        model = BodyPerception(width=4, separate_motion=True)
        rgb, q = torch.rand(2, 3, 32, 48), torch.rand(2, 7)
        before = model(rgb, q)
        before['logits'].mean().backward()
        self.assertFalse(before['jacobian'].requires_grad)
        self.assertTrue(all(p.grad is None for p in model.motion_branch.parameters()))
        self.assertIsNotNone(model.enc1[0].weight.grad)
        with torch.no_grad():
            model.enc1[0].weight.add_(torch.randn_like(model.enc1[0].weight))
        torch.testing.assert_close(model(rgb, q)['jacobian'], before['jacobian'], rtol=0, atol=0)

    def checkpoints(self, root):
        config = {'channels': 7, 'width': 4, 'use_feedback': True}
        preprocessing = {'q_center': [0.]*7, 'q_scale': [1.]*7, 'probe_scales': [.1]*7}
        nets = []
        for name in ('body', 'motion'):
            net = BodyPerception(**config)
            # A zero-initialized response head would make the preservation test trivial.
            torch.nn.init.normal_(net.motion_head.weight)
            torch.save({'model_config': config, 'preprocessing': preprocessing,
                        'state_dict': net.state_dict()}, root/f'{name}.pt')
            nets.append(net.eval())
        return nets

    def test_composition_preserves_both_outputs_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            body, motion = self.checkpoints(root)
            compose_checkpoints(root/'body.pt', root/'motion.pt', root/'composed.pt')
            checkpoint = torch.load(root/'composed.pt', weights_only=True)
            model = BodyPerception(**checkpoint['model_config']).eval()
            model.load_state_dict(checkpoint['state_dict'])
            rgb, q = torch.rand(1, 3, 32, 48), torch.rand(1, 7)
            torch.testing.assert_close(model(rgb, q)['logits'], body(rgb, q)['logits'], rtol=0, atol=0)
            torch.testing.assert_close(model(rgb, q)['jacobian'], motion(rgb, q)['jacobian'], rtol=0, atol=0)

    def test_mismatched_feedback_units_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.checkpoints(root)
            checkpoint = torch.load(root/'motion.pt', weights_only=True)
            checkpoint['preprocessing']['probe_scales'][6] = .9
            torch.save(checkpoint, root/'motion.pt')
            with self.assertRaisesRegex(ValueError, 'preprocessing'):
                compose_checkpoints(root/'body.pt', root/'motion.pt', root/'composed.pt')
            self.assertFalse((root/'composed.pt').exists())


if __name__ == '__main__':
    unittest.main()
