import unittest
import numpy as np
import torch
from embodiment.model import EntityState, SpatialEntity, discover, masked_position_loss


class EntityModelTests(unittest.TestCase):
    def test_discovery_rejects_stationary_and_independent_features(self):
        rng = np.random.default_rng(12)
        q = rng.uniform(-1, 1, (180, 7)).astype(np.float32)
        xyz = np.zeros((180, 3, 3), np.float32)
        xyz[:, :, 2] = 1
        xyz[:, 0, 0] = q[:, 0]*.1
        xyz[:, 2, 0] = rng.normal(0, .1, 180)
        train = np.arange(180) < 120
        selected, _, _, _ = discover(q, xyz, np.ones((180, 3), bool), train, ~train)
        np.testing.assert_array_equal(selected, [0])

    def test_missing_tracks_do_not_supply_learning_targets(self):
        prediction = torch.zeros(2, 4, 3, requires_grad=True)
        observed = torch.ones_like(prediction)
        valid = torch.ones(2, 4, dtype=torch.bool)
        valid[0, 1] = False
        masked_position_loss(prediction, observed, valid).backward()
        self.assertEqual(float(prediction.grad[0, 1].abs().sum()), 0.)
        self.assertGreater(float(prediction.grad[1, 1].abs().sum()), 0.)

    def test_no_evidence_is_unknown_and_camera_loss_is_not_new_observation(self):
        self.assertEqual(EntityState().observe(np.zeros(7), 0)['status'], 'unknown')
        model = SpatialEntity(4, np.zeros(7), np.ones(7), np.array([[0, 0, 1]]*4))
        state = EntityState(model)
        visible = state.observe(np.zeros(7), 1, np.array([[0, 0, 1]]*4), np.ones(4, bool))
        missing = state.observe(np.zeros(7), 5)
        self.assertEqual(visible['identity'], missing['identity'])
        self.assertEqual(missing['status'], 'predicted_unobserved')
        self.assertFalse(missing['camera_evidence'])
        self.assertGreater(missing['uncertainty_radius_m'], visible['uncertainty_radius_m'])
        contradiction = state.observe(np.zeros(7), 6, np.array([[1, 0, 1]]*4), np.ones(4, bool))
        self.assertEqual(contradiction['status'], 'visual_contradiction')
        self.assertEqual(state.last_confirmed_time, 1)
