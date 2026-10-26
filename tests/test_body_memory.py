import tempfile
from pathlib import Path
import unittest

import mujoco
import numpy as np

from embodiment.body_memory import BodyMemory, ResidualBelief, sample_mask
from embodiment.body_cycle import choose, recovery_events
from embodiment.body_fixture import make_fixture, render_truth, EvaluationLandmarks


class ToyBody:
    selected = np.arange(4)

    def __call__(self, q):
        q = np.atleast_2d(q)
        result = np.zeros((len(q), 4, 3), np.float32)
        result[:, :, 2] = .8
        result[:, :, 0] = q[:, :1]*.1
        return result


class BodyMemoryTests(unittest.TestCase):
    def test_mask_rejects_outside_and_nonfinite_without_clamping(self):
        mask = np.ones((5, 7), bool)
        uv = np.array([[1, 2], [-1, 2], [7, 2], [np.nan, 2], [2, 5]])
        np.testing.assert_array_equal(sample_mask(mask, uv), [1, 0, 0, 0, 0])

    def test_occlusion_grows_uncertainty_without_fabricating_visual_support(self):
        memory = BodyMemory(4)
        prior = ToyBody()(np.zeros(7))[0]
        observed = prior+np.array([.005, 0, 0])
        seen = memory.update(prior, np.zeros(7), 1., observed, np.ones(4, bool))
        missing = memory.update(prior, np.zeros(7), 4.)
        self.assertEqual(len(missing['prediction']), 4)
        self.assertFalse(missing['visual_evidence'].any())
        self.assertTrue(np.all(missing['variance'] > seen['variance']))
        np.testing.assert_array_equal(missing['last_seen'], np.full(4, 1.))
        impostor = memory.update(prior, np.zeros(7), 5., prior+.2, np.ones(4, bool))
        self.assertFalse(impostor['accepted'].any())
        np.testing.assert_array_equal(impostor['last_seen'], np.full(4, 1.))

    def test_residual_learning_retains_same_decoder_and_only_updates_admitted_points(self):
        body = ToyBody()
        belief = ResidualBelief(body, np.zeros(7), np.ones(7))
        qs = np.zeros((3, 7))
        np.testing.assert_allclose(belief.predict(qs)[0], body(qs))
        original = belief.online.covariance.copy()
        q = np.zeros(7)
        xyz = body(q)[0]+np.array([.008, 0, 0])
        belief.update(q, xyz, np.array([1, 0, 0, 0]))
        self.assertEqual(belief.online.count.tolist(), [1, 0, 0, 0])
        np.testing.assert_array_equal(belief.online.covariance[1:], original[1:])
        restored = ResidualBelief.from_state(body, belief.state())
        np.testing.assert_allclose(restored.predict(qs)[0], belief.predict(qs)[0])

    def test_recovery_requires_original_surface_location_after_real_occlusion(self):
        visible = np.array([[1], [0], [0], [0], [1], [1]], bool)
        hidden = ~visible
        truth = np.zeros((6, 1, 3))
        impostor = np.full_like(truth, .2)
        accepted = np.ones((6, 1), bool)
        events = recovery_events(visible, hidden, accepted, impostor, truth, np.ones(1, bool))
        self.assertEqual(len(events), 1)
        self.assertFalse(events[0]['recovered'])
        impostor[5] = 0
        events = recovery_events(visible, hidden, accepted, impostor, truth, np.ones(1, bool))
        self.assertTrue(events[0]['recovered'])
        self.assertEqual(events[0]['latency_frames'], 1)

    def test_common_bootstrap_and_action_bounds_do_not_require_truth(self):
        belief = ResidualBelief(ToyBody(), np.zeros(7), np.ones(7))
        choices = [choose(belief, np.zeros(7), np.random.default_rng(25), p, 3)[0]
                   for p in ('active', 'random', 'preset')]
        np.testing.assert_array_equal(choices[0], choices[1])
        np.testing.assert_array_equal(choices[1], choices[2])
        for p in ('active', 'random', 'preset'):
            target, _ = choose(belief, np.zeros(7), np.random.default_rng(25), p, 30)
            self.assertTrue(np.all(np.abs(target) <= .5+1e-7))

    def test_delayed_return_after_self_occlusion_remains_a_recovery_event(self):
        visible = np.array([[1], [0], [0], [0], [0], [0], [1]], bool)
        behind = np.array([[0], [1], [1], [1], [0], [0], [0]], bool)
        truth = np.zeros((7, 1, 3))
        events = recovery_events(visible, behind, visible, truth, truth, np.ones(1, bool))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['frame'], 6)
        self.assertTrue(events[0]['recovered'])

    def test_world_fixed_screen_actually_occludes_body_and_props_stay_fixed(self):
        with tempfile.TemporaryDirectory() as directory:
            model, data, calibration = make_fixture(Path(directory), panel=True)
            props = [model.body(f'table_object_{i}').id for i in range(3)]
            start = data.xpos[props].copy()
            panel = model.geom('fixed_occlusion_panel').id
            screen = data.geom_xpos[panel].copy()
            renderer = mujoco.Renderer(model, height=480, width=640)
            try:
                truth = render_truth(renderer, model, data)
                self.assertGreater(int(truth['panel_mask'].sum()), 8000)
                # Reference rays through the upper arm hit body with the screen excluded.
                anchors = EvaluationLandmarks(model, data, np.array([[245, 110], [238, 80], [150, 230]]), calibration)
                score = anchors.score_truth(model, data, truth)
                self.assertTrue(score['behind_panel'].any())
                data.ctrl[0] = .22
                mujoco.mj_step(model, data, nstep=300)
                mujoco.mj_forward(model, data)
                np.testing.assert_array_equal(start, data.xpos[props])
                np.testing.assert_array_equal(screen, data.geom_xpos[panel])
            finally:
                renderer.close()
