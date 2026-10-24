import unittest
import numpy as np

from embodiment.exploration import ExplorationBelief, choose_action


class ExplorationTests(unittest.TestCase):
    def test_valid_observations_reduce_uncertainty_and_missing_points_do_not_update(self):
        model = ExplorationBelief(2, np.zeros(7), np.ones(7))
        q = np.full(7, .2)
        before = model.predict(q[None])[1].copy()
        for _ in range(20):
            model.update(q, np.array([[.1, .2, .9], [5, 5, 5]]), np.array([True, False]))
        after = model.predict(q[None])[1]
        self.assertLess(after[0, 0], before[0, 0]/10)
        self.assertEqual(after[0, 1], before[0, 1])
        np.testing.assert_array_equal(model.count, [20, 0])
        self.assertTrue(np.all(model.coefficients[1] == 0))

    def test_visual_movement_changes_action_preference_with_identical_feedback(self):
        q = np.zeros((40, 7))
        q[:20, 0] = np.linspace(-.6, .6, 20)
        q[20:, 1] = np.linspace(-.6, .6, 20)
        choices = np.zeros((2, 7))
        choices[0, 0], choices[1, 1] = .9, .9
        picked = []
        for moving in (0, 1):
            model = ExplorationBelief(2, np.zeros(7), np.ones(7))
            for index, measured in enumerate(q):
                xyz = np.array([[0., 0., 1.], [0., 0., 1.]])
                xyz[moving, 0] = .1*measured[moving]
                valid = np.array([index < 20, index >= 20])
                model.update(measured, xyz, valid)
            picked.append(int(np.argmax(model.scores(choices))))
        self.assertEqual(picked, [1, 0])

    def test_every_policy_obeys_the_same_command_bounds(self):
        model = ExplorationBelief(4, np.zeros(7), np.ones(7))
        current = np.full(7, .85)
        for policy in ('active', 'random', 'preset'):
            target, info = choose_action(model, current, np.random.default_rng(5), policy, 30)
            self.assertTrue(np.all(np.abs(target) <= 1.+1e-7))
            self.assertTrue(np.all(np.abs(target-current) <= .5+1e-7))
            self.assertTrue(np.isfinite(info['score']))

    def test_saved_belief_preserves_prediction_and_action_scores(self):
        model = ExplorationBelief(1, np.zeros(7), np.ones(7))
        rng = np.random.default_rng(6)
        for _ in range(15):
            q = rng.uniform(-.5, .5, 7)
            model.update(q, np.array([[q[0]*.1, 0., 1.]]), np.array([True]))
        restored = ExplorationBelief.from_state(model.state())
        candidates = rng.uniform(-1, 1, (12, 7))
        np.testing.assert_allclose(restored.predict(candidates)[0], model.predict(candidates)[0])
        np.testing.assert_allclose(restored.scores(candidates), model.scores(candidates))
