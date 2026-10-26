from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from embodiment.action_state import ActionTransition, ActionBodyState
from embodiment.action_cycle import inputs


class LinearBody:
    saved = dict(scale=np.ones(7))

    def __init__(self):
        self.jacobian = np.random.default_rng(17).normal(0., .09, (12, 3, 7))

    def __call__(self, q):
        return np.einsum('bq,pcq->bpc', np.atleast_2d(q), self.jacobian)+[0, 0, 1.]


class ImmediateActuator:
    def predict(self, q, velocity, target, duration):
        return np.asarray(target).copy()


class SavedObservation(dict):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class ActionStateTests(unittest.TestCase):
    def make_state(self, **kwargs):
        return ActionBodyState(ImmediateActuator(), LinearBody(), np.zeros(7), **kwargs)

    def test_prediction_precedes_any_future_observation_read(self):
        before = SavedObservation(target=np.ones(7), duration=np.float32(.1))
        after = dict(q=np.zeros(7), left=np.zeros((10, 10, 3), np.uint8),
                     right=np.zeros((10, 10, 3), np.uint8), time=1.)
        with patch('embodiment.action_cycle.read_json', return_value={'frames': [{'file': 'frame.npz'}]}), \
             patch('embodiment.action_cycle.np.load', return_value=before), \
             patch('embodiment.action_cycle.load_npz', return_value=after) as future:
            stream = inputs(Path('unused'), 'clean')
            kind, command = next(stream)
            self.assertEqual(kind, 'command')
            np.testing.assert_array_equal(command['target'], np.ones(7))
            future.assert_not_called()
            self.assertEqual(next(stream)[0], 'observation')
            future.assert_called_once()

    def test_filter_enforces_prediction_then_feedback_then_vision(self):
        state = self.make_state()
        with self.assertRaises(RuntimeError):
            state.feedback(np.zeros(7))
        with self.assertRaises(RuntimeError):
            state.finish()
        state.forecast(np.zeros(7), .1)
        with self.assertRaises(RuntimeError):
            state.forecast(np.zeros(7), .1)
        state.feedback(np.zeros(7))
        state.finish()
        state.forecast(np.ones(7)*.01, .1)

    def test_consistent_vision_can_correct_feedback_that_conflicts_with_action(self):
        state = self.make_state()
        state.forecast(np.zeros(7), .1)
        q = state.feedback(np.array([.08, .04, -.04, 0., 0., 0., 0.]))
        observed = state.body(np.zeros(7))[0]
        old_error = np.linalg.norm(state.body(q)[0]-observed)
        result = state.finish(observed, np.ones(12, bool), 1.)
        self.assertTrue(result['visual_update'])
        self.assertLess(np.linalg.norm(result['xyz']-observed), old_error*.1)
        self.assertGreater(np.linalg.norm(result['bias']), 0.)

    def test_wrong_vision_cannot_override_agreeing_action_and_feedback(self):
        state = self.make_state()
        state.forecast(np.zeros(7), .1)
        state.feedback(np.zeros(7))
        result = state.finish(state.body(np.ones(7)*.04)[0], np.ones(12, bool), 1.)
        self.assertFalse(result['visual_update'])
        self.assertEqual(result['reason'], 'action_and_feedback_already_agree')
        np.testing.assert_array_equal(result['q'], np.zeros(7))

    def test_missing_vision_never_updates_bias_or_invents_support(self):
        state = self.make_state()
        state.forecast(np.zeros(7), .1)
        state.feedback(np.ones(7)*.04)
        result = state.finish()
        self.assertFalse(result['visual_update'])
        self.assertEqual(result['visual_support'], 0)
        np.testing.assert_array_equal(result['bias'], np.zeros(7))

    def test_no_feedback_ablation_is_independent_of_new_sensor_readings(self):
        outputs = []
        for reading in (np.ones(7)*.3, np.ones(7)*-.4):
            state = self.make_state(feedback_enabled=False)
            state.forecast(np.ones(7)*.02, .1)
            state.feedback(reading)
            outputs.append(state.finish()['q'])
        np.testing.assert_array_equal(*outputs)

    def test_no_action_model_cannot_obtain_command_information(self):
        torch.manual_seed(12)
        model = ActionTransition(np.zeros(7), np.ones(7), use_action=False).eval()
        q, velocity = np.zeros(7), np.zeros(7)
        a = model.predict(q, velocity, np.ones(7), .1)
        b = model.predict(q, velocity, -np.ones(7), .1)
        np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()
