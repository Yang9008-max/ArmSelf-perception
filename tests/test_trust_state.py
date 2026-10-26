import unittest

import numpy as np
import torch

from embodiment.trust_state import ReliabilityGate, LearnedBodyState
from embodiment.trust_cycle import rollout_train
from embodiment.action_state import ActionTransition


class Body:
    saved = {'scale': np.ones(7, np.float32)}

    def __call__(self, q):
        return np.atleast_2d(q)[:, :6].reshape(-1, 2, 3)


class Transition:
    def predict(self, q, velocity, target, duration):
        return np.asarray(q)+.2*(np.asarray(target)-q)


class TrustStateTests(unittest.TestCase):
    def state(self):
        torch.manual_seed(31)
        return LearnedBodyState(Transition(), ReliabilityGate().eval(), Body(), np.zeros(7))

    def test_missing_sources_have_exactly_zero_contribution(self):
        state = self.state()
        state.forecast(np.ones(7), .1)
        result = state.finish(None, np.ones(7)*100, visual_available=False)
        np.testing.assert_array_equal(result['weights'], [1, 0, 0])
        np.testing.assert_allclose(result['q'], .2)

    def test_absent_visual_value_cannot_change_estimate_or_recurrent_memory(self):
        states = [self.state(), self.state()]
        for step in range(6):
            results = []
            for index, state in enumerate(states):
                state.forecast(np.ones(7)*step*.01, .06)
                results.append(state.finish(np.ones(7)*.03, np.ones(7)*(index+1)*100,
                                            visual_available=False))
            np.testing.assert_array_equal(results[0]['q'], results[1]['q'])
            np.testing.assert_array_equal(states[0].hidden.numpy(), states[1].hidden.numpy())

    def test_causal_prefix_independent_of_later_sensor_changes(self):
        prefixes = []
        for branch in range(2):
            state = self.state()
            rows = []
            for step in range(10):
                state.forecast(np.ones(7)*.03, .06)
                reading = np.ones(7)*(.02 if step < 6 else branch*.5)
                rows.append(state.finish(reading, np.ones(7)*.01, visual_available=True)['q'])
            prefixes.append(np.stack(rows))
        np.testing.assert_array_equal(prefixes[0][:6], prefixes[1][:6])
        self.assertFalse(np.array_equal(prefixes[0][6:], prefixes[1][6:]))

    def test_forecast_cannot_be_replaced_after_observation(self):
        state = self.state()
        with self.assertRaises(RuntimeError):
            state.finish()
        state.forecast(np.zeros(7), .1)
        with self.assertRaises(RuntimeError):
            state.forecast(np.ones(7), .1)
        state.finish()
        state.forecast(np.ones(7), .1)

    def test_available_source_weights_receive_training_gradients(self):
        torch.manual_seed(31)
        gate = ReliabilityGate()
        x = torch.zeros(3, 7)
        q, weights, _ = gate(x, torch.ones_like(x), -torch.ones_like(x), x, x, x,
            torch.ones(1, 7), torch.zeros(3, 5), torch.tensor([[True,True],[True,False],[False,False]]),
            torch.ones(3)*.1, torch.tensor([[1.,0,0]]).expand(3,-1), torch.zeros(3,32))
        (q-.8).square().mean().backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in gate.parameters()), 0.)
        np.testing.assert_array_equal(weights[2].detach().numpy(), [1,0,0])

    def test_training_teacher_never_enters_prediction_or_weights(self):
        torch.manual_seed(7)
        gate = ReliabilityGate()
        transition = ActionTransition(np.zeros(7), np.ones(7))
        body = type('TrainingBody', (), {'model': torch.nn.Linear(7, 6)})()
        a = dict(initial=torch.zeros(2,7), target=torch.randn(2,5,7)*.02,
                 duration=torch.ones(2,5)*.06, feedback=torch.randn(2,5,7)*.01,
                 visual=torch.randn(2,5,7)*.01, available=torch.ones(2,5,2),
                 quality=torch.ones(2,5,5)*.2, teacher=torch.zeros(2,5,7))
        _, first, weights = rollout_train(gate, transition, body, a, reliability_supervision=.05)
        a['teacher'] = torch.ones(2,5,7)*100
        _, second, other_weights = rollout_train(gate, transition, body, a, reliability_supervision=.05)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(weights, other_weights, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
