import unittest

import numpy as np
import torch

from embodiment.action_state import ActionTransition
from embodiment.reliable_state import JointReliability, ReliableBodyState
from embodiment.reliable_cycle import rollout


class Body:
    saved = {'scale':np.ones(7,np.float32)}

    def __call__(self,q):
        return np.atleast_2d(q)[:,:6].reshape(-1,2,3)


class Transition:
    def predict(self,q,velocity,target,duration):
        return np.asarray(q)+.2*(np.asarray(target)-q)


class ReliableStateTests(unittest.TestCase):
    def state(self,bias=True):
        torch.manual_seed(28)
        return ReliableBodyState(Transition(),JointReliability(bias_enabled=bias).eval(),Body(),np.zeros(7))

    def test_missing_sources_have_zero_weight_for_every_joint(self):
        state = self.state()
        state.forecast(np.ones(7),.1)
        a = state.finish()
        np.testing.assert_array_equal(a['weights'],np.tile([1,0,0],(7,1)))
        np.testing.assert_allclose(a['q'],.2)

    def test_absent_visual_values_and_quality_cannot_pollute_memory(self):
        states = [self.state(),self.state()]
        for i in range(6):
            outputs = []
            for branch,state in enumerate(states):
                state.forecast(np.ones(7)*i*.03,.06)
                outputs.append(state.finish(np.ones(7)*.02,np.ones(7)*branch*100,
                                            np.ones(5)*branch*100,False))
            np.testing.assert_array_equal(outputs[0]['q'],outputs[1]['q'])
            np.testing.assert_array_equal(states[0].hidden.numpy(),states[1].hidden.numpy())

    def test_future_changes_leave_past_prefix_identical(self):
        trajectories = []
        for branch in range(2):
            state,rows = self.state(),[]
            for i in range(10):
                state.forecast(np.ones(7)*.02,.06)
                rows.append(state.finish(np.ones(7)*(.03 if i<6 else branch*.4))['q'])
            trajectories.append(np.stack(rows))
        np.testing.assert_array_equal(trajectories[0][:6],trajectories[1][:6])
        self.assertFalse(np.array_equal(trajectories[0][6:],trajectories[1][6:]))

    def test_order_and_invalid_duration_are_rejected(self):
        state = self.state()
        with self.assertRaises(RuntimeError):
            state.finish()
        with self.assertRaises(ValueError):
            state.forecast(np.zeros(7),0)
        state.forecast(np.zeros(7),.06)
        with self.assertRaises(RuntimeError):
            state.forecast(np.zeros(7),.06)

    def test_disabled_bias_is_exactly_zero_and_sigma_positive(self):
        state = self.state(False)
        for i in range(5):
            state.forecast(np.ones(7)*.03,.06)
            a = state.finish(np.ones(7)*.2,np.zeros(7),visual_available=True)
            np.testing.assert_array_equal(a['bias_normalized'],np.zeros(7))
            self.assertTrue(np.isfinite(a['sigma_normalized']).all())
            self.assertTrue((a['sigma_normalized']>0).all())

    def test_teacher_changes_loss_but_not_outputs(self):
        torch.manual_seed(28)
        gate = JointReliability()
        transition = ActionTransition(np.zeros(7),np.ones(7))
        body = type('TrainingBody',(),{'model':torch.nn.Linear(7,6)})()
        a = dict(initial=torch.zeros(2,7),target=torch.randn(2,5,7)*.02,
                 duration=torch.ones(2,5)*.06,feedback=torch.randn(2,5,7)*.01,
                 visual=torch.randn(2,5,7)*.01,available=torch.ones(2,5,2),
                 quality=torch.ones(2,5,5)*.2,teacher=torch.zeros(2,5,7))
        first = rollout(gate,transition,body,a)
        a['teacher'] = torch.ones(2,5,7)*.5
        second = rollout(gate,transition,body,a)
        for index in range(2,len(first)):
            torch.testing.assert_close(first[index],second[index],rtol=0,atol=0)
        self.assertNotEqual(float(first[0].detach()),float(second[0].detach()))

    def test_training_reaches_weight_bias_and_uncertainty_parameters(self):
        torch.manual_seed(28)
        gate = JointReliability()
        transition = ActionTransition(np.zeros(7),np.ones(7))
        body = type('TrainingBody',(),{'model':torch.nn.Linear(7,6)})()
        a = dict(initial=torch.zeros(2,7),target=torch.ones(2,4,7)*.03,
                 duration=torch.ones(2,4)*.06,feedback=torch.ones(2,4,7)*.1,
                 visual=torch.zeros(2,4,7),available=torch.ones(2,4,2),
                 quality=torch.ones(2,4,5)*.2,teacher=torch.zeros(2,4,7))
        # Output heads start at a neutral correction. A first optimizer step
        # opens their gradient path into the recurrent representation.
        optimizer = torch.optim.Adam(gate.parameters(),lr=.002)
        rollout(gate,transition,body,a)[0].backward()
        optimizer.step()
        optimizer.zero_grad()
        rollout(gate,transition,body,a)[0].backward()
        for module in (gate.cell,gate.weight_head,gate.bias_head,gate.sigma_head):
            gradient = sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)
            self.assertGreater(gradient,0)


if __name__=='__main__':
    unittest.main()
