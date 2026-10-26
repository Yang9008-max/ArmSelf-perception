import unittest
import numpy as np
import torch
from embodiment.branch_state import BranchBodyState
from embodiment.reliable_state import JointReliability
from tests.test_hypothesis_state import Transition,Body,Fitter


class BranchTests(unittest.TestCase):
    def state(self,override=True):
        torch.manual_seed(12)
        return BranchBodyState(Transition(),JointReliability(bias_enabled=False).eval(),Body(),Fitter(),
            np.zeros(7,np.float32),dict(association_m=.08,association_q=.12,surface_m=.08),
            confirm=2,allow_override=override)

    def step(self,state,target=1.,feedback=0.,surface=None,camera=True):
        prior=state.forecast(np.full(7,target,np.float32),.1)
        f=None if feedback is None else np.full(7,feedback,np.float32)
        xyz=np.full((12,7),feedback if surface is None and feedback is not None else (surface or 0.),np.float32)
        result=state.finish(f,np.zeros(7,np.float32),np.ones(5,np.float32),camera,xyz,np.full(12,camera,bool))
        return prior,result

    def test_healthy_observation_overrides_wrong_command_while_command_continues(self):
        state=self.state()
        outputs=[self.step(state)[1] for _ in range(6)]
        self.assertTrue(any(r['override_action'] for r in outputs))
        np.testing.assert_array_equal(outputs[-1]['q'],np.zeros(7))
        self.assertFalse(outputs[-1]['unknown'])
        self.assertGreater(outputs[-1]['open_action'].mean(),.9)

    def test_correlated_sensor_jump_cannot_invent_confirming_motion_branch(self):
        state=self.state()
        outputs=[self.step(state,feedback=-1.)[1] for _ in range(8)]
        self.assertTrue(all(r['unknown'] for r in outputs))
        self.assertFalse(any(r['sensor_selected'] for r in outputs))
        np.testing.assert_array_equal(outputs[-1]['bank'][3],np.zeros(7))

    def test_new_evidence_can_select_stop_history_without_action_agreement(self):
        state=self.state()
        for _ in range(8):
            self.step(state,feedback=-1.)
        outputs=[self.step(state,feedback=0.)[1] for _ in range(3)]
        self.assertTrue(outputs[-1]['override_action'])
        np.testing.assert_array_equal(outputs[-1]['q'],np.zeros(7))

    def test_observation_commit_changes_next_forecast_but_not_competing_action(self):
        state=self.state()
        for _ in range(3):
            self.step(state)
        prior,result=self.step(state)
        np.testing.assert_array_equal(prior,np.zeros(7))
        self.assertGreater(result['open_action'].mean(),.9)

    def test_missing_camera_does_not_confirm_a_new_observation_override(self):
        state=self.state()
        outputs=[self.step(state,camera=False)[1] for _ in range(6)]
        self.assertFalse(any(r['sensor_selected'] for r in outputs))
        self.assertTrue(outputs[-1]['unknown'])

    def test_already_established_hold_survives_camera_outage(self):
        state=self.state()
        for _ in range(3):
            self.step(state)
        _,result=self.step(state,camera=False)
        np.testing.assert_array_equal(result['q'],np.zeros(7))
        self.assertEqual(result['support'],0)

    def test_override_ablation_affects_actual_state(self):
        a,b=self.state(),self.state(False)
        for _ in range(5):
            _,x=self.step(a);_,y=self.step(b)
        self.assertGreater(np.max(np.abs(x['q']-y['q'])),.01)
        self.assertFalse(y['override_action'])

    def test_missing_feedback_does_not_create_a_feedback_candidate(self):
        _,result=self.step(self.state(),feedback=None)
        self.assertTrue(result['unknown'])
        self.assertFalse(result['bank_valid'][4])

    def test_future_evidence_does_not_change_prior_outputs(self):
        traces=[]
        for future in (-1.,1.):
            state=self.state();values=[self.step(state)[1] for _ in range(4)]
            self.step(state,feedback=future);traces.append(values)
        for x,y in zip(*traces):
            for key in ('q','bank','open_action','branch_innovation'):
                np.testing.assert_array_equal(x[key],y[key])

    def test_velocity_after_gap_uses_elapsed_observation_time(self):
        state=self.state()
        self.step(state,target=0.,feedback=0.)
        for _ in range(4):
            self.step(state,target=0.,feedback=None,camera=False)
        self.step(state,target=0.,feedback=.02)
        np.testing.assert_allclose(state.observed_v,np.full(7,.04),atol=1e-6)

    def test_state_api_rejects_out_of_order_evidence(self):
        state=self.state()
        with self.assertRaises(RuntimeError):
            state.finish(None,np.zeros(7),np.ones(5),False,np.zeros((12,7)),np.zeros(12,bool))
        state.forecast(np.zeros(7),.1)
        with self.assertRaises(RuntimeError):
            state.forecast(np.zeros(7),.1)

    def test_forecast_rejects_invalid_duration(self):
        for dt in (0.,-1.,float('nan')):
            with self.assertRaises(ValueError):
                self.state().forecast(np.zeros(7),dt)


if __name__=='__main__':
    unittest.main()
