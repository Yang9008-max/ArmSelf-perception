import unittest
import numpy as np
import torch
from embodiment.evidence_state import EvidenceBodyState,branch_envelope,MotionMeasurements
from embodiment.reliable_state import JointReliability
from tests.test_hypothesis_state import Transition,Body


class Surface:
    scale=np.ones(7,np.float32)
    def values(self,q):return np.tile(np.asarray(q)[:3],(18,1))
    def distance(self,a,b):return float(np.linalg.norm(np.asarray(a)[:3]-np.asarray(b)[:3]))
    def residual(self,q,xyz,valid):
        return float(np.median(np.linalg.norm(self.values(q)[valid]-xyz[valid],axis=-1))) if np.sum(valid)>=12 else float('inf')
    def fit(self,q,xyz,valid):return q.copy(),False


class EvidenceTests(unittest.TestCase):
    def state(self,enabled=True):
        torch.manual_seed(13)
        return EvidenceBodyState(Transition(),JointReliability(bias_enabled=False).eval(),Body(),Surface(),
            np.zeros(7,np.float32),dict(association_m=.08,association_q=.12,surface_m=.006),
            confirm=2,visual_confirm=3,visual_enabled=enabled)

    def step(self,state,feedback=None,camera=True,surface=0.):
        prior=state.forecast(np.ones(7,np.float32),.1)
        f=None if feedback is None else np.full(7,feedback,np.float32)
        return prior,state.finish(f,np.zeros(7,np.float32),np.ones(5,np.float32),camera,
            np.full((18,3),surface,np.float32),np.full(18,camera,bool))

    def test_visual_can_correct_actual_hold_without_feedback(self):
        state=self.state()
        outputs=[self.step(state)[1] for _ in range(5)]
        self.assertTrue(outputs[-1]['visual_selected'])
        np.testing.assert_array_equal(outputs[-1]['q'],np.zeros(7))
        self.assertGreater(outputs[-1]['open_action'].mean(),.9)

    def test_visual_selection_requires_consecutive_support(self):
        state=self.state()
        a=self.step(state)[1];b=self.step(state)[1]
        self.assertFalse(a['visual_selected']);self.assertFalse(b['visual_selected'])
        self.step(state,camera=False)
        self.assertFalse(self.step(state)[1]['visual_selected'])

    def test_visual_alone_does_not_remove_source_ambiguity(self):
        state=self.state()
        for _ in range(5):_,r=self.step(state)
        self.assertTrue(r['unknown']);self.assertGreaterEqual(r['bank_valid'].sum(),4)
        self.assertFalse(r['bank_valid'][4])

    def test_visual_correction_changes_future_while_open_action_survives(self):
        state=self.state()
        for _ in range(5):self.step(state)
        prior,r=self.step(state)
        np.testing.assert_array_equal(prior,np.zeros(7))
        self.assertGreater(r['open_action'].mean(),.9)

    def test_disabled_visual_has_real_counterfactual_difference(self):
        yes,no=self.state(),self.state(False)
        for _ in range(5):
            _,a=self.step(yes);_,b=self.step(no)
        self.assertGreater(np.linalg.norm(a['q']-b['q']),1.)
        self.assertFalse(b['visual_selected'])

    def test_missing_camera_cannot_create_visual_commit(self):
        state=self.state()
        for _ in range(6):
            _,r=self.step(state,camera=False)
            self.assertFalse(r['visual_selected']);self.assertEqual(r['visual_history'],-1)

    def test_propagated_visual_history_is_not_fresh_observation(self):
        state=self.state()
        for _ in range(5):self.step(state)
        for _ in range(4):
            _,r=self.step(state,camera=False)
            self.assertFalse(r['visual_selected'])
        self.assertAlmostEqual(state.observed_age,.4)
        np.testing.assert_array_equal(r['q'],np.zeros(7))

    def test_one_spatial_subset_cannot_validate_visual_explanation(self):
        state=self.state();branches=np.array([[1.]*7,[1.]*7,[0.]*7,[0.]*7])
        valid=np.arange(18)%2==0
        candidate,_=state.visual_choice(branches,None,np.zeros((18,3)),valid)
        self.assertEqual(candidate,-1)

    def test_visual_outage_keeps_stop_velocity_decay_instead_of_restarting_it(self):
        state=self.state()
        for _ in range(5):self.step(state)
        state.hold_v[:]=.04;state.observed_v[:]=.04
        self.step(state,camera=False)
        np.testing.assert_allclose(state.hold_v[1],.02)
        self.step(state,camera=False)
        np.testing.assert_allclose(state.hold_v[1],.01)

    def test_better_coherent_sensor_pair_rejects_nearby_physical_candidate(self):
        state=self.state();branches=np.array([[1.]*7,[1.]*7,[.002]*7,[.002]*7])
        xyz=np.zeros((18,3));valid=np.ones(18,bool)
        self.assertGreaterEqual(state.visual_choice(branches,None,xyz,valid)[0],0)
        self.assertEqual(state.visual_choice(branches,np.zeros(7),xyz,valid)[0],-1)

    def test_healthy_sensor_can_clear_previous_visual_uncertainty(self):
        state=self.state()
        for _ in range(5):self.step(state)
        for _ in range(3):_,r=self.step(state,feedback=0.)
        self.assertFalse(r['unknown']);self.assertEqual(r['visual_history'],-1)

    def test_branch_radius_belongs_only_to_active_candidates(self):
        bank=np.zeros((2,6,7));active=np.zeros((2,6),bool);active[0,-1]=True;active[1,0]=True
        radii=np.full((6,7),.02);radii[0]=.2
        lo,hi=branch_envelope(bank,active,np.ones(7),radii)
        np.testing.assert_allclose(hi[0],.02);np.testing.assert_allclose(hi[1],.2)
        np.testing.assert_allclose(lo,-hi)

    def test_missing_images_preserve_both_observer_budgets_as_empty(self):
        original,augmented=MotionMeasurements(None,np.arange(18)).observe(None,None,[None]*4)
        self.assertFalse(original['valid'].any());self.assertFalse(augmented['valid'].any())
        self.assertTrue((augmented['origin']==0).all())

if __name__=='__main__':unittest.main()
