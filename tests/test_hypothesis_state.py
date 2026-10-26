import unittest
import numpy as np
import torch

from embodiment.hypothesis_state import AtlasMeasurements, AtlasFitter, CompetingBodyState
from embodiment.reliable_state import JointReliability


class Transition:
    def predict(self,q,velocity,target,duration):
        return (np.asarray(q)+.5*(np.asarray(target)-q)).astype(np.float32)


class Body:
    saved={'scale':np.ones(7,np.float32)}
    def __call__(self,q):
        return np.atleast_2d(q)[:,:6].reshape(-1,2,3)


class Fitter:
    scale=np.ones(7,np.float32)
    def distance(self,a,b):
        return float(np.sqrt(np.mean((np.asarray(a)-b)**2)))
    def residual(self,q,xyz,valid):
        return float(np.linalg.norm(np.asarray(q)-np.asarray(xyz)[0])) if np.asarray(valid).sum()>=12 else float('inf')
    def fit(self,q,xyz,valid):
        return np.asarray(q,np.float32).copy(),False


class HypothesisTests(unittest.TestCase):
    def state(self,update=True):
        torch.manual_seed(31)
        return CompetingBodyState(Transition(),JointReliability(bias_enabled=False).eval(),Body(),Fitter(),
            np.zeros(7,np.float32),dict(action_feedback=.08,surface=.08,dynamics=.08),gain=0,confirm=3,update_state=update)

    def step(self,state,target,feedback):
        state.forecast(np.ones(7,np.float32)*target,.1)
        observed=np.zeros(7,np.float32) if feedback is None else np.ones(7,np.float32)*feedback
        return state.finish(None if feedback is None else observed,observed,np.ones(5,np.float32),
            feedback is not None,np.tile(observed,(12,1)),np.ones(12,bool))

    def test_correlated_observations_cannot_delete_protected_explanation(self):
        state=self.state()
        result=self.step(state,1.,0.)
        self.assertTrue(result['unknown'])
        self.assertTrue(result['bank_valid'][0])
        self.assertTrue(result['bank_valid'][1])
        self.assertGreater(result['bank'][0].mean(),result['bank'][1].mean())

    def test_missing_feedback_preserves_alternatives_and_marks_unknown(self):
        result=self.step(self.state(),1.,None)
        self.assertTrue(result['unknown'])
        self.assertFalse(result['bank_valid'][1])
        self.assertTrue(result['bank_valid'][0])

    def test_new_consistent_evidence_commits_and_resets_recurrent_memory(self):
        state=self.state()
        self.step(state,0.,1.)
        # Force the learned update to preserve its contaminated action state;
        # fresh evidence must then change the actual state at explicit commit.
        with torch.no_grad():
            state.base.gate.weight_head[-1].bias.copy_(torch.tensor([6.,-6.,-6.]).repeat(7))
        outputs=[self.step(state,0.,0.) for _ in range(5)]
        commits=[x for x in outputs if x['committed']]
        self.assertTrue(commits)
        self.assertFalse(outputs[-1]['unknown'])
        self.assertTrue(any(np.max(np.abs(x['q']-x['proposal']))>1e-4 for x in commits))
        np.testing.assert_array_equal(state.base.q,outputs[-1]['q'])

    def test_readout_ablation_cannot_commit_or_apply_corrections(self):
        state=self.state(False)
        self.step(state,0.,1.)
        outputs=[self.step(state,0.,0.) for _ in range(5)]
        self.assertFalse(any(x['committed'] for x in outputs))
        for x in outputs:
            np.testing.assert_array_equal(x['q'],x['proposal'])

    def test_surface_evidence_changes_recovery_with_same_actions_and_feedback(self):
        traces=[]
        for surface in (0.,1.):
            state=self.state()
            self.step(state,0.,1.)
            values=[]
            for _ in range(5):
                state.forecast(np.zeros(7,np.float32),.1)
                values.append(state.finish(np.zeros(7,np.float32),np.zeros(7,np.float32),
                    np.ones(5,np.float32),True,np.full((12,7),surface,np.float32),np.ones(12,bool)))
            traces.append(values)
        self.assertTrue(any(x['committed'] for x in traces[0]))
        self.assertFalse(any(x['committed'] for x in traces[1]))
        self.assertTrue(traces[1][-1]['unknown'])

    def test_stopped_hypothesis_survives_prolonged_unresolved_motion(self):
        state=self.state()
        for _ in range(8):
            x=self.step(state,1.,0.)
        self.assertTrue(x['unknown'])
        np.testing.assert_array_equal(x['bank'][3],np.zeros(7))

    def test_future_evidence_does_not_change_prefix(self):
        traces=[]
        for future in (0.,2.):
            state=self.state()
            values=[self.step(state,.2,0.) for _ in range(4)]
            values.append(self.step(state,.2,future))
            traces.append(values)
        for a,b in zip(traces[0][:4],traces[1][:4]):
            np.testing.assert_array_equal(a['q'],b['q'])
            np.testing.assert_array_equal(a['bank'],b['bank'])

    def test_absent_camera_does_not_invent_surface_measurements(self):
        observer=AtlasMeasurements(None,np.arange(82))
        result=observer.observe(None,None,[np.zeros(7)])
        self.assertFalse(result['valid'].any())
        self.assertFalse(result['origin'].any())

    def test_observational_twins_share_exact_prefix_but_keep_future_evidence(self):
        from embodiment.hypothesis_cycle import share_prefix
        moving={'visual':np.ones((8,7),np.float32),'available':np.ones((8,2),bool)}
        held={'visual':np.ones((8,7),np.float32)+1e-6,'available':np.zeros((8,2),bool)}
        future=held['visual'][4:].copy()
        share_prefix(moving,held,4)
        for key in moving:
            np.testing.assert_array_equal(moving[key][:4],held[key][:4])
        np.testing.assert_array_equal(held['visual'][4:],future)
        self.assertFalse(held['available'][4:].any())


if __name__=='__main__':
    unittest.main()
