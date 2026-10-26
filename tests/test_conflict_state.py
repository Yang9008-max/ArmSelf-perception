import unittest
import numpy as np

from embodiment.conflict_state import (ConsistencyAudit, classify, interval_envelope,
                                       pair_distances, recovery_event)


class ConflictStateTests(unittest.TestCase):
    def test_three_incompatible_explanations_remain_unresolved(self):
        status, conflict = classify([3., 3., 3.], [True]*3, [1.]*3)
        self.assertEqual(status, 4)
        self.assertTrue(conflict)

    def test_two_sources_cannot_identify_which_one_is_wrong(self):
        status, conflict = classify([3., 0., 0.], [True, True, False], [1.]*3)
        self.assertEqual(status, 4)
        self.assertTrue(conflict)
        self.assertEqual(classify([100.]*3, [True, False, False], [1.]*3)[0], 5)

    def test_outlier_is_not_a_fault_oracle_when_two_bad_sources_agree(self):
        # Action=0, feedback=vision=2: the rule flags action, even if action
        # is actually correct. This explicit limitation must remain visible.
        q = np.stack((np.zeros(7), np.ones(7)*2, np.ones(7)*2))
        d = pair_distances(q, scale=np.ones(7))
        self.assertEqual(classify(d, [True]*3, [1.]*3)[0], 1)

    def test_a_single_transient_cannot_become_a_confirmed_outlier(self):
        audit = ConsistencyAudit([1.]*3)
        self.assertFalse(audit.update([0., 0., 0.], [True]*3)['conflict'])
        self.assertFalse(audit.update([0., 3., 3.], [True]*3)['conflict'])
        self.assertFalse(audit.update([0., 0., 0.], [True]*3)['conflict'])

    def test_conflict_envelope_preserves_all_available_hypotheses(self):
        q = np.zeros(7)
        candidates = np.stack((q, q+2, q-100))
        low, high = interval_envelope(q, np.ones(7)*.1, candidates, [True, True, False], True)
        np.testing.assert_allclose(low, -.1)
        np.testing.assert_allclose(high, 2.1)
        low0, high0 = interval_envelope(q, np.ones(7)*.1, candidates, [True]*3, False)
        np.testing.assert_allclose(low0, -.1)
        np.testing.assert_allclose(high0, .1)

    def test_recovery_requires_sustained_evidence_and_keeps_censored_events(self):
        distance = np.array([.001]*4+[.01]+[.001]*5)
        event = recovery_event(distance, distance, 0, 10)
        self.assertTrue(event['convergence']['recovered'])
        self.assertAlmostEqual(event['convergence']['time_s'], 1.)
        failed = recovery_event(distance, distance, 0, 9)
        self.assertFalse(failed['convergence']['recovered'])
        self.assertIsNone(failed['convergence']['time_s'])

    def test_future_observations_cannot_change_past_audit_outputs(self):
        traces = []
        for future in ([0., 0., 0.], [10., 10., 10.]):
            audit = ConsistencyAudit([1.]*3)
            values = [[0., 0., 0.], [0., 2., 2.], [0., 2., 2.], future]
            traces.append([audit.update(x, [True]*3) for x in values])
        self.assertEqual(traces[0][:3], traces[1][:3])


if __name__ == '__main__':
    unittest.main()
