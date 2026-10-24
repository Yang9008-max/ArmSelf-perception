import unittest
import numpy as np

from perception.evaluate import mask_metrics, _motion_scores, distractor_metrics


class EvaluationTests(unittest.TestCase):
    def test_distractor_rate_counts_only_visible_prop_pixels(self):
        prop = np.eye(4, dtype=bool)
        prediction = np.ones_like(prop)
        self.assertEqual(distractor_metrics(prediction, prop)['false_positive_rate'], 1.)
        self.assertIsNone(distractor_metrics(prediction, np.zeros_like(prop))['false_positive_rate'])
        prediction[:] = False
        prediction[0, 0] = True
        self.assertEqual(distractor_metrics(prediction, prop)['false_positive_rate'], .25)

    def test_empty_prediction_is_not_good_accuracy(self):
        truth = np.zeros((8, 8), dtype=bool)
        truth[3:5, 3:5] = True
        scores = mask_metrics(np.zeros_like(truth), truth)
        self.assertEqual(scores['iou'], 0)
        self.assertEqual(scores['recall'], 0)
        self.assertEqual(scores['precision'], 0)

    def test_perfect_and_excess_foreground(self):
        truth = np.eye(4, dtype=bool)
        self.assertEqual(mask_metrics(truth, truth)['iou'], 1)
        self.assertEqual(mask_metrics(np.ones_like(truth), truth)['iou'], .25)

    def test_motion_compares_same_correspondence_and_feedback(self):
        jacobian = np.zeros((7,2,4,4), np.float32)
        jacobian[:,0] = np.arange(1,8)[:,None,None]
        dq = np.repeat(np.eye(7,dtype=np.float32), 2, axis=0)
        dq[1::2] *= -1
        flow = np.einsum('kchw,nk->nhwc',jacobian,dq)
        result = _motion_scores(jacobian, {'flow':flow,'dq_scaled':dq,
                                         'valid':np.ones((14,4,4),bool)}, np.random.default_rng(4))
        self.assertAlmostEqual(result['model']['moving']['epe'],0)
        self.assertGreater(result['zero_motion']['moving']['epe'],0)
        self.assertGreater(result['shuffled_dq']['moving']['epe'],0)


if __name__ == '__main__':
    unittest.main()
