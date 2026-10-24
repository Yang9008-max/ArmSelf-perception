import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from embodiment.collect import evaluation_protocol
from embodiment.experiment import evaluate
from embodiment.model import SpatialEntity


class EntityEvaluationTests(unittest.TestCase):
    def test_fresh_protocol_is_evaluation_only_and_blind_phase_moves(self):
        rows = evaluation_protocol(991)
        self.assertEqual(len(rows), 162)
        self.assertEqual(set(r[0] for r in rows), {'test', 'hold', 'blind'})
        blind = [r for r in rows if r[0] == 'blind']
        self.assertTrue(all(not r[2] for r in blind))
        self.assertGreater(float(np.ptp(np.stack([r[1] for r in blind]), axis=0).max()), .02)

    def test_legacy_model_cannot_redetect_identity_in_unrelated_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, output = root/'data', root/'model'
            data.mkdir()
            output.mkdir()
            calibration = dict(focal_px=360, baseline_m=.06, cx=319.5, cy=239.5)
            (data/'metadata.json').write_text(json.dumps(dict(calibration=calibration, seed=5)))
            model = SpatialEntity(4, np.zeros(7), np.ones(7), np.ones((4, 3)))
            torch.save(dict(point_count=4, selected=[0, 1, 2, 3], calibration=calibration,
                            center=[0]*7, scale=[1]*7, origin=[[1]*3]*4,
                            state_dict=model.state_dict()), output/'model.pt')
            (output/'freeze.json').write_text(json.dumps(dict(
                checkpoint_sha256=hashlib.sha256((output/'model.pt').read_bytes()).hexdigest(),
                data_metadata_sha256='different-training-reference')))
            with self.assertRaisesRegex(ValueError, 'original reference-data'):
                evaluate(data, output)
            self.assertFalse((output/'metrics.json').exists())
