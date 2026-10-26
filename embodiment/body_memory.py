"""RGB/feedback body observations and persistent sparse state; no simulator API.

All modes preserve the same feature IDs. Missing or rejected measurements never
remove a point from the predicted representation or supply a training target.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
import torch

from perception.live import LiveBodyPredictor
from perception.boundary import prediction_mask, GRABCUT_CONFIG
from .vision import StereoTracks, project
from .model import SpatialEntity
from .exploration import ExplorationBelief


def sample_mask(mask, uv):
    uv = np.asarray(uv)
    finite = np.isfinite(uv).all(-1)
    pixels = np.rint(np.where(finite[:, None], uv, 0)).astype(int)
    h, w = mask.shape
    inside = finite & (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
    result = np.zeros(len(uv), dtype=bool)
    result[inside] = mask[pixels[inside, 1], pixels[inside, 0]]
    return result


class BodyMasks:
    """One frozen network inference produces the V6 and V7 mask ablations."""
    def __init__(self, checkpoint):
        self.predictor = LiveBodyPredictor(checkpoint, threads=4)

    def __call__(self, rgb, q):
        result = self.predictor.predict(rgb, q)
        size = self.predictor.input_size
        small = np.asarray(Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR))
        probability = result['probability']
        def resize(mask):
            return np.asarray(Image.fromarray(mask.astype(np.uint8)).resize(
                (rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST)).astype(bool)
        return {'none': np.ones(rgb.shape[:2], bool), 'v6': resize(probability >= .5),
                'v7': resize(prediction_mask(small, probability, GRABCUT_CONFIG))}


class BodyModel:
    def __init__(self, checkpoint):
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.selected = np.asarray(saved['selected'], dtype=int)
        self.model = SpatialEntity(len(self.selected), saved['center'], saved['scale'], saved['origin'])
        self.model.load_state_dict(saved['state_dict'])
        self.model.eval()
        self.saved = saved

    def __call__(self, q):
        values = np.asarray(q, dtype=np.float32)
        if values.ndim == 1:
            values = values[None]
        with torch.inference_mode():
            return self.model(torch.from_numpy(values)).numpy()


class BodyObserver:
    def __init__(self, reference, calibration, selected):
        self.tracker = StereoTracks(reference, calibration, 'refined11')
        self.selected = np.asarray(selected)
        self.calibration = calibration

    def observe(self, left, right, mask, prediction=None):
        if prediction is not None:
            uv = project(np.asarray(prediction), self.calibration).astype(np.float32)
            usable = np.isfinite(uv).all(-1) & (np.asarray(prediction)[:, 2] > .15)
            self.tracker.previous_points[self.selected[usable], 0] = uv[usable]
        tracks = self.tracker.observe(left, right)
        result = {k: v[self.selected].copy() for k, v in tracks.items()}
        result['raw_valid'] = result['valid'].copy()
        result['valid'] &= sample_mask(mask, result['uv'])
        return result


class BodyMemory:
    """Model motion prior plus gated RGB corrections; covariance is a heuristic.

    Reference patches carry correspondence identity. This explicit association
    procedure is not a learned object-permanence or learned re-ID network.
    """
    gate_m = .04
    measurement_variance = .008**2

    def __init__(self, count):
        self.correction = np.zeros((count, 3), np.float32)
        self.variance = np.full(count, .012**2)
        self.last_time = None
        self.last_q = None
        self.last_seen = np.full(count, np.nan)

    def update(self, prediction, q, time, xyz=None, valid=None):
        prediction, q = np.asarray(prediction), np.asarray(q)
        dt = 0. if self.last_time is None else max(0., time-self.last_time)
        motion = 0. if self.last_q is None else np.linalg.norm(q-self.last_q)
        self.variance += (.002**2)*dt + (.004**2)*motion
        self.correction *= np.exp(-motion)
        accepted = np.zeros(len(prediction), bool)
        if xyz is not None and valid is not None:
            xyz = np.asarray(xyz)
            accepted = np.asarray(valid, bool) & np.isfinite(xyz).all(-1)
            accepted &= np.linalg.norm(xyz-prediction, axis=1) <= self.gate_m
            gain = self.variance[accepted]/(self.variance[accepted]+self.measurement_variance)
            residual = xyz[accepted]-prediction[accepted]-self.correction[accepted]
            self.correction[accepted] += gain[:, None]*residual
            self.variance[accepted] *= 1-gain
            self.last_seen[accepted] = time
        self.last_time, self.last_q = time, q.copy()
        return {'prediction': prediction+self.correction, 'accepted': accepted,
                'variance': self.variance.copy(), 'visual_evidence': accepted.copy(),
                'last_seen': self.last_seen.copy()}


class ResidualBelief:
    """Online residual learning around the exact same frozen body decoder."""
    def __init__(self, body, center, scale):
        self.body = body
        self.online = ExplorationBelief(len(body.selected), center, scale)
        self.online.origin[:] = 0
        self.center, self.scale = self.online.center, self.online.scale

    @property
    def observations(self):
        return self.online.observations

    def predict(self, qs):
        residual, variance = self.online.predict(qs)
        return self.body(qs)+residual, variance

    def update(self, q, xyz, valid):
        prior = self.body(q)[0]
        valid = np.asarray(valid, bool) & (np.linalg.norm(xyz-prior, axis=-1) <= BodyMemory.gate_m)
        # ExplorationBelief initializes each first origin to its first observation.
        result = self.online.update(q, np.asarray(xyz)-prior, valid)
        return result, valid

    def scores(self, qs):
        _, variance = self.online.predict(qs)
        weight = self.online.count/max(self.online.observations, 1)
        weight *= .008/(self.online.error+.003)
        weight[self.online.count < 4] = 0
        return np.log1p(variance) @ weight/max(weight.sum(), 1e-8)

    def state(self):
        return self.online.state()

    @classmethod
    def from_state(cls, body, state):
        result = cls(body, state['center'], state['scale'])
        result.online = ExplorationBelief.from_state(state)
        return result
