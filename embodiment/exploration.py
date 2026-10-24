"""Observation-driven action selection; explicit acquisition, no RL policy."""
from __future__ import annotations

import numpy as np

from .model import basis


class ExplorationBelief:
    """Online quadratic position belief; covariance is a regression proxy, not truth."""
    def __init__(self, point_count, center, scale):
        self.center = np.asarray(center, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        dimensions = basis(np.zeros((1, 7))).shape[1]
        self.coefficients = np.zeros((point_count, dimensions, 3))
        self.covariance = np.tile(np.eye(dimensions)/.05, (point_count, 1, 1))
        self.origin = np.tile([0., 0., .9], (point_count, 1))
        self.count = np.zeros(point_count, dtype=int)
        self.mean = np.zeros((point_count, 3))
        self.motion_sum = np.zeros((point_count, 3))
        self.error = np.full(point_count, .003)
        self.observations = 0

    def features(self, q):
        q = np.asarray(q)
        return basis((q-self.center)/self.scale)

    def predict(self, qs):
        features = self.features(qs)
        means = self.origin[None] + np.einsum('bd,ndk->bnk', features, self.coefficients)
        variance = np.einsum('bd,ndk,bk->bn', features, self.covariance, features)
        return means, np.maximum(variance, 0.)

    def update(self, measured_q, xyz, valid):
        xyz, valid = np.asarray(xyz), np.asarray(valid, dtype=bool).copy()
        if np.shape(measured_q) != (7,) or not np.isfinite(measured_q).all():
            raise ValueError('Seven finite actual measured joint positions required')
        valid &= np.isfinite(xyz).all(-1)
        self.observations += 1
        new = valid & (self.count == 0)
        self.origin[new] = xyz[new]
        x = self.features(np.asarray(measured_q)[None])[0]
        ids = np.flatnonzero(valid)
        if not len(ids):
            return dict(accepted=0, median_residual_m=None)
        prediction = self.origin[ids] + np.einsum('d,ndk->nk', x, self.coefficients[ids])
        residual = xyz[ids]-prediction
        magnitude = np.linalg.norm(residual, axis=-1)
        # A wrong depth match must not be rewarded merely for having large error.
        weight = np.minimum(1., .02/np.maximum(magnitude, 1e-8))
        projection = self.covariance[ids] @ x
        denominator = 1./weight + np.einsum('nd,d->n', projection, x)
        gain = projection/denominator[:, None]
        self.coefficients[ids] += gain[:, :, None]*residual[:, None, :]
        self.covariance[ids] -= gain[:, :, None]*projection[:, None, :]
        self.error[ids] = .9*self.error[ids] + .1*np.minimum(magnitude, .05)
        old_mean = self.mean[ids].copy()
        self.count[ids] += 1
        # Clip exceptional depth jumps in the motion statistic as well.
        delta = xyz[ids]-old_mean
        first = self.count[ids] == 1
        delta[~first] *= weight[~first, None]
        self.mean[ids] += delta/self.count[ids, None]
        self.motion_sum[ids] += delta*(xyz[ids]-self.mean[ids])
        return dict(accepted=len(ids), median_residual_m=float(np.median(magnitude)))

    def scores(self, qs):
        _, variance = self.predict(qs)
        motion = np.sqrt(np.maximum(self.motion_sum.sum(-1)/np.maximum(self.count, 1), 0))
        reliability = np.clip(motion/(self.error+.003), 0, 3)
        reliability *= self.count/np.maximum(self.observations, 1)
        reliability[self.count < 4] = 0
        if reliability.sum() < 1e-8:
            return np.zeros(len(qs))
        return np.log1p(variance) @ reliability/reliability.sum()

    def state(self):
        return {key: np.asarray(value).copy() for key, value in vars(self).items()}

    @classmethod
    def from_state(cls, state):
        result = cls(len(state['count']), state['center'], state['scale'])
        for key, value in state.items():
            setattr(result, key, int(value) if key == 'observations' else np.asarray(value).copy())
        return result


def choose_action(belief, current_q, rng, policy, step):
    """Candidate order depends on past observations; no future simulator queries."""
    current = (np.asarray(current_q)-belief.center)/belief.scale
    lo, hi = np.maximum(-1., current-.5), np.minimum(1., current+.5)
    if np.any(lo > hi):
        raise ValueError('Measured q outside the recoverable local command envelope')
    if step < 16 or policy == 'random':
        normalized = rng.uniform(lo, hi)
        source = 'shared_random_bootstrap' if step < 16 else 'random'
        score, rank, count = 0., 0, 1
    elif policy == 'preset':
        normalized = np.clip(.8*np.sin(step*.19*np.linspace(.7, 1.4, 7)+np.arange(7)*.8), lo, hi)
        source, score, rank, count = 'preset_sine', 0., 0, 1
    elif policy == 'active':
        proposals = np.concatenate([rng.uniform(lo, hi, (64, 7)),
                                    np.clip(current[None]+np.r_[np.eye(7)*.5, -np.eye(7)*.5], lo, hi)])
        candidates = belief.center + proposals*belief.scale
        scores = belief.scores(candidates)
        chosen = int(np.argmax(scores))
        normalized, score = proposals[chosen], float(scores[chosen])
        source, rank, count = 'learned_belief_information', chosen, len(proposals)
    else:
        raise ValueError(f'Unknown policy: {policy}')
    return (belief.center+normalized*belief.scale).astype(np.float32), dict(
        source=source, score=score, candidate_index=rank, candidates=count,
        observed_before_choice=int(belief.observations))
