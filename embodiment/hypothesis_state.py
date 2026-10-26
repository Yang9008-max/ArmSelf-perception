"""E11 explicit set-valued body observer around frozen learned models.

RGB fits inherit their initialization provenance. Repeated matches of one
patch are one observation, never independent votes. Unresolved hypotheses
are retained, not assigned calibrated fault probabilities.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

from .reliable_state import ReliableBodyState
from .surface_extent import track_patch
from .vision import project


class AtlasMeasurements:
    def __init__(self, atlas, indices):
        self.atlas = atlas
        self.indices = np.asarray(indices, int)

    def observe(self, left, right, seeds):
        n = len(self.indices)
        xyz = np.zeros((n, 3), np.float32)
        best = np.full(n, np.inf)
        origin = np.zeros(n, np.uint8)
        if left is None or right is None:
            return dict(xyz=xyz, valid=np.zeros(n, bool), origin=origin)
        gray = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
        other = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
        saved = self.atlas.saved
        for seed_index, q in enumerate(seeds):
            if q is None or not np.isfinite(q).all():
                continue
            predicted = self.atlas.predict(q)[self.indices]
            initial = project(predicted, self.atlas.calibration)
            for chart in np.unique(saved['chart'][self.indices]):
                positions = np.flatnonzero(saved['chart'][self.indices] == chart)
                ids = self.indices[positions]
                measured, good, uv = track_patch(saved['gray'][chart], gray, other,
                    saved['uv'][ids], predicted[positions], self.atlas.calibration, initial[positions])
                good &= np.linalg.norm(measured-predicted[positions], axis=-1) < .05
                for j in np.flatnonzero(good):
                    # Rank using actual RGB appearance, not hidden geometry or
                    # agreement with a favored hypothesis. One winner per patch.
                    old = cv2.getRectSubPix(saved['gray'][chart], (7, 7), tuple(map(float, saved['uv'][ids[j]])))
                    new = cv2.getRectSubPix(gray, (7, 7), tuple(map(float, uv[j])))
                    cost = float(np.abs(old.astype(float)-new).mean())
                    k = positions[j]
                    if cost < best[k]:
                        best[k], xyz[k], origin[k] = cost, measured[j], 1 << seed_index
        return dict(xyz=xyz, valid=np.isfinite(best), origin=origin)


class AtlasFitter:
    def __init__(self, atlas, indices):
        self.atlas, self.indices = atlas, np.asarray(indices, int)
        self.scale = np.asarray(atlas.scale, np.float32)

    def values(self, q):
        return self.atlas.predict(np.asarray(q, np.float32))[self.indices]

    def distance(self, a, b):
        return float(np.sqrt(np.mean(np.sum((self.values(a)-self.values(b))**2, -1))))

    def residual(self, q, xyz, valid):
        good = np.asarray(valid, bool) & np.isfinite(xyz).all(-1)
        if good.sum() < 12:
            return float('inf')
        # Median reduces a few wrong correspondences; no body labels prune IDs.
        return float(np.median(np.linalg.norm(self.values(q)[good]-xyz[good], axis=-1)))

    def fit(self, q, xyz, valid):
        start = np.asarray(q, np.float32)
        q = start.copy()
        good = np.asarray(valid, bool) & np.isfinite(xyz).all(-1)
        if good.sum() < 12:
            return q, False
        for _ in range(3):
            predicted = self.values(q)
            residual = np.asarray(xyz)-predicted
            norm = np.linalg.norm(residual, axis=-1)
            keep = good & (norm < .05)
            if keep.sum() < 12:
                return start.copy(), False
            eps = .005
            jac = np.stack([(self.values(q+np.eye(7, dtype=np.float32)[j]*self.scale*eps)-
                             self.values(q-np.eye(7, dtype=np.float32)[j]*self.scale*eps))/(2*eps)
                            for j in range(7)], -1)[keep].reshape(-1, 7)
            weights = np.repeat(np.minimum(1., .008/np.maximum(norm[keep], 1e-8)), 3)
            delta = np.linalg.solve(jac.T@(weights[:, None]*jac)+.001*np.eye(7),
                                    jac.T@(weights*residual[keep].ravel()))
            q = start+np.clip((q-start)/self.scale+np.clip(delta, -.04, .04), -.12, .12)*self.scale
        before, after = self.residual(start, xyz, good), self.residual(q, xyz, good)
        return q.astype(np.float32), bool(after < before*.98 and self.distance(start, q) < .006)


class CompetingBodyState:
    """A protected action path, observation paths and unresolved body memory.

There is no majority vote or multiplication of prior-derived RGB evidence.
The E8 point estimate is provisional while hypotheses disagree. A commit
requires consecutive cross-time consistency; explicit RGB geometry can then
change both the state and subsequent action forecasts.
"""
    def __init__(self, transition, gate, body, fitter, initial, thresholds,
                 gain=.2, confirm=3, update_state=True):
        self.base = ReliableBodyState(transition, gate, body, initial)
        self.transition, self.fitter = transition, fitter
        self.scale = np.asarray(body.saved['scale'], np.float32)
        self.thresholds = thresholds
        self.gain, self.confirm, self.update_state = gain, confirm, update_state
        self.q = np.asarray(initial, np.float32).copy()
        self.protected, self.protected_velocity = self.q.copy(), np.zeros(7, np.float32)
        self.stationary, self.last_safe = self.q.copy(), self.q.copy()
        self.old_feedback, self.feedback_velocity = None, np.zeros(7, np.float32)
        self.unknown, self.stable = False, 0
        self.pending = None

    def forecast(self, target, duration):
        self.base.forecast(target, duration)
        protected = self.transition.predict(self.protected, self.protected_velocity, target, duration)
        feedback_prior = None if self.old_feedback is None else self.transition.predict(
            self.old_feedback, self.feedback_velocity, target, duration)
        self.pending = dict(protected=protected, feedback_prior=feedback_prior,
                            target=np.asarray(target).copy(), dt=float(duration), previous=self.q.copy())
        return protected.copy()

    def finish(self, feedback, visual, quality, visual_available, xyz, valid):
        if self.pending is None:
            raise RuntimeError('Forecast must precede evidence')
        p = self.pending
        f_ok = feedback is not None and np.isfinite(feedback).all()
        f = np.asarray(feedback, np.float32) if f_ok else p['protected']
        proposal = self.base.finish(feedback, visual, quality, visual_available)
        q = proposal['q'].copy()
        fit_q, fit_ok = self.fitter.fit(q, xyz, valid)
        raw_residual = self.fitter.residual(f, xyz, valid) if f_ok else float('inf')
        af = self.fitter.distance(p['protected'], f) if f_ok else float('inf')
        dynamic = self.fitter.distance(p['feedback_prior'], f) if f_ok and p['feedback_prior'] is not None else 0.
        has_rgb = int(np.asarray(valid).sum()) >= 12
        conflict = not f_ok or af > self.thresholds['action_feedback'] or (
            has_rgb and raw_residual > self.thresholds['surface'])
        if conflict and not self.unknown:
            self.stationary = self.last_safe.copy()
        self.unknown = bool(self.unknown or conflict)
        fresh_support = bool(f_ok and has_rgb and af <= self.thresholds['action_feedback']
            and raw_residual <= self.thresholds['surface'] and dynamic <= self.thresholds['dynamics'])
        self.stable = self.stable+1 if fresh_support else 0
        commit = bool(self.unknown and self.stable >= self.confirm)
        changed = False
        if commit:
            measured_fit, measured_ok = self.fitter.fit(f, xyz, valid)
            if self.update_state:
                q = f+self.gain*(measured_fit-f) if measured_ok else f.copy()
            self.unknown = False
            self.stable = 0
        elif not self.unknown and fit_ok and self.update_state:
            q += self.gain*(fit_q-q)
            changed = bool(self.gain and self.fitter.distance(q, proposal['q']) > 1e-7)
        memory_q = self.transition.predict(p['previous'], np.zeros(7, np.float32), p['target'], p['dt'])
        bank = np.stack((p['protected'], f, memory_q, self.stationary, fit_q, q)).astype(np.float32)
        bank_valid = np.array([True, f_ok, True, True, has_rgb, True]) if self.unknown else np.array([False]*5+[True])
        if not self.unknown:
            self.last_safe = q.copy()
        if self.update_state:
            self.base.velocity = np.clip((q-p['previous'])/p['dt'], -self.scale*20, self.scale*20)
            self.base.q = q.copy()
            if commit:
                self.base.hidden = torch.zeros(1, self.base.gate.hidden_size)
                self.base.weights = torch.tensor([1., 0., 0.]).repeat(1, 7, 1)
                self.base.old_feedback = f.copy()
                self.base.old_visual = np.asarray(visual, np.float32).copy() if visual_available else q.copy()
        old_protected = self.protected.copy()
        self.protected = q.copy() if commit and self.update_state else p['protected'].copy()
        self.protected_velocity = np.clip((self.protected-old_protected)/p['dt'], -self.scale*20, self.scale*20)
        if f_ok:
            self.feedback_velocity = np.zeros(7, np.float32) if self.old_feedback is None else np.clip(
                (f-self.old_feedback)/p['dt'], -self.scale*20, self.scale*20)
            self.old_feedback = f.copy()
        else:
            self.old_feedback = None
        self.q = q.astype(np.float32)
        self.pending = None
        return dict(q=self.q.copy(), bank=bank, bank_valid=bank_valid,
            unknown=np.bool_(self.unknown), conflict=np.bool_(conflict), committed=np.bool_(commit and self.update_state),
            surface_corrected=np.bool_(changed), proposal=proposal['q'], weights=proposal['weights'],
            raw_residual=np.float32(raw_residual if np.isfinite(raw_residual) else 1.),
            action_feedback=np.float32(af if np.isfinite(af) else 1.),
            dynamics=np.float32(dynamic), support=np.int32(np.asarray(valid).sum()))
