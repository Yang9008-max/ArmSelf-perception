"""Multi-pose RGB memory and visual correction of the learned body decoder.

No simulator, depth map, body labels, true encoder bias or surface identity is
available here. Reference identities are inherited self-supervised stereo tracks.
The association and bias-filter rules are explicit algorithms, not a trained
object-permanence objective.
"""
from __future__ import annotations

import cv2
import numpy as np

from .body_memory import sample_mask
from .vision import project, triangulate


def farthest_poses(q, scale, budget=64):
    q = np.asarray(q)/np.asarray(scale)
    selected = [0]
    distance = np.sum((q-q[0])**2, axis=1)
    for _ in range(min(budget, len(q))-1):
        index = int(np.argmax(distance))
        selected.append(index)
        distance = np.minimum(distance, np.sum((q-q[index])**2, axis=1))
        distance[selected] = -1
    return np.array(selected)


def guided_stereo(left, right, uv, prediction, calibration, window=11):
    points = np.asarray(uv, np.float32).reshape(-1, 1, 2)
    initial = points.copy()
    initial[:, 0, 0] -= calibration['focal_px']*calibration['baseline_m']/np.maximum(prediction[:, 2], .15)
    kwargs = dict(winSize=(window, window), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, .01))
    match, ok, error = cv2.calcOpticalFlowPyrLK(left, right, points, initial,
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **kwargs)
    reverse, reverse_ok, reverse_error = cv2.calcOpticalFlowPyrLK(right, left, match, points.copy(),
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **kwargs)
    xyz, valid = triangulate(points[:, 0], match[:, 0], calibration)
    valid &= (ok[:, 0].astype(bool) & reverse_ok[:, 0].astype(bool) &
              (error[:, 0] < 22) & (reverse_error[:, 0] < 22) &
              (np.linalg.norm(reverse[:, 0]-points[:, 0], axis=1) < 1.))
    return xyz, valid, error[:, 0]


class VisualBodyBank:
    """Re-identify original feature IDs from RGB observations in multiple poses."""
    def __init__(self, bank, calibration, references=4, window=11, error_limit=22.,
                 spatial_gate_m=.05):
        self.gray = np.asarray(bank['gray'], np.uint8)
        self.q = np.asarray(bank['q'])
        self.uv = np.asarray(bank['uv'], np.float32)
        self.valid = np.asarray(bank['valid'], bool)
        self.scale = np.asarray(bank['scale'])
        self.calibration = calibration
        self.references = references
        self.error_limit = error_limit
        self.spatial_gate_m = spatial_gate_m
        self.kwargs = dict(winSize=(window, window), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, .01))
        self.window = window

    def observe(self, left, right, mask, measured_q, prediction):
        count = self.uv.shape[1]
        prediction = np.asarray(prediction)
        predicted_uv = project(prediction, self.calibration).astype(np.float32)
        gray = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
        other = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        best = np.full(count, np.inf)
        xyz = np.zeros((count, 3), np.float32)
        uv = predicted_uv.copy()
        source = np.full(count, -1, int)
        distances = np.linalg.norm((self.q-measured_q)/self.scale, axis=1)
        # Reference selection uses measured/corrected feedback, not evaluation IDs.
        references = np.argsort(distances)[:self.references]
        for frame in references:
            ids = np.flatnonzero(self.valid[frame] & np.isfinite(prediction).all(-1))
            if not len(ids):
                continue
            old = self.uv[frame, ids].reshape(-1, 1, 2).copy()
            current, ok, error = cv2.calcOpticalFlowPyrLK(self.gray[frame], gray,
                old, predicted_uv[ids, None].copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **self.kwargs)
            back, back_ok, back_error = cv2.calcOpticalFlowPyrLK(gray, self.gray[frame],
                current, old.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **self.kwargs)
            proposed, stereo, stereo_error = guided_stereo(gray, other, current[:, 0],
                                                           prediction[ids], self.calibration, self.window)
            current_uv = current[:, 0]
            discrepancy = np.linalg.norm(proposed-prediction[ids], axis=1)
            valid = (ok[:, 0].astype(bool) & back_ok[:, 0].astype(bool) & stereo &
                (error[:, 0] < self.error_limit) & (back_error[:, 0] < self.error_limit) &
                (np.linalg.norm(back[:, 0]-old[:, 0], axis=1) < 1.2) &
                (np.linalg.norm(current_uv-predicted_uv[ids], axis=1) < 24.) &
                (current_uv[:, 0] > 5) & (current_uv[:, 0] < w-6) &
                (current_uv[:, 1] > 5) & (current_uv[:, 1] < h-6) &
                sample_mask(mask, current_uv) & (discrepancy < self.spatial_gate_m))
            # Appearance plus stereo evidence determine the match. The prior only
            # initializes, bounds and ranks; it never becomes a visual observation.
            score = error[:, 0]+.3*stereo_error+discrepancy*100
            chosen = valid & (score < best[ids])
            chosen_ids = ids[chosen]
            best[chosen_ids] = score[chosen]
            xyz[chosen_ids] = proposed[chosen]
            uv[chosen_ids] = current_uv[chosen]
            source[chosen_ids] = frame
        accepted = np.isfinite(best)
        return dict(xyz=xyz, uv=uv, valid=accepted, reference=source,
                    appearance_cost=np.where(accepted, best, -1).astype(np.float32))


class VisualFeedbackCorrection:
    """Estimate persistent encoder offsets from independent stereo evidence.

    Optimize an offset in the learned decoder's joint input. The body network
    remains frozen. No physical robot equations or true offset labels are used.
    """
    def __init__(self, body, scale, damping=.0002, max_normalized_offset=.65):
        self.body = body
        self.scale = np.asarray(scale, float)
        self.offset = np.zeros(7)
        self.damping = damping
        self.max_offset = max_normalized_offset
        self.last_support = None

    def corrected_q(self, reading):
        return np.asarray(reading)-self.offset*self.scale

    def predict(self, reading):
        return self.body(self.corrected_q(reading))[0]

    def update(self, reading, xyz, valid, time):
        valid = np.asarray(valid, bool) & np.isfinite(xyz).all(-1)
        if valid.sum() < 6:
            return dict(updated=False, support=int(valid.sum()), offset=self.offset.copy())
        start = self.offset.copy()
        for _ in range(3):
            q = self.corrected_q(reading)
            prediction = self.body(q)[0]
            residual = np.asarray(xyz)[valid]-prediction[valid]
            magnitude = np.linalg.norm(residual, axis=1)
            keep = magnitude < .05
            if keep.sum() < 6:
                self.offset = start
                return dict(updated=False, support=int(keep.sum()), offset=self.offset.copy())
            ids = np.flatnonzero(valid)[keep]
            epsilon = .01
            probes = np.r_[q[None]+np.eye(7)*self.scale*epsilon,
                           q[None]-np.eye(7)*self.scale*epsilon]
            values = self.body(probes)
            derivative = -(values[:7, ids]-values[7:, ids])/(2*epsilon)
            jacobian = derivative.transpose(1, 2, 0).reshape(-1, 7)
            weights = np.repeat(np.minimum(1., .012/np.maximum(magnitude[keep], 1e-8)), 3)
            target = residual[keep].reshape(-1)
            delta = np.linalg.solve(jacobian.T @ (weights[:, None]*jacobian)+self.damping*np.eye(7),
                                    jacobian.T @ (weights*target))
            self.offset = np.clip(self.offset+np.clip(delta, -.10, .10), -self.max_offset, self.max_offset)
        self.last_support = float(time)
        return dict(updated=True, support=int(valid.sum()), offset=self.offset.copy())
