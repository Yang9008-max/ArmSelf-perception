"""RGB-only multi-chart surface evidence; no simulator or identity labels.

Points are learned surface samples, not occupied volume. Matching and evidence
thresholds are explicit. The atlas never declares unobserved space free.
"""
from __future__ import annotations

import cv2
import numpy as np

from .model import basis
from .vision import triangulate, project, match_stereo


def patch_seeds(gray, spacing=6, tile_budget=60):
    """Equal full-image tile budgets; no robot crop or segmentation input."""
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    texture = cv2.boxFilter(dx*dx+dy*dy, -1, (7, 7))
    rows = []
    for y in range(8, gray.shape[0]-8, 60):
        for x in range(8, gray.shape[1]-8, 80):
            xx, yy = np.meshgrid(np.arange(x, min(x+80, gray.shape[1]-8), spacing),
                                 np.arange(y, min(y+60, gray.shape[0]-8), spacing))
            uv = np.c_[xx.ravel(), yy.ravel()]
            strength = texture[uv[:, 1], uv[:, 0]]
            uv = uv[strength > 12.]
            # Deterministic spatial subsampling, not strongest-edge-only ranking.
            if len(uv) > tile_budget:
                uv = uv[np.linspace(0, len(uv)-1, tile_budget).astype(int)]
            rows.extend(uv)
    return np.asarray(rows, np.float32).reshape(-1, 2)


def farthest_views(q, center, scale, count=6):
    values = (np.asarray(q)-center)/scale
    chosen = [0]
    distance = np.sum((values-values[0])**2, -1)
    for _ in range(min(count, len(q))-1):
        i = int(np.argmax(distance))
        chosen.append(i)
        distance = np.minimum(distance, np.sum((values-values[i])**2, -1))
        distance[chosen] = -1
    return np.asarray(chosen)


def track_patch(reference, gray, right, uv, reference_xyz, calibration, initial=None):
    n = len(uv)
    if not n:
        return np.empty((0, 3), np.float32), np.empty(0, bool), np.empty((0, 2), np.float32)
    old = np.asarray(uv, np.float32).reshape(-1, 1, 2)
    guess = old.copy() if initial is None else np.asarray(initial, np.float32).reshape(-1, 1, 2)
    kw = dict(winSize=(11, 11), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
    current, ok, error = cv2.calcOpticalFlowPyrLK(reference, gray, old, guess,
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **kw)
    back, ok_back, _ = cv2.calcOpticalFlowPyrLK(gray, reference, current, old.copy(),
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **kw)
    other = current.copy()
    other[:, 0, 0] -= calibration['focal_px']*calibration['baseline_m']/np.maximum(reference_xyz[:, 2], .15)
    stereo, ok_s, err_s = cv2.calcOpticalFlowPyrLK(gray, right, current, other,
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **kw)
    reverse, ok_r, _ = cv2.calcOpticalFlowPyrLK(right, gray, stereo, current.copy(),
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **kw)
    xyz, good = triangulate(current[:, 0], stereo[:, 0], calibration)
    good &= (ok[:, 0].astype(bool) & ok_back[:, 0].astype(bool) & ok_s[:, 0].astype(bool)
        & ok_r[:, 0].astype(bool) & (error[:, 0] < 22) & (err_s[:, 0] < 22)
        & (np.linalg.norm(back[:, 0]-old[:, 0], axis=1) < .8)
        & (np.linalg.norm(reverse[:, 0]-current[:, 0], axis=1) < .8)
        & (current[:, 0, 0] > 7) & (current[:, 0, 0] < gray.shape[1]-8)
        & (current[:, 0, 1] > 7) & (current[:, 0, 1] < gray.shape[0]-8))
    xyz[~good] = 0
    return xyz, good, current[:, 0]


def ridge_fit(features, xyz, valid, ridge=.08):
    usable = np.asarray(valid, bool) & np.isfinite(xyz).all(-1)
    if usable.sum() < features.shape[1]+2:
        return None
    x, y = features[usable], xyz[usable]
    penalty = np.eye(x.shape[1])*ridge
    penalty[0, 0] = 1e-8
    return np.linalg.solve(x.T@x+penalty, x.T@y).astype(np.float32)


def discover_surface(q, xyz, valid, center, scale, ridge=.08):
    """Only sensor trajectories: constant/shuffled controls reject static tracks."""
    features = basis((q-center)/scale)
    fit = np.arange(len(q)) % 5 != 0
    check = ~fit
    shuffled = features[np.random.default_rng(2026102900).permutation(len(q))]
    selected, weights, diagnostics = [], [], []
    for j in range(xyz.shape[1]):
        good = valid[:, j] & np.isfinite(xyz[:, j]).all(-1)
        train, val = good & fit, good & check
        row = dict(id=j, train=int(train.sum()), validation=int(val.sum()), selected=False)
        if train.sum() < 40 or val.sum() < 10:
            diagnostics.append(row)
            continue
        w = ridge_fit(features, xyz[:, j], train, ridge)
        mean = xyz[train, j].mean(0)
        error = float(np.sqrt(np.mean(np.sum((features[val]@w-xyz[val, j])**2, -1))))
        constant = float(np.sqrt(np.mean(np.sum((xyz[val, j]-mean)**2, -1))))
        wrong = float(np.sqrt(np.mean(np.sum((shuffled[val]@w-xyz[val, j])**2, -1))))
        excursion = float(np.sqrt(np.mean(np.sum((xyz[train, j]-mean)**2, -1))))
        keep = excursion > .006 and error < .025 and error < .75*constant and error < .8*wrong
        row.update(error_m=error, constant_m=constant, shuffled_m=wrong, excursion_m=excursion,
                   selected=bool(keep))
        diagnostics.append(row)
        if keep:
            selected.append(j)
            weights.append(ridge_fit(features, xyz[:, j], good, ridge))
    return np.asarray(selected, int), np.asarray(weights, np.float32), diagnostics


class SurfaceAtlas:
    """One q indexes all registered patches; no known link topology is needed."""
    def __init__(self, saved):
        self.saved = {k: np.asarray(v) for k, v in saved.items()}
        self.center, self.scale = saved['center'], saved['scale']
        self.weights = saved['weights']
        self.calibration = {k: float(saved[k]) for k in ('focal_px', 'baseline_m', 'cx', 'cy')}

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as a:
            return cls({k: a[k].copy() for k in a.files})

    def predict(self, q):
        q = np.asarray(q, np.float32)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError('Seven finite current joint estimates required')
        return np.einsum('k,nkc->nc', basis(((q-self.center)/self.scale)[None])[0], self.weights).astype(np.float32)

    def evidence(self, q, left=None, right=None):
        xyz = self.predict(q)
        train = (self.saved['training_q']-self.center)/self.scale
        distance = float(np.linalg.norm(train-(q-self.center)/self.scale, axis=-1).min())
        supported = np.full(len(xyz), distance <= 1.25)
        # 0 unknown support; 1 model-predicted; 2 current stereo-supported.
        status = supported.astype(np.uint8)
        observed_xyz = np.zeros_like(xyz)
        if left is not None and right is not None:
            gray = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
            other = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
            predicted_uv = project(xyz, self.calibration)
            for chart in np.unique(self.saved['chart']):
                ids = np.flatnonzero(self.saved['chart'] == chart)
                measured, good, _ = track_patch(self.saved['gray'][chart], gray, other,
                    self.saved['uv'][ids], xyz[ids], self.calibration, predicted_uv[ids])
                good &= np.linalg.norm(measured-xyz[ids], axis=-1) < .025
                status[ids[good]] = 2
                observed_xyz[ids[good]] = measured[good]
        return dict(xyz=xyz, status=status, observed_xyz=observed_xyz,
                    pose_distance=np.float32(distance))

    def query(self, q, positions, tolerance=.015):
        """Positive surface evidence or unknown; never infer occupied interiors."""
        from scipy.spatial import cKDTree
        evidence = self.evidence(q)
        points = evidence['xyz'][evidence['status'] > 0]
        if not len(points):
            return np.full(len(positions), 'unknown')
        distance, _ = cKDTree(points).query(positions)
        return np.where(distance <= tolerance, 'predicted_surface', 'unknown')
