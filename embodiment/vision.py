"""Sparse stereo observations. No arm identity, simulator depth or q input."""
from __future__ import annotations

import cv2
import numpy as np


def triangulate(left, right, calibration):
    left, right = np.asarray(left), np.asarray(right)
    disparity = left[:, 0] - right[:, 0]
    valid = (np.isfinite(left).all(1) & np.isfinite(right).all(1)
             & (disparity > 2.) & (np.abs(left[:, 1]-right[:, 1]) < 1.2))
    z = calibration['focal_px'] * calibration['baseline_m'] / np.maximum(disparity, 1e-6)
    valid &= (z > .15) & (z < 4.)
    points = np.column_stack(((left[:, 0]-calibration['cx'])*z/calibration['focal_px'],
                             (left[:, 1]-calibration['cy'])*z/calibration['focal_px'], z))
    points[~valid] = 0
    return points.astype(np.float32), valid


def project(points, calibration):
    z = np.maximum(np.asarray(points)[..., 2], 1e-6)
    return np.stack((points[..., 0]/z*calibration['focal_px']+calibration['cx'],
                     points[..., 1]/z*calibration['focal_px']+calibration['cy']), axis=-1)


def match_stereo(gray, other, current, calibration, mode='legacy21'):
    """Metric observations from RGB only; small patches reduce depth-edge mixing."""
    if mode not in ('legacy21', 'refined11'):
        raise ValueError(f'Unknown stereo mode: {mode}')
    size = 21 if mode == 'legacy21' else 11
    kwargs = dict(winSize=(size, size), maxLevel=4,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, .01))
    initial = current.copy()
    initial[:, :, 0] -= calibration['focal_px']*calibration['baseline_m']/.9
    matched, ok, error = cv2.calcOpticalFlowPyrLK(
        gray, other, current, initial, flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **kwargs)
    reverse, reverse_ok, _ = cv2.calcOpticalFlowPyrLK(other, gray, matched, None, **kwargs)
    points, valid = triangulate(current[:, 0], matched[:, 0], calibration)
    valid &= (ok[:, 0].astype(bool) & reverse_ok[:, 0].astype(bool)
              & (error[:, 0] < 22) & (np.linalg.norm(reverse[:, 0]-current[:, 0], axis=1) < 1.))
    points[~valid] = 0
    return points, valid


class StereoTracks:
    """Reference-anchored image features retain identity without arm labels."""
    def __init__(self, left, calibration, mode='legacy21'):
        if mode not in ('legacy21', 'refined11'):
            raise ValueError(f'Unknown stereo mode: {mode}')
        self.mode = mode
        self.calibration = calibration
        self.reference = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
        height, width = self.reference.shape
        points = []
        # Equal tile budgets avoid letting the textured floor monopolize features.
        for y in range(0, height, 60):
            for x in range(0, width, 80):
                found = cv2.goodFeaturesToTrack(self.reference[y:y+60, x:x+80],
                                                maxCorners=8, qualityLevel=.015, minDistance=7,
                                                blockSize=5)
                if found is not None:
                    points.extend(found[:, 0] + np.array([x, y]))
        if not points:
            raise ValueError('No trackable image features; entity remains unknown')
        self.reference_points = np.asarray(points, np.float32).reshape(-1, 1, 2)
        self.previous_points = self.reference_points.copy()
        self.kwargs = dict(winSize=(21, 21), maxLevel=4,
                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, .01))

    def observe(self, left, right):
        gray = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
        other = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
        current, ok, error = cv2.calcOpticalFlowPyrLK(
            self.reference, gray, self.reference_points, self.previous_points.copy(),
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **self.kwargs)
        back, back_ok, _ = cv2.calcOpticalFlowPyrLK(gray, self.reference, current, None, **self.kwargs)
        valid = (ok[:, 0].astype(bool) & back_ok[:, 0].astype(bool)
                 & (np.linalg.norm(back[:, 0]-self.reference_points[:, 0], axis=1) < 1.2)
                 & (error[:, 0] < 28))
        points, stereo_valid = match_stereo(gray, other, current, self.calibration, self.mode)
        height, width = gray.shape
        valid &= (stereo_valid
                  & (current[:, 0, 0] > 2) & (current[:, 0, 0] < width-3)
                  & (current[:, 0, 1] > 2) & (current[:, 0, 1] < height-3))
        self.previous_points = current.copy()
        self.previous_points[~valid] = self.reference_points[~valid]
        points[~valid] = 0
        return {'xyz': points, 'valid': valid, 'uv': current[:, 0].copy()}
