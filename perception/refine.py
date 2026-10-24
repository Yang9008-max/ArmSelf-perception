"""Refine action-derived partial labels with each RGB image's appearance.

Only observed pixels and existing action/background seeds are consumed. GrabCut
learns its color models independently in every image; there is no robot color,
geometry, location, shape or privileged mask prior. Inferred labels remain weak
supervision and can inherit errors in the original action/background evidence.
"""
from __future__ import annotations

import cv2
import numpy as np


def refine_partial_labels(rgb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    image, seeds = np.asarray(rgb), np.asarray(labels)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError('rgb must be HxWx3 uint8')
    if seeds.shape != image.shape[:2] or not np.isin(seeds, [-1, 0, 1]).all():
        raise ValueError('labels must match RGB and contain only -1, 0, 1')
    positive, negative = seeds == 1, seeds == 0
    # OpenCV fits five mixture components per class. Do not fabricate evidence
    # when either class has too few action/background seed pixels.
    if min(int(positive.sum()), int(negative.sum())) < 5:
        return seeds.astype(np.float32, copy=True)
    mask = np.full(seeds.shape, cv2.GC_PR_BGD, np.uint8)
    mask[positive] = cv2.GC_FGD
    mask[negative] = cv2.GC_BGD
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    cv2.setRNGSeed(20261024)
    cv2.grabCut(np.ascontiguousarray(image), mask, None, background_model,
                foreground_model, 5, cv2.GC_INIT_WITH_MASK)
    result = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.float32)
    # Keep this invariant explicit if the underlying implementation changes.
    result[positive], result[negative] = 1., 0.
    return result
