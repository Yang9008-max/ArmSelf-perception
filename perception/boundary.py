"""RGB boundary refinement seeded only by a network's body probabilities."""
from __future__ import annotations

import cv2
import numpy as np


GRABCUT_CONFIG = {'method': 'grabcut-v1', 'foreground_threshold': .9,
                  'possible_threshold': .1, 'erode_radius': 2,
                  'dilate_radius': 3, 'iterations': 3}


def prediction_mask(rgb, probability, config=None):
    """Return an uncalibrated binary mask; keep the network probabilities intact.

    The graph cut can revise uncertain boundaries using this frame's colors. It
    cannot use scene identities, robot colors, geometry, or evaluation masks.
    Degenerate all-background/all-foreground inputs retain their raw prediction.
    """
    image, scores = np.asarray(rgb), np.asarray(probability)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError('Expected HxWx3 uint8 RGB')
    if (scores.shape != image.shape[:2] or not np.isfinite(scores).all()
            or np.any((scores < 0) | (scores > 1))):
        raise ValueError('Expected finite HxW probabilities in [0, 1]')
    raw = scores >= .5
    if config is None:
        return raw
    if config != GRABCUT_CONFIG:
        raise ValueError('Unsupported boundary refinement configuration')
    foreground = cv2.erode((scores >= config['foreground_threshold']).astype(np.uint8),
                           np.ones((2 * config['erode_radius'] + 1,) * 2, np.uint8)) > 0
    possible = cv2.dilate((scores >= config['possible_threshold']).astype(np.uint8),
                         np.ones((2 * config['dilate_radius'] + 1,) * 2, np.uint8)) > 0
    if foreground.sum() < 5 or (~possible).sum() < 5:
        return raw
    seeds = np.where(raw, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    seeds[~possible] = cv2.GC_BGD
    seeds[foreground] = cv2.GC_FGD
    cv2.setRNGSeed(20261024)
    cv2.grabCut(np.ascontiguousarray(image), seeds, None,
                np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64),
                config['iterations'], cv2.GC_INIT_WITH_MASK)
    return (seeds == cv2.GC_FGD) | (seeds == cv2.GC_PR_FGD)
