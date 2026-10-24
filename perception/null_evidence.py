"""Conservative negative seeds from RGB changes under measured stationary q.

Applicable to this controlled experiment with a fixed camera and lighting within
a group. It is not a general proof of non-body: occlusion and correspondence
errors remain possible. No simulator labels or environment metadata are read.
"""
from __future__ import annotations
import numpy as np
from .flow import estimate_flow, _correspondence_benefit


def stationary_control_evidence(rgb, controls, dq, probe_scales, labels):
    controls = np.asarray(controls)
    dq, scales = np.asarray(dq), np.asarray(probe_scales)
    if controls.shape != (2, *rgb.shape) or dq.shape != (2, len(scales)):
        raise ValueError('Require two independent stationary-feedback controls')
    if (not np.isfinite(dq).all() or not np.isfinite(scales).all()
            or np.any(scales <= 0) or labels.shape != rgb.shape[:2]):
        raise ValueError('Invalid measured feedback, scales or seed dimensions')
    stationary = bool(np.max(np.abs(dq / scales[None])) < .025)
    evidence = np.zeros(labels.shape, dtype=bool)
    if stationary:
        estimates = [estimate_flow(rgb, control) for control in controls]
        flow = np.stack([row[0] for row in estimates])
        valid = np.stack([row[1] for row in estimates])
        benefit, _ = _correspondence_benefit(rgb, controls, flow, valid)
        evidence = np.all(benefit & (np.linalg.norm(flow, axis=-1) > .35), axis=0)
    result = labels.copy()
    conflict = evidence & (labels == 1)
    result[evidence & ~conflict] = 0
    result[conflict] = -1
    diagnostic = {'feedback_stationary': stationary,
                  'max_normalized_feedback_change': float(np.max(np.abs(dq / scales[None]))),
                  'negative_evidence_pixels': int(evidence.sum()),
                  'positive_conflicts': int(conflict.sum())}
    return result, evidence, diagnostic
