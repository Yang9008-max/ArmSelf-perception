"""Explicit observation-only audits; inconsistency is NOT fault attribution.

Candidates are dependent (the RGB frontend uses an action prior). These rules
are diagnostic readouts, not learned source-correctness probabilities. They do
not modify the frozen E8 estimate. No condition names or truth enter this API.
"""
from __future__ import annotations

from collections import deque
import numpy as np

PAIRS = ((0, 1), (0, 2), (1, 2))


def pair_distances(candidates, decoder=None, scale=None):
    q = np.asarray(candidates, np.float32)
    values = np.stack([decoder.predict(x) for x in q]) if decoder is not None else q / np.asarray(scale)
    if decoder is not None:
        return np.array([np.sqrt(np.mean(np.sum((values[a]-values[b])**2, -1)))
                         for a, b in PAIRS])
    return np.array([np.sqrt(np.mean((values[a]-values[b])**2)) for a, b in PAIRS])


def classify(distances, available, thresholds):
    available = np.asarray(available, bool)
    present = np.array([available[a] and available[b] for a, b in PAIRS])
    bad = present & (np.asarray(distances) > np.asarray(thresholds))
    # 0: mutually consistent; 1..3: candidate i is the unique outlier;
    # 4: multiple explanations unresolved; 5: insufficient observations.
    if available.sum() < 2:
        return 5, bool(bad.any())
    if not bad.any():
        return 0, False
    if available.all() and bad.sum() == 2:
        compatible = np.flatnonzero(~bad)[0]
        pair = PAIRS[compatible]
        return next(i+1 for i in range(3) if i not in pair), True
    return 4, True


class ConsistencyAudit:
    def __init__(self, thresholds, smoothing=True):
        self.thresholds = np.asarray(thresholds)
        self.smoothing = smoothing
        self.history = deque(maxlen=3)

    def update(self, distances, available):
        status, conflict = classify(distances, available, self.thresholds)
        self.history.append((status, conflict))
        if self.smoothing:
            conflict = sum(int(x[1]) for x in self.history) >= 2
            if status in (1, 2, 3) and sum(x[0] == status for x in self.history) < 2:
                status = 4
            if not conflict and status != 5:
                status = 0
            elif conflict and status == 0:
                status = 4
        return dict(status=np.int8(status), conflict=np.bool_(conflict),
                    unknown=np.bool_(status in (4, 5)))


def interval_envelope(q, half_width, candidates, available, conflict):
    q, half_width = np.asarray(q), np.asarray(half_width)
    lower, upper = q-half_width, q+half_width
    if conflict:
        active = np.asarray(candidates)[np.asarray(available, bool)]
        if len(active):
            lower = np.minimum(lower, active.min(0)-half_width)
            upper = np.maximum(upper, active.max(0)+half_width)
    return lower, upper


def recovery_event(distance, error_excess, start, stop, dt=.1, tolerance=.002, run=5):
    """Keep censored events in the denominator, measure time at confirmation."""
    def first(values):
        count = 0
        for i in range(start, stop):
            count = count+1 if np.isfinite(values[i]) and values[i] <= tolerance else 0
            if count >= run:
                return dict(recovered=True, time_s=float((i-start+1)*dt), confirmation_frame=i)
        return dict(recovered=False, time_s=None, confirmation_frame=None)
    return dict(start=start, stop=stop, convergence=first(np.asarray(distance)),
                actual_error=first(np.asarray(error_excess)))
