"""Conservative static-background seeds from repeated RGB observations.

The reference is fitted once from training observations, then kept frozen for
all splits. It supplies *training labels*, not inference-time subtraction. No
body annotation, simulator state, geometry or chosen robot color is consulted.
"""

from __future__ import annotations

import cv2
import numpy as np


BACKGROUND_THRESHOLDS = {
    "reference_agreement_mean_rgb": 3.0,
    "minimum_reference_support": 0.60,
    "maximum_probe_change_mean_rgb": 2.0,
    "positive_exclusion_radius_px": 1,
}


def _check_rgb(rgb: np.ndarray, name: str) -> np.ndarray:
    image = np.asarray(rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"{name} must be an HxWx3 uint8 RGB array")
    if min(image.shape[:2]) < 1:
        raise ValueError(f"{name} must not be empty")
    return image


def fit_background_reference(rgb_frames: np.ndarray) -> dict:
    """Fit a pixelwise median and repeated-observation support using train RGB.

    ``support[y,x]`` is the fraction of supplied frames whose mean absolute RGB
    distance to the median is at most 3. Calling code must use only training
    observations and keep this reference frozen on validation/test observations.
    Distinct explored poses are needed; duplicate frames do not establish that a
    consistently observed surface is environment rather than an unexcited part.
    """
    frames = np.asarray(rgb_frames)
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("rgb_frames must be an NxHxWx3 uint8 RGB array")
    if frames.shape[0] < 2 or min(frames.shape[1:3]) < 1:
        raise ValueError("At least two nonempty RGB observations are required")
    median = np.median(frames, axis=0).astype(np.uint8)
    distance = np.abs(frames.astype(np.float32) - median.astype(np.float32)[None]).mean(axis=-1)
    support = np.mean(distance <= BACKGROUND_THRESHOLDS["reference_agreement_mean_rgb"], axis=0)
    return {
        "median": median,
        "support": support.astype(np.float32),
        "metadata": {
            "method": "training_rgb_temporal_median_v1",
            "frame_count": int(frames.shape[0]),
            "thresholds": dict(BACKGROUND_THRESHOLDS),
            "fit_policy": "Fit on training RGB only; freeze for all other splits.",
            "use": "Additional negative supervision for currently unknown pixels; not an inference input.",
            "assumptions": [
                "The camera, lighting and environment are fixed during collection.",
                "Training observations explore sufficiently varied robot poses.",
                "Repeatedly observed background is visible in a majority of frames where it is seeded.",
            ],
            "limitations": [
                "A persistently visible unexcited part can resemble static environment.",
                "Uniform moving surfaces can have no observable same-pixel RGB change.",
                "Agreement is evidence under this experiment, not a general proof of non-body membership.",
            ],
        },
    }


def extend_background_labels(rgb: np.ndarray, probe_rgb: np.ndarray,
                             labels: np.ndarray, reference: dict) -> np.ndarray:
    """Add background seeds only where current unknown pixels meet all checks.

    The source pixel must agree with a repeatedly observed training reference
    and remain unchanged at the same image coordinate in all 14 signed probes.
    Existing positives and negatives are preserved. A one-pixel neighborhood of
    current positives is excluded to avoid immediately labeling their boundary.
    Neither the input labels nor the frozen reference is modified.
    """
    source = _check_rgb(rgb, "rgb")
    probes = np.asarray(probe_rgb)
    if probes.dtype != np.uint8 or probes.shape != (14, *source.shape):
        raise ValueError("probe_rgb must be (14,H,W,3) uint8 and match rgb")
    partial = np.asarray(labels)
    if partial.shape != source.shape[:2] or not np.isfinite(partial).all():
        raise ValueError("labels must be a finite HxW array matching rgb")
    if not np.isin(partial, [-1, 0, 1]).all():
        raise ValueError("labels must contain only -1 (unknown), 0 and 1")
    median = _check_rgb(reference["median"], "reference median")
    support = np.asarray(reference["support"], dtype=np.float32)
    if median.shape != source.shape or support.shape != source.shape[:2]:
        raise ValueError("Reference dimensions must match rgb")
    if not np.isfinite(support).all() or np.any((support < 0) | (support > 1)):
        raise ValueError("Reference support must be finite and between 0 and 1")

    source_float = source.astype(np.float32)
    source_distance = np.abs(source_float - median.astype(np.float32)).mean(axis=-1)
    probe_change = np.abs(probes.astype(np.float32) - source_float[None]).mean(axis=-1).max(axis=0)
    positive_neighborhood = cv2.dilate((partial == 1).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    seed = ((partial == -1)
            & (source_distance <= BACKGROUND_THRESHOLDS["reference_agreement_mean_rgb"])
            & (support >= BACKGROUND_THRESHOLDS["minimum_reference_support"])
            & (probe_change <= BACKGROUND_THRESHOLDS["maximum_probe_change_mean_rgb"])
            & ~positive_neighborhood)
    result = partial.copy()
    result[seed] = 0
    return result
