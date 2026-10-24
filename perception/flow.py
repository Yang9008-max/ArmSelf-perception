"""Image-derived supervision for independently executed joint probes.

This module accepts RGB pixels and measured feedback increments only. It never
reads simulator labels, depth, body IDs, geometry or kinematic Jacobians. A
partial label is evidence under the controlled fixed-camera experiment, not a
claim that arbitrary action-correlated pixels necessarily belong to a body.
"""

from __future__ import annotations

import cv2
import numpy as np


THRESHOLDS = {
    "forward_backward_absolute_px": 0.60,
    "forward_backward_relative": 0.15,
    "photometric_absolute_gray": 38.0,
    "identity_error_minimum_rgb": 5.0,
    "photometric_improvement_absolute_rgb": 2.0,
    "photometric_improvement_relative": 0.35,
    "positive_motion_px": 0.35,
    "negative_motion_px": 0.15,
    "motion_exclusion_px": 0.25,
    "negative_prediction_relative_error": 0.60,
    "negative_prediction_absolute_error_px": 0.75,
    "texture_gradient_gray_per_px": 1.20,
    "minimum_valid_probe_fraction": 0.85,
    "maximum_excitation_condition": 100.0,
}


def _gray(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Images must be HxWx3 uint8 RGB arrays")
    if min(rgb.shape[:2]) < 16:
        raise ValueError("Flow images must be at least 16 pixels in each dimension")
    return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)


def estimate_flow(source: np.ndarray, destination: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return source-to-destination Farneback flow and a consistency mask.

    ``valid`` means the estimated correspondence passed bounds, photometric and
    forward/backward checks. It does *not* mean a textureless zero-flow pixel is
    known to be stationary. ``probe_supervision`` treats that ambiguity explicitly.
    Displacements have (horizontal, vertical) order and pixel units.
    """
    source_gray, destination_gray = _gray(source), _gray(destination)
    if source_gray.shape != destination_gray.shape:
        raise ValueError("Source and destination image sizes must match")
    kwargs = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=4,
                  poly_n=5, poly_sigma=1.1, flags=0)
    forward = cv2.calcOpticalFlowFarneback(source_gray, destination_gray, None, **kwargs)
    backward = cv2.calcOpticalFlowFarneback(destination_gray, source_gray, None, **kwargs)
    height, width = source_gray.shape
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    map_x, map_y = xx + forward[..., 0], yy + forward[..., 1]
    backward_warped = cv2.remap(backward, map_x, map_y, cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
    destination_warped = cv2.remap(destination_gray.astype(np.float32), map_x, map_y,
                                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    magnitude = np.linalg.norm(forward, axis=-1)
    fb_error = np.linalg.norm(forward + backward_warped, axis=-1)
    photometric_error = np.abs(source_gray.astype(np.float32) - destination_warped)
    in_bounds = (map_x >= 1) & (map_x <= width - 2) & (map_y >= 1) & (map_y <= height - 2)
    valid = (in_bounds & np.isfinite(forward).all(axis=-1)
             & (fb_error <= THRESHOLDS["forward_backward_absolute_px"]
                + THRESHOLDS["forward_backward_relative"] * magnitude)
             & (photometric_error <= THRESHOLDS["photometric_absolute_gray"]))
    return forward.astype(np.float32), valid


def _texture_support(gray: np.ndarray) -> np.ndarray:
    """Local gradients supply observation support, without any color prior."""
    gray_float = gray.astype(np.float32)
    gx = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3, scale=1.0 / 8.0)
    gy = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3, scale=1.0 / 8.0)
    gradient = np.sqrt(cv2.boxFilter(gx * gx + gy * gy, -1, (5, 5)))
    return gradient > THRESHOLDS["texture_gradient_gray_per_px"]


def _correspondence_benefit(rgb: np.ndarray, probes: np.ndarray,
                            flow: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Require observed image evidence for a non-identity correspondence.

    Dense optical flow can interpolate a moving foreground's flow into unchanged
    background, passing forward/backward checks there. Warping must actually
    explain RGB changes better than leaving the source pixel at its location.
    Textureless and unchanged pixels cannot pass this positive-evidence gate.
    """
    height, width = rgb.shape[:2]
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    source = rgb.astype(np.float32)
    supported, benefits = [], []
    for destination, displacement, correspondence_valid in zip(probes, flow, valid):
        destination = destination.astype(np.float32)
        identity_error = np.mean(np.abs(destination - source), axis=-1)
        warped = cv2.remap(destination, xx + displacement[..., 0], yy + displacement[..., 1],
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        warped_error = np.mean(np.abs(warped - source), axis=-1)
        improvement = identity_error - warped_error
        supported.append(correspondence_valid
                         & (identity_error > THRESHOLDS["identity_error_minimum_rgb"])
                         & (improvement > np.maximum(
                             THRESHOLDS["photometric_improvement_absolute_rgb"],
                             THRESHOLDS["photometric_improvement_relative"] * identity_error)))
        benefits.append(np.clip(improvement / np.maximum(identity_error, 1.0), 0.0, 1.0))
    return np.stack(supported), np.stack(benefits).astype(np.float32)


def probe_supervision(rgb: np.ndarray, probe_rgb: np.ndarray, dq: np.ndarray,
                      probe_scales: np.ndarray) -> dict:
    """Build flow observations, a local response teacher and partial labels.

    Fourteen probes must have order axis0+, axis0-, ..., axis6+, axis6-.
    Actual feedback increments, rather than commanded targets, form the design
    matrix. The seven positive probes fit J; seven held-out negative probes test
    its predictions. No negative probe participates in the fit.

    Labels are 1 (repeatably moving, feedback-predictable and photometrically
    corroborated in both directions), 0 (reliably static,
    textured and away from movement), or -1 (unknown). Uniform no-flow pixels
    remain unknown. Coverage will be incomplete, especially for tiny parts,
    occlusions, textureless surfaces and weak or nonlinear probe responses.
    """
    gray = _gray(rgb)
    probes = np.asarray(probe_rgb)
    increments = np.asarray(dq, dtype=np.float32)
    scales = np.asarray(probe_scales, dtype=np.float32)
    if probes.dtype != np.uint8 or probes.shape != (14, *rgb.shape):
        raise ValueError("probe_rgb must have shape (14,H,W,3) and dtype uint8")
    if increments.shape != (14, 7) or scales.shape != (7,):
        raise ValueError("dq must be (14,7) and probe_scales must be (7,)")
    if not np.isfinite(increments).all() or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Feedback must be finite and all probe scales strictly positive")

    observations = [estimate_flow(rgb, probe) for probe in probes]
    flow = np.stack([item[0] for item in observations]).astype(np.float32)
    valid = np.stack([item[1] for item in observations])
    correspondence_supported, correspondence_benefit = _correspondence_benefit(rgb, probes, flow, valid)
    dq_scaled = increments / scales[None]
    positive_design, negative_design = dq_scaled[::2], dq_scaled[1::2]
    singular_values = np.linalg.svd(positive_design.astype(np.float64), compute_uv=False)
    rank = int(np.sum(singular_values > 1e-3))
    condition = float(singular_values[0] / singular_values[-1]) if singular_values[-1] > 1e-12 else None
    signed_axes = (np.diag(positive_design) > 0.05) & (np.diag(negative_design) < -0.05)
    excitation_ok = (rank == 7 and condition is not None
                     and condition <= THRESHOLDS["maximum_excitation_condition"]
                     and bool(signed_axes.all()))

    height, width = gray.shape
    positive_flow = flow[::2]
    negative_flow = flow[1::2]
    # Fixed, small ridge guards numerical noise; excitation checks are separate.
    ridge = 1e-4
    solver = np.linalg.solve(positive_design.T @ positive_design + ridge * np.eye(7),
                             positive_design.T).astype(np.float32)
    teacher_flat = solver @ positive_flow.reshape(7, -1)
    teacher_hwc = teacher_flat.reshape(7, height, width, 2)
    predicted_negative = (negative_design @ teacher_flat).reshape(7, height, width, 2)
    teacher_j = teacher_hwc.transpose(0, 3, 1, 2).astype(np.float32)
    if not excitation_ok:
        teacher_j.fill(0)

    positive_magnitude = np.linalg.norm(positive_flow, axis=-1)
    negative_magnitude = np.linalg.norm(negative_flow, axis=-1)
    predicted_magnitude = np.linalg.norm(predicted_negative, axis=-1)
    error = np.linalg.norm(predicted_negative - negative_flow, axis=-1)
    pair_valid = valid[::2] & valid[1::2]
    counts = np.maximum(pair_valid.sum(axis=0), 1)
    mean_error = (error * pair_valid).sum(axis=0) / counts
    zero_error = (negative_magnitude * pair_valid).sum(axis=0) / counts
    relative_error = mean_error / np.maximum(zero_error, 0.03)
    alignment = ((predicted_negative * negative_flow).sum(axis=-1)
                 / np.maximum(predicted_magnitude * negative_magnitude, 1e-6))
    repeated_response = (pair_valid
                         & (positive_magnitude > THRESHOLDS["positive_motion_px"])
                         & (negative_magnitude > THRESHOLDS["positive_motion_px"])
                         & (alignment > 0.5)
                         & (error < 0.55 + 0.30 * negative_magnitude)
                         & (error < 0.65 * negative_magnitude + 0.05))
    # A local J is trusted only when all seven fit observations are reliable.
    # Some negative probes can lose visibility; their checks are omitted.
    enough_observations = valid[::2].all(axis=0) & (valid[1::2].sum(axis=0) >= 5)
    response_trusted = (enough_observations
                        & (relative_error < THRESHOLDS["negative_prediction_relative_error"])
                        & (mean_error < THRESHOLDS["negative_prediction_absolute_error_px"]))
    supported_pairs = (repeated_response
                       & correspondence_supported[::2] & correspondence_supported[1::2])
    foreground = supported_pairs.any(axis=0) & response_trusted
    source_confidence = np.max(
        np.minimum(correspondence_benefit[::2], correspondence_benefit[1::2]) * supported_pairs,
        axis=0)

    magnitude = np.linalg.norm(flow, axis=-1)
    maximum_motion = magnitude.max(axis=0)
    observed_motion = (maximum_motion > THRESHOLDS["motion_exclusion_px"]).astype(np.uint8)
    near_motion = cv2.dilate(observed_motion, np.ones((9, 9), np.uint8)).astype(bool)
    textured = _texture_support(gray)
    background = ((maximum_motion < THRESHOLDS["negative_motion_px"])
                  & (valid.mean(axis=0) >= THRESHOLDS["minimum_valid_probe_fraction"])
                  & textured & ~near_motion)

    labels = np.full((height, width), -1.0, dtype=np.float32)
    confidence = np.zeros((height, width), dtype=np.float32)
    if excitation_ok:
        labels[background] = 0.0
        labels[foreground] = 1.0
        confidence[background] = np.clip(
            1.0 - maximum_motion[background] / THRESHOLDS["negative_motion_px"], 0.2, 1.0)
        confidence[foreground] = (np.clip(1.0 - relative_error[foreground], 0.2, 1.0)
                                  * np.clip(maximum_motion[foreground] / 1.0, 0.35, 1.0)
                                  * source_confidence[foreground])

    moving_pairs = pair_valid & (negative_magnitude > THRESHOLDS["positive_motion_px"])
    diagnostics = {
        "excitation_ok": bool(excitation_ok),
        "excitation_rank": rank,
        "excitation_condition": condition,
        "signed_axes_valid": signed_axes.tolist(),
        "foreground_fraction": float(np.mean(labels == 1)),
        "background_fraction": float(np.mean(labels == 0)),
        "ignored_fraction": float(np.mean(labels < 0)),
        "valid_flow_fraction": float(valid.mean()),
        "uncorroborated_motion_fraction": float(np.mean(
            repeated_response.any(axis=0) & response_trusted & ~foreground)),
        "heldout_negative_moving_error_px": float(error[moving_pairs].mean()) if moving_pairs.any() else None,
        "heldout_negative_zero_error_px": float(negative_magnitude[moving_pairs].mean()) if moving_pairs.any() else None,
        "thresholds": dict(THRESHOLDS),
        "caution": "Partial image-derived evidence only; textureless static regions are unknown. Cast shadows and correlated external objects are not resolved.",
    }
    return {"flow": flow, "valid": valid, "dq_scaled": dq_scaled.astype(np.float32),
            "teacher_j": teacher_j, "labels": labels, "confidence": confidence,
            "diagnostics": diagnostics}
