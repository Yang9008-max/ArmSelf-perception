"""Evaluate frozen weights with perturbed current q and unchanged RGB/dq.

This is an inference-input perturbation diagnostic, not a retrained image-only
control. No checkpoint selection, threshold fitting or label generation occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .evaluate import _aggregate_masks, _bound_preprocessing, mask_metrics
from .model import BodyPerception


CONDITIONS = ("correct_q", "shuffled_q", "center_q")
MASK_THRESHOLD = 0.5
MOVING_THRESHOLD_PX = 0.35


def _derangement(size: int, seed: int) -> np.ndarray:
    if size < 2:
        raise ValueError("Cross-pose q shuffling requires at least two poses")
    rng = np.random.default_rng(seed)
    identity = np.arange(size)
    for _ in range(100):
        permutation = rng.permutation(size)
        if np.all(permutation != identity):
            return permutation
    return np.roll(identity, 1)


def _motion_score(jacobian: np.ndarray, dq: np.ndarray, flow: np.ndarray,
                  selector: np.ndarray) -> dict[str, Any]:
    prediction = np.einsum("kchw,nk->nhwc", jacobian, dq, optimize=True)
    errors = np.linalg.norm(prediction - flow, axis=-1)[selector]
    return {"pixels": int(errors.size),
            "sum_error": float(errors.sum(dtype=np.float64)),
            "epe": float(errors.mean()) if errors.size else None}


def ablate_feedback(data: str | Path, checkpoint: str | Path, cache: str | Path,
                    output: str | Path, split: str = "test", seed: int = 20261026,
                    threads: int = 4) -> dict[str, Any]:
    if threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    data, checkpoint, cache, output = map(Path, (data, checkpoint, cache, output))
    if output.exists():
        raise FileExistsError(f"Preserving existing diagnostic: {output}")
    metadata_path = data / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        raise ValueError("Collection must be complete before this diagnostic")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    center, scale = _bound_preprocessing(saved, metadata)
    probe_scales = np.asarray(saved["preprocessing"]["probe_scales"], dtype=np.float32)
    model = BodyPerception(**saved["model_config"])
    model.load_state_dict(saved["state_dict"])
    model.eval()

    sources = sorted((data / "observations" / split).glob("*.npz"))
    permutation = _derangement(len(sources), seed)
    if len(sources) != metadata["counts"][split]:
        raise ValueError("Observation count differs from completed dataset metadata")
    missing = [path.name for path in sources if not (cache / split / path.name).is_file()]
    if missing:
        raise FileNotFoundError(f"Observation caches are incomplete: {missing}")
    measured_q = []
    for source in sources:
        with np.load(source, allow_pickle=False) as observation:
            measured_q.append(observation["q"].astype(np.float32))
    normalized_q = (np.stack(measured_q) - center) / scale
    if not np.isfinite(normalized_q).all():
        raise ValueError("Non-finite current feedback")

    rows = []
    with torch.inference_mode():
        for index, source in enumerate(sources):
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            with np.load(source, allow_pickle=False) as observation:
                rgb = observation["rgb"]
                correct_dq = observation["dq"].astype(np.float32) / probe_scales
            with np.load(cache / split / source.name, allow_pickle=False) as observed:
                if str(observed["source_hash"]) != source_hash:
                    raise ValueError(f"Cache belongs to different observations: {source.name}")
                np.testing.assert_allclose(observed["dq_scaled"], correct_dq, rtol=1e-6, atol=1e-6)
                flow = observed["flow"].astype(np.float32)[1::2]
                valid = observed["valid"].astype(bool)[1::2]
            correct_dq = correct_dq[1::2]
            moving = valid & np.isfinite(flow).all(axis=-1) & (
                np.linalg.norm(flow, axis=-1) > MOVING_THRESHOLD_PX)
            q_conditions = np.stack((normalized_q[index], normalized_q[permutation[index]],
                                     np.zeros_like(normalized_q[index])))
            image = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float() / 255.
            result = model(image[None].expand(len(CONDITIONS), -1, -1, -1),
                           torch.from_numpy(q_conditions))
            if not all(torch.isfinite(result[key]).all() for key in ("logits", "jacobian")):
                raise ValueError(f"Non-finite predictions for {source.name}")
            probabilities = torch.sigmoid(result["logits"])[:, 0].cpu().numpy()
            predictions = probabilities >= MASK_THRESHOLD
            jacobians = result["jacobian"].cpu().numpy()

            # All three predictions are fixed before this evaluation-only read.
            with np.load(data / "evaluation" / split / source.name, allow_pickle=False) as evaluation:
                truth = evaluation["body_mask"].astype(bool)
            conditions = {}
            for condition_index, name in enumerate(CONDITIONS):
                changed = predictions[condition_index] != predictions[0]
                union = predictions[condition_index] | predictions[0]
                conditions[name] = {
                    "segmentation": mask_metrics(predictions[condition_index], truth),
                    "moving_flow": _motion_score(jacobians[condition_index], correct_dq, flow, moving),
                    "changed_pixels": int(changed.sum()),
                    "image_pixels": int(changed.size),
                    "prediction_union_pixels": int(union.sum()),
                    "mask_change_fraction_of_image": float(changed.mean()),
                    "mask_change_fraction_of_prediction_union": float(changed.sum() / union.sum()) if union.any() else 0.,
                    "probability_mean_absolute_change": float(np.abs(probabilities[condition_index] - probabilities[0]).mean()),
                    "normalized_q_change_l2": float(np.linalg.norm(q_conditions[condition_index] - q_conditions[0])),
                }
            rows.append({"id": source.stem, "shuffled_q_donor": sources[permutation[index]].stem,
                         "source_sha256": source_hash, "conditions": conditions})
            print(f"feedback perturbation {index + 1}/{len(sources)}", flush=True)

    summaries = {}
    for name in CONDITIONS:
        values = [row["conditions"][name] for row in rows]
        moving_pixels = sum(value["moving_flow"]["pixels"] for value in values)
        error_sum = sum(value["moving_flow"]["sum_error"] for value in values)
        changed = sum(value["changed_pixels"] for value in values)
        image_pixels = sum(value["image_pixels"] for value in values)
        union_pixels = sum(value["prediction_union_pixels"] for value in values)
        summaries[name] = {
            "segmentation": _aggregate_masks([value["segmentation"] for value in values]),
            "moving_flow": {"pixels": moving_pixels, "epe": error_sum / moving_pixels if moving_pixels else None},
            "mask_change_fraction_of_image": changed / image_pixels,
            "mask_change_fraction_of_prediction_union": changed / union_pixels if union_pixels else 0.,
            "probability_mean_absolute_change": float(np.mean([value["probability_mean_absolute_change"] for value in values])),
            "normalized_q_change_l2_mean": float(np.mean([value["normalized_q_change_l2"] for value in values])),
        }
    report = {
        "diagnostic": "Frozen-model inference-input perturbation; not a retrained image-only control",
        "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "data": str(data), "data_seed": metadata["seed"], "split": split,
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "model_config": saved["model_config"], "preprocessing": saved["preprocessing"],
        "pose_count": len(sources), "permutation_seed": seed,
        "permutation": permutation.tolist(), "mask_threshold": MASK_THRESHOLD,
        "moving_threshold_px": MOVING_THRESHOLD_PX, "threads": threads,
        "protocol": {
            "rgb": "Identical source RGB for all three conditions within each pose",
            "correct_q": "Measured current q normalized with the checkpoint's fixed preprocessing",
            "shuffled_q": "One deterministic whole-vector derangement across different source poses",
            "center_q": "Normalized q is zero, corresponding to the checkpoint's physical q_center",
            "dq": "Correct source-pose measured negative-probe dq in every condition; never shuffled",
            "flow_target": "Cached RGB-derived flow, restricted to valid moving pixels; not privileged physical flow",
            "truth": "Visible-body mask opened after predictions, solely for evaluation",
            "fit_or_selection": "None; weights, pseudo-labels and thresholds remain unchanged",
        },
        "summary": summaries, "per_pose": rows,
        "limitations": [
            "A single q permutation measures sensitivity, not a retrained model's need for feedback.",
            "RGB and q become deliberately inconsistent under perturbation.",
            "Fixed-camera RGB may already reveal pose, so small sensitivity does not establish that feedback is generally useless.",
            "This tests current-q conditioning separately from the explicit correct-dq motion contraction.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)
    print(f"Saved {output}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--threads", type=int, default=4)
    ablate_feedback(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
