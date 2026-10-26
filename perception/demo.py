"""Run trained perception on a deterministic moving-then-held MuJoCo arm."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from .collect import DEFAULT_SCENE, HOME
from .evaluate import mask_metrics, distractor_metrics
from .model import BodyPerception
from .boundary import prediction_mask


WIDTH, HEIGHT = 192, 144
AMPLITUDES = np.array([0.35, 0.08, 0.15, 0.25, 0.15, 0.25, 0.006])


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("C:/Windows/Fonts/arial.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _overlay(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    result = rgb.copy()
    result[mask] = (0.50 * rgb[mask] + 0.50 * np.asarray(color)).astype(np.uint8)
    return result


def _summary(records: list[dict], phase: str) -> dict:
    selected = [row for row in records if row["phase"] == phase]
    return {
        "frames": len(selected),
        "mean_iou": float(np.mean([row["iou"] for row in selected])) if selected else None,
        "mean_recall": float(np.mean([row["recall"] for row in selected])) if selected else None,
        "mean_abs_rotary_speed_rad_s": float(np.mean([row["mean_abs_rotary_speed_rad_s"] for row in selected])) if selected else None,
        "mean_abs_gripper_speed_m_s": float(np.mean([row["abs_gripper_speed_m_s"] for row in selected])) if selected else None,
        "mean_normalized_joint_speed_per_s": float(np.mean([row["mean_normalized_joint_speed_per_s"] for row in selected])) if selected else None,
    }


def generate_demo(checkpoint: str | Path, output: str | Path, frames: int = 90,
                  fps: int = 15, threads: int = 4, environment: str = 'clean',
                  scene: str | Path = DEFAULT_SCENE, shadows: bool = False) -> dict:
    """Save inference animation and measured trajectory; no optimization occurs."""
    if frames < 6 or fps < 1 or fps > 120 or threads < 1:
        raise ValueError("Require frames >= 6, fps between 1 and 120, and positive threads")
    checkpoint, output = Path(checkpoint), Path(output)
    scene = Path(scene)
    if output.suffix.lower() != ".gif":
        raise ValueError("--output must name a .gif file")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Completed model checkpoint not found: {checkpoint}")
    torch.set_num_threads(threads)
    torch.manual_seed(20261024)
    np.random.seed(20261024)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "preprocessing" not in saved or "training" not in saved:
        raise ValueError("Use the completed model.pt checkpoint with preprocessing and training records")
    q_center = np.asarray(saved["preprocessing"]["q_center"], dtype=np.float32)
    q_scale = np.asarray(saved["preprocessing"]["q_scale"], dtype=np.float32)
    if q_center.shape != (7,) or q_scale.shape != (7,) or np.any(q_scale <= 0):
        raise ValueError("Demo requires valid seven-channel checkpoint normalization")
    if not np.isfinite(q_center).all() or not np.isfinite(q_scale).all():
        raise ValueError("Checkpoint normalization must be finite")
    network = BodyPerception(**saved["model_config"])
    network.load_state_dict(saved["state_dict"])
    network.eval()

    controller, environment_xml_hash = None, None
    if environment != 'clean':
        from .environment import create_environment_model, EnvironmentController
        model, environment_xml_hash = create_environment_model(scene)
        controller = EnvironmentController(model)
    else:
        from .environment import create_static_scene_model
        model = create_static_scene_model(scene)
    data = mujoco.MjData(model)
    if model.nu != 7 or np.any(model.actuator_trntype != int(mujoco.mjtTrn.mjTRN_JOINT)):
        raise ValueError("Demo scene must have seven joint actuators")
    joint_ids = model.actuator_trnid[:, 0].astype(int)
    q_indices = model.jnt_qposadr[joint_ids].astype(int)
    v_indices = model.jnt_dofadr[joint_ids].astype(int)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    if controller:
        demo_rng = np.random.default_rng(20261103)
        controller.configure(data, environment, demo_rng)
    mujoco.mj_step(model, data, nstep=round(0.8 / model.opt.timestep))
    if controller:
        controller.mark_source(data)
        controller.begin_probe(demo_rng)
    start_time = float(data.time)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    moving_frames = 2 * frames // 3
    final_command = HOME - AMPLITUDES
    pictures, records = [], []
    font = _font(19)
    small_font = _font(16)

    def update_scene() -> None:
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="fixed")
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = shadows

    try:
        with torch.inference_mode():
            for index in range(frames):
                if index < moving_frames:
                    phase_angle = index / (moving_frames - 1) * 1.5 * np.pi
                    command = HOME + AMPLITUDES * np.sin(phase_angle)
                    phase = "moving"
                else:
                    command = final_command.copy()
                    hold_elapsed = (index - moving_frames + 1) / fps
                    phase = "hold_transient" if hold_elapsed <= 0.3 else "hold"
                data.ctrl[:] = command
                # Alternate integer step counts to keep accumulated simulation time
                # within half a physics step of the requested video timeline.
                steps = (round((index + 1) / (fps * model.opt.timestep))
                         - round(index / (fps * model.opt.timestep)))
                mujoco.mj_step(model, data, nstep=steps)
                if controller:
                    controller.advance_bounded(data)
                q = data.qpos[q_indices].astype(np.float32).copy()
                speed = data.qvel[v_indices].copy()
                if not np.isfinite(q).all() or not np.isfinite(speed).all():
                    raise RuntimeError("Non-finite physical feedback during demo")
                update_scene()
                rgb = renderer.render().copy()
                tensor_rgb = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float()[None] / 255
                tensor_q = torch.from_numpy((q - q_center) / q_scale)[None]
                logits = network(tensor_rgb, tensor_q)["logits"]
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Network produced non-finite logits")
                probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
                prediction = prediction_mask(rgb, probability, saved.get('postprocessing'))

                # Privileged geometry enters only the comparison AFTER inference.
                update_scene()
                renderer.enable_segmentation_rendering()
                try:
                    segmentation = renderer.render().copy()
                finally:
                    renderer.disable_segmentation_rendering()
                body_ids = np.flatnonzero(model.geom_group == 1)
                truth = ((segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
                         & np.isin(segmentation[..., 0], body_ids))
                score = mask_metrics(prediction, truth)
                if controller:
                    prop_ids = np.flatnonzero(model.geom_group == 2)
                    prop_mask = ((segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
                                 & np.isin(segmentation[..., 0], prop_ids))
                    score['distractors'] = distractor_metrics(prediction, prop_mask)
                    score['prop_positions'] = data.mocap_pos[controller.mocap_ids].tolist()
                elapsed = float(data.time - start_time)
                records.append({"frame": index, "time_seconds": elapsed, "phase": phase,
                                "command": command.tolist(), "q": q.tolist(), "qvel": speed.tolist(),
                                "mean_abs_rotary_speed_rad_s": float(np.mean(np.abs(speed[:6]))),
                                "abs_gripper_speed_m_s": float(abs(speed[6])),
                                "mean_normalized_joint_speed_per_s": float(np.mean(np.abs(speed / q_scale))),
                                **score})
                panel = Image.new("RGB", (WIDTH * 6, HEIGHT * 2 + 70), "#17202c")
                draw = ImageDraw.Draw(panel)
                phase_label = {"moving": "MOVING", "hold_transient": "HOLD (settling)", "hold": "HOLD"}[phase]
                draw.text((12, 8), f"t={elapsed:4.2f}s   {phase_label}   IoU={score['iou']:.3f}", font=font, fill="white")
                images = [rgb, _overlay(rgb, prediction, (40, 240, 90)), _overlay(rgb, truth, (255, 150, 35))]
                for col, (pixels, label) in enumerate(zip(images, ("Camera RGB", "Model prediction", "Evaluation truth only"))):
                    x = col * WIDTH * 2
                    draw.text((x + 12, 38), label, font=small_font, fill="#c8d8ee")
                    panel.paste(Image.fromarray(pixels).resize((WIDTH * 2, HEIGHT * 2), Image.Resampling.NEAREST), (x, 70))
                pictures.append(panel)
    finally:
        renderer.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    pictures[0].save(output, save_all=True, append_images=pictures[1:], loop=0,
                     duration=round(1000 / fps), disposal=2, optimize=False)
    summary = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "model_config": saved["model_config"], "preprocessing": saved["preprocessing"],
        "scene_file": str(scene.resolve()),
        "scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
        "repeatability_seed": 20261024, "random_motion": False,
        "environment": environment, "environment_seed": 20261103 if controller else None,
        "environment_xml_sha256": environment_xml_hash,
        "environment_trajectory": ('Independent bounded sinusoidal prop motion continues during arm hold'
                                   if environment in ('moving', 'occlusion') else 'Props remain fixed throughout the trajectory'),
        "versions": {"mujoco": mujoco.__version__, "torch": str(torch.__version__), "numpy": np.__version__},
        "frames": frames, "fps": fps, "moving_frames": moving_frames,
        "resolution": [WIDTH, HEIGHT], "threshold": 0.5, "shadows": bool(shadows),
        "postprocessing": saved.get('postprocessing'),
        "settle_before_motion_seconds": 0.8,
        "hold_metric_excludes_first_seconds": 0.3,
        "trajectory": "All seven commands follow home + amplitude * sin(phase); phase progresses from 0 to 1.5*pi, then the final non-home target is held.",
        "amplitudes": AMPLITUDES.tolist(), "final_command": final_command.tolist(),
        "moving": _summary(records, "moving"), "hold_transient": _summary(records, "hold_transient"),
        "hold": _summary(records, "hold"), "per_frame": records,
        "ground_truth_usage": "Segmentation render after prediction, only for metric and right-hand comparison panel; no training or model input.",
        "limitations": ["One deterministic trajectory, not a statistically independent benchmark",
                        "Generic seven-channel simulator, not SO-ARM or real hardware",
                        "Ideal feedback; fixed camera and appearance; configured shadow setting",
                        "No learned temporal memory: each inference uses its current image and measured positions"],
    }
    summary_path = output.with_suffix(".json")
    if controller:
        for phase in ('moving', 'hold'):
            prop_rows = [row['distractors'] for row in records if row['phase'] == phase]
            pixels = sum(row['pixels'] for row in prop_rows)
            summary[phase]['distractor_false_positive_rate'] = (
                sum(row['predicted_body_pixels'] for row in prop_rows)/pixels if pixels else None)
        summary['limitations'][2] = 'Ideal feedback; fixed camera; noncontact visual props; configured shadow setting'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"GIF: {output.resolve()}", flush=True)
    print(f"Moving mean IoU: {summary['moving']['mean_iou']}; held mean IoU: {summary['hold']['mean_iou']}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument('--environment', choices=('clean', 'static', 'moving', 'occlusion', 'lighting'), default='clean')
    parser.add_argument('--scene', type=Path, default=DEFAULT_SCENE)
    parser.add_argument('--shadows', action='store_true')
    generate_demo(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
