"""Collect independent, physically executed camera/feedback probe groups.

Only ``evaluation/`` contains simulator body labels. The observation files contain
camera pixels and actuator-aligned position feedback, never simulator geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = PROJECT_ROOT / "sim" / "scene.xml"
HOME = np.array([0.0, 0.4, 1.0, 0.0, -0.5, 0.0, 0.022])
POSE_HALF_RANGE = np.array([0.65, 0.12, 0.20, 0.50, 0.25, 0.50, 0.006])
PROBE_SCALES = np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.009])


def _prepare_output(output: Path, overwrite: bool) -> None:
    """Replace only a previous collector dataset, never an arbitrary directory."""
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {output}; choose a new path or --overwrite")
        metadata_path = output / "metadata.json"
        if not metadata_path.is_file():
            raise ValueError("--overwrite requires a dataset metadata.json; choose a fresh directory")
        old = json.loads(metadata_path.read_text(encoding="utf-8"))
        if old.get("dataset_format") != "arm-self-perception-probes-v1":
            raise ValueError("Refusing to overwrite a directory that is not a collector dataset")
        # Resolve and check each known child before any recursive removal.
        for name in ("observations", "evaluation"):
            child = output / name
            if child.is_symlink() or child.resolve().parent != output.resolve():
                raise ValueError(f"Unsafe dataset child path: {child}")
            if child.exists():
                shutil.rmtree(child)
    output.mkdir(parents=True, exist_ok=True)


def collect_dataset(
    output: str | Path,
    train: int = 64,
    val: int = 16,
    test: int = 16,
    width: int = 192,
    height: int = 144,
    seed: int = 20261024,
    overwrite: bool = False,
    scene: str | Path = DEFAULT_SCENE,
    settle_seconds: float = 0.8,
    probe_seconds: float = 0.35,
    stop_seconds: float = 0.5,
    environment: str = 'clean',
    shadows: bool = False,
) -> dict:
    """Write a reproducible dataset and return its metadata.

    Every signed probe starts from the same complete settled integration state.
    Feedback and probe increments are measured after stepping real dynamics;
    commanded target positions are not substituted for feedback.
    """
    output = Path(output).resolve()
    scene = Path(scene).resolve()
    counts = {"train": train, "val": val, "test": test}
    if any(n < 0 for n in counts.values()) or sum(counts.values()) < 1:
        raise ValueError("Split sizes must be nonnegative and at least one must be positive")
    if width < 16 or height < 16:
        raise ValueError("Render dimensions must be at least 16 pixels")
    if min(settle_seconds, probe_seconds, stop_seconds) <= 0:
        raise ValueError("All stepping durations must be positive")
    _prepare_output(output, overwrite)

    environment_controller = None
    environment_xml_sha256 = None
    if environment != 'clean':
        from .environment import create_environment_model, EnvironmentController, PROFILES
        if environment != 'mixed' and environment not in PROFILES:
            raise ValueError(f'Unknown environment: {environment}')
        model, environment_xml_sha256 = create_environment_model(scene)
        environment_controller = EnvironmentController(model)
    else:
        model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    if model.nu != 7:
        raise ValueError(f"This generic prototype expects 7 actuator channels, got {model.nu}")
    if np.any(model.actuator_trntype != int(mujoco.mjtTrn.mjTRN_JOINT)):
        raise ValueError("This collector requires direct joint actuators")
    feedback_joint_ids = model.actuator_trnid[:, 0].astype(int)
    feedback_qpos_indices = model.jnt_qposadr[feedback_joint_ids].astype(int)
    actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(i)) for i in feedback_joint_ids]
    right_finger = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_right")
    right_finger_qpos = model.jnt_qposadr[right_finger]
    body_geom_ids = np.flatnonzero(model.geom_group == 1)
    step_counts = {
        "settle": max(1, int(round(settle_seconds / model.opt.timestep))),
        "probe": max(1, int(round(probe_seconds / model.opt.timestep))),
        "stop": max(1, int(round(stop_seconds / model.opt.timestep))),
    }
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    source_state = np.empty(mujoco.mj_stateSize(model, state_spec))
    stop_start_state = np.empty_like(source_state)
    split_sequences = np.random.SeedSequence(seed).spawn(3)
    metadata = {
        "dataset_format": "arm-self-perception-probes-v1",
        "status": "collecting",
        "robot": "generic-six-rotary-joints-plus-gripper; not SO-ARM",
        "seed": seed,
        "counts": counts,
        "width": width,
        "height": height,
        "camera": "fixed",
        "shadows": bool(shadows),
        "state_alignment": "mj_forward refreshes derived positions before RGB rendering",
        "actuator_names": actuator_names,
        "feedback_joint_names": joint_names,
        "position_units": ["radian"] * 6 + ["metre"],
        "feedback_source": "ideal simulated measured joint position; no encoder noise or latency",
        "q_center": HOME.tolist(),
        "q_scale": POSE_HALF_RANGE.tolist(),
        "pose_half_range": POSE_HALF_RANGE.tolist(),
        "probe_scales": PROBE_SCALES.tolist(),
        "probe_axes": np.repeat(np.arange(7), 2).tolist(),
        "probe_signs": np.tile([1, -1], 7).tolist(),
        "probe_order": "axis0+,axis0-,axis1+,axis1-,...,axis6+,axis6-",
        "stop_protocol": "Restore the completed positive shoulder probe (probe index 2), retain its target command, then physically hold for stop duration.",
        "stop_probe_index": 2,
        "timestep_seconds": float(model.opt.timestep),
        "steps": step_counts,
        "durations_seconds": {k: float(v * model.opt.timestep) for k, v in step_counts.items()},
        "scene_file": str(scene.relative_to(PROJECT_ROOT)) if scene.is_relative_to(PROJECT_ROOT) else str(scene),
        "scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
        "versions": {"python": platform.python_version(), "mujoco": mujoco.__version__, "numpy": np.__version__},
        "evaluation_policy": "Only evaluation files contain visible geom-group-1 masks; fixed base is environment.",
        "split_seed_spawn_keys": {name: list(seq.spawn_key) for name, seq in zip(counts, split_sequences)},
        "samples": [],
        "environment": environment,
        "environment_xml_sha256": environment_xml_sha256,
        "null_control_count": 2 if environment_controller else 0,
        "distractor_policy": ('Static noncontact props within each pose group; placement randomized between groups; group 2 used only for evaluation'
                              if environment == 'static' else
                              'Independent randomized per-probe visual mocap trajectories; noncontact; group 2 used only for evaluation'),
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def feedback() -> np.ndarray:
        return data.qpos[feedback_qpos_indices].copy()

    def step(n: int) -> None:
        mujoco.mj_step(model, data, nstep=n)
        if environment_controller:
            environment_controller.advance(data)
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            raise RuntimeError("Simulation produced a non-finite state")

    renderer = mujoco.Renderer(model, height=height, width=width)

    def render_rgb() -> np.ndarray:
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="fixed")
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = shadows
        return renderer.render().copy()

    def render_eval_masks() -> dict:
        renderer.update_scene(data, camera="fixed")
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        renderer.enable_segmentation_rendering()
        try:
            segmentation = renderer.render().copy()
        finally:
            renderer.disable_segmentation_rendering()
        is_geom = segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
        result = {'body': is_geom & np.isin(segmentation[..., 0], body_geom_ids)}
        if environment_controller:
            distractor_ids = np.flatnonzero(model.geom_group == 2)
            result['distractor'] = is_geom & np.isin(segmentation[..., 0], distractor_ids)
            # Benchmark-only counterfactual: measure actual occlusion coverage.
            # This image/mask is never written under observations/.
            option = mujoco.MjvOption()
            option.geomgroup[2] = 0
            renderer.update_scene(data, camera='fixed', scene_option=option)
            renderer.enable_segmentation_rendering()
            try:
                hidden = renderer.render().copy()
            finally:
                renderer.disable_segmentation_rendering()
            result['unoccluded_body'] = ((hidden[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
                                         & np.isin(hidden[..., 0], body_geom_ids))
        return result

    def restore_source() -> None:
        mujoco.mj_setState(model, data, source_state, state_spec)
        mujoco.mj_forward(model, data)

    start = time.perf_counter()
    try:
        for (split, count), sequence in zip(counts.items(), split_sequences):
            (output / "observations" / split).mkdir(parents=True, exist_ok=True)
            (output / "evaluation" / split).mkdir(parents=True, exist_ok=True)
            for index, sample_sequence in enumerate(sequence.spawn(count)):
                rng = np.random.default_rng(sample_sequence)
                target = HOME + rng.uniform(-1, 1, 7) * POSE_HALF_RANGE
                mujoco.mj_resetDataKeyframe(model, data, 0)
                profile = 'clean'
                if environment_controller:
                    from .environment import PROFILES
                    profile = PROFILES[index % len(PROFILES)] if environment == 'mixed' else environment
                    environment_controller.configure(data, profile, rng)
                data.ctrl[:] = target
                data.qpos[feedback_qpos_indices] = target
                data.qpos[right_finger_qpos] = target[6]
                mujoco.mj_forward(model, data)
                step(step_counts["settle"])
                source_q = feedback()
                source_time = float(data.time)
                if environment_controller:
                    environment_controller.mark_source(data)
                mujoco.mj_getState(model, data, source_state, state_spec)
                rgb = render_rgb()
                source_masks = render_eval_masks()

                probe_images, commands, probe_feedback, durations = ([None]*14 for _ in range(4))
                probe_order = rng.permutation(14) if environment_controller else np.arange(14)
                stop_environment_velocity = None
                for probe_index in probe_order:
                    axis, sign = int(probe_index)//2, 1 if int(probe_index)%2 == 0 else -1
                    restore_source()
                    if environment_controller:
                        environment_controller.begin_probe(rng)
                    command = target.copy()
                    command[axis] = np.clip(command[axis] + sign * PROBE_SCALES[axis],
                                            model.actuator_ctrlrange[axis, 0],
                                            model.actuator_ctrlrange[axis, 1])
                    data.ctrl[:] = command
                    step(step_counts["probe"])
                    probe_images[probe_index] = render_rgb()
                    commands[probe_index] = command
                    probe_feedback[probe_index] = feedback()
                    durations[probe_index] = float(data.time - source_time)
                    if axis == 1 and sign == 1:
                        mujoco.mj_getState(model, data, stop_start_state, state_spec)
                        if environment_controller:
                            stop_environment_velocity = environment_controller.velocities.copy()

                null_observations = {}
                null_masks = []
                if environment_controller:
                    null_rgb, null_dq = [], []
                    for _ in range(2):
                        restore_source()
                        environment_controller.begin_probe(rng)
                        data.ctrl[:] = target
                        step(step_counts['probe'])
                        null_rgb.append(render_rgb())
                        null_dq.append(feedback() - source_q)
                        null_masks.append(render_eval_masks())
                    null_observations = {'null_rgb': np.asarray(null_rgb, dtype=np.uint8),
                                         'null_dq': np.asarray(null_dq, dtype=np.float32)}

                mujoco.mj_setState(model, data, stop_start_state, state_spec)
                mujoco.mj_forward(model, data)
                if environment_controller:
                    environment_controller.velocities[:] = stop_environment_velocity
                step(step_counts["stop"])
                stop_rgb = render_rgb()
                stop_q = feedback()
                stop_masks = render_eval_masks()
                probe_q = np.asarray(probe_feedback, dtype=np.float32)
                dq = (np.asarray(probe_feedback) - source_q).astype(np.float32)
                sample_id = f"{split}_{index:05d}"
                obs_path = output / "observations" / split / f"{sample_id}.npz"
                eval_path = output / "evaluation" / split / f"{sample_id}.npz"
                if obs_path.exists() or eval_path.exists():
                    raise FileExistsError(f"Duplicate sample ID: {sample_id}")
                np.savez_compressed(obs_path, rgb=rgb, q=source_q.astype(np.float32),
                                    probe_rgb=np.asarray(probe_images, dtype=np.uint8), dq=dq,
                                    commands=np.asarray(commands, dtype=np.float32), probe_q=probe_q,
                                    dt=np.asarray(durations, dtype=np.float32), stop_rgb=stop_rgb,
                                    stop_q=stop_q.astype(np.float32), **null_observations)
                eval_arrays = {'body_mask': source_masks['body'], 'stop_mask': stop_masks['body']}
                if environment_controller:
                    eval_arrays.update(distractor_mask=source_masks['distractor'],
                                       unoccluded_body_mask=source_masks['unoccluded_body'],
                                       stop_distractor_mask=stop_masks['distractor'],
                                       null_body_mask=np.stack([item['body'] for item in null_masks]),
                                       null_distractor_mask=np.stack([item['distractor'] for item in null_masks]))
                np.savez_compressed(eval_path, **eval_arrays)
                normalized_dq = dq / PROBE_SCALES
                rank = int(np.linalg.matrix_rank(normalized_dq))
                metadata["samples"].append({"id": sample_id, "split": split,
                                            "seed_spawn_key": list(sample_sequence.spawn_key),
                                            "environment_profile": profile,
                                            "probe_execution_order": probe_order.tolist(),
                                            "measured_probe_rank": rank})
                print(f"{sample_id}: actual dq rank={rank}/7; elapsed={time.perf_counter()-start:.1f}s", flush=True)
                if rank < 7:
                    raise RuntimeError(f"Insufficient independent measured excitation in {sample_id}")
        metadata["status"] = "complete"
    finally:
        renderer.close()
        metadata["collection_seconds"] = round(time.perf_counter() - start, 3)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "perception" / "data")
    parser.add_argument("--train", type=int, default=64)
    parser.add_argument("--val", type=int, default=16)
    parser.add_argument("--test", type=int, default=16)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=144)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--settle-seconds", type=float, default=0.8)
    parser.add_argument("--probe-seconds", type=float, default=0.35)
    parser.add_argument("--stop-seconds", type=float, default=0.5)
    parser.add_argument('--environment', choices=('clean', 'mixed', 'static', 'moving', 'occlusion', 'lighting'), default='clean')
    parser.add_argument('--shadows', action='store_true')
    args = parser.parse_args()
    collect_dataset(**vars(args))


if __name__ == "__main__":
    main()
