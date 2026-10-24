"""Closed-loop MuJoCo acquisition with identical budgets and no privileged inputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from project_paths import portable_path

import mujoco
import numpy as np

from perception.collect import HOME
from perception.environment import create_environment_model, EnvironmentController
from .collect import stereo_scene
from .experiment import write_json
from .exploration import ExplorationBelief, choose_action
from .vision import StereoTracks


AMPLITUDE = np.array([.22, .09, .14, .22, .16, .22, .005])
BUDGETS = (16, 32, 64, 112)


def run_one(output, prior, calibration_data, seed, policy, budget=112):
    output.mkdir(parents=True)
    (output/'observations').mkdir()
    calibration = stereo_scene('sim/scenes/robot-eye-v6.xml', output/'stereo.xml')
    metadata = json.loads((calibration_data/'metadata.json').read_text(encoding='utf-8'))
    if calibration != metadata['calibration']:
        raise ValueError('Exploration must use the same calibrated optical frame')
    selected = np.array(json.loads((prior/'discovery.json').read_text())['selected'])
    with np.load(prior/'reference.npz') as ref:
        tracker = StereoTracks(ref['left'], calibration, 'refined11')
        np.testing.assert_allclose(tracker.reference_points, ref['points'])
    model, environment_hash = create_environment_model(output/'stereo.xml')
    data = mujoco.MjData(model)
    q_indices = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    mujoco.mj_resetDataKeyframe(model, data, 0)
    environment = EnvironmentController(model)
    environment_rng = np.random.default_rng(seed+500)
    environment.configure(data, 'moving', environment_rng)
    mujoco.mj_step(model, data, nstep=round(.8/model.opt.timestep))
    environment.mark_source(data)
    environment.begin_probe(environment_rng)
    action_rng = np.random.default_rng(seed)
    belief = ExplorationBelief(len(selected), HOME, AMPLITUDE)
    records, tracks = [], []
    renderer = mujoco.Renderer(model, height=calibration['height'], width=calibration['width'])
    try:
        for step in range(budget):
            # Only the actual current feedback and already updated belief are accessible here.
            measured_before = data.qpos[q_indices].copy()
            target, decision = choose_action(belief, measured_before, action_rng, policy, step)
            data.ctrl[:] = target
            mujoco.mj_step(model, data, nstep=round(.2/model.opt.timestep))
            environment.advance_bounded(data)
            mujoco.mj_forward(model, data)
            measured = data.qpos[q_indices].astype(np.float32).copy()
            images = []
            for camera in ('fixed', 'entity_right'):
                renderer.update_scene(data, camera=camera)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
                images.append(renderer.render().copy())
            observed = tracker.observe(*images)
            xyz, valid = observed['xyz'][selected], observed['valid'][selected]
            update = belief.update(measured, xyz, valid)
            np.savez_compressed(output/'observations'/f'{step:05d}.npz',
                                left=images[0], right=images[1], q=measured, time=data.time)
            tracks.append(dict(q=measured, target=target, xyz=xyz, valid=valid,
                               uv=observed['uv'][selected]))
            records.append(dict(step=step, time=float(data.time), policy=policy,
                                target=target.tolist(), measured_before=measured_before.tolist(),
                                measured_q=measured.tolist(), visible_features=int(valid.sum()),
                                **decision, **update))
            if step+1 in BUDGETS:
                np.savez_compressed(output/f'belief-{step+1:03d}.npz', **belief.state())
            if (step+1) % 32 == 0:
                print(f'{seed}/{policy}: {step+1}/{budget}, visible {valid.sum()}, score {decision["score"]:.3f}', flush=True)
    finally:
        renderer.close()
    np.savez_compressed(output/'tracks.npz', selected=selected,
                        **{key: np.stack([r[key] for r in tracks]) for key in tracks[0]})
    write_json(output/'decisions.json', records)
    write_json(output/'freeze.json', dict(seed=seed, policy=policy, budget=budget,
        at_utc=datetime.now(timezone.utc).isoformat(), calibration=calibration,
        environment_xml_sha256=environment_hash,
        checkpoints={f'belief-{n:03d}.npz': hashlib.sha256((output/f'belief-{n:03d}.npz').read_bytes()).hexdigest()
                     for n in BUDGETS if n <= budget},
        inputs='RGB stereo, calibration and actual measured q; target requests saved for audit only',
        prior='E2 reference image and discovered feature IDs; no E2 3D network weights',
        no_privileged_evaluation_inputs=True))


def run(output, prior='outputs/embodiment/entity-v2/run-v2',
        calibration_data='outputs/embodiment/entity-v1/data-v1', seeds=(410, 411, 412)):
    output, prior, calibration_data = Path(output), Path(prior), Path(calibration_data)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Preserving active-exploration runs; choose a fresh output')
    output.mkdir(parents=True, exist_ok=True)
    config = dict(seeds=list(seeds), policies=['active', 'random', 'preset'], budget=112,
                  checkpoints=list(BUDGETS), bootstrap=16, seconds_per_observation=.2,
                  center=HOME.tolist(), amplitude=AMPLITUDE.tolist(), max_normalized_step=.5,
                  prior_directory=portable_path(prior),
                  calibration_data_directory=portable_path(calibration_data),
                  algorithm='Online quadratic Bayesian position belief + explicit information acquisition; no RL policy',
                  rules_frozen_utc=datetime.now(timezone.utc).isoformat(),
                  source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                 for name in ('exploration.py', 'explore_run.py', 'vision.py')})
    write_json(output/'config.json', config)
    for seed in seeds:
        for policy in config['policies']:
            run_one(output/f'{seed}-{policy}', prior, calibration_data, seed, policy)
    config['all_models_frozen_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(output/'config.json', config)
    print('All exploratory models frozen; fresh evaluation data may now be collected', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--prior', default='outputs/embodiment/entity-v2/run-v2')
    parser.add_argument('--calibration-data', default='outputs/embodiment/entity-v1/data-v1')
    parser.add_argument('--seeds', type=int, nargs='+', default=[410, 411, 412])
    run(**vars(parser.parse_args()))
