"""E6: prospective action prediction and shared-state sensory updates.

Training reads only transitions of measured feedback. Final action predictions
are written before the corresponding simulator action. Replay filters consume
the command first, then the sensor reading and RGB. Scoring is a separate pass.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import cv2
import mujoco
import numpy as np
import torch

from perception.collect import HOME
from .body_cycle import reference, read_json, write_json, sha, utc, stack
from .body_fixture import make_fixture, render_rgb, render_truth, EvaluationLandmarks
from .body_memory import BodyModel, BodyMasks
from .explore_run import AMPLITUDE
from .recovery import VisualBodyBank
from .recovery_cycle import BODY, SEGMENTATION, BIAS
from .action_state import ActionTransition, ActionBodyState

ROOT = Path('outputs/local-check-20261024/e6-action-body')
BANK = Path('outputs/local-check-20261024/e5-body-memory/visual-bank.npz')
TRAIN_SEEDS = (2026102601, 2026102602, 2026102603)
VAL_SEED, DEV_SEED = 2026102611, 2026102621
FINAL_SEEDS = (2026102631, 2026102632, 2026102633)
CONDITIONS = ('clean', 'biased_feedback', 'delayed_feedback', 'camera_missing', 'wrong_rgb')


def schedule(seed, count):
    rng = np.random.default_rng(seed)
    previous = HOME.copy()
    for i in range(count):
        duration = (.02, .06, .12)[i % 3]
        if i % 8 in (3, 4):
            target, phase = previous.copy(), 'hold_command'
        elif i % 8 == 5:
            target, phase = 2*HOME-previous, 'reverse'
        else:
            target, phase = HOME+rng.uniform(-.8, .8, 7)*AMPLITUDE, 'new_command'
        previous = target.copy()
        yield dict(target=target.astype(np.float32), duration=duration, phase=phase)


def fresh(path):
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError(f'Preserving existing experiment: {path}')


def collect_transitions(output, seed, count):
    destination = output/f'transitions-{seed}.npz'
    if destination.exists():
        return destination
    fixture = output/f'fixture-{seed}'
    fixture.mkdir(parents=True, exist_ok=True)
    model, data, _ = make_fixture(fixture, panel=False)
    indices = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    props = [model.body(f'table_object_{i}').id for i in range(3)]
    original_props = data.xpos[props].copy()
    previous, previous_dt = data.qpos[indices].copy(), .1
    rows = []
    for request in schedule(seed, count):
        q = data.qpos[indices].copy()
        velocity = (q-previous)/previous_dt
        target, duration = request['target'], request['duration']
        data.ctrl[:] = target
        mujoco.mj_step(model, data, nstep=round(duration/model.opt.timestep))
        mujoco.mj_forward(model, data)
        rows.append(dict(q=q.astype(np.float32), velocity=velocity.astype(np.float32), target=target,
                         duration=np.float32(duration), next_q=data.qpos[indices].astype(np.float32).copy()))
        np.testing.assert_array_equal(original_props, data.xpos[props])
        previous, previous_dt = q, duration
    np.savez_compressed(destination, **stack(rows))
    print(f'feedback training observations {seed}: {count}', flush=True)
    return destination


def load_npz(path):
    with np.load(path) as a:
        return {k: a[k].copy() for k in a.files}


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    if (output/'training-freeze.json').exists():
        return
    training = [load_npz(collect_transitions(output, s, 700)) for s in TRAIN_SEEDS]
    validation = load_npz(collect_transitions(output, VAL_SEED, 350))
    training = {k: np.concatenate([r[k] for r in training]) for k in training[0]}
    body = BodyModel(BODY)
    data = [{k: torch.from_numpy(v) for k, v in a.items()} for a in (training, validation)]
    history = {}
    for use_action, label in [(True, 'action'), (False, 'no-action')]:
        torch.manual_seed(2026102600)
        model = ActionTransition(body.saved['center'], body.saved['scale'], use_action)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.0001)
        best, best_state, trace = float('inf'), None, []
        for epoch in range(650):
            model.train()
            train = data[0]
            optimizer.zero_grad()
            predicted = model(train['q'], train['velocity'], train['target'], train['duration'])
            loss = (((predicted-train['next_q'])/model.scale)**2).mean()
            loss.backward()
            optimizer.step()
            model.eval()
            val = data[1]
            with torch.inference_mode():
                score = float((((model(val['q'], val['velocity'], val['target'], val['duration'])-val['next_q'])/model.scale)**2).mean())
            trace.append(dict(epoch=epoch, train=float(loss.detach()), validation=score))
            if score < best:
                best, best_state = score, copy.deepcopy(model.state_dict())
        torch.save(dict(center=body.saved['center'], scale=body.saved['scale'], use_action=use_action,
                        state_dict=best_state), output/f'{label}.pt')
        history[label] = trace
        print(f'{label} validation normalized joint RMSE: {np.sqrt(best):.4f}', flush=True)
    write_json(output/'training-history.json', history)
    write_json(output/'training-freeze.json', dict(frozen_utc=utc(), train_seeds=list(TRAIN_SEEDS), val_seed=VAL_SEED,
        training_transitions=2100, validation_transitions=350, geometry_used=False,
        models={k: sha(output/f'{k}.pt') for k in ('action', 'no-action')},
        body_sha256=sha(BODY), bank_sha256=sha(BANK)))


def collect_evaluation(output, seed, count=120):
    source = output/f'data-{seed}'
    if (source/'metadata.json').exists():
        return source
    fresh(source)
    for name in ('before', 'after', 'evaluation'):
        (source/name).mkdir()
    model, data, calibration = make_fixture(source, panel=False)
    saved, _, pixels = reference()
    anchors = EvaluationLandmarks(model, data, pixels[saved['selected'], 0], calibration)
    body = BodyModel(BODY)
    models = {k: ActionTransition.load(output/f'{k}.pt') for k in ('action', 'no-action')}
    indices = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    props = [model.body(f'table_object_{i}').id for i in range(3)]
    original_props = data.xpos[props].copy()
    renderer = mujoco.Renderer(model, height=480, width=640)
    previous, previous_dt = data.qpos[indices].copy(), .1
    np.savez_compressed(source/'initial.npz', q=previous.astype(np.float32), time=data.time)
    rows, pre_hashes = [], {}
    previous_command = HOME.copy()
    try:
        for i, request in enumerate(schedule(seed, count)):
            q = data.qpos[indices].astype(np.float32).copy()
            velocity = (q-previous)/previous_dt
            target, duration = request['target'], request['duration']
            wrong_target = np.asarray(HOME*2-target, np.float32)
            next_q = {k: model_.predict(q, velocity, target, duration) for k, model_ in models.items()}
            next_q.update(hold=q.copy(), target=target.copy(), wrong_action=models['action'].predict(q, velocity, wrong_target, duration))
            filename = f'{i:05d}.npz'
            before = dict(q=q, velocity=velocity.astype(np.float32), target=target, duration=np.float32(duration), time=np.float64(data.time),
                          **{f'q_{k}': v for k, v in next_q.items()}, **{f'xyz_{k}': body(v)[0] for k, v in next_q.items()})
            # This write completes BEFORE the action and any future observation.
            np.savez_compressed(source/'before'/filename, **before)
            pre_hashes[filename] = sha(source/'before'/filename)
            # Unknown actuator delay probes after otherwise normal observations.
            delay = min(duration, .02) if 90 <= i < 100 else 0.
            if delay:
                data.ctrl[:] = previous_command
                mujoco.mj_step(model, data, nstep=round(delay/model.opt.timestep))
            data.ctrl[:] = target
            if duration > delay:
                mujoco.mj_step(model, data, nstep=round((duration-delay)/model.opt.timestep))
            mujoco.mj_forward(model, data)
            np.savez_compressed(source/'after'/filename, q=data.qpos[indices].astype(np.float32).copy(), time=data.time,
                left=render_rgb(renderer, data), right=render_rgb(renderer, data, 'entity_right'))
            truth = render_truth(renderer, model, data)
            truth.update(anchors.score_truth(model, data, truth))
            # Sparse truth, body mask and depth remain in the evaluation directory.
            np.savez_compressed(source/'evaluation'/filename, **truth)
            np.testing.assert_array_equal(original_props, data.xpos[props])
            rows.append(dict(file=filename, phase=request['phase'], duration=duration, actuator_delay_s=delay))
            previous, previous_dt, previous_command = q, duration, target.copy()
            if (i+1) % 40 == 0:
                print(f'prospective collection {seed}: {i+1}/{count}', flush=True)
    finally:
        renderer.close()
    write_json(source/'metadata.json', dict(seed=seed, frames=rows, calibration=calibration,
        created_utc=utc(), before_action_prediction_hashes=pre_hashes, fixed_body_points=int(anchors.is_body.sum()),
        prop_max_displacement_m=0., predictions_saved_before_actuation=True,
        action_delay_hidden_from_predictor=True, initial_feedback_clean=True))
    return source


def inputs(source, condition):
    metadata = read_json(source/'metadata.json')
    for i, row in enumerate(metadata['frames']):
        # Future observations are not opened until the caller resumes this
        # generator after consuming the command and making its forecast.
        with np.load(source/'before'/row['file']) as before:
            yield 'command', dict(target=before['target'].copy(), duration=float(before['duration']))
        actual = load_npz(source/'after'/row['file'])
        reading = actual['q'].copy()
        if condition == 'biased_feedback':
            reading += BIAS*np.clip((i-20)/5., 0., 1.)
        elif condition == 'delayed_feedback' and i >= 25:
            with np.load(source/'after'/metadata['frames'][i-1]['file']) as old:
                reading = old['q'].copy()
        available = not (condition == 'camera_missing' and 35 <= i < 80)
        if condition == 'wrong_rgb' and i >= 25:
            # Replay an earlier RGB pair; future RGB is never exposed.
            wrong = load_npz(source/'after'/metadata['frames'][i % 20]['file'])
            actual['left'], actual['right'] = wrong['left'], wrong['right']
        if not available:
            actual['left'][:] = actual['right'][:] = 0
        yield 'observation', dict(q=reading, left=actual['left'], right=actual['right'],
                                  time=float(actual['time']), available=available)


def paired_actions(output, seed, count=12):
    """Execute both commands from the exact same saved physical initial state."""
    destination = output/f'paired-{seed}'
    if (destination/'scores.json').exists():
        return read_json(destination/'scores.json')
    fresh(destination)
    (destination/'before').mkdir()
    (destination/'evaluation').mkdir()
    model, data, calibration = make_fixture(destination, panel=False)
    saved, _, pixels = reference()
    anchors = EvaluationLandmarks(model, data, pixels[saved['selected'], 0], calibration)
    body = BodyModel(BODY)
    dynamics = {k: ActionTransition.load(output/f'{k}.pt') for k in ('action', 'no-action')}
    indices = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    props = [model.body(f'table_object_{i}').id for i in range(3)]
    original_props = data.xpos[props].copy()
    rng = np.random.default_rng(seed+1000)
    signature = mujoco.mjtState.mjSTATE_INTEGRATION
    snapshot = np.empty(mujoco.mj_stateSize(model, signature))
    predictions, truths, hashes = [], [], {}
    for i in range(count):
        pose = HOME+rng.uniform(-.45, .45, 7)*AMPLITUDE
        data.ctrl[:] = pose
        mujoco.mj_step(model, data, nstep=200)
        previous = data.qpos[indices].copy()
        mujoco.mj_step(model, data, nstep=10)
        mujoco.mj_forward(model, data)
        q = data.qpos[indices].copy()
        velocity = (q-previous)/.02
        mujoco.mj_getState(model, data, snapshot, signature)
        direction = rng.uniform(-1., 1., 7)*AMPLITUDE*.45
        commands = np.stack([q-direction, q+direction]).astype(np.float32)
        duration = (.02, .06, .12)[i % 3]
        predicted = {name: np.stack([body(dyn.predict(q, velocity, target, duration))[0] for target in commands])
                     for name, dyn in dynamics.items()}
        predicted.update(target=body(commands), hold=body(np.stack([q, q])))
        filename = f'{i:03d}.npz'
        np.savez_compressed(destination/'before'/filename, q=q, velocity=velocity, commands=commands, duration=duration, **predicted)
        hashes[filename] = sha(destination/'before'/filename)
        pair_truth = []
        for target in commands:
            mujoco.mj_setState(model, data, snapshot, signature)
            mujoco.mj_forward(model, data)
            np.testing.assert_array_equal(data.qpos[indices], q)
            data.ctrl[:] = target
            mujoco.mj_step(model, data, nstep=round(duration/model.opt.timestep))
            mujoco.mj_forward(model, data)
            pair_truth.append(anchors.positions(model, data))
            np.testing.assert_array_equal(original_props, data.xpos[props])
        truth = np.stack(pair_truth)
        np.savez_compressed(destination/'evaluation'/filename, xyz=truth, body=anchors.is_body)
        predictions.append(predicted)
        truths.append(truth)
    truth = np.stack(truths)
    actual_change = truth[:, 1]-truth[:, 0]
    scores = {}
    for mode in ('action', 'no-action', 'target', 'hold', 'wrong_action'):
        values = np.stack([r['action' if mode == 'wrong_action' else mode] for r in predictions])
        if mode == 'wrong_action':
            values = values[:, ::-1]
        change = values[:, 1]-values[:, 0]
        scores[mode] = dict(all_body_rmse_mm=float(np.sqrt(np.mean(np.sum((values-truth)**2, axis=-1)[:, :, anchors.is_body]))*1000),
            action_contrast_rmse_mm=float(np.sqrt(np.mean(np.sum((change-actual_change)**2, axis=-1)[:, anchors.is_body]))*1000))
    result = dict(seed=seed, pairs=count, modes=scores, exact_initial_physical_state_restored=True,
                  before_prediction_hashes=hashes, prop_max_displacement_m=0.,
                  truth_opened_after_predictions=True)
    write_json(destination/'scores.json', result)
    print(f'paired actions {seed}: {count} same-state pairs', flush=True)
    return result


def infer(output, source, condition, config, suffix=''):
    name = f'{source.name}-{condition}{suffix}'
    destination = output/'predictions'/f'{name}.npz'
    if destination.exists():
        return destination
    destination.parent.mkdir(exist_ok=True)
    metadata = read_json(source/'metadata.json')
    initial_q = load_npz(source/'initial.npz')['q']
    body, masks = BodyModel(BODY), BodyMasks(SEGMENTATION)
    models = {k: ActionTransition.load(output/f'{k}.pt') for k in ('action', 'no-action')}
    with np.load(BANK) as saved:
        bank = VisualBodyBank(saved, metadata['calibration'], references=4, window=7)
    states = {name: ActionBodyState(models['no-action' if name == 'no_action' else 'action'], body, initial_q,
              vision_enabled=name in ('full', 'no_action', 'no_feedback'),
              feedback_enabled=name not in ('action_only', 'no_feedback'), **config)
              for name in ('full', 'action_feedback', 'action_only', 'no_action', 'no_feedback')}
    rows, pending = [], None
    cv2.setRNGSeed(20261026)
    for kind, observation in inputs(source, condition):
        if kind == 'command':
            pending = {name: state.forecast(observation['target'], observation['duration']) for name, state in states.items()}
            continue
        obs = observation
        row = dict(q_reading=obs['q'], prediction_feedback_only=body(obs['q'])[0], available=np.bool_(obs['available']))
        feedback = {name: state.feedback(obs['q'] if name != 'no_feedback' else None) for name, state in states.items()}
        common_mask = masks(obs['left'], obs['q'])['v7'] if obs['available'] else np.zeros((480, 640), bool)
        for name, state in states.items():
            q = feedback[name]
            measurement = None
            # Unavailable cameras are not submitted as measurements.
            if state.vision_enabled and obs['available']:
                mask = masks(obs['left'], q)['v7'] if name == 'no_feedback' else common_mask
                measurement = bank.observe(obs['left'], obs['right'], mask, q, body(q)[0])
            final = state.finish(None if measurement is None else measurement['xyz'],
                                 None if measurement is None else measurement['valid'], obs['time'])
            row.update({f'prior_{name}': pending[name]['xyz'], f'after_feedback_{name}': body(q)[0],
                        f'prediction_{name}': final['xyz'], f'visual_update_{name}': np.bool_(final['visual_update']),
                        f'support_{name}': np.int32(final['visual_support']), f'innovation_{name}': np.float32(final['innovation_m']),
                        f'bias_{name}': final['bias'], f'q_{name}': final['q']})
        rows.append(row)
        if len(rows) % 40 == 0:
            print(f'{destination.stem}: {len(rows)}/120', flush=True)
    np.savez_compressed(destination, **stack(rows))
    write_json(destination.with_suffix('.freeze.json'), dict(frozen_utc=utc(), prediction_sha256=sha(destination),
        truth_used=False, condition=condition, config=config, initial_feedback_clean=True,
        all_modes_use_same_body_decoder=True))
    return destination


def score(output, source, predictions):
    metadata = read_json(source/'metadata.json')
    for filename, digest in metadata['before_action_prediction_hashes'].items():
        assert sha(source/'before'/filename) == digest
    truths = [load_npz(source/'evaluation'/r['file']) for r in metadata['frames']]
    xyz = np.stack([t['anchor_xyz'] for t in truths])
    body = np.stack([t['anchor_body'] for t in truths])
    def error(prediction, selection=None):
        selection = np.ones(len(body), bool) if selection is None else selection
        return float(np.sqrt(np.mean(np.sum((prediction-xyz)**2, axis=-1)[body & selection[:, None]]))*1000)
    out = dict(seed=metadata['seed'], frames=len(body), fixed_body_points=int(body[0].sum()), conditions={})
    before = [load_npz(source/'before'/r['file']) for r in metadata['frames']]
    after_q = np.stack([load_npz(source/'after'/r['file'])['q'] for r in metadata['frames']])
    out['prospective'] = {}
    scale = np.asarray(BodyModel(BODY).saved['scale'])
    normal = np.array([r['actuator_delay_s'] == 0 for r in metadata['frames']])
    for mode in ('action', 'no-action', 'hold', 'target', 'wrong_action'):
        p = np.stack([r[f'xyz_{mode}'] for r in before])
        q = np.stack([r[f'q_{mode}'] for r in before])
        out['prospective'][mode] = dict(body_rmse_mm=error(p), normal_rmse_mm=error(p, normal),
            unexpected_actuator_delay_rmse_mm=error(p, ~normal),
            normalized_joint_rmse=float(np.sqrt(np.mean(((q-after_q)/scale)**2))),
            by_duration={str(dt): error(p, np.array([r['duration'] == dt for r in metadata['frames']]) & normal)
                         for dt in (.02, .06, .12)})
    for prediction in predictions:
        assert sha(prediction) == read_json(prediction.with_suffix('.freeze.json'))['prediction_sha256']
        a = load_npz(prediction)
        condition = read_json(prediction.with_suffix('.freeze.json'))['condition']
        values = {}
        for mode in ('feedback_only', 'full', 'action_feedback', 'action_only', 'no_action', 'no_feedback'):
            values[mode] = dict(body_rmse_mm=error(a[f'prediction_{mode}']),
                normal_actuation_rmse_mm=error(a[f'prediction_{mode}'], normal),
                actuator_delay_rmse_mm=error(a[f'prediction_{mode}'], ~normal),
                after_intervention_rmse_mm=error(a[f'prediction_{mode}'], np.arange(len(body)) >= 30))
            if mode != 'feedback_only':
                values[mode].update(visual_update_frames=int(a[f'visual_update_{mode}'].sum()),
                    prior_rmse_mm=error(a[f'prior_{mode}']), feedback_stage_rmse_mm=error(a[f'after_feedback_{mode}']),
                    after_intervention_visual_updates=int(a[f'visual_update_{mode}'][30:].sum()))
        out['conditions'][condition] = values
    (output/'scores').mkdir(exist_ok=True)
    suffix = '-'+predictions[0].stem.split('--')[-1] if '--' in predictions[0].stem else ''
    write_json(output/'scores'/f'{source.name}{suffix}.json', out)
    print('scores', source.name, {c: {m: round(r['body_rmse_mm'], 2) for m, r in v.items()}
                                for c, v in out['conditions'].items()}, flush=True)
    return out


def develop(output):
    prepare(output)
    source = collect_evaluation(output, DEV_SEED)
    paired_actions(output, DEV_SEED)
    for gate in (.008, .014):
        config = dict(feedback_gain=.9, innovation_gate_m=gate)
        suffix = f'--gate{round(gate*1000)}'
        paths = [infer(output, source, condition, config, suffix) for condition in ('clean', 'biased_feedback', 'wrong_rgb')]
        score(output, source, paths)


def freeze(output, gate):
    if (output/'final-protocol.json').exists():
        raise FileExistsError('Final protocol is already frozen')
    write_json(output/'final-protocol.json', dict(frozen_utc=utc(), seeds=list(FINAL_SEEDS), conditions=list(CONDITIONS),
        config=dict(feedback_gain=.9, innovation_gate_m=gate), durations_s=[.02, .06, .12],
        normal_noninferiority_tolerance_mm=.5, fixed_body_points=28, frames_per_seed=120,
        selection='8 mm gate selected on development seed: clean/wrong-RGB unchanged, biased-feedback error lower than 14 mm gate',
        bias=BIAS.tolist(), bias_ramp_frames=[21, 25], camera_missing_frames=[35, 79],
        delayed_feedback='one past frame after frame 25', wrong_rgb='cycling first 20 recorded frames after frame 25',
        actuator_delay_frames=[90, 99], actuator_delay_s=.02, uncertainty_calibrated=False,
        same_state_action_pairs_per_seed=12,
        models={k: sha(output/f'{k}.pt') for k in ('action', 'no-action')},
        body_sha256=sha(BODY), bank_sha256=sha(BANK), segmentation_sha256=sha(SEGMENTATION),
        sources={name: sha(Path(__file__).with_name(name)) for name in
                 ('action_state.py', 'action_cycle.py', 'body_memory.py', 'body_fixture.py', 'recovery.py')},
        evidence='A learned action transition plus explicit shared-state filtering; not an end-to-end learned latent body'))


def final(output):
    protocol = read_json(output/'final-protocol.json')
    for name, digest in protocol['sources'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    for name, digest in protocol['models'].items():
        assert sha(output/f'{name}.pt') == digest
    assert sha(BODY) == protocol['body_sha256']
    assert sha(BANK) == protocol['bank_sha256']
    assert sha(SEGMENTATION) == protocol['segmentation_sha256']
    for seed in protocol['seeds']:
        source = collect_evaluation(output, seed)
        paired_actions(output, seed, protocol['same_state_action_pairs_per_seed'])
        paths = [infer(output, source, condition, protocol['config']) for condition in protocol['conditions']]
        score(output, source, paths)
    write_json(output/'complete.json', dict(completed_utc=utc(), seeds=protocol['seeds']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'develop', 'freeze', 'final'))
    parser.add_argument('--output', type=Path, default=ROOT)
    parser.add_argument('--gate', type=float, default=.014)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    if args.mode == 'freeze':
        freeze(args.output, args.gate)
    else:
        globals()[args.mode](args.output)
