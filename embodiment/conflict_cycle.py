"""E10 frozen-model conflict lifecycle experiment. Run prepare, final, score.

Sensor emulation, inference and privileged evaluation are separate stages.
Final predictions for ALL conditions are sealed before scoring opens truth.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import mujoco
import numpy as np
import torch

from perception.collect import HOME
from . import trust_cycle as e7
from . import reliable_cycle_v2 as e8
from . import extent_cycle_v2 as e9
from .action_state import ActionTransition
from .body_cycle import reference, read_json, write_json, sha, utc, stack
from .body_fixture import make_fixture, render_rgb, render_truth, EvaluationLandmarks
from .body_memory import BodyModel
from .conflict_state import pair_distances, ConsistencyAudit, interval_envelope, recovery_event
from .explore_run import AMPLITUDE
from .reliable_state import JointReliability, ReliableBodyState
from .surface_extent import SurfaceAtlas

ROOT = Path('outputs/local-check-20261024/e10-body-conflict')
DEV = 2026103011
FINAL = (2026103031, 2026103032, 2026103033)
CONDITIONS = ('clean', 'feedback_bias', 'stale_rgb', 'both', 'no_feedback', 'no_camera', 'coherent_replay')
MODES = ('full', 'no_visual', 'no_feedback', 'action', 'feedback', 'fixed_equal')
AUDITS = ('body', 'joint', 'body_instant')
WINDOWS = ((30, 70), (110, 140))
RETURNS = ((70, 110), (140, 180))
SOURCES = ('conflict_cycle.py', 'conflict_state.py')
BIAS = np.array([-.075, .028, .04, -.035, .025, .03, -.0015], np.float32)


def active_mask():
    mask = np.zeros(180, bool)
    for start, stop in WINDOWS:
        mask[start:stop] = True
    return mask


def schedule(seed):
    rng = np.random.default_rng(seed)
    frequency = rng.uniform(.55, 1.35, 7)
    offset = rng.uniform(-np.pi, np.pi, 7)
    previous = HOME.copy()
    for i in range(180):
        if i < 10:
            target, phase = HOME.copy(), 'initial'
        elif 80 <= i < 100:
            target, phase = previous.copy(), 'stopped'
        else:
            target = HOME+.68*AMPLITUDE*np.sin(i*.15*frequency+offset)
            target[0] = .22*np.sin(i*.075+offset[0])
            phase = 'moving'
        previous = target.copy()
        yield target.astype(np.float32), .1, phase


def collect(output, seed):
    source = output/f'data-{seed}'
    if (source/'metadata.json').exists():
        return source
    source.mkdir(parents=True, exist_ok=True)
    if any(source.iterdir()):
        raise FileExistsError(f'Partial collection preserved: {source}')
    for name in ('before', 'after', 'evaluation'):
        (source/name).mkdir()
    model, data, calibration = make_fixture(source, panel=True)
    saved, _, pixels = reference()
    anchors = EvaluationLandmarks(model, data, pixels[saved['selected'], 0], calibration)
    indices = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    props = [model.body(f'table_object_{i}').id for i in range(3)]
    fixed = data.xpos[props].copy()
    np.savez_compressed(source/'initial.npz', q=data.qpos[indices].astype(np.float32))
    renderer = mujoco.Renderer(model, height=480, width=640)
    rows, hashes = [], {}
    previous = HOME.copy()
    try:
        for i, (target, duration, phase) in enumerate(schedule(seed)):
            name = f'{i:05d}.npz'
            np.savez_compressed(source/'before'/name, target=target, duration=np.float32(duration))
            hashes[name] = sha(source/'before'/name)
            lag = .06 if 50 <= i < 75 or 110 <= i < 135 else 0.
            if lag:
                data.ctrl[:] = previous
                mujoco.mj_step(model, data, nstep=round(lag/model.opt.timestep))
            data.ctrl[:] = target
            mujoco.mj_step(model, data, nstep=round((duration-lag)/model.opt.timestep))
            mujoco.mj_forward(model, data)
            q = data.qpos[indices].astype(np.float32).copy()
            np.savez_compressed(source/'after'/name, q=q, left=render_rgb(renderer, data),
                                right=render_rgb(renderer, data, 'entity_right'), time=data.time)
            truth = render_truth(renderer, model, data)
            truth.update(anchors.score_truth(model, data, truth), actual_q=q)
            body_ids = np.flatnonzero(model.geom_group == 1)
            camera = model.camera('fixed').id
            optical = np.diag([1., -1., -1.])@data.cam_xmat[camera].reshape(3, 3).T
            truth.update(body_positions=(data.geom_xpos[body_ids]-data.cam_xpos[camera])@optical.T,
                body_rotations=np.einsum('ij,njk->nik', optical, data.geom_xmat[body_ids].reshape(-1, 3, 3)),
                body_sizes=model.geom_size[body_ids], body_types=model.geom_type[body_ids])
            np.savez_compressed(source/'evaluation'/name, **truth)
            np.testing.assert_array_equal(fixed, data.xpos[props])
            rows.append(dict(file=name, phase=phase, duration=duration, actuator_delay_s=lag))
            previous = target.copy()
            if (i+1) % 30 == 0:
                print(f'E10 collect {seed}: {i+1}/180', flush=True)
    finally:
        renderer.close()
    write_json(source/'metadata.json', dict(seed=seed, frames=rows, calibration=calibration,
        created_utc=utc(), prop_max_displacement_m=0., fixed_body_points=int(anchors.is_body.sum()),
        before_action_hashes=hashes, evaluation_isolated=True))
    return source


def sensors(output, seed):
    """Sensor emulator only. Its clean readings are NOT inference teachers."""
    source = output/f'data-{seed}'
    cache = e7.visual_cache(output, source)
    a = e7.load_npz(cache)
    metadata = read_json(source/'metadata.json')
    readings = np.stack([e7.load_npz(source/'after'/row['file'])['q'] for row in metadata['frames']])
    mask = active_mask()
    destination = output/'sensors'
    destination.mkdir(exist_ok=True)
    for condition in CONDITIONS:
        path = destination/f'{seed}-{condition}.npz'
        if path.exists():
            continue
        feedback = readings.copy()
        wrong = mask & (condition in ('stale_rgb', 'both', 'coherent_replay'))
        if condition in ('feedback_bias', 'both'):
            feedback[mask] += BIAS
        if condition == 'coherent_replay':
            ids = np.flatnonzero(mask)
            feedback[ids] = readings[ids % 20]
        visual = np.where(wrong[:, None], a['visual_wrong'], a['visual_clean'])
        quality = np.where(wrong[:, None], a['quality_wrong'], a['quality_clean'])
        valid = np.where(wrong, a['valid_wrong'], a['valid_clean'])
        f_ok = np.ones(len(mask), bool)
        if condition == 'no_feedback':
            f_ok[mask] = False
        if condition == 'no_camera':
            valid[mask] = False
        available = np.stack((f_ok, valid), -1)
        # Missing values are removed before any inference module sees the file.
        np.savez_compressed(path, target=a['target'], duration=a['duration'],
            feedback=np.where(f_ok[:, None], feedback, a['action_q']),
            visual=np.where(valid[:, None], visual, a['action_q']),
            quality=np.where(valid[:, None], quality, 0.), available=available,
            action_reference=a['action_q'])
    return cache


def candidates(stream, i):
    return np.stack((stream['action_reference'][i], stream['feedback'][i], stream['visual'][i]))


def calibrate(output):
    source = collect(output, DEV)
    sensors(output, DEV)
    a = e7.load_npz(output/'sensors'/f'{DEV}-clean.npz')
    atlas = SurfaceAtlas.load(e9.ROOT/'atlas-multi.npz')
    scale = np.asarray(BodyModel(e7.BODY).saved['scale'])
    distances = {k:[] for k in ('body', 'joint')}
    availability = []
    for i in range(len(a['duration'])):
        q = candidates(a, i)
        distances['body'].append(pair_distances(q, decoder=atlas))
        distances['joint'].append(pair_distances(q, scale=scale))
        f, v = a['available'][i]
        availability.append([bool(f), bool(v), bool(f and v)])
    available = np.asarray(availability)
    thresholds, counts = {}, {}
    for key, rows in distances.items():
        d = np.asarray(rows)
        thresholds[key] = [float(max(1e-6, np.quantile(d[available[:, j], j], .99))) for j in range(3)]
        counts[key] = available.sum(0).tolist()
    old = read_json(e8.ROOT/'calibration.json')['profiles']['per_joint']['factor']
    write_json(output/'audit-calibration.json', dict(created_utc=utc(), development_seed=DEV,
        quantile=.99, thresholds=thresholds, counts=counts, e8_sigma_factor=old,
        clean_only=True, geometry_truth_used=False, smoothing='2 of trailing 3 frames'))
    return source


def prepare(output):
    if (output/'protocol-freeze.json').exists():
        raise FileExistsError('Experiment already frozen; preserve it')
    output.mkdir(parents=True, exist_ok=True)
    if not (output/'action.pt').exists():
        shutil.copyfile(e7.ROOT/'action.pt', output/'action.pt')
    e8.verify_freeze(e8.ROOT)
    e9.verify(e9.ROOT)
    calibrate(output)
    dependencies = (e7.BODY, e7.BANK, e7.SEGMENTATION, e8.ROOT/'per_joint.pt',
        e8.ROOT/'calibration.json', e9.ROOT/'atlas-multi.npz', output/'action.pt')
    # Transitive inference sources are frozen too, not merely the new runner.
    names = (*SOURCES, 'reliable_state.py', 'action_state.py', 'body_memory.py',
             'trust_cycle.py', 'recovery.py', 'surface_extent.py', 'body_fixture.py')
    write_json(output/'protocol-freeze.json', dict(created_utc=utc(), development_seed=DEV,
        final_seeds=FINAL, conditions=CONDITIONS, frames=180, models_retrained=False,
        source_digests={name:sha(Path(__file__).with_name(name)) for name in names},
        dependencies={str(p):sha(p) for p in dependencies},
        protocol_sha256=sha('docs/body-conflict-e10.md'), calibration_sha256=sha(output/'audit-calibration.json'),
        fault_windows=WINDOWS, recovery_windows=RETURNS, priors='clean initial q, known channels and stereo calibration',
        goals=dict(fault_rmse_improvement_over_equal=.10, clean_guard_mm=.5,
            interval_coverage_min=.85, outlier_precision_min=.8, geometry_f1_gain=.05,
            clean_false_alarm_max=.05, recovered_events_min=5, recovery_median_max_s=1.)))
    print('E10 PROTOCOL FROZEN before final collection', flush=True)


def verify(output):
    frozen = read_json(output/'protocol-freeze.json')
    for name, digest in frozen['source_digests'].items():
        assert sha(Path(__file__).with_name(name)) == digest, name
    for path, digest in frozen['dependencies'].items():
        assert sha(path) == digest, path
    assert sha('docs/body-conflict-e10.md') == frozen['protocol_sha256']
    assert sha(output/'audit-calibration.json') == frozen['calibration_sha256']
    return frozen


def predict(output, seed, condition):
    path = output/'predictions'/f'{seed}-{condition}.npz'
    if path.exists():
        return path
    path.parent.mkdir(exist_ok=True)
    stream_path = output/'sensors'/f'{seed}-{condition}.npz'
    a = e7.load_npz(stream_path)
    initial = e7.load_npz(output/f'data-{seed}'/'initial.npz')['q']
    body, transition = BodyModel(e7.BODY), ActionTransition.load(output/'action.pt')
    atlas = SurfaceAtlas.load(e9.ROOT/'atlas-multi.npz')
    scale = np.asarray(body.saved['scale'])
    calibration = read_json(output/'audit-calibration.json')
    states = {name:ReliableBodyState(transition, JointReliability.load(e8.ROOT/'per_joint.pt'), body, initial)
              for name in ('full', 'no_visual', 'no_feedback')}
    q_old = {name:initial.copy() for name in ('action', 'feedback', 'fixed_equal')}
    velocity = {name:np.zeros(7, np.float32) for name in q_old}
    audits = {name:ConsistencyAudit(calibration['thresholds']['joint' if name == 'joint' else 'body'],
                                   smoothing=name != 'body_instant') for name in AUDITS}
    rows = []
    for i in range(len(a['duration'])):
        target, dt = a['target'][i], float(a['duration'][i])
        # Forecast occurs before touching this step's observation arrays.
        for state in states.values():
            state.forecast(target, dt)
        priors = {name:transition.predict(q_old[name], velocity[name], target, dt) for name in q_old}
        q_prior_full = states['full'].pending[0].copy()
        f_ok, v_ok = a['available'][i]
        row = dict(prior_full=q_prior_full, available=a['available'][i])
        for name, state in states.items():
            result = state.finish(a['feedback'][i] if f_ok and name != 'no_feedback' else None,
                a['visual'][i], a['quality'][i], bool(v_ok and name != 'no_visual'))
            row[f'q_{name}'] = result['q']
            if name == 'full':
                row.update(weights=result['weights'], sigma=result['sigma_normalized'],
                           half_width=result['sigma_normalized']*scale*calibration['e8_sigma_factor'])
        q_simple = dict(action=priors['action'], feedback=a['feedback'][i] if f_ok else priors['feedback'])
        options = [priors['fixed_equal']]
        if f_ok:
            options.append(a['feedback'][i])
        if v_ok:
            options.append(a['visual'][i])
        q_simple['fixed_equal'] = np.mean(options, axis=0)
        for name, q in q_simple.items():
            row[f'q_{name}'] = q.copy()
            velocity[name] = np.clip((q-q_old[name])/dt, -scale*20, scale*20)
            q_old[name] = q.copy()
        q_candidates = candidates(a, i)
        available = np.array([True, f_ok, v_ok])
        db = pair_distances(q_candidates, decoder=atlas)
        dj = pair_distances(q_candidates, scale=scale)
        row.update(candidates=q_candidates, candidate_available=available,
                   distances_body=db, distances_joint=dj)
        for name, audit in audits.items():
            result = audit.update(dj if name == 'joint' else db, available)
            row.update({f'{k}_{name}':v for k,v in result.items()})
        row['envelope_low'], row['envelope_high'] = interval_envelope(row['q_full'], row['half_width'],
            q_candidates, available, row['conflict_body'])
        rows.append(row)
    np.savez_compressed(path, **stack(rows))
    write_json(path.with_suffix('.json'), dict(created_utc=utc(), sha256=sha(path),
        sensor_file_sha256=sha(stream_path), inference_truth_read=False,
        condition_as_model_input=False, audit_changes_estimate=False, forecasts_before_observations=True))
    print(f'E10 predicted {seed} {condition}', flush=True)
    return path


def final(output):
    verify(output)
    if (output/'predictions-complete.json').exists():
        raise FileExistsError('Final predictions already sealed')
    for seed in FINAL:
        collect(output, seed)
        sensors(output, seed)
        for condition in CONDITIONS:
            predict(output, seed, condition)
    paths = sorted((output/'predictions').glob('*.npz'))
    write_json(output/'predictions-complete.json', dict(created_utc=utc(), seeds=FINAL,
        prediction_hashes={str(p):sha(p) for p in paths}, predictions_before_scoring=True))
    print('E10 all predictions SEALED before privileged scoring', flush=True)


def binary_metrics(predicted, target):
    predicted, target = np.asarray(predicted, bool), np.asarray(target, bool)
    tp, fp, fn, tn = (int(x.sum()) for x in (predicted & target, predicted & ~target,
                                           ~predicted & target, ~predicted & ~target))
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=tp/max(1, tp+fp),
                recall=tp/max(1, tp+fn), f1=2*tp/max(1, 2*tp+fp+fn))


def score(output):
    verify(output)
    if (output/'scores.json').exists():
        raise FileExistsError('Preserving final scores')
    sealed = read_json(output/'predictions-complete.json')
    assert len(sealed['prediction_hashes']) == len(FINAL)*len(CONDITIONS)
    for name, digest in sealed['prediction_hashes'].items():
        assert sha(name) == digest, name
    body = BodyModel(e7.BODY)
    atlas = SurfaceAtlas.load(e9.ROOT/'atlas-multi.npz')
    scale = np.asarray(body.saved['scale'])
    frame_rows, events, summary = [], [], []
    mask = active_mask()
    for seed in FINAL:
        source = output/f'data-{seed}'
        metadata = read_json(source/'metadata.json')
        truth = [e7.load_npz(source/'evaluation'/r['file']) for r in metadata['frames']]
        actual_q = np.stack([x['actual_q'] for x in truth])
        actual_body = body(actual_q)
        actual_atlas = np.stack([atlas.predict(q) for q in actual_q])
        anchor = np.stack([x['anchor_xyz'] for x in truth])
        anchors = truth[0]['anchor_body'].astype(bool)
        clean = e7.load_npz(output/'predictions'/f'{seed}-clean.npz')
        clean_errors = {mode:np.sqrt(np.mean(np.sum((body(clean[f'q_{mode}'])[:, anchors]-anchor[:, anchors])**2, -1), -1))
                        for mode in MODES}
        for condition in CONDITIONS:
            p = e7.load_npz(output/'predictions'/f'{seed}-{condition}.npz')
            candidate_points = np.stack([[atlas.predict(q) for q in row] for row in p['candidates']])
            candidate_error = np.sqrt(np.mean(np.sum((candidate_points-actual_atlas[:, None])**2, -1), -1))
            bad_candidates = (candidate_error > .008) & p['candidate_available']
            # Unavailability is a sensor intervention but not a visible conflict.
            target_fault = mask & (condition != 'clean')
            for mode in MODES:
                predicted = body(p[f'q_{mode}'])
                error = np.sqrt(np.mean(np.sum((predicted[:, anchors]-anchor[:, anchors])**2, -1), -1))
                separation = np.sqrt(np.mean(np.sum((predicted-body(clean[f'q_{mode}']))**2, -1), -1))
                q_error = np.max(np.abs((p[f'q_{mode}']-actual_q)/scale), -1)
                surface_points = [atlas.predict(q) for q in p[f'q_{mode}']]
                outside = [float(np.mean(e9.primitive_surface_distance(x, t)[0] > .015))
                           for x, t in zip(surface_points, truth)]
                joint_covered = np.all(np.abs(p['q_full']-actual_q) <= p['half_width'], -1)
                envelope_covered = np.all((actual_q >= p['envelope_low']) & (actual_q <= p['envelope_high']), -1)
                width = np.mean(p['half_width']/scale, -1)
                envelope_width = np.mean((p['envelope_high']-p['envelope_low'])/(2*scale), -1)
                item = dict(seed=seed, condition=condition, mode=mode,
                    rmse_mm=float(np.sqrt(np.mean(error**2))*1000),
                    fault_rmse_mm=float(np.sqrt(np.mean(error[mask]**2))*1000),
                    return_rmse_mm=float(np.sqrt(np.mean(error[~mask & (np.arange(180)>=70)]**2))*1000),
                    off_surface_rate=float(np.mean(outside)), anchors=int(anchors.sum()),
                    joint_interval_coverage=float(joint_covered.mean()) if mode == 'full' else None,
                    fault_joint_coverage=float(joint_covered[mask].mean()) if mode == 'full' else None,
                    envelope_coverage=float(envelope_covered.mean()) if mode == 'full' else None,
                    half_width_normalized=float(width.mean()) if mode == 'full' else None,
                    envelope_half_width_normalized=float(envelope_width.mean()) if mode == 'full' else None)
                summary.append(item)
                for i in range(180):
                    row = dict(seed=seed, condition=condition, mode=mode, frame=i,
                        phase=metadata['frames'][i]['phase'], fault_window=bool(target_fault[i]),
                        execution_lag=bool(metadata['frames'][i]['actuator_delay_s']),
                        anchor_rmse_mm=float(error[i]*1000), separation_mm=float(separation[i]*1000),
                        state_error_max_normalized=float(q_error[i]), off_surface_rate=outside[i])
                    if mode == 'full':
                        row.update(joint_covered=bool(joint_covered[i]), envelope_covered=bool(envelope_covered[i]),
                            half_width_normalized=float(width[i]), envelope_half_width_normalized=float(envelope_width[i]),
                            candidate_error_m=candidate_error[i].tolist(), bad_candidates=bad_candidates[i].tolist(),
                            candidate_available=p['candidate_available'][i].tolist(),
                            weights=p['weights'][i].mean(0).tolist())
                        for audit in AUDITS:
                            row.update({f'{name}_{audit}':p[f'{name}_{audit}'][i].item()
                                        for name in ('status', 'conflict', 'unknown')})
                    frame_rows.append(row)
                if condition != 'clean':
                    for start, stop in RETURNS:
                        event = recovery_event(separation, error-clean_errors[mode], start, stop)
                        event.update(seed=seed, condition=condition, mode=mode)
                        events.append(event)
            print(f'E10 scored {seed} {condition}', flush=True)
    write_json(output/'frame-scores.json', frame_rows)
    write_json(output/'recovery-events.json', events)
    write_json(output/'scores.json', dict(created_utc=utc(), rows=summary, final_seeds=FINAL,
        prediction_seal_sha256=sha(output/'predictions-complete.json'), truth_read_after_seal=True,
        evaluation_frame_rows=len(frame_rows), recovery_events=len(events)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'final', 'score', 'verify'))
    parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args()
    torch.set_num_threads(4)
    globals()[args.stage](args.output)


if __name__ == '__main__':
    main()
