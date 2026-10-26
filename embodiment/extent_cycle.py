"""E9 isolated acquisition, RGB atlas learning, frozen predictions and scoring."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import cv2
import mujoco
import numpy as np
import torch
from scipy.spatial import cKDTree

from perception.collect import HOME
from . import trust_cycle as e7
from .action_state import ActionTransition
from .body_cycle import read_json, write_json, sha, utc, stack
from .body_fixture import make_fixture, render_rgb, render_truth
from .body_memory import BodyModel
from .explore_run import AMPLITUDE
from .model import basis
from .reliable_state import JointReliability, ReliableBodyState
from .surface_extent import (SurfaceAtlas, patch_seeds, farthest_views, track_patch,
                             discover_surface, ridge_fit)
from .vision import match_stereo, project

ROOT = Path('outputs/local-check-20261024/e9-body-extent')
TRAIN_SOURCE = Path('outputs/local-check-20261024/e4-body-cycle/training-data')
E8 = Path('outputs/local-check-20261024/e8b-reliable-state')
DEV = 2026102911
FINAL = (2026102931, 2026102932, 2026102933)
CONDITIONS = ('clean', 'biased_feedback', 'wrong_rgb', 'bias_and_wrong_rgb', 'no_feedback', 'camera_missing')
CONFIG = dict(charts=6, training_stride=2, patch_spacing=6, tile_budget=60, ridge=.08,
              tolerance_m=.015, surface_limit_m=.020, minimum_dev_matches=8,
              dev_error_m=.025, dev_constant_ratio=.95, seed=2026102900)
SOURCES = ('surface_extent.py', 'extent_cycle.py')


def init(output):
    output.mkdir(parents=True, exist_ok=True)
    path = output/'protocol.json'
    if path.exists():
        return
    write_json(path, dict(created_utc=utc(), configuration=CONFIG, training_seed=2026102471,
        development_seed=DEV, final_seeds=FINAL, final_frames=150, conditions=CONDITIONS,
        training_source=str(TRAIN_SOURCE), training_roles='all 414 E4 frames reused as E9 training',
        input_permissions=['calibrated stereo RGB', 'recorded actual feedback', 'commands and time'],
        forbidden_training_inputs=['depth', 'body labels', 'simulator geometry', 'CAD', 'part identities', 'V7 masks'],
        primary_goals=dict(coverage_gain_over_sparse=.10, off_surface_regression_max=.02,
            mean_surface_distance_max_m=.020, multi_over_single_coverage=.03),
        scoring='all clear-left visible body pixels at stride 3, common 15mm center distance; full primitive union surface distance',
        scope='observed surface atlas, not full occupied volume; absent evidence is unknown'))
    for name, source in [('action', e7.ROOT/'action.pt'), ('gate', e7.ROOT/'gate.pt'), ('fixed', e7.ROOT/'fixed.pt')]:
        shutil.copyfile(source, output/f'{name}.pt')


def schedule(seed):
    rng = np.random.default_rng(seed)
    frequency, offset = rng.uniform(.6, 1.5, 7), rng.uniform(-np.pi, np.pi, 7)
    previous = HOME.copy()
    for i in range(150):
        if i < 10:
            target, phase = HOME.copy(), 'home'
        elif 70 <= i < 90:
            target, phase = previous.copy(), 'stopped'
        else:
            target = HOME+.70*AMPLITUDE*np.sin(i*.14*frequency+offset)
            target[0] = .22*np.sin((i-10)*.075+offset[0])
            phase = 'reappearance' if i >= 130 else 'crossing' if i >= 90 else 'moving'
        previous = target.copy()
        yield target.astype(np.float32), .10, phase


def collect(output, seed, panel=False):
    source = output/f'data-{seed}'
    if (source/'metadata.json').exists():
        return source
    source.mkdir(parents=True, exist_ok=True)
    if any(source.iterdir()):
        raise FileExistsError(f'Partial collection preserved: {source}')
    for folder in ('before', 'after', 'evaluation'):
        (source/folder).mkdir()
    model, data, calibration = make_fixture(source, panel)
    indices = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    props = [model.body(f'table_object_{i}').id for i in range(3)]
    fixed = data.xpos[props].copy()
    np.savez_compressed(source/'initial.npz', q=data.qpos[indices].astype(np.float32))
    renderer = mujoco.Renderer(model, height=480, width=640)
    rows, before_hashes = [], {}
    previous = HOME.copy()
    try:
        for i, (target, duration, phase) in enumerate(schedule(seed)):
            name = f'{i:05d}.npz'
            np.savez_compressed(source/'before'/name, target=target, duration=np.float32(duration))
            before_hashes[name] = sha(source/'before'/name)
            lag = .06 if 50 <= i < 75 or 110 <= i < 135 else 0.
            if lag:
                data.ctrl[:] = previous
                mujoco.mj_step(model, data, nstep=round(lag/model.opt.timestep))
            data.ctrl[:] = target
            mujoco.mj_step(model, data, nstep=round((duration-lag)/model.opt.timestep))
            mujoco.mj_forward(model, data)
            np.savez_compressed(source/'after'/name, q=data.qpos[indices].astype(np.float32).copy(),
                left=render_rgb(renderer, data), right=render_rgb(renderer, data, 'entity_right'), time=data.time)
            # Privileged data are written to evaluation, never used by train().
            truth = render_truth(renderer, model, data)
            clear = render_truth(renderer, model, data, panel_visible=False)
            body_ids = np.flatnonzero(model.geom_group == 1)
            camera = model.camera('fixed').id
            rotation = data.cam_xmat[camera].reshape(3, 3)
            optical = np.diag([1., -1., -1.])@rotation.T
            truth.update(clear_body_mask=clear['body_mask'], clear_depth=clear['depth'],
                body_positions=(data.geom_xpos[body_ids]-data.cam_xpos[camera])@optical.T,
                body_rotations=np.einsum('ij,njk->nik', optical, data.geom_xmat[body_ids].reshape(-1, 3, 3)),
                body_sizes=model.geom_size[body_ids], body_types=model.geom_type[body_ids])
            np.savez_compressed(source/'evaluation'/name, **truth)
            np.testing.assert_array_equal(fixed, data.xpos[props])
            rows.append(dict(file=name, phase=phase, duration=duration, actuator_delay_s=lag))
            previous = target.copy()
            if (i+1) % 30 == 0:
                print(f'E9 collect {seed}: {i+1}/150 panel={panel}', flush=True)
    finally:
        renderer.close()
    write_json(source/'metadata.json', dict(seed=seed, frames=rows, calibration=calibration,
        created_utc=utc(), panel=panel, prop_max_displacement_m=0., before_action_hashes=before_hashes))
    return source


def sensor_images(source, folder, stride=1):
    metadata = read_json(source/'metadata.json')
    records, grays, others, readings, hashes = [], [], [], [], {}
    for i, row in enumerate(metadata['frames']):
        if i % stride:
            continue
        path = source/folder/row['file']
        with np.load(path) as a:
            if 'camera_available' in a and not bool(a['camera_available']):
                continue
            grays.append(cv2.cvtColor(a['left'], cv2.COLOR_RGB2GRAY))
            others.append(cv2.cvtColor(a['right'], cv2.COLOR_RGB2GRAY))
            readings.append(a['q'].copy())
        hashes[row['file']] = sha(path)
        records.append(row['file'])
    return np.asarray(readings), np.asarray(grays), np.asarray(others), metadata, hashes


def extract(output):
    path = output/'training-tracks.npz'
    if path.exists():
        return path
    q, gray, other, metadata, hashes = sensor_images(TRAIN_SOURCE, 'observations', CONFIG['training_stride'])
    body = BodyModel(e7.BODY)
    center, scale = np.asarray(body.saved['center']), np.asarray(body.saved['scale'])
    references = farthest_views(q, center, scale, CONFIG['charts'])
    all_xyz, all_valid, all_uv, all_chart, all_refxyz = [], [], [], [], []
    for chart, index in enumerate(references):
        uv = patch_seeds(gray[index], CONFIG['patch_spacing'], CONFIG['tile_budget'])
        initial_xyz, good = match_stereo(gray[index], other[index], uv[:, None].copy(), metadata['calibration'], 'refined11')
        uv, initial_xyz = uv[good], initial_xyz[good]
        values, available = [], []
        for i in range(len(q)):
            xyz, valid, _ = track_patch(gray[index], gray[i], other[i], uv, initial_xyz, metadata['calibration'])
            values.append(xyz)
            available.append(valid)
        all_xyz.append(np.asarray(values))
        all_valid.append(np.asarray(available))
        all_uv.append(uv)
        all_refxyz.append(initial_xyz)
        all_chart.append(np.full(len(uv), chart))
        print(f'E9 chart {chart+1}/{len(references)}: reference={index} seeds={len(uv)}', flush=True)
    np.savez_compressed(path, q=q, xyz=np.concatenate(all_xyz, 1), valid=np.concatenate(all_valid, 1),
        uv=np.concatenate(all_uv), chart=np.concatenate(all_chart), reference_xyz=np.concatenate(all_refxyz),
        gray=gray[references], center=center, scale=scale, references=references,
        **{k:v for k,v in metadata['calibration'].items() if isinstance(v, (int, float))})
    write_json(output/'training-input-audit.json', dict(created_utc=utc(), input_hashes=hashes,
        training_source=str(TRAIN_SOURCE), frames=len(q), references=references.tolist(),
        cache_sha256=sha(path), evaluation_files_read=False, body_masks_used=False))
    return path


def train(output):
    init(output)
    if (output/'training-freeze.json').exists():
        raise FileExistsError('Preserving frozen E9 atlas')
    dev = collect(output, DEV, panel=False)
    path = extract(output)
    a = e7.load_npz(path)
    selected, weights, diagnostics = discover_surface(a['q'], a['xyz'], a['valid'], a['center'], a['scale'], CONFIG['ridge'])
    write_json(output/'discovery.json', dict(candidates=diagnostics, training_selected=selected.tolist()))
    print('E9 training motion-selected patches:', len(selected), flush=True)
    if not len(selected):
        raise ValueError('No motion-consistent surface evidence; preserve failure')
    dq, dg, dr, metadata, dhashes = sensor_images(dev, 'after', 2)
    features = basis((dq-a['center'])/a['scale'])
    keep, dev_rows = [], []
    for chart in range(CONFIG['charts']):
        subset = np.flatnonzero(a['chart'][selected] == chart)
        ids = selected[subset]
        values, valid = [], []
        for i in range(len(dq)):
            prediction = np.einsum('k,nkc->nc', features[i], weights[subset])
            xyz, good, _ = track_patch(a['gray'][chart], dg[i], dr[i], a['uv'][ids], prediction,
                metadata['calibration'], project(prediction, metadata['calibration']))
            values.append(xyz)
            valid.append(good)
        values, valid = np.asarray(values), np.asarray(valid)
        for j, k in enumerate(subset):
            good = valid[:, j]
            predicted = features@weights[k]
            count = int(good.sum())
            error = float(np.sqrt(np.mean(np.sum((predicted[good]-values[good, j])**2, -1)))) if count else 1e6
            train_good = a['valid'][:, selected[k]]
            mean = a['xyz'][train_good, selected[k]].mean(0)
            constant = float(np.sqrt(np.mean(np.sum((values[good, j]-mean)**2, -1)))) if count else 0.
            accepted = count >= CONFIG['minimum_dev_matches'] and error < CONFIG['dev_error_m'] and error < CONFIG['dev_constant_ratio']*constant
            dev_rows.append(dict(id=int(selected[k]), chart=chart, matches=count, error_m=error,
                                 constant_m=constant, accepted=bool(accepted)))
            if accepted:
                keep.append(int(k))
        print(f'E9 dev chart {chart}: kept total={len(keep)}', flush=True)
    keep = np.asarray(sorted(keep), int)
    selected, weights = selected[keep], weights[keep]
    if not len(selected):
        raise ValueError('Independent development rejects all patches; preserve failure')
    common = dict(center=a['center'], scale=a['scale'], training_q=a['q'], gray=a['gray'],
                  **{k:a[k] for k in ('focal_px', 'baseline_m', 'cx', 'cy')})
    for name, mask in [('multi', np.ones(len(selected), bool)), ('single', a['chart'][selected] == 0)]:
        ids = selected[mask]
        np.savez_compressed(output/f'atlas-{name}.npz', **common, weights=weights[mask],
            uv=a['uv'][ids], chart=a['chart'][ids], reference_xyz=a['reference_xyz'][ids],
            training_candidate_id=ids)
    write_json(output/'development.json', dict(seed=DEV, rows=dev_rows, input_hashes=dhashes,
        selected_count=len(selected), counts_by_chart=np.bincount(a['chart'][selected], minlength=CONFIG['charts']).tolist(),
        evaluation_geometry_used=False))
    write_json(output/'training-freeze.json', dict(created_utc=utc(), configuration=CONFIG,
        training_seed=2026102471, development_seed=DEV, final_seeds=FINAL,
        selected_count=len(selected), model_hashes={k:sha(output/f'atlas-{k}.npz') for k in ('multi', 'single')},
        sources={name:sha(Path(__file__).with_name(name)) for name in SOURCES},
        dependencies={str(p):sha(p) for p in (e7.BODY, E8/'per_joint.pt', E8/'training-freeze.json',
            e7.ROOT/'action.pt', e7.ROOT/'gate.pt', e7.ROOT/'fixed.pt')},
        geometry_used=False, segmentation_used=False))
    print('E9 FROZEN', len(selected), 'surface patches', flush=True)


def verify(output):
    freeze = read_json(output/'training-freeze.json')
    for name, digest in freeze['sources'].items():
        assert sha(Path(__file__).with_name(name)) == digest, name
    for name, digest in freeze['model_hashes'].items():
        assert sha(output/f'atlas-{name}.npz') == digest, name
    for name, digest in freeze['dependencies'].items():
        assert sha(Path(name)) == digest, name
    return freeze


def predict(output, seed, condition):
    destination = output/'extent-predictions'/f'{seed}-{condition}.npz'
    if destination.exists():
        return destination
    destination.parent.mkdir(exist_ok=True)
    source = output/f'data-{seed}'
    cache = e7.visual_cache(output, source)
    stream = e7.sensor_stream(source, cache, condition, training=False)
    initial = e7.load_npz(source/'initial.npz')['q']
    body = BodyModel(e7.BODY)
    transition = ActionTransition.load(output/'action.pt')
    state = ReliableBodyState(transition, JointReliability.load(E8/'per_joint.pt'), body, initial)
    atlas, single = SurfaceAtlas.load(output/'atlas-multi.npz'), SurfaceAtlas.load(output/'atlas-single.npz')
    # E7 baseline predictions use exactly the same sensor proposals.
    e7path = output/'predictions'/f'data-{seed}-{condition}.npz'
    if not e7path.exists():
        e7.predict(output, seed, condition)
    baseline = e7.load_npz(e7path)
    rows = []
    metadata = read_json(source/'metadata.json')
    # E7's q for each baseline is reconstructed from saved joint estimates below.
    old_q, old_v = {k:initial.copy() for k in ('action', 'feedback', 'fixed')}, {k:np.zeros(7) for k in ('action', 'feedback', 'fixed')}
    for i, frame in enumerate(metadata['frames']):
        target, dt = stream['target'][i], float(stream['duration'][i])
        state.forecast(target, dt)
        f_ok, v_ok = stream['available'][i]
        result = state.finish(stream['feedback'][i] if f_ok else None, stream['visual'][i], stream['quality'][i], v_ok)
        q = result['q']
        action_q = transition.predict(old_q['action'], old_v['action'], target, dt)
        feedback_q = stream['feedback'][i] if f_ok else transition.predict(old_q['feedback'], old_v['feedback'], target, dt)
        prior = transition.predict(old_q['fixed'], old_v['fixed'], target, dt)
        candidates = [prior]
        if f_ok:
            candidates.append(stream['feedback'][i])
        if v_ok:
            candidates.append(stream['visual'][i])
        fixed_q = np.mean(candidates, axis=0)
        for name, value in [('action', action_q), ('feedback', feedback_q), ('fixed', fixed_q)]:
            old_v[name] = np.clip((value-old_q[name])/dt, -transition.scale.numpy()*20, transition.scale.numpy()*20)
            old_q[name] = value.copy()
        # E7 saves q_<mode> in addition to sparse coordinates.
        e7_q = baseline['q_learned'][i]
        obs = e7.load_npz(source/'after'/frame['file'])
        # Current evidence uses the actual sensor condition, including past RGB.
        rgb = obs
        if 'wrong_rgb' in condition and i >= 30:
            rgb = e7.load_npz(source/'after'/metadata['frames'][i % 20]['file'])
        rgb_available = not ('camera_missing' in condition and 40 <= i < 95)
        evidence = atlas.evidence(q, rgb['left'] if rgb_available else None,
                                  rgb['right'] if rgb_available else None)
        matched_count = len(single.weights)
        matched_ids = np.linspace(0, len(atlas.weights)-1, matched_count).astype(int)
        row = dict(q_e8=q, q_e7=e7_q, q_feedback=feedback_q, q_action=action_q, q_fixed=fixed_q,
            multi=evidence['xyz'], status=evidence['status'], observed_xyz=evidence['observed_xyz'],
            single=single.predict(q), budget_matched=evidence['xyz'][matched_ids],
            sparse=body(q)[0], e7=atlas.predict(e7_q),
            feedback=atlas.predict(feedback_q), action=atlas.predict(action_q), fixed=atlas.predict(fixed_q),
            static=atlas.predict(initial), mismatch=atlas.predict(2*HOME-q),
            pose_distance=evidence['pose_distance'], available=stream['available'][i])
        rows.append(row)
    np.savez_compressed(destination, **stack(rows))
    write_json(destination.with_suffix('.json'), dict(created_utc=utc(), sha256=sha(destination),
        geometry_read=False, fault_label_as_model_input=False, frames=len(rows)))
    print(f'E9 predicted {seed} {condition}', flush=True)
    return destination


def primitive_surface_distance(points, truth):
    """Privileged approximate signed distance to union of exact box/cylinder solids."""
    distances = []
    for position, rotation, size, kind in zip(truth['body_positions'], truth['body_rotations'], truth['body_sizes'], truth['body_types']):
        local = (points-position)@rotation
        if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
            d = np.abs(local)-size
            signed = np.linalg.norm(np.maximum(d, 0), axis=-1)+np.minimum(np.max(d, axis=-1), 0)
        elif kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            d = np.c_[np.linalg.norm(local[:, :2], axis=-1)-size[0], np.abs(local[:, 2])-size[1]]
            signed = np.linalg.norm(np.maximum(d, 0), axis=-1)+np.minimum(np.max(d, axis=-1), 0)
        else:
            raise ValueError(f'Unimplemented evaluation primitive {kind}')
        distances.append(signed)
    signed = np.min(distances, axis=0)
    return np.abs(signed), signed


def scoring(output):
    verify(output)
    manifest = read_json(output/'predictions-complete.json')
    for path, digest in manifest['hashes'].items():
        assert sha(Path(path)) == digest
    modes = ('sparse', 'single', 'budget_matched', 'multi', 'static', 'mismatch', 'e7', 'feedback', 'action', 'fixed')
    all_rows = []
    for seed in FINAL:
        source = output/f'data-{seed}'
        metadata = read_json(source/'metadata.json')
        cal = metadata['calibration']
        for condition in CONDITIONS:
            prediction = e7.load_npz(output/'extent-predictions'/f'{seed}-{condition}.npz')
            for i, frame in enumerate(metadata['frames']):
                truth = e7.load_npz(source/'evaluation'/frame['file'])
                vv, uu = np.mgrid[0:480:3, 0:640:3]
                use = truth['clear_body_mask'][vv, uu]
                u, v = uu[use], vv[use]
                z = truth['clear_depth'][v, u]
                targets = np.c_[(u-cal['cx'])*z/cal['focal_px'], (v-cal['cy'])*z/cal['focal_px'], z]
                hidden = truth['panel_mask'][v, u] & (np.abs(truth['depth'][v, u]-z) > .015)
                for mode in modes:
                    points = prediction[mode][i]
                    if not len(points):
                        raise ValueError(f'Empty final representation: {mode}')
                    assert np.isfinite(points).all()
                    distance = cKDTree(points).query(targets)[0]
                    surface, signed = primitive_surface_distance(points, truth)
                    covered = distance <= CONFIG['tolerance_m']
                    uv = np.rint(project(points, cal)).astype(int)
                    inside = (uv[:, 0] >= 0) & (uv[:, 0] < 640) & (uv[:, 1] >= 0) & (uv[:, 1] < 480)
                    ids = np.flatnonzero(inside)
                    onprop = np.zeros(len(points), bool)
                    if len(ids):
                        pu, pv = uv[ids].T
                        onprop[ids] = truth['prop_mask'][pv, pu] & (np.abs(points[ids, 2]-truth['depth'][pv, pu]) < .015)
                    row = dict(seed=seed, condition=condition, frame=i, phase=frame['phase'], mode=mode,
                        target_count=len(targets), covered_count=int(covered.sum()),
                        hidden_count=int(hidden.sum()), hidden_covered=int((covered & hidden).sum()),
                        visible_count=int((~hidden).sum()), visible_covered=int((covered & ~hidden).sum()),
                        point_count=len(points), surface_distance_sum_m=float(surface.sum()),
                        off_surface_count=int((surface > .015).sum()), outside_count=int((signed > .015).sum()),
                        prop_point_count=int(onprop.sum()), target_distance_sum_m=float(distance.sum()),
                        observed_count=int((prediction['status'][i] == 2).sum()) if mode == 'multi' else 0,
                        unknown_support_count=int((prediction['status'][i] == 0).sum()) if mode == 'multi' else 0)
                    all_rows.append(row)
            print(f'E9 scored {seed} {condition}', flush=True)
    write_json(output/'frame-scores.json', all_rows)
    write_json(output/'complete.json', dict(created_utc=utc(), final_seeds=FINAL,
        predictions_frozen_before_geometry_reads=True, score_rows=len(all_rows)))


def final(output):
    verify(output)
    if (output/'complete.json').exists():
        raise FileExistsError('E9 final complete; preserve frozen results')
    for seed in FINAL:
        collect(output, seed, panel=True)
        for condition in CONDITIONS:
            predict(output, seed, condition)
    hashes = {str(p):sha(p) for p in sorted((output/'extent-predictions').glob('*.npz'))}
    write_json(output/'predictions-complete.json', dict(created_utc=utc(), hashes=hashes,
        all_final_trajectories_saved=True, geometry_not_scored_yet=True))
    scoring(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('train', 'final', 'score'))
    parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    globals()['scoring' if args.mode == 'score' else args.mode](args.output)
