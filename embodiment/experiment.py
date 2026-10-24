"""Fit/freeze the entity using observations, then run separated evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch

from .vision import StereoTracks
from .model import discover, train_entity, SpatialEntity, EntityState


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def fit(data, output):
    data, output = Path(data), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'model.pt').exists():
        raise FileExistsError('Preserving completed model; select a new output directory')
    metadata = json.loads((data/'metadata.json').read_text(encoding='utf-8'))
    fingerprint = hashlib.sha256((data/'metadata.json').read_bytes()
                                  + Path(__file__).with_name('vision.py').read_bytes()
                                  + str(data.resolve()).encode()).hexdigest()
    with np.load(data/'observations'/metadata['frames'][0]['file']) as first:
        tracker = StereoTracks(first['left'], metadata['calibration'])
    cache_path = output/'training-tracks.npz'
    if cache_path.exists():
        with np.load(cache_path) as cache:
            if 'fingerprint' not in cache or str(cache['fingerprint']) != fingerprint:
                raise ValueError('Training track cache does not match this data/vision version; use a new output')
            arrays = {key: cache[key] for key in cache.files}
    else:
        collected = []
        for index, row in enumerate(metadata['frames']):
            if row['phase'] not in ('train', 'val'):
                continue
            with np.load(data/'observations'/row['file']) as observation:
                assert bool(observation['camera_available'])
                tracks = tracker.observe(observation['left'], observation['right'])
                collected.append({**tracks, 'q': observation['q'], 'index': index})
            if index % 50 == 0:
                print(f'Training observations {index+1}: {int(tracks["valid"].sum())} stereo features', flush=True)
        arrays = {key: np.stack([row[key] for row in collected]) for key in collected[0]}
        arrays['previous_points'] = tracker.previous_points
        arrays['reference_points'] = tracker.reference_points
        arrays['fingerprint'] = np.array(fingerprint)
        np.savez_compressed(cache_path, **arrays)
    phases = np.array([metadata['frames'][int(i)]['phase'] for i in arrays['index']])
    train, val = phases == 'train', phases == 'val'
    selected, center, scale, diagnostic = discover(arrays['q'], arrays['xyz'], arrays['valid'], train, val)
    write_json(output/'discovery.json', dict(selected=selected.tolist(), candidates=len(diagnostic),
                                           per_feature=diagnostic,
                                           selection='Train/validation measured q and stereo RGB tracks only'))
    print(f'Discovered {len(selected)} of {len(diagnostic)} features', flush=True)
    model, history = train_entity(arrays['q'], arrays['xyz'], arrays['valid'], train, val, selected, center, scale)
    write_json(output/'history.json', history)
    best = min(history, key=lambda item: item['validation_squared_m'])
    saved = dict(state_dict=model.state_dict(), point_count=len(selected), selected=selected.tolist(),
                 center=center.tolist(), scale=scale.tolist(), origin=model.origin.tolist(),
                 calibration=metadata['calibration'], training=dict(seed=20261210, epochs=400,
                     best=best, ground_truth_used=False, observation_count=int(train.sum()),
                     validation_count=int(val.sum()), input='Seven actual measured joint positions',
                     target='Metric surface feature positions triangulated from calibrated stereo RGB'))
    torch.save(saved, output/'model.pt')
    freeze = dict(at_utc=datetime.now(timezone.utc).isoformat(),
                  checkpoint_sha256=hashlib.sha256((output/'model.pt').read_bytes()).hexdigest(),
                  data_metadata_sha256=hashlib.sha256((data/'metadata.json').read_bytes()).hexdigest(),
                  files_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in ('model.py', 'vision.py', 'collect.py')},
                  selection='No test/hold/blind frames opened by fitting; all rules and checkpoint now frozen',
                  scope='Sparse tracked surface features of a feedback-coupled entity, not complete body volume')
    write_json(output/'freeze.json', freeze)
    print(f'Frozen model: {len(selected)} points; validation RMS {np.sqrt(best["validation_squared_m"])*1000:.2f} mm', flush=True)
    return saved


def controls(data, output):
    """Discovery ablations on the development observations; never refit final weights."""
    data, output = Path(data), Path(output)
    metadata = json.loads((data/'metadata.json').read_text(encoding='utf-8'))
    with np.load(output/'training-tracks.npz') as cache:
        arrays = {key: cache[key] for key in cache.files}
    phases = np.array([metadata['frames'][int(i)]['phase'] for i in arrays['index']])
    q = arrays['q']
    variants = {'correct_q': q,
                'shuffled_time_q': q[np.random.default_rng(522).permutation(len(q))],
                'constant_q': np.repeat(q[:1], len(q), axis=0)}
    result = {}
    for name, feedback in variants.items():
        selected, _, _, _ = discover(feedback, arrays['xyz'], arrays['valid'], phases == 'train', phases == 'val')
        result[name] = dict(selected_count=len(selected), selected=selected.tolist())
    write_json(output/'feedback-discovery-controls.json', result)
    print(json.dumps(result), flush=True)
    return result


def evaluate(data, output):
    data, output = Path(data), Path(output)
    metadata = json.loads((data/'metadata.json').read_text(encoding='utf-8'))
    freeze = json.loads((output/'freeze.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((output/'model.pt').read_bytes()).hexdigest() == freeze['checkpoint_sha256']
    saved = torch.load(output/'model.pt', weights_only=True, map_location='cpu')
    torch.set_num_threads(4)
    model = SpatialEntity(saved['point_count'], saved['center'], saved['scale'], saved['origin'])
    model.load_state_dict(saved['state_dict'])
    model.eval()
    selected = np.array(saved['selected'])
    with np.load(data/'observations'/metadata['frames'][0]['file']) as first:
        tracker = StereoTracks(first['left'], metadata['calibration'])
    with np.load(output/'training-tracks.npz') as cache:
        np.testing.assert_allclose(tracker.reference_points, cache['reference_points'])
        tracker.previous_points = cache['previous_points'].copy()
    state = EntityState(model, tolerance_m=max(.02, 3*np.sqrt(saved['training']['best']['validation_squared_m'])))
    rows, point_records = [], []
    last_visible = None
    for index, row in enumerate(metadata['frames']):
        if row['phase'] in ('train', 'val'):
            continue
        with np.load(data/'observations'/row['file']) as observation:
            available = bool(observation['camera_available'])
            q = observation['q'].copy()
            if available:
                tracks = tracker.observe(observation['left'], observation['right'])
                visible_xyz, visible_valid = tracks['xyz'][selected], tracks['valid'][selected]
                result = state.observe(q, row['time'], visible_xyz, visible_valid)
                if last_visible is None:
                    last_visible = np.asarray(result['points'])
                last_visible[visible_valid] = visible_xyz[visible_valid]
            else:
                # The entity has no access to hidden images or oracle feature tracks.
                result = state.observe(q, row['time'])
        prediction = np.asarray(result['points'])
        # Privileged evaluation and hidden reference images are read only AFTER prediction.
        with np.load(data/'evaluation'/row['file']) as evaluation:
            if not available:
                tracks = tracker.observe(evaluation['oracle_left'], evaluation['oracle_right'])
            truth = tracks['xyz'][selected]
            valid = tracks['valid'][selected]
            uv = tracks['uv'][selected]
            h, w = evaluation['body_mask'].shape
            pixels = np.rint(uv).astype(int)
            pixels[:, 0] = np.clip(pixels[:, 0], 0, w-1)
            pixels[:, 1] = np.clip(pixels[:, 1], 0, h-1)
            on_body = evaluation['body_mask'][pixels[:, 1], pixels[:, 0]] & valid
            on_prop = evaluation['prop_mask'][pixels[:, 1], pixels[:, 0]] & valid
            z = evaluation['depth'][pixels[:, 1], pixels[:, 0]]
            c = metadata['calibration']
            geometric = np.column_stack(((uv[:, 0]-c['cx'])*z/c['focal_px'],
                                          (uv[:, 1]-c['cy'])*z/c['focal_px'], z))
        error = np.sum((prediction-truth)**2, axis=1)
        fixed_error = np.sum((np.array(saved['origin'])-truth)**2, axis=1)
        held_error = np.sum((last_visible-truth)**2, axis=1)
        record = dict(frame=index, phase=row['phase'], time=row['time'], **result,
                      valid_features=int(valid.sum()), body_features=int(on_body.sum()), prop_features=int(on_prop.sum()),
                      squared_error_sum=float(error[valid].sum()),
                      fixed_squared_error_sum=float(fixed_error[valid].sum()),
                      last_seen_squared_error_sum=float(held_error[valid].sum()),
                      body_geometric_error_sum=float(np.sum((prediction[on_body]-geometric[on_body])**2)),
                      stereo_body_geometric_error_sum=float(np.sum((truth[on_body]-geometric[on_body])**2)))
        rows.append(record)
        point_records.append(dict(q=q, truth=truth, valid=valid, prediction=prediction))
    rng = np.random.default_rng(20261211)
    permutation = rng.permutation(len(rows))
    with torch.inference_mode():
        wrong = model(torch.from_numpy(np.stack([point_records[i]['q'] for i in permutation]))).numpy()
    for index, (row, points) in enumerate(zip(rows, point_records)):
        row['shuffled_q_squared_error_sum'] = float(np.sum((wrong[index]-points['truth'])[points['valid']]**2))
    summaries = {}
    for phase in ('test', 'hold', 'blind'):
        group = [row for row in rows if row['phase'] == phase]
        count = sum(row['valid_features'] for row in group)
        body_count = sum(row['body_features'] for row in group)
        score = dict(frames=len(group), valid_feature_observations=count,
                     body_purity=body_count/max(count, 1),
                     prop_fraction=sum(row['prop_features'] for row in group)/max(count, 1),
                     visually_supported_frames=sum(row['status'] == 'visually_supported' for row in group))
        for name, field in [('model_rmse_m', 'squared_error_sum'), ('fixed_rmse_m', 'fixed_squared_error_sum'),
                            ('last_seen_rmse_m', 'last_seen_squared_error_sum'),
                            ('shuffled_q_rmse_m', 'shuffled_q_squared_error_sum')]:
            score[name] = float(np.sqrt(sum(row[field] for row in group)/count)) if count else None
        score['body_geometric_rmse_m'] = float(np.sqrt(sum(row['body_geometric_error_sum'] for row in group)/body_count)) if body_count else None
        score['stereo_body_geometric_rmse_m'] = float(np.sqrt(sum(row['stereo_body_geometric_error_sum'] for row in group)/body_count)) if body_count else None
        summaries[phase] = score
    report = dict(checkpoint_sha256=freeze['checkpoint_sha256'], point_count=len(selected),
                  primary='3D predicted surface feature positions in metres, no segmentation objective',
                  calibration=metadata['calibration'], summary=summaries, per_frame=rows,
                  limitations=['One local trajectory family, no full-workspace or real-hardware validation',
                     'Sparse textured surfaces, not a solid occupied volume or complete arm topology',
                     'Persistent identifier and uncertainty handling are engineered state management',
                     'Stereo supplies metric depth, so this does not demonstrate monocular 3D learning',
                     'Temporal smoothness and correlation are not conclusive causality',
                     'No tools, contact understanding, force feedback or learned permanence objective'])
    write_json(output/'metrics.json', report)
    print(json.dumps(summaries, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('fit', 'evaluate', 'controls'))
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    {'fit': fit, 'evaluate': evaluate, 'controls': controls}[args.mode](args.data, args.output)
