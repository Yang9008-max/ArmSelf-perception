"""E5: test RGB correspondence recovery and correction of biased feedback.

Memory construction uses the E4 training RGB/tracks only. Prediction files are
saved and hashed before the separate scorer opens simulator evaluation labels.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from .body_cycle import (SEGMENTATION, collect_data, read_json, reference, sha,
                         stack, utc, recovery_events, rmse, write_json)
from .body_memory import BodyModel, BodyMasks, BodyObserver, BodyMemory
from .recovery import farthest_poses, VisualBodyBank, VisualFeedbackCorrection

E4 = Path('outputs/local-check-20261024/e4-body-cycle')
ROOT = Path('outputs/local-check-20261024/e5-body-memory')
BODY = E4/'bridge/model-v7.pt'
DEV_SEED = 2026102495
FINAL_SEEDS = (2026102501, 2026102502, 2026102503)
BIAS = np.array([.055, .018, -.015, 0., 0., 0., 0.], np.float32)


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    destination = output/'visual-bank.npz'
    if destination.exists():
        return
    body = BodyModel(BODY)
    metadata = read_json(E4/'training-data/metadata.json')
    with np.load(E4/'bridge/training-observations.npz') as a:
        q, uv = a['q'].copy(), a['uv'].copy()
        usable = a['valid_v7'] & (np.linalg.norm(a['xyz']-body(q), axis=-1) < .035)
        train = np.flatnonzero(a['phase'] == 'train')
    indices = train[farthest_poses(q[train], body.saved['scale'], budget=64)]
    gray = []
    for index in indices:
        with np.load(E4/'training-data/observations'/metadata['frames'][index]['file']) as obs:
            gray.append(cv2.cvtColor(obs['left'], cv2.COLOR_RGB2GRAY))
    np.savez_compressed(destination, gray=np.stack(gray), q=q[indices], uv=uv[indices],
                        valid=usable[indices], scale=body.saved['scale'], training_indices=indices)
    write_json(output/'bank-provenance.json', dict(created_utc=utc(), frames=len(indices),
        source='E4 training split only; RGB identities inherited from original LK tracks',
        training_indices=indices.tolist(), accepted_reference_observations=int(usable[indices].sum()),
        body_sha256=sha(BODY), segmentation_sha256=sha(SEGMENTATION), bank_sha256=sha(destination),
        used_evaluation_labels=False, neural_weights_changed=False))


def inputs(source, biased=False):
    """Only RGB, encoder reading, time and availability cross into inference."""
    metadata = read_json(source/'metadata.json')
    for index, row in enumerate(metadata['frames']):
        with np.load(source/'observations'/row['file']) as obs:
            # Controlled readout fault: physical motion and RGB remain identical.
            # True pre-fault joint values and the fault vector are not passed on.
            amount = np.clip((index-25)/5., 0., 1.) if biased else 0.
            yield dict(left=obs['left'].copy(), right=obs['right'].copy(),
                q=obs['q'].copy()+BIAS*amount, time=float(obs['time']),
                available=bool(obs['camera_available']))


def predict(output, source, name, configuration, biased=False):
    destination = output/'predictions'/f'{name}.npz'
    if destination.exists():
        return destination
    destination.parent.mkdir(exist_ok=True)
    body, masks = BodyModel(BODY), BodyMasks(SEGMENTATION)
    _, image, _ = reference()
    metadata = read_json(source/'metadata.json')
    calibration = metadata['calibration']
    original = BodyObserver(image, calibration, body.selected)
    with np.load(output/'visual-bank.npz') as saved:
        bank = VisualBodyBank(saved, calibration, **configuration)
    correction = VisualFeedbackCorrection(body, body.saved['scale'])
    memories = {key: BodyMemory(len(body.selected)) for key in ('original', 'bank', 'corrected')}
    rows = []
    for index, obs in enumerate(inputs(source, biased)):
        q, time = obs['q'], obs['time']
        prior, corrected_prior = body(q)[0], correction.predict(q)
        mask = masks(obs['left'], q)['v7'] if obs['available'] else np.zeros(obs['left'].shape[:2], bool)
        a = original.observe(obs['left'], obs['right'], mask, prediction=prior)
        b = bank.observe(obs['left'], obs['right'], mask, q, prior)
        c = bank.observe(obs['left'], obs['right'], mask, correction.corrected_q(q), corrected_prior)
        if not obs['available']:
            a['valid'][:] = b['valid'][:] = c['valid'][:] = False
        update = correction.update(q, c['xyz'], c['valid'], time)
        corrected_prior = correction.predict(q)
        row = dict(q_reading=q, time=np.float64(time), prediction_feedback=prior,
                   prediction_state_only=corrected_prior, offset=correction.offset.copy(),
                   state_updated=np.bool_(update['updated']), support=np.int32(update['support']))
        for key, observation, prediction in [('original', a, prior), ('bank', b, prior),
                                               ('corrected', c, corrected_prior)]:
            state = memories[key].update(prediction, q, time, observation['xyz'], observation['valid'])
            row.update({f'prediction_{key}': state['prediction'], f'accepted_{key}': state['accepted'],
                        f'observed_{key}': observation['xyz'], f'uv_{key}': observation['uv']})
        rows.append(row)
        if (index+1) % 35 == 0:
            print(f'{name}: {index+1}/{len(metadata["frames"])}', flush=True)
    np.savez_compressed(destination, **stack(rows))
    write_json(destination.with_suffix('.freeze.json'), dict(frozen_utc=utc(),
        prediction_sha256=sha(destination), body_sha256=sha(BODY), bank_sha256=sha(output/'visual-bank.npz'),
        configuration=configuration, biased_readout=biased, truth_opened_during_prediction=False))
    return destination


def score(output, source, prediction):
    freeze = read_json(prediction.with_suffix('.freeze.json'))
    assert sha(prediction) == freeze['prediction_sha256']
    with np.load(prediction) as p:
        arrays = {k: p[k].copy() for k in p.files}
    metadata = read_json(source/'metadata.json')
    truth = []
    for row in metadata['frames']:
        with np.load(source/'evaluation'/row['file']) as t:
            truth.append({key: t[key].copy() for key in
                          ('anchor_xyz', 'anchor_body', 'anchor_visible', 'behind_panel')})
    truth = stack(truth)
    body, visible, hidden = truth['anchor_body'], truth['anchor_visible'], truth['behind_panel']
    result = dict(name=prediction.stem, frames=len(body), fixed_body_points=int(body[0].sum()),
                  prop_max_displacement_m=metadata['prop_max_displacement_m'], modes={})
    events = {}
    for key in ('feedback', 'original', 'bank', 'state_only', 'corrected'):
        error = np.sum((arrays[f'prediction_{key}']-truth['anchor_xyz'])**2, axis=-1)
        metrics = dict(all_body_rmse_mm=rmse(error, body), hidden_body_rmse_mm=rmse(error, hidden & body),
                       after_fault_rmse_mm=rmse(error[35:], body[35:]),
                       before_fault_rmse_mm=rmse(error[:25], body[:25]))
        if f'accepted_{key}' in arrays:
            accepted = arrays[f'accepted_{key}']
            observed = arrays[f'observed_{key}']
            correct = accepted & body & visible & (np.linalg.norm(observed-truth['anchor_xyz'], axis=-1) <= .02)
            events[key] = recovery_events(visible, hidden, accepted, observed, truth['anchor_xyz'], body[0])
            count, recovered = len(events[key]), sum(e['recovered'] for e in events[key])
            metrics.update(recovery_count=recovered, recovery_events=count, recovery_rate=recovered/max(count, 1),
                visible_correct_coverage=float(correct.sum()/max((body & visible).sum(), 1)),
                visible_correspondence_precision=float(correct.sum()/max((accepted & body & visible).sum(), 1)),
                accepted_per_frame=float(accepted.sum()/len(body)),
                covered_body_points=int((correct.sum(0) >= 3).sum()),
                false_reappearance=sum(e['false_accepted'] for e in events[key]),
                accepted_reappearance=sum(e['accepted'] for e in events[key]))
        result['modes'][key] = metrics
    result['state_update_frames'] = int(arrays['state_updated'].sum())
    result['final_estimated_offset_rad_or_m'] = (arrays['offset'][-1]*np.asarray(BodyModel(BODY).saved['scale'])).tolist()
    (output/'scores').mkdir(exist_ok=True)
    write_json(output/'scores'/f'{prediction.stem}.json', result)
    write_json(output/'scores'/f'{prediction.stem}-events.json', events)
    print({k: (round(v['all_body_rmse_mm'], 2), f"{v.get('recovery_count', '-')}/{v.get('recovery_events', '-')} recoveries")
           for k, v in result['modes'].items()}, flush=True)
    return result


def develop(output):
    prepare(output)
    source = output/f'development-{DEV_SEED}'
    collect_data(source, DEV_SEED, panel=True)
    configurations = {f'w{window}-r4': dict(window=window, references=4) for window in (11, 21, 7)}
    for name, config in configurations.items():
        for biased in (False, True):
            prediction = predict(output, source, f'dev-{name}-{"biased" if biased else "clean"}', config, biased)
            score(output, source, prediction)


def freeze(output, configuration):
    destination = output/'final-protocol.json'
    if destination.exists():
        raise FileExistsError('Final protocol already frozen')
    write_json(destination, dict(frozen_utc=utc(), development_seed=DEV_SEED, final_seeds=list(FINAL_SEEDS),
        configuration=configuration, bias_rad_or_m=BIAS.tolist(), bias_ramp_frames=[26, 30],
        selection='Window 7: more development recoveries than 11/21; 11 had lower biased-state error. No final scores used.',
        recovery_tolerance_m=.02, recovery_window_frames=5, hidden_min_frames=3,
        scoring='All 28 fixed body anchors; visual recovery requires an actual stereo observation',
        body_sha256=sha(BODY), bank_sha256=sha(output/'visual-bank.npz'),
        source_sha256={name: sha(Path(__file__).with_name(name)) for name in
                       ('recovery.py', 'recovery_cycle.py', 'body_memory.py', 'body_cycle.py',
                        'body_fixture.py', 'vision.py')},
        weights_frozen=True, evaluation_labels_for_learning=False))


def final(output):
    protocol = read_json(output/'final-protocol.json')
    assert sha(BODY) == protocol['body_sha256']
    assert sha(output/'visual-bank.npz') == protocol['bank_sha256']
    for name, digest in protocol['source_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    for seed in protocol['final_seeds']:
        source = output/f'final-data-{seed}'
        collect_data(source, seed, panel=True)
        for biased in (False, True):
            prediction = predict(output, source, f'final-{seed}-{"biased" if biased else "clean"}',
                                 protocol['configuration'], biased)
            score(output, source, prediction)
    write_json(output/'complete.json', dict(completed_utc=utc(), final_seeds=protocol['final_seeds']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'develop', 'freeze', 'final'))
    parser.add_argument('--output', type=Path, default=ROOT)
    parser.add_argument('--window', type=int, default=11)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.mode == 'freeze':
        freeze(args.output, dict(window=args.window, references=4))
    else:
        globals()[args.mode](args.output)
