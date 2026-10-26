"""E4: segmentation bridge, actual occlusion recovery, shared-decoder exploration.

Run prepare, acquire, evaluate, then embodiment.body_report. Fitting/acquisition
never open evaluation files; each evaluation prediction precedes truth reads.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from perception.collect import HOME
from .collect import protocol, evaluation_protocol
from .experiment import write_json
from .model import masked_position_loss
from .vision import project
from .explore_run import AMPLITUDE
from .body_memory import BodyMasks, BodyModel, BodyObserver, BodyMemory, ResidualBelief, sample_mask
from .body_fixture import make_fixture, render_rgb, render_truth, EvaluationLandmarks, occlusion_protocol


PRIOR = Path('outputs/embodiment/entity-v2/run-v2')
SEGMENTATION = Path('outputs/perception/v7-static/model/model.pt')
MODES = ('none', 'v6', 'v7')
POLICIES = ('active', 'random', 'preset')
ACQUISITION_SEEDS = (521, 522, 523, 524, 525)
EVALUATION_SEEDS = (2026102481, 2026102482, 2026102483)
BUDGETS = (16, 32, 64, 112)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def reference():
    saved = torch.load(PRIOR/'model.pt', map_location='cpu', weights_only=True)
    with np.load(PRIOR/'reference.npz') as data:
        return saved, data['left'].copy(), data['points'].copy()


def stack(rows):
    return {key: np.stack([row[key] for row in rows]) for key in rows[0]}


def fresh(path):
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError(f'Preserving partial/completed experiment: {path}')


def collect_data(output, seed, training=False, panel=False):
    if (output/'metadata.json').exists():
        return read_json(output/'metadata.json')
    fresh(output)
    (output/'observations').mkdir()
    (output/'evaluation').mkdir()
    saved, image, points = reference()
    model, data, calibration = make_fixture(output, panel)
    anchors = EvaluationLandmarks(model, data, points[saved['selected'], 0], calibration)
    q_indices = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    props = [model.body(f'table_object_{i}').id for i in range(3)]
    initial_props = data.xpos[props].copy()
    renderer = mujoco.Renderer(model, height=480, width=640)
    schedule = ([(p, q, v) for p, q, v in protocol(seed) if p in ('train', 'val')] if training
                else list(occlusion_protocol(seed)) if panel else evaluation_protocol(seed))
    rows, max_prop_change = [], 0.
    try:
        for i, (phase, target, available) in enumerate(schedule):
            data.ctrl[:] = target
            mujoco.mj_step(model, data, nstep=50)
            mujoco.mj_forward(model, data)
            q = data.qpos[q_indices].astype(np.float32).copy()
            left, right = render_rgb(renderer, data), render_rgb(renderer, data, 'entity_right')
            name = f'{i:05d}.npz'
            np.savez_compressed(output/'observations'/name, q=q, time=data.time,
                                camera_available=np.bool_(available),
                                left=left if available else np.zeros_like(left),
                                right=right if available else np.zeros_like(right))
            truth = render_truth(renderer, model, data)
            truth.update(anchors.score_truth(model, data, truth))
            if panel:
                clear = render_truth(renderer, model, data, panel_visible=False)
                clear_score = anchors.score_truth(model, data, clear)
                truth['anchor_visible_without_panel'] = clear_score['anchor_visible']
                # Self-occluded points do not count as screen-caused occlusion.
                truth['behind_panel'] &= clear_score['anchor_visible']
                truth['occluded_body_pixel_fraction'] = np.float64(
                    (clear['body_mask'] & truth['panel_mask']).sum()/max(clear['body_mask'].sum(), 1))
            if not available:
                truth.update(oracle_left=left, oracle_right=right)
            np.savez_compressed(output/'evaluation'/name, **truth)
            max_prop_change = max(max_prop_change, float(np.max(np.abs(data.xpos[props]-initial_props))))
            rows.append(dict(file=name, phase=phase, time=float(data.time), camera_available=bool(available)))
            if (i+1) % 100 == 0:
                print(f'collect {output.name} {i+1}/{len(schedule)}', flush=True)
    finally:
        renderer.close()
    metadata = dict(seed=seed, frames=rows, calibration=calibration, panel=panel,
                    training=training, created_utc=utc(), reference_sha256=sha(PRIOR/'reference.npz'),
                    fixture_sha256=sha(output/'stereo.xml'), fixed_reference_body_points=int(anchors.is_body.sum()),
                    prop_max_displacement_m=max_prop_change, props_initial_xyz=initial_props.tolist(),
                    learner_inputs='RGB stereo, actual measured q, timestamps, camera availability only',
                    evaluation_only='Masks, depth, physical surface anchors, hidden RGB; never fitting inputs')
    if max_prop_change > 1e-12:
        raise AssertionError('Tabletop props moved')
    write_json(output/'metadata.json', metadata)
    return metadata


def cache_training(data, output, masks):
    destination = output/'training-observations.npz'
    if destination.exists():
        with np.load(destination) as result:
            return {k: result[k] for k in result.files}
    saved, ref, _ = reference()
    metadata = read_json(data/'metadata.json')
    observer = BodyObserver(ref, metadata['calibration'], saved['selected'])
    rows = []
    for i, row in enumerate(metadata['frames']):
        if row['phase'] not in ('train', 'val'):
            raise ValueError('Fitting dataset must contain train/val observations only')
        with np.load(data/'observations'/row['file']) as obs:
            mm = masks(obs['left'], obs['q'])
            tracks = observer.observe(obs['left'], obs['right'], mm['none'])
            rows.append(dict(q=obs['q'].copy(), xyz=tracks['xyz'], uv=tracks['uv'],
                             phase=np.array(row['phase']),
                             **{f'valid_{name}': tracks['raw_valid'] & sample_mask(mm[name], tracks['uv']) for name in MODES}))
        if (i+1) % 100 == 0:
            print(f'train RGB/mask tracks {i+1}/{len(metadata["frames"])}', flush=True)
    arrays = stack(rows)
    np.savez_compressed(destination, **arrays)
    return arrays


def fit_bridge(output, arrays):
    for name in MODES:
        destination = output/f'model-{name}.pt'
        if destination.exists():
            continue
        torch.manual_seed(2026102471)
        body = BodyModel(PRIOR/'model.pt')
        optimizer = torch.optim.Adam(body.model.parameters(), lr=.0005)
        q = torch.from_numpy(arrays['q'])
        xyz = torch.from_numpy(arrays['xyz'])
        usable = torch.from_numpy(arrays[f'valid_{name}'])
        train, val = arrays['phase'] == 'train', arrays['phase'] == 'val'
        best, state, history = float('inf'), None, []
        for epoch in range(300):
            body.model.train()
            optimizer.zero_grad()
            loss = masked_position_loss(body.model(q[train]), xyz[train], usable[train])/.01
            loss.backward()
            optimizer.step()
            body.model.eval()
            with torch.inference_mode():
                score = float(masked_position_loss(body.model(q[val]), xyz[val], usable[val]))
            history.append(dict(epoch=epoch+1, validation_squared_m=score))
            if score < best:
                best, state = score, copy.deepcopy(body.model.state_dict())
        saved = dict(body.saved)
        saved['state_dict'] = state
        saved['origin'] = state['origin'].tolist()
        saved['e4_training'] = dict(mask=name, epochs=300, learning_rate=.0005, seed=2026102471,
                                   fixed_feature_ids=True, inherited_e2_weights=True,
                                   valid_train_per_point=usable[train].sum(0).tolist(),
                                   best_validation_squared_m=best, used_evaluation_labels=False)
        torch.save(saved, destination)
        write_json(output/f'history-{name}.json', history)
        print(f'bridge model {name} frozen: val {np.sqrt(best)*1000:.3f}mm, targets {int(usable[train].sum())}', flush=True)


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    config_path = output/'protocol.json'
    if not config_path.exists():
        write_json(config_path, dict(created_utc=utc(), acquisition_seeds=list(ACQUISITION_SEEDS),
            evaluation_seeds=list(EVALUATION_SEEDS), training_seed=2026102471, modes=list(MODES),
            selected_ids=reference()[0]['selected'], shared_decoder='model-v7.pt fixed before final evaluation',
            budgets=list(BUDGETS), policies=list(POLICIES), bootstrap=16,
            gate_m=BodyMemory.gate_m, recovery_tolerance_m=.02, recovery_hidden_min_frames=3,
            recovery_window_frames=5, information_travel_penalty=.15,
            evaluation='All 28 physical reference anchors, including never observed/lost points',
            segmentation_sha256=sha(SEGMENTATION), e2_sha256=sha(PRIOR/'model.pt'),
            source_sha256={name: sha(Path(__file__).with_name(name)) for name in
                           ('body_cycle.py', 'body_memory.py', 'body_fixture.py')}))
    (output/'bridge').mkdir(exist_ok=True)
    collect_data(output/'training-data', 2026102471, training=True)
    masks = BodyMasks(SEGMENTATION)
    arrays = cache_training(output/'training-data', output/'bridge', masks)
    fit_bridge(output/'bridge', arrays)
    if not (output/'bridge'/'freeze.json').exists():
        write_json(output/'bridge'/'freeze.json', dict(at_utc=utc(),
            models={name: sha(output/'bridge'/f'model-{name}.pt') for name in MODES},
            training_metadata_sha256=sha(output/'training-data'/'metadata.json'),
            segmentation_sha256=sha(SEGMENTATION), final_evaluation_not_opened=True))


def choose(belief, q, rng, policy, step):
    current = (q-belief.center)/belief.scale
    lo, hi = np.maximum(-1., current-.5), np.minimum(1., current+.5)
    if np.any(lo > hi):
        raise ValueError('Feedback outside local envelope')
    if step < 16 or policy == 'random':
        candidate = rng.uniform(lo, hi)
        source, score = ('shared_bootstrap' if step < 16 else 'random'), 0.
    elif policy == 'preset':
        candidate = np.clip(.8*np.sin(step*.19*np.linspace(.7, 1.4, 7)+np.arange(7)*.8), lo, hi)
        source, score = 'preset', 0.
    elif policy == 'active':
        proposals = np.r_[rng.uniform(lo, hi, (64, 7)),
                          np.clip(current+np.r_[np.eye(7)*.5, -np.eye(7)*.5], lo, hi)]
        information = belief.scores(belief.center+proposals*belief.scale)
        scores = information/(1+.15*np.linalg.norm(proposals-current, axis=1))
        best = int(np.argmax(scores))
        candidate, score, source = proposals[best], float(scores[best]), 'past_observations_information_per_travel'
    else:
        raise ValueError(policy)
    return (belief.center+candidate*belief.scale).astype(np.float32), dict(
        source=source, score=score, observed_before_choice=int(belief.observations))


def acquire(output):
    bridge_freeze = read_json(output/'bridge'/'freeze.json')
    body = BodyModel(output/'bridge'/'model-v7.pt')
    assert sha(output/'bridge'/'model-v7.pt') == bridge_freeze['models']['v7']
    saved, ref, _ = reference()
    masks = BodyMasks(SEGMENTATION)
    runs = output/'exploration'
    runs.mkdir(exist_ok=True)
    for seed in ACQUISITION_SEEDS:
        for policy in POLICIES:
            destination = runs/f'{seed}-{policy}'
            if (destination/'freeze.json').exists():
                continue
            fresh(destination)
            model, data, calibration = make_fixture(destination)
            indices = model.jnt_qposadr[model.actuator_trnid[:, 0]]
            observer = BodyObserver(ref, calibration, saved['selected'])
            belief = ResidualBelief(body, HOME, AMPLITUDE)
            rng, rows = np.random.default_rng(seed), []
            props = [model.body(f'table_object_{i}').id for i in range(3)]
            original_props = data.xpos[props].copy()
            renderer = mujoco.Renderer(model, height=480, width=640)
            cost = 0.
            travel = np.zeros(7)
            try:
                for step in range(112):
                    before = data.qpos[indices].copy()
                    target, decision = choose(belief, before, rng, policy, step)
                    data.ctrl[:] = target
                    previous = before.copy()
                    for _ in range(20):
                        mujoco.mj_step(model, data, nstep=5)
                        current = data.qpos[indices].copy()
                        delta = current-previous
                        cost += float(np.linalg.norm(delta/AMPLITUDE))
                        travel += np.abs(delta)
                        previous = current
                    mujoco.mj_forward(model, data)
                    q = data.qpos[indices].astype(np.float32).copy()
                    left, right = render_rgb(renderer, data), render_rgb(renderer, data, 'entity_right')
                    mm = masks(left, q)
                    predicted = body(q)[0]
                    observed = observer.observe(left, right, mm['v7'], prediction=predicted)
                    update, accepted = belief.update(q, observed['xyz'], observed['valid'])
                    rows.append(dict(q=q, target=target, xyz=observed['xyz'], uv=observed['uv'],
                        valid=accepted, raw_valid=observed['raw_valid'], normalized_travel=np.float64(cost),
                        rotary_travel_rad=np.float64(travel[:6].sum()), gripper_travel_m=np.float64(travel[6]),
                        score=np.float64(decision['score'])))
                    if step+1 in BUDGETS:
                        np.savez_compressed(destination/f'belief-{step+1:03d}.npz', **belief.state())
                    if (step+1) % 32 == 0:
                        print(f'explore {seed}/{policy} {step+1}/112 accepted {accepted.sum()} travel {cost:.1f}', flush=True)
                if not np.array_equal(original_props, data.xpos[props]):
                    raise AssertionError('Props moved')
            finally:
                renderer.close()
            np.savez_compressed(destination/'tracks.npz', **stack(rows))
            write_json(destination/'freeze.json', dict(at_utc=utc(), seed=seed, policy=policy,
                body_decoder_sha256=sha(output/'bridge'/'model-v7.pt'), segmentation_sha256=sha(SEGMENTATION),
                checkpoints={str(b): sha(destination/f'belief-{b:03d}.npz') for b in BUDGETS},
                inputs='RGB stereo, measured q, calibration; zero simulator evaluation calls',
                fixed_props=True, acquisition_observations_saved='derived stereo/mask tracks and actual q',
                feature_ids=saved['selected'], budget=112, observation_seconds=.2,
                normalized_travel=cost, rotary_travel_rad=float(travel[:6].sum()), gripper_travel_m=float(travel[6])))
    if not (runs/'freeze.json').exists():
        write_json(runs/'freeze.json', dict(all_models_frozen_utc=utc(), seeds=list(ACQUISITION_SEEDS),
            policies=list(POLICIES), checkpoints=list(BUDGETS), body_decoder_sha256=sha(output/'bridge'/'model-v7.pt'),
            runs={f'{s}-{p}': sha(runs/f'{s}-{p}'/'freeze.json') for s in ACQUISITION_SEEDS for p in POLICIES}))


def rmse(error, mask):
    return float(np.sqrt(np.mean(error[mask]))*1000) if np.any(mask) else None


def bridge_evaluate(output, seed, masks):
    destination = output/'bridge'/f'eval-{seed}.npz'
    if destination.exists():
        return
    data = output/f'evaluation-{seed}'
    metadata = read_json(data/'metadata.json')
    saved, ref, _ = reference()
    observer = BodyObserver(ref, metadata['calibration'], saved['selected'])
    models = {name: BodyModel(output/'bridge'/f'model-{name}.pt') for name in MODES}
    rows = []
    for row in metadata['frames']:
        with np.load(data/'observations'/row['file']) as obs:
            q = obs['q'].copy()
            available = bool(obs['camera_available'])
            predictions = {name: model(q)[0] for name, model in models.items()}
            if available:
                mm = masks(obs['left'], q)
                track = observer.observe(obs['left'], obs['right'], mm['none'])
                valid = {name: track['valid'] & sample_mask(mm[name], track['uv']) for name in MODES}
            else:
                track = dict(uv=np.zeros((len(saved['selected']), 2)), xyz=np.zeros((len(saved['selected']), 3)))
                valid = {name: np.zeros(len(saved['selected']), bool) for name in MODES}
        # No evaluation file is opened until all predictions and gates are fixed.
        with np.load(data/'evaluation'/row['file']) as truth:
            on_body = sample_mask(truth['body_mask'], track['uv'])
            result = dict(q=q, phase=np.array(row['phase']), available=np.bool_(available),
                          uv=track['uv'], observed_xyz=track['xyz'], on_body=on_body,
                          anchor_xyz=truth['anchor_xyz'].copy(), anchor_body=truth['anchor_body'].copy(),
                          anchor_visible=truth['anchor_visible'].copy())
        result.update({f'prediction_{name}': predictions[name] for name in MODES})
        result.update({f'valid_{name}': valid[name] for name in MODES})
        rows.append(result)
    np.savez_compressed(destination, **stack(rows))


def recovery_events(visible, behind, accepted, observed, truth, is_body, tolerance=.02):
    """First visible return after >=3 screen-hidden frames, all IDs fixed.

    If self-occlusion follows screen occlusion, keep the pending event until
    actual visibility returns; difficult delayed returns stay in the denominator.
    """
    result = []
    correct = accepted & (np.linalg.norm(observed-truth, axis=-1) <= tolerance)
    for point in np.flatnonzero(is_body):
        hidden_run, pending_hidden = 0, 0
        for frame in range(len(visible)):
            if behind[frame, point]:
                hidden_run += 1
                if hidden_run >= 3:
                    pending_hidden = max(pending_hidden, hidden_run)
                continue
            if pending_hidden >= 3 and visible[frame, point]:
                stop = min(frame+5, len(visible))
                matches = np.flatnonzero(correct[frame:stop, point] & visible[frame:stop, point])
                candidate = np.flatnonzero(accepted[frame:stop, point] & visible[frame:stop, point])
                result.append(dict(point=int(point), frame=frame, hidden_frames=pending_hidden,
                    recovered=bool(len(matches)), latency_frames=int(matches[0]) if len(matches) else None,
                    false_accepted=int(np.sum(accepted[frame:stop, point] & visible[frame:stop, point] & ~correct[frame:stop, point])),
                    accepted=int(len(candidate))))
                pending_hidden = 0
            if visible[frame, point] or not behind[frame, point]:
                hidden_run = 0
    return result


def occlusion_evaluate(output, seed, masks):
    destination = output/'occlusion'
    destination.mkdir(exist_ok=True)
    if (destination/f'{seed}.npz').exists():
        return
    source = output/f'occlusion-data-{seed}'
    metadata = read_json(source/'metadata.json')
    body = BodyModel(output/'bridge'/'model-v7.pt')
    saved, ref, _ = reference()
    visual = BodyObserver(ref, metadata['calibration'], saved['selected'])
    fused = BodyObserver(ref, metadata['calibration'], saved['selected'])
    memory = BodyMemory(len(saved['selected']))
    last_visual = body(HOME)[0].copy()
    rows = []
    for row in metadata['frames']:
        with np.load(source/'observations'/row['file']) as obs:
            q = obs['q'].copy()
            prior = body(q)[0]
            mm = masks(obs['left'], q)
            a = visual.observe(obs['left'], obs['right'], mm['v7'])
            b = fused.observe(obs['left'], obs['right'], mm['v7'], prediction=prior)
            last_visual[a['valid']] = a['xyz'][a['valid']]
            state = memory.update(prior, q, float(obs['time']), b['xyz'], b['valid'])
            result = dict(q=q, phase=np.array(row['phase']), time=np.float64(obs['time']),
                prediction_visual=last_visual.copy(), prediction_feedback=prior, prediction_fusion=state['prediction'],
                accepted_visual=a['valid'], accepted_fusion=state['accepted'],
                observed_visual=a['xyz'], observed_fusion=b['xyz'],
                uv_visual=a['uv'], uv_fusion=b['uv'], variance=state['variance'])
        with np.load(source/'evaluation'/row['file']) as truth:
            result.update({k: truth[k].copy() for k in ('anchor_xyz', 'anchor_body', 'anchor_visible', 'behind_panel',
                                                      'anchor_visible_without_panel', 'occluded_body_pixel_fraction')})
            for mode, tracks in [('visual', a), ('fusion', b)]:
                result[f'on_body_{mode}'] = sample_mask(truth['body_mask'], tracks['uv'])
        rows.append(result)
    arrays = stack(rows)
    events = {mode: recovery_events(arrays['anchor_visible'], arrays['behind_panel'], arrays[f'accepted_{mode}'],
        arrays[f'observed_{mode}'], arrays['anchor_xyz'], arrays['anchor_body'][0]) for mode in ('visual', 'fusion')}
    np.savez_compressed(destination/f'{seed}.npz', **arrays)
    write_json(destination/f'{seed}-events.json', events)
    print(f'occlusion {seed}: {int(arrays["behind_panel"].sum())} hidden point-frames; '+
          ', '.join(f'{mode} {sum(e["recovered"] for e in items)}/{len(items)} recovered' for mode, items in events.items()), flush=True)


def exploration_evaluate(output, seed):
    destination = output/'exploration'/f'eval-{seed}.json'
    if destination.exists():
        return
    read_json(output/'exploration'/'freeze.json')  # Required before opening final truth.
    source = output/f'evaluation-{seed}'
    metadata = read_json(source/'metadata.json')
    body = BodyModel(output/'bridge'/'model-v7.pt')
    qs = []
    for row in metadata['frames']:
        with np.load(source/'observations'/row['file']) as obs:
            qs.append(obs['q'].copy())
    qs = np.stack(qs)
    predictions, states = {}, {}
    for acquisition in ACQUISITION_SEEDS:
        for policy in POLICIES:
            run = output/'exploration'/f'{acquisition}-{policy}'
            freeze = read_json(run/'freeze.json')
            for budget in BUDGETS:
                checkpoint = run/f'belief-{budget:03d}.npz'
                assert sha(checkpoint) == freeze['checkpoints'][str(budget)]
                with np.load(checkpoint) as saved:
                    state = {k: saved[k].copy() for k in saved.files}
                key = acquisition, policy, budget
                predictions[key] = ResidualBelief.from_state(body, state).predict(qs)[0]
                states[key] = state
    baseline = body(qs)
    truths = []
    for row in metadata['frames']:
        with np.load(source/'evaluation'/row['file']) as truth:
            truths.append(dict(xyz=truth['anchor_xyz'].copy(), body=truth['anchor_body'].copy(), visible=truth['anchor_visible'].copy()))
    truth = stack(truths)
    scores = []
    for key, prediction in predictions.items():
        acquisition, policy, budget = key
        run = output/'exploration'/f'{acquisition}-{policy}'
        with np.load(run/'tracks.npz') as tracks:
            admitted = tracks['valid'][:budget].copy()
            xyz = tracks['xyz'][:budget].copy()
            normalized_travel = float(tracks['normalized_travel'][budget-1])
            rotary = float(tracks['rotary_travel_rad'][budget-1])
            gripper = float(tracks['gripper_travel_m'][budget-1])
            qq = tracks['q'][:budget].copy()
        # Physical anchors depend on actual q; compare on an independent fixture
        # only after the acquired models and their predictions have been frozen.
        fixture = output/'exploration'/'evaluation-fixture'
        fixture.mkdir(exist_ok=True)
        model, data, calibration = make_fixture(fixture)
        saved, _, refpoints = reference()
        anchors = EvaluationLandmarks(model, data, refpoints[saved['selected'], 0], calibration)
        indices = model.jnt_qposadr[model.actuator_trnid[:, 0]]
        expected = []
        for q in qq:
            data.qpos[indices] = q
            right = model.joint('finger_right').qposadr[0]
            data.qpos[right] = q[6]
            mujoco.mj_forward(model, data)
            expected.append(anchors.positions(model, data))
        expected = np.stack(expected)
        correct = admitted & anchors.is_body[None] & (np.linalg.norm(xyz-expected, axis=-1) <= .02)
        coverage = (correct.sum(0) >= 4) & anchors.is_body
        score = dict(acquisition_seed=acquisition, evaluation_seed=seed, policy=policy, budget=budget,
            all_body_rmse_mm=rmse(np.sum((prediction-truth['xyz'])**2, axis=-1), truth['body']),
            visible_body_rmse_mm=rmse(np.sum((prediction-truth['xyz'])**2, axis=-1), truth['body'] & truth['visible']),
            fixed_body_points=int(truth['body'][0].sum()), covered_body_points=int(coverage.sum()),
            accepted_body_purity=float((admitted & anchors.is_body).sum()/max(admitted.sum(), 1)),
            correspondence_precision=float(correct.sum()/max(admitted.sum(), 1)),
            accepted_points_per_frame=float(admitted.sum()/budget),
            normalized_travel=normalized_travel, rotary_travel_rad=rotary, gripper_travel_m=gripper,
            baseline_body_rmse_mm=rmse(np.sum((baseline-truth['xyz'])**2, axis=-1), truth['body']))
        scores.append(score)
    write_json(destination, scores)
    print(f'exploration evaluation {seed} complete: {len(scores)} checkpoints', flush=True)


def evaluate(output):
    freeze = read_json(output/'bridge'/'freeze.json')
    for name in MODES:
        assert sha(output/'bridge'/f'model-{name}.pt') == freeze['models'][name]
    read_json(output/'exploration'/'freeze.json')
    if not (output/'evaluation-protocol.json').exists():
        write_json(output/'evaluation-protocol.json', dict(frozen_utc=utc(),
            source_sha256={name: sha(Path(__file__).with_name(name)) for name in
                           ('body_cycle.py', 'body_memory.py', 'body_fixture.py')},
            occlusion='Panel must be in front AND same physical point visible in a counterfactual render with panel hidden',
            truth_is_evaluation_only=True, spatial_error_denominator='All 28 reference body anchors, including lost points'))
    masks = BodyMasks(SEGMENTATION)
    for seed in EVALUATION_SEEDS:
        collect_data(output/f'evaluation-{seed}', seed)
        bridge_evaluate(output, seed, masks)
        collect_data(output/f'occlusion-data-{seed}', seed, panel=True)
        occlusion_evaluate(output, seed, masks)
        exploration_evaluate(output, seed)
    write_json(output/'complete.json', dict(completed_utc=utc(), evaluation_seeds=list(EVALUATION_SEEDS),
        note='Full results include negative controls and all fixed reference body points'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'acquire', 'evaluate'))
    parser.add_argument('--output', default='outputs/local-check-20261024/e4-body-cycle')
    args = parser.parse_args()
    torch.set_num_threads(4)
    globals()[args.mode](Path(args.output))
