"""E7: train sensor reliability on past runs; freeze before new final runs.

Clean recorded encoder readings supervise training only. Geometry is opened by
score() after predictions have been saved. Front-end visual proposals are made
from an independent causal action rollout and RGB, never current clean labels.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import shutil

import cv2
import mujoco
import numpy as np
import torch

from perception.collect import HOME
from .action_cycle import ROOT as E6, schedule, load_npz
from .action_state import ActionTransition, ActionBodyState
from .body_cycle import reference, read_json, write_json, sha, utc, stack
from .body_fixture import make_fixture, render_rgb, render_truth, EvaluationLandmarks
from .body_memory import BodyModel, BodyMasks
from .recovery import VisualBodyBank, VisualFeedbackCorrection
from .recovery_cycle import BODY, SEGMENTATION
from .action_cycle import BANK
from .trust_state import ReliabilityGate, LearnedBodyState

ROOT = Path('outputs/local-check-20261024/e7-learned-trust')
OLD_TRAIN = (2026102621, 2026102631, 2026102632, 2026102633)
NEW_TRAIN = (2026102701, 2026102702)
DEV = 2026102711
FINAL = (2026102731, 2026102732, 2026102733)
CONDITIONS = ('clean', 'biased_feedback', 'delayed_feedback', 'camera_missing',
              'wrong_rgb', 'bias_and_wrong_rgb', 'bias_and_camera_missing', 'no_feedback')
MODES = ('learned', 'fixed_equal', 'fixed_learned', 'action_only', 'feedback_only', 'e6_rules')
SOURCES = ('trust_state.py','trust_cycle.py','action_state.py','action_cycle.py',
           'body_memory.py','body_fixture.py','recovery.py')


def source_for(output, seed):
    return (E6 if seed in OLD_TRAIN else output)/f'data-{seed}'


def collect(output, seed, count=150, training=False):
    source = output/f'data-{seed}'
    if (source/'metadata.json').exists():
        return source
    source.mkdir(parents=True, exist_ok=True)
    if any(source.iterdir()):
        raise FileExistsError(f'Preserving partial collection: {source}')
    for directory in ('before', 'after', 'evaluation'):
        (source/directory).mkdir()
    model, data, calibration = make_fixture(source, panel=False)
    saved, _, pixels = reference()
    anchors = EvaluationLandmarks(model, data, pixels[saved['selected'], 0], calibration)
    transition = ActionTransition.load(output/'action.pt')
    body = BodyModel(BODY)
    renderer = mujoco.Renderer(model, height=480, width=640)
    indices = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    props = [model.body(f'table_object_{i}').id for i in range(3)]
    fixed = data.xpos[props].copy()
    q = data.qpos[indices].astype(np.float32).copy()
    np.savez_compressed(source/'initial.npz', q=q)
    previous, previous_dt, last_command = q.copy(), .1, HOME.copy()
    rows, hashes = [], {}
    try:
        for i, request in enumerate(schedule(seed, count)):
            q = data.qpos[indices].astype(np.float32).copy()
            target, dt = request['target'], request['duration']
            prediction = transition.predict(q, (q-previous)/previous_dt, target, dt)
            filename = f'{i:05d}.npz'
            np.savez_compressed(source/'before'/filename, target=target, duration=dt,
                                xyz_action=body(prediction)[0])
            hashes[filename] = sha(source/'before'/filename)
            # Two execution faults, unknown to learner; original props stay fixed.
            # Final delay magnitude is larger than the new training magnitude.
            active = 50 <= i < 75 or 110 <= i < 135
            lag = min(dt, .04 if training else .06) if active else 0.
            if lag:
                data.ctrl[:] = last_command
                mujoco.mj_step(model, data, nstep=round(lag/model.opt.timestep))
            data.ctrl[:] = target
            if dt > lag:
                mujoco.mj_step(model, data, nstep=round((dt-lag)/model.opt.timestep))
            mujoco.mj_forward(model, data)
            np.savez_compressed(source/'after'/filename, q=data.qpos[indices].astype(np.float32).copy(),
                left=render_rgb(renderer, data), right=render_rgb(renderer, data, 'entity_right'), time=data.time)
            if not training:
                truth = render_truth(renderer, model, data)
                truth.update(anchors.score_truth(model, data, truth))
                np.savez_compressed(source/'evaluation'/filename, **truth)
            np.testing.assert_array_equal(fixed, data.xpos[props])
            rows.append(dict(file=filename, phase=request['phase'], duration=dt, actuator_delay_s=lag))
            previous, previous_dt, last_command = q, dt, target.copy()
            if (i+1) % 50 == 0:
                print(f'collect {seed}: {i+1}/{count}', flush=True)
    finally:
        renderer.close()
    write_json(source/'metadata.json', dict(seed=seed, frames=rows, calibration=calibration,
        before_action_prediction_hashes=hashes, fixed_body_points=int(anchors.is_body.sum()),
        prop_max_displacement_m=0., training=training, created_utc=utc(),
        execution_fault_hidden_from_learner=True))
    return source


def observations(source):
    """Yield command before opening any corresponding post-action observation."""
    for row in read_json(source/'metadata.json')['frames']:
        with np.load(source/'before'/row['file']) as before:
            yield 'command', dict(target=before['target'].copy(), duration=float(before['duration']))
        yield 'observation', load_npz(source/'after'/row['file'])


def visual_cache(output, source):
    """Sensor proposals; only initial q is read, subsequent q labels are ignored.

    Both correct RGB and PAST replayed RGB are processed, rather than replacing
    visual candidates with artificially clean or directly corrupted joint labels.
    """
    path = output/'visual'/f'{source.name}.npz'
    if path.exists():
        return path
    path.parent.mkdir(exist_ok=True)
    metadata = read_json(source/'metadata.json')
    body, masks = BodyModel(BODY), BodyMasks(SEGMENTATION)
    scale = np.asarray(body.saved['scale'])
    transition = ActionTransition.load(output/'action.pt')
    with np.load(BANK) as a:
        bank = VisualBodyBank(a, metadata['calibration'], references=4, window=7)
    q = load_npz(source/'initial.npz')['q']
    velocity = np.zeros(7, np.float32)
    rows, past, command = [], [], None
    cv2.setRNGSeed(20261027)
    for kind, obs in observations(source):
        if kind == 'command':
            command = obs
            prior = transition.predict(q, velocity, obs['target'], obs['duration'])
            continue
        i = len(rows)
        row = dict(target=command['target'], duration=np.float32(command['duration']), action_q=prior,
                   time=np.float64(obs['time']))
        # q is NEVER used here; only RGB and clean initialization are available.
        pairs = [('clean', obs)]
        if i >= 20:
            pairs.append(('wrong', past[i % 20]))
        for label, image in pairs:
            mask = masks(image['left'], prior)['v7']
            measurement = bank.observe(image['left'], image['right'], mask, prior, body(prior)[0])
            corrector = VisualFeedbackCorrection(body, body.saved['scale'])
            fit = corrector.update(prior, measurement['xyz'], measurement['valid'], obs['time'])
            candidate = corrector.corrected_q(prior).astype(np.float32)
            good = measurement['valid']
            residual = np.linalg.norm(body(candidate)[0][good]-measurement['xyz'][good], axis=-1)
            quality = np.array([good.mean(), np.median(measurement['appearance_cost'][good])/30 if good.any() else 0,
                np.median(residual)/.03 if good.any() else 0,
                np.linalg.norm((candidate-prior)/body.saved['scale']), float(fit['updated'])], np.float32)
            row.update({f'visual_{label}': candidate, f'quality_{label}': quality,
                        f'valid_{label}': np.bool_(fit['updated']), f'xyz_{label}': measurement['xyz'],
                        f'points_{label}': measurement['valid']})
        if i < 20:
            for key in ('visual', 'quality', 'valid', 'xyz', 'points'):
                row[f'{key}_wrong'] = row[f'{key}_clean'].copy()
            past.append(dict(left=obs['left'].copy(), right=obs['right'].copy()))
        rows.append(row)
        velocity = np.clip((prior-q)/command['duration'], -scale*20, scale*20)
        q = prior
        if (i+1) % 40 == 0:
            print(f'visual {source.name}: {i+1}/{len(metadata["frames"])}', flush=True)
    np.savez_compressed(path, **stack(rows))
    write_json(path.with_suffix('.json'), dict(created_utc=utc(), sha256=sha(path),
        q_labels_used=False, evaluation_used=False, action_only_frontend=True,
        visual_matching_still_explicit=True, wrong_images='causal replay of first 20 RGB pairs'))
    return path


def sensor_stream(source, cache, condition, variant=0, training=False):
    """Construct observation-only inputs. The training teacher is loaded separately.

    Fault family/amplitude is never passed to the gate. No future q or RGB is
    required: delayed readings and wrong images always come from the past.
    """
    a = load_npz(cache)
    rows = read_json(source/'metadata.json')['frames']
    readings = np.stack([load_npz(source/'after'/r['file'])['q'] for r in rows])
    count = len(readings)
    feedback = readings.copy()
    rng = np.random.default_rng(int(source.name.split('-')[-1])+variant*53)
    start = 20 if training else 30
    bias = np.zeros(7, np.float32)
    if 'bias' in condition:
        if training:
            bias = rng.uniform(-1, 1, 7).astype(np.float32)*np.array([.07,.035,.045,.06,.04,.05,.002], np.float32)
        else:
            # New sign pattern and magnitude; not E6's constant bias.
            bias = np.array([-.075,.028,.04,-.035,.025,.03,-.0015], np.float32)
        feedback += np.clip((np.arange(count)-start)/7, 0, 1)[:, None]*bias
    if 'delay' in condition:
        lag = (1+variant % 2) if training else 2
        feedback[start:] = readings[np.maximum(np.arange(start, count)-lag, 0)]
    if condition == 'jitter':
        feedback += rng.normal(size=feedback.shape)*np.array([.012,.008,.01,.015,.01,.012,.0005])
    use_wrong = 'wrong' in condition
    switch = np.arange(count) >= start
    a['visual'] = np.where(switch[:, None], a['visual_wrong'], a['visual_clean']) if use_wrong else a['visual_clean'].copy()
    a['quality'] = np.where(switch[:, None], a['quality_wrong'], a['quality_clean']) if use_wrong else a['quality_clean'].copy()
    valid = np.where(switch, a['valid_wrong'], a['valid_clean']) if use_wrong else a['valid_clean'].copy()
    if 'camera_missing' in condition:
        valid[40:95] = False
        a['quality'][40:95] = 0
    available = np.stack((np.full(count, condition != 'no_feedback'), valid), -1)
    # Remove the absent measurement's value, not just its flag.
    a.update(feedback=np.where(available[:, :1], feedback, a['action_q']), available=available,
             visual=np.where(valid[:, None], a['visual'], a['action_q']))
    return {k: a[k] for k in ('target', 'duration', 'visual', 'quality', 'feedback', 'available')}


def training_set(output, seeds, training=True):
    examples = []
    conditions = (*CONDITIONS, 'jitter')
    for seed in seeds:
        source = source_for(output, seed)
        cache = visual_cache(output, source)
        q = np.stack([load_npz(source/'after'/r['file'])['q'] for r in read_json(source/'metadata.json')['frames']])
        initial = load_npz(source/'initial.npz')['q']
        for condition in conditions:
            repeats = 3 if training and ('bias' in condition or condition in ('clean','camera_missing','wrong_rgb')) else 1
            for variant in range(repeats):
                a = sensor_stream(source, cache, condition, variant, training=training)
                # Equal 120-frame training sequences; no truth geometry is read.
                for start in range(0, len(q)-119, 120):
                    examples.append(dict(**{k: v[start:start+120] for k, v in a.items()}, teacher=q[start:start+120],
                        initial=initial if start == 0 else q[start-1]))
    return {k: torch.from_numpy(np.stack([x[k] for x in examples])).float() for k in examples[0]}


def rollout_train(gate, transition, body, a, mode='learned', fixed_logits=None, reliability_supervision=0.):
    batch, length = a['duration'].shape
    q = a['initial']
    velocity, old_f, old_v = torch.zeros_like(q), q, q
    weights = torch.tensor([1., 0., 0.]).expand(batch, -1)
    hidden = torch.zeros(batch, gate.hidden_size)
    predictions, contributions, reliability_losses = [], [], []
    for i in range(length):
        prior = transition(q, velocity, a['target'][:, i], a['duration'][:, i])
        available = a['available'][:, i].bool()
        f = torch.where(available[:, :1], a['feedback'][:, i], prior)
        v = torch.where(available[:, 1:2], a['visual'][:, i], prior)
        if mode == 'learned':
            new_q, weights, hidden = gate(prior, f, v, q, old_f, old_v, transition.scale[None],
                a['quality'][:, i], available, a['duration'][:, i], weights, hidden)
        else:
            logits = torch.zeros(batch, 3) if fixed_logits is None else fixed_logits.expand(batch, -1)
            mask = torch.cat((torch.ones(batch, 1, dtype=torch.bool), available), -1)
            weights = logits.masked_fill(~mask, -torch.inf).softmax(-1)
            new_q = prior+weights[:, 1:2]*(f-prior)+weights[:, 2:3]*(v-prior)
        if reliability_supervision:
            # Training-only comparative reliability targets, detached so the
            # estimator cannot make a candidate worse to manipulate its label.
            candidates = torch.stack((prior, f, v), 1)
            errors = (((candidates-a['teacher'][:,i:i+1])/transition.scale)**2).mean(-1).detach()
            mask = torch.cat((torch.ones(batch,1,dtype=torch.bool),available),-1)
            oracle = (-errors/.00001).masked_fill(~mask,-torch.inf).softmax(-1)
            reliability_losses.append(-(oracle*weights.clamp_min(1e-8).log()).sum(-1).mean())
        velocity = ((new_q-q)/a['duration'][:, i:i+1]).clamp(-transition.scale*20, transition.scale*20)
        q, old_f, old_v = new_q, f, v
        predictions.append(q)
        contributions.append(weights)
        # Truncated backpropagation through 20 past observation updates.
        if (i+1) % 20 == 0:
            q, velocity, hidden, weights = (x.detach() for x in (q, velocity, hidden, weights))
            old_f, old_v = old_f.detach(), old_v.detach()
    predicted = torch.stack(predictions, 1)
    target = a['teacher']
    loss = (((predicted-target)/transition.scale)**2).mean()
    # Frozen learned decoder consistency, NOT simulator body coordinates.
    xyz = body.model(predicted.reshape(-1, 7))
    with torch.no_grad():
        teacher_xyz = body.model(target.reshape(-1, 7))
    loss = loss+((xyz-teacher_xyz)/.03).square().mean()
    if reliability_losses:
        loss = loss+reliability_supervision*torch.stack(reliability_losses).mean()
    return loss, predicted, torch.stack(contributions, 1)


def train(output, epochs=180):
    output.mkdir(parents=True, exist_ok=True)
    if (output/'training-freeze.json').exists():
        return
    if not (output/'action.pt').exists():
        shutil.copyfile(E6/'action.pt', output/'action.pt')
    for seed in NEW_TRAIN:
        collect(output, seed, training=True)
    collect(output, DEV, training=False)
    training = training_set(output, (*OLD_TRAIN, *NEW_TRAIN))
    validation = training_set(output, (DEV,), training=False)
    transition, body = ActionTransition.load(output/'action.pt'), BodyModel(BODY)
    for module in (transition, body.model):
        module.requires_grad_(False)
    history = {}
    for mode in ('learned', 'fixed_learned'):
        torch.manual_seed(2026102700)
        gate = ReliabilityGate()
        fixed_logits = torch.nn.Parameter(torch.zeros(3))
        parameters = gate.parameters() if mode == 'learned' else [fixed_logits]
        optimizer = torch.optim.Adam(parameters, lr=.002 if mode == 'learned' else .05)
        best, best_state, trace = float('inf'), None, []
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss, _, _ = rollout_train(gate, transition, body, training, mode, fixed_logits,
                                      reliability_supervision=.2 if mode == 'learned' else 0.)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(gate.parameters()) if mode == 'learned' else [fixed_logits], 2.)
            optimizer.step()
            with torch.no_grad():
                val_loss, _, _ = rollout_train(gate, transition, body, validation, mode, fixed_logits)
                clean_loss, _, _ = rollout_train(gate, transition, body,
                    {k:v[:1] for k,v in validation.items()}, mode, fixed_logits)
            value = float(val_loss+3*clean_loss)
            trace.append(dict(epoch=epoch, train=float(loss.detach()), validation=float(val_loss),
                              clean_validation=float(clean_loss), selection_score=value))
            if value < best:
                best = value
                best_state = copy.deepcopy(gate.state_dict()) if mode == 'learned' else fixed_logits.detach().clone()
            if (epoch+1) % 20 == 0:
                print(f'{mode} {epoch+1}/{epochs}: train={float(loss.detach()):.6f}, dev={value:.6f}, best={best:.6f}', flush=True)
        if mode == 'learned':
            torch.save(dict(hidden=gate.hidden_size, state_dict=best_state), output/'gate.pt')
        else:
            torch.save(dict(logits=best_state), output/'fixed.pt')
        history[mode] = trace
    write_json(output/'training-history.json', history)
    write_json(output/'training-freeze.json', dict(created_utc=utc(), training_seeds=[*OLD_TRAIN,*NEW_TRAIN],
        dev_seed=DEV, training_sequences=len(training['initial']), sequence_length=120,
        teacher='clean recorded measured joint readings; no geometry or fault labels as gate input',
        old_e6_final_repurposed_as_training=True, epochs=epochs, truncation_frames=20,
        reliability_supervision=dict(weight=.2,temperature_normalized_mse=.00001,
            labels='candidate errors against clean TRAINING readings; unavailable sources masked'),
        validation_selection='mean state/decoder loss over conditions plus three times clean loss; no geometry',
        normal_training_examples_repeated=3,
        fixed_baseline_lr=.05,
        models={k: sha(output/f'{k}.pt') for k in ('gate','fixed','action')},
        sources={s: sha(Path(__file__).with_name(s)) for s in SOURCES},
        visual_caches={p.name: sha(p) for p in (output/'visual').glob('*.npz')},
        body_sha256=sha(BODY), segmentation_sha256=sha(SEGMENTATION), bank_sha256=sha(BANK)))


def predict(output, seed, condition):
    source = source_for(output, seed)
    cache = visual_cache(output, source)
    destination = output/'predictions'/f'data-{seed}-{condition}.npz'
    if destination.exists():
        return destination
    destination.parent.mkdir(exist_ok=True)
    sensors = sensor_stream(source, cache, condition, training=False)
    initial = load_npz(source/'initial.npz')['q']
    body = BodyModel(BODY)
    scale = np.asarray(body.saved['scale'])
    transition = ActionTransition.load(output/'action.pt')
    gate = ReliabilityGate.load(output/'gate.pt')
    learned = LearnedBodyState(transition, gate, body, initial)
    rule = ActionBodyState(transition, body, initial, feedback_gain=.9, innovation_gate_m=.008,
                           feedback_enabled=condition != 'no_feedback')
    fixed_logits = torch.load(output/'fixed.pt', weights_only=True)['logits'].numpy()
    fixed_states = {mode: dict(q=initial.copy(), velocity=np.zeros(7)) for mode in ('fixed_equal','fixed_learned','action_only')}
    visuals = load_npz(cache)
    rows = []
    # No teacher or privileged evaluation file is opened by this inference loop.
    for i in range(len(sensors['duration'])):
        target, dt = sensors['target'][i], float(sensors['duration'][i])
        prior = learned.forecast(target, dt)
        prior_q = learned.pending[0].copy()
        rule.forecast(target, dt)
        f_ok, v_ok = sensors['available'][i]
        f, v = sensors['feedback'][i], sensors['visual'][i]
        result = learned.finish(f if f_ok else None, v, sensors['quality'][i], v_ok)
        rule.feedback(f if f_ok else None)
        label = 'wrong' if 'wrong' in condition and i >= 30 else 'clean'
        rule_result = rule.finish(visuals[f'xyz_{label}'][i] if v_ok else None,
                                 visuals[f'points_{label}'][i] if v_ok else None, visuals['time'][i])
        row = dict(prediction_learned=result['xyz'], q_learned=result['q'], weights=result['weights'],
                   q_prior=prior_q, q_feedback=f, q_visual=v,
                   prior_learned=prior, available=sensors['available'][i],
                   prediction_feedback_only=body(f if f_ok else visuals['action_q'][i])[0],
                   prediction_e6_rules=rule_result['xyz'])
        for mode, state in fixed_states.items():
            p = transition.predict(state['q'], state['velocity'], target, dt)
            if mode == 'action_only':
                q = p
            else:
                logits = np.zeros(3) if mode == 'fixed_equal' else fixed_logits.copy()
                valid = np.array([True, f_ok, v_ok])
                weights = np.where(valid, np.exp(logits-logits.max()), 0.)
                weights /= weights.sum()
                q = p+weights[1]*(f-p)+weights[2]*(v-p)
            row[f'prediction_{mode}'] = body(q)[0]
            state['velocity'] = np.clip((q-state['q'])/dt, -scale*20, scale*20)
            state['q'] = q
        rows.append(row)
    np.savez_compressed(destination, **stack(rows))
    write_json(destination.with_suffix('.json'), dict(created_utc=utc(), prediction_sha256=sha(destination),
        teacher_used=False, geometry_used=False, sensor_condition_not_passed_to_gate=True,
        observation_inputs_sha256=sha(cache),
        e6_rules_same_frontend=True, feedback_only_fallback_to_action_if_absent=True))
    return destination


def score(output, seed):
    source = source_for(output, seed)
    paths = [output/'predictions'/f'data-{seed}-{c}.npz' for c in CONDITIONS]
    for p in paths:
        assert sha(p) == read_json(p.with_suffix('.json'))['prediction_sha256']
    # ALL predictions must exist and be hashed before geometry is opened.
    metadata = read_json(source/'metadata.json')
    truths = [load_npz(source/'evaluation'/r['file']) for r in metadata['frames']]
    truth = np.stack([t['anchor_xyz'] for t in truths])
    mask = np.stack([t['anchor_body'] for t in truths])
    assert (mask.sum(-1) == 28).all()
    faulty = np.array([r['actuator_delay_s'] > 0 for r in metadata['frames']])
    result = dict(seed=seed, frames=len(mask), points=28, conditions={})
    for condition, path in zip(CONDITIONS, paths):
        a = load_npz(path)
        values = {}
        for mode in MODES:
            squared = np.sum((a[f'prediction_{mode}']-truth)**2, -1)
            values[mode] = dict(body_rmse_mm=float(np.sqrt(squared[mask].mean())*1000),
                actuator_fault_rmse_mm=float(np.sqrt(squared[mask & faulty[:, None]].mean())*1000),
                after_intervention_rmse_mm=float(np.sqrt(squared[mask & (np.arange(len(mask)) >= 40)[:, None]].mean())*1000))
        result['conditions'][condition] = dict(modes=values, mean_weights=a['weights'].mean(0).tolist(),
            post_weights=a['weights'][40:].mean(0).tolist(),
            actuator_fault_weights=a['weights'][faulty].mean(0).tolist())
    (output/'scores').mkdir(exist_ok=True)
    write_json(output/'scores'/f'data-{seed}.json', result)
    print('scored', seed, {c: round(r['modes']['learned']['body_rmse_mm'], 3) for c,r in result['conditions'].items()}, flush=True)
    return result


def develop(output):
    train(output)
    for condition in CONDITIONS:
        predict(output, DEV, condition)
    score(output, DEV)


def freeze(output):
    path = output/'final-protocol.json'
    if path.exists():
        raise FileExistsError('Final protocol already frozen')
    training = read_json(output/'training-freeze.json')
    for name, digest in training['sources'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    write_json(path, dict(frozen_utc=utc(), final_seeds=list(FINAL), conditions=list(CONDITIONS),
        frames_per_seed=150, geometry_points_denominator=28,
        source_digests={s: sha(Path(__file__).with_name(s)) for s in SOURCES},
        models=training['models'], body_sha256=sha(BODY), bank_sha256=sha(BANK), segmentation_sha256=sha(SEGMENTATION),
        clean_tolerance_mm=.5, selected_using='development clean recorded feedback loss only; no test tuning',
        test_faults='new bias signs/magnitudes, two-frame sensor lag; 60ms hidden actuation delay vs train 40ms',
        missing_camera_frames=[40,94], bias_start=30, bias_ramp=7,
        fault_labels_as_gate_input=False, uncertainty_calibrated=False,
        comparisons='shared visual proposal frontend; E6-style rules, not an identical rerun of E6 frontend',
        limitations=['fixed camera and morphology','synthetic sensor faults','clean training teacher','explicit RGB matcher and inverse solver']))


def final(output):
    protocol = read_json(output/'final-protocol.json')
    for name, digest in protocol['source_digests'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    for name, digest in protocol['models'].items():
        assert sha(output/f'{name}.pt') == digest
    for seed in FINAL:
        source = collect(output, seed, training=False)
        visual_cache(output, source)
        for condition in CONDITIONS:
            predict(output, seed, condition)
    # Score only after ALL three independent trajectories' predictions are saved.
    for seed in FINAL:
        score(output, seed)
    write_json(output/'complete.json', dict(completed_utc=utc(), seeds=list(FINAL)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('train','develop','freeze','final'))
    parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    globals()[args.mode](args.output)
