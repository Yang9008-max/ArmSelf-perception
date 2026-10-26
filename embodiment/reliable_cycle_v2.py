"""E8B: sharper reliability supervision and stronger clean/missing-feedback balance.

E8A remains frozen. This version uses NEW calibration and final trajectories.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import shutil

import cv2
import numpy as np
import torch

from . import trust_cycle as e7
from .action_state import ActionTransition
from .body_cycle import read_json, write_json, sha, utc, stack
from .body_memory import BodyModel
from .reliable_state import JointReliability, ReliableBodyState

ROOT = Path('outputs/local-check-20261024/e8b-reliable-state')
TRAIN = (*e7.OLD_TRAIN, *e7.NEW_TRAIN)
DEV = (e7.DEV, 2026102811)
CALIBRATION = 2026102851
FINAL = (2026102861, 2026102862, 2026102863)
CONDITIONS = e7.CONDITIONS
PROFILES = ('per_joint', 'bias_state')
SOURCES = ('reliable_state.py', 'reliable_cycle_v2.py', *e7.SOURCES)
CONFIG = dict(hidden=64, epochs=180, learning_rate=.002, gradient_clip=2.,
              truncation_frames=20, reliability_weight=.10, reliability_temperature=.00001,
              bias_weight=.05, bias_loss_scale=.2, bias_bound=.8, uncertainty_weight=.001,
              dev_clean_weight=8., init_feedback_logit=2., seed=2026102800)


def source_for(output, seed):
    if seed in e7.OLD_TRAIN:
        return e7.E6/f'data-{seed}'
    return (e7.ROOT if seed in (*e7.NEW_TRAIN, e7.DEV) else output)/f'data-{seed}'


def cache_for(output, seed):
    source = source_for(output, seed)
    return e7.visual_cache(e7.ROOT if seed in (*TRAIN, e7.DEV) else output, source)


def dataset(output, seeds, training):
    examples, labels = [], []
    for seed in seeds:
        source, cache = source_for(output, seed), cache_for(output, seed)
        rows = read_json(source/'metadata.json')['frames']
        teacher = np.stack([e7.load_npz(source/'after'/r['file'])['q'] for r in rows])
        initial = e7.load_npz(source/'initial.npz')['q']
        for condition in (*CONDITIONS, 'jitter'):
            repeats = (5 if condition=='clean' else 3 if ('bias' in condition or condition in ('camera_missing','wrong_rgb','no_feedback')) else 1) if training else 1
            for variant in range(repeats):
                sensors = e7.sensor_stream(source, cache, condition, variant, training=training)
                # Same 120-frame training budget as E7; full dev trajectories.
                length = 120 if training else len(teacher)
                examples.append(dict(**{k:v[:length] for k,v in sensors.items()},
                                     initial=initial, teacher=teacher[:length]))
                labels.append(dict(seed=seed, condition=condition, variant=variant))
    return ({k:torch.from_numpy(np.stack([x[k] for x in examples])).float()
             for k in examples[0]}, labels)


def rollout(gate, transition, body, a, auxiliary=True):
    batch, length = a['duration'].shape
    q = a['initial']
    velocity, old_f, old_v = torch.zeros_like(q), q, q
    weights = torch.tensor([1.,0.,0.]).repeat(batch,7,1)
    bias, hidden = torch.zeros_like(q), torch.zeros(batch,gate.hidden_size)
    predictions, deviations, contributions, biases, aux = [], [], [], [], []
    for i in range(length):
        prior = transition(q,velocity,a['target'][:,i],a['duration'][:,i])
        available = a['available'][:,i].bool()
        f = torch.where(available[:,:1],a['feedback'][:,i],prior)
        v = torch.where(available[:,1:2],a['visual'][:,i],prior)
        new_q,weights,bias,sigma,hidden = gate(prior,f,v,q,old_f,old_v,transition.scale[None],
            a['quality'][:,i],available,a['duration'][:,i],weights,bias,hidden)
        if auxiliary:
            # Teachers only define losses. They never enter the gate/state API.
            candidates = torch.stack((prior,f-bias*transition.scale,v),-1)
            errors = ((candidates-a['teacher'][:,i,:,None])/transition.scale[None,:,None]).square().detach()
            mask = torch.cat((torch.ones(batch,1,dtype=torch.bool),available),-1)
            oracle = (-errors/CONFIG['reliability_temperature']).masked_fill(~mask[:,None],-torch.inf).softmax(-1)
            ce = -(oracle*weights.clamp_min(1e-8).log()).sum(-1).mean()
            truth_bias = ((f-a['teacher'][:,i])/transition.scale).detach()
            b_loss = (((bias-truth_bias)/CONFIG['bias_loss_scale']).square()*available[:,:1]).mean()
            error = ((new_q-a['teacher'][:,i])/transition.scale).detach()
            nll = (.5*(error/sigma).square()+sigma.log()).mean()
            aux.append(CONFIG['reliability_weight']*ce+
                       (CONFIG['bias_weight']*b_loss if gate.bias_enabled else 0.)+
                       CONFIG['uncertainty_weight']*nll)
        velocity = ((new_q-q)/a['duration'][:,i:i+1]).clamp(-transition.scale*20,transition.scale*20)
        q,old_f,old_v = new_q,f,v
        predictions.append(q)
        contributions.append(weights)
        biases.append(bias)
        deviations.append(sigma)
        if (i+1)%CONFIG['truncation_frames']==0:
            q,velocity,hidden,weights,bias,old_f,old_v = (x.detach() for x in
                (q,velocity,hidden,weights,bias,old_f,old_v))
    predicted = torch.stack(predictions,1)
    state_loss = ((predicted-a['teacher'])/transition.scale).square().mean()
    xyz = body.model(predicted.reshape(-1,7))
    with torch.no_grad():
        target_xyz = body.model(a['teacher'].reshape(-1,7))
    state_loss = state_loss+((xyz-target_xyz)/.03).square().mean()
    loss = state_loss+(torch.stack(aux).mean() if aux else 0.)
    return loss,state_loss,predicted,torch.stack(deviations,1),torch.stack(contributions,1),torch.stack(biases,1)


def train(output, epochs=180):
    output.mkdir(parents=True,exist_ok=True)
    if (output/'training-freeze.json').exists():
        raise FileExistsError('Training already selected; preserve this experiment')
    for name in ('action','gate','fixed'):
        if not (output/f'{name}.pt').exists():
            shutil.copyfile(e7.ROOT/f'{name}.pt',output/f'{name}.pt')
    for seed in (DEV[1],CALIBRATION):
        e7.collect(output,seed,training=False)
        cache_for(output,seed)
    training,train_labels = dataset(output,TRAIN,True)
    validation,dev_labels = dataset(output,DEV,False)
    np.savez_compressed(output/'training-inputs.npz',**{k:v.numpy() for k,v in training.items()})
    write_json(output/'data-split.json',dict(training=train_labels,development=dev_labels,
        calibration_seed=CALIBRATION,final_seeds=list(FINAL),
        unique_training_steps=720,training_input_sha256=sha(output/'training-inputs.npz')))
    transition,body = ActionTransition.load(output/'action.pt'),BodyModel(e7.BODY)
    transition.requires_grad_(False)
    body.model.requires_grad_(False)
    clean_ids = [i for i,x in enumerate(dev_labels) if x['condition']=='clean']
    clean = {k:v[clean_ids] for k,v in validation.items()}
    history,bests = {},{}
    for profile in PROFILES:
        torch.manual_seed(CONFIG['seed'])
        gate = JointReliability(CONFIG['hidden'],profile=='bias_state',CONFIG['bias_bound'])
        optimizer = torch.optim.Adam(gate.parameters(),lr=CONFIG['learning_rate'])
        best,best_epoch,best_state,trace = float('inf'),None,None,[]
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss,*_ = rollout(gate,transition,body,training)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite training loss {profile}:{epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(),CONFIG['gradient_clip'])
            optimizer.step()
            with torch.no_grad():
                _,val,*_ = rollout(gate,transition,body,validation,False)
                _,normal,*_ = rollout(gate,transition,body,clean,False)
            value = float(val+CONFIG['dev_clean_weight']*normal)
            trace.append(dict(epoch=epoch+1,training=float(loss.detach()),validation=float(val),
                              clean=float(normal),selection=value))
            if value<best:
                best,best_epoch,best_state = value,epoch+1,copy.deepcopy(gate.state_dict())
            if (epoch+1)%10==0:
                print(f'{profile} {epoch+1}/{epochs}: train={float(loss.detach()):.5f} dev={value:.5f} best={best:.5f}@{best_epoch}',flush=True)
        gate.load_state_dict(best_state)
        torch.save(dict(hidden=gate.hidden_size,bias_enabled=gate.bias_enabled,bias_bound=gate.bias_bound,
                        state_dict=best_state),output/f'{profile}.pt')
        history[profile],bests[profile] = trace,dict(epoch=best_epoch,selection=best,
            parameters=sum(p.numel() for p in gate.parameters()),sha256=sha(output/f'{profile}.pt'))
        write_json(output/'training-history.json',history)
        write_json(output/'development-selection.json',bests)
    selected = min(bests,key=lambda k:bests[k]['selection'])
    write_json(output/'training-freeze.json',dict(created_utc=utc(),selected=selected,profiles=bests,
        configuration={**CONFIG,'epochs':epochs},training_seeds=list(TRAIN),development_seeds=list(DEV),
        training_sequences=len(train_labels),unique_training_steps=720,
        selection='mean normalized state and frozen-decoder loss plus eight times clean development loss',
        geometry_used_for_training_or_selection=False,old_e7_final_used=False,old_e8_final_used=False,
        previous_experiment='E8A informed the hypothesis; its data did not enter training or selection',
        sources={s:sha(Path(__file__).with_name(s)) for s in SOURCES}))
    print('Selected',selected,bests,flush=True)


def predict(output,seed,condition):
    destination = output/'e8-predictions'/f'data-{seed}-{condition}.npz'
    if destination.exists():
        raise FileExistsError(f'Preserving prediction {destination}')
    destination.parent.mkdir(exist_ok=True)
    source,cache = source_for(output,seed),cache_for(output,seed)
    sensors = e7.sensor_stream(source,cache,condition,training=False)
    initial = e7.load_npz(source/'initial.npz')['q']
    body,transition = BodyModel(e7.BODY),ActionTransition.load(output/'action.pt')
    states = {p:ReliableBodyState(transition,JointReliability.load(output/f'{p}.pt'),body,initial) for p in PROFILES}
    rows = []
    for i in range(len(sensors['duration'])):
        row = dict(available=sensors['available'][i])
        for profile,state in states.items():
            prior = state.forecast(sensors['target'][i],float(sensors['duration'][i]))
            f_ok,v_ok = sensors['available'][i]
            result = state.finish(sensors['feedback'][i] if f_ok else None,
                                  sensors['visual'][i],sensors['quality'][i],v_ok)
            row.update({f'{k}_{profile}':v for k,v in result.items()})
            row[f'prior_{profile}'] = prior
        rows.append(row)
    np.savez_compressed(destination,**stack(rows))
    write_json(destination.with_suffix('.json'),dict(created_utc=utc(),sha256=sha(destination),
        teacher_used=False,geometry_used=False,fault_condition_as_model_input=False,
        visual_cache_sha256=sha(cache)))
    return destination


def calibrate(output):
    training = read_json(output/'training-freeze.json')
    source = source_for(output,CALIBRATION)
    rows = read_json(source/'metadata.json')['frames']
    teacher = np.stack([e7.load_npz(source/'after'/r['file'])['q'] for r in rows])
    scale = np.asarray(BodyModel(e7.BODY).saved['scale'])
    scores = {p:[] for p in PROFILES}
    for condition in CONDITIONS:
        path = output/'e8-predictions'/f'data-{CALIBRATION}-{condition}.npz'
        if not path.exists():
            predict(output,CALIBRATION,condition)
        a = e7.load_npz(path)
        for p in PROFILES:
            # 90% simultaneous seven-joint box at a time step, pooled conditions.
            scores[p].append((np.abs((a[f'q_{p}']-teacher)/scale)/a[f'sigma_normalized_{p}']).max(-1))
    result = {}
    for p,parts in scores.items():
        values = np.concatenate(parts)
        rank = min(len(values),int(np.ceil((len(values)+1)*.9)))
        result[p] = dict(factor=float(np.sort(values)[rank-1]),samples=len(values),nominal_coverage=.9,
                         simultaneous_joints=7,calibration_coverage=float((values<=np.sort(values)[rank-1]).mean()))
    write_json(output/'calibration.json',dict(created_utc=utc(),seed=CALIBRATION,profiles=result,
        selected=training['selected'],state_model_retrained=False,
        geometry_used=False,coverage_guarantee='empirical; correlated frames and limited synthetic fault families'))


def freeze(output):
    path = output/'final-protocol.json'
    if path.exists():
        raise FileExistsError('Final protocol already frozen')
    training = read_json(output/'training-freeze.json')
    calibration = read_json(output/'calibration.json')
    assert set(TRAIN).isdisjoint((*DEV,CALIBRATION,*FINAL))
    assert set(DEV).isdisjoint((CALIBRATION,*FINAL))
    for name,digest in training['sources'].items():
        assert sha(Path(__file__).with_name(name))==digest
    write_json(path,dict(frozen_utc=utc(),selected=training['selected'],final_seeds=list(FINAL),
        conditions=list(CONDITIONS),frames_per_seed=150,body_denominator=28,
        source_digests=training['sources'],
        models={p:sha(output/f'{p}.pt') for p in (*PROFILES,'action','gate','fixed')},
        dependencies=dict(body=sha(e7.BODY),bank=sha(e7.BANK),segmentation=sha(e7.SEGMENTATION)),
        calibration_sha256=sha(output/'calibration.json'),
        goals=dict(wrong_rgb_relative_improvement=.10,bias_relative_improvement=.05,
                   clean_vs_feedback_tolerance_mm=.5,clean_vs_e7_tolerance_mm=.3,
                   other_condition_regression_warning_mm=.5,nominal_joint_box_coverage=.9),
        comparison='all baselines share same RGB matching proposals, fixed anchors and sensor streams',
        synthetic_final_faults='E7 sensor intervention schedule; hidden 60ms actuation delay',
        geometric_truth_used_for_selection=False))
    print('Frozen final protocol:',path,flush=True)


def verify_freeze(output):
    protocol = read_json(output/'final-protocol.json')
    for name,digest in protocol['source_digests'].items():
        assert sha(Path(__file__).with_name(name))==digest,name
    for name,digest in protocol['models'].items():
        assert sha(output/f'{name}.pt')==digest,name
    assert sha(output/'calibration.json')==protocol['calibration_sha256']
    assert sha(e7.BODY)==protocol['dependencies']['body']
    assert sha(e7.BANK)==protocol['dependencies']['bank']
    assert sha(e7.SEGMENTATION)==protocol['dependencies']['segmentation']
    return protocol


def score(output,seed):
    protocol = verify_freeze(output)
    # Verify every final prediction before ANY final geometry is opened here.
    for final_seed in FINAL:
        for condition in CONDITIONS:
            for directory,key in (('predictions','prediction_sha256'),('e8-predictions','sha256')):
                path = output/directory/f'data-{final_seed}-{condition}.npz'
                assert sha(path)==read_json(path.with_suffix('.json'))[key]
    source = source_for(output,seed)
    meta = read_json(source/'metadata.json')
    truths = [e7.load_npz(source/'evaluation'/r['file']) for r in meta['frames']]
    xyz,mask = np.stack([t['anchor_xyz'] for t in truths]),np.stack([t['anchor_body'] for t in truths])
    assert (mask.sum(-1)==28).all()
    teacher = np.stack([e7.load_npz(source/'after'/r['file'])['q'] for r in meta['frames']])
    scale = np.asarray(BodyModel(e7.BODY).saved['scale'])
    faulty = np.array([r['actuator_delay_s']>0 for r in meta['frames']])
    factors = read_json(output/'calibration.json')['profiles']
    result = dict(seed=seed,frames=len(mask),points=28,conditions={})
    for condition in CONDITIONS:
        a = e7.load_npz(output/'e8-predictions'/f'data-{seed}-{condition}.npz')
        baseline = e7.load_npz(output/'predictions'/f'data-{seed}-{condition}.npz')
        modes = {}
        for profile in PROFILES:
            assert np.isfinite(a[f'xyz_{profile}']).all()
            squared = np.sum((a[f'xyz_{profile}']-xyz)**2,-1)
            error = (a[f'q_{profile}']-teacher)/scale
            radius = a[f'sigma_normalized_{profile}']*factors[profile]['factor']
            covered = np.abs(error)<=radius
            weights = a[f'weights_{profile}']
            assert np.all(weights[~a['available'][:,0],:,1]==0)
            assert np.all(weights[~a['available'][:,1],:,2]==0)
            modes[profile] = dict(body_rmse_mm=float(np.sqrt(squared[mask].mean())*1000),
                after_intervention_rmse_mm=float(np.sqrt(squared[mask & (np.arange(len(mask))>=40)[:,None]].mean())*1000),
                actuator_fault_rmse_mm=float(np.sqrt(squared[mask & faulty[:,None]].mean())*1000),
                joint_normalized_rmse=float(np.sqrt((error**2).mean())),
                joint_box_coverage=float(covered.all(-1).mean()),joint_marginal_coverage=float(covered.mean()),
                interval_mean_width_normalized=float((2*radius).mean()),
                interval_width_per_joint=(2*radius*scale).mean(0).tolist(),
                post_weights=weights[40:].mean((0,1)).tolist(),
                post_weights_per_joint=weights[40:].mean(0).tolist(),
                predicted_bias_normalized=a[f'bias_normalized_{profile}'][40:].mean(0).tolist())
        # Compare original E7 and simple alternatives on these NEW final data.
        for mode in e7.MODES:
            squared = np.sum((baseline[f'prediction_{mode}']-xyz)**2,-1)
            modes[f'e7_{mode}'] = dict(body_rmse_mm=float(np.sqrt(squared[mask].mean())*1000),
                after_intervention_rmse_mm=float(np.sqrt(squared[mask & (np.arange(len(mask))>=40)[:,None]].mean())*1000),
                actuator_fault_rmse_mm=float(np.sqrt(squared[mask & faulty[:,None]].mean())*1000))
        result['conditions'][condition] = dict(modes=modes)
    (output/'scores').mkdir(exist_ok=True)
    write_json(output/'scores'/f'data-{seed}.json',result)
    print('Scored',seed,{c:round(r['modes'][protocol['selected']]['body_rmse_mm'],3) for c,r in result['conditions'].items()},flush=True)
    return result


def final(output):
    verify_freeze(output)
    if (output/'complete.json').exists():
        raise FileExistsError('Final evaluation already complete')
    for seed in FINAL:
        source = e7.collect(output,seed,training=False)
        e7.visual_cache(output,source)
        for condition in CONDITIONS:
            e7.predict(output,seed,condition)
            path = output/'e8-predictions'/f'data-{seed}-{condition}.npz'
            if not path.exists():
                predict(output,seed,condition)
    for seed in FINAL:
        score(output,seed)
    write_json(output/'complete.json',dict(completed_utc=utc(),seeds=list(FINAL)))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('train','calibrate','freeze','final'))
    parser.add_argument('--output',type=Path,default=ROOT)
    parser.add_argument('--epochs',type=int,default=CONFIG['epochs'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    if args.mode=='train':
        train(args.output,args.epochs)
    else:
        globals()[args.mode](args.output)
