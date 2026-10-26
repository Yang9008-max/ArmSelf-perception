"""E12: dependency-aware competing body explanations and actual state updates."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import numpy as np
import mujoco
import torch

from . import conflict_cycle as e10
from . import trust_cycle as e7
from . import reliable_cycle_v2 as e8
from . import extent_cycle_v2 as e9
from .action_state import ActionTransition
from .body_cycle import reference, read_json, write_json, sha, utc, stack
from .body_fixture import make_fixture, render_rgb, render_truth, EvaluationLandmarks
from .body_memory import BodyModel
from .conflict_state import recovery_event
from .hypothesis_state import AtlasMeasurements, AtlasFitter, CompetingBodyState
from .reliable_state import JointReliability, ReliableBodyState
from .surface_extent import SurfaceAtlas

ROOT = Path('outputs/local-check-20261024/e12-branch-body')
DEV = 2026103211
FINAL = (2026103231, 2026103232, 2026103233)
CONDITIONS = (*e10.CONDITIONS, 'execution_hold', 'twin_moving', 'twin_held', 'hold_feedback_bias', 'hold_no_camera')
MODES = ('branch', 'e11', 'e8', 'single_surface', 'no_override', 'action', 'feedback')
PROFILES = {'c2_a1':(2,1.), 'c3_a1':(3,1.), 'c2_a15':(2,1.5), 'c3_a15':(3,1.5)}
SOURCES = ('branch_cycle.py', 'branch_state.py', 'hypothesis_cycle.py', 'hypothesis_state.py')
from .branch_state import BranchBodyState
from . import hypothesis_cycle as e11
HELD = ('execution_hold','twin_held','hold_feedback_bias','hold_no_camera')


def source_for(output, seed, condition):
    held = condition in HELD
    return output/f'data-{seed}{"-held" if held else ""}'


def collect_held(output, seed):
    source = source_for(output, seed, 'execution_hold')
    if (source/'metadata.json').exists():
        return source
    source.mkdir(parents=True, exist_ok=True)
    if any(source.iterdir()):
        raise FileExistsError(f'Preserve partial collection: {source}')
    for name in ('before', 'after', 'evaluation'):
        (source/name).mkdir()
    model, data, cal = make_fixture(source, panel=True)
    saved, _, pixels = reference()
    anchors = EvaluationLandmarks(model, data, pixels[saved['selected'], 0], cal)
    indices = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    props = [model.body(f'table_object_{i}').id for i in range(3)]
    fixed = data.xpos[props].copy()
    np.savez_compressed(source/'initial.npz', q=data.qpos[indices].astype(np.float32))
    renderer = mujoco.Renderer(model, height=480, width=640)
    rows, hashes = [], {}
    previous = data.ctrl.copy()
    try:
        for i, (target, dt, phase) in enumerate(e10.schedule(seed)):
            name = f'{i:05d}.npz'
            np.savez_compressed(source/'before'/name, target=target, duration=np.float32(dt))
            hashes[name] = sha(source/'before'/name)
            held = bool(30 <= i < 150)
            lag = .06 if not held and (50 <= i < 75 or 110 <= i < 135) else 0.
            if held:
                data.ctrl[:] = previous
                mujoco.mj_step(model, data, nstep=round(dt/model.opt.timestep))
            else:
                if lag:
                    data.ctrl[:] = previous
                    mujoco.mj_step(model, data, nstep=round(lag/model.opt.timestep))
                data.ctrl[:] = target
                mujoco.mj_step(model, data, nstep=round((dt-lag)/model.opt.timestep))
                previous = target.copy()
            mujoco.mj_forward(model, data)
            q = data.qpos[indices].astype(np.float32).copy()
            np.savez_compressed(source/'after'/name, q=q, left=render_rgb(renderer, data),
                right=render_rgb(renderer, data, 'entity_right'), time=data.time)
            truth = render_truth(renderer, model, data)
            truth.update(anchors.score_truth(model, data, truth), actual_q=q)
            ids = np.flatnonzero(model.geom_group == 1)
            camera = model.camera('fixed').id
            optical = np.diag([1.,-1.,-1.])@data.cam_xmat[camera].reshape(3,3).T
            truth.update(body_positions=(data.geom_xpos[ids]-data.cam_xpos[camera])@optical.T,
                body_rotations=np.einsum('ij,njk->nik',optical,data.geom_xmat[ids].reshape(-1,3,3)),
                body_sizes=model.geom_size[ids],body_types=model.geom_type[ids])
            np.savez_compressed(source/'evaluation'/name, **truth)
            np.testing.assert_array_equal(fixed, data.xpos[props])
            rows.append(dict(file=name, phase=phase, duration=dt, held=held, actuator_delay_s=lag))
            if (i+1)%60 == 0:
                print(f'E12 held collect {seed}: {i+1}/180', flush=True)
    finally:
        renderer.close()
    write_json(source/'metadata.json',dict(seed=seed,frames=rows,calibration=cal,created_utc=utc(),
        before_action_hashes=hashes,prop_max_displacement_m=0.,fixed_body_points=int(anchors.is_body.sum())))
    return source


def rgb_files(source):
    folder = source/'rgb'; folder.mkdir(exist_ok=True)
    paths = []
    for row in read_json(source/'metadata.json')['frames']:
        path = folder/row['file']
        if not path.exists():
            a = e7.load_npz(source/'after'/row['file'])
            np.savez_compressed(path, left=a['left'], right=a['right'])
        paths.append(str(path))
    return np.array(paths)


def share_prefix(moving, held, stop=30):
    """The observational-twin fixture reuses the same recorded causal prefix."""
    assert moving.keys() == held.keys()
    for key in held:
        held[key][:stop] = moving[key][:stop]


def inputs(output, seed):
    marker=output/f'inputs-{seed}-complete.json'
    if marker.exists():
        return
    collect_held(output,seed)
    e11.inputs(output,seed)
    a=e7.load_npz(output/'sensors'/f'{seed}-execution_hold.npz')
    for condition in ('hold_feedback_bias','hold_no_camera'):
        b={k:v.copy() for k,v in a.items()}
        mask=e10.active_mask()
        if condition=='hold_feedback_bias':
            b['feedback'][mask]+=e10.BIAS
        else:
            b['available'][mask,1]=False;b['quality'][mask]=0.
            b['rgb_paths'][mask]=''
        np.savez_compressed(output/'sensors'/f'{seed}-{condition}.npz',**b)
    note=read_json(marker)
    note.update(sensor_hashes={c:sha(output/'sensors'/f'{seed}-{c}.npz') for c in CONDITIONS},
                physical_hold_frames=[30,150],new_evidence_while_still_held=[[70,110],[140,150]])
    write_json(marker,note)


def measurements(output, seed):
    atlas = SurfaceAtlas.load(e9.ROOT/'atlas-multi.npz')
    ids = np.load(e9.ROOT/'matched-budget-indices.npy')
    observer = AtlasMeasurements(atlas,ids)
    destination = output/'measurements'; destination.mkdir(exist_ok=True)
    cache = {}
    for condition in CONDITIONS:
        path = destination/f'{seed}-{condition}.npz'
        if path.exists():
            continue
        a = e7.load_npz(output/'sensors'/f'{seed}-{condition}.npz')
        rows = []
        for i in range(180):
            f_ok = bool(a['available'][i,0])
            key = (a['rgb_paths'][i],a['action_reference'][i].tobytes(),a['feedback'][i].tobytes() if f_ok else None)
            if key not in cache:
                rgb = e7.load_npz(Path(key[0])) if key[0] else dict(left=None,right=None)
                cache[key] = observer.observe(rgb['left'],rgb['right'],
                    [a['action_reference'][i],a['feedback'][i] if f_ok else None])
            rows.append(cache[key])
        np.savez_compressed(path, **stack(rows))
        write_json(path.with_suffix('.json'),dict(created_utc=utc(),sha256=sha(path),
            sensor_sha256=sha(output/'sensors'/f'{seed}-{condition}.npz'),
            geometry_truth_used=False, patches=len(ids), duplicates_counted_once=True))
        print(f'E12 RGB surface measurements {seed} {condition}', flush=True)


def components(output):
    body = BodyModel(e7.BODY)
    transition = ActionTransition.load(output/'action.pt')
    atlas = SurfaceAtlas.load(e9.ROOT/'atlas-multi.npz')
    fitter = AtlasFitter(atlas,np.load(e9.ROOT/'matched-budget-indices.npy'))
    return body,transition,fitter


def calibrate(output):
    _,_,fitter=components(output)
    af,aq,raw=[],[],[]
    for condition in ('clean','execution_hold'):
        a=e7.load_npz(output/'sensors'/f'{DEV}-{condition}.npz')
        m=e7.load_npz(output/'measurements'/f'{DEV}-{condition}.npz')
        if condition=='clean':
            af.extend(fitter.distance(x,y) for x,y in zip(a['action_reference'],a['feedback']))
            aq.extend(np.max(np.abs((a['action_reference']-a['feedback'])/fitter.scale),axis=-1))
        raw.extend(r for i,f in enumerate(a['feedback']) if np.isfinite(r:=fitter.residual(f,m['xyz'][i],m['valid'][i])))
    if len(raw)<80:
        raise ValueError('Insufficient surface evidence')
    thresholds=dict(association_m=max(.006,float(np.quantile(af,.995))),
        association_q=max(.08,float(np.quantile(aq,.995))),surface_m=max(.006,float(np.quantile(raw,.99))))
    write_json(output/'calibration.json',dict(created_utc=utc(),development_seed=DEV,thresholds=thresholds,
        counts=dict(af=len(af),raw=len(raw)),network_retrained=False))


def rollout(output,seed,condition,profile,all_modes=False):
    body,transition,fitter=components(output)
    a=e7.load_npz(output/'sensors'/f'{seed}-{condition}.npz')
    m=e7.load_npz(output/'measurements'/f'{seed}-{condition}.npz')
    initial=e7.load_npz(source_for(output,seed,condition)/'initial.npz')['q']
    thresholds=read_json(output/'calibration.json')['thresholds']
    confirm,factor=PROFILES[profile]
    gate=JointReliability.load(e8.ROOT/'per_joint.pt')
    state=BranchBodyState(transition,gate,body,fitter,initial,thresholds,confirm,factor)
    no_override=BranchBodyState(transition,gate,body,fitter,initial,thresholds,confirm,factor,False) if all_modes else None
    old=CompetingBodyState(transition,gate,body,fitter,initial,
        read_json(e11.ROOT/'calibration.json')['thresholds'],0.,3) if all_modes else None
    baselines={k:ReliableBodyState(transition,gate,body,initial) for k in ('e8','single_surface')} if all_modes else {}
    previous_f,velocity_f=initial.copy(),np.zeros(7,np.float32)
    rows=[]
    for i in range(180):
        target,dt=a['target'][i],float(a['duration'][i])
        prior=state.forecast(target,dt)
        for baseline in baselines.values():
            baseline.forecast(target,dt)
        if old is not None:
            old.forecast(target,dt);no_override.forecast(target,dt)
        fallback=transition.predict(previous_f,velocity_f,target,dt)
        f_ok,v_ok=a['available'][i]
        feedback=a['feedback'][i] if f_ok else None
        args=(feedback,a['visual'][i],a['quality'][i],v_ok,m['xyz'][i],m['valid'][i])
        result=state.finish(*args)
        row=dict(q_branch=result.pop('q'),**result,prior=prior)
        if all_modes:
            row['q_e11']=old.finish(*args)['q']
            row['q_no_override']=no_override.finish(*args)['q']
            for name,baseline in baselines.items():
                oldq=baseline.q.copy()
                estimate=baseline.finish(*args[:4])['q']
                if name=='single_surface':
                    fitted,ok=fitter.fit(estimate,m['xyz'][i],m['valid'][i])
                    if ok:
                        estimate+=.2*(fitted-estimate)
                    baseline.q=estimate.copy()
                    baseline.velocity=np.clip((estimate-oldq)/dt,-fitter.scale*20,fitter.scale*20)
                row[f'q_{name}']=estimate.copy()
            row['q_action']=a['action_reference'][i]
            qf=feedback if f_ok else fallback
            row['q_feedback']=qf.copy()
            velocity_f=np.clip((qf-previous_f)/dt,-fitter.scale*20,fitter.scale*20)
            previous_f=qf.copy()
        rows.append(row)
    return stack(rows)


def prepare(output):
    output.mkdir(parents=True,exist_ok=True)
    if (output/'protocol-freeze.json').exists():
        raise FileExistsError('E12 already frozen')
    e11.verify(e11.ROOT);e10.verify(e10.ROOT);e8.verify_freeze(e8.ROOT);e9.verify(e9.ROOT)
    if not (output/'action.pt').exists():
        shutil.copyfile(e7.ROOT/'action.pt',output/'action.pt')
    inputs(output,DEV);measurements(output,DEV);calibrate(output)
    destination=output/'development';destination.mkdir(exist_ok=True)
    profiles={};_,_,fitter=components(output)
    teachers={}
    for condition in CONDITIONS:
        source=source_for(output,DEV,condition)
        teachers[condition]=np.stack([e7.load_npz(source/'after'/r['file'])['q'] for r in read_json(source/'metadata.json')['frames']])
    for profile in PROFILES:
        errors=[];margins=[]
        for condition in CONDITIONS:
            prediction=rollout(output,DEV,condition,profile)
            np.savez_compressed(destination/f'{profile}-{condition}.npz',**prediction)
            teacher=teachers[condition]
            error=float(np.mean(((prediction['q_branch']-teacher)/fitter.scale)**2))
            weight=8 if condition in ('clean','execution_hold') else (4 if condition in ('both','coherent_replay') else 1)
            errors.append(error*weight)
            if condition=='clean':
                margins.extend(np.abs((prediction['q_branch']-teacher)/fitter.scale))
            if condition=='twin_held':
                candidates=prediction['bank'][30:70,:4]
                distances=np.array([[fitter.distance(q,t) for q in row] for row,t in zip(candidates,teacher[30:70])])
                nearest=candidates[np.arange(40),distances.argmin(-1)]
                margins.extend(np.abs((nearest-teacher[30:70])/fitter.scale))
            print('E12 DEV',profile,condition,'normalized RMSE',round(error**.5,5),flush=True)
        value=float(np.mean(errors))
        margin=np.maximum(np.quantile(np.asarray(margins),.99,axis=0),.02)
        profiles[profile]=dict(selection=value,margin_normalized=margin.tolist(),
            confirm=PROFILES[profile][0],association_factor=PROFILES[profile][1])
    selected=min(profiles,key=lambda k:profiles[k]['selection'])
    write_json(output/'development-selection.json',dict(created_utc=utc(),selected=selected,profiles=profiles,
        teacher='development recorded joint readings only; never evaluation geometry',network_retrained=False,
        engineering_seed=2026103111,development_seed=DEV))
    dependencies=(e7.BODY,e7.BANK,e7.SEGMENTATION,e8.ROOT/'per_joint.pt',e9.ROOT/'atlas-multi.npz',
                  e9.ROOT/'matched-budget-indices.npy',output/'action.pt',e11.ROOT/'calibration.json',e11.ROOT/'protocol-freeze.json')
    names=(*SOURCES,'conflict_cycle.py','conflict_state.py','trust_cycle.py','reliable_state.py',
           'action_state.py','surface_extent.py','body_memory.py','body_fixture.py','recovery.py')
    write_json(output/'protocol-freeze.json',dict(created_utc=utc(),selected=selected,development_seed=DEV,
        final_seeds=FINAL,conditions=CONDITIONS,models_retrained=False,
        sources={n:sha(Path(__file__).with_name(n)) for n in names},dependencies={str(p):sha(p) for p in dependencies},
        protocol_sha256=sha('docs/body-branches-e12.md'),calibration_sha256=sha(output/'calibration.json'),
        selection_sha256=sha(output/'development-selection.json'),frames=180,
        physical_hold_frames=[30,150],single_surface_gain=.2,revision='persistent-hold-disentangled-return'))
    print('E12 FROZEN',selected,flush=True)


def verify(output):
    frozen=read_json(output/'protocol-freeze.json')
    for n,digest in frozen['sources'].items():
        assert sha(Path(__file__).with_name(n))==digest,n
    for p,digest in frozen['dependencies'].items():
        assert sha(p)==digest,p
    for name,key in ((Path('docs/body-branches-e12.md'),'protocol_sha256'),
                     (output/'calibration.json','calibration_sha256'),
                     (output/'development-selection.json','selection_sha256')):
        assert sha(name)==frozen[key],name
    return frozen


def final(output):
    frozen=verify(output)
    if (output/'predictions-complete.json').exists():
        raise FileExistsError('Final predictions already sealed')
    destination=output/'predictions';destination.mkdir(exist_ok=True)
    for seed in FINAL:
        inputs(output,seed);measurements(output,seed)
        for condition in CONDITIONS:
            path=destination/f'{seed}-{condition}.npz'
            if path.exists():
                continue
            prediction=rollout(output,seed,condition,frozen['selected'],all_modes=True)
            np.savez_compressed(path,**prediction)
            write_json(path.with_suffix('.json'),dict(created_utc=utc(),sha256=sha(path),
                sensor_sha256=sha(output/'sensors'/f'{seed}-{condition}.npz'),
                measurement_sha256=sha(output/'measurements'/f'{seed}-{condition}.npz'),
                geometry_used=False,teacher_used=False,fault_label_as_input=False))
            print('E12 predicted',seed,condition,flush=True)
    paths=sorted(destination.glob('*.npz'))
    write_json(output/'predictions-complete.json',dict(created_utc=utc(),
        hashes={str(p):sha(p) for p in paths},all_predictions_before_privileged_scoring=True))
    print('E12 ALL PREDICTIONS SEALED',flush=True)


def score(output):
    frozen=verify(output)
    if (output/'scores.json').exists():
        raise FileExistsError('Preserve completed scores')
    seal=read_json(output/'predictions-complete.json')
    assert len(seal['hashes'])==len(FINAL)*len(CONDITIONS)
    for p,digest in seal['hashes'].items():
        assert sha(p)==digest
    body,_,fitter=components(output)
    selection=read_json(output/'development-selection.json')
    margin=np.array(selection['profiles'][frozen['selected']]['margin_normalized'])*fitter.scale
    frame_rows,events,summary,twins=[],[],[],[]
    mask=e10.active_mask()
    for seed in FINAL:
        truth_cache={}
        for kind in ('clean','execution_hold'):
            source=source_for(output,seed,kind)
            truth_cache[kind]=[e7.load_npz(source/'evaluation'/r['file']) for r in read_json(source/'metadata.json')['frames']]
        for condition in CONDITIONS:
            held=condition in HELD
            truth=truth_cache['execution_hold' if held else 'clean']
            true_q=np.stack([t['actual_q'] for t in truth])
            anchor=np.stack([t['anchor_xyz'] for t in truth]);anchors=truth[0]['anchor_body'].astype(bool)
            p=e7.load_npz(output/'predictions'/f'{seed}-{condition}.npz')
            clean=e7.load_npz(output/'predictions'/f'{seed}-{"execution_hold" if held else "clean"}.npz')
            if condition in ('twin_held','hold_feedback_bias','hold_no_camera'):
                for mode in MODES:
                    xyz=body(p[f'q_{mode}'])
                    err=np.sqrt(np.mean(np.sum((xyz[:,anchors]-anchor[:,anchors])**2,-1),-1))
                    floor=np.sqrt(np.mean(np.sum((body(true_q)[:,anchors]-anchor[:,anchors])**2,-1),-1))
                    distance=np.array([fitter.distance(q,t) for q,t in zip(p[f'q_{mode}'],true_q)])
                    good=(distance<=.005)&(err<=floor+.002)
                    for start,stop in ((70,110),(140,150)):
                        ends=[i for i in range(start+4,stop) if good[i-4:i+1].all()]
                        events.append(dict(seed=seed,condition=condition,mode=mode,start=start,stop=stop,
                            held_absolute=dict(recovered=bool(ends),time_s=(ends[0]-start+1)*.1 if ends else None)))
            active=p['bank_valid']
            lower=np.where(active[:,:,None],p['bank'],np.inf).min(1)-margin
            upper=np.where(active[:,:,None],p['bank'],-np.inf).max(1)+margin
            covered=np.all((true_q>=lower)&(true_q<=upper),-1)
            widths=np.mean((upper-lower)/(2*fitter.scale),-1)
            for mode in MODES:
                xyz=body(p[f'q_{mode}'])
                error=np.sqrt(np.mean(np.sum((xyz[:,anchors]-anchor[:,anchors])**2,-1),-1))
                clear_xyz=body(clean[f'q_{mode}'])
                clear_error=np.sqrt(np.mean(np.sum((clear_xyz[:,anchors]-anchor[:,anchors])**2,-1),-1))
                separation=np.sqrt(np.mean(np.sum((xyz-clear_xyz)**2,-1),-1))
                off=[float(np.mean(e9.primitive_surface_distance(fitter.atlas.predict(q),t)[0]>.015)) for q,t in zip(p[f'q_{mode}'],truth)]
                summary.append(dict(seed=seed,condition=condition,mode=mode,anchors=int(anchors.sum()),
                    rmse_mm=float(np.sqrt(np.mean(error**2))*1000),fault_rmse_mm=float(np.sqrt(np.mean(error[mask]**2))*1000),
                    hold_rmse_mm=float(np.sqrt(np.mean(error[30:150]**2))*1000) if held else None,
                    off_surface=float(np.mean(off)),unknown=float(p['unknown'].mean()) if mode=='branch' else None,
                    coverage=float(covered.mean()) if mode=='branch' else None,
                    width_normalized=float(widths.mean()) if mode=='branch' else None,
                    commits=int(p['committed'].sum()) if mode=='branch' else None,
                    surface_corrections=int(p['surface_corrected'].sum()) if mode=='branch' else None))
                for i in range(180):
                    row=dict(seed=seed,condition=condition,mode=mode,frame=i,error_mm=float(error[i]*1000),
                        separation_mm=float(separation[i]*1000),off_surface=off[i],fault_window=bool(mask[i]),
                        held_physics=held)
                    if mode=='branch':
                        row.update(unknown=bool(p['unknown'][i]),covered=bool(covered[i]),width_normalized=float(widths[i]),
                            committed=bool(p['committed'][i]),surface_corrected=bool(p['surface_corrected'][i]),
                            support=int(p['support'][i]),override_action=bool(p['override_action'][i]),sensor_credible=bool(p['sensor_credible'][i]),hypotheses=int(p['bank_valid'][i].sum()),
                            raw_residual_mm=float(p['raw_residual'][i]*1000))
                    frame_rows.append(row)
                if condition not in ('clean','execution_hold'):
                    for start,stop in e10.RETURNS:
                        event=recovery_event(separation,error-clear_error,start,stop)
                        event.update(seed=seed,condition=condition,mode=mode)
                        events.append(event)
            print('E12 scored',seed,condition,flush=True)
        a=e7.load_npz(output/'predictions'/f'{seed}-twin_moving.npz')
        b=e7.load_npz(output/'predictions'/f'{seed}-twin_held.npz')
        np.testing.assert_array_equal(a['q_branch'][:70],b['q_branch'][:70])
        np.testing.assert_array_equal(a['bank'][:70],b['bank'][:70])
        qa=np.stack([t['actual_q'] for t in truth_cache['clean']])
        qb=np.stack([t['actual_q'] for t in truth_cache['execution_hold']])
        gap=np.array([fitter.distance(x,y) for x,y in zip(qa,qb)])
        valid=np.arange(180)<70
        valid&=np.arange(180)>=30
        valid&=gap>.015
        low=np.where(a['bank_valid'][:,:,None],a['bank'],np.inf).min(1)-margin
        high=np.where(a['bank_valid'][:,:,None],a['bank'],-np.inf).max(1)+margin
        both=np.all((qa>=low)&(qa<=high)&(qb>=low)&(qb<=high),-1)
        twins.append(dict(seed=seed,distinguishable_physical_frames=int(valid.sum()),
            unknown_frames=int((valid&a['unknown']).sum()),both_states_covered=int((valid&both).sum()),
            input_prefix_identical=True,output_prefix_identical=True,
            unknown_rate=float(a['unknown'][valid].mean()) if valid.any() else None,
            simultaneous_coverage=float(both[valid].mean()) if valid.any() else None))
    write_json(output/'frame-scores.json',frame_rows)
    write_json(output/'recovery-events.json',events)
    write_json(output/'scores.json',dict(created_utc=utc(),rows=summary,twins=twins,
        predictions_seal_sha256=sha(output/'predictions-complete.json'),truth_read_after_seal=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','final','score','verify'))
    parser.add_argument('--output',type=Path,default=ROOT)
    args=parser.parse_args();torch.set_num_threads(4)
    globals()[args.stage](args.output)


if __name__=='__main__':
    main()
