"""E11: dependency-aware competing body explanations and actual state updates."""
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

ROOT = Path('outputs/local-check-20261024/e11-body-hypotheses')
DEV = 2026103111
FINAL = (2026103161, 2026103162, 2026103163)
CONDITIONS = (*e10.CONDITIONS, 'execution_hold', 'twin_moving', 'twin_held')
MODES = ('competing', 'e8', 'single_surface', 'naive_vote', 'action', 'feedback')
PROFILES = {'g0_c3':(0.,3), 'g02_c3':(.2,3), 'g0_c5':(0.,5), 'g02_c5':(.2,5)}
SOURCES = ('hypothesis_cycle.py', 'hypothesis_state.py')


def source_for(output, seed, condition):
    held = condition in ('execution_hold', 'twin_held')
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
            held = bool(e10.active_mask()[i])
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
                print(f'E11 held collect {seed}: {i+1}/180', flush=True)
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
    marker = output/f'inputs-{seed}-complete.json'
    if marker.exists():
        return
    e10.collect(output, seed)
    collect_held(output, seed)
    e10.sensors(output, seed)
    normal = source_for(output, seed, 'clean')
    held = source_for(output, seed, 'execution_hold')
    rn, rh = rgb_files(normal), rgb_files(held)
    frames = read_json(held/'metadata.json')['frames']
    hc = e7.load_npz(e7.visual_cache(output, held))
    feedback = np.stack([e7.load_npz(held/'after'/r['file'])['q'] for r in frames])
    execution = dict(target=hc['target'], duration=hc['duration'], feedback=feedback,
        visual=np.where(hc['valid_clean'][:,None],hc['visual_clean'],hc['action_q']),
        quality=np.where(hc['valid_clean'][:,None],hc['quality_clean'],0.),
        available=np.stack((np.ones(180,bool),hc['valid_clean']),-1), action_reference=hc['action_q'])
    data = {c:{k:v for k,v in e7.load_npz(output/'sensors'/f'{seed}-{c}.npz').items()
               if k!='rgb_paths'} for c in e10.CONDITIONS}
    data['execution_hold'] = execution
    data['twin_moving'] = {k:v.copy() for k,v in data['coherent_replay'].items()}
    data['twin_held'] = {k:v.copy() for k,v in execution.items()}
    mask = e10.active_mask()
    for k in data['twin_held']:
        data['twin_held'][k][mask] = data['coherent_replay'][k][mask]
    # Separate GPU contexts may differ by one grayscale level. Reuse the exact
    # recorded prefix rather than claiming approximate pixels are identical.
    share_prefix(data['twin_moving'],data['twin_held'])
    for condition, a in data.items():
        paths = rh.copy() if condition in ('execution_hold','twin_held') else rn.copy()
        if condition in ('stale_rgb','both','coherent_replay','twin_moving','twin_held'):
            ids = np.flatnonzero(mask); paths[ids] = rn[ids%20]
        if condition == 'no_camera':
            paths[mask] = ''
        if condition == 'twin_held':
            paths[:30] = rn[:30]
        a['rgb_paths'] = paths
        np.savez_compressed(output/'sensors'/f'{seed}-{condition}.npz', **a)
    for key in data['twin_moving']:
        np.testing.assert_array_equal(data['twin_moving'][key][:70],data['twin_held'][key][:70])
    np.testing.assert_array_equal(e7.load_npz(normal/'initial.npz')['q'],e7.load_npz(held/'initial.npz')['q'])
    write_json(marker,dict(created_utc=utc(), sensor_hashes={c:sha(output/'sensors'/f'{seed}-{c}.npz') for c in CONDITIONS},
        emulator_uses_actual_readings=True, teacher_as_inference_input=False,
        twins='same causal inputs before first return, different hidden physical trajectories',
        exact_input_prefix_verified_frames=70,shared_recorded_prefix_frames=30))


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
        print(f'E11 RGB surface measurements {seed} {condition}', flush=True)


def components(output):
    body = BodyModel(e7.BODY)
    transition = ActionTransition.load(output/'action.pt')
    atlas = SurfaceAtlas.load(e9.ROOT/'atlas-multi.npz')
    fitter = AtlasFitter(atlas,np.load(e9.ROOT/'matched-budget-indices.npy'))
    return body,transition,fitter


def calibrate(output):
    body,transition,fitter = components(output)
    a = e7.load_npz(output/'sensors'/f'{DEV}-clean.npz')
    m = e7.load_npz(output/'measurements'/f'{DEV}-clean.npz')
    af,raw,dyn = [],[],[]
    old,velocity = None,np.zeros(7,np.float32)
    for i in range(180):
        f = a['feedback'][i]
        af.append(fitter.distance(a['action_reference'][i],f))
        r = fitter.residual(f,m['xyz'][i],m['valid'][i])
        if np.isfinite(r):
            raw.append(r)
        if old is not None:
            dyn.append(fitter.distance(transition.predict(old,velocity,a['target'][i],float(a['duration'][i])),f))
            velocity = np.clip((f-old)/float(a['duration'][i]),-fitter.scale*20,fitter.scale*20)
        old = f.copy()
    if len(raw)<40:
        raise ValueError('Insufficient real stereo surface evidence; preserve failure')
    thresholds = dict(action_feedback=max(.004,float(np.quantile(af,.99))),
        surface=max(.004,float(np.quantile(raw,.99))),dynamics=max(.004,float(np.quantile(dyn,.99))))
    write_json(output/'calibration.json',dict(created_utc=utc(),development_seed=DEV,
        thresholds=thresholds,counts=dict(af=len(af),raw=len(raw),dynamics=len(dyn)),quantile=.99))


def rollout(output, seed, condition, profile, all_modes=False):
    body,transition,fitter = components(output)
    a = e7.load_npz(output/'sensors'/f'{seed}-{condition}.npz')
    m = e7.load_npz(output/'measurements'/f'{seed}-{condition}.npz')
    initial = e7.load_npz(source_for(output,seed,condition)/'initial.npz')['q']
    thresholds = read_json(output/'calibration.json')['thresholds']
    gain,confirm = PROFILES[profile]
    gate = JointReliability.load(e8.ROOT/'per_joint.pt')
    state = CompetingBodyState(transition,gate,body,fitter,initial,thresholds,gain,confirm)
    baselines = {k:ReliableBodyState(transition,gate,body,initial) for k in ('e8','single_surface')} if all_modes else {}
    previous_f,velocity_f = initial.copy(),np.zeros(7,np.float32)
    rows = []
    for i in range(180):
        target,dt = a['target'][i],float(a['duration'][i])
        protected = state.forecast(target,dt)
        for baseline in baselines.values():
            baseline.forecast(target,dt)
        fallback = transition.predict(previous_f,velocity_f,target,dt)
        f_ok,v_ok = a['available'][i]
        feedback = a['feedback'][i] if f_ok else None
        result = state.finish(feedback,a['visual'][i],a['quality'][i],v_ok,m['xyz'][i],m['valid'][i])
        row = dict(q_competing=result.pop('q'), **result, prior=protected)
        if all_modes:
            for name,baseline in baselines.items():
                oldq = baseline.q.copy()
                estimate = baseline.finish(feedback,a['visual'][i],a['quality'][i],v_ok)['q']
                if name == 'single_surface':
                    fitted,ok = fitter.fit(estimate,m['xyz'][i],m['valid'][i])
                    if ok:
                        estimate += .2*(fitted-estimate)
                    baseline.q = estimate.copy()
                    baseline.velocity = np.clip((estimate-oldq)/dt,-fitter.scale*20,fitter.scale*20)
                row[f'q_{name}'] = estimate.copy()
            opts = [a['action_reference'][i]]
            if f_ok:
                opts.append(a['feedback'][i])
            if v_ok:
                opts.append(a['visual'][i])
            if len(opts)>2:
                pairs = [(fitter.distance(opts[j],opts[k]),j,k) for j in range(len(opts)) for k in range(j+1,len(opts))]
                _,j,k = min(pairs)
                voted = (opts[j]+opts[k])*.5
            else:
                voted = np.mean(opts,axis=0)
            row['q_naive_vote'] = voted
            row['q_action'] = a['action_reference'][i]
            qf = a['feedback'][i] if f_ok else fallback
            row['q_feedback'] = qf.copy()
            velocity_f = np.clip((qf-previous_f)/dt,-fitter.scale*20,fitter.scale*20)
            previous_f = qf.copy()
        rows.append(row)
    return stack(rows)


def prepare(output):
    output.mkdir(parents=True,exist_ok=True)
    if (output/'protocol-freeze.json').exists():
        raise FileExistsError('E11 already frozen')
    e10.verify(e10.ROOT); e8.verify_freeze(e8.ROOT); e9.verify(e9.ROOT)
    if not (output/'action.pt').exists():
        shutil.copyfile(e7.ROOT/'action.pt',output/'action.pt')
    inputs(output,DEV); measurements(output,DEV); calibrate(output)
    destination=output/'development';destination.mkdir(exist_ok=True)
    profiles = {}
    body = BodyModel(e7.BODY)
    scale = np.asarray(body.saved['scale'])
    for profile in PROFILES:
        errors,normal,qnormal = [],None,None
        for condition in CONDITIONS:
            prediction = rollout(output,DEV,condition,profile)
            np.savez_compressed(destination/f'{profile}-{condition}.npz',**prediction)
            source = source_for(output,DEV,condition)
            teacher = np.stack([e7.load_npz(source/'after'/r['file'])['q'] for r in read_json(source/'metadata.json')['frames']])
            error = float(np.mean(((prediction['q_competing']-teacher)/scale)**2))
            errors.append(error*(8 if condition=='clean' else 1))
            if condition=='clean':
                normal,qnormal = teacher,prediction['q_competing']
        value = float(np.mean(errors))
        margin = np.maximum(np.quantile(np.abs((qnormal-normal)/scale),.99,axis=0),.02)
        profiles[profile] = dict(selection=value,margin_normalized=margin.tolist(),
            gain=PROFILES[profile][0],confirm=PROFILES[profile][1])
        print('E11 development profile',profile,value,flush=True)
    selected = min(profiles,key=lambda k:profiles[k]['selection'])
    write_json(output/'development-selection.json',dict(created_utc=utc(),selected=selected,profiles=profiles,
        teacher='normal recorded actual feedback; not evaluation geometry',network_retrained=False))
    dependencies=(e7.BODY,e7.BANK,e7.SEGMENTATION,e8.ROOT/'per_joint.pt',e9.ROOT/'atlas-multi.npz',
                  e9.ROOT/'matched-budget-indices.npy',output/'action.pt')
    names=(*SOURCES,'conflict_cycle.py','conflict_state.py','trust_cycle.py','reliable_state.py',
           'action_state.py','surface_extent.py','body_memory.py','body_fixture.py','recovery.py')
    write_json(output/'protocol-freeze.json',dict(created_utc=utc(),selected=selected,development_seed=DEV,
        final_seeds=FINAL,conditions=CONDITIONS,models_retrained=False,
        sources={n:sha(Path(__file__).with_name(n)) for n in names},dependencies={str(p):sha(p) for p in dependencies},
        protocol_sha256=sha('docs/body-hypotheses-e11.md'),calibration_sha256=sha(output/'calibration.json'),
        selection_sha256=sha(output/'development-selection.json'),frames=180,
        revision_sha256=sha('docs/body-hypotheses-e11-revision.json'),
        single_surface_gain=.2,revision='shared-prefix input fixture; old partial final data are unscored preflight'))
    print('E11 FROZEN',selected,flush=True)


def verify(output):
    frozen=read_json(output/'protocol-freeze.json')
    for n,digest in frozen['sources'].items():
        assert sha(Path(__file__).with_name(n))==digest,n
    for p,digest in frozen['dependencies'].items():
        assert sha(p)==digest,p
    assert sha('docs/body-hypotheses-e11.md')==frozen['protocol_sha256']
    assert sha(output/'calibration.json')==frozen['calibration_sha256']
    assert sha(output/'development-selection.json')==frozen['selection_sha256']
    assert sha('docs/body-hypotheses-e11-revision.json')==frozen['revision_sha256']
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
            print('E11 predicted',seed,condition,flush=True)
    paths=sorted(destination.glob('*.npz'))
    write_json(output/'predictions-complete.json',dict(created_utc=utc(),
        hashes={str(p):sha(p) for p in paths},all_predictions_before_privileged_scoring=True))
    print('E11 ALL PREDICTIONS SEALED',flush=True)


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
            held=condition in ('execution_hold','twin_held')
            truth=truth_cache['execution_hold' if held else 'clean']
            true_q=np.stack([t['actual_q'] for t in truth])
            anchor=np.stack([t['anchor_xyz'] for t in truth]);anchors=truth[0]['anchor_body'].astype(bool)
            p=e7.load_npz(output/'predictions'/f'{seed}-{condition}.npz')
            clean=e7.load_npz(output/'predictions'/f'{seed}-{"execution_hold" if held else "clean"}.npz')
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
                    off_surface=float(np.mean(off)),unknown=float(p['unknown'].mean()) if mode=='competing' else None,
                    coverage=float(covered.mean()) if mode=='competing' else None,
                    width_normalized=float(widths.mean()) if mode=='competing' else None,
                    commits=int(p['committed'].sum()) if mode=='competing' else None,
                    surface_corrections=int(p['surface_corrected'].sum()) if mode=='competing' else None))
                for i in range(180):
                    row=dict(seed=seed,condition=condition,mode=mode,frame=i,error_mm=float(error[i]*1000),
                        separation_mm=float(separation[i]*1000),off_surface=off[i],fault_window=bool(mask[i]),
                        held_physics=held)
                    if mode=='competing':
                        row.update(unknown=bool(p['unknown'][i]),covered=bool(covered[i]),width_normalized=float(widths[i]),
                            committed=bool(p['committed'][i]),surface_corrected=bool(p['surface_corrected'][i]),
                            support=int(p['support'][i]),hypotheses=int(p['bank_valid'][i].sum()),
                            raw_residual_mm=float(p['raw_residual'][i]*1000))
                    frame_rows.append(row)
                if condition not in ('clean','execution_hold'):
                    for start,stop in e10.RETURNS:
                        event=recovery_event(separation,error-clear_error,start,stop)
                        event.update(seed=seed,condition=condition,mode=mode)
                        events.append(event)
            print('E11 scored',seed,condition,flush=True)
        a=e7.load_npz(output/'predictions'/f'{seed}-twin_moving.npz')
        b=e7.load_npz(output/'predictions'/f'{seed}-twin_held.npz')
        np.testing.assert_array_equal(a['q_competing'][:70],b['q_competing'][:70])
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
