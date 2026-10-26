"""E13: frozen visual branch selection and separately calibrated error tubes."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import numpy as np
import torch
from . import branch_cycle as e12, trust_cycle as e7, reliable_cycle_v2 as e8
from . import extent_cycle_v2 as e9
from .body_cycle import read_json,write_json,sha,utc,stack
from .branch_state import BranchBodyState
from .evidence_state import EvidenceBodyState,MotionMeasurements,branch_envelope
from .hypothesis_state import AtlasMeasurements
from .reliable_state import JointReliability,ReliableBodyState

ROOT=Path('outputs/local-check-20261024/e13-visual-evidence')
DEV=2026103311
CAL=2026103312
FINAL=(2026103331,2026103332,2026103333)
CONDITIONS=(*e12.CONDITIONS,'hold_no_feedback')
HELD=(*e12.HELD,'hold_no_feedback')
MODES=('evidence','e12','e12_original','no_visual','e8','single_surface','action','feedback')
PROFILES={'v3_g0':(3,0.),'v5_g0':(5,0.),'v3_g02':(3,.2),'v5_g02':(5,.2)}
PROTOCOL=Path('docs/body-evidence-e13.md')


def source_for(output,seed,condition):
    return output/f'data-{seed}{"-held" if condition in HELD else ""}'


def components(output):
    return e12.components(output)


def inputs(output,seed):
    e12.inputs(output,seed)
    path=output/'sensors'/f'{seed}-hold_no_feedback.npz'
    if not path.exists():
        a=e7.load_npz(output/'sensors'/f'{seed}-execution_hold.npz')
        mask=e12.e10.active_mask()
        a['available'][mask,0]=False;a['feedback'][mask]=0.
        np.savez_compressed(path,**a)
    marker=output/f'inputs-{seed}-complete.json'
    note=read_json(marker)
    note['sensor_hashes']={c:sha(output/'sensors'/f'{seed}-{c}.npz') for c in CONDITIONS}
    write_json(marker,note)


def measurements(output,seed):
    body,transition,fitter=components(output)
    observer=MotionMeasurements(fitter.atlas,fitter.indices)
    gate=JointReliability.load(e8.ROOT/'per_joint.pt')
    thresholds=read_json(e12.ROOT/'calibration.json')['thresholds']
    destination=output/'measurements';destination.mkdir(exist_ok=True)
    cache={};checks=0
    for condition in CONDITIONS:
        path=destination/f'{seed}-{condition}.npz'
        if path.exists():continue
        a=e7.load_npz(output/'sensors'/f'{seed}-{condition}.npz')
        initial=e7.load_npz(source_for(output,seed,condition)/'initial.npz')['q']
        neutral=BranchBodyState(transition,gate,body,fitter,initial,thresholds,2,1.5)
        rows=[]
        for i in range(180):
            neutral.forecast(a['target'][i],float(a['duration'][i]))
            f=a['feedback'][i] if a['available'][i,0] else None
            bank=np.vstack((neutral.pending['action'],neutral.pending['local'],neutral.pending['holds']))
            seeds=[a['action_reference'][i],f,*neutral.pending['holds']]
            key=(a['rgb_paths'][i],*(q.tobytes() if q is not None else None for q in seeds))
            if key not in cache:
                rgb=e7.load_npz(Path(key[0])) if key[0] else dict(left=None,right=None)
                original,m=observer.observe(rgb['left'],rgb['right'],seeds)
                if condition=='clean' and i in (0,30,60,110,150):
                    expected=AtlasMeasurements(fitter.atlas,fitter.indices).observe(rgb['left'],rgb['right'],seeds[:2])
                    for k in expected:np.testing.assert_array_equal(original[k],expected[k])
                    checks+=1
                cache[key]=(original,m)
            original,m=cache[key]
            result=neutral.finish(f,a['visual'][i],a['quality'][i],a['available'][i,1],m['xyz'],m['valid'])
            rows.append(dict(**m,**{f'original_{k}':v for k,v in original.items()},
                neutral_bank=bank,neutral_q=result['q']))
        np.savez_compressed(path,**stack(rows))
        write_json(path.with_suffix('.json'),dict(created_utc=utc(),sha256=sha(path),
            sensor_sha256=sha(output/'sensors'/f'{seed}-{condition}.npz'),geometry_used=False,
            patches=len(fitter.indices),seed_budget=4,original_seed_budget=2,duplicates_counted_once=True,
            original_observer_exact_comparisons=checks if condition=='clean' else 0))
        print('E13 RGB',seed,condition,flush=True)


def joint_teacher(output,seed,condition):
    source=source_for(output,seed,condition)
    return np.stack([e7.load_npz(source/'after'/r['file'])['q'] for r in read_json(source/'metadata.json')['frames']])


def calibrate(output):
    _,_,fitter=components(output)
    clean=e7.load_npz(output/'measurements'/f'{CAL}-clean.npz')
    held=e7.load_npz(output/'measurements'/f'{CAL}-twin_held.npz')
    normalq=joint_teacher(output,CAL,'clean');heldq=joint_teacher(output,CAL,'execution_hold')
    groups=[(clean['neutral_bank'][:,0]-normalq)/fitter.scale,
        (held['neutral_bank'][30:70,2]-heldq[30:70])/fitter.scale,
        (held['neutral_bank'][30:70,3]-heldq[30:70])/fitter.scale,
        (clean['neutral_q']-normalq)/fitter.scale]
    results=[]
    for errors in groups:
        base=np.maximum(np.quantile(np.abs(errors),.95,axis=0),.02)
        scores=np.max(np.abs(errors)/base,axis=-1)
        factor=max(1.,float(np.quantile(scores,.995)))*1.25
        results.append(dict(frames=len(errors),base=base.tolist(),factor=factor,radius=(base*factor).tolist()))
    motion,servo,coast,main=[x['radius'] for x in results]
    radii=[motion,motion,servo,coast,[.02]*7,main]
    write_json(output/'calibration.json',dict(created_utc=utc(),calibration_seed=CAL,
        thresholds=read_json(e12.ROOT/'calibration.json')['thresholds'],radius_normalized=radii,
        groups=results,quantile_per_joint=.95,quantile_max_joint=.995,safety_factor=1.25,
        model_retrained=False,teacher='separate calibration joint readings; no evaluation geometry',
        measurement_hashes={c:sha(output/'measurements'/f'{CAL}-{c}.npz') for c in ('clean','twin_held')}))


def rollout(output,seed,condition,profile,all_modes=False):
    body,transition,fitter=components(output)
    a=e7.load_npz(output/'sensors'/f'{seed}-{condition}.npz')
    m=e7.load_npz(output/'measurements'/f'{seed}-{condition}.npz')
    initial=e7.load_npz(source_for(output,seed,condition)/'initial.npz')['q']
    thresholds=read_json(output/'calibration.json')['thresholds']
    confirm,gain=PROFILES[profile];gate=JointReliability.load(e8.ROOT/'per_joint.pt')
    state=EvidenceBodyState(transition,gate,body,fitter,initial,thresholds,2,1.5,
        visual_confirm=confirm,visual_gain=gain)
    others={}
    if all_modes:
        others['no_visual']=EvidenceBodyState(transition,gate,body,fitter,initial,thresholds,2,1.5,
            visual_confirm=confirm,visual_gain=gain,visual_enabled=False)
        for k in ('e12','e12_original'):
            others[k]=BranchBodyState(transition,gate,body,fitter,initial,thresholds,2,1.5)
    baselines={k:ReliableBodyState(transition,gate,body,initial) for k in ('e8','single_surface')} if all_modes else {}
    previous_f=initial.copy();velocity_f=np.zeros(7,np.float32);rows=[]
    for i in range(180):
        target,dt=a['target'][i],float(a['duration'][i])
        prior=state.forecast(target,dt)
        for baseline in (*others.values(),*baselines.values()):baseline.forecast(target,dt)
        f_ok,v_ok=a['available'][i];feedback=a['feedback'][i] if f_ok else None
        args=(feedback,a['visual'][i],a['quality'][i],v_ok,m['xyz'][i],m['valid'][i])
        result=state.finish(*args);row=dict(q_evidence=result.pop('q'),**result,prior=prior)
        if all_modes:
            for name,other in others.items():
                evidence=args if name!='e12_original' else (*args[:4],m['original_xyz'][i],m['original_valid'][i])
                r=other.finish(*evidence);row[f'q_{name}']=r['q']
                if name=='e12':
                    row['e12_bank']=r['bank'];row['e12_bank_valid']=r['bank_valid'];row['e12_unknown']=r['unknown']
            for name,baseline in baselines.items():
                oldq=baseline.q.copy();estimate=baseline.finish(*args[:4])['q']
                if name=='single_surface':
                    fitted,ok=fitter.fit(estimate,m['xyz'][i],m['valid'][i])
                    if ok:estimate+=.2*(fitted-estimate)
                    baseline.q=estimate.copy();baseline.velocity=np.clip((estimate-oldq)/dt,-fitter.scale*20,fitter.scale*20)
                row[f'q_{name}']=estimate.copy()
            row['q_action']=a['action_reference'][i]
            qf=feedback if f_ok else transition.predict(previous_f,velocity_f,target,dt)
            row['q_feedback']=qf.copy();velocity_f=np.clip((qf-previous_f)/dt,-fitter.scale*20,fitter.scale*20);previous_f=qf.copy()
        rows.append(row)
    return stack(rows)


def develop(output):
    output.mkdir(parents=True,exist_ok=True)
    if (output/'protocol-freeze.json').exists():raise FileExistsError('Frozen E13')
    e12.verify(e12.ROOT)
    if not (output/'action.pt').exists():shutil.copyfile(e7.ROOT/'action.pt',output/'action.pt')
    for seed in (DEV,CAL):inputs(output,seed);measurements(output,seed)
    calibrate(output)
    destination=output/'development';destination.mkdir(exist_ok=True)
    profiles={};_,_,fitter=components(output)
    teachers={c:joint_teacher(output,DEV,c) for c in CONDITIONS}
    for profile in PROFILES:
        errors=[];details={}
        for condition in CONDITIONS:
            prediction=rollout(output,DEV,condition,profile)
            np.savez_compressed(destination/f'{profile}-{condition}.npz',**prediction)
            error=float(np.mean(((prediction['q_evidence']-teachers[condition])/fitter.scale)**2))
            weight=8 if condition in ('clean','execution_hold','hold_feedback_bias','hold_no_feedback') else (4 if condition in ('both','coherent_replay') else 1)
            errors.append(error*weight);details[condition]=dict(mse=error,visual_updates=int(prediction['visual_selected'].sum()))
            print('E13 DEV',profile,condition,'normalized RMSE',round(error**.5,6),flush=True)
        profiles[profile]=dict(selection=float(np.mean(errors)),conditions=details,
            visual_confirm=PROFILES[profile][0],visual_gain=PROFILES[profile][1])
    selected=min(profiles,key=lambda k:profiles[k]['selection'])
    write_json(output/'development-selection.json',dict(created_utc=utc(),selected=selected,profiles=profiles,
        engineering_seed=2026103211,development_seed=DEV,calibration_seed=CAL,
        teacher='development joint readings; not evaluation geometry',models_retrained=False))
    review(output)
    print('E13 DEVELOPED; NOT YET FROZEN',selected,flush=True)


def review(output):
    selected=read_json(output/'development-selection.json')['selected']
    body,transition,fitter=components(output)
    thresholds=read_json(output/'calibration.json')['thresholds'];results={};mask=e12.e10.active_mask()
    for c in CONDITIONS:
        a=e7.load_npz(output/'sensors'/f'{DEV}-{c}.npz')
        m=e7.load_npz(output/'measurements'/f'{DEV}-{c}.npz')
        p=e7.load_npz(output/'development'/f'{selected}-{c}.npz')
        initial=e7.load_npz(source_for(output,DEV,c)/'initial.npz')['q']
        state=BranchBodyState(transition,JointReliability.load(e8.ROOT/'per_joint.pt'),body,fitter,initial,thresholds,2,1.5)
        baseline=[]
        for i in range(180):
            state.forecast(a['target'][i],float(a['duration'][i]))
            baseline.append(state.finish(a['feedback'][i] if a['available'][i,0] else None,a['visual'][i],
                a['quality'][i],a['available'][i,1],m['xyz'][i],m['valid'][i])['q'])
        teacher=joint_teacher(output,DEV,c)
        new=np.array([fitter.distance(q,t) for q,t in zip(p['q_evidence'],teacher)])
        old=np.array([fitter.distance(q,t) for q,t in zip(baseline,teacher)])
        results[c]=dict(learned_state_fault_distance_e13_mm=float(np.mean(new[mask]**2)**.5*1000),
            learned_state_fault_distance_e12_mm=float(np.mean(old[mask]**2)**.5*1000),
            normalized_q_rmse_e13=float(np.mean(((p['q_evidence']-teacher)/fitter.scale)**2)**.5),
            visual_updates=int(p['visual_selected'].sum()))
    p=e7.load_npz(output/'development'/f'{selected}-twin_moving.npz')
    qa=joint_teacher(output,DEV,'clean');qb=joint_teacher(output,DEV,'execution_hold')
    lo,hi=branch_envelope(p['bank'],p['bank_valid'],fitter.scale,read_json(output/'calibration.json')['radius_normalized'])
    ids=(np.arange(180)>=30)&(np.arange(180)<70)
    ids&=np.array([fitter.distance(a,b)>.015 for a,b in zip(qa,qb)])
    both=np.all((qa>=lo)&(qa<=hi)&(qb>=lo)&(qb<=hi),-1)
    write_json(output/'development-review.json',dict(created_utc=utc(),selected=selected,conditions=results,
        development_twin_coverage=float(both[ids].mean()),
        scoring='joint teachers and learned body mapping only; no evaluation geometry',
        state_source_sha256=sha(Path(__file__).with_name('evidence_state.py'))))


def freeze(output):
    if (output/'protocol-freeze.json').exists():raise FileExistsError('Already frozen')
    assert not any(source_for(output,seed,'clean').exists() or source_for(output,seed,'execution_hold').exists() for seed in FINAL), 'Final data must follow freeze'
    e12.verify(e12.ROOT)
    selected=read_json(output/'development-selection.json')['selected']
    review=read_json(output/'development-review.json')
    assert review['selected']==selected
    assert review['state_source_sha256']==sha(Path(__file__).with_name('evidence_state.py'))
    names=('evidence_cycle.py','evidence_state.py','body_cycle.py','vision.py',
        'reliable_cycle_v2.py','extent_cycle_v2.py',*e12.verify(e12.ROOT)['sources'].keys())
    dependencies=(e7.BODY,e7.BANK,e7.SEGMENTATION,e8.ROOT/'per_joint.pt',e9.ROOT/'atlas-multi.npz',
        e9.ROOT/'matched-budget-indices.npy',output/'action.pt',e12.ROOT/'protocol-freeze.json',e12.ROOT/'calibration.json',
        e12.ROOT/'development-selection.json',output/'development-review.json')
    write_json(output/'protocol-freeze.json',dict(created_utc=utc(),selected=selected,development_seed=DEV,
        calibration_seed=CAL,final_seeds=FINAL,conditions=CONDITIONS,frames=180,models_retrained=False,
        sources={n:sha(Path(__file__).with_name(n)) for n in dict.fromkeys(names)},
        dependencies={str(p):sha(p) for p in dependencies},protocol_sha256=sha(PROTOCOL),
        calibration_sha256=sha(output/'calibration.json'),selection_sha256=sha(output/'development-selection.json'),
        baseline_confirm=2,association_factor=1.5,patches=82,seed_budget=4,
        final_data_absent_before_freeze=True,revision='visual-motion-evidence'))
    print('E13 FROZEN',selected,flush=True)


def verify(output):
    frozen=read_json(output/'protocol-freeze.json')
    for n,d in frozen['sources'].items():assert sha(Path(__file__).with_name(n))==d,n
    for p,d in frozen['dependencies'].items():assert sha(p)==d,p
    for p,k in ((PROTOCOL,'protocol_sha256'),(output/'calibration.json','calibration_sha256'),
        (output/'development-selection.json','selection_sha256')):assert sha(p)==frozen[k],p
    e12.verify(e12.ROOT)
    return frozen


def final(output):
    frozen=verify(output)
    if (output/'predictions-complete.json').exists():raise FileExistsError('Predictions sealed')
    destination=output/'predictions';destination.mkdir(exist_ok=True)
    for seed in FINAL:
        inputs(output,seed);measurements(output,seed)
        for condition in CONDITIONS:
            path=destination/f'{seed}-{condition}.npz'
            if path.exists():continue
            prediction=rollout(output,seed,condition,frozen['selected'],all_modes=True)
            np.savez_compressed(path,**prediction)
            write_json(path.with_suffix('.json'),dict(created_utc=utc(),sha256=sha(path),
                sensor_sha256=sha(output/'sensors'/f'{seed}-{condition}.npz'),
                measurement_sha256=sha(output/'measurements'/f'{seed}-{condition}.npz'),
                geometry_used=False,teacher_used=False,fault_label_as_input=False))
            print('E13 predicted',seed,condition,flush=True)
    paths=sorted(destination.glob('*.npz'))
    write_json(output/'predictions-complete.json',dict(created_utc=utc(),
        hashes={str(p):sha(p) for p in paths},all_predictions_before_privileged_scoring=True))
    print('E13 ALL PREDICTIONS SEALED',flush=True)


def score(output):
    """Privileged evaluation only after all predictions are sealed."""
    frozen=verify(output)
    if (output/'scores.json').exists():raise FileExistsError('Preserve completed scores')
    seal=read_json(output/'predictions-complete.json')
    assert len(seal['hashes'])==len(FINAL)*len(CONDITIONS)
    for p,d in seal['hashes'].items():assert sha(p)==d
    body,_,fitter=components(output)
    radii=np.array(read_json(output/'calibration.json')['radius_normalized'])
    oldsel=read_json(e12.ROOT/'development-selection.json')
    oldmargin=np.array(oldsel['profiles'][oldsel['selected']]['margin_normalized'])
    frames=[];rows=[];twins=[];events=[];mask=e12.e10.active_mask()
    for seed in FINAL:
        truth_cache={}
        for kind in ('clean','execution_hold'):
            source=source_for(output,seed,kind)
            truth_cache[kind]=[e7.load_npz(source/'evaluation'/r['file']) for r in read_json(source/'metadata.json')['frames']]
        for condition in CONDITIONS:
            held=condition in HELD;truth=truth_cache['execution_hold' if held else 'clean']
            tq=np.stack([t['actual_q'] for t in truth]);anchor=np.stack([t['anchor_xyz'] for t in truth]);ids=truth[0]['anchor_body'].astype(bool)
            p=e7.load_npz(output/'predictions'/f'{seed}-{condition}.npz')
            clean=e7.load_npz(output/'predictions'/f'{seed}-{"execution_hold" if held else "clean"}.npz')
            low,high=branch_envelope(p['bank'],p['bank_valid'],fitter.scale,radii)
            narrowlo,narrowhi=branch_envelope(p['bank'],p['bank_valid'],fitter.scale,oldmargin)
            covered=np.all((tq>=low)&(tq<=high),-1);narrow=np.all((tq>=narrowlo)&(tq<=narrowhi),-1)
            widths=np.mean((high-low)/(2*fitter.scale),-1)
            for mode in MODES:
                xyz=body(p[f'q_{mode}']);error=np.sqrt(np.mean(np.sum((xyz[:,ids]-anchor[:,ids])**2,-1),-1))
                clearxyz=body(clean[f'q_{mode}']);clearerr=np.sqrt(np.mean(np.sum((clearxyz[:,ids]-anchor[:,ids])**2,-1),-1))
                separation=np.sqrt(np.mean(np.sum((xyz-clearxyz)**2,-1),-1))
                off=[float(np.mean(e9.primitive_surface_distance(fitter.atlas.predict(q),t)[0]>.015)) for q,t in zip(p[f'q_{mode}'],truth)]
                rows.append(dict(seed=seed,condition=condition,mode=mode,anchors=int(ids.sum()),
                    rmse_mm=float(np.sqrt(np.mean(error**2))*1000),fault_rmse_mm=float(np.sqrt(np.mean(error[mask]**2))*1000),
                    hold_rmse_mm=float(np.sqrt(np.mean(error[30:150]**2))*1000) if held else None,off_surface=float(np.mean(off)),
                    unknown=float(p['unknown'].mean()) if mode=='evidence' else None,
                    coverage=float(covered.mean()) if mode=='evidence' else None,
                    narrow_coverage=float(narrow.mean()) if mode=='evidence' else None,
                    width_normalized=float(widths.mean()) if mode=='evidence' else None,
                    visual_updates=int(p['visual_selected'].sum()) if mode=='evidence' else None))
                for i in range(180):
                    item=dict(seed=seed,condition=condition,mode=mode,frame=i,error_mm=float(error[i]*1000),
                        separation_mm=float(separation[i]*1000),off_surface=off[i],fault_window=bool(mask[i]),held_physics=held)
                    if mode=='evidence':item.update(unknown=bool(p['unknown'][i]),covered=bool(covered[i]),
                        narrow_covered=bool(narrow[i]),width_normalized=float(widths[i]),committed=bool(p['committed'][i]),
                        visual_selected=bool(p['visual_selected'][i]),visual_history=int(p['visual_history'][i]),
                        support=int(p['support'][i]),override_action=bool(p['override_action'][i]),sensor_credible=bool(p['sensor_credible'][i]),
                        hypotheses=int(p['bank_valid'][i].sum()),raw_residual_mm=float(p['raw_residual'][i]*1000))
                    frames.append(item)
                if condition in ('twin_held','hold_feedback_bias','hold_no_camera','hold_no_feedback'):
                    floor=np.sqrt(np.mean(np.sum((body(tq)[:,ids]-anchor[:,ids])**2,-1),-1))
                    distance=np.array([fitter.distance(q,t) for q,t in zip(p[f'q_{mode}'],tq)])
                    good=(distance<=.005)&(error<=floor+.002)
                    for start,stop in ((70,110),(140,150)):
                        ends=[i for i in range(start+4,stop) if good[i-4:i+1].all()]
                        events.append(dict(seed=seed,condition=condition,mode=mode,start=start,stop=stop,
                            held_absolute=dict(recovered=bool(ends),time_s=(ends[0]-start+1)*.1 if ends else None)))
                if condition not in ('clean','execution_hold'):
                    for start,stop in e12.e10.RETURNS:
                        event=e12.recovery_event(separation,error-clearerr,start,stop)
                        event.update(seed=seed,condition=condition,mode=mode);events.append(event)
            print('E13 scored',seed,condition,flush=True)
        a=e7.load_npz(output/'predictions'/f'{seed}-twin_moving.npz');b=e7.load_npz(output/'predictions'/f'{seed}-twin_held.npz')
        for k in a:np.testing.assert_array_equal(a[k][:70],b[k][:70])
        qa=np.stack([t['actual_q'] for t in truth_cache['clean']]);qb=np.stack([t['actual_q'] for t in truth_cache['execution_hold']])
        gap=np.array([fitter.distance(x,y) for x,y in zip(qa,qb)])
        valid=(np.arange(180)>=30)&(np.arange(180)<70)&(gap>.015)
        item=dict(seed=seed,distinguishable_physical_frames=int(valid.sum()),unknown_frames=int((valid&a['unknown']).sum()),
            unknown_rate=float(a['unknown'][valid].mean()),input_prefix_identical=True,output_prefix_identical=True)
        for name,bank,active,rs in (('evidence',a['bank'],a['bank_valid'],radii),
            ('narrow',a['bank'],a['bank_valid'],oldmargin),('e12',a['e12_bank'],a['e12_bank_valid'],oldmargin)):
            lo,hi=branch_envelope(bank,active,fitter.scale,rs)
            both=np.all((qa>=lo)&(qa<=hi)&(qb>=lo)&(qb<=hi),-1)
            item[f'{name}_both_covered']=int((both&valid).sum());item[f'{name}_coverage']=float(both[valid].mean())
            item[f'{name}_width']=float(np.mean((hi-lo)[valid]/(2*fitter.scale)))
        twins.append(item)
    write_json(output/'frame-scores.json',frames);write_json(output/'recovery-events.json',events)
    write_json(output/'scores.json',dict(created_utc=utc(),rows=rows,twins=twins,
        predictions_seal_sha256=sha(output/'predictions-complete.json'),truth_read_after_seal=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('develop','review','freeze','final','score','verify'))
    parser.add_argument('--output',type=Path,default=ROOT)
    args=parser.parse_args();torch.set_num_threads(4);globals()[args.stage](args.output)

if __name__=='__main__':main()
