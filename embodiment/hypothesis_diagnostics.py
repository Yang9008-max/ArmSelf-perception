"""Post-seal descriptive checks; these do not select models or revise acceptance."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from .body_cycle import read_json,write_json,utc
from . import trust_cycle as e7
from .hypothesis_cycle import ROOT,FINAL,CONDITIONS,verify,source_for,components


def diagnose(output):
    verify(output)
    assert read_json(output/'scores.json')['truth_read_after_seal']
    body,transition,fitter=components(output)
    twins=[];updates=[]
    for seed in FINAL:
        qa=[];qb=[]
        for condition,qs in (('clean',qa),('execution_hold',qb)):
            source=source_for(output,seed,condition)
            for row in read_json(source/'metadata.json')['frames']:
                qs.append(e7.load_npz(source/'evaluation'/row['file'])['actual_q'])
        p=e7.load_npz(output/'predictions'/f'{seed}-twin_moving.npz')
        nearest_a=[];nearest_b=[]
        for i in range(30,70):
            if fitter.distance(qa[i],qb[i])<=.015:
                continue
            bank=p['bank'][i][p['bank_valid'][i]]
            nearest_a.append(min(fitter.distance(q,qa[i]) for q in bank)*1000)
            nearest_b.append(min(fitter.distance(q,qb[i]) for q in bank)*1000)
        twins.append(dict(seed=seed,frames=len(nearest_a),
            nearest_moving_median_mm=float(np.median(nearest_a)) if nearest_a else None,
            nearest_moving_p95_mm=float(np.quantile(nearest_a,.95)) if nearest_a else None,
            nearest_held_median_mm=float(np.median(nearest_b)) if nearest_b else None,
            nearest_held_p95_mm=float(np.quantile(nearest_b,.95)) if nearest_b else None))
        for condition in CONDITIONS:
            source=source_for(output,seed,condition)
            metadata=read_json(source/'metadata.json')
            p=e7.load_npz(output/'predictions'/f'{seed}-{condition}.npz')
            a=e7.load_npz(output/'sensors'/f'{seed}-{condition}.npz')
            for i in np.flatnonzero(p['committed']):
                truth=e7.load_npz(source/'evaluation'/metadata['frames'][i]['file'])
                ids=truth['anchor_body'].astype(bool)
                def err(q,t=truth):
                    return float(np.sqrt(np.mean(np.sum((body(q)[0,ids]-t['anchor_xyz'][ids])**2,-1)))*1000)
                before,after=err(p['proposal'][i]),err(p['q_competing'][i])
                row=dict(seed=seed,condition=condition,frame=int(i),before_mm=before,after_mm=after,
                         improvement_mm=before-after)
                if i<179:
                    old=p['q_competing'][i-1] if i and p['committed'][i-1] else (
                        p['prior'][i-1] if i else e7.load_npz(source/'initial.npz')['q'])
                    velocity=np.clip((p['prior'][i]-old)/float(a['duration'][i]),-fitter.scale*20,fitter.scale*20)
                    uncorrected=transition.predict(p['prior'][i],velocity,a['target'][i+1],float(a['duration'][i+1]))
                    next_truth=e7.load_npz(source/'evaluation'/metadata['frames'][i+1]['file'])
                    row['next_forecast_improvement_mm']=err(uncorrected,next_truth)-err(p['prior'][i+1],next_truth)
                updates.append(row)
    values=np.array([r['improvement_mm'] for r in updates])
    result=dict(created_utc=utc(),post_hoc_descriptive_only=True,selection_unchanged=True,
        nearest_candidates=twins,commit_events=updates,
        commits_helped_gt_001mm=int((values>.01).sum()),commits_hurt_gt_001mm=int((values<-.01).sum()),
        median_instant_improvement_mm=float(np.median(values)) if len(values) else None)
    write_json(output/'diagnostics.json',result)
    print('E11 descriptive diagnostics complete:',len(updates),'commits',flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT)
    diagnose(parser.parse_args().output)
