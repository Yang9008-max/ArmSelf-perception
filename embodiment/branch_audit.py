"""Independent E12 delivery, causal-prefix and state-update audit."""
from __future__ import annotations
import ast
from pathlib import Path
import numpy as np
from . import branch_cycle as run, hypothesis_cycle as e11, trust_cycle as e7
from .body_cycle import read_json,write_json,sha,utc


def audit(output=run.ROOT):
    run.verify(output);e11.verify(e11.ROOT)
    seal=read_json(output/'predictions-complete.json')
    assert len(seal['hashes'])==len(run.FINAL)*len(run.CONDITIONS)
    assert seal['all_predictions_before_privileged_scoring']
    assert read_json(output/'scores.json')['truth_read_after_seal']
    for path,digest in seal['hashes'].items():
        assert sha(path)==digest,path
    _,_,fitter=run.components(output)
    commands=commits=changed=overrides=future_changes=0;twins=[];nearest=[]
    assert len(fitter.atlas.weights)==434
    assert all(r['anchors']==28 for r in read_json(output/'scores.json')['rows'])
    for seed in run.FINAL:
        for condition in ('clean','execution_hold'):
            source=run.source_for(output,seed,condition)
            meta=read_json(source/'metadata.json')
            assert meta['prop_max_displacement_m']==0
            assert meta['fixed_body_points']==28
            if condition=='execution_hold':
                np.testing.assert_array_equal([r['held'] for r in meta['frames']],(np.arange(180)>=30)&(np.arange(180)<150))
            for name,digest in meta['before_action_hashes'].items():
                assert sha(source/'before'/name)==digest
                commands+=1
        inputs=read_json(output/f'inputs-{seed}-complete.json')
        for condition in run.CONDITIONS:
            path=output/'predictions'/f'{seed}-{condition}.npz'
            a=e7.load_npz(output/'sensors'/f'{seed}-{condition}.npz')
            m=e7.load_npz(output/'measurements'/f'{seed}-{condition}.npz')
            p=e7.load_npz(path);note=read_json(path.with_suffix('.json'))
            assert sha(path)==note['sha256']
            assert sha(output/'sensors'/f'{seed}-{condition}.npz')==note['sensor_sha256']==inputs['sensor_hashes'][condition]
            assert sha(output/'measurements'/f'{seed}-{condition}.npz')==note['measurement_sha256']
            assert not any(note[k] for k in ('geometry_used','teacher_used','fault_label_as_input'))
            assert m['xyz'].shape==(180,82,3)
            assert set(np.unique(m['origin']))<={0,1,2}
            np.testing.assert_array_equal(m['valid'],m['origin']!=0)
            for key,value in p.items():
                assert np.isfinite(value).all(),(seed,condition,key)
            for mode in run.MODES:
                assert p[f'q_{mode}'].shape==(180,7)
            np.testing.assert_array_equal(p['q_action'],a['action_reference'])
            assert np.all(p['bank_valid'].sum(-1)[p['unknown']]>=2)
            assert np.all(p['bank_valid'].sum(-1)[~p['unknown']]==1)
            commits+=int(p['committed'].sum());overrides+=int(p['override_action'].sum())
            changed+=int((p['committed']&(np.max(np.abs(p['q_branch']-p['proposal']),axis=-1)>1e-7)).sum())
            # Actual forecast can select the corrected stop branch while the
            # separately propagated open-loop command history remains different.
            active=p['override_action'][:-1]&p['sensor_mode'][:-1]
            future_changes+=int((active&(np.max(np.abs(p['prior'][1:]-p['open_action'][1:]),axis=-1)>1e-7)).sum())
        for folder in ('sensors','measurements','predictions'):
            a=e7.load_npz(output/folder/f'{seed}-twin_moving.npz')
            b=e7.load_npz(output/folder/f'{seed}-twin_held.npz')
            for key in a:
                np.testing.assert_array_equal(a[key][:70],b[key][:70])
        p=e7.load_npz(output/'predictions'/f'{seed}-twin_moving.npz')
        actual=[]
        for condition in ('clean','execution_hold'):
            source=run.source_for(output,seed,condition)
            actual.append(np.stack([e7.load_npz(source/'evaluation'/r['file'])['actual_q'] for r in read_json(source/'metadata.json')['frames']]))
        gap=np.array([fitter.distance(x,y) for x,y in zip(*actual)])
        ids=np.flatnonzero((np.arange(180)>=30)&(np.arange(180)<70)&(gap>.015))
        item=dict(seed=seed,frames=len(ids))
        for name,truth in zip(('moving','held'),actual):
            d=[]
            for i in ids:
                retained=[fitter.distance(q,truth[i]) for q,v in zip(p['bank'][i,:4],p['bank_valid'][i,:4]) if v]
                if retained:
                    d.append(min(retained))
            item[f'{name}_frames_without_retained_motion_branch']=int(len(ids)-len(d))
            item[f'{name}_nearest_median_mm']=float(np.median(d)*1000) if d else None
            item[f'{name}_nearest_p95_mm']=float(np.quantile(d,.95)*1000) if d else None
        nearest.append(item)
    tree=ast.parse(Path(run.__file__).read_text(encoding='utf8'))
    for name in ('rollout','measurements'):
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        literals=[n.value for n in ast.walk(fn) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
        assert not any('evaluation' in x or 'actual_q' in x or '/after' in x for x in literals)
    result=dict(created_utc=utc(),prediction_files=len(seal['hashes']),command_files=commands,
        commits=commits,commits_changing_point_estimate=changed,overrides=overrides,
        corrected_future_differs_from_open_action=future_changes,nearest_motion_candidates=nearest,
        exact_twin_prefix_frames=70,all_prefix_fields_equal=True,fixed_props=True,
        shared_28_anchor_434_surface=True,old_e11_freeze_verified=True,truth_read_after_seal=True,
        inference_truth_path_static_check=True)
    write_json(output/'audit.json',result)
    print('E12 AUDIT',result,flush=True)
    return result


if __name__=='__main__':
    audit()
