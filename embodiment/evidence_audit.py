"""E13 provenance, causal-prefix, actual-update and frozen-baseline audit."""
from __future__ import annotations
import ast
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
from . import evidence_cycle as run,trust_cycle as e7
from .body_cycle import read_json,write_json,sha,utc


def audit(output=run.ROOT):
    frozen=run.verify(output);seal=read_json(output/'predictions-complete.json')
    assert seal['all_predictions_before_privileged_scoring']
    assert frozen['final_data_absent_before_freeze']
    assert read_json(output/'scores.json')['truth_read_after_seal']
    assert len(seal['hashes'])==len(run.FINAL)*len(run.CONDITIONS)
    for p,d in seal['hashes'].items():assert sha(p)==d,p
    _,_,fitter=run.components(output)
    commands=visual=changed=future=original_checks=total_support=extra_winners=0
    nearest=[];visual_events=[];fault_selections=0;selected_by_condition={c:0 for c in run.CONDITIONS}
    assert len(fitter.atlas.weights)==434
    assert all(r['anchors']==28 for r in read_json(output/'scores.json')['rows'])
    for seed in run.FINAL:
        for c in ('clean','execution_hold'):
            source=run.source_for(output,seed,c);meta=read_json(source/'metadata.json')
            assert datetime.fromisoformat(meta['created_utc'])>datetime.fromisoformat(frozen['created_utc'])
            assert meta['prop_max_displacement_m']==0.;assert meta['fixed_body_points']==28
            for name,d in meta['before_action_hashes'].items():
                assert sha(source/'before'/name)==d;commands+=1
        initial=e7.load_npz(run.source_for(output,seed,'clean')/'initial.npz')['q']
        np.testing.assert_array_equal(initial,e7.load_npz(run.source_for(output,seed,'execution_hold')/'initial.npz')['q'])
        inputs=read_json(output/f'inputs-{seed}-complete.json')
        for c in run.CONDITIONS:
            path=output/'predictions'/f'{seed}-{c}.npz';note=read_json(path.with_suffix('.json'))
            a=e7.load_npz(output/'sensors'/f'{seed}-{c}.npz')
            m=e7.load_npz(output/'measurements'/f'{seed}-{c}.npz');p=e7.load_npz(path)
            assert sha(path)==note['sha256']
            assert sha(output/'sensors'/f'{seed}-{c}.npz')==note['sensor_sha256']==inputs['sensor_hashes'][c]
            assert sha(output/'measurements'/f'{seed}-{c}.npz')==note['measurement_sha256']
            assert not any(note[k] for k in ('geometry_used','teacher_used','fault_label_as_input'))
            assert m['xyz'].shape==(180,82,3)
            for prefix,origins in (('',{0,1,2,4,8}),('original_',{0,1,2})):
                assert set(np.unique(m[prefix+'origin']))<=origins
                np.testing.assert_array_equal(m[prefix+'valid'],m[prefix+'origin']!=0)
            for k,v in p.items():assert np.isfinite(v).all(),(seed,c,k)
            for mode in run.MODES:assert p[f'q_{mode}'].shape==(180,7)
            np.testing.assert_array_equal(p['q_action'],a['action_reference'])
            np.testing.assert_array_equal(p['q_e12'],p['q_no_visual'])
            assert np.all(p['bank_valid'].sum(-1)[p['unknown']]>=2)
            assert np.all(p['bank_valid'].sum(-1)[~p['unknown']]==1)
            assert not np.any(p['visual_selected']&(p['support']<12))
            selected=p['visual_selected'];selected_by_condition[c]+=int(selected.sum())
            assert not (selected&p['sensor_credible']).any()
            assert p['unknown'][selected].all()
            visual+=int(selected.sum())
            fault_selections+=int((selected&run.e12.e10.active_mask()).sum())
            if c in ('hold_feedback_bias','hold_no_feedback'):
                for start,stop in ((30,70),(110,140)):
                    ids=np.flatnonzero(selected[start:stop])+start
                    visual_events.append(dict(seed=seed,condition=c,start=start,stop=stop,
                        visual_selections=len(ids),first_update_frame=int(ids[0]) if len(ids) else None,
                        first_update_delay_s=float((ids[0]-start+1)*.1) if len(ids) else None,
                        feedback_available_at_first=bool(a['available'][ids[0],0]) if len(ids) else None))
            changed+=int((selected&(np.max(np.abs(p['q_evidence']-p['q_no_visual']),-1)>1e-7)).sum())
            future+=int((selected[:-1]&(np.max(np.abs(p['prior'][1:]-p['open_action'][1:]),-1)>1e-7)).sum())
            total_support+=int(m['valid'].sum());extra_winners+=int((m['origin']>=4).sum())
            original_checks+=read_json((output/'measurements'/f'{seed}-{c}.json'))['original_observer_exact_comparisons']
        for folder in ('sensors','measurements','predictions'):
            a=e7.load_npz(output/folder/f'{seed}-twin_moving.npz');b=e7.load_npz(output/folder/f'{seed}-twin_held.npz')
            for k in a:np.testing.assert_array_equal(a[k][:70],b[k][:70])
        pred=e7.load_npz(output/'predictions'/f'{seed}-twin_moving.npz');actual=[]
        for c in ('clean','execution_hold'):
            source=run.source_for(output,seed,c)
            actual.append(np.stack([e7.load_npz(source/'evaluation'/r['file'])['actual_q'] for r in read_json(source/'metadata.json')['frames']]))
        gap=np.array([fitter.distance(x,y) for x,y in zip(*actual)])
        ids=np.flatnonzero((np.arange(180)>=30)&(np.arange(180)<70)&(gap>.015))
        item=dict(seed=seed,frames=len(ids))
        for label,truth in zip(('moving','held'),actual):
            distances=[min(fitter.distance(q,truth[i]) for q,v in zip(pred['bank'][i,:4],pred['bank_valid'][i,:4]) if v) for i in ids]
            item[label+'_nearest_median_mm']=float(np.median(distances)*1000)
            item[label+'_nearest_p95_mm']=float(np.quantile(distances,.95)*1000)
        nearest.append(item)
    tree=ast.parse(Path(run.__file__).read_text(encoding='utf8'))
    for name in ('rollout','measurements'):
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        literals=[n.value for n in ast.walk(fn) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
        assert not any('evaluation' in x or 'actual_q' in x or '/after' in x for x in literals)
    result=dict(created_utc=utc(),prediction_files=len(seal['hashes']),command_files=commands,
        visual_selections=visual,visual_selections_differing_from_ablation=changed,
        visual_selections_inside_fault_window=fault_selections,visual_acquisition_events=visual_events,
        visually_corrected_future_differs_from_open_action=future,visual_updates_by_condition=selected_by_condition,
        valid_patch_frames=total_support,stop_seed_winning_patch_frames=extra_winners,
        original_observer_exact_comparisons=original_checks,nearest_motion_candidates=nearest,
        exact_twin_prefix_frames=70,all_prefix_fields_equal=True,fixed_props=True,
        shared_28_anchor_434_surface=True,old_e12_freeze_verified=True,truth_read_after_seal=True,
        no_visual_ablation_exactly_matches_e12=True,inference_truth_path_static_check=True)
    write_json(output/'audit.json',result);print('E13 AUDIT',result,flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=run.ROOT)
    audit(parser.parse_args().output)
