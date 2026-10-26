"""Verify E11 causal inputs, provenance, frozen artifacts and actual updates."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import numpy as np

from .body_cycle import read_json, write_json, sha, utc
from . import trust_cycle as e7
from . import conflict_cycle as e10
from . import reliable_cycle_v2 as e8
from . import extent_cycle_v2 as e9
from .hypothesis_cycle import ROOT, FINAL, CONDITIONS, MODES, verify, source_for, components


def audit(output):
    frozen=verify(output)
    e10.verify(e10.ROOT)
    e8.verify_freeze(e8.ROOT)
    e9.verify(e9.ROOT)
    seal=read_json(output/'predictions-complete.json')
    assert len(seal['hashes'])==30
    for path,digest in seal['hashes'].items():
        assert sha(path)==digest,path
    assert seal['all_predictions_before_privileged_scoring']
    assert read_json(output/'scores.json')['truth_read_after_seal']
    _,transition,fitter=components(output)
    assert len(fitter.atlas.weights)==434
    assert all(r['anchors']==28 for r in read_json(output/'scores.json')['rows'])
    checks=[];commands=0;changed_commits=0;changed_future=0;commit_count=0
    for seed in FINAL:
        for condition in ('clean','execution_hold'):
            source=source_for(output,seed,condition)
            meta=read_json(source/'metadata.json')
            assert meta['prop_max_displacement_m']==0.
            assert meta['fixed_body_points']==28
            for name,digest in meta['before_action_hashes'].items():
                assert sha(source/'before'/name)==digest
                commands+=1
        inputs=read_json(output/f'inputs-{seed}-complete.json')
        assert inputs['exact_input_prefix_verified_frames']==70
        clean=e7.load_npz(output/'sensors'/f'{seed}-clean.npz')
        held=e7.load_npz(output/'sensors'/f'{seed}-execution_hold.npz')
        for condition in CONDITIONS:
            path=output/'predictions'/f'{seed}-{condition}.npz'
            sensor_path=output/'sensors'/f'{seed}-{condition}.npz'
            measured_path=output/'measurements'/f'{seed}-{condition}.npz'
            p=e7.load_npz(path);a=e7.load_npz(sensor_path);m=e7.load_npz(measured_path)
            note=read_json(path.with_suffix('.json'))
            assert sha(path)==note['sha256']
            assert sha(sensor_path)==note['sensor_sha256']==inputs['sensor_hashes'][condition]
            assert sha(measured_path)==note['measurement_sha256']
            assert not any(note[k] for k in ('geometry_used','teacher_used','fault_label_as_input'))
            assert set(a)=={'target','duration','feedback','visual','quality','available','action_reference','rgb_paths'}
            assert m['xyz'].shape==(180,82,3)
            assert set(np.unique(m['origin']))<={0,1,2}
            np.testing.assert_array_equal(m['valid'],m['origin']!=0)
            for key,value in p.items():
                assert np.isfinite(value).all(),(seed,condition,key)
            for mode in MODES:
                assert p[f'q_{mode}'].shape==(180,7)
            np.testing.assert_array_equal(p['q_action'],a['action_reference'])
            np.testing.assert_array_equal(a['action_reference'],clean['action_reference'])
            assert np.all(p['bank_valid'].sum(-1)[p['unknown']]>=2)
            assert np.all(p['bank_valid'].sum(-1)[~p['unknown']]==1)
            for j in (0,1):
                assert not p['weights'][~a['available'][:,j],:,j+1].any()
            if condition=='no_camera':
                assert not m['valid'][e10.active_mask()].any()
            normal=held if condition in ('execution_hold','twin_held') else clean
            outside=~e10.active_mask()
            if condition=='twin_held':
                outside[:30]=False
            for key in ('target','duration','feedback','visual','quality','available'):
                np.testing.assert_array_equal(a[key][outside],normal[key][outside])
            for i in np.flatnonzero(p['committed']):
                commit_count+=1
                changed_commits+=bool(np.max(np.abs(p['q_competing'][i]-p['proposal'][i]))>1e-7)
                if i==179:
                    continue
                old=p['q_competing'][i-1] if i and p['committed'][i-1] else (
                    p['prior'][i-1] if i else e7.load_npz(source_for(output,seed,condition)/'initial.npz')['q'])
                velocity=np.clip((p['q_competing'][i]-old)/float(a['duration'][i]),-fitter.scale*20,fitter.scale*20)
                expected=transition.predict(p['q_competing'][i],velocity,a['target'][i+1],float(a['duration'][i+1]))
                np.testing.assert_allclose(expected,p['prior'][i+1],atol=2e-6,rtol=0)
                uncorrected_velocity=np.clip((p['prior'][i]-old)/float(a['duration'][i]),-fitter.scale*20,fitter.scale*20)
                counterfactual=transition.predict(p['prior'][i],uncorrected_velocity,a['target'][i+1],float(a['duration'][i+1]))
                changed_future+=bool(np.max(np.abs(expected-counterfactual))>1e-7)
            checks.append(dict(seed=seed,condition=condition,finite=True,patches=82,duplicate_votes=False))
        ma=e7.load_npz(output/'sensors'/f'{seed}-twin_moving.npz')
        mb=e7.load_npz(output/'sensors'/f'{seed}-twin_held.npz')
        for key in ma:
            np.testing.assert_array_equal(ma[key][:70],mb[key][:70])
        for folder in ('measurements','predictions'):
            ma=e7.load_npz(output/folder/f'{seed}-twin_moving.npz')
            mb=e7.load_npz(output/folder/f'{seed}-twin_held.npz')
            for key in ma:
                np.testing.assert_array_equal(ma[key][:70],mb[key][:70])
    tree=ast.parse(Path(__file__).with_name('hypothesis_cycle.py').read_text(encoding='utf8'))
    for name in ('rollout','measurements'):
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        literals=[n.value for n in ast.walk(function) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
        assert not any('evaluation' in x or 'actual_q' in x or '/after' in x for x in literals)
    if (output/'report-artifacts.json').exists():
        for path,digest in read_json(output/'report-artifacts.json')['hashes'].items():
            assert sha(path)==digest,path
    result=dict(created_utc=utc(),revision=frozen['revision'],prediction_files=len(checks),
        command_files=commands,all_old_model_freezes_verified=True,
        exact_twin_prefix_frames=70,all_prediction_fields_prefix_equal=True,
        commits=commit_count,commits_changing_point_estimate=changed_commits,
        commits_changing_future_forecast=changed_future,
        fixed_props=True,shared_28_anchor_and_434_surface_denominators=True,
        inference_truth_path_static_check=True,predictions_sealed_before_scoring=True,checks=checks)
    write_json(output/'audit.json',result)
    print('E11 audit passed:',len(checks),'predictions;',commands,'commands;',changed_future,'future forecast changes',flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT)
    audit(parser.parse_args().output)
