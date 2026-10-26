"""Verify E10 frozen artifacts and intervention/ablation invariants."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import numpy as np

from .body_cycle import read_json, write_json, sha, utc
from . import trust_cycle as e7
from . import reliable_cycle_v2 as e8
from . import extent_cycle_v2 as e9
from .conflict_cycle import ROOT, FINAL, CONDITIONS, MODES, verify, active_mask


def audit(output):
    verify(output)
    e8.verify_freeze(e8.ROOT)
    e9.verify(e9.ROOT)
    sealed = read_json(output/'predictions-complete.json')
    for path, digest in sealed['prediction_hashes'].items():
        assert sha(path) == digest
    comparisons = []
    command_files = 0
    for seed in FINAL:
        source = output/f'data-{seed}'
        meta = read_json(source/'metadata.json')
        assert meta['prop_max_displacement_m'] == 0.
        assert meta['fixed_body_points'] == 28
        for name, digest in meta['before_action_hashes'].items():
            assert sha(source/'before'/name) == digest
            command_files += 1
        clean = e7.load_npz(output/'predictions'/f'{seed}-clean.npz')
        clean_sensor = e7.load_npz(output/'sensors'/f'{seed}-clean.npz')
        for condition in CONDITIONS:
            p = e7.load_npz(output/'predictions'/f'{seed}-{condition}.npz')
            sensor_path = output/'sensors'/f'{seed}-{condition}.npz'
            sensors = e7.load_npz(sensor_path)
            sidecar = read_json((output/'predictions'/f'{seed}-{condition}.npz').with_suffix('.json'))
            assert sha(sensor_path) == sidecar['sensor_file_sha256']
            assert not sidecar['inference_truth_read']
            for value in p.values():
                assert np.isfinite(value).all()
            for mode in MODES:
                np.testing.assert_array_equal(p[f'q_{mode}'][:30], clean[f'q_{mode}'][:30])
            np.testing.assert_array_equal(p['q_action'], clean['q_action'])
            np.testing.assert_allclose(p['q_action'], sensors['action_reference'], atol=2e-6, rtol=0)
            for key in ('feedback', 'visual', 'quality', 'available'):
                np.testing.assert_array_equal(sensors[key][~active_mask()], clean_sensor[key][~active_mask()])
            if condition == 'stale_rgb':
                np.testing.assert_array_equal(p['q_no_visual'], clean['q_no_visual'])
            if condition == 'feedback_bias':
                np.testing.assert_array_equal(p['q_no_feedback'], clean['q_no_feedback'])
            if condition == 'coherent_replay':
                clean_readings = np.stack([e7.load_npz(source/'after'/r['file'])['q'] for r in meta['frames']])
                ids = np.flatnonzero(active_mask())
                np.testing.assert_array_equal(sensors['feedback'][ids], clean_readings[ids % 20])
            # Missing channels have exact zero contribution, at every joint.
            for source_index, available_index in ((1, 0), (2, 1)):
                missing = ~sensors['available'][:, available_index]
                assert np.all(p['weights'][missing, :, source_index] == 0)
            assert np.all(p['envelope_high'] >= p['envelope_low'])
            comparisons.append(dict(seed=seed, condition=condition, finite=True, causal_prefix=True,
                                    sensor_recovery_exact=True, action_independent=True))
    source_path = Path(__file__).with_name('conflict_cycle.py')
    tree = ast.parse(source_path.read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'predict')
    literals = [n.value for n in ast.walk(function) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert not any('evaluation' in x or 'actual_q' in x for x in literals)
    scores = read_json(output/'scores.json')
    assert scores['truth_read_after_seal']
    report_manifest = read_json(output/'report-artifacts.json') if (output/'report-artifacts.json').exists() else None
    if report_manifest:
        for name, digest in report_manifest['hashes'].items():
            assert sha(name) == digest, name
    result = dict(created_utc=utc(), final_seeds=FINAL, prediction_files=len(comparisons),
        command_before_action_files=command_files, fixed_props_asserted=True,
        original_e8_e9_freezes_verified=True, inference_ast_has_no_truth_paths=True,
        sensor_emulation_separate_from_inference=True, intervention_checks=comparisons,
        reports_verified=bool(report_manifest), scoring_seal_verified=True)
    write_json(output/'audit.json', result)
    print('E10 audit passed:', len(comparisons), 'prediction files;', command_files, 'commands', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT)
    audit(parser.parse_args().output)
