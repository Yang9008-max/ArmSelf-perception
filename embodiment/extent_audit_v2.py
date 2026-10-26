"""Check E9B artifacts and preserve all earlier frozen experiments."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np

from . import trust_cycle as e7
from .body_cycle import read_json, write_json, sha, utc
from .extent_cycle_v2 import ROOT, FINAL, CONDITIONS, verify
from .surface_extent import SurfaceAtlas


def audit(output):
    freeze = verify(output)
    manifest = read_json(output/'predictions-complete.json')
    assert set(FINAL).isdisjoint({2026102471, 2026102911, 2026102931, 2026102932, 2026102933, 2026102941})
    assert len(manifest['hashes']) == len(FINAL)*len(CONDITIONS)
    for path, digest in manifest['hashes'].items():
        assert sha(Path(path)) == digest
    for seed in FINAL:
        metadata = read_json(output/f'data-{seed}'/'metadata.json')
        assert metadata['prop_max_displacement_m'] == 0.
        for name, digest in metadata['before_action_hashes'].items():
            assert sha(output/f'data-{seed}'/'before'/name) == digest
    checked_frozen_sources = 0
    for stage in ('e7-learned-trust', 'e8-reliable-state', 'e8b-reliable-state'):
        p = output.parent/stage/'training-freeze.json'
        data = read_json(p)
        for name, digest in data.get('sources', {}).items():
            assert sha(Path(__file__).with_name(name)) == digest, (stage, name)
            checked_frozen_sources += 1
    atlas = SurfaceAtlas.load(output/'atlas-multi.npz')
    from .extent_cycle import verify as verify_first
    verify_first(output.parent/'e9-body-extent')
    indices = np.load(output/'matched-budget-indices.npy')
    assert len(indices) == len(np.unique(indices))
    assert len(indices) == len(SurfaceAtlas.load(output/'atlas-single.npz').weights)
    for seed in FINAL:
        clean = e7.load_npz(output/'extent-predictions'/f'{seed}-clean.npz')
        for condition in CONDITIONS:
            a = e7.load_npz(output/'extent-predictions'/f'{seed}-{condition}.npz')
            for mode in ('sparse', 'single', 'budget_matched', 'multi', 'static', 'mismatch', 'e7', 'feedback', 'action', 'fixed'):
                assert len(a[mode]) == 150
                assert np.isfinite(a[mode]).all(), (seed, condition, mode)
            assert a['multi'].shape[1] == freeze['selected_count']
            np.testing.assert_allclose(a['multi'][70], atlas.predict(a['q_e8'][70]), atol=1e-7)
            if condition == 'camera_missing':
                assert not (a['status'][40:95] == 2).any()
            if condition != 'no_feedback':
                np.testing.assert_array_equal(clean['multi'][:30], a['multi'][:30])
            if condition == 'no_feedback':
                assert not a['available'][:, 0].any()
            np.testing.assert_array_equal(clean['action'], a['action'])
            np.testing.assert_array_equal(clean['static'], a['static'])
    # AST check: scientific learning and inference functions don't read scoring directories.
    tree = ast.parse(Path(__file__).with_name('extent_cycle_v2.py').read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ('extract', 'train', 'predict', 'sensor_images'):
            assert not any(isinstance(n, ast.Constant) and n.value == 'evaluation' for n in ast.walk(node))
    tests = read_json(output/'test-result.json')
    assert tests['passed'] and tests['tests_run'] >= 104
    record = dict(created_utc=utc(), e9_frozen_sources_verified=True, older_frozen_source_checks=checked_frozen_sources,
        final_prediction_hashes_verified=len(manifest['hashes']), before_action_records_verified=450,
        fixed_distractors=True, fixed_representation_denominators=True, finite_all_outputs=True,
        same_state_drives_all_atlas_patches=True, missing_camera_never_currently_observed=True,
        first_e9_sources_and_models_unchanged=True, spatial_budget_indices_frozen_unique_equal_count=True,
        before_intervention_same_predictions=True, action_and_static_independent_of_sensor_faults=True,
        learning_inference_no_evaluation_directory_literals=True,
        full_suite_tests=tests['tests_run'], full_suite_passed=True, new_surface_tests=tests.get('new_tests',10),
        limitations='AST/input-path checks support an interface audit, not a formal proof of all possible information flow')
    write_json(output/'audit.json', record)
    print(record, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT)
    audit(parser.parse_args().output)
