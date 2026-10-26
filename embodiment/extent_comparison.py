"""Post-hoc paired diagnosis of already frozen A/B atlases on B clean states.

This comparison was added after B final scoring; it selects no parameters and
does not replace either round's preregistered tests.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from . import trust_cycle as e7
from .body_cycle import read_json, write_json, sha, utc
from .extent_cycle import verify as verify_a
from .extent_cycle_v2 import ROOT, PREVIOUS, FINAL, verify, primitive_surface_distance
from .surface_extent import SurfaceAtlas
from .model import basis


def compare(output=ROOT):
    verify(output)
    verify_a(PREVIOUS)
    destination = output/'paired-registration'
    destination.mkdir(exist_ok=True)
    atlas = SurfaceAtlas.load(PREVIOUS/'atlas-multi.npz')
    for seed in FINAL:
        a = e7.load_npz(output/'extent-predictions'/f'{seed}-clean.npz')
        predicted = np.einsum('fk,nkc->fnc',basis((a['q_e8']-atlas.center)/atlas.scale),atlas.weights).astype(np.float32)
        np.savez_compressed(destination/f'{seed}.npz', earlier_atlas=predicted, later_atlas=a['multi'], sparse=a['sparse'])
    write_json(destination/'prediction-hashes.json',dict(created_utc=utc(),
        posthoc=True, all_paired_predictions_saved_before_paired_scoring=True,
        prior_final_scores_already_known=True, state='same frozen E8B clean q in all methods',
        hashes={p.name:sha(p) for p in sorted(destination.glob('*.npz'))}))
    rows = []
    for seed in FINAL:
        source = output/f'data-{seed}'
        metadata = read_json(source/'metadata.json')
        cal = metadata['calibration']
        a = e7.load_npz(destination/f'{seed}.npz')
        for i, frame in enumerate(metadata['frames']):
            truth = e7.load_npz(source/'evaluation'/frame['file'])
            vv,uu = np.mgrid[0:480:3,0:640:3]
            use = truth['clear_body_mask'][vv,uu]
            u,v = uu[use],vv[use]
            z = truth['clear_depth'][v,u]
            targets = np.c_[(u-cal['cx'])*z/cal['focal_px'],(v-cal['cy'])*z/cal['focal_px'],z]
            for mode in a:
                points = a[mode][i]
                distances = cKDTree(points).query(targets)[0]
                surface,_ = primitive_surface_distance(points,truth)
                rows.append(dict(seed=seed,frame=i,mode=mode,targets=len(targets),
                    covered=int((distances<=.015).sum()),points=len(points),
                    off=int((surface>.015).sum()),distance_sum=float(surface.sum())))
    def aggregate(part):
        return dict(coverage=sum(r['covered'] for r in part)/sum(r['targets'] for r in part),
            off_surface=sum(r['off'] for r in part)/sum(r['points'] for r in part),
            surface_distance_mm=1000*sum(r['distance_sum'] for r in part)/sum(r['points'] for r in part),
            point_count=part[0]['points'])
    modes = ('sparse','earlier_atlas','later_atlas')
    result = dict(created_utc=utc(),posthoc=True,selected_no_parameters=True,
        interpretation='same-state paired accuracy/coverage tradeoff, secondary to frozen primary tests',
        pooled={m:aggregate([r for r in rows if r['mode']==m]) for m in modes},
        per_seed={str(s):{m:aggregate([r for r in rows if r['mode']==m and r['seed']==s]) for m in modes} for s in FINAL})
    write_json(destination/'summary.json',result)
    print(result,flush=True)


if __name__ == '__main__':
    compare()
