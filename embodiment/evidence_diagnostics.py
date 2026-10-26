"""Post-seal E13 coverage diagnosis, without changing any frozen parameter."""
import numpy as np
import argparse
from pathlib import Path
from . import evidence_cycle as run,trust_cycle as e7
from .evidence_state import branch_envelope
from .body_cycle import read_json,write_json,utc,sha


def main(output=run.ROOT):
    run.verify(output)
    assert (output/'predictions-complete.json').exists() and (output/'scores.json').exists()
    radii=np.array(read_json(output/'calibration.json')['radius_normalized'])
    _,_,fitter=run.components(output);rows=[]
    for seed in run.FINAL:
        p=e7.load_npz(output/'predictions'/f'{seed}-twin_moving.npz');q=[]
        for c in ('clean','execution_hold'):
            source=run.source_for(output,seed,c)
            q.append(np.stack([e7.load_npz(source/'evaluation'/r['file'])['actual_q'] for r in read_json(source/'metadata.json')['frames']]))
        gap=np.array([fitter.distance(a,b) for a,b in zip(*q)])
        ids=np.flatnonzero((np.arange(180)>=30)&(np.arange(180)<70)&(gap>.015))
        lo,hi=branch_envelope(p['bank'],p['bank_valid'],fitter.scale,radii)
        for name,truth in zip(('moving','held'),q):
            fail=(truth<lo)|(truth>hi)
            minimum=[np.max(np.abs((p['bank'][i,:4]-truth[i])/fitter.scale)/radii[:4],axis=-1).min() for i in ids]
            rows.append(dict(seed=seed,physical=name,frames=len(ids),covered=int((~fail[ids].any(-1)).sum()),
                failed_frames=ids[fail[ids].any(-1)].tolist(),failed_joint_counts=fail[ids].sum(0).tolist(),
                min_motion_tube_max_joint_score_median=float(np.median(minimum)),
                min_motion_tube_max_joint_score_max=float(np.max(minimum)),
                max_outside_margin_normalized=np.maximum((lo[ids]-truth[ids])/fitter.scale,
                    (truth[ids]-hi[ids])/fitter.scale).max(0).clip(0).tolist()))
    result=dict(created_utc=utc(),posthoc_only=True,parameters_unchanged=True,
        frozen_radii_normalized=radii.tolist(),rows=rows,freeze_sha256=sha(output/'protocol-freeze.json'))
    write_json(output/'diagnostics.json',result);print(result,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=run.ROOT)
    main(parser.parse_args().output)
