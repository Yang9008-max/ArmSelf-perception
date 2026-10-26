"""Post-seal E12 set failures; diagnostic only, never change frozen margins."""
import numpy as np
from . import branch_cycle as run,trust_cycle as e7
from .body_cycle import read_json,write_json,utc,sha


def main(output=run.ROOT):
    frozen=run.verify(output)
    assert (output/'predictions-complete.json').exists() and (output/'scores.json').exists()
    margin=np.array(read_json(output/'development-selection.json')['profiles'][frozen['selected']]['margin_normalized'])
    _,_,fitter=run.components(output)
    rows=[]
    for seed in run.FINAL:
        p=e7.load_npz(output/'predictions'/f'{seed}-twin_moving.npz')
        q=[]
        for c in ('clean','execution_hold'):
            source=run.source_for(output,seed,c)
            q.append(np.stack([e7.load_npz(source/'evaluation'/r['file'])['actual_q'] for r in read_json(source/'metadata.json')['frames']]))
        gap=np.array([fitter.distance(a,b) for a,b in zip(*q)])
        ids=np.flatnonzero((np.arange(180)>=30)&(np.arange(180)<70)&(gap>.015))
        lower=np.where(p['bank_valid'][:,:,None],p['bank'],np.inf).min(1)-margin*fitter.scale
        upper=np.where(p['bank_valid'][:,:,None],p['bank'],-np.inf).max(1)+margin*fitter.scale
        for name,truth in zip(('moving','held'),q):
            fail=(truth<lower)|(truth>upper)
            rows.append(dict(seed=seed,physical=name,frames=len(ids),covered=int((~fail[ids].any(-1)).sum()),
                failed_joint_counts=fail[ids].sum(0).tolist(),
                max_outside_margin_normalized=np.maximum((lower[ids]-truth[ids])/fitter.scale,
                    (truth[ids]-upper[ids])/fitter.scale).max(0).clip(0).tolist()))
    result=dict(created_utc=utc(),posthoc_only=True,parameters_unchanged=True,
        frozen_margin=margin.tolist(),rows=rows,freeze_sha256=sha(output/'protocol-freeze.json'))
    write_json(output/'diagnostics.json',result)
    print(result,flush=True)


if __name__=='__main__':
    main()
