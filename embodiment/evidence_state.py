"""E13: branch-specific error tubes and visual selection without good feedback.

Spatial patch checks share one image source; they are not independent votes.
Visual-only selection changes the point estimate but retains source ambiguity.
"""
from __future__ import annotations
import cv2
import numpy as np
import torch
from .branch_state import BranchBodyState
from .surface_extent import track_patch
from .vision import project


class MotionMeasurements:
    """One measurement per patch, with original and augmented seed budgets."""
    def __init__(self,atlas,indices):
        self.atlas=atlas;self.indices=np.asarray(indices,int)

    def observe(self,left,right,seeds):
        n=len(self.indices);xyz=np.zeros((n,3),np.float32)
        best=np.full(n,np.inf);origin=np.zeros(n,np.uint8)
        def snapshot():
            return dict(xyz=xyz.copy(),valid=np.isfinite(best),origin=origin.copy())
        if left is None or right is None:
            return snapshot(),snapshot()
        gray=cv2.cvtColor(left,cv2.COLOR_RGB2GRAY);other=cv2.cvtColor(right,cv2.COLOR_RGB2GRAY)
        saved=self.atlas.saved;original=None
        for k,q in enumerate(seeds):
            if q is not None and np.isfinite(q).all():
                predicted=self.atlas.predict(q)[self.indices]
                initial=project(predicted,self.atlas.calibration)
                for chart in np.unique(saved['chart'][self.indices]):
                    positions=np.flatnonzero(saved['chart'][self.indices]==chart);ids=self.indices[positions]
                    measured,good,uv=track_patch(saved['gray'][chart],gray,other,saved['uv'][ids],
                        predicted[positions],self.atlas.calibration,initial[positions])
                    good&=np.linalg.norm(measured-predicted[positions],axis=-1)<.05
                    for j in np.flatnonzero(good):
                        old=cv2.getRectSubPix(saved['gray'][chart],(7,7),tuple(map(float,saved['uv'][ids[j]])))
                        new=cv2.getRectSubPix(gray,(7,7),tuple(map(float,uv[j])))
                        cost=float(np.abs(old.astype(float)-new).mean());index=positions[j]
                        if cost<best[index]:
                            best[index],xyz[index],origin[index]=cost,measured[j],1<<k
            if k==1:
                original=snapshot()
        return original if original is not None else snapshot(),snapshot()


def branch_envelope(bank,active,scale,radii):
    radii=np.asarray(radii)*np.asarray(scale)
    lower=np.where(np.asarray(active)[...,None],bank-radii,np.inf).min(-2)
    upper=np.where(np.asarray(active)[...,None],bank+radii,-np.inf).max(-2)
    return lower,upper


class EvidenceBodyState(BranchBodyState):
    def __init__(self,*args,visual_confirm=3,visual_gain=0.,visual_enabled=True,**kwargs):
        super().__init__(*args,**kwargs)
        self.visual_confirm=visual_confirm;self.visual_gain=visual_gain;self.visual_enabled=visual_enabled
        self.visual_streak=0;self.visual_id=-1;self.visual_last=-1

    def visual_choice(self,branches,feedback,xyz,valid):
        good=np.asarray(valid,bool)&np.isfinite(xyz).all(-1)
        halves=[good&(np.arange(len(good))%2==k) for k in (0,1)]
        # At least six separate patches in EACH spatial subset. Use a median
        # directly because the legacy residual requires 12 in one subset.
        if any(int(h.sum())<6 for h in halves):
            return -1,np.full((2,4),1.,np.float32)
        distances=np.array([np.linalg.norm(self.fitter.values(q)-xyz,axis=-1) for q in branches])
        residual=np.array([[np.median(d[h]) for d in distances] for h in halves])
        winners=residual.argmin(-1);candidate=int(np.argmin(residual.mean(0)))
        # Similar physical branches describe one spatial explanation.
        separation=np.array([self.fitter.distance(branches[candidate],q) for q in branches])
        if not all(separation[w]<.012 for w in winners):
            return -1,residual.astype(np.float32)
        rivals=separation>.015
        if not rivals.any() or (residual[:,candidate]>self.thresholds['surface_m']).any():
            return -1,residual.astype(np.float32)
        if (residual[:,rivals].min(-1)-residual[:,candidate]<.003).any():
            return -1,residual.astype(np.float32)
        # A replayed RGB/feedback pair that explains the image better than
        # physical histories must not make a physical history look verified.
        if feedback is not None and np.isfinite(feedback).all():
            fr=self.fitter.residual(feedback,xyz,valid)
            if fr+.001<float(residual[:,candidate].mean()):
                return -1,residual.astype(np.float32)
        return candidate,residual.astype(np.float32)

    def finish(self,feedback,visual,quality,visual_available,xyz,valid):
        if self.pending is None:
            raise RuntimeError('Forecast must precede evidence')
        p=self.pending
        branches=np.vstack((p['action'],p['local'],p['holds'])).copy()
        old_observed=self.observed.copy();age=self.observed_age
        candidate,residual=self.visual_choice(branches,feedback,xyz,valid)
        result=super().finish(feedback,visual,quality,visual_available,xyz,valid)
        selected=False;fit_used=False
        if result['sensor_credible']:
            self.visual_streak=0;self.visual_last=-1;self.visual_id=-1
        else:
            same=candidate>=0 and self.visual_last>=0 and self.fitter.distance(
                branches[candidate],branches[self.visual_last])<.012
            self.visual_streak=(self.visual_streak+1 if same else 1) if candidate>=0 else 0
            self.visual_last=candidate
            selected=bool(self.visual_enabled and candidate>=0 and self.visual_streak>=self.visual_confirm)
            if selected:
                self.visual_id=candidate
            if self.visual_enabled and self.visual_id>=0:
                q=branches[self.visual_id].copy()
                if selected and self.visual_gain:
                    fitted,ok=self.fitter.fit(q,xyz,valid)
                    if ok:
                        q+=self.visual_gain*(fitted-q);fit_used=True
                self.q=q.astype(np.float32)
                self.local=self.q.copy()
                # Propagation through an image outage is not a fresh observation.
                if selected:
                    self.observed=self.q.copy()
                    self.observed_v=self._velocity(self.q,old_observed,max(age,p['dt']))
                    self.observed_age=0.
                self.local_v=(self.observed_v.copy() if selected else
                    self._velocity(self.q,p['previous'],p['dt']))
                self.sensor_mode=bool(self.visual_id>=2)
                if self.sensor_mode:
                    self.holds[:]=self.q
                    if selected:
                        self.hold_v[:]=self.observed_v
                self.base.q=self.q.copy();self.base.velocity=self._velocity(self.q,p['previous'],p['dt'])
                self.base.hidden=torch.zeros(1,self.base.gate.hidden_size)
                self.base.old_feedback=feedback.copy() if feedback is not None else self.q.copy()
                self.base.old_visual=np.asarray(visual,np.float32).copy() if visual_available else self.q.copy()
                # One possibly faulty RGB source cannot remove source ambiguity.
                self.unresolved=True
                result.update(q=self.q.copy(),unknown=np.bool_(True),
                    committed=np.bool_(np.max(np.abs(self.q-result['proposal']))>1e-7),
                    bank_valid=np.array([True,True,True,True,
                        feedback is not None and np.isfinite(feedback).all(),True]),
                    sensor_mode=np.bool_(self.sensor_mode))
                result['bank'][-1]=self.q
        result.update(visual_selected=np.bool_(selected),visual_history=np.int32(self.visual_id),
            visual_candidate=np.int32(candidate),visual_checks=residual,visual_fitted=np.bool_(fit_used))
        return result
