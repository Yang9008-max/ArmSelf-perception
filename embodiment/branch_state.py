"""E12: causal motion branches, one RGB source, and defeasible action priors.

The branch compatibility values are explicit residuals, not calibrated fault
probabilities. Neither condition names nor simulator truth enter this API.
"""
from __future__ import annotations
import numpy as np
import torch
from .reliable_state import ReliableBodyState


class BranchBodyState:
    def __init__(self, transition, gate, body, fitter, initial, thresholds,
                 confirm=3, association_factor=1., allow_override=True):
        self.base=ReliableBodyState(transition,gate,body,initial)
        self.transition,self.fitter=transition,fitter
        self.scale=np.asarray(body.saved['scale'],np.float32)
        self.thresholds=thresholds
        self.confirm,self.factor,self.allow_override=confirm,association_factor,allow_override
        self.q=np.asarray(initial,np.float32).copy()
        self.action=self.q.copy();self.action_v=np.zeros(7,np.float32)
        self.local=self.q.copy();self.local_v=self.action_v.copy()
        self.observed=self.q.copy();self.observed_v=self.action_v.copy()
        self.holds=np.stack((self.q,self.q));self.hold_v=np.zeros((2,7),np.float32)
        self.held_target=self.q.copy();self.last_target=self.q.copy()
        self.unresolved=False;self.sensor_mode=False;self.stable=0;self.pending=None
        self.observed_age=0.

    def _velocity(self,new,old,dt):
        return np.clip((new-old)/dt,-self.scale*20,self.scale*20)

    def forecast(self,target,duration):
        if self.pending is not None:
            raise RuntimeError('Finish evidence before next command')
        if duration<=0 or not np.isfinite(duration):
            raise ValueError('Duration must be finite and positive')
        target=np.asarray(target,np.float32);dt=float(duration)
        self.observed_age+=dt
        self.base.forecast(target,dt)
        action=self.transition.predict(self.action,self.action_v,target,dt)
        local=self.transition.predict(self.local,self.local_v,target,dt)
        if not self.unresolved and not self.sensor_mode:
            self.holds[:]=self.observed
            self.hold_v[:]=self.observed_v
            self.held_target=self.last_target.copy()
        servo=self.transition.predict(self.holds[0],self.hold_v[0],self.held_target,dt)
        coast=(self.holds[1]+.5*self.hold_v[1]*dt).astype(np.float32)
        prior=coast if self.sensor_mode else local
        self.pending=dict(action=action,local=local,holds=np.stack((servo,coast)),
            previous=self.q.copy(),target=target.copy(),dt=dt,prior=prior.copy())
        return prior.copy()

    def finish(self,feedback,visual,quality,visual_available,xyz,valid):
        if self.pending is None:
            raise RuntimeError('Forecast must precede evidence')
        p=self.pending;dt=p['dt']
        f_ok=feedback is not None and np.isfinite(feedback).all()
        f=np.asarray(feedback,np.float32) if f_ok else p['local']
        proposal=self.base.finish(feedback,visual,quality,visual_available)
        branches=np.vstack((p['action'],p['local'],p['holds']))
        support=int(np.asarray(valid).sum())
        has_rgb=support>=12
        rgb=np.array([self.fitter.residual(q,xyz,valid) for q in branches])
        innovation=np.array([self.fitter.distance(q,f) for q in branches]) if f_ok else np.full(4,np.inf)
        joint=np.max(np.abs((branches-f)/self.scale),axis=-1) if f_ok else np.full(4,np.inf)
        compatible=(innovation<=self.thresholds['association_m']*self.factor)&(joint<=self.thresholds['association_q']*self.factor)
        fr=self.fitter.residual(f,xyz,valid) if f_ok else float('inf')
        surface_ok=bool(has_rgb and fr<=self.thresholds['surface_m'])
        associated=bool(f_ok and compatible.any())
        credible=bool(associated and surface_ok)
        action_disagrees=bool(not f_ok or not compatible[0])
        conflict=bool(not f_ok or not associated or action_disagrees or (has_rgb and not surface_ok))
        self.unresolved=bool(self.unresolved or conflict)
        self.stable=self.stable+1 if credible else 0
        accepted=bool(credible and self.stable>=self.confirm)
        best=int(np.argmin(innovation)) if f_ok else 0
        # A sensor jump cannot create its own confirming physical branch.
        # Association is with states predicted BEFORE the current readings.
        selected=bool(accepted and self.allow_override)
        q=proposal['q'].copy()
        if selected:
            if action_disagrees or self.sensor_mode or self.fitter.distance(q,f)>.002:
                q=f.copy()
            self.sensor_mode=bool(action_disagrees)
            self.unresolved=False
        elif not associated:
            q=(p['holds'][1] if self.sensor_mode else p['action']).copy()
        elif self.sensor_mode and self.allow_override:
            # Previously established execution failure survives an RGB outage.
            # This is a propagated estimate; no fresh visual evidence is claimed.
            q=f.copy() if f_ok and compatible[2:].any() else p['holds'][1].copy()
        if not self.allow_override and action_disagrees:
            q=proposal['q'].copy()
        committed=bool(selected and np.max(np.abs(q-proposal['q']))>1e-7)
        old_action=self.action.copy()
        self.action=p['action'].copy();self.action_v=self._velocity(self.action,old_action,dt)
        old_local=self.local.copy()
        self.local=f.copy() if credible else p['local'].copy()
        self.local_v=self._velocity(self.local,old_local,dt)
        self.hold_v[0]=self._velocity(p['holds'][0],self.holds[0],dt)
        self.hold_v[1]*=.5
        self.holds=p['holds'].copy()
        if credible:
            old_observed=self.observed.copy()
            self.observed=f.copy();self.observed_v=self._velocity(f,old_observed,self.observed_age)
            self.observed_age=0.
            self.local_v=self.observed_v.copy()
            if selected and self.sensor_mode:
                # Both stop branches assimilate the chosen observation, while
                # the open-loop command branch keeps its competing history.
                self.holds[:]=f
                self.hold_v[:]=self.observed_v
        self.q=q.astype(np.float32)
        self.base.q=self.q.copy();self.base.velocity=self._velocity(self.q,p['previous'],dt)
        if committed:
            self.base.hidden=torch.zeros(1,self.base.gate.hidden_size)
            self.base.old_feedback=f.copy()
            self.base.old_visual=np.asarray(visual,np.float32).copy() if visual_available else self.q.copy()
        self.last_target=p['target'].copy()
        bank=np.vstack((branches,f,self.q)).astype(np.float32)
        bank_valid=np.array([True,True,True,True,f_ok,True]) if self.unresolved else np.array([False]*5+[True])
        self.pending=None
        return dict(q=self.q.copy(),bank=bank,bank_valid=bank_valid,unknown=np.bool_(self.unresolved),
            conflict=np.bool_(conflict),committed=np.bool_(committed),surface_corrected=np.bool_(False),
            proposal=proposal['q'],weights=proposal['weights'],raw_residual=np.float32(fr if np.isfinite(fr) else 1.),
            action_feedback=np.float32(innovation[0] if np.isfinite(innovation[0]) else 1.),
            dynamics=np.float32(innovation[1] if np.isfinite(innovation[1]) else 1.),support=np.int32(support),
            sensor_credible=np.bool_(credible),sensor_selected=np.bool_(selected),sensor_mode=np.bool_(self.sensor_mode),
            override_action=np.bool_(selected and action_disagrees),branch_selected=np.int32(best),
            branch_rgb=np.nan_to_num(rgb,nan=1.,posinf=1.).astype(np.float32),
            branch_innovation=np.nan_to_num(innovation,nan=1.,posinf=1.).astype(np.float32),open_action=p['action'])
