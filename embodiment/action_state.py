"""Learned action transition and explicit updates of a shared body state.

Learner inputs are joint readings, their past differences, command targets and
duration. The existing learned body decoder is shared by every state estimate.
No simulator state, geometry or future observation is available in this module.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .recovery import VisualFeedbackCorrection


class ActionTransition(nn.Module):
    def __init__(self, center, scale, use_action=True):
        super().__init__()
        self.register_buffer('center', torch.as_tensor(center, dtype=torch.float32))
        self.register_buffer('scale', torch.as_tensor(scale, dtype=torch.float32))
        self.use_action = use_action
        self.network = nn.Sequential(nn.Linear(22 if use_action else 15, 96), nn.SiLU(),
                                     nn.Linear(96, 96), nn.SiLU(), nn.Linear(96, 7))

    def forward(self, q, velocity, target, duration):
        duration = duration.reshape(-1, 1)
        values = [(q-self.center)/self.scale, velocity/self.scale*.05, duration/.1]
        if self.use_action:
            values.append((target-q)/self.scale)
        delta = self.network(torch.cat(values, dim=-1))
        return q + delta*self.scale

    def predict(self, q, velocity, target, duration):
        with torch.inference_mode():
            tensors = [torch.as_tensor(np.atleast_2d(v), dtype=torch.float32) for v in (q, velocity, target)]
            return self(*tensors, torch.as_tensor([duration], dtype=torch.float32))[0].numpy()

    @classmethod
    def load(cls, path):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        result = cls(saved['center'], saved['scale'], saved['use_action'])
        result.load_state_dict(saved['state_dict'])
        return result.eval()


class ActionBodyState:
    """Explicit filter around learned transition and body networks.

The state uses measured joint coordinates, not a learned latent coordinate
system. Bias correction needs visual evidence AND an independent discrepancy
between action prediction and feedback. These are heuristic admission rules.
"""
    def __init__(self, transition, body, initial_q, feedback_gain=.9,
                 innovation_gate_m=.012, vision_enabled=True, feedback_enabled=True):
        self.transition, self.body = transition, body
        self.scale = np.asarray(body.saved['scale'])
        self.q = np.asarray(initial_q, float).copy()
        self.velocity = np.zeros(7)
        self.bias = np.zeros(7)
        self.feedback_gain = feedback_gain
        self.gate = innovation_gate_m
        self.vision_enabled, self.feedback_enabled = vision_enabled, feedback_enabled
        self.pending = None

    def forecast(self, target, duration):
        if self.pending is not None:
            raise RuntimeError('Finish the current observation update before another action')
        prior = self.transition.predict(self.q, self.velocity, target, duration)
        self.pending = dict(previous=self.q.copy(), prior=prior.copy(), duration=float(duration))
        return dict(q=prior.copy(), xyz=self.body(prior)[0].copy())

    def feedback(self, reading):
        if self.pending is None:
            raise RuntimeError('Forecast must precede feedback')
        prior = self.pending['prior']
        if self.feedback_enabled and reading is not None:
            measured = np.asarray(reading)-self.bias
            q = prior+self.feedback_gain*(measured-prior)
            discrepancy = np.sqrt(np.mean(np.sum((self.body(measured)[0]-self.body(prior)[0])**2, axis=-1)))
        else:
            q, discrepancy = prior.copy(), 0.
        self.pending.update(feedback=q.copy(), innovation=float(discrepancy), reading=None if reading is None else np.asarray(reading).copy())
        return q.copy()

    def finish(self, xyz=None, valid=None, time=0.):
        if self.pending is None or 'feedback' not in self.pending:
            raise RuntimeError('Feedback stage must precede visual update')
        p = self.pending
        q = p['feedback'].copy()
        accepted, support, reason = False, 0, 'vision_missing'
        if self.vision_enabled and xyz is not None and valid is not None:
            support = int(np.asarray(valid, bool).sum())
            reason = 'insufficient_visual_support'
            eligible = not self.feedback_enabled or p['innovation'] > self.gate
            if support >= 8 and eligible:
                corrector = VisualFeedbackCorrection(self.body, self.scale)
                result = corrector.update(q, xyz, valid, time)
                candidate = corrector.corrected_q(q)
                before = self.body(q)[0]
                after = self.body(candidate)[0]
                action_prior = self.body(p['prior'])[0]
                old_distance = np.mean(np.sum((before-action_prior)**2, axis=-1))
                new_distance = np.mean(np.sum((after-action_prior)**2, axis=-1))
                change = np.sqrt(np.mean(np.sum((after-before)**2, axis=-1)))
                # Vision must support reducing the independent action/feedback
                # disagreement. For the no-feedback ablation only the finite
                # correction bound is used and its weaker protection is reported.
                consistent = not self.feedback_enabled or new_distance < old_distance*.75
                accepted = bool(result['updated'] and consistent and .002 < change < .04)
                reason = 'accepted' if accepted else 'inconsistent_with_action_prior'
                if accepted:
                    if self.feedback_enabled:
                        self.bias += (q-candidate)*.5
                    q = candidate
            elif support >= 8:
                reason = 'action_and_feedback_already_agree'
        self.velocity = np.clip((q-p['previous'])/p['duration'], -self.scale*20, self.scale*20)
        self.q = q.copy()
        self.pending = None
        return dict(q=q, xyz=self.body(q)[0], visual_update=accepted,
                    visual_support=support, reason=reason, bias=self.bias.copy(),
                    innovation_m=p['innovation'])
