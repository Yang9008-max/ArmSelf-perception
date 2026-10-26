"""Learned causal sensor weighting; no simulator or evaluation labels here.

The three softmax weights express contributions to an estimate, not calibrated
probabilities that a sensor is correct. Missing measurements are hard-masked;
all decisions between AVAILABLE sources are learned from observation history.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class ReliabilityGate(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.hidden_size = hidden
        self.cell = nn.GRUCell(60, hidden)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.SiLU(), nn.Linear(32, 3))

    def forward(self, prior, feedback, visual, previous, old_feedback, old_visual,
                scale, quality, available, duration, weights, hidden):
        df = (feedback-prior)/scale
        dv = (visual-prior)/scale
        motion = (prior-previous)/scale
        tf = (feedback-old_feedback)/scale-motion
        tv = (visual-old_visual)/scale-motion
        features = torch.cat((df, dv, df.abs(), dv.abs(), tf, tv, motion,
                              quality, available.float(), duration[:, None]/.1, weights), -1)
        hidden = self.cell(features.clamp(-10, 10), hidden)
        logits = self.head(hidden)
        mask = torch.cat((torch.ones_like(available[:, :1]), available), -1).bool()
        weights = logits.masked_fill(~mask, -torch.inf).softmax(-1)
        q = prior+weights[:, 1:2]*(feedback-prior)+weights[:, 2:3]*(visual-prior)
        return q, weights, hidden

    @classmethod
    def load(cls, path):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        result = cls(saved['hidden'])
        result.load_state_dict(saved['state_dict'])
        return result.eval()


class LearnedBodyState:
    """Predict with action, then update ONE state using learned contributions."""
    def __init__(self, transition, gate, body, initial_q):
        self.transition, self.gate, self.body = transition, gate, body
        self.scale = torch.as_tensor(np.asarray(body.saved['scale']), dtype=torch.float32)[None]
        self.q = np.asarray(initial_q, np.float32).copy()
        self.velocity = np.zeros(7, np.float32)
        self.old_feedback = self.q.copy()
        self.old_visual = self.q.copy()
        self.weights = torch.tensor([[1., 0., 0.]])
        self.hidden = torch.zeros(1, gate.hidden_size)
        self.pending = None

    def forecast(self, target, duration):
        if self.pending is not None:
            raise RuntimeError('Finish the observation update before another forecast')
        self.pending = (self.transition.predict(self.q, self.velocity, target, duration), float(duration))
        return self.body(self.pending[0])[0]

    def finish(self, feedback=None, visual=None, quality=None, visual_available=False):
        if self.pending is None:
            raise RuntimeError('Forecast must precede observation update')
        prior, duration = self.pending
        f_ok = feedback is not None and np.isfinite(feedback).all()
        v_ok = visual_available and visual is not None and np.isfinite(visual).all()
        f = np.asarray(feedback, np.float32) if f_ok else prior
        v = np.asarray(visual, np.float32) if v_ok else prior
        quality = np.zeros(5, np.float32) if quality is None else np.asarray(quality, np.float32)
        def t(x):
            return torch.as_tensor(np.asarray(x), dtype=torch.float32)[None]
        with torch.inference_mode():
            q, self.weights, self.hidden = self.gate(t(prior), t(f), t(v), t(self.q),
                t(self.old_feedback), t(self.old_visual), self.scale, t(quality),
                torch.tensor([[f_ok, v_ok]]), torch.tensor([duration]), self.weights, self.hidden)
        q = q[0].numpy().copy()
        self.velocity = np.clip((q-self.q)/duration, -self.scale[0].numpy()*20, self.scale[0].numpy()*20)
        self.q = q
        self.old_feedback, self.old_visual = f.copy(), v.copy()
        self.pending = None
        return dict(q=q.copy(), xyz=self.body(q)[0], weights=self.weights[0].numpy().copy())
