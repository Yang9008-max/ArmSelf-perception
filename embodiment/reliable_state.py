"""E8 causal per-joint fusion with an optional learned encoder-bias state.

Only observation candidates enter this module. Training labels, fault families,
simulator geometry and future observations are deliberately outside its API.
Weights are contributions, not sensor-correctness probabilities. Standard
deviations concern joint-state error; they exclude body-decoder geometry error.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class JointReliability(nn.Module):
    def __init__(self, hidden=64, bias_enabled=True, bias_bound=.8):
        super().__init__()
        self.hidden_size = hidden
        self.bias_enabled = bias_enabled
        self.bias_bound = bias_bound
        self.cell = nn.GRUCell(99, hidden)
        self.weight_head = nn.Sequential(nn.Linear(hidden, 64), nn.SiLU(), nn.Linear(64, 21))
        self.bias_head = nn.Linear(hidden, 14)
        self.sigma_head = nn.Sequential(nn.Linear(hidden, 32), nn.SiLU(), nn.Linear(32, 7))
        with torch.no_grad():
            self.weight_head[-1].weight.zero_()
            self.weight_head[-1].bias.copy_(torch.tensor([0., 2., -2.]).repeat(7))
            self.bias_head.weight.zero_()
            self.bias_head.bias.zero_()
            self.bias_head.bias[7:] = -1.
            self.sigma_head[-1].bias.fill_(-3.)

    def forward(self, prior, feedback, visual, previous, old_feedback, old_visual,
                scale, quality, available, duration, weights, bias, hidden):
        available = available.bool()
        # Remove unavailable VALUES and quality before recurrent feature creation.
        feedback = torch.where(available[:, :1], feedback, prior)
        visual = torch.where(available[:, 1:2], visual, prior)
        quality = torch.where(available[:, 1:2], quality, torch.zeros_like(quality))
        df, dv = (feedback-prior)/scale, (visual-prior)/scale
        pair = df-dv
        motion = (prior-previous)/scale
        tf = torch.where(available[:, :1], (feedback-old_feedback)/scale-motion, 0.)
        tv = torch.where(available[:, 1:2], (visual-old_visual)/scale-motion, 0.)
        features = torch.cat((df, dv, pair, df.abs(), dv.abs(), pair.abs(), tf, tv,
                              motion, bias, quality, available.float(),
                              duration[:, None]/.1, weights.flatten(1)), -1)
        hidden = self.cell(features.clamp(-10, 10), hidden)
        if self.bias_enabled:
            proposal, rate = self.bias_head(hidden).chunk(2, -1)
            proposal = proposal.tanh()*self.bias_bound
            updated_bias = bias+rate.sigmoid()*(proposal-bias)
            bias = torch.where(available[:, :1], updated_bias, bias)
        else:
            bias = torch.zeros_like(bias)
        corrected = feedback-bias*scale
        candidates = torch.stack((prior, corrected, visual), -1)
        mask = torch.cat((torch.ones_like(available[:, :1]), available), -1)
        logits = self.weight_head(hidden).reshape(-1, 7, 3)
        weights = logits.masked_fill(~mask[:, None], -torch.inf).softmax(-1)
        q = (weights*candidates).sum(-1)
        # Uncertainty fitting must not change the selected state through a
        # shortcut in which the estimator enlarges its own errors or variances.
        sigma = self.sigma_head(hidden.detach()).clamp(-7., 1.).exp()
        return q, weights, bias, sigma, hidden

    @classmethod
    def load(cls, path):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        result = cls(saved['hidden'], saved['bias_enabled'], saved['bias_bound'])
        result.load_state_dict(saved['state_dict'])
        return result.eval()


class ReliableBodyState:
    def __init__(self, transition, gate, body, initial_q):
        self.transition, self.gate, self.body = transition, gate, body
        self.scale = torch.as_tensor(np.asarray(body.saved['scale']), dtype=torch.float32)[None]
        self.q = np.asarray(initial_q, np.float32).copy()
        self.velocity = np.zeros(7, np.float32)
        self.old_feedback, self.old_visual = self.q.copy(), self.q.copy()
        self.weights = torch.tensor([1., 0., 0.]).repeat(1, 7, 1)
        self.bias = torch.zeros(1, 7)
        self.hidden = torch.zeros(1, gate.hidden_size)
        self.pending = None

    def forecast(self, target, duration):
        if self.pending is not None:
            raise RuntimeError('Finish observation update before the next forecast')
        if duration <= 0 or not np.isfinite(duration):
            raise ValueError('Duration must be positive and finite')
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
        quality = np.zeros(5, np.float32) if quality is None or not v_ok else np.asarray(quality, np.float32)
        def tensor(value):
            return torch.as_tensor(np.asarray(value), dtype=torch.float32)[None]
        with torch.inference_mode():
            q, self.weights, self.bias, sigma, self.hidden = self.gate(
                tensor(prior), tensor(f), tensor(v), tensor(self.q),
                tensor(self.old_feedback), tensor(self.old_visual), self.scale,
                tensor(quality), torch.tensor([[f_ok, v_ok]]), torch.tensor([duration]),
                self.weights, self.bias, self.hidden)
        q = q[0].numpy().copy()
        self.velocity = np.clip((q-self.q)/duration, -self.scale[0].numpy()*20, self.scale[0].numpy()*20)
        self.q, self.old_feedback, self.old_visual = q, f.copy(), v.copy()
        self.pending = None
        return dict(q=q.copy(), xyz=self.body(q)[0], weights=self.weights[0].numpy().copy(),
                    bias_normalized=self.bias[0].numpy().copy(),
                    sigma_normalized=sigma[0].numpy().copy())
