"""A learned feedback-to-3D entity model, independent of simulator geometry."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn


def basis(q):
    q = np.asarray(q)
    return np.column_stack([np.ones(len(q)), q, q*q,
                            *[q[:, a]*q[:, b] for a in range(q.shape[1]) for b in range(a+1, q.shape[1])]])


def discover(q, xyz, valid, train, val):
    """Select repeatable moving features using train/validation observations only."""
    center = np.median(q[train], axis=0)
    scale = np.maximum(np.ptp(q[train], axis=0)/2, 1e-5)
    normalized = (q-center)/scale
    features = basis(normalized)
    shuffled = np.random.default_rng(1701).permutation(np.flatnonzero(val))
    diagnostics, selected = [], []
    for point in range(xyz.shape[1]):
        fit = train & valid[:, point]
        check = val & valid[:, point]
        row = dict(id=point, train_observations=int(fit.sum()), val_observations=int(check.sum()), selected=False)
        if fit.sum() < max(50, .35*train.sum()) or check.sum() < max(15, .35*val.sum()):
            diagnostics.append(row)
            continue
        x, y = features[fit], xyz[fit, point]
        weight = np.linalg.solve(x.T@x+.08*np.eye(x.shape[1]), x.T@y)
        prediction = features[check]@weight
        observed = xyz[check, point]
        fixed = y.mean(axis=0)
        wrong_features = features[shuffled][valid[val, point]]
        error = float(np.sqrt(np.mean(np.sum((prediction-observed)**2, axis=1))))
        fixed_error = float(np.sqrt(np.mean(np.sum((fixed-observed)**2, axis=1))))
        wrong_error = float(np.sqrt(np.mean(np.sum((wrong_features@weight-observed)**2, axis=1))))
        excursion = float(np.sqrt(np.mean(np.sum((y-fixed)**2, axis=1))))
        keep = (excursion > .006 and error < .6*fixed_error and error < .65*wrong_error and error < .025)
        row.update(response_rms_m=excursion, val_error_m=error, fixed_error_m=fixed_error,
                   shuffled_error_m=wrong_error, selected=bool(keep))
        diagnostics.append(row)
        if keep:
            selected.append(point)
    return np.array(selected, dtype=int), center.astype(np.float32), scale.astype(np.float32), diagnostics


class SpatialEntity(nn.Module):
    def __init__(self, point_count, center, scale, origin):
        super().__init__()
        self.register_buffer('center', torch.as_tensor(center, dtype=torch.float32))
        self.register_buffer('scale', torch.as_tensor(scale, dtype=torch.float32))
        self.register_buffer('origin', torch.as_tensor(origin, dtype=torch.float32))
        self.decoder = nn.Sequential(nn.Linear(7, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(),
                                     nn.Linear(64, point_count*3))
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, measured_q):
        if measured_q.shape[-1] != 7 or not torch.isfinite(measured_q).all():
            raise ValueError('Seven finite measured joint positions required')
        residual = self.decoder((measured_q-self.center)/self.scale)
        return self.origin + .1*residual.reshape(-1, len(self.origin), 3)


def masked_position_loss(prediction, observed, valid):
    error = (prediction-observed).square().sum(-1)
    return error[valid].mean() if valid.any() else prediction.sum()*0


def train_entity(q, xyz, valid, train, val, selected, center, scale, epochs=400, seed=20261210):
    if len(selected) < 4:
        raise ValueError(f'Only {len(selected)} supported features; insufficient evidence for a spatial entity')
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    points = xyz[:, selected].astype(np.float32)
    usable = valid[:, selected]
    origin = np.stack([points[train & usable[:, index], index].mean(0) for index in range(len(selected))])
    model = SpatialEntity(len(selected), center, scale, origin)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    inputs, target, mask = torch.from_numpy(q), torch.from_numpy(points), torch.from_numpy(usable)
    history, best, best_state = [], float('inf'), None
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = masked_position_loss(model(inputs[train]), target[train], mask[train])/.01
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = masked_position_loss(model(inputs[val]), target[val], mask[val])
        score = float(validation)
        history.append(dict(epoch=epoch+1, train_squared_m=float(loss.detach())*.01,
                            validation_squared_m=score))
        if score < best:
            best, best_state = score, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model.eval(), history


@dataclass
class EntityState:
    """Persistent identity plus predicted geometry; no scalar 'existence probability'."""
    model: SpatialEntity | None = None
    tolerance_m: float = .03
    identity: str | None = None
    last_confirmed_time: float | None = None

    def observe(self, measured_q, time, observed_points=None, valid=None):
        if self.model is None:
            return dict(identity=None, status='unknown', points=None, camera_evidence=False)
        if self.identity is None:
            self.identity = 'learned-entity-1'
        with torch.inference_mode():
            points = self.model(torch.as_tensor(measured_q, dtype=torch.float32)[None])[0].numpy()
        normalized = (np.asarray(measured_q)-self.model.center.numpy())/self.model.scale.numpy()
        outside = float(np.maximum(np.abs(normalized)-1, 0).max())
        status, mismatch = 'predicted_unobserved', None
        if observed_points is not None and valid is not None and np.asarray(valid).sum() >= 4:
            mismatch = float(np.median(np.linalg.norm(points[valid]-observed_points[valid], axis=-1)))
            if mismatch < self.tolerance_m:
                status = 'visually_supported'
                self.last_confirmed_time = float(time)
            else:
                status = 'visual_contradiction'
        age = None if self.last_confirmed_time is None else max(0., float(time)-self.last_confirmed_time)
        # This is an engineering uncertainty indicator, not calibrated probability.
        radius = self.tolerance_m*(1+outside) + .002*(age if age is not None else 5)
        return dict(identity=self.identity, status=status, points=points.tolist(),
                    camera_evidence=status == 'visually_supported', mismatch_m=mismatch,
                    seconds_since_visual_support=age, uncertainty_radius_m=radius,
                    extrapolation=outside > 0, measured_q=np.asarray(measured_q).tolist())
