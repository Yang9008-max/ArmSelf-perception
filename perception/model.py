"""Small RGB/position-conditioned network; no kinematic or segmentation truth."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, incoming: int, outgoing: int):
        super().__init__(
            nn.Conv2d(incoming, outgoing, 3, padding=1),
            nn.GroupNorm(4, outgoing), nn.SiLU(),
            nn.Conv2d(outgoing, outgoing, 3, padding=1),
            nn.GroupNorm(4, outgoing), nn.SiLU(),
        )


class BodyPerception(nn.Module):
    def __init__(self, channels: int = 7, width: int = 8, use_feedback: bool = True,
                 separate_motion: bool = False, body_support: bool = False):
        super().__init__()
        if width < 4 or width % 4:
            raise ValueError('width must be a positive multiple of four')
        self.channels, self.use_feedback = channels, use_feedback
        self.enc1 = ConvBlock(3, width)
        self.enc2 = ConvBlock(width, width * 2)
        self.enc3 = ConvBlock(width * 2, width * 4)
        self.middle = ConvBlock(width * 4, width * 8)
        self.feedback = nn.Sequential(nn.Linear(channels, 32), nn.SiLU(),
                                      nn.Linear(32, width * 16))
        self.dec3 = ConvBlock(width * 12, width * 4)
        self.dec2 = ConvBlock(width * 6, width * 2)
        self.dec1 = ConvBlock(width * 3, width)
        self.motion_head = nn.Conv2d(width, channels * 2, 1)
        self.body_head = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.motion_head.weight)
        nn.init.zeros_(self.motion_head.bias)
        self.motion_branch = None
        if separate_motion:
            self.motion_branch = BodyPerception(channels, width, use_feedback)
            self.motion_branch.requires_grad_(False)
        self.support_branch = None
        if body_support:
            self.support_branch = BodyPerception(channels, width, use_feedback)
            self.support_branch.requires_grad_(False)

    @staticmethod
    def up(value, skip):
        return torch.cat([F.interpolate(value, size=skip.shape[-2:], mode='bilinear',
                                        align_corners=False), skip], dim=1)

    def forward(self, rgb, q):
        if q.shape[-1] != self.channels:
            raise ValueError('Feedback channel count does not match model')
        a = self.enc1(rgb)
        b = self.enc2(F.avg_pool2d(a, 2))
        c = self.enc3(F.avg_pool2d(b, 2))
        hidden = self.middle(F.avg_pool2d(c, 2))
        if self.use_feedback:
            scale, shift = self.feedback(q).chunk(2, dim=1)
            hidden = hidden * (1 + .1 * scale[:, :, None, None]) + .1 * shift[:, :, None, None]
        hidden = self.dec3(self.up(hidden, c))
        hidden = self.dec2(self.up(hidden, b))
        features = self.dec1(self.up(hidden, a))
        jacobian = self.motion_head(features).reshape(
            rgb.shape[0], self.channels, 2, *rgb.shape[-2:])
        if self.motion_branch is not None:
            # Body fitting cannot alter the independently selected response.
            # Keep eval semantics even if the containing model is being trained.
            self.motion_branch.eval()
            with torch.no_grad():
                jacobian = self.motion_branch(rgb, q)['jacobian']
        logits = self.body_head(features)
        if self.support_branch is not None:
            self.support_branch.eval()
            with torch.no_grad():
                support = self.support_branch(rgb, q)['logits']
            # A conservative intersection, not a calibrated new probability.
            # Thresholding cannot introduce body pixels absent from the reference.
            logits = torch.minimum(logits, support)
        return {'jacobian': jacobian, 'logits': logits}


def predict_flow(jacobian, dq):
    """Contract B,C,2,H,W with measured increments B,K,C, in probe units."""
    return torch.einsum('bcuhw,bkc->bkuhw', jacobian, dq)


def motion_loss(jacobian, dq, observed, valid):
    prediction = predict_flow(jacobian, dq)
    error = F.smooth_l1_loss(prediction, observed, reduction='none').mean(2)
    moving = observed.square().sum(2).sqrt() > .35
    terms = []
    for region in (valid & moving, valid & ~moving):
        if region.any():
            terms.append(error[region].mean())
    return sum(terms) / len(terms) if terms else error.sum() * 0


def partial_segmentation_loss(logits, labels):
    """Balance trusted positives/negatives; unknown -1 has exactly zero gradient."""
    logits = logits[:, 0]
    terms = []
    for value in (0., 1.):
        selected = labels == value
        if selected.any():
            terms.append(F.binary_cross_entropy_with_logits(
                logits[selected], torch.full_like(logits[selected], value)))
    return sum(terms) / len(terms) if terms else logits.sum() * 0


def stationary_negative_loss(logits, evidence):
    """Balance independent-motion negatives separately from abundant background."""
    selected = logits[:, 0][evidence.bool()]
    return (F.binary_cross_entropy_with_logits(selected, torch.zeros_like(selected))
            if selected.numel() else logits.sum() * 0)
