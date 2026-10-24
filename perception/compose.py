"""Compose learned body segmentation with an independently frozen response."""
from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path

import torch

from .model import BodyPerception


def compose_checkpoints(body_checkpoint, motion_checkpoint, output, support_checkpoint=None):
    body_path, motion_path, output = map(Path, (body_checkpoint, motion_checkpoint, output))
    if output.exists():
        raise FileExistsError(f'Preserving existing checkpoint: {output}')
    body = torch.load(body_path, map_location='cpu', weights_only=True)
    motion = torch.load(motion_path, map_location='cpu', weights_only=True)
    if body['model_config'] != motion['model_config']:
        raise ValueError('Body and motion model configurations must match')
    if body['model_config'].get('separate_motion', False) or body['model_config'].get('body_support', False):
        raise ValueError('Compose original single-branch checkpoints, not an existing composition')
    if not body.get('preprocessing') or body['preprocessing'] != motion.get('preprocessing'):
        raise ValueError('Body and motion preprocessing must match, including feedback units')
    config = {**body['model_config'], 'separate_motion': True}
    support = None
    if support_checkpoint:
        support_path = Path(support_checkpoint)
        support = torch.load(support_path, map_location='cpu', weights_only=True)
        if support['model_config'] != body['model_config'] or support.get('preprocessing') != body['preprocessing']:
            raise ValueError('Body support must share the original configuration and preprocessing')
        config['body_support'] = True
    model = BodyPerception(**config)
    combined_state = {**body['state_dict'],
                      **{f'motion_branch.{key}': value for key, value in motion['state_dict'].items()}}
    if support is not None:
        combined_state.update({f'support_branch.{key}': value for key, value in support['state_dict'].items()})
    model.load_state_dict(combined_state, strict=True)
    record = copy.deepcopy(body.get('training', {}))
    record['composition'] = {
        'method': 'Body branch from segmentation training; frozen response branch from motion training',
        'body_checkpoint': str(body_path),
        'body_sha256': hashlib.sha256(body_path.read_bytes()).hexdigest(),
        'motion_checkpoint': str(motion_path),
        'motion_sha256': hashlib.sha256(motion_path.read_bytes()).hexdigest(),
        'response_training_after_composition': False,
        'new_optimization_or_threshold_selection': False,
    }
    if support is not None:
        record['composition']['body_support'] = {
            'checkpoint': str(support_path),
            'sha256': hashlib.sha256(support_path.read_bytes()).hexdigest(),
            'rule': 'Minimum body logit: visible mask is an intersection at the common threshold',
            'selection': 'Rule chosen using development failures; no fitting or final-test threshold search',
            'limitation': 'Cannot recover body pixels missed by the frozen support model'}
    record['model_config'] = config
    record['parameters'] = sum(p.numel() for p in model.parameters())
    record['frozen_motion_parameters'] = sum(p.numel() for p in model.motion_branch.parameters())
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_config': config, 'state_dict': model.state_dict(),
                'preprocessing': body['preprocessing'], 'training': record}, output)
    print(f'Saved composed checkpoint: {output}', flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--body-checkpoint', required=True)
    parser.add_argument('--motion-checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--support-checkpoint')
    compose_checkpoints(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
