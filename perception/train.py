"""Train dense response and partial body labels without simulator truth."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ObservationDataset, prepare_cache
from .model import BodyPerception, motion_loss, partial_segmentation_loss, stationary_negative_loss


def objective(model, batch, stage, probe_slice, motion_weight=.1, control_weight=0.):
    result = model(batch['rgb'], batch['q'])
    motion = motion_loss(result['jacobian'], batch['dq'][:, probe_slice],
                         batch['flow'][:, probe_slice], batch['valid'][:, probe_slice])
    segment = partial_segmentation_loss(result['logits'], batch['labels'])
    if control_weight:
        segment = segment + control_weight * stationary_negative_loss(result['logits'], batch['control_negative'])
    total = motion if stage == 'motion' else segment + motion_weight * motion
    return total, motion, segment


def train(data, cache, output, motion_epochs=25, segmentation_epochs=35,
          batch_size=4, width=8, threads=4, seed=20261024, use_feedback=True,
          init_checkpoint=None, reuse_cache=None, refine_labels=False,
          use_null_evidence=False, motion_weight=.1, control_weight=0.):
    started = time.perf_counter()
    if min(batch_size, threads) < 1 or min(motion_epochs, segmentation_epochs) < 0:
        raise ValueError('Batch size/threads must be positive; epochs must be nonnegative')
    if motion_epochs + segmentation_epochs == 0:
        raise ValueError('At least one training stage must run')
    if motion_weight < 0 or not np.isfinite(motion_weight):
        raise ValueError('Motion weight must be finite and nonnegative')
    if control_weight < 0 or not np.isfinite(control_weight) or (control_weight > 0 and not use_null_evidence):
        raise ValueError('Positive control weight requires stationary-control evidence')
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    output = Path(output)
    if (output / 'model.pt').exists():
        raise FileExistsError('Use a new run directory; existing model.pt is preserved')
    output.mkdir(parents=True, exist_ok=True)
    meta = json.loads((Path(data) / 'metadata.json').read_text(encoding='utf-8'))
    preprocessing = {key: meta[key] for key in ('q_center', 'q_scale', 'probe_scales')}
    cache_summary = prepare_cache(data, cache, reuse_cache=reuse_cache, refine_labels=refine_labels,
                                  use_null_evidence=use_null_evidence)
    for split in ('train', 'val'):
        if cache_summary[split]['positive'] <= 0 or cache_summary[split]['negative'] <= 0:
            raise RuntimeError(f'{split} lacks trusted positive or negative labels')
    training = ObservationDataset(data, cache, 'train')
    validation = ObservationDataset(data, cache, 'val')
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(training, batch_size=batch_size, shuffle=True,
                        num_workers=0, generator=generator)
    val_loader = DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=0)
    config = {'channels': len(meta['q_center']), 'width': width, 'use_feedback': use_feedback}
    model = BodyPerception(**config)
    initialization = None
    if init_checkpoint:
        previous = torch.load(init_checkpoint, map_location='cpu', weights_only=True)
        if previous['model_config'] != config:
            raise ValueError('Initialization model configuration differs')
        if previous.get('preprocessing') != preprocessing:
            raise ValueError('Initialization preprocessing differs')
        model.load_state_dict(previous['state_dict'])
        initialization = {'file': str(init_checkpoint),
                          'sha256': hashlib.sha256(Path(init_checkpoint).read_bytes()).hexdigest(),
                          'preprocessing_verified': 'preprocessing' in previous}
        history_path = Path(init_checkpoint).parent / 'history.json'
        if history_path.exists():
            initialization['source_history'] = json.loads(history_path.read_text(encoding='utf-8'))
        print(f'Initialized from {init_checkpoint}', flush=True)
    history = []
    best_by_stage = {}
    for stage, epochs, learning_rate in [('motion', motion_epochs, .002),
                                         ('segmentation', segmentation_epochs, .001)]:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(epochs, 1), eta_min=learning_rate*.1)
        best, best_state = float('inf'), None
        for epoch in range(epochs):
            model.train()
            totals = np.zeros(3)
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                # Positive directions fit the predictor; negative directions remain a check.
                losses = objective(model, batch, stage, slice(0, None, 2), motion_weight, control_weight)
                if not torch.isfinite(losses[0]):
                    raise RuntimeError('Nonfinite training objective')
                losses[0].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                totals += [float(loss.detach()) for loss in losses]
            model.eval()
            val_totals = np.zeros(3)
            with torch.inference_mode():
                for batch in val_loader:
                    val_totals += [float(x) for x in objective(model, batch, stage, slice(1, None, 2), motion_weight, control_weight)]
            totals /= len(loader)
            val_totals /= len(val_loader)
            if val_totals[0] < best:
                best = float(val_totals[0])
                best_state = copy.deepcopy(model.state_dict())
            row = {'stage': stage, 'epoch': epoch + 1, 'train_loss': totals.tolist(),
                   'val_loss': val_totals.tolist(), 'elapsed_seconds': time.perf_counter() - started}
            history.append(row)
            (output / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
            print(f'{stage} {epoch+1}/{epochs}: train={totals[0]:.4f} val={val_totals[0]:.4f} '
                  f'({row["elapsed_seconds"]:.1f}s)', flush=True)
            scheduler.step()
        if best_state is not None:
            model.load_state_dict(best_state)
        best_by_stage[stage] = best if np.isfinite(best) else None
        torch.save({'model_config': config, 'state_dict': model.state_dict(),
                    'preprocessing': preprocessing}, output / f'{stage}.pt')
    record = {'seed': seed, 'torch_version': str(torch.__version__), 'device': 'cpu',
              'threads': threads, 'batch_size': batch_size, 'motion_epochs': motion_epochs,
              'segmentation_epochs': segmentation_epochs, 'best_validation': best_by_stage,
              'cache_summary': cache_summary, 'parameters': sum(x.numel() for x in model.parameters()),
              'elapsed_seconds': time.perf_counter() - started,
              'teacher': 'Image-flow local ridge Jacobian: positive probes fitted, negative probes checked',
              'ground_truth_used_for_training': False,
              'selection': 'Observation-only validation objective; no evaluation mask loaded',
              'model_config': config}
    record['initialization'] = initialization
    record['segmentation_motion_weight'] = motion_weight
    record['stationary_control_evidence'] = bool(use_null_evidence)
    record['stationary_control_loss_weight'] = control_weight
    record['background_reference_sha256'] = hashlib.sha256(
        (Path(cache) / 'background-reference.npz').read_bytes()).hexdigest()
    record['negative_label_extension'] = 'Fixed-camera background reference fitted only on training source RGB; same-coordinate probe appearance must remain unchanged.'
    record['boundary_refinement'] = {
        'enabled': bool(refine_labels),
        'method': 'Per-image seeded GrabCut inferred from RGB and existing action/background seeds',
        'seed_labels_retained_in_cache': bool(refine_labels)}
    torch.save({'model_config': config, 'state_dict': model.state_dict(), 'training': record,
                'preprocessing': preprocessing}, output / 'model.pt')
    (output / 'training.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(f'Saved {output / "model.pt"}', flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--motion-epochs', type=int, default=25)
    parser.add_argument('--segmentation-epochs', type=int, default=35)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--width', type=int, default=8)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261024)
    parser.add_argument('--image-only', action='store_true')
    parser.add_argument('--init-checkpoint')
    parser.add_argument('--reuse-cache')
    parser.add_argument('--refine-labels', action='store_true')
    parser.add_argument('--use-null-evidence', action='store_true')
    parser.add_argument('--motion-weight', type=float, default=.1)
    parser.add_argument('--control-weight', type=float, default=0.)
    args = vars(parser.parse_args())
    args['use_feedback'] = not args.pop('image_only')
    train(**args)


if __name__ == '__main__':
    main()
