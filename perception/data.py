"""Observation-only loading and RGB/feedback-derived supervision caching."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import flow as flow_module
from . import background as background_module
from . import refine as refine_module
from . import null_evidence as null_module


def prepare_cache(data_dir, cache_dir, splits=('train', 'val', 'test'),
                  background_reference=None, reuse_cache=None, refine_labels=False,
                  use_null_evidence=False):
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    meta = json.loads((data_dir / 'metadata.json').read_text(encoding='utf-8'))
    normalization = {key: meta[key] for key in ('q_center', 'q_scale', 'probe_scales')}
    flow_fingerprint = hashlib.sha256(Path(flow_module.__file__).read_bytes()
                                      + json.dumps(normalization, sort_keys=True).encode()).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    if background_reference is not None:
        with np.load(background_reference, allow_pickle=False) as saved:
            reference = {key: saved[key] for key in ('median', 'support')}
    else:
        training_sources = sorted((data_dir / 'observations' / 'train').glob('*.npz'))
        if not training_sources:
            raise ValueError('Test-only caching requires a frozen training background_reference')
        frames = []
        for path in training_sources:
            with np.load(path, allow_pickle=False) as observation:
                frames.append(observation['rgb'])
        reference = background_module.fit_background_reference(np.stack(frames))
    np.savez_compressed(cache_dir / 'background-reference.npz',
                        median=reference['median'], support=reference['support'])
    fingerprint_bytes = (flow_fingerprint.encode()
                         + Path(background_module.__file__).read_bytes()
                         + reference['median'].tobytes()
                         + reference['support'].tobytes())
    if refine_labels:
        fingerprint_bytes += b'seeded-grabcut-v1' + Path(refine_module.__file__).read_bytes()
    if use_null_evidence:
        fingerprint_bytes += b'stationary-controls-v1' + Path(null_module.__file__).read_bytes()
    fingerprint = hashlib.sha256(fingerprint_bytes).hexdigest()
    summaries = {}
    for split in splits:
        sources = sorted((data_dir / 'observations' / split).glob('*.npz'))
        if not sources:
            raise ValueError(f'No observation groups for split {split}')
        (cache_dir / split).mkdir(parents=True, exist_ok=True)
        rows = []
        for index, source in enumerate(sources):
            output = cache_dir / split / source.name
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            reuse = False
            if output.exists():
                with np.load(output, allow_pickle=False) as cached:
                    reuse = ('fingerprint' in cached and str(cached['fingerprint']) == fingerprint
                             and str(cached['source_hash']) == source_hash)
            if not reuse:
                arrays = None
                candidate = Path(reuse_cache) / split / source.name if reuse_cache else output
                if candidate.exists():
                    with np.load(candidate, allow_pickle=False) as previous:
                        prior_fingerprint = str(previous['flow_fingerprint'] if 'flow_fingerprint' in previous else previous['fingerprint'])
                        if prior_fingerprint == flow_fingerprint and str(previous['source_hash']) == source_hash:
                            arrays = {key: previous[key] for key in previous.files}
                with np.load(source, allow_pickle=False) as observation:
                    # Deliberately whitelist only images and measured position increments.
                    if arrays is None:
                        result = flow_module.probe_supervision(
                            observation['rgb'], observation['probe_rgb'], observation['dq'],
                            np.asarray(meta['probe_scales'], dtype=np.float32))
                        arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
                        diagnostics = result.get('diagnostics', {})
                    else:
                        diagnostics = json.loads(str(arrays['diagnostics_json']))
                    base_labels = arrays.get('base_labels', arrays['labels']).copy()
                    arrays['base_labels'] = base_labels
                    arrays['labels'] = background_module.extend_background_labels(
                        observation['rgb'], observation['probe_rgb'], base_labels, reference)
                    extended = (base_labels == -1) & (arrays['labels'] == 0)
                    arrays['confidence'][extended] = 1.
                    diagnostics['background_extension_pixels'] = int(extended.sum())
                    diagnostics['background_reference_fit'] = 'training source RGB only; frozen for other splits'
                    if use_null_evidence:
                        if 'null_rgb' not in observation or 'null_dq' not in observation:
                            raise ValueError('Stationary-control evidence requires null_rgb and measured null_dq')
                        arrays['labels'], arrays['null_negative_mask'], null_diagnostic = null_module.stationary_control_evidence(
                            observation['rgb'], observation['null_rgb'], observation['null_dq'],
                            np.asarray(meta['probe_scales']), arrays['labels'])
                        diagnostics['stationary_controls'] = null_diagnostic
                    else:
                        arrays.pop('null_negative_mask', None)
                        diagnostics.pop('stationary_controls', None)
                    arrays['seed_labels'] = arrays['labels'].copy()
                    if refine_labels:
                        arrays['labels'] = refine_module.refine_partial_labels(
                            observation['rgb'], arrays['seed_labels'])
                        inferred = (arrays['seed_labels'] == -1) & (arrays['labels'] != -1)
                        arrays['confidence'][inferred] = .5
                    arrays['confidence'][arrays['labels'] == -1] = 0.
                    diagnostics['boundary_refinement'] = {
                        'enabled': bool(refine_labels),
                        'method': 'Per-image seeded GrabCut; no geometry or chosen robot color',
                        'seed_positive_fraction': float(np.mean(arrays['seed_labels'] == 1)),
                        'inferred_positive_fraction': float(np.mean(
                            (arrays['seed_labels'] == -1) & (arrays['labels'] == 1))),
                    }
                arrays.update(fingerprint=np.array(fingerprint), flow_fingerprint=np.array(flow_fingerprint),
                              source_hash=np.array(source_hash),
                              diagnostics_json=np.array(json.dumps(diagnostics)))
                np.savez_compressed(output, **arrays)
            with np.load(output, allow_pickle=False) as cached:
                labels = cached['labels']
                rows.append({'positive': float(np.mean(labels == 1)),
                             'negative': float(np.mean(labels == 0)),
                             'ignored': float(np.mean(labels == -1))})
            if (index + 1) % 16 == 0 or index + 1 == len(sources):
                print(f'flow cache {split}: {index + 1}/{len(sources)}', flush=True)
        summaries[split] = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    (cache_dir / 'summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
    return summaries


class ObservationDataset(Dataset):
    def __init__(self, data_dir, cache_dir, split):
        self.root, self.cache = Path(data_dir), Path(cache_dir)
        self.split = split
        meta = json.loads((self.root / 'metadata.json').read_text(encoding='utf-8'))
        self.center = np.asarray(meta['q_center'], dtype=np.float32)
        self.scale = np.asarray(meta['q_scale'], dtype=np.float32)
        self.paths = sorted((self.root / 'observations' / split).glob('*.npz'))
        if not self.paths:
            raise ValueError(f'Empty {split} dataset')
        # Load only selected fields. No path under evaluation/ is accessed here.
        self.items = []
        for source in self.paths:
            with np.load(source, allow_pickle=False) as record:
                image = record['rgb'].astype(np.float32).transpose(2, 0, 1) / 255.
                q = (record['q'].astype(np.float32) - self.center) / self.scale
            with np.load(self.cache / split / source.name, allow_pickle=False) as record:
                item = {'rgb': image, 'q': q,
                        'flow': record['flow'].astype(np.float32).transpose(0, 3, 1, 2),
                        'valid': record['valid'].astype(bool),
                        'dq': record['dq_scaled'].astype(np.float32),
                        'labels': record['labels'].astype(np.float32)}
                if 'null_negative_mask' in record:
                    # Positive conflicts were made unknown and must stay excluded.
                    item['control_negative'] = (record['null_negative_mask'].astype(bool)
                                                 & (record['seed_labels'] == 0))
            self.items.append({key: torch.from_numpy(value.copy()) for key, value in item.items()})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]
