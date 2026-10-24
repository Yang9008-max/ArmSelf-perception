"""CPU inference adapter: camera RGB plus seven measured positions, no truth."""
from __future__ import annotations

import hashlib
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from .model import BodyPerception


class LiveBodyPredictor:
    input_size = (192, 144)
    threshold = .5

    def __init__(self, checkpoint, threads=2):
        checkpoint = Path(checkpoint).resolve()
        torch.set_num_threads(threads)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        normalization = saved['preprocessing']
        self.center = np.asarray(normalization['q_center'], dtype=np.float32)
        self.scale = np.asarray(normalization['q_scale'], dtype=np.float32)
        if (self.center.shape != (7,) or self.scale.shape != (7,)
                or not np.isfinite(self.center).all() or not np.isfinite(self.scale).all()
                or np.any(self.scale <= 0) or saved['model_config']['channels'] != 7):
            raise ValueError('Model requires finite seven-channel position normalization')
        self.network = BodyPerception(**saved['model_config'])
        self.network.load_state_dict(saved['state_dict'])
        self.network.eval()
        deployment = saved.get('deployment', {})
        self.metadata = {
            'model': deployment.get('model', 'V5' if checkpoint.parent.name == 'run-v5-supported' else checkpoint.parent.name),
            'checkpoint': str(checkpoint),
            'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'input_size': list(self.input_size), 'threshold': self.threshold,
            'q_center': self.center.tolist(), 'q_scale': self.scale.tolist(),
            'feedback_units': ['radian']*6+['metre'],
            'training_scene': deployment.get('training_scene', '旧场景；当前眼位尚未重新训练'),
            'training_scene_sha256': deployment.get('scene_sha256'),
            'live_scene_sha256': deployment.get('live_scene_sha256'),
        }

    def predict(self, rgb, measured_q):
        rgb = np.asarray(rgb)
        q = np.asarray(measured_q, dtype=np.float32)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError('Camera input must be uint8 H×W×3 RGB')
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError('Feedback must contain seven finite measured positions')
        started = time.perf_counter()
        image = np.asarray(Image.fromarray(rgb).resize(self.input_size, Image.Resampling.BILINEAR))
        tensor_rgb = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()[None]/255
        tensor_q = torch.from_numpy((q-self.center)/self.scale)[None]
        with torch.inference_mode():
            logits = self.network(tensor_rgb, tensor_q)['logits']
            if not torch.isfinite(logits).all():
                raise RuntimeError('Model produced non-finite body logits')
            probability = torch.sigmoid(logits)[0, 0].cpu().numpy().copy()
        low_mask = (probability >= self.threshold).astype(np.uint8)*255
        mask = np.asarray(Image.fromarray(low_mask).resize(
            (rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST)) > 0
        overlay = rgb.copy()
        overlay[mask] = (.5*rgb[mask]+.5*np.array([40, 240, 90])).astype(np.uint8)
        return {'probability': probability, 'mask': mask, 'overlay': overlay,
                'inference_ms': (time.perf_counter()-started)*1000}
