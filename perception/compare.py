"""Create a matched-pose report from two completed checkpoint evaluations."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from .evaluate import _bound_preprocessing, _font
from .model import BodyPerception


def compare_reports(baseline_metrics, candidate_metrics, output):
    baseline_path, candidate_path, output = map(Path, (baseline_metrics, candidate_metrics, output))
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    candidate = json.loads(candidate_path.read_text(encoding='utf-8'))
    for key in ('data', 'data_seed', 'split', 'threshold'):
        if baseline[key] != candidate[key]:
            raise ValueError(f'Matched evaluations must share {key}')
    old_rows = {row['id']: row for row in baseline['per_pose']}
    new_rows = {row['id']: row for row in candidate['per_pose']}
    if old_rows.keys() != new_rows.keys():
        raise ValueError('Matched evaluations must contain identical pose IDs')
    deltas = np.array([new_rows[key]['source']['iou'] - old_rows[key]['source']['iou']
                       for key in sorted(new_rows)])
    summary = {
        'data_seed': candidate['data_seed'], 'pose_count': len(new_rows),
        'baseline_checkpoint_sha256': baseline['checkpoint_sha256'],
        'candidate_checkpoint_sha256': candidate['checkpoint_sha256'],
        'baseline': {key: baseline[key] for key in ('source', 'stopped_after_motion', 'motion')},
        'candidate': {key: candidate[key] for key in ('source', 'stopped_after_motion', 'motion')},
        'paired_source_iou_delta': {'mean': float(deltas.mean()), 'min': float(deltas.min()),
                                  'max': float(deltas.max()), 'improved_poses': int((deltas > 0).sum())},
    }
    torch.set_num_threads(4)
    networks = []
    data = Path(candidate['data'])
    metadata = json.loads((data/'metadata.json').read_text(encoding='utf-8'))
    for metrics in (baseline, candidate):
        checkpoint = Path(metrics['checkpoint'])
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != metrics['checkpoint_sha256']:
            raise ValueError('Checkpoint changed after evaluation')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        center, scale = _bound_preprocessing(saved, metadata)
        network = BodyPerception(**saved['model_config']).eval()
        network.load_state_dict(saved['state_dict'])
        networks.append((network, center, scale))
    ranked = sorted(new_rows, key=lambda key: (new_rows[key]['source']['iou'], key))
    selected = [ranked[i] for i in sorted(set((0, len(ranked)//2, len(ranked)-1)))]
    width, height = metadata['width'], metadata['height']
    row_height = height*2 + 36
    panel = Image.new('RGB', (width*8, row_height*len(selected)*2 + 32), '#17202c')
    draw, font = ImageDraw.Draw(panel), _font(17)
    for col, caption in enumerate(('Camera RGB', 'V3 prediction', 'V4 prediction', 'Evaluation truth only')):
        draw.text((col*width*2+8, 7), caption, font=font, fill='white')
    with torch.inference_mode():
        for sample_index, sample_id in enumerate(selected):
            with np.load(data/'observations'/candidate['split']/f'{sample_id}.npz') as record:
                observations = [(record['rgb'], record['q']), (record['stop_rgb'], record['stop_q'])]
            for phase_index, (rgb, q) in enumerate(observations):
                predictions = []
                for network, center, scale in networks:
                    image = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()[None]/255
                    feedback = torch.from_numpy(((q-center)/scale).astype(np.float32))[None]
                    probability = torch.sigmoid(network(image, feedback)['logits'])[0, 0].numpy()
                    predictions.append((probability >= candidate['threshold']).astype(np.uint8)*255)
                # As in the evaluator, privileged truth is read only after inference.
                with np.load(data/'evaluation'/candidate['split']/f'{sample_id}.npz') as truth:
                    mask = truth['body_mask' if phase_index == 0 else 'stop_mask'].astype(np.uint8)*255
                phase = 'source' if phase_index == 0 else 'stopped_after_motion'
                top = 32 + (sample_index*2+phase_index)*row_height
                label = (f'{sample_id} | {phase} | V3 {old_rows[sample_id][phase]["iou"]:.3f}'
                         f' | V4 {new_rows[sample_id][phase]["iou"]:.3f}')
                draw.text((8, top+5), label, font=font, fill='white')
                for col, pixels in enumerate([rgb, *predictions, mask]):
                    picture = Image.fromarray(pixels).convert('RGB')
                    panel.paste(picture.resize((width*2, height*2), Image.Resampling.NEAREST),
                                (col*width*2, top+34))
    output.mkdir(parents=True, exist_ok=True)
    image_path = output/'comparison.png'
    panel.save(image_path)
    summary['selected_pose_ids'] = selected
    (output/'comparison.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    encoded = base64.b64encode(image_path.read_bytes()).decode('ascii')
    rows = ''
    for key, label in (('source', '新初始姿态 IoU'), ('stopped_after_motion', '运动后保持姿态 IoU')):
        old, new = baseline[key]['macro']['iou'], candidate[key]['macro']['iou']
        rows += f'<tr><td>{label}</td><td>{old*100:.2f}%</td><td>{new*100:.2f}%</td></tr>'
    for key, label in (('moving', '运动区域 EPE'), ('static', '静止区域 EPE'), ('all', '整图 EPE')):
        old = baseline['motion']['model'][key]['epe']
        new = candidate['motion']['model'][key]['epe']
        rows += f'<tr><td>{label}（像素，越低越好）</td><td>{old:.4f}</td><td>{new:.4f}</td></tr>'
    report = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>V3 与 V4：相同新姿态对比</title><style>body{{background:#111827;color:#e5e7eb;font:16px/1.7 system-ui;margin:auto;max-width:1400px;padding:28px}}table{{border-collapse:collapse;width:100%;background:#1f2937}}td,th{{text-align:left;padding:12px;border-bottom:1px solid #374151}}img{{width:100%;height:auto}}a{{color:#93c5fd}}</style>
<h1>身体感知优化：V3 与 V4</h1><p>权重与规则冻结后，在同一批 {len(new_rows)} 个新姿态上比较，种子 {candidate['data_seed']}，分割阈值均为 0.5。</p>
<table><tr><th>指标</th><th>原模型 V3</th><th>优化模型 V4</th></tr>{rows}</table>
<p>逐姿态 IoU 平均增加 {deltas.mean()*100:.2f} 个百分点，{int((deltas>0).sum())}/{len(deltas)} 个姿态改善。EPE 的目标为 RGB 估计光流，不是仿真物理光流。静止背景仍须与全部预测不动的基线比较，完整指标见新模型报告。</p>
<p>改动：从动作与静态背景种子出发，利用逐图像颜色及边缘补全弱标签；分割训练后组合独立冻结的运动响应分支。推理不运行 GrabCut。没有把身体真值用于训练。</p>
<p>边界：固定相机、固定外观、无投影阴影、理想反馈，使用通用七通道仿真机构。当前位置反馈融合仍需单独检验，身体边界与夹爪仍有误差。</p>
<h2>按 V4 IoU 选择最低、中位与最高样本</h2><p>每个样本同时展示初始图像和运动后保持图像。新旧模型各列对应完全相同的相机观测。</p>
<img alt="相同新姿态上的旧模型、新模型和真值" src="data:image/png;base64,{encoded}">
<p><a href="report.html">优化模型的完整评价</a> · <a href="motion-demo.gif">连续运动与保持演示</a></p></html>'''
    (output/'comparison.html').write_text(report, encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-metrics', required=True)
    parser.add_argument('--candidate-metrics', required=True)
    parser.add_argument('--output', required=True)
    compare_reports(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
