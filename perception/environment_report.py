"""Matched three-checkpoint environment benchmark, including the ablation."""
from __future__ import annotations
import argparse
import base64
import hashlib
import html
import json
from pathlib import Path
from project_paths import recorded_path
import numpy as np
from PIL import Image, ImageDraw
import torch
from .evaluate import _bound_preprocessing, _font
from .model import BodyPerception


def build_report(baseline, exposure, controls, output):
    paths = list(map(Path, (baseline, exposure, controls)))
    results = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
    names = ('V4', 'V5 exposure only', 'V5 + stationary controls')
    names = tuple(name + (' + support' if result['model_config'].get('body_support') else '')
                  for name, result in zip(names, results))
    supports = [result.get('training', {}).get('composition', {}).get('body_support', {}).get('sha256') for result in results[1:]]
    if supports[0] != supports[1]:
        raise ValueError('The control ablation must use identical frozen body support')
    control_weight = results[-1].get('training', {}).get('stationary_control_loss_weight', 0.)
    support_note = ('它们使用相同的冻结身体候选范围限制：输出与 V4 身体预测取交集，防止新分支扩张原先的候选范围。'
                    if supports[0] else '两个 V5 分支均未施加额外的冻结身体候选约束。')
    for key in ('data', 'data_seed', 'split', 'threshold'):
        if len({json.dumps(result[key]) for result in results}) != 1:
            raise ValueError(f'Benchmarks must share {key}')
    pose_ids = [set(row['id'] for row in result['per_pose']) for result in results]
    if not all(ids == pose_ids[0] for ids in pose_ids):
        raise ValueError('Benchmarks must contain the same poses')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    root = recorded_path(results[0]['data'])
    metadata = json.loads((root/'metadata.json').read_text(encoding='utf-8'))
    networks = []
    for result in results:
        checkpoint = recorded_path(result['checkpoint'])
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != result['checkpoint_sha256']:
            raise ValueError('Checkpoint changed after evaluation')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        center, scale = _bound_preprocessing(saved, metadata)
        model = BodyPerception(**saved['model_config']).eval()
        model.load_state_dict(saved['state_dict'])
        networks.append((model, center, scale))
    selected = []
    for profile in ('clean', 'static', 'moving', 'occlusion', 'lighting'):
        group = [row for row in results[-1]['per_pose'] if row['environment_profile'] == profile]
        selected.append(min(group, key=lambda row: (row['source']['iou'], row['id'])))
    width, height = metadata['width'], metadata['height']
    panel = Image.new('RGB', (width*10, (height*2+38)*len(selected)+32), '#17202c')
    draw, font = ImageDraw.Draw(panel), _font(16)
    for col, name in enumerate(('Camera RGB', *names, 'Evaluation truth only')):
        draw.text((col*width*2+6, 7), name, font=font, fill='white')
    with torch.inference_mode():
        for index, row in enumerate(selected):
            with np.load(root/'observations'/results[0]['split']/f"{row['id']}.npz") as record:
                rgb, q = record['rgb'], record['q']
            predictions = []
            for network, center, scale in networks:
                image = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()[None]/255
                feedback = torch.from_numpy(((q-center)/scale).astype(np.float32))[None]
                mask = torch.sigmoid(network(image, feedback)['logits'])[0, 0].numpy() >= .5
                overlay = rgb.copy()
                overlay[mask] = (.5*rgb[mask]+np.array([20, 120, 45])).astype(np.uint8)
                predictions.append(overlay)
            with np.load(root/'evaluation'/results[0]['split']/f"{row['id']}.npz") as record:
                truth = record['body_mask'].astype(np.uint8)*255
            top = 32 + index*(height*2+38)
            draw.text((8, top+5), f"{row['environment_profile']} | lowest V5+control IoU | {row['id']}", font=font, fill='white')
            for col, pixels in enumerate((rgb, *predictions, truth)):
                picture = Image.fromarray(pixels).convert('RGB').resize((width*2, height*2), Image.Resampling.NEAREST)
                panel.paste(picture, (col*width*2, top+35))
    image_path = output/'environment-comparison.png'
    panel.save(image_path)
    headings = ''.join(f'<th>{html.escape(name)}</th>' for name in names)

    def table_rows():
        rows = ''
        for profile in ('clean', 'static', 'moving', 'occlusion', 'lighting'):
            for key, label in (('iou', '身体 IoU ↑'), ('fpr', '杂物误判率 ↓')):
                values = []
                for result in results:
                    group = result['by_environment'][profile]
                    value = group['source']['macro']['iou'] if key == 'iou' else group['distractors']['false_positive_rate']
                    values.append('—' if value is None else f'{value*100:.2f}%')
                rows += f'<tr><td>{profile} / {label}</td>'+''.join(f'<td>{value}</td>' for value in values)+'</tr>'
        for key, label in (('source', '全部初始姿态 IoU ↑'), ('stopped_after_motion', '运动后保持 IoU ↑')):
            rows += f'<tr><td>{label}</td>'+''.join(f"<td>{result[key]['macro']['iou']*100:.2f}%</td>" for result in results)+'</tr>'
        if all('null_controls' in result for result in results):
            for key, label in (('iou', 'q 保持、环境变化：身体 IoU ↑'), ('fpr', 'q 保持、环境变化：杂物误判率 ↓')):
                values = [(result['null_controls']['body']['macro']['iou'] if key == 'iou' else result['null_controls']['distractors']['false_positive_rate']) for result in results]
                rows += f'<tr><td>{label}（全部环境）</td>'+''.join(f'<td>{value*100:.2f}%</td>' for value in values)+'</tr>'
            for profile in ('moving', 'occlusion'):
                for key, label in (('iou', '保持身体 IoU ↑'), ('fpr', '保持时杂物误判率 ↓')):
                    groups = [result['by_environment'][profile]['null_controls'] for result in results]
                    values = [group['body']['macro']['iou'] if key == 'iou' else group['distractors']['false_positive_rate'] for group in groups]
                    rows += f'<tr><td>{profile} / {label}</td>'+''.join(f'<td>{value*100:.2f}%</td>' for value in values)+'</tr>'
        return rows

    summary = {'data': results[0]['data'], 'data_seed': results[0]['data_seed'],
               'pose_count': len(pose_ids[0]), 'selected_pose_ids': [row['id'] for row in selected],
               'models': {name: {'checkpoint_sha256': result['checkpoint_sha256'],
                                 'source': result['source'], 'stopped_after_motion': result['stopped_after_motion'],
                                 'by_environment': result['by_environment'],
                                 'null_controls': result.get('null_controls')} for name, result in zip(names, results)},
               'occlusion_coverage': results[0].get('occlusion_coverage'),
               'stationary_control_loss_weight': control_weight,
               'control_ablation_iou_delta': results[2]['source']['macro']['iou']-results[1]['source']['macro']['iou']}
    (output/'environment-comparison.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    encoded = base64.b64encode(image_path.read_bytes()).decode()
    occlusion = html.escape(json.dumps(summary['occlusion_coverage'], ensure_ascii=False, indent=2))
    report = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>环境干扰：实测与对照</title>
<style>body{{background:#111827;color:#e5e7eb;font:16px/1.7 system-ui;max-width:1500px;margin:auto;padding:28px}}table{{border-collapse:collapse;width:100%;background:#1f2937}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #374151}}img{{width:100%;height:auto}}pre{{white-space:pre-wrap}}a{{color:#93c5fd}}</style>
<h1>加入环境干扰后的身体感知</h1><p>同一批 {summary['pose_count']} 个新姿态，种子 {summary['data_seed']}，分割阈值统一 0.5。五类环境等量采集。身体 IoU 为逐姿态平均；杂物误判率为误判像素数除以可见杂物像素总数。</p>
<table><tr><th>指标</th>{headings}</tr>{table_rows()}</table>
<p>两个 V5 分支使用相同初始化、数据、训练种子及 25 轮分割训练。对照分支在 GrabCut 之前加入两次实测关节保持的负证据，并为可靠负种子单独计算平均 BCE（权重 {control_weight:g}）。{support_note}该对照方案额外改变初始姿态 IoU {summary['control_ablation_iou_delta']*100:+.2f} 个百分点。它不是新的传感器模态，也不证明当前 q 在推理时发挥作用。</p>
<h2>每类环境中 V5 加对照的最低 IoU 样本</h2><p>绿色为模型预测身体，最右为仅评价使用的身体真值。这些是失败侧样本；完整逐姿态结果保存在各模型 metrics.json。</p><img src="data:image/png;base64,{encoded}" alt="五类环境、三个模型预测与评价真值">
<h2>真实遮挡覆盖</h2><p>只在评价目录中额外渲染隐藏杂物的身体掩膜，用来统计实际遮挡。前景物体配置不保证每个姿态都有遮挡；不将该额外掩膜用于训练。</p><pre>{occlusion}</pre>
<h2>经验与限制</h2><p>相似颜色的外物揭示了旧模型的外观捷径。新环境训练与静止对照的贡献必须分开检验。对照只给有可靠对应关系的像素增加负种子；与身体正种子冲突时退回未知。其余未知像素通过逐图像 GrabCut 推断，仍是有噪声的弱标签。</p>
<p>相机固定、阴影关闭、反馈理想；外物为独立运动且无接触的可视干扰，未模拟碰撞。每组探测内光照不变，组间改变照明和地面外观。这里只测试有限形状、外观和姿态范围，不等于任意环境泛化。模型逐帧推理，未学习时间记忆，未完成 SO-ARM 真机验证。冻结原运动分支可防止被新训练改写，不能保证它已学会排除外物运动。候选范围限制增加一套网络，且无法补回 V4 已漏掉的身体区域。</p>
<p><a href="report.html">V5 完整报告</a> · <a href="environment-demo.gif">机械臂运动后停止、外物继续运动</a></p></html>'''
    (output/'environment-comparison.html').write_text(report, encoding='utf-8')
    print(f'Report: {(output/"environment-comparison.html").resolve()}', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('baseline', 'exposure', 'controls', 'output'):
        parser.add_argument(f'--{argument}', required=True)
    build_report(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
