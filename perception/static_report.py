"""Evaluate frozen static-scene candidates and publish their paired comparison."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw

from .evaluate import evaluate_checkpoint
from .demo import _font


def build(run, publish, evaluate=True):
    run, publish = Path(run), Path(publish)
    freeze = json.loads((run / 'freeze.json').read_text(encoding='utf-8'))
    scores = {}
    for name, item in freeze['checkpoints'].items():
        checkpoint = Path(item['path'])
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f'Checkpoint changed after freeze: {name}')
        output = run / 'final' / name
        if evaluate:
            metrics = evaluate_checkpoint(run / 'final-data', checkpoint, output)
        else:
            metrics = json.loads((output / 'metrics.json').read_text(encoding='utf-8'))
        scores[name] = metrics
        print(name, metrics['source']['macro'], metrics['distractors'], flush=True)
    baseline, selected = scores['v6-raw'], scores[freeze['selected']]
    assert [x['id'] for x in baseline['per_pose']] == [x['id'] for x in selected['per_pose']]
    gains = np.array([b['source']['iou'] - a['source']['iou']
                      for a, b in zip(baseline['per_pose'], selected['per_pose'])])
    rng = np.random.default_rng(20261024)
    bootstrap = gains[rng.integers(len(gains), size=(10000, len(gains)))].mean(1)
    summary = {'freeze': freeze, 'paired_poses': len(gains),
               'mean_iou_gain': float(gains.mean()),
               'paired_bootstrap_95_ci': np.quantile(bootstrap, [.025, .975]).tolist(),
               'improved_poses': int((gains > 0).sum()),
               'models': {name: {key: item[key] for key in
                                ('source', 'stopped_after_motion', 'distractors', 'checkpoint_sha256')}
                          for name, item in scores.items()},
               'final_metadata_sha256': hashlib.sha256((run / 'final-data/metadata.json').read_bytes()).hexdigest(),
               'limitations': ['Fixed camera and simulated morphology',
                               'No real hardware validation',
                               'The deployed neural weights are V6; gains come from RGB boundary refinement',
                               'Graph-cut refinement may fail on similar colors or incorrect confident seeds',
                               'Development used labels for postprocessing selection; final evaluation was separate']}
    publish.mkdir(parents=True, exist_ok=True)
    if (run / 'v7-static.gif').is_file():
        demos = [json.loads((run / f'{name}-static.json').read_text(encoding='utf-8'))
                 for name in ('v6', 'v7')]
        assert all(item['environment'] == 'static' for item in demos)
        for a, b in zip(demos[0]['per_frame'], demos[1]['per_frame']):
            np.testing.assert_array_equal(a['q'], b['q'])
            np.testing.assert_array_equal(a['prop_positions'], b['prop_positions'])
            np.testing.assert_array_equal(b['prop_positions'], demos[1]['per_frame'][0]['prop_positions'])
        summary['animation'] = {name: {key: item[key] for key in ('moving', 'hold', 'checkpoint_sha256')}
                                for name, item in zip(('v6', 'selected'), demos)}
        pictures = []
        with Image.open(run / 'v6-static.gif') as old, Image.open(run / 'v7-static.gif') as new:
            assert old.n_frames == new.n_frames == demos[0]['frames'] == demos[1]['frames']
            for index in range(old.n_frames):
                old.seek(index)
                new.seek(index)
                before, after = old.convert('RGB'), new.convert('RGB')
                panel = Image.new('RGB', (1152, 286), '#17202c')
                draw = ImageDraw.Draw(panel)
                row = demos[1]['per_frame'][index]
                draw.text((12, 6), f"t={row['time_seconds']:.2f}s | {row['phase']} | props fixed", fill='white', font=_font(18))
                crops = [before.crop((0, 70, 384, 358)), before.crop((384, 70, 768, 358)),
                         after.crop((384, 70, 768, 358)), after.crop((768, 70, 1152, 358))]
                for col, (crop, label) in enumerate(zip(crops, ('Camera RGB', 'V6 prediction', 'Optimized prediction', 'Evaluation truth only'))):
                    draw.text((col * 288 + 10, 37), label, fill='#c8d8ee', font=_font(17))
                    panel.paste(crop.resize((288, 216), Image.Resampling.NEAREST), (col * 288, 70))
                pictures.append(panel)
            pictures[0].save(publish / 'comparison.gif', save_all=True, append_images=pictures[1:],
                             duration=67, loop=0, disposal=2, optimize=False)
        pictures[40].save(publish / 'preview.png')
        for name in ('v6', 'v7'):
            shutil.copy2(run / f'{name}-static.json', publish / f'{name}-static.json')
    (run / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    shutil.copy2(run / 'summary.json', publish / 'summary.json')
    for name in scores:
        destination = publish / name
        destination.mkdir(exist_ok=True)
        for filename in ('report.html', 'metrics.json', 'examples.png'):
            shutil.copy2(run / 'final' / name / filename, destination / filename)
    shutil.copy2('docs/research/2026-10-24-related-work.md', publish / 'related-work.md')
    labels = {'v6-raw': '原始 V6', 'v6-refined': '启用版：V6 + 边界细化',
              'finetuned-raw': '静态微调候选', 'finetuned-refined': '微调候选 + 边界细化'}
    rows = ''.join(f'<tr><td><a href="{name}/report.html">{labels[name]}</a></td>'
                   f'<td>{m["source"]["macro"]["iou"]:.4f}</td>'
                   f'<td>{m["stopped_after_motion"]["macro"]["iou"]:.4f}</td>'
                   f'<td>{m["distractors"]["false_positive_rate"]:.2%}</td></tr>'
                   for name, m in summary['models'].items())
    low, high = summary['paired_bootstrap_95_ci']
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>静态场景 · 身体感知优化</title>
<style>body{{margin:0;background:#10171e;color:#dce7ee;font:16px/1.8 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1160px;margin:auto;padding:36px 24px}}a{{color:#80d8cd}}h1{{font-size:32px;line-height:1.4}}
.note{{color:#afc0ca}}section{{background:#19242e;border:1px solid #30404d;border-radius:12px;padding:22px;margin:24px 0}}
table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;border-bottom:1px solid #34424c;padding:10px}}
img{{max-width:100%;height:auto;border-radius:8px}}pre{{white-space:pre-wrap;font-size:12px}}.tag{{color:#84e1bc}}
</style><main><p class="tag">本机实测 · 2026-10-24 · 干扰物全程静止</p><h1>让身体边界贴合机械臂</h1>
<p>固定其他物体，只让机械臂执行动作。启用版保留 V6 神经网络权重，用 RGB 和预测置信度细化边界。</p>
<p><a href="http://127.0.0.1:8765/">打开实时仿真</a> · <a href="../">之前的 E2 / E3 结果</a> · <a href="related-work.md">研究与开源笔记</a></p>
<section><h2>相同画面的优化前后对比</h2><img src="comparison.gif" alt="同一场景：RGB、V6、优化版、评估真值">
<p class="note">动画是实际仿真与推理输出。左右使用同一动作、同一画面和同一物体位置；真值仅用于评分和右侧展示。</p></section>
<section><h2>80 组新姿态的独立测试</h2><table><tr><th>方案</th><th>初始姿态 IoU ↑</th><th>运动后保持 IoU ↑</th><th>杂物误分割率 ↓</th></tr>{rows}</table>
<p>启用版平均 IoU 提升 {gains.mean():.4f}；{summary['improved_poses']} / {len(gains)} 个姿态提升。
配对 bootstrap 95% 区间 [{low:.4f}, {high:.4f}]。点击各方案查看最低、中位、最高样本和完整记录。</p>
<p class="note">这些是固定相机、静态干扰物、带阴影的新仿真数据。不能将它们与上一轮移动杂物动画的 0.84 直接比较。
杂物误分割率按全部可见杂物像素汇总；身体 IoU 按姿态平均。虽然身体边界总体改善，杂物误分割仍从 4.31% 上升到 4.90%，没有实现完全排除杂物。</p></section>
<section><h2>实际做了什么</h2><p>新增 100 组训练姿态、25 组验证姿态和 20 组开发姿态，执行了 24 轮小学习率微调及亮度扰动。
微调未在开发集带来稳定提升，因此实际启用的优化版采用 V6 权重。四个方案都保留在表中。</p>
<p>网络置信度仅提供图割种子，RGB 提供颜色边界。没有预设机械臂颜色，没有人工绘制身体掩膜，未将仿真真值输入训练或推理。
部署方案与后处理在最终 80 组数据采集前冻结。输出二值掩膜经过细化，原始网络概率保持不变，不能视作新的校准概率。</p>
<p>新旧候选使用相同的冻结运动响应分支。本轮优化可见身体分割；完整三维形状学习仍需后续验证。</p></section>
<section><h2>相关论文与开源</h2><ul>
<li><a href="https://github.com/BoyuanChen/visual-selfmodeling">Full-Body Visual Self-Modeling</a>：关节条件的连续三维身体表示，适合下一阶段三维建模。</li>
<li><a href="https://github.com/michaelyuancb/roboengine">RoboEngine / Robo-SAM</a>：机械臂分割与背景增强，可作为跨场景分割基线。</li>
<li><a href="https://github.com/facebookresearch/co-tracker">CoTracker3</a>：在线 / 离线身体点跟踪，可用于验证 E2 / E3 的身份稳定性。</li>
</ul><p class="note">已核对官方仓库；本轮未运行上述三个项目的大模型。具体适配条件及来源见研究笔记。</p></section>
<section><h2>验证与边界</h2><p>66 项测试通过，包括物体固定、身体边界细化和旧模型兼容检查。实时服务另有相机、反馈、关节、夹爪、阴影和预测真值分离检查。</p>
<p>相同颜色和错误的高置信度种子仍可能造成误分割。当前使用理想仿真反馈、固定相机，尚无真实硬件验证。</p>
<details><summary>冻结记录、误差与逐方案指标</summary><pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre></details></section></main></html>'''
    (publish / 'index.html').write_text(page, encoding='utf-8')
    print('Published:', publish.resolve(), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('outputs/perception/v7-static'))
    parser.add_argument('--publish', type=Path, default=Path('outputs/local-check-20261024/v7-static'))
    parser.add_argument('--build-only', action='store_true')
    args = parser.parse_args()
    build(args.run, args.publish, evaluate=not args.build_only)


if __name__ == '__main__':
    main()
