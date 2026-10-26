"""E5 report and an evaluation-only binocular visibility audit."""
from __future__ import annotations

import csv
import html
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .body_cycle import read_json, write_json, reference, sha, utc
from .body_fixture import make_fixture, EvaluationLandmarks
from .recovery_cycle import ROOT, FINAL_SEEDS, BIAS
from .vision import project


LABELS = dict(feedback='Feedback only', original='E4 fusion', bank='Multi-pose RGB',
              state_only='Corrected decoder', corrected='RGB + state correction')
COLORS = dict(feedback='#dd9933', original='#778899', bank='#247bb6',
              state_only='#8b68ad', corrected='#008b7c')


def load(path):
    with np.load(path) as a:
        return {k: a[k].copy() for k in a.files}


def truths(source):
    rows = [load(source/'evaluation'/row['file']) for row in read_json(source/'metadata.json')['frames']]
    return {k: np.stack([r[k] for r in rows]) for k in
            ('anchor_xyz', 'anchor_body', 'anchor_visible', 'behind_panel')}


def binocular_audit(output, seed):
    """Right-camera visibility is a diagnostic, never a changed main denominator."""
    destination = output/'audit'/f'{seed}.json'
    if destination.exists():
        return read_json(destination)
    prediction = output/'predictions'/f'final-{seed}-clean.npz'
    assert sha(prediction) == read_json(prediction.with_suffix('.freeze.json'))['prediction_sha256']
    destination.parent.mkdir(exist_ok=True)
    fixture = destination.parent/'fixture'
    fixture.mkdir(exist_ok=True)
    model, data, calibration = make_fixture(fixture, panel=True)
    saved, _, pixels = reference()
    anchors = EvaluationLandmarks(model, data, pixels[saved['selected'], 0], calibration)
    source = output/f'final-data-{seed}'
    truth = truths(source)
    metadata = read_json(source/'metadata.json')
    indices = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    renderer = mujoco.Renderer(model, height=480, width=640)
    option = mujoco.MjvOption()
    option.geomgroup[3] = 1
    right_visible, max_replay_error = [], 0.
    try:
        for i, row in enumerate(metadata['frames']):
            with np.load(source/'observations'/row['file']) as obs:
                data.qpos[indices] = obs['q']
                data.qpos[model.joint('finger_right').qposadr[0]] = obs['q'][6]
            mujoco.mj_forward(model, data)
            xyz = anchors.positions(model, data)
            # The non-actuated coupled finger is not one of the seven recorded
            # channels. Replaying its partner can differ by a few micrometres.
            replay_error = float(np.max(np.abs(xyz[anchors.is_body]-truth['anchor_xyz'][i, anchors.is_body])))
            max_replay_error = max(max_replay_error, replay_error)
            if replay_error > .0001:
                raise AssertionError('Replay differs by more than 0.1 mm')
            xyz = truth['anchor_xyz'][i].copy()
            uv = project(xyz, calibration)
            uv[:, 0] -= calibration['focal_px']*calibration['baseline_m']/np.maximum(xyz[:, 2], .15)
            px = np.rint(uv).astype(int)
            inside = anchors.is_body & (px[:, 0] >= 0) & (px[:, 0] < 640) & (px[:, 1] >= 0) & (px[:, 1] < 480)
            px[:, 0] = np.clip(px[:, 0], 0, 639)
            px[:, 1] = np.clip(px[:, 1], 0, 479)
            renderer.update_scene(data, camera='entity_right', scene_option=option)
            renderer.enable_depth_rendering()
            depth = renderer.render().copy()
            renderer.disable_depth_rendering()
            right_visible.append(inside & (np.abs(xyz[:, 2]-depth[px[:, 1], px[:, 0]]) < .015))
    finally:
        renderer.close()
    right_visible = np.stack(right_visible)
    original_events = read_json(output/'scores'/f'final-{seed}-clean-events.json')['original']
    events = []
    for e in original_events:
        p, start = e['point'], e['frame']
        both = truth['anchor_visible'][start:start+5, p] & right_visible[start:start+5, p]
        events.append(dict(point=p, frame=start, binocular_opportunity=bool(both.any()),
                           simultaneous_visible_frames=int(both.sum())))
    result = dict(seed=seed, evaluation_only=True, main_recovery_denominator_unchanged=True,
        events=len(events), binocular_opportunities=sum(e['binocular_opportunity'] for e in events),
        events_detail=events, maximum_replay_anchor_difference_m=max_replay_error,
        visibility_depth_tolerance_m=.015)
    np.savez_compressed(destination.with_suffix('.npz'), right_visible=right_visible)
    write_json(destination, result)
    return result


def summary(output):
    rows = [read_json(output/'scores'/f'final-{seed}-{condition}.json')
            for seed in FINAL_SEEDS for condition in ('clean', 'biased')]
    result = dict(created_utc=utc(), final_seeds=list(FINAL_SEEDS), frames_per_condition=420,
                  methods={}, per_seed=rows, binocular_audit=[binocular_audit(output, seed) for seed in FINAL_SEEDS])
    for condition in ('clean', 'biased'):
        selected = [r for r in rows if r['name'].endswith(condition)]
        result['methods'][condition] = {}
        for mode in LABELS:
            values = [r['modes'][mode] for r in selected]
            aggregate = {key: float(np.sqrt(np.mean([v[key]**2 for v in values])))
                         for key in ('all_body_rmse_mm', 'after_fault_rmse_mm', 'before_fault_rmse_mm', 'hidden_body_rmse_mm')}
            if 'recovery_count' in values[0]:
                for key in ('recovery_count', 'recovery_events', 'false_reappearance', 'accepted_reappearance'):
                    aggregate[key] = sum(v[key] for v in values)
                aggregate['recovery_rate'] = aggregate['recovery_count']/aggregate['recovery_events']
                for key in ('visible_correct_coverage', 'visible_correspondence_precision', 'covered_body_points', 'accepted_per_frame'):
                    aggregate[key] = float(np.mean([v[key] for v in values]))
            result['methods'][condition][mode] = aggregate
    result['fixed_props_max_displacement_m'] = max(r['prop_max_displacement_m'] for r in rows)
    return result


def figures(output, report, result):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    modes = list(LABELS)
    for shift, condition, alpha in [(-.18, 'clean', .45), (.18, 'biased', 1.)]:
        axes[0, 0].bar(np.arange(len(modes))+shift,
            [result['methods'][condition][m]['all_body_rmse_mm'] for m in modes], width=.36,
            color=[COLORS[m] for m in modes], alpha=alpha, label=condition)
    axes[0, 0].set(title='All 28 body anchors: three fresh seeds', ylabel='3D RMSE / mm',
                   xticks=np.arange(len(modes)), xticklabels=['Feedback', 'E4 fusion', 'RGB bank', 'State only', 'Corrected'])
    axes[0, 0].legend()
    for i, mode in enumerate(('original', 'bank', 'corrected')):
        row = result['methods']['clean'][mode]
        axes[0, 1].bar(i, row['recovery_rate']*100, color=COLORS[mode])
        axes[0, 1].text(i, row['recovery_rate']*100+2, f"{row['recovery_count']}/{row['recovery_events']}", ha='center')
    axes[0, 1].set(title='Same surface point recovered within 5 frames', ylabel='Recovery / %', ylim=(0, 100),
                   xticks=range(3), xticklabels=['E4 fusion', 'RGB bank', 'Corrected'])
    seed = FINAL_SEEDS[0]
    a = load(output/'predictions'/f'final-{seed}-biased.npz')
    truth = truths(output/f'final-data-{seed}')
    times = np.arange(140)*.1
    for mode in ('feedback', 'original', 'bank', 'corrected'):
        error = np.mean(np.sum((a[f'prediction_{mode}']-truth['anchor_xyz'])**2, axis=-1)[:, truth['anchor_body'][0]], axis=1)**.5*1000
        axes[1, 0].plot(times, error, label=LABELS[mode], color=COLORS[mode])
    axes[1, 0].axvspan(2.6, 3., color='#dd9933', alpha=.15)
    axes[1, 0].set(title='Injected encoder fault: first final seed', xlabel='Time / s', ylabel='All-body RMSE / mm')
    axes[1, 0].legend(fontsize=8)
    from .body_memory import BodyModel
    from .recovery_cycle import BODY
    scale = np.asarray(BodyModel(BODY).saved['scale'])
    for channel, color in zip(range(3), ['#b14c63', '#277ca3', '#007d65']):
        axes[1, 1].plot(times, a['offset'][:, channel]*scale[channel], color=color, label=f'Estimated joint {channel+1}')
        axes[1, 1].plot(times, BIAS[channel]*np.clip((np.arange(140)-25)/5., 0, 1), '--', color=color, alpha=.6)
    axes[1, 1].set(title='Dashed: injected offset (evaluation only)', xlabel='Time / s', ylabel='Encoder offset / rad')
    axes[1, 1].legend(fontsize=8)
    fig.savefig(report/'results.png', dpi=150)
    plt.close(fig)


def animation(output, report):
    seed = FINAL_SEEDS[0]
    source = output/f'final-data-{seed}'
    metadata = read_json(source/'metadata.json')
    a = load(output/'predictions'/f'final-{seed}-biased.npz')
    truth = truths(source)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17)
    frames = []
    for i in range(0, 140, 2):
        with np.load(source/'observations'/metadata['frames'][i]['file']) as obs:
            camera = Image.fromarray(obs['left'])
        canvas = Image.new('RGB', (1280, 550), '#101c26')
        for col, mode in enumerate(('feedback', 'corrected')):
            image = camera.copy()
            draw = ImageDraw.Draw(image)
            predicted = project(a[f'prediction_{mode}'][i], metadata['calibration'])
            actual = project(truth['anchor_xyz'][i], metadata['calibration'])
            for point in np.flatnonzero(truth['anchor_body'][i]):
                u, v = predicted[point]
                x, y = actual[point]
                if np.isfinite([u, v, x, y]).all():
                    draw.line([(float(x), float(y)), (float(u), float(v))], fill=COLORS[mode], width=1)
                    draw.ellipse((u-3, v-3, u+3, v+3), fill=COLORS[mode])
                    draw.ellipse((x-2, y-2, x+2, y+2), outline='#f5dff7')
            canvas.paste(image, (col*640, 35))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 7), 'Biased feedback: orange predictions', font=font, fill='#efa33d')
        draw.text((652, 7), 'RGB + state correction: green predictions', font=font, fill='#48d0b6')
        draw.text((12, 519), f't={i*.1:.1f}s | pale dots: evaluation-only true anchors | fixed screen and props | fault starts 2.6s', font=font, fill='white')
        frames.append(canvas.resize((1024, 440)))
    frames[0].save(report/'correction.gif', save_all=True, append_images=frames[1:], duration=160, loop=0, optimize=False)


def make_report(output=ROOT):
    read_json(output/'complete.json')
    report = output/'report'
    report.mkdir(exist_ok=True)
    result = summary(output)
    write_json(report/'summary.json', result)
    with (report/'scores.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        fieldnames = ['name', 'mode', 'all_body_rmse_mm', 'hidden_body_rmse_mm', 'after_fault_rmse_mm',
                      'recovery_count', 'recovery_events', 'visible_correct_coverage',
                      'visible_correspondence_precision', 'covered_body_points', 'false_reappearance', 'accepted_reappearance']
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in result['per_seed']:
            for mode, metrics in row['modes'].items():
                writer.writerow(dict(name=row['name'], mode=mode, **metrics))
    figures(output, report, result)
    animation(output, report)
    c, b = result['methods']['clean'], result['methods']['biased']
    labels = dict(feedback='只用原始反馈', original='E4 融合对照', bank='多姿态视觉记忆',
                  state_only='只取纠正后的状态预测', corrected='视觉记忆＋状态纠错')
    rows = ''.join(f'<tr><td>{labels[m]}</td><td>{c[m]["all_body_rmse_mm"]:.2f}</td><td>{b[m]["all_body_rmse_mm"]:.2f}</td>'
        f'<td>{str(c[m].get("recovery_count", "—"))+"/"+str(c[m].get("recovery_events", "—"))}</td></tr>' for m in labels)
    opportunities = sum(r['binocular_opportunities'] for r in result['binocular_audit'])
    events = sum(r['events'] for r in result['binocular_audit'])
    development = [read_json(p) for p in sorted((output/'scores').glob('dev-*.json')) if not p.stem.endswith('-events')]
    dev_rows = ''.join(f'<tr><td>{html.escape(r["name"])}</td><td>{r["modes"]["corrected"]["all_body_rmse_mm"]:.2f}</td>'
        f'<td>{r["modes"]["corrected"]["recovery_count"]}/{r["modes"]["corrected"]["recovery_events"]}</td></tr>' for r in development)
    validation = read_json(output/'validation.json') if (output/'validation.json').exists() else None
    verification = (f'{validation["full_unittest_count"]} 项回归测试通过。真实记忆的 5 次空白相机对照均无视觉接受、无状态更新；六份预测哈希和全部 41 点输出通过检查。'
                    if validation and validation['unittest_exit_code'] == 0 else '此目录尚未记录完整测试与负面对照验证。')
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>E5 · 身体状态纠错与遮挡恢复</title><style>*{{box-sizing:border-box}}body{{background:#101b24;color:#e2edf3;font:16px/1.8 system-ui,'Microsoft YaHei',sans-serif;margin:0}}main{{max-width:1120px;margin:auto;padding:32px 22px 80px}}h1{{font-size:32px}}h2{{font-size:23px;margin-top:38px}}p{{color:#bfccd6}}a{{color:#6bdec3}}img{{width:100%;height:auto;background:#fff;border-radius:8px}}table{{border-collapse:collapse;width:100%;font-size:15px}}th,td{{padding:12px;border-bottom:1px solid #354654;text-align:left}}.note{{border-left:3px solid #e4af61;padding:12px 18px;background:#23313e}}.tag{{color:#6bdec3}}details{{margin-top:24px}}code{{word-break:break-all}}@media(max-width:650px){{main{{padding:20px 12px}}table{{font-size:12px}}th,td{{padding:7px}}}}</style>
<main><p class="tag">E5 / 三个新种子 / 原有干扰物与遮挡板固定</p><h1>视觉能帮助纠正身体状态，持续身份仍未可靠建立</h1>
<p>这里的“反馈模型”把测到的关节位置映射为 41 个身体候选点的三维位置。评价始终包含其中全部 28 个真实身体点，丢失或未被看见的点也保留。</p>
<p class="note">加入读数偏差后，整体误差由 {b['feedback']['all_body_rmse_mm']:.2f} mm 降至 {b['corrected']['all_body_rmse_mm']:.2f} mm。正常反馈时为 {c['feedback']['all_body_rmse_mm']:.2f} → {c['corrected']['all_body_rmse_mm']:.2f} mm；纠错并非在所有条件下都有收益。</p>
<h2>新数据上的结果</h2><p>三条轨迹各 140 帧。正常与偏差条件使用相同的真实运动和 RGB；偏差仅注入传感器读数。误差单位为 mm，越低越好。</p>
<table><tr><th>方案</th><th>正常反馈 RMSE</th><th>偏差反馈 RMSE</th><th>正常条件下认回原点</th></tr>{rows}</table>
<p>恢复标准沿用 E4：同一点被挡至少 3 帧，左相机重新可见后 5 帧内，算法接受的双目测量须在原物理表面点 20 mm 内。预测位置接近真值不能独自计为恢复。</p>
<h2>反馈出错时，预测怎样变化</h2><p>左为原始关节反馈的空间预测，右为视觉辅助纠正后的预测。浅色小点仅供评价；它们没有提供给算法。橙色/绿色线显示预测偏差。</p>
<img src="correction.gif" alt="关节读数偏差下，原始反馈与视觉纠错的身体点预测对比">
<h2>空间误差、恢复率与状态估计</h2><img src="results.png" alt="三种子误差与恢复率，以及第一种子的误差曲线和关节偏差估计">
<p>偏差估计是对冻结空间模型输入的潜在偏移修正，会混合模型误差和视觉误差，不能直接当作准确的关节传感器标定。</p>
<h2>遮挡恢复的限制</h2><p>恢复率仍低，不能认定已经学会持续身体身份。评价后的双目可见性检查发现：{events} 个左相机重现事件中，{opportunities} 个在原 5 帧窗口内存在至少一帧双目同时可见。其余事件对当前双目重建流程存在可观测性限制；原评分分母保持不变，没有因此删掉困难事件。</p>
<p>这项诊断只检查几何可见性，不能保证图像纹理足够，也不能用来替错误匹配开脱。下一步应分别评价单目身份恢复和双目三维更新。</p>
<h2>本次试验的边界</h2><p>空间网络和 V7 分割权重均冻结。新增的是多姿态 RGB 参考记忆、显式光流关联和视觉残差纠错，没有训练新的身份网络。记忆只取 E4 的训练帧，既未使用 CAD，也未使用评价真值拟合。双目相机已标定，运动范围较小；三个新种子是同一路线的小扰动，并非三个独立场景。</p>
<p>桌面物体最大位移为 {result['fixed_props_max_displacement_m']:.1f} m。偏差从第 26–30 帧逐渐注入：前三个旋转关节为 +0.055、+0.018、−0.015 rad；算法不知道偏差值或起始时刻。未测试真实机械臂、未知相机或身体结构变化。</p>
<details><summary>开发集选择与负面结果</summary><p>开发种子 2026102495 与最终三种子分开。比较 7/11/21 像素窗口；7 像素在开发集恢复更多点，11 像素在偏差条件下误差更小，21 像素明显退化。最终选 7，配置及源码在采集最终数据前冻结。</p><table><tr><th>开发条件</th><th>纠错后 RMSE</th><th>认回原点</th></tr>{dev_rows}</table></details>
<h2>原始结果与研究依据</h2><p>{verification}</p><p><a href="summary.json">全部指标 JSON</a> · <a href="scores.csv">逐种子 CSV</a> · <a href="../validation.json">验证记录</a> · <a href="../final-protocol.json">最终协议与哈希</a> · <a href="../bank-provenance.json">视觉记忆来源</a> · <a href="/e4-body-cycle/report/">E4 对照</a></p>
<p><a href="https://arxiv.org/abs/2111.06389">Full-Body Visual Self-Modeling</a> 支持从视觉学习身体空间模型的可行性；<a href="https://cotracker3.github.io/">CoTracker3 官方项目</a>提供遮挡点跟踪的后续对照方向。本次没有运行 CoTracker3，不能声称优于它。</p></main></html>'''
    (report/'index.html').write_text(page, encoding='utf-8')
    print(report.resolve(), flush=True)
    print({condition: {m: round(v['all_body_rmse_mm'], 3) for m, v in scores.items()}
           for condition, scores in result['methods'].items()}, flush=True)


if __name__ == '__main__':
    make_report()
