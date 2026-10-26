"""Report every E4 denominator, failed correspondence and acquisition seed."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .body_cycle import (MODES, POLICIES, ACQUISITION_SEEDS, EVALUATION_SEEDS, BUDGETS,
                         read_json, write_json, rmse, sha, utc)
from .body_memory import BodyModel
from .vision import project


def arrays(path):
    with np.load(path) as data:
        return {k: data[k].copy() for k in data.files}


def join(items):
    return {k: np.concatenate([r[k] for r in items]) for k in items[0]}


def bridge_summary(output):
    data = join([arrays(output/'bridge'/f'eval-{s}.npz') for s in EVALUATION_SEEDS])
    result = []
    raw_body = data['valid_none'] & data['on_body']
    for name in MODES:
        valid = data[f'valid_{name}']
        correct = (valid & data['anchor_body'] & data['anchor_visible'] &
                   (np.linalg.norm(data['observed_xyz']-data['anchor_xyz'], axis=-1) <= .02))
        error = np.sum((data[f'prediction_{name}']-data['anchor_xyz'])**2, axis=-1)
        row = dict(mode=name, fixed_body_points=int(data['anchor_body'][0].sum()),
            accepted_point_observations=int(valid.sum()),
            pixel_body_purity=float((valid & data['on_body']).sum()/max(valid.sum(), 1)),
            raw_body_observations_retained=float((valid & data['on_body']).sum()/max(raw_body.sum(), 1)),
            correctly_matched_visible_body_coverage=float(correct.sum()/max((data['anchor_body'] & data['anchor_visible'] & data['available'][:, None]).sum(), 1)),
            all_body_rmse_mm=rmse(error, data['anchor_body']),
            visible_body_rmse_mm=rmse(error, data['anchor_body'] & data['anchor_visible']),
            blind_body_rmse_mm=rmse(error, data['anchor_body'] & (data['phase'] == 'blind')[:, None]),
            rmse_by_evaluation_seed=[])
        for seed in EVALUATION_SEEDS:
            a = arrays(output/'bridge'/f'eval-{seed}.npz')
            err = np.sum((a[f'prediction_{name}']-a['anchor_xyz'])**2, axis=-1)
            row['rmse_by_evaluation_seed'].append(dict(seed=seed, all_body_rmse_mm=rmse(err, a['anchor_body'])))
        result.append(row)
    q = data['q']
    frozen = BodyModel(output/'bridge'/'model-v7.pt')
    shuffled = q[np.random.default_rng(2026102492).permutation(len(q))]
    fixed_q = np.repeat(q[:1], len(q), axis=0)
    controls = {name: rmse(np.sum((frozen(values)-data['anchor_xyz'])**2, axis=-1), data['anchor_body'])
                for name, values in [('actual_q', q), ('shuffled_q', shuffled), ('constant_q', fixed_q)]}
    return result, controls


def occlusion_summary(output):
    data = join([arrays(output/'occlusion'/f'{s}.npz') for s in EVALUATION_SEEDS])
    result = []
    for mode in ('visual', 'feedback', 'fusion'):
        error = np.sum((data[f'prediction_{mode}']-data['anchor_xyz'])**2, axis=-1)
        row = dict(mode=mode, all_body_rmse_mm=rmse(error, data['anchor_body']),
                   panel_hidden_body_rmse_mm=rmse(error, data['behind_panel']),
                   visible_body_rmse_mm=rmse(error, data['anchor_body'] & data['anchor_visible']))
        if mode != 'feedback':
            events = [e for seed in EVALUATION_SEEDS for e in read_json(output/'occlusion'/f'{seed}-events.json')[mode]]
            accepted = data[f'accepted_{mode}']
            row.update(events=len(events), recovered=sum(e['recovered'] for e in events),
                recovery_rate=sum(e['recovered'] for e in events)/max(len(events), 1),
                median_recovery_latency_frames=float(np.median([e['latency_frames'] for e in events if e['recovered']])) if any(e['recovered'] for e in events) else None,
                false_reappearance_observations=sum(e['false_accepted'] for e in events),
                accepted_reappearance_observations=sum(e['accepted'] for e in events),
                pixel_body_purity=float((accepted & data[f'on_body_{mode}']).sum()/max(accepted.sum(), 1)),
                accepted_point_observations=int(accepted.sum()))
        result.append(row)
    protocol = dict(total_frames=len(data['q']), hidden_point_frames=int(data['behind_panel'].sum()),
                    frames_with_occluded_body=int(data['behind_panel'].any(1).sum()),
                    maximum_hidden_fraction=float(data['behind_panel'].sum(1).max()/data['anchor_body'][0].sum()),
                    maximum_occluded_body_pixel_fraction=float(data['occluded_body_pixel_fraction'].max()),
                    seconds_per_frame=.1, point_tolerance_m=.02, recovery_window_frames=5,
                    hidden_min_frames=3,
                    scoring='Same physical reference surface point, not any point on the arm')
    return result, protocol


def exploration_summary(output):
    scores = [r for seed in EVALUATION_SEEDS for r in read_json(output/'exploration'/f'eval-{seed}.json')]
    final = [r for r in scores if r['budget'] == 112]
    per_seed = []
    for seed in ACQUISITION_SEEDS:
        for policy in POLICIES:
            rows = [r for r in final if r['acquisition_seed'] == seed and r['policy'] == policy]
            per_seed.append(dict(seed=seed, policy=policy,
                mean_body_rmse_mm=float(np.mean([r['all_body_rmse_mm'] for r in rows])),
                covered_body_points=rows[0]['covered_body_points'],
                correspondence_precision=rows[0]['correspondence_precision'],
                normalized_travel=rows[0]['normalized_travel'],
                rotary_travel_rad=rows[0]['rotary_travel_rad'], gripper_travel_m=rows[0]['gripper_travel_m']))
    policies = []
    for policy in POLICIES:
        rows = [r for r in per_seed if r['policy'] == policy]
        policies.append(dict(policy=policy,
            mean_body_rmse_mm=float(np.mean([r['mean_body_rmse_mm'] for r in rows])),
            acquisition_seed_sd_mm=float(np.std([r['mean_body_rmse_mm'] for r in rows], ddof=1)),
            covered_body_points=float(np.mean([r['covered_body_points'] for r in rows])),
            correspondence_precision=float(np.mean([r['correspondence_precision'] for r in rows])),
            normalized_travel=float(np.mean([r['normalized_travel'] for r in rows])),
            rotary_travel_rad=float(np.mean([r['rotary_travel_rad'] for r in rows])),
            gripper_travel_m=float(np.mean([r['gripper_travel_m'] for r in rows]))))
    differences = []
    for seed in ACQUISITION_SEEDS:
        active = next(r for r in per_seed if r['seed'] == seed and r['policy'] == 'active')
        random = next(r for r in per_seed if r['seed'] == seed and r['policy'] == 'random')
        differences.append(active['mean_body_rmse_mm']-random['mean_body_rmse_mm'])
    rng = np.random.default_rng(2026102491)
    samples = rng.choice(differences, (10000, len(differences)), replace=True).mean(1)
    comparison = dict(active_minus_random_mm=float(np.mean(differences)),
        active_better_acquisition_seeds=int(np.sum(np.asarray(differences) < 0)), acquisition_seeds=5,
        acquisition_seed_bootstrap_95_ci_mm=np.quantile(samples, [.025, .975]).tolist(),
        seed_differences_mm=differences,
        inference_scope='Five acquisition seeds; three evaluation trajectories averaged within each acquisition seed, not 15 independent replicates')
    return scores, per_seed, policies, comparison


def plot_results(output, summary, scores):
    fig, axs = plt.subplots(2, 3, figsize=(16, 9), layout='constrained')
    colors = ['#74879a', '#c58d49', '#208b83']
    bridge = summary['bridge']
    labels = ['No mask', 'V6 mask', 'V7 mask']
    for i, row in enumerate(bridge):
        axs[0, 0].bar(i, row['all_body_rmse_mm'], color=colors[i])
        axs[0, 1].bar(i-.18, row['pixel_body_purity']*100, width=.35, color=colors[i])
        axs[0, 1].bar(i+.18, row['raw_body_observations_retained']*100, width=.35, color=colors[i], alpha=.35)
    axs[0, 0].set(title='Spatial predictions: ALL fixed body anchors', ylabel='RMSE / mm', xticks=range(3), xticklabels=labels)
    axs[0, 1].set(title='Opaque: pixel purity | light: retained body observations', ylabel='Percent', xticks=range(3), xticklabels=labels, ylim=(0, 105))
    data = arrays(output/'occlusion'/f'{EVALUATION_SEEDS[0]}.npz')
    time = np.arange(len(data['q']))*.1
    for mode, color in zip(('visual', 'feedback', 'fusion'), colors):
        error = np.linalg.norm(data[f'prediction_{mode}']-data['anchor_xyz'], axis=-1)
        value = np.sqrt(np.mean(error[:, data['anchor_body'][0]]**2, axis=1))*1000
        axs[0, 2].plot(time, value, label=mode, color=color)
    hidden = data['behind_panel'].sum(1)/data['anchor_body'][0].sum()
    ax2 = axs[0, 2].twinx()
    ax2.fill_between(time, 0, hidden*100, color='#74879a', alpha=.12)
    ax2.set(ylabel='Body anchors behind fixed screen / %', ylim=(0, 105))
    axs[0, 2].set(title='Actual occlusion: first evaluation seed', xlabel='Time / s', ylabel='All-body RMSE / mm')
    axs[0, 2].legend()
    for policy, color in zip(POLICIES, ['#208b83', '#c58d49', '#74879a']):
        matrix = []
        for budget in BUDGETS:
            matrix.append([np.mean([r['all_body_rmse_mm'] for r in scores if r['budget'] == budget and r['policy'] == policy and r['acquisition_seed'] == s]) for s in ACQUISITION_SEEDS])
        matrix = np.asarray(matrix)
        axs[1, 0].plot(BUDGETS, matrix.mean(1), '-o', color=color, label=policy)
        axs[1, 0].fill_between(BUDGETS, matrix.mean(1)-matrix.std(1), matrix.mean(1)+matrix.std(1), color=color, alpha=.12)
        rows = [r for r in summary['exploration_per_seed'] if r['policy'] == policy]
        axs[1, 1].scatter([r['normalized_travel'] for r in rows], [r['mean_body_rmse_mm'] for r in rows], color=color, label=policy)
        index = POLICIES.index(policy)
        axs[1, 2].bar(index, np.mean([r['covered_body_points'] for r in rows]), color=color)
        axs[1, 2].scatter(index+np.linspace(-.12, .12, len(rows)), [r['covered_body_points'] for r in rows], color='#233d4d', s=18)
    baseline = np.mean([r['baseline_body_rmse_mm'] for r in scores])
    axs[1, 0].axhline(baseline, color='#c54949', ls='--', label='frozen shared decoder')
    axs[1, 0].set(title='Same decoder + online residual learning', xlabel='Acquisition observations', ylabel='All-body RMSE / mm')
    axs[1, 0].legend(fontsize=8)
    axs[1, 1].set(title='Five acquisition seeds: cost vs spatial error', xlabel='Actual normalized joint path length', ylabel='All-body RMSE / mm')
    axs[1, 1].legend()
    axs[1, 2].set(title='Physical anchors observed correctly >=4 times', ylabel='Covered anchors / 28', xticks=range(3), xticklabels=POLICIES, ylim=(0, 29))
    for ax in axs.ravel():
        ax.grid(axis='y', alpha=.15)
    fig.suptitle('E4 | Fixed props | 28 physical body anchors | 5 acquisition x 3 evaluation seeds', fontsize=16)
    fig.savefig(output/'report'/'results.png', dpi=160)
    plt.close(fig)


def animation(output):
    seed = EVALUATION_SEEDS[0]
    a = arrays(output/'occlusion'/f'{seed}.npz')
    source = output/f'occlusion-data-{seed}'
    meta = read_json(source/'metadata.json')
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16)
    small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 13)
    frames = []
    for i in range(0, len(a['q']), 2):
        with np.load(source/'observations'/meta['frames'][i]['file']) as obs:
            rgb = Image.fromarray(obs['left']).resize((320, 240), Image.Resampling.LANCZOS)
        canvas = Image.new('RGB', (1280, 300), '#12222c')
        draw = ImageDraw.Draw(canvas)
        for column, mode in enumerate(('rgb', 'visual', 'feedback', 'fusion')):
            canvas.paste(rgb, (column*320, 35))
            draw.text((column*320+10, 9), {'rgb':'Camera / fixed screen', 'visual':'Vision / hold missing points',
                      'feedback':'Learned feedback decoder', 'fusion':'RGB + learned motion prior'}[mode], font=font, fill='#e0ecf4')
            if mode != 'rgb':
                uv = project(a[f'prediction_{mode}'][i], meta['calibration'])/2
                for j in np.flatnonzero(a['anchor_body'][i]):
                    u, v = uv[j]
                    if 1 < u < 319 and 1 < v < 239:
                        x, y = column*320+float(u), 35+float(v)
                        colour = '#f6c15c' if a['behind_panel'][i, j] else '#88d8ed'
                        draw.ellipse((x-2, y-2, x+2, y+2), fill=colour)
                if mode != 'feedback':
                    for j in np.flatnonzero(a[f'accepted_{mode}'][i] & a['anchor_body'][i]):
                        p = project(a[f'observed_{mode}'][i, j:j+1], meta['calibration'])[0]/2
                        u, v = p
                        if 1 < u < 319 and 1 < v < 239:
                            x, y = column*320+float(u), 35+float(v)
                            draw.ellipse((x-3, y-3, x+3, y+3), outline='#67f391')
        draw.text((12, 281), f't={i*.1:4.1f}s | {int(a["behind_panel"][i].sum())}/28 anchors behind screen | yellow: hidden prediction | green ring: admitted RGB', font=small, fill='#c4d7e5')
        frames.append(canvas)
    frames[0].save(output/'report'/'occlusion.gif', save_all=True, append_images=frames[1:], duration=200, loop=0, optimize=True)
    frames[len(frames)//2].save(output/'report'/'preview.png')


def table(headers, rows):
    return '<table><thead><tr>'+''.join(f'<th>{html.escape(str(v))}</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'


def build(output):
    read_json(output/'complete.json')
    report = output/'report'
    report.mkdir(exist_ok=True)
    bridge, feedback_controls = bridge_summary(output)
    occlusion, occlusion_protocol = occlusion_summary(output)
    scores, per_seed, policies, comparison = exploration_summary(output)
    meta = [read_json(output/f'{kind}-{seed}'/'metadata.json') for seed in EVALUATION_SEEDS for kind in ('evaluation', 'occlusion-data')]
    summary = dict(created_utc=utc(), bridge=bridge, feedback_controls_rmse_mm=feedback_controls, occlusion=occlusion,
        actual_occlusion=occlusion_protocol, exploration=policies, exploration_per_seed=per_seed,
        active_random_comparison=comparison, acquisition_seeds=list(ACQUISITION_SEEDS), evaluation_seeds=list(EVALUATION_SEEDS),
        decoder_sha256=sha(output/'bridge'/'model-v7.pt'),
        prop_max_displacement_m=max(m['prop_max_displacement_m'] for m in meta),
        protocol='41 common candidate IDs; 28 reference rays hit physical body; all 28 retained in spatial-error denominators',
        limitations=['Fixed camera/appearance, one local simulation envelope, ideal calibrated stereo and joint feedback',
            'Inherited E2 feature IDs, reference image and learned weights; this is incremental learning, not discovery from zero',
            'Reference-patch matching, spatial gating and state memory are engineered; no learned permanence/re-ID objective',
            'Physical reference anchors use simulator geometry only in evaluation, never in masks/training/policy/fusion',
            'Mask purity measures observed pixel ownership; correspondence precision additionally requires original surface identity',
            'Covariance is not a calibrated probability, 20mm recovery threshold is a declared local tolerance',
            'Five acquisition seeds are independent replicates; three evaluation seeds share scene/appearance and are averaged within replicates',
            'Sparse anchors do not measure complete surface or volume coverage; no hardware validation'])
    test_log = output/'unittest-final.log'
    if test_log.exists():
        log = test_log.read_text(encoding='utf-8')
        match = re.search(r'Ran (\d+) tests', log)
        summary['validation'] = dict(unit_tests_passed=int(match[1]) if match and '\nOK' in log else None,
                                    test_log='../unittest-final.log')
    if (output/'scoring-correction.json').exists():
        summary['scoring_correction'] = read_json(output/'scoring-correction.json')
    write_json(report/'summary.json', summary)
    write_json(report/'exploration-scores.json', scores)
    with (report/'exploration-scores.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, list(scores[0]))
        writer.writeheader()
        writer.writerows(scores)
    plot_results(output, summary, scores)
    animation(output)
    btable = table(['分割辅助', '身体像素纯度', '保留的身体观测', '正确对应覆盖', '全部28点 RMSE', '相机中断 RMSE'],
        [[{'none':'不加分割', 'v6':'V6', 'v7':'V7'}[r['mode']], f'{r["pixel_body_purity"]:.1%}', f'{r["raw_body_observations_retained"]:.1%}',
          f'{r["correctly_matched_visible_body_coverage"]:.1%}', f'{r["all_body_rmse_mm"]:.2f} mm', f'{r["blind_body_rmse_mm"]:.2f} mm'] for r in bridge])
    otable = table(['方案', '全部28点 RMSE', '屏幕后点 RMSE', '重现后恢复', '错误接纳次数 / 接纳次数'],
        [[{'visual':'视觉＋失踪点保持', 'feedback':'仅学习到的反馈模型', 'fusion':'视觉＋反馈融合'}[r['mode']],
          f'{r["all_body_rmse_mm"]:.2f} mm', f'{r["panel_hidden_body_rmse_mm"]:.2f} mm',
          f'{r["recovered"]}/{r["events"]} ({r["recovery_rate"]:.1%})' if 'events' in r else '无视觉对应，不能报告恢复率',
          f'{r["false_reappearance_observations"]}/{r["accepted_reappearance_observations"]}' if 'events' in r else '—'] for r in occlusion])
    etable = table(['策略', '空间 RMSE ± 采集种子SD', '覆盖的身体点 /28', '正确对应率', '实测归一化路程', '旋转关节总路程'],
        [[{'active':'主动', 'random':'随机', 'preset':'预设'}[r['policy']], f'{r["mean_body_rmse_mm"]:.2f} ± {r["acquisition_seed_sd_mm"]:.2f} mm',
          f'{r["covered_body_points"]:.1f}', f'{r["correspondence_precision"]:.1%}', f'{r["normalized_travel"]:.1f}', f'{r["rotary_travel_rad"]:.1f} rad'] for r in policies])
    stable = '仍包含零，优势尚未得到稳定支持' if comparison['acquisition_seed_bootstrap_95_ci_mm'][0] <= 0 <= comparison['acquisition_seed_bootstrap_95_ci_mm'][1] else '未包含零，但仅覆盖当前局部仿真分布'
    fusion_score = next(r for r in occlusion if r['mode'] == 'fusion')
    mask_effect = '改善了三维误差' if bridge[-1]['all_body_rmse_mm'] < bridge[0]['all_body_rmse_mm'] else '未改善三维误差'
    correction = summary.get('scoring_correction', {}).get('counts', [])
    old_events = sum(r['previous_event_count'] for r in correction)
    new_events = sum(r['revised_event_count'] for r in correction)
    passed = summary.get('validation', {}).get('unit_tests_passed')
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>E4 · 持续身体表示的三项实测</title><style>
*{{box-sizing:border-box}}body{{background:#101b24;color:#e1eaf1;font:16px/1.75 system-ui,'Microsoft YaHei',sans-serif;margin:0}}main{{max-width:1240px;margin:auto;padding:32px 22px 70px}}h1{{font-size:32px}}h2{{font-size:23px}}p,li{{color:#b9ccd9}}a{{color:#7bddbc}}section{{background:#172a36;border:1px solid #34515f;border-radius:12px;padding:23px;margin:24px 0}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:11px 8px;border-bottom:1px solid #3b5364;text-align:left}}img{{width:100%;height:auto}}.note{{padding:14px;background:#293946;border-left:3px solid #dbaa59}}.scroll{{overflow-x:auto}}small{{color:#92aebe}}</style>
<main><small>E4 / 新训练与冻结评价 / 原有桌面物体保持固定</small><h1>从身体分割，走到可检验的持续身体表示</h1>
<p>三件事都已实际运行：分割辅助空间模型、真正的固定遮挡与对应恢复、共享空间模型的主动探索。空间误差始终包含同样的 28 个真实参考身体点；观测丢失也不会从分母中删掉。</p>
<p class="note">本轮结论：V7 {mask_effect}；遮挡后原点恢复 {fusion_score['recovered']}/{fusion_score['events']} 次；主动减随机误差 {comparison['active_minus_random_mm']:+.2f} mm，差值区间{stable}。当前最明确的短板是视觉对应的恢复，尚不能宣称模型已经学会持续身体身份。</p>
<p><a href="../../">返回实验入口</a> · <a href="summary.json">完整指标 JSON</a> · <a href="exploration-scores.csv">逐种子与预算 CSV</a></p>
<section><h2>1 · 更好的分割是否带来更好的三维身体表示</h2><p>从相同 E2 权重出发，在全新固定物体数据上用三种观测掩码训练，各 300 轮。414 帧训练/验证，随后在三条新轨迹共 486 帧上评价。41 个候选点输出一直保留；评价参考射线中 28 个真正落在机械臂表面。</p>
<div class="scroll">{btable}</div><p>共享 V7 辅助空间模型的反馈对照：实际 q 为 {feedback_controls['actual_q']:.2f} mm，打乱 q 为 {feedback_controls['shuffled_q']:.2f} mm，恒定 q 为 {feedback_controls['constant_q']:.2f} mm。它检验三维预测对反馈的依赖，不能替代身体归属和恒存验证。</p><p class="note">纯度提高必须连同观测保留率与固定分母的空间误差解释。只删除难点、换到更容易的评价点，不能算表示改善。这里的“正确对应覆盖”要求误差不超过 20 mm，仅衡量这 28 个点，不能代表完整表面覆盖。</p></section>
<section><h2>2 · 机械臂进入固定遮挡、继续运动、再出现</h2><p>新增一块固定的非接触遮挡板；原有三个桌面物体保持原位，最大位移 {summary['prop_max_displacement_m']:.1e} m。3 条不同偏置的 140 帧轨迹，共 {occlusion_protocol['frames_with_occluded_body']}/{occlusion_protocol['total_frames']} 帧实际发生身体点遮挡，最严重时遮住 {occlusion_protocol['maximum_hidden_fraction']:.1%} 的参考身体点。评分同时渲染同姿态下隐藏遮挡板的画面，排除机械臂自身遮挡；屏幕最多遮住 {occlusion_protocol['maximum_occluded_body_pixel_fraction']:.1%} 的身体像素。这些画面只供评价。</p>
<img src="occlusion.gif" alt="固定遮挡板下，RGB、视觉保持、反馈预测、融合表示的实际运行对比">
<div class="scroll">{otable}</div><p>三个方案从共同的已学 HOME 空间先验起步；视觉方案随后只追踪并保持最后看到的位置。恢复事件：同一物理表面点被遮挡至少 3 帧，重新可见后 5 帧内，视觉重新测得原始点的 3D 位置，误差不超过 20 mm。只靠反馈继续输出位置不会获得视觉恢复分数。错误接纳次数保留在表中。</p>
<p class="note">融合使用学到的 q→3D 运动模型引导参考图像对应，再做显式空间门控与状态更新。对应恢复是可测的新证据，但记忆管理和匹配规则仍由程序设计；这不等于网络已经自主学会物体恒存。</p></section>
<section><h2>3 · 同一个身体模型下，主动探索是否值得</h2><p>三种策略复用上面冻结的 V7 辅助空间解码器、同样的 41 个候选点和在线残差学习器。每次 112 帧，前 16 帧为同种子的共同随机起步。5 个采集种子 × 3 个策略，共 15 次完整采集；全部冻结后才采集 3 条评价轨迹。路程由物理仿真过程中每 10 ms 的实际反馈累计。</p>
<div class="scroll">{etable}</div><p>主动减随机的空间误差差值：{comparison['active_minus_random_mm']:+.2f} mm；主动在 {comparison['active_better_acquisition_seeds']}/5 个采集种子上更好。按采集种子重采样的 95% 区间 [{comparison['acquisition_seed_bootstrap_95_ci_mm'][0]:+.2f}, {comparison['acquisition_seed_bootstrap_95_ci_mm'][1]:+.2f}] mm，{stable}。三条评价轨迹先在每个采集种子内平均，未把它们当成 15 次独立采集。</p>
<img src="results.png" alt="分割桥接纯度与覆盖、实际遮挡误差、主动探索误差与运动代价六张实测图"></section>
<section><h2>当前证据的范围</h2><p>这是一套共享稀疏身体状态的实际学习与评价闭环。它复用了 E2 已发现的特征和预训练权重，使用理想关节反馈与已标定双目，仍限定在同一场景、相机和局部动作范围。完整形态、身体拓扑、实体体积和真机泛化尚未得到验证。</p><p>仿真几何、身体标签、深度、隐藏图像只用于隔离的评分文件，不进入分割、空间训练、融合门控或动作选择。模型的协方差是工程估计，不能当作已校准的存在概率。</p><p>评分检查修正了一处漏计：被屏幕遮挡后又被自身遮挡的点，也必须等到真实重现后纳入恢复事件。分母从 {old_events} 增至 {new_events}，成功仍为 {fusion_score['recovered']}；模型、输入、预测和动作没有改变。原评分与<a href="../scoring-correction.json">修正记录</a>均保留。</p><p><a href="../unittest-final.log">{str(passed)+' 项测试通过' if passed else '查看测试记录'}</a>；已核对冻结权重、全部 15 次固定物体采集和相机中断时 RGB 清零。</p></section></main></html>'''
    (report/'index.html').write_text(page, encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ('bridge', 'occlusion', 'exploration', 'active_random_comparison')}, ensure_ascii=False, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/local-check-20261024/e4-body-cycle')
    build(Path(parser.parse_args().output))
