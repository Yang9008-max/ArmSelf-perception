"""E10 complete report, paired timelines and a local interactive replay."""
from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import trust_cycle as e7
from .body_cycle import read_json, write_json, sha, utc
from .body_memory import BodyModel
from .conflict_cycle import (ROOT, FINAL, CONDITIONS, MODES, AUDITS, WINDOWS,
                             active_mask, binary_metrics, verify)
from .surface_extent import SurfaceAtlas
from . import extent_cycle_v2 as e9
from .vision import project

CN = dict(clean='传感器正常', feedback_bias='反馈偏差', stale_rgb='过去图像', both='偏差＋过去图像',
          no_feedback='反馈缺失', no_camera='相机缺失', coherent_replay='反馈＋图像协同回放')
LABEL = dict(full='E8B 完整', no_visual='E8B 去视觉', no_feedback='E8B 去反馈',
             action='仅动作', feedback='仅反馈／缺失时递推', fixed_equal='固定等权')
STATUS = ['候选相容', '动作候选不一致', '反馈候选不一致', '视觉候选不一致', '多解释未决', '证据不足']


def summarize(output):
    score = read_json(output/'scores.json')
    rows = read_json(output/'frame-scores.json')
    events = read_json(output/'recovery-events.json')
    conditions = {}
    for condition in CONDITIONS:
        result = {}
        for mode in MODES:
            chosen = [r for r in score['rows'] if r['condition'] == condition and r['mode'] == mode]
            item = dict(rmse_mm=float(np.sqrt(np.mean([r['rmse_mm']**2 for r in chosen]))),
                        fault_rmse_mm=float(np.sqrt(np.mean([r['fault_rmse_mm']**2 for r in chosen]))),
                        off_surface_rate=float(np.mean([r['off_surface_rate'] for r in chosen])))
            if mode == 'full':
                for k in ('joint_interval_coverage', 'fault_joint_coverage', 'envelope_coverage',
                          'half_width_normalized', 'envelope_half_width_normalized'):
                    item[k] = float(np.mean([r[k] for r in chosen]))
            selected_events = [e for e in events if e['condition'] == condition and e['mode'] == mode]
            for key in ('convergence', 'actual_error'):
                successful = [e[key]['time_s'] for e in selected_events if e[key]['recovered']]
                item[key] = dict(events=len(selected_events), recovered=len(successful),
                    censored=len(selected_events)-len(successful),
                    median_success_time_s=float(np.median(successful)) if successful else None,
                    # Conservative restricted mean includes 4 s for censored events.
                    restricted_mean_time_s=float(np.mean([e[key]['time_s'] if e[key]['recovered'] else 4.
                                                          for e in selected_events])) if selected_events else None)
            result[mode] = item
        conditions[condition] = result
    diagnostic = {}
    full = [r for r in rows if r['mode'] == 'full']
    for audit in AUDITS:
        by_condition = {}
        for condition in CONDITIONS:
            subset = [r for r in full if r['condition'] == condition]
            detect = binary_metrics([r[f'conflict_{audit}'] for r in subset], [r['fault_window'] for r in subset])
            outliers = [r for r in subset if r[f'status_{audit}'] in (1, 2, 3)]
            correct = 0
            harmful = 0
            for r in outliers:
                rejected = r[f'status_{audit}']-1
                correct += int(r['bad_candidates'][rejected])
                others = [e for j, e in enumerate(r['candidate_error_m'])
                          if j != rejected and r['candidate_available'][j]]
                harmful += int(bool(others) and r['candidate_error_m'][rejected]+.002 < min(others))
            impactful = [any(r['bad_candidates']) for r in subset]
            by_condition[condition] = dict(**detect, frames=len(subset), outlier_frames=len(outliers),
                outlier_precision=correct/max(1, len(outliers)), outlier_coverage=len(outliers)/len(subset),
                harmful_rejection_rate=harmful/max(1, len(outliers)), harmful_rejection_frames=harmful,
                unknown_rate=float(np.mean([r[f'unknown_{audit}'] for r in subset])),
                impactful_detection=binary_metrics([r[f'conflict_{audit}'] for r in subset], impactful))
        primary = ('feedback_bias', 'stale_rgb', 'both', 'coherent_replay')
        chosen = [r for r in full if r['condition'] in primary]
        overall = binary_metrics([r[f'conflict_{audit}'] for r in chosen], [r['fault_window'] for r in chosen])
        overall['clean_false_alarm'] = float(np.mean([r[f'conflict_{audit}'] for r in full if r['condition'] == 'clean']))
        overall['macro_f1'] = float(np.mean([by_condition[c]['f1'] for c in primary]))
        outlier_n = sum(x['outlier_frames'] for x in by_condition.values())
        overall['outlier_precision'] = sum(x['outlier_precision']*x['outlier_frames'] for x in by_condition.values())/max(1, outlier_n)
        overall['outlier_coverage'] = outlier_n/len(full)
        diagnostic[audit] = dict(overall=overall, conditions=by_condition)
    primary = ('feedback_bias', 'stale_rgb', 'both', 'coherent_replay')
    full_mean = np.mean([conditions[c]['full']['fault_rmse_mm'] for c in primary])
    fixed_mean = np.mean([conditions[c]['fixed_equal']['fault_rmse_mm'] for c in primary])
    improvement = float(1-full_mean/fixed_mean)
    action_mean = float(np.mean([conditions[c]['action']['fault_rmse_mm'] for c in primary]))
    guards = []
    for seed in FINAL:
        c = {r['mode']:r['rmse_mm'] for r in score['rows'] if r['seed'] == seed and r['condition'] == 'clean'}
        guards.append(dict(seed=seed, regression_mm=c['full']-c['feedback'], passed=c['full'] <= c['feedback']+.5))
    interval_pass = all(conditions[c]['full']['joint_interval_coverage'] >= .85 for c in CONDITIONS)
    recovery_pass = {}
    for c in CONDITIONS[1:]:
        r = conditions[c]['full']['convergence']
        recovery_pass[c] = r['recovered'] >= 5 and r['median_success_time_s'] is not None and r['median_success_time_s'] <= 1.
    goals = dict(state_benefit=dict(passed=bool(improvement >= .1 and all(x['passed'] for x in guards)),
                    relative_improvement=improvement, normal_guards=guards),
        uncertainty=dict(passed=interval_pass, minimum_coverage=.85),
        outlier_precision=dict(passed=diagnostic['body']['overall']['outlier_precision'] >= .8,
                              value=diagnostic['body']['overall']['outlier_precision'],
                              coverage=diagnostic['body']['overall']['outlier_coverage']),
        geometry_extra_value=dict(passed=bool(diagnostic['body']['overall']['macro_f1'] >= diagnostic['joint']['overall']['macro_f1']+.05
            and diagnostic['body']['overall']['clean_false_alarm'] <= .05),
            macro_f1_gain=diagnostic['body']['overall']['macro_f1']-diagnostic['joint']['overall']['macro_f1']),
        recovery=dict(passed=all(recovery_pass.values()), conditions=recovery_pass))
    result = dict(created_utc=utc(), seeds=FINAL, conditions=conditions, diagnostics=diagnostic, goals=goals,
        strong_baseline=dict(four_fault_macro_full_mm=float(full_mean), four_fault_macro_action_mm=action_mean,
                             full_regression_vs_action=float(full_mean/action_mean-1)),
        original_models_frozen=True, explicit_audit_changes_state=False,
        audit_outputs_are_not_fault_probabilities=True, correlated_frontend=True,
        protocol='body-conflict-e10.md')
    write_json(output/'report'/'summary.json', result)
    return result, rows, events


def figures(report, summary, rows):
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei', 'DejaVu Sans'], 'axes.unicode_minus':False})
    x = np.arange(len(CONDITIONS))
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    for offset, mode in enumerate(('full', 'action', 'feedback', 'fixed_equal')):
        y = [summary['conditions'][c][mode]['fault_rmse_mm'] for c in CONDITIONS]
        ax.bar(x+(offset-1.5)*.19, y, .19, label=LABEL[mode])
    ax.set(xticks=x, xticklabels=[CN[c] for c in CONDITIONS], ylabel='干预窗口固定锚点 RMSE / mm',
           title='同一物理轨迹、共同锚点：故障窗口中的身体状态误差')
    ax.tick_params(axis='x', labelsize=9)
    ax.legend(ncol=4)
    fig.tight_layout(); fig.savefig(report/'errors.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for key, label in [('joint_interval_coverage', '原 E8B 名义 90% 区间'), ('envelope_coverage', '显式候选包络（非概率）')]:
        axes[0].plot(x, [summary['conditions'][c]['full'][key]*100 for c in CONDITIONS], 'o-', label=label)
    axes[0].axhline(90, color='gray', ls='--'); axes[0].set(ylabel='同时七关节覆盖 / %', ylim=(0, 103))
    for key, label in [('half_width_normalized', 'E8B'), ('envelope_half_width_normalized', '候选包络')]:
        axes[1].plot(x, [summary['conditions'][c]['full'][key] for c in CONDITIONS], 'o-', label=label)
    axes[1].set(ylabel='平均归一化半宽（越宽越保守）')
    for ax in axes:
        ax.set_xticks(x, [CN[c] for c in CONDITIONS], rotation=22, ha='right', fontsize=8)
        ax.legend(fontsize=9)
    fig.suptitle('覆盖必须结合区间宽度判断；包络扩张不能等同学会可信度')
    fig.tight_layout(); fig.savefig(report/'uncertainty.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for audit, label in [('body', '身体几何＋时间'), ('joint', '关节残差＋时间'), ('body_instant', '身体几何，无平滑')]:
        d = summary['diagnostics'][audit]
        axes[0].plot(x[1:], [d['conditions'][c]['f1'] for c in CONDITIONS[1:]], 'o-', label=label)
        axes[1].plot(x, [d['conditions'][c]['harmful_rejection_rate']*100 for c in CONDITIONS], 'o-', label=label)
    axes[0].set(ylabel='干预窗口检测 F1', ylim=(0, 1.05), xticks=x[1:])
    axes[1].set(ylabel='离群判定中错误排除更好候选 / %', xticks=x)
    for ax in axes:
        ax.set_xticklabels([CN[c] for c in (CONDITIONS[1:] if ax == axes[0] else CONDITIONS)], rotation=22, ha='right', fontsize=8)
        ax.legend(fontsize=8)
    fig.suptitle('发现不一致不等于找到了故障来源')
    fig.tight_layout(); fig.savefig(report/'audit.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))
    for k, mode in enumerate(('full', 'no_visual', 'no_feedback', 'fixed_equal')):
        values = [summary['conditions'][c][mode]['convergence']['recovered'] for c in CONDITIONS[1:]]
        axes[0].bar(np.arange(6)+(k-1.5)*.19, values, .19, label=LABEL[mode])
        y = [summary['conditions'][c][mode]['convergence']['restricted_mean_time_s'] for c in CONDITIONS[1:]]
        axes[1].plot(np.arange(6), y, 'o-', label=LABEL[mode])
    axes[0].set(ylabel='收敛恢复事件 / 6', ylim=(0, 6.8))
    axes[1].set(ylabel='受限平均确认时间 / s（未恢复计 4s）', ylim=(0, 4.2))
    for ax in axes:
        ax.set_xticks(np.arange(6), [CN[c] for c in CONDITIONS[1:]], rotation=22, ha='right', fontsize=8)
        ax.legend(fontsize=8)
    fig.suptitle('故障解除后恢复：连续五帧，未恢复事件保留在分母')
    fig.tight_layout(); fig.savefig(report/'recovery.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for condition in ('clean', 'both', 'coherent_replay'):
        chosen = [r for r in rows if r['seed'] == FINAL[0] and r['condition'] == condition and r['mode'] == 'full']
        axes[0].plot(np.arange(180)*.1, [r['anchor_rmse_mm'] for r in chosen], label=CN[condition])
        axes[1].plot(np.arange(180)*.1, [r['half_width_normalized'] for r in chosen], label=CN[condition])
        axes[2].plot(np.arange(180)*.1, [r['separation_mm'] for r in chosen], label=CN[condition])
    for ax in axes:
        for start, stop in WINDOWS:
            ax.axvspan(start*.1, stop*.1, color='orange', alpha=.15)
        ax.legend(fontsize=8)
    axes[0].set(ylabel='锚点误差 / mm'); axes[1].set(ylabel='区间归一化半宽')
    axes[2].set(ylabel='与正常分支差距 / mm', xlabel='时间 / s；橙色为干预窗口')
    fig.tight_layout(); fig.savefig(report/'timeline.png', dpi=150); plt.close(fig)


def replay(output, report, rows):
    seed = FINAL[0]
    source = output/f'data-{seed}'
    meta = read_json(source/'metadata.json')
    atlas = SurfaceAtlas.load(e9.ROOT/'atlas-multi.npz')
    cal = meta['calibration']
    indices = np.linspace(0, len(atlas.weights)-1, 110, dtype=int)
    (report/'frames').mkdir(exist_ok=True)
    ids = list(range(0, 180, 2))
    for i in sorted(set(ids+list(range(20)))):
        obs = e7.load_npz(source/'after'/meta['frames'][i]['file'])
        Image.fromarray(obs['left']).resize((480, 360)).save(report/'frames'/f'{i:03d}.jpg', quality=83)
    result = dict(seed=seed, labels=CN, modes=LABEL, statuses=STATUS, conditions={})
    lookup = {(r['condition'], r['frame'], r['mode']):r for r in rows if r['seed'] == seed}
    def uv(q):
        points = project(atlas.predict(q)[indices], cal)*.75
        return np.clip(np.nan_to_num(points), -2000, 2000).round(1).tolist()
    for condition in CONDITIONS:
        p = e7.load_npz(output/'predictions'/f'{seed}-{condition}.npz')
        frames = []
        for i in ids:
            score = lookup[condition, i, 'full']
            wrong = active_mask()[i] and condition in ('stale_rgb', 'both', 'coherent_replay')
            missing = active_mask()[i] and condition == 'no_camera'
            truth = e7.load_npz(source/'evaluation'/meta['frames'][i]['file'])
            true_points = project(truth['anchor_xyz'][truth['anchor_body']], cal)*.75
            frames.append(dict(i=i, current=f'frames/{i:03d}.jpg', image=None if missing else f'frames/{i%20 if wrong else i:03d}.jpg',
                wrong_image=bool(wrong), error=score['anchor_rmse_mm'], interval=score['half_width_normalized'],
                envelope=score['envelope_half_width_normalized'], status=int(p['status_body'][i]),
                unknown=bool(p['unknown_body'][i]), conflict=bool(p['conflict_body'][i]),
                covered=score['joint_covered'], separation=score['separation_mm'],
                weights=p['weights'][i].mean(0).round(3).tolist(),
                points={m:uv(p[f'q_{m}'][i]) for m in MODES},
                truth=np.clip(true_points, -2000, 2000).round(1).tolist()))
        result['conditions'][condition] = frames
    write_json(report/'replay.json', result)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 18)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 15)
    animation = []
    frames = result['conditions']['coherent_replay']
    for j, row in enumerate(frames):
        image = Image.new('RGB', (960, 430), '#101c29')
        image.paste(Image.open(report/row['current']), (0, 38))
        if row['image'] is not None:
            image.paste(Image.open(report/row['image']), (480, 38))
        draw = ImageDraw.Draw(image)
        draw.text((12, 7), '当前真实相机＋身体预测（绿色）', font=font, fill='white')
        draw.text((492, 7), '模型收到的相机：'+('过去图像' if row['wrong_image'] else '当前图像'), font=font, fill='white')
        for x, y in row['points']['full']:
            if 0 <= x < 480 and 0 <= y < 360:
                draw.ellipse((x-2, y+36, x+2, y+40), fill='#53ef95')
        for x, y in row['truth']:
            if 0 <= x < 480 and 0 <= y < 360:
                draw.ellipse((x-2, y+36, x+2, y+40), outline='#66bfff', width=1)
        draw.text((12, 403), f"t={row['i']*.1:.1f}s  误差={row['error']:.2f}mm  {STATUS[row['status']]}  蓝色：评价锚点", font=small, fill='white')
        animation.append(image)
    animation[0].save(report/'body-conflict.gif', save_all=True, append_images=animation[1:], duration=160, loop=0)
    animation[25].save(report/'preview.jpg', quality=90)


def markdown(output, report, summary):
    s = summary
    strong = s['strong_baseline']
    lines = ['# E10：冲突、未知与恢复实验报告', '', '完成：2026-10-25（Asia/Shanghai）。', '',
        '**本轮是冻结模型的独立检验，没有重新训练，也没有把显式一致性审计伪装成模型学会的能力。**', '',
        '**结论：已有部分冲突抑制和观测恢复能力；错误来源识别仍不可靠，共同身体几何在本轮尚未显示足够额外收益。** 协同回放揭示两路同意仍会诱发错误离群判定；恢复与故障来源识别不能混为一项。', '',
        '[本地交互回放](http://127.0.0.1:8767/e10-body-conflict/report/) · [执行前协议](body-conflict-e10.md)', '',
        '## 做了什么', '',
        '开发轨迹 1 条，最终轨迹 3 条，每条 180 帧。原有干扰物固定，固定遮挡屏沿用。每条轨迹有两段未知执行延迟，七种观测条件共享同一物理过程，每段故障之后有明确恢复窗口；中途包含停止命令。', '',
        '冻结 E8B、动作网络、视觉前端和 E9B 的 434 片段。完整、去视觉、去反馈、仅动作、仅反馈、固定等权使用相同观测。动作递推与 RGB 前端相关，不能把它们当作独立多数票。', '',
        '新增显式读出：三个候选通过共同身体表面转换后比较残差；正常开发轨迹的 99% 分位定门限，2/3 帧确认。相同候选的关节距离审计和无平滑审计作为对照。读出没有修改状态。没有增加神经网络参数；不应把本轮称为重新训练优化成功。', '',
        '## 状态误差', '',
        '下表是干预窗口内固定物理身体锚点 RMSE，单位 mm；正常列同样含执行延迟，不是完全理想运动。', '',
        '| 条件 | 完整 E8B | 去视觉 | 去反馈 | 仅动作 | 仅反馈 | 等权 |', '|---|---:|---:|---:|---:|---:|---:|']
    for c in CONDITIONS:
        lines.append('| '+CN[c]+' | '+' | '.join(f"{s['conditions'][c][m]['fault_rmse_mm']:.3f}" for m in MODES)+' |')
    g = s['goals']
    lines += ['', f"四个错误观测条件的宏平均相对等权改善 **{g['state_benefit']['relative_improvement']*100:.2f}%**。正常逐种子保护："+
        '；'.join(f"{x['seed']} {x['regression_mm']:+.3f}mm" for x in g['state_benefit']['normal_guards'])+'。', '',
        f"但强对照仍未胜出：四条件宏平均完整为 **{strong['four_fault_macro_full_mm']:.3f}mm**，仅动作为 **{strong['four_fault_macro_action_mm']:.3f}mm**，完整高 **{strong['full_regression_vs_action']*100:.2f}%**。等权在协同回放中受到严重误导，因此相对它的大幅改善不能视为统一表示已胜过强基线。共同受损中完整也差于去视觉与去反馈。", '',
        '## 不确定性和恢复', '',
        '| 条件 | E8B 全段覆盖 | E8B 故障段覆盖 | 包络全段覆盖 | E8B 半宽 | 包络半宽 | 收敛恢复 /6 | 成功中位确认时间 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for c in CONDITIONS:
        r = s['conditions'][c]['full']; rec = r['convergence']
        timing = '—' if rec['median_success_time_s'] is None else f"{rec['median_success_time_s']:.2f}s"
        lines.append(f"| {CN[c]} | {r['joint_interval_coverage']*100:.2f}% | {r['fault_joint_coverage']*100:.2f}% | {r['envelope_coverage']*100:.2f}% | {r['half_width_normalized']:.4f} | {r['envelope_half_width_normalized']:.4f} | {rec['recovered']}/{rec['events']} | {timing} |")
    lines += ['', '半宽按训练关节尺度归一化。E8B 是旧校准的名义 90% 同时七关节区间；候选包络是显式保守集合，不具备 90% 概率含义。增宽包络提高覆盖只能说明保留更多解释，不能单独证明学会了可信度。', '',
        '当前运动比 E8 原有随机、短时距测试平滑，全段覆盖约 97%–100%，只通过预设最低覆盖保护，不能宣称概率精确校准。共同故障的第三个种子在故障段覆盖仅 84.29%；必须保留逐种子与故障段结果，不能用正常帧稀释失败。协同回放平均半宽约为正常的 6.7 倍，覆盖提升伴随保守性增加。', '',
        '恢复是相对同轨迹正常分支的身体预测距离连续五帧 ≤2mm；同时报告实际锚点误差相对正常分支 ≤2mm 的另一项恢复。确认时间包含五帧观察时间，最快 0.5s；未恢复保留分母，受限均值中计为 4s。恢复到正常分支仍继承正常分支误差，不表示绝对定位无误。', '',
        '## 共同身体表示是否提供额外帮助', '',
        '| 显式审计 | 四错误条件宏 F1 | 正常误报 | 离群解释确有 >8mm 状态影响 | 离群判定覆盖 |',
        '|---|---:|---:|---:|---:|']
    for k, label in [('body', '身体几何＋时间'), ('joint', '关节距离＋时间'), ('body_instant', '身体几何，无平滑')]:
        d = s['diagnostics'][k]['overall']
        lines.append(f"| {label} | {d['macro_f1']:.3f} | {d['clean_false_alarm']*100:.2f}% | {d['outlier_precision']*100:.2f}% | {d['outlier_coverage']*100:.2f}% |")
    coherent = s['diagnostics']['body']['conditions']['coherent_replay']
    lines += ['', f"协同回放中，有 {coherent['outlier_frames']} 帧给出唯一离群候选，其中 **{coherent['harmful_rejection_frames']} 帧（{coherent['harmful_rejection_rate']*100:.2f}%）排除了比其余候选都更接近真实状态的解释**。这直接检验两个受损通道一致时的错误确定性。", '',
        f"整体离群错误精度 {s['diagnostics']['body']['overall']['outlier_precision']*100:.2f}% 仅覆盖全部条件帧的 {s['diagnostics']['body']['overall']['outlier_coverage']*100:.2f}%；协同回放自身精度只有 {coherent['outlier_precision']*100:.2f}%。因此整体精度验收通过不能解释成逐条件可靠，更不能作为直接拒绝观测的依据。", '',
        '检测目标是已知干预窗口，不是唯一故障真相。报告也给出按候选误差定义的实质冲突检测。候选距离来自同一个 q 条件表面映射，不能自动证明身体身份或因果归属。正常误报与时序平滑、缺失信息、外观匹配局限分别保留。', '',
        '## 冻结验收', '', '| 项目 | 结果 |', '|---|---|']
    for key, label in [('state_benefit', '状态收益及正常保护'), ('uncertainty', '逐条件区间覆盖至少85%'),
                       ('outlier_precision', '离群解释错误精度至少80%'), ('geometry_extra_value', '几何相对关节审计额外收益'),
                       ('recovery', '每条件至少5/6事件且成功中位≤1s')]:
        lines.append(f"| {label} | {'通过' if g[key]['passed'] else '未通过'} |")
    lines += ['', '## 结果应怎样解释', '',
        '本轮将“发现不一致”“排除错误解释”“概率区间可信”“故障解除后恢复”分开，不能用其中一项替代其他项。审计显示的冲突／未知是显式算法输出，原 E8B 没有独立的来源故障分类头。', '',
        f"几何审计宏 F1 为 {s['diagnostics']['body']['overall']['macro_f1']:.3f}，关节审计为 {s['diagnostics']['joint']['overall']['macro_f1']:.3f}，只增 {s['goals']['geometry_extra_value']['macro_f1_gain']:.3f}，未达到预设 +0.05。正常误报较低，但没有足够证据将可靠性收益归因于更丰富的身体几何。几何审计在协同回放中的错误排除比关节审计更明显，增加表面点不自动提高故障可辨识性。", '',
        '普通故障恢复通过，只支持当前条件下有新观测时的状态收敛。协同回放 1/6 事件未在 4s 窗口恢复，其余成功事件中位确认 1.2s，未达 1s 目标；所有事件的实际锚点误差恢复则通过，说明状态回归与位置误差保护回答不同问题。', '',
        '下一步应针对相依证据与竞争解释设计真正更新身体状态的机制；只有独立新证据足以区分时才排除解释，并用新最终数据验证。不能把当前显式审计的离群结果直接接成硬门控，也不应在本次最终数据上调门限后宣称问题解决。', '',
        '尚未证明：完整身体发现、形态自适应、实机迁移、任意故障分布、完整实体身份及端到端实时性能。本轮是离线因果回放；视觉匹配由独立动作递推提供初始化，错误相关仍存在。', '',
        '## 数据与复现', '',
        '输出：`outputs/local-check-20261024/e10-body-conflict/`。`protocol-freeze.json` 固定源码、旧权重、协议、开发门限；`predictions-complete.json` 封存 21 份预测；`scores.json`、`frame-scores.json`、`recovery-events.json` 保留共同分母与删失。', '',
        '```powershell', r'.\.venv\Scripts\python.exe -X utf8 -m embodiment.conflict_cycle prepare',
        r'.\.venv\Scripts\python.exe -X utf8 -m embodiment.conflict_cycle final',
        r'.\.venv\Scripts\python.exe -X utf8 -m embodiment.conflict_cycle score',
        r'.\.venv\Scripts\python.exe -X utf8 -m embodiment.conflict_report', '```', '',
        '已经封存的阶段拒绝覆盖；复现请使用新的 `--output` 文件夹。所有最终预测封存后才打开特权评价数据。', '']
    text = '\n'.join(lines)
    (report/'详细报告.md').write_text(text, encoding='utf-8')
    Path('docs/body-conflict-e10-summary.md').write_text(text, encoding='utf-8')
    return text


def webpage(report, summary):
    rows = ''.join('<tr><td>'+CN[c]+'</td>'+''.join(f"<td>{summary['conditions'][c][m]['fault_rmse_mm']:.2f}</td>" for m in MODES)+'</tr>' for c in CONDITIONS)
    goal_names = dict(state_benefit='相对等权的状态收益及正常保护', uncertainty='逐条件最低区间覆盖（不代表概率校准）',
                      outlier_precision='总体离群错误精度（覆盖仅8.31%，协同回放失败）', geometry_extra_value='身体几何相对关节距离的额外收益', recovery='各条件恢复要求')
    goals = ''.join(f'<li>{html.escape(goal_names[k])}：<b>{"通过" if v["passed"] else "未通过"}</b></li>' for k,v in summary['goals'].items())
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>E10 · 冲突、未知与恢复</title>
<style>*{box-sizing:border-box}body{margin:0;background:#101b29;color:#e8f0f6;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1180px;margin:auto;padding:30px 20px 70px}h1{font-size:32px}h2{font-size:23px}section{background:#192b3d;border:1px solid #345069;border-radius:12px;padding:20px;margin:22px 0}a{color:#75dac3}p{color:#c4d3e1}.note{border-left:4px solid #edb75b;padding-left:16px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}canvas{width:100%;background:#070c12;border-radius:8px}img{width:100%;border-radius:8px}select,button{background:#28435a;color:white;border:1px solid #5c7991;padding:9px;border-radius:6px;margin:4px}input{width:100%}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:8px;border-bottom:1px solid #3d5265;text-align:left}.scroll{overflow-x:auto}#stats{font-variant-numeric:tabular-nums;white-space:pre-line}.legend{color:#8decb7}small{color:#a7bfd2}@media(max-width:750px){.grid{grid-template-columns:1fr}main{padding:16px}h1{font-size:26px}}</style>
<main><p><a href="/">← 实验总入口</a> · <a href="详细报告.md">详细报告</a> · <a href="summary.json">全部指标</a> · <a href="frame-scores.csv">逐帧数据</a></p><h1>冲突出现以后，身体认识能否恢复？</h1>
<p>冻结 E8B 与 E9B · 3 条新轨迹 · 7 种观测条件 · 两次故障与恢复 · 原有干扰物固定</p>
<p class="note">本轮没有重新训练。绿色身体由同一个状态生成；新增“候选不一致／未知”是显式诊断，尚未反馈修改状态。发现冲突、判断谁错、表达可信区间与恢复正确状态分别验收。</p>
<p class="note">实际结论：普通条件有恢复能力；协同回放中23次唯一离群判断有22次排除了更好的候选，且1/6事件未收敛恢复。完整模型仍差于仅动作强对照；身体几何相对关节距离的额外收益未达标。</p>
<section><h2>配对回放</h2><select id="condition"></select><select id="mode"></select><button id="play">暂停</button><label><input id="overlay" type="checkbox" checked style="width:auto">显示表面预测与评价锚点</label>
<div class="grid"><div><p>当前物理相机＋所选状态的身体预测</p><canvas id="actual" width="480" height="360"></canvas></div><div><p>提供给模型的相机（可能为过去图像或缺失）</p><canvas id="sensor" width="480" height="360"></canvas></div></div>
<input id="frame" type="range" min="0" max="89" value="0"><p id="stats">加载回放…</p><small class="legend">绿色：110 个展示用表面采样；蓝色：共同评价锚点。实际评分保留全部 434 表面片段和全部身体锚点。统计读数为完整 E8B，模型下拉框仅切换投影。</small></section>
<section><h2>状态、解释与置信的验收</h2><ul>GOALS</ul><p>未通过会原样保留。相对正常分支恢复仍继承正常误差；保守包络扩大不能单独证明学会了不确定性。</p></section>
<section><h2>干预窗口误差 / mm</h2><div class="scroll"><table><thead><tr><th>条件</th><th>完整</th><th>去视觉</th><th>去反馈</th><th>仅动作</th><th>仅反馈</th><th>等权</th></tr></thead><tbody>ROWS</tbody></table></div><img src="errors.png"></section>
<section><h2>区间覆盖与宽度同时看</h2><img src="uncertainty.png"></section><section><h2>相互同意不保证正确</h2><img src="audit.png"></section><section><h2>故障解除后的恢复</h2><img src="recovery.png"></section><section><h2>完整时间过程</h2><img src="timeline.png"></section>
<section><h2>本轮能回答的范围</h2><p>已知关节通道、干净初始化、固定双目相机和当前仿真机械臂。动作与视觉候选存在依赖。没有证明完整身体身份、任意故障识别、形态自适应或实机性能。源码和旧模型在最终采集前冻结，21 份预测全部封存后才评分。</p><p><a href="body-conflict.gif">协同回放动画</a> · <a href="recovery-events.csv">逐事件与未恢复记录</a></p></section></main>
<script>let data,playing=true,busy=false;const $=id=>document.getElementById(id);const images=new Map();function load(src){if(!src)return Promise.resolve(null);if(images.has(src))return images.get(src);const p=new Promise(resolve=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=()=>resolve(null);im.src=src;});images.set(src,p);return p;}
async function draw(){if(!data||busy)return;busy=true;try{const c=$('condition').value,m=$('mode').value,k=+$('frame').value;const r=data.conditions[c][k];const [a,b]=await Promise.all([load(r.current),load(r.image)]);for(const [id,im] of [['actual',a],['sensor',b]]){const ctx=$(id).getContext('2d');ctx.fillStyle='#070c12';ctx.fillRect(0,0,480,360);if(im)ctx.drawImage(im,0,0,480,360);else{ctx.fillStyle='#c4d3e1';ctx.font='20px sans-serif';ctx.fillText('相机缺失',180,180);}}if($('overlay').checked){const ctx=$('actual').getContext('2d');ctx.fillStyle='#53ef95';for(const [x,y] of r.points[m]){ctx.beginPath();ctx.arc(x,y,2,0,Math.PI*2);ctx.fill();}ctx.strokeStyle='#66bfff';for(const [x,y] of r.truth){ctx.beginPath();ctx.arc(x,y,3,0,Math.PI*2);ctx.stroke();}}$('stats').textContent=`t=${(r.i*.1).toFixed(1)}s / 帧 ${r.i} · ${data.statuses[r.status]} · ${r.conflict?'发现不一致':'尚无确认冲突'}\n完整模型锚点误差 ${r.error.toFixed(2)}mm；距正常分支 ${r.separation.toFixed(2)}mm；真实关节是否落在原区间：${r.covered?'是':'否'}\n融合贡献 动作/反馈/视觉：${r.weights.map(x=>(x*100).toFixed(1)+'%').join(' / ')}；半宽 原区间 ${r.interval.toFixed(4)} / 包络 ${r.envelope.toFixed(4)}\n输入图像：${r.image?(r.wrong_image?'过去回放':'当前'):'缺失'}。离群解释不是已确认故障。`;}finally{busy=false;}}
fetch('replay.json').then(r=>r.json()).then(x=>{data=x;for(const [k,v] of Object.entries(x.labels))$('condition').add(new Option(v,k));for(const [k,v] of Object.entries(x.modes))$('mode').add(new Option(v,k));$('condition').value='coherent_replay';draw();}).catch(e=>$('stats').textContent='加载失败：'+e.message);for(const id of ['condition','mode','frame','overlay'])$(id).addEventListener('input',draw);$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'暂停':'播放';};setInterval(()=>{if(playing&&data&&!busy){$('frame').value=(+$('frame').value+1)%90;draw();}},180);</script></html>'''
    page = page.replace('GOALS', goals).replace('ROWS', rows)
    (report/'index.html').write_text(page, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args()
    verify(args.output)
    report = args.output/'report'; report.mkdir(exist_ok=True)
    (report/'body-conflict-e10.md').write_text(Path('docs/body-conflict-e10.md').read_text(encoding='utf-8'), encoding='utf-8')
    summary, rows, events = summarize(args.output)
    for name, records in [('frame-scores.csv', rows), ('recovery-events.csv', events)]:
        keys = list(dict.fromkeys(k for row in records for k in row))
        with (report/name).open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, keys); writer.writeheader(); writer.writerows(records)
    figures(report, summary, rows)
    replay(args.output, report, rows)
    markdown(args.output, report, summary)
    webpage(report, summary)
    write_json(args.output/'report-artifacts.json', dict(created_utc=utc(),
        hashes={str(p):sha(p) for p in report.rglob('*') if p.is_file()},
        generator_sha256=sha(Path(__file__))))
    print('E10 report complete:', report, flush=True)
    print({k:v['passed'] for k,v in summary['goals'].items()}, flush=True)


if __name__ == '__main__':
    main()
