"""E9B delivery; the first E9 experiment remains preserved."""
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

from .body_cycle import read_json, write_json, sha, utc
from . import trust_cycle as e7
from .extent_cycle_v2 import ROOT, FINAL, CONDITIONS, CONFIG, verify, primitive_surface_distance
from .surface_extent import SurfaceAtlas
from .vision import project

LABELS = dict(sparse='旧 41 候选点', single='单参考图谱', budget_matched='多参考，等点数',
    multi='多参考完整图谱 + E8B', static='保持初始身体', mismatch='反向错配反馈',
    e7='完整图谱 + E7', feedback='完整图谱 + 反馈', action='完整图谱 + 动作', fixed='完整图谱 + 等权融合')
CN = dict(clean='正常', biased_feedback='反馈偏差', wrong_rgb='错误图像',
    bias_and_wrong_rgb='反馈与图像共同受损', no_feedback='缺反馈', camera_missing='相机缺失')
GOALS = dict(coverage_gain='相比旧点覆盖提高至少 10 个百分点',
    off_surface_protection='离体点增幅不超过 2 个百分点',
    mean_surface_distance='平均表面距离不超过 20 mm',
    full_multi_over_single='完整多参考比单参考覆盖提高至少 3 个百分点且保持离体保护',
    equal_output_count_multi_over_single='等点数多参考比单参考覆盖提高至少 3 个百分点且保持离体保护')


def aggregate(rows):
    def total(key):
        return sum(r[key] for r in rows)
    return dict(frames=len(rows), point_count=rows[0]['point_count'],
        coverage=total('covered_count')/max(total('target_count'), 1),
        visible_coverage=total('visible_covered')/max(total('visible_count'), 1),
        hidden_coverage=total('hidden_covered')/max(total('hidden_count'), 1),
        surface_distance_mm=1000*total('surface_distance_sum_m')/max(total('point_count'), 1),
        off_surface=total('off_surface_count')/max(total('point_count'), 1),
        outside=total('outside_count')/max(total('point_count'), 1),
        prop_fraction=total('prop_point_count')/max(total('point_count'), 1),
        target_distance_mm=1000*total('target_distance_sum_m')/max(total('target_count'), 1),
        stereo_supported=total('observed_count')/max(total('point_count'), 1),
        unknown_support=total('unknown_support_count')/max(total('point_count'), 1))


def pct(value):
    return f'{100*value:.2f}%'


def diagnostic(output):
    result = {}
    for condition in CONDITIONS:
        count, bad, residual, hidden_count, false_current = 0, 0, [], 0, 0
        for seed in FINAL:
            a = e7.load_npz(output/'extent-predictions'/f'{seed}-{condition}.npz')
            source = output/f'data-{seed}'
            metadata = read_json(source/'metadata.json')
            for i, frame in enumerate(metadata['frames']):
                ok = a['status'][i] == 2
                points = a['observed_xyz'][i][ok]
                if not len(points):
                    continue
                truth = e7.load_npz(source/'evaluation'/frame['file'])
                distance, _ = primitive_surface_distance(points, truth)
                count += len(points)
                bad += int((distance > .015).sum())
                residual.extend(distance.tolist())
                uv = np.rint(project(a['multi'][i][ok], metadata['calibration'])).astype(int)
                inside = (uv[:, 0] >= 0) & (uv[:, 0] < 640) & (uv[:, 1] >= 0) & (uv[:, 1] < 480)
                ids = np.flatnonzero(inside)
                u, v = uv[ids].T
                hidden = truth['panel_mask'][v, u] & (truth['depth'][v, u] < a['multi'][i][ok][ids, 2]-.015)
                hidden_count += int(hidden.sum())
                if condition == 'camera_missing' and 40 <= i < 95:
                    false_current += len(points)
        result[condition] = dict(stereo_supported_samples=count, off_surface_samples=bad,
            off_surface_fraction=bad/max(count, 1), mean_surface_distance_mm=1000*float(np.mean(residual)) if residual else None,
            matches_projecting_behind_panel=hidden_count, matches_while_camera_absent=false_current,
            interpretation='matcher support is not a calibrated current-body correctness probability')
    return result


def make_figures(report, rows, summary):
    modes = ['sparse', 'single', 'budget_matched', 'multi']
    labels = ['41 points', 'Single chart', 'Multi, same count', 'Multi, all']
    clean = summary['conditions']['clean']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    colors = ['#8b99ac', '#32a59a', '#5479d5', '#e39b3c']
    quantities = [('coverage', 'Clear-body surface coverage (%)', 100),
                  ('off_surface', 'Samples >15 mm from body (%)', 100),
                  ('surface_distance_mm', 'Mean surface distance (mm)', 1)]
    for ax, (key, title, factor) in zip(axes, quantities):
        ax.bar(labels, [clean[m][key]*factor for m in modes], color=colors)
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis='x', labelrotation=20, labelsize=9)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('E9: three held-out trajectories; common 15 mm tolerance')
    fig.tight_layout()
    fig.savefig(report/'comparison.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(CONDITIONS))
    for mode, color in [('multi', '#e19a37'), ('e7', '#5c7bd3'), ('feedback', '#2b9c88'), ('action', '#9a74b8'), ('fixed', '#76818d')]:
        axes[0].plot(x, [100*summary['conditions'][c][mode]['coverage'] for c in CONDITIONS], 'o-', label=mode, color=color)
        axes[1].plot(x, [summary['conditions'][c][mode]['surface_distance_mm'] for c in CONDITIONS], 'o-', label=mode, color=color)
    for ax, title in zip(axes, ['Same atlas: surface coverage (%)', 'Same atlas: surface distance (mm)']):
        ax.set_xticks(x, ['Clean', 'Bias', 'Wrong RGB', 'Bias+RGB', 'No F', 'No camera'], rotation=20)
        ax.set_title(title)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(report/'state-stress.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for mode, color in [('sparse', '#8b99ac'), ('single', '#32a59a'), ('multi', '#e39b3c')]:
        a = [r for r in rows if r['seed'] == FINAL[0] and r['condition'] == 'clean' and r['mode'] == mode]
        axes[0].plot([r['frame'] for r in a], [100*r['covered_count']/r['target_count'] for r in a], label=mode, color=color)
    a = [r for r in rows if r['seed'] == FINAL[0] and r['condition'] == 'clean' and r['mode'] == 'multi']
    axes[1].plot([r['frame'] for r in a], [100*r['observed_count']/r['point_count'] for r in a], color='#32a59a', label='Current stereo-supported patches')
    axes[1].plot([r['frame'] for r in a], [100*r['hidden_count']/r['target_count'] for r in a], color='#9368bc', label='Body surface hidden by fixed panel')
    for ax in axes:
        ax.axvspan(70, 89, color='#8b99ac', alpha=.17, label='Stop command')
        ax.set_ylabel('Percent')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set_title(f'New motion, stop, fixed-panel occlusion: seed {FINAL[0]}')
    axes[1].set_xlabel('Frame')
    fig.tight_layout()
    fig.savefig(report/'timeline.png', dpi=160)
    plt.close(fig)


def animation(output, report):
    seed, condition = FINAL[0], 'clean'
    source = output/f'data-{seed}'
    cal = read_json(source/'metadata.json')['calibration']
    a = e7.load_npz(output/'extent-predictions'/f'{seed}-{condition}.npz')
    atlas = SurfaceAtlas.load(output/'atlas-multi.npz')
    frames = []
    font_path = 'C:/Windows/Fonts/msyh.ttc'
    font = ImageFont.truetype(font_path, 17)
    small = ImageFont.truetype(font_path, 14)
    for i in range(0, 150, 5):
        obs = e7.load_npz(source/'after'/f'{i:05d}.npz')
        canvas = Image.new('RGB', (1280, 536), '#112133')
        canvas.paste(Image.fromarray(obs['left']), (0, 48))
        canvas.paste(Image.fromarray(obs['left']), (640, 48))
        draw = ImageDraw.Draw(canvas)
        draw.text((15, 10), f'旧 41 候选点 · 新轨迹帧 {i}', font=font, fill='white')
        draw.text((655, 10), 'E9 多姿态表面图谱 · 同一 E8B 状态', font=font, fill='white')
        for x, y in project(a['sparse'][i], cal):
            if np.isfinite(x+y):
                draw.ellipse((x-2, y+46, x+2, y+50), fill='#ec9a46')
        uv = project(a['multi'][i], cal)
        for (x, y), status in zip(uv, a['status'][i]):
            if np.isfinite(x+y) and 0 <= x < 640 and 0 <= y < 480:
                color = '#40d9b3' if status == 2 else '#f1b34c' if status == 1 else '#b5b8c1'
                draw.ellipse((640+x-1.5, y+46.5, 640+x+1.5, y+49.5), fill=color)
        draw.text((15, 515), '固定干扰物及固定遮挡屏；左图候选点也可能含误认', font=small, fill='#d3e0ed')
        draw.text((655, 515), '绿：匹配支持　橙：模型预测　灰：姿态支持不足；未标区域未知', font=small, fill='#d3e0ed')
        frames.append(canvas)
    frames[0].save(report/'body-extent.gif', save_all=True, append_images=frames[1:], duration=160, loop=0)
    frames[12].save(report/'preview.jpg', quality=92)
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(projection='3d')
    point = a['multi'][60]
    ax.scatter(point[:, 0], point[:, 2], -point[:, 1], c=atlas.saved['chart'], cmap='tab10', s=6, alpha=.7)
    ax.set(xlabel='Camera X (m)', ylabel='Camera Z (m)', zlabel='Camera up (m)',
           title='Registered surface samples, colored by reference chart\nGaps and interior remain unknown; no CAD used for registration')
    ax.set_box_aspect(np.maximum(np.ptp(point[:, [0, 2, 1]], axis=0), .01))
    fig.tight_layout()
    fig.savefig(report/'surface-3d.png', dpi=160)
    plt.close(fig)


def generate(output):
    freeze = verify(output)
    assert (output/'complete.json').exists()
    report = output/'report'
    report.mkdir(exist_ok=True)
    rows = read_json(output/'frame-scores.json')
    modes = list(LABELS)
    by = {c:{m:aggregate([r for r in rows if r['condition'] == c and r['mode'] == m]) for m in modes} for c in CONDITIONS}
    seeds = {str(s):{m:aggregate([r for r in rows if r['seed'] == s and r['condition'] == 'clean' and r['mode'] == m]) for m in modes} for s in FINAL}
    phases = {p:aggregate([r for r in rows if r['condition'] == 'clean' and r['mode'] == 'multi' and r['phase'] == p]) for p in sorted(set(r['phase'] for r in rows))}
    baseline, multi, single, matched = (by['clean'][k] for k in ('sparse', 'multi', 'single', 'budget_matched'))
    goals = dict(coverage_gain=multi['coverage']-baseline['coverage'] >= .10,
        off_surface_protection=multi['off_surface']-baseline['off_surface'] <= .02,
        mean_surface_distance=multi['surface_distance_mm'] <= 20.,
        full_multi_over_single=multi['coverage']-single['coverage'] >= .03 and multi['off_surface']-single['off_surface'] <= .02,
        equal_output_count_multi_over_single=matched['coverage']-single['coverage'] >= .03 and matched['off_surface']-single['off_surface'] <= .02)
    atlas = SurfaceAtlas.load(output/'atlas-multi.npz')
    home = e7.load_npz(output/f'data-{FINAL[0]}'/'initial.npz')['q']
    points = atlas.predict(home)
    unique = len(np.unique(np.floor(points/.003).astype(int), axis=0))
    previous_summary = read_json(report/'summary.json') if (report/'summary.json').exists() else {}
    manifest = read_json(output/'predictions-complete.json')
    for path,digest in manifest['hashes'].items():
        assert sha(Path(path)) == digest
    cached_diagnostics = previous_summary.get('diagnostics') if previous_summary.get('model_sha256') == freeze['model_hashes'] else None
    paired = read_json(output/'paired-registration'/'summary.json') if (output/'paired-registration'/'summary.json').exists() else None
    summary = dict(created_utc=utc(), selected_patches=freeze['selected_count'], unique_3mm_cells_at_initial_pose=unique,
        conditions=by, per_seed_clean=seeds, clean_phases=phases, goals=goals,
        diagnostics=cached_diagnostics if cached_diagnostics is not None else diagnostic(output),
        posthoc_paired_registration=paired, registration=read_json(output/'development.json')['counts_by_chart'],
        model_sha256=freeze['model_hashes'], generator_sha256=sha(Path(__file__)),
        limitations=['one morphology and calibrated stereo pair', 'one old training trajectory, three new test trajectories',
            'surface points, not watertight volume', 'explicit patch registration and quadratic regression',
            'shared feedback coordinates and clean initialization', 'matching support not calibrated probability',
            'candidate duplicates and additional search budget', 'same synthetic fault families as E7/E8',
            'offline causal replay, not end-to-end real-time control'])
    write_json(report/'summary.json', summary)
    with (report/'metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        fieldnames = ['condition', 'mode', *list(baseline)]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in CONDITIONS:
            for m in modes:
                writer.writerow(dict(condition=c, mode=m, **by[c][m]))
    make_figures(report, rows, summary)
    animation(output, report)
    header = '| 方法 | 点数 | 覆盖 | 屏挡表面覆盖 | 平均表面距离 mm | 离体点比例 |\n|---|---:|---:|---:|---:|---:|\n'
    table = header+''.join(f"| {LABELS[m]} | {by['clean'][m]['point_count']} | {pct(by['clean'][m]['coverage'])} | {pct(by['clean'][m]['hidden_coverage'])} | {by['clean'][m]['surface_distance_mm']:.3f} | {pct(by['clean'][m]['off_surface'])} |\n" for m in modes)
    fault_table = '| 条件 | E8B 覆盖 | E7 覆盖 | 反馈覆盖 | 动作覆盖 | 等权覆盖 | E8B 表面距离 mm | E8B 离体点 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'+''.join(f"| {CN[c]} | {pct(by[c]['multi']['coverage'])} | {pct(by[c]['e7']['coverage'])} | {pct(by[c]['feedback']['coverage'])} | {pct(by[c]['action']['coverage'])} | {pct(by[c]['fixed']['coverage'])} | {by[c]['multi']['surface_distance_mm']:.3f} | {pct(by[c]['multi']['off_surface'])} |\n" for c in CONDITIONS)
    seed_table = '| 最终种子 | 旧点覆盖 | 新图谱覆盖 | 新图谱离体点 | 新图谱表面距离 mm |\n|---|---:|---:|---:|---:|\n'+''.join(f"| {s} | {pct(seeds[str(s)]['sparse']['coverage'])} | {pct(seeds[str(s)]['multi']['coverage'])} | {pct(seeds[str(s)]['multi']['off_surface'])} | {seeds[str(s)]['multi']['surface_distance_mm']:.3f} |\n" for s in FINAL)
    phase_table = '| 正常轨迹阶段 | 新图谱覆盖 | 屏挡部分覆盖 | 离体点 |\n|---|---:|---:|---:|\n'+''.join(f"| {p} | {pct(v['coverage'])} | {pct(v['hidden_coverage'])} | {pct(v['off_surface'])} |\n" for p,v in phases.items())
    paired_text = ''
    if paired:
        pa,pb = paired['pooled']['earlier_atlas'],paired['pooled']['later_atlas']
        paired_text = f'''## 事后同状态配对诊断

两版冻结图谱同时接到第二轮相同 E8B 状态：首轮图谱覆盖 {pct(pa['coverage'])}、表面距离 {pa['surface_distance_mm']:.3f} mm、离体点 {pct(pa['off_surface'])}；严格图谱覆盖 {pct(pb['coverage'])}、距离 {pb['surface_distance_mm']:.3f} mm、离体点 {pct(pb['off_surface'])}。这是减少部分覆盖换取更干净证据的取舍，不是所有指标都改善。配对分析在主要最终评分之后增加，不用于参数选择，不替代预设测试；文件在 paired-registration 中。

'''
    verdict = '全部主要目标通过' if all(goals[k] for k in ('coverage_gain','off_surface_protection','mean_surface_distance')) else '主要目标部分通过；未通过项不能宣称解决'
    text = f'''# E9B 身体表面范围实验：详细报告

{verdict}。正常条件下，覆盖从旧 41 点的 {pct(baseline['coverage'])} 变为 {pct(multi['coverage'])}，增加 {(multi['coverage']-baseline['coverage'])*100:.2f} 个百分点；离体点比例从 {pct(baseline['off_surface'])} 变为 {pct(multi['off_surface'])}。

## 做了什么

把原来只输出固定 41 个点的身体表示扩展为多姿态表面图谱。复用 E4 的 414 帧固定干扰物数据作为本轮训练，间隔两帧使用 207 帧；用状态归一化距离最远采样选 6 个参考姿态。在全图按 80×60 像素瓦片、6 像素间距、每瓦片最多 60 个片段均匀取样，不以身体掩膜限定区域。双目三角化、11×11 光流及往返一致性获得三维轨迹。

候选发现先比较关节条件二次回归与恒定／错配反馈。7 维归一化反馈产生 36 维二次特征，每个点有 36×3 个回归系数，岭系数 0.08。训练内部按每第五帧留出，只用于初筛。新增参考片段五点双目检查：中心及上下左右 2 像素至少四次有效，与参考三维位置中位差 <6 mm，邻域三维散布 <12 mm。独立开发种子 {2026102941} 的整条轨迹进一步要求至少 8 次匹配、三维预测 RMS 小于 6 mm、优于恒定位置。该选择只使用 RGB 和实际反馈，身体真值、深度、CAD、连杆拓扑和 V7 身体掩膜都没有进入新图谱学习。输入归一化沿用既有身体模型的反馈通道和尺度。

最终保留 {multi['point_count']} 个片段，各图谱数量为 {summary['registration']}，回归系数 {multi['point_count']*108:,} 个；单参考保留 {single['point_count']} 个。片段并非 {multi['point_count']} 个独立身体部分：在初始姿态按 3 mm 空间单元去重仅有 {unique} 个单元。多参考用更多参考图像及搜索预算，等点数对照只控制输出数，没有消除搜索预算差异；本轮用训练初始姿态预测位置做最远点采样，减少重复覆盖，索引在最终前冻结。

本轮由 E9A 聚合失败启发，首轮的模型、报告和预测保留在 e9-body-extent。E9A 的覆盖提高但离体保护失败，等点数多参考覆盖也未超过单参考。因此本轮新增严格 RGB 表面一致性和空间采样；开发及最终都换了新种子。两个实验的绝对分数不能直接作为配对优化收益，判断各自在同批旧点／单参考／状态基线上是否通过协议。

## 怎样与动作—反馈—视觉连接

冻结 E8B 先按命令预测，再由反馈和旧视觉前端更新同一个 7 维身体状态；新表面图谱把这个状态映射成全部表面点。E7、仅反馈、仅动作与等权融合使用同一个新图谱，因此可以观察状态错误是否传到身体范围。新图谱不修改 E8B，也未启用它开发较差的偏差状态分支。

当前双目片段匹配通过且与预测相距小于 25 mm，才标为“当前匹配支持”；只有模型输出时标为“预测”；与训练姿态最近归一化距离超过 1.25 时标为支持未知。图谱外查询返回未知，不以包围盒填充身体。这个支持阈值是显式规则，不是新校准概率。新片段的当前测量用于核验与来源标记，没有再次更新 E8B 状态；E8B 的视觉更新仍来自旧匹配前端。

## 独立最终结果

三个新种子 {FINAL}，各 150 帧，在模型与源代码冻结后采集。三个原干扰物每帧位置完全一致，固定遮挡屏也不移动。所有最终预测保存并哈希之后才打开评价真值。共同目标集合为移除屏后从左相机可见的活动身体全部表面，每 3 像素采样；不是只评已选成功点。所有方法用相同 15 mm 三维最近点覆盖标准；离体指距实体表面超过 15 mm，内部很深的错误点也计入。表面距离利用独立评价中的箱体／圆柱并集近似距离，重叠形状处不是严格网格测地距离。

{table}

预设验收：{goals}。多参考完整图谱比单参考覆盖差为 {(multi['coverage']-single['coverage'])*100:.2f} 个百分点；等点数多参考比单参考差为 {(matched['coverage']-single['coverage'])*100:.2f} 个百分点。需要同时看离体比例，不能把更多点等同于更准确。

{seed_table}

{phase_table}

三个种子来自相同相机和机械臂设置，不是三个不同形态。大量相关帧不能当作同样数量的独立重复；本轮列出逐轨迹结果，没有虚构统计显著性。停止阶段为相同目标保持命令，最初若干帧可能仍在收敛。

离体点比例是三维几何接近程度，不是语义身体纯度：离身体不足 15 mm 的错误对象可能通过这一容差，预测落到另一块身体表面也可能通过。因此它不能单独证明本我发现或精确身体点身份。覆盖按共同目标表面点数加权，逐帧共同目标数保存在 frame-scores.json，不会删除未覆盖区域。

本轮最近表面距离不能与 E8 固定对应点 RMSE 直接比较；两个指标的对象与分母不同。

{paired_text}

## 可靠状态怎样影响范围

{fault_table}

故障仍来自既有 E7/E8 家族。新轨迹采用 0.1 秒采样及平滑动作、停止和固定屏遮挡，不能将其绝对误差直接与 E8 的随机短时命令实验排名。冻结的 E8B 不保证在新设置胜过简单反馈或等权融合。

## 已观测、预测与未知的边界

正常条件中 {pct(multi['stereo_supported'])} 的片段帧具有当前匹配支持。匹配支持仍可能错误：独立评分发现这些测量有 {pct(summary['diagnostics']['clean']['off_surface_fraction'])} 距真实身体表面超过 15 mm。错误图像更不能因为匹配通过就宣称当前身体被看到了。应结合 summary.json 的逐条件诊断阅读，匹配标记并非真实性证明。

本轮对“实体”的进展是增加了从自身运动获得、可随身体状态变化的表面证据；不是完整封闭体积，不是连杆语义分割，不是完整六自由度位姿，也不是主观自我。未见背面、内部及无纹理区域保留未知。精确旧点身份恢复和真实硬件都没有在本轮解决。

## 复现与文件

运行 `python -m embodiment.extent_cycle_v2 train`，冻结后 `python -m embodiment.extent_cycle_v2 final`，最后 `python -m embodiment.extent_report_v2`。已完成目录不可当空目录重跑；要复现实验应传入新的 --output 路径。训练只打开 observations / after，评分才打开 evaluation。协议、训练输入哈希、候选筛选、邻域检查、开发结果、模型冻结、全部预测哈希和逐帧分母保存在上级目录。复用的是 E9A 原始训练 RGB 轨迹缓存；E9A 最终数据没有进入拟合或选择。

本报告图表来自冻结输出，无额外模型拟合。测试、输入权限及旧模型哈希核验另见 audit.json。展示为离线因果回放，图像处理、匹配和绘图耗时未形成实时控制性能证明。
'''
    (report/'详细报告.md').write_text(text, encoding='utf-8')
    def html_table(data, keys):
        body = ''.join('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in r)+'</tr>' for r in data)
        return '<table><thead><tr>'+''.join(f'<th>{k}</th>' for k in keys)+'</tr></thead><tbody>'+body+'</tbody></table>'
    main_table = html_table([[LABELS[m], by['clean'][m]['point_count'], pct(by['clean'][m]['coverage']),
        pct(by['clean'][m]['hidden_coverage']), f"{by['clean'][m]['surface_distance_mm']:.3f}", pct(by['clean'][m]['off_surface'])] for m in modes],
        ['方法', '点数', '覆盖', '屏挡表面覆盖', '平均表面距离 mm', '离体点'])
    state_table = html_table([[CN[c], *[pct(by[c][m]['coverage']) for m in ('multi','e7','feedback','action','fixed')],
        pct(by[c]['multi']['off_surface'])] for c in CONDITIONS], ['条件', 'E8B 覆盖', 'E7', '反馈', '动作', '等权', 'E8B 离体点'])
    goal_table = html_table([[GOALS[k], '通过' if v else '未通过'] for k,v in goals.items()], ['预设目标', '结果'])
    paired_html = ''
    if paired:
        paired_table = html_table([[name,pct(paired['pooled'][key]['coverage']),
            f"{paired['pooled'][key]['surface_distance_mm']:.3f}",pct(paired['pooled'][key]['off_surface'])]
            for key,name in [('earlier_atlas','首轮 877 点'),('later_atlas','严格注册 434 点')]],
            ['同一 E8B 状态的图谱','覆盖','表面距离 mm','离体点'])
        paired_html = '<section><h2>同批状态：覆盖与错误的取舍</h2>'+paired_table+'<p>这是最终评分之后补充的配对诊断，未用于参数选择。严格注册减少部分覆盖，换来更干净的表面证据；不能宣称所有指标都改善。</p></section>'
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>E9 身体表面范围</title>
<style>body{{background:#0d1929;color:#deebfa;font:16px/1.8 system-ui,"Microsoft YaHei",sans-serif;max-width:1180px;margin:36px auto;padding:0 22px}}h1,h2{{line-height:1.4}}a{{color:#63ddc5}}section{{background:#15253a;padding:24px;border-radius:14px;margin:24px 0}}img{{max-width:100%;border-radius:8px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px 7px;border-bottom:1px solid #35445a;text-align:left}}.hero{{font-size:22px;color:#72e2ca}}.muted{{color:#a8b9cf}}.scroll{{overflow:auto}}</style>
<a href="../../">实验入口</a> · <a href="../../e9-body-extent/report/">首轮结果与失败</a><h1>E9B：更严格的身体表面范围</h1>
<p class="hero">正常表面覆盖 {pct(baseline['coverage'])} → {pct(multi['coverage'])}；离体点 {pct(baseline['off_surface'])} → {pct(multi['off_surface'])}</p>
<p>{verdict}。三条新轨迹，固定干扰物，固定遮挡屏，同一身体状态。</p>
<p><a href="详细报告.md">详细中文报告</a> · <a href="metrics.csv">完整指标 CSV</a> · <a href="summary.json">逐种子与故障结果 JSON</a></p>
<section><h2>表面证据如何随身体移动</h2><img src="body-extent.gif"><p class="muted">绿表示当前匹配支持，橙表示状态预测；未标区域未知。屏后的点可以由模型预测，不能算作被相机直接看见。</p></section>
<section><h2>覆盖、离体错误一起看</h2><img src="comparison.png"><div class="scroll">{main_table}</div></section>
<section><h2>同一个身体状态，解释更多表面</h2><img src="surface-3d.png"><p>6 个参考姿态，{multi['point_count']} 个注册片段；单参考 {single['point_count']} 个。片段坐标由关节反馈条件回归学习。多个片段可能重复覆盖同一表面，初始姿态按 3 mm 去重为 {unique} 个空间单元。</p><p>发现阶段没有身体掩膜、仿真深度、CAD 或连杆拓扑。它仍是显式视觉注册与小型回归组成的模型。</p></section>
<section><h2>停止、遮挡与重现</h2><img src="timeline.png"><p>预测点保留在共同分母中；匹配失败不删除困难身体区域。遮挡屏不移动，机械臂自己穿过遮挡。</p></section>
<section><h2>状态错误会传到身体范围</h2><img src="state-stress.png"><div class="scroll">{state_table}</div><p>本轮未调 E8B。它在新轨迹未必优于简单反馈；共同受损仍需单独看待。</p></section>
{paired_html}
<section><h2>验收与当前边界</h2>{goal_table}<p>这是一份部分可见表面的图谱。背面、内部与未见区域未知，包围范围不等于被身体填满的体积。当前匹配支持也不是校准过的身体真实性概率。</p><p>正常条件中通过当前匹配的测量，仍有 {pct(summary['diagnostics']['clean']['off_surface_fraction'])} 距真实身体表面超过 15 mm。全部失败和逐种子结果均保留在详细报告与 JSON 中。</p><p class="muted">一次旧训练采集、独立开发、三个新最终种子；固定相机和形态；离线因果回放，尚非实机或端到端实时证明。</p></section></html>'''
    (report/'index.html').write_text(page, encoding='utf-8')
    write_json(output/'report-artifacts.json', dict(created_utc=utc(), generator_sha256=sha(Path(__file__)),
        hashes={str(p.relative_to(output)):sha(p) for p in sorted(report.iterdir()) if p.is_file()}))
    print('E9 report', {k:v for k,v in goals.items()}, flush=True)
    print('Clean', {m:by['clean'][m] for m in ('sparse','single','budget_matched','multi')}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT)
    generate(parser.parse_args().output)
