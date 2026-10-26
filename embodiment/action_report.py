"""Report E6 prospective predictions, sensor ablations and negative results."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .action_cycle import ROOT, FINAL_SEEDS, CONDITIONS, load_npz
from .body_cycle import read_json, write_json, sha, utc
from .vision import project

MODES = ('feedback_only', 'action_only', 'action_feedback', 'no_feedback', 'no_action', 'full')
LABELS = dict(feedback_only='仅反馈', action_only='仅动作递推', action_feedback='动作＋反馈',
              no_feedback='动作＋视觉', no_action='反馈＋视觉（无动作）', full='动作＋反馈＋视觉')
CONDITION_LABELS = dict(clean='正常观测', biased_feedback='反馈偏差', delayed_feedback='反馈延迟',
                        camera_missing='相机中断', wrong_rgb='错误图像')
COLORS = dict(feedback_only='#dc9639', action_only='#5186b0', action_feedback='#9585ae',
              no_feedback='#679f93', no_action='#c07378', full='#007d70')


def aggregate(rows):
    result = {}
    for key in rows[0]:
        if key == 'by_duration':
            result[key] = {duration: float(np.sqrt(np.mean([r[key][duration]**2 for r in rows]))) for duration in rows[0][key]}
        elif key.endswith('frames') or key.endswith('updates'):
            result[key] = sum(r[key] for r in rows)
        else:
            result[key] = float(np.sqrt(np.mean([r[key]**2 for r in rows])))
    return result


def summarize(output):
    per_seed = [read_json(output/'scores'/f'data-{seed}.json') for seed in FINAL_SEEDS]
    pairs = [read_json(output/f'paired-{seed}/scores.json') for seed in FINAL_SEEDS]
    result = dict(completed_utc=utc(), seeds=list(FINAL_SEEDS), total_frames=360, same_state_action_pairs=36,
        prospective={mode: aggregate([r['prospective'][mode] for r in per_seed]) for mode in per_seed[0]['prospective']},
        paired={mode: aggregate([r['modes'][mode] for r in pairs]) for mode in pairs[0]['modes']},
        conditions={condition: {mode: aggregate([r['conditions'][condition][mode] for r in per_seed]) for mode in MODES}
                    for condition in CONDITIONS}, per_seed=per_seed, paired_per_seed=pairs)
    result['checks'] = dict(
        normal_within_predeclared_half_mm=all(r['conditions']['clean']['full']['body_rmse_mm']-r['conditions']['clean']['feedback_only']['body_rmse_mm'] <= .5 for r in per_seed),
        action_prediction_beats_no_action_all_seeds=all(r['prospective']['action']['body_rmse_mm'] < r['prospective']['no-action']['body_rmse_mm'] for r in per_seed),
        action_prediction_beats_target_all_seeds=all(r['prospective']['action']['body_rmse_mm'] < r['prospective']['target']['body_rmse_mm'] for r in per_seed),
        biased_full_beats_action_feedback_all_seeds=all(r['conditions']['biased_feedback']['full']['body_rmse_mm'] < r['conditions']['biased_feedback']['action_feedback']['body_rmse_mm'] for r in per_seed),
        biased_full_beats_action_only_all_seeds=all(r['conditions']['biased_feedback']['full']['body_rmse_mm'] < r['conditions']['biased_feedback']['action_only']['body_rmse_mm'] for r in per_seed),
        wrong_rgb_within_half_mm_of_clean_all_seeds=all(r['conditions']['wrong_rgb']['full']['body_rmse_mm']-r['conditions']['clean']['full']['body_rmse_mm'] <= .5 for r in per_seed))
    return result


def validate(output):
    protocol = read_json(output/'final-protocol.json')
    for name, digest in protocol['sources'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    pre_count, prediction_count = 0, 0
    for seed in FINAL_SEEDS:
        source = output/f'data-{seed}'
        metadata = read_json(source/'metadata.json')
        assert metadata['prop_max_displacement_m'] == 0.
        for filename, digest in metadata['before_action_prediction_hashes'].items():
            assert sha(source/'before'/filename) == digest
            pre_count += 1
        pair = read_json(output/f'paired-{seed}/scores.json')
        for filename, digest in pair['before_prediction_hashes'].items():
            assert sha(output/f'paired-{seed}/before'/filename) == digest
            pre_count += 1
        for condition in CONDITIONS:
            path = output/'predictions'/f'data-{seed}-{condition}.npz'
            assert sha(path) == read_json(path.with_suffix('.freeze.json'))['prediction_sha256']
            a = load_npz(path)
            for mode in MODES:
                assert a[f'prediction_{mode}'].shape == (120, 41, 3)
                assert np.isfinite(a[f'prediction_{mode}']).all()
            if condition == 'camera_missing':
                for mode in ('full', 'no_action', 'no_feedback'):
                    assert not a[f'visual_update_{mode}'][35:80].any()
                    assert not a[f'support_{mode}'][35:80].any()
            prediction_count += 1
        clean = load_npz(output/'predictions'/f'data-{seed}-clean.npz')
        for condition in ('biased_feedback', 'delayed_feedback'):
            altered = load_npz(output/'predictions'/f'data-{seed}-{condition}.npz')
            for mode in ('action_only', 'no_feedback'):
                np.testing.assert_array_equal(clean[f'prediction_{mode}'], altered[f'prediction_{mode}'])
    result = dict(verified_utc=utc(), prospective_files_verified=pre_count, sensory_replay_files_verified=prediction_count,
        frozen_source_hashes_pass=True, all_41_point_predictions_finite=True, fixed_props=True,
        no_feedback_modes_invariant_to_corrupted_feedback=True,
        camera_missing_produces_no_visual_support=True)
    write_json(output/'validation.json', result)
    return result


def charts(output, report, summary):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    methods = ('action', 'no-action', 'hold', 'target', 'wrong_action')
    short = ('Learned action', 'No action', 'Keep state', 'Target = state', 'Wrong action')
    palette = ['#007d70', '#9585ae', '#778899', '#dc9639', '#c07378']
    axes[0, 0].bar(range(5), [summary['prospective'][m]['body_rmse_mm'] for m in methods], color=palette)
    axes[0, 0].set(title='Predictions saved BEFORE action execution', ylabel='All-body RMSE / mm', xticks=range(5), xticklabels=short)
    axes[0, 0].tick_params(axis='x', labelrotation=18)
    axes[0, 1].bar(range(5), [summary['paired'][m]['action_contrast_rmse_mm'] for m in methods], color=palette)
    axes[0, 1].set(title='Different commands from identical physical state', ylabel='Error in predicted outcome difference / mm', xticks=range(5), xticklabels=short)
    axes[0, 1].tick_params(axis='x', labelrotation=18)
    for index, mode in enumerate(('feedback_only', 'action_only', 'action_feedback', 'full')):
        axes[1, 0].bar(np.arange(5)+(index-1.5)*.18,
            [summary['conditions'][c][mode]['body_rmse_mm'] for c in CONDITIONS], width=.18,
            color=COLORS[mode], label=mode.replace('_', ' '))
    axes[1, 0].set(title='Sensor interventions: same body decoder', ylabel='All-body RMSE / mm',
                   xticks=range(5), xticklabels=['Clean', 'Biased q', 'Delayed q', 'No camera', 'Wrong RGB'])
    axes[1, 0].legend(fontsize=8)
    seed = FINAL_SEEDS[0]
    source = output/f'data-{seed}'
    metadata = read_json(source/'metadata.json')
    actual = np.stack([load_npz(source/'evaluation'/r['file'])['anchor_xyz'] for r in metadata['frames']])
    body = load_npz(source/'evaluation'/metadata['frames'][0]['file'])['anchor_body']
    a = load_npz(output/'predictions'/f'data-{seed}-biased_feedback.npz')
    times = np.cumsum([r['duration'] for r in metadata['frames']])
    for mode in ('feedback_only', 'action_only', 'action_feedback', 'full'):
        err = np.sqrt(np.mean(np.sum((a[f'prediction_{mode}']-actual)**2, axis=-1)[:, body], axis=1))*1000
        axes[1, 1].plot(times, err, color=COLORS[mode], label=mode.replace('_', ' '))
    axes[1, 1].axvspan(times[20], times[25], color='#dc9639', alpha=.15)
    axes[1, 1].set(title='Biased feedback: first final seed', ylabel='All-body RMSE / mm', xlabel='Physical time / s')
    axes[1, 1].legend(fontsize=8)
    fig.savefig(report/'results.png', dpi=150)
    plt.close(fig)


def movie(output, report):
    seed = FINAL_SEEDS[0]
    source = output/f'data-{seed}'
    metadata = read_json(source/'metadata.json')
    a = load_npz(output/'predictions'/f'data-{seed}-biased_feedback.npz')
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17)
    frames = []
    stages = [('prior_full', 'Before action: forecast', '#72b2e1'),
              ('after_feedback_full', 'After biased feedback', '#e4a242'),
              ('prediction_full', 'After visual update', '#42cab0')]
    for i in range(0, 120, 2):
        obs = load_npz(source/'after'/metadata['frames'][i]['file'])
        truth = load_npz(source/'evaluation'/metadata['frames'][i]['file'])
        actual = project(truth['anchor_xyz'], metadata['calibration'])
        canvas = Image.new('RGB', (1920, 550), '#101a24')
        for j, (key, label, color) in enumerate(stages):
            image = Image.fromarray(obs['left'])
            draw = ImageDraw.Draw(image)
            predicted = project(a[key][i], metadata['calibration'])
            for point in np.flatnonzero(truth['anchor_body']):
                u, v = predicted[point]
                x, y = actual[point]
                draw.line([(float(u), float(v)), (float(x), float(y))], fill=color, width=1)
                draw.ellipse((u-3, v-3, u+3, v+3), fill=color)
                draw.ellipse((x-2, y-2, x+2, y+2), outline='#f2d6f0')
            canvas.paste(image, (j*640, 35))
            ImageDraw.Draw(canvas).text((j*640+12, 7), label, font=font, fill=color)
        note = f'Frame {i} | forecast horizon {metadata["frames"][i]["duration"]*1000:.0f} ms | pale dots = evaluation only | visual update: {bool(a["visual_update_full"][i])}'
        ImageDraw.Draw(canvas).text((12, 520), note, font=font, fill='white')
        frames.append(canvas.resize((1440, 413)))
    frames[0].save(report/'shared-state.gif', save_all=True, append_images=frames[1:], duration=200, loop=0, optimize=False)


def build(output=ROOT):
    read_json(output/'complete.json')
    report = output/'report'
    report.mkdir(exist_ok=True)
    validation = validate(output)
    summary = summarize(output)
    write_json(report/'summary.json', summary)
    with (report/'scores.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        columns = ['seed', 'condition', 'mode', 'body_rmse_mm', 'after_intervention_rmse_mm', 'actuator_delay_rmse_mm', 'visual_update_frames']
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in summary['per_seed']:
            for condition, values in row['conditions'].items():
                for mode, metrics in values.items():
                    writer.writerow(dict(seed=row['seed'], condition=condition, mode=mode, **metrics))
    charts(output, report, summary)
    movie(output, report)
    prospective = summary['prospective']
    paired = summary['paired']
    conditions = summary['conditions']
    rows = ''.join('<tr><td>'+LABELS[mode]+'</td>'+''.join(f'<td>{conditions[c][mode]["body_rmse_mm"]:.2f}</td>' for c in CONDITIONS)+'</tr>' for mode in MODES)
    forecast_rows = ''.join(f'<tr><td>{label}</td><td>{prospective[mode]["body_rmse_mm"]:.2f}</td><td>{paired[mode]["action_contrast_rmse_mm"]:.2f}</td></tr>'
        for mode, label in [('action', '学习动作后果'), ('no-action', '不输入动作'), ('hold', '保持当前位置'), ('target', '将目标当作到达状态'), ('wrong_action', '错误配对动作')])
    dev = [read_json(output/'scores'/f'data-2026102621-gate{gate}.json') for gate in (8, 14)]
    dev_rows = ''.join(f'<tr><td>{gate} mm</td><td>{r["conditions"]["clean"]["full"]["body_rmse_mm"]:.2f}</td><td>{r["conditions"]["biased_feedback"]["full"]["body_rmse_mm"]:.2f}</td><td>{r["conditions"]["wrong_rgb"]["full"]["body_rmse_mm"]:.2f}</td></tr>' for gate, r in zip((8, 14), dev))
    clean, biased, delayed, wrong = (conditions[k] for k in ('clean', 'biased_feedback', 'delayed_feedback', 'wrong_rgb'))
    checks = summary['checks']
    tests = read_json(output/'test-result.json') if (output/'test-result.json').exists() else None
    test_text = f'{tests["tests"]} 项回归测试通过。' if tests and tests['exit_code'] == 0 else '完整回归测试记录尚未生成。'
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>E6 · 动作、反馈与视觉</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#101b24;color:#e4edf3;font:16px/1.8 system-ui,'Microsoft YaHei',sans-serif}}main{{max-width:1240px;margin:auto;padding:32px 22px 70px}}h1{{font-size:32px}}h2{{font-size:23px;margin-top:36px}}p{{color:#bdcdd9}}a{{color:#71d9c3}}img{{width:100%;height:auto;border-radius:8px}}.note{{border-left:3px solid #dca456;background:#23323e;padding:14px 18px}}table{{width:100%;border-collapse:collapse;font-size:15px}}td,th{{padding:11px;border-bottom:1px solid #344958;text-align:left}}.scroll{{overflow:auto}}details{{margin-top:24px}}@media(max-width:650px){{main{{padding:20px 12px}}td,th{{padding:7px}}table{{font-size:12px}}}}</style>
<main><p>E6 / 三个新种子 / 同一个身体空间解码器</p><h1>开始检验动作、实际反馈和视觉之间的联系</h1>
<p>这轮先学习动作会导致怎样的实际关节变化，再将预测状态、反馈更新后的状态、视觉更新后的状态输入同一个身体模型。动作基准预测在物理执行前落盘；融合回放严格按“动作输入→预测→读取未来观测”的顺序处理，全部预测冻结后再评分。</p>
<p class="note">动作预测的身体位置误差为 {prospective['action']['body_rmse_mm']:.2f} mm，不输入动作的对照为 {prospective['no-action']['body_rmse_mm']:.2f} mm。反馈偏差条件下，完整方案为 {biased['full']['body_rmse_mm']:.2f} mm，动作＋反馈为 {biased['action_feedback']['body_rmse_mm']:.2f} mm，单靠动作递推为 {biased['action_only']['body_rmse_mm']:.2f} mm。增加传感器并不保证在每种条件下都优于动作预测。</p>
<h2>动作执行前，能否预测不同的后果</h2><p>三个新种子各 120 步，预测时间跨度为 20/60/120 ms。另有 36 组配对试验，每组从完全相同的物理状态执行两个不同动作。右列衡量两种动作造成的身体变化之差是否预测正确。</p>
<table><tr><th>对照</th><th>提前预测 RMSE / mm</th><th>配对动作结果差的误差 / mm</th></tr>{forecast_rows}</table>
<p>全部误差保留同样的 28 个物理身体锚点。动作训练只用 2,100 条实际反馈转移，另用 350 条独立轨迹观测选择权重；身体空间网络和视觉记忆沿用冻结版本。</p>
<h2>不同观测条件下，同一个状态怎样更新</h2><p>各条件使用完全相同的运动和 RGB 来源，分别改变传感器读数或图像配对。下表单位为 mm，越低越好；“仅动作递推”和“动作＋视觉”均获得一次干净的初始关节读数，随后不读取新的反馈。</p>
<div class="scroll"><table><tr><th>方案</th>{''.join('<th>'+CONDITION_LABELS[c]+'</th>' for c in CONDITIONS)}</tr>{rows}</table></div>
<p>正常条件完整方案与仅反馈之差为 {clean['full']['body_rmse_mm']-clean['feedback_only']['body_rmse_mm']:+.3f} mm；逐种子是否均在预先规定的 +0.5 mm 容差内：{'是' if checks['normal_within_predeclared_half_mm'] else '否'}。错误图像相对正常条件是否均在 +0.5 mm 内：{'是' if checks['wrong_rgb_within_half_mm_of_clean_all_seeds'] else '否'}。这是预设误差容差检查，不是统计非劣效检验。</p>
<p>反馈偏差下，完整方案是否在三个种子均优于动作＋反馈：{'是' if checks['biased_full_beats_action_feedback_all_seeds'] else '否'}；是否均优于仅动作递推：{'是' if checks['biased_full_beats_action_only_all_seeds'] else '否'}。同时报告强对照，避免只挑一个较差基线说明融合有效。</p>
<h2>同一个身体状态的三个阶段</h2><p>回放展示第一个最终种子的反馈偏差条件。三幅图均叠加在动作执行后的相机画面上，依次显示执行前预测、读取反馈后的估计、视觉更新后的估计。浅色点为隔离评价真值；若视觉证据未被接受，最后两幅状态相同。回放放慢，画面上注明真实预测时间跨度。</p>
<img src="shared-state.gif" alt="同一身体表示在动作预测、实际反馈更新和视觉更新三个阶段的回放">
<h2>完整对照图</h2><img src="results.png" alt="动作前预测、同状态不同动作和五种传感器条件的实验结果">
<h2>意外执行延迟与融合局限</h2><p>每条最终轨迹第 90–99 步插入预测器不知道的 20 ms 动作执行延迟。在这段时间，正常传感器条件下仅动作递推误差为 {clean['action_only']['actuator_delay_rmse_mm']:.2f} mm，完整方案为 {clean['full']['actuator_delay_rmse_mm']:.2f} mm。反馈本身延迟一个采样步时，完整方案整体误差为 {delayed['full']['body_rmse_mm']:.2f} mm；这种诊断不能替代复杂实机故障。</p>
<p>视觉更新要求至少 8 个有效双目对应，并要求它降低动作预测与反馈的分歧。完整方案在正常条件与错误图像条件分别接受 {clean['full']['visual_update_frames']}/360 和 {wrong['full']['visual_update_frames']}/360 次视觉更新；反馈偏差条件接受 {biased['full']['visual_update_frames']}/360 次。这意味着错误图像下的稳定性来自显式规则阻止更新，不能解释成网络已经识别了错误图像。未同时组合错误视觉和错误反馈，也未校准不确定性。</p>
<h2>这轮证明到哪里</h2><p>新增网络学习了局部动作到反馈变化的映射，全部方案通过同一冻结身体解码器输出位置。内部状态仍采用人工定义的关节坐标，视觉关联和融合为显式算法；这不能称为已学得统一神经身体表示、完整形态或持续身体身份。结果来自一个固定双目相机、固定机械臂、有限动作范围的仿真，三种子不是三个独立场景。</p>
<p>三个原有桌面干扰物始终固定，本轮使用无遮挡基础场景。仿真身体标签、深度和物理锚点仅用于评价。{test_text}已核验 {validation['prospective_files_verified']} 份动作前预测文件、{validation['sensory_replay_files_verified']} 份融合预测文件的哈希及全部 41 点输出，相机中断区间没有伪造视觉支持。</p>
<details><summary>开发集选择与复现资料</summary><p>开发种子 2026102621；比较 8/14 mm 的动作—反馈分歧门槛后选 8 mm。最终种子 2026102631/32/33 在配置和源码冻结后采集。</p><table><tr><th>开发门槛</th><th>正常</th><th>反馈偏差</th><th>错误图像</th></tr>{dev_rows}</table><p>偏差在第 21–25 步逐渐注入前三关节，分别为 +0.055/+0.018/−0.015 rad；反馈延迟条件从第 25 步起使用上一帧读数；相机在第 35–79 步中断；错误图像从第 25 步起循环重放前 20 帧。索引均从 0 开始。这些干预值与时刻不作为预测器输入。</p></details>
<p><a href="summary.json">完整指标 JSON</a> · <a href="scores.csv">逐种子 CSV</a> · <a href="../final-protocol.json">冻结协议</a> · <a href="../training-freeze.json">训练记录</a> · <a href="../validation.json">验证记录</a> · <a href="/e5-body-memory/report/">E5 历史报告</a></p></main></html>'''
    (report/'index.html').write_text(page, encoding='utf-8')
    print(report.resolve(), flush=True)
    print(summary['checks'], flush=True)


if __name__ == '__main__':
    build()
