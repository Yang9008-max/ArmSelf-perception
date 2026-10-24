"""Paired, coverage-aware plots of frozen models on the same new trajectory."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from project_paths import recorded_path
from project_plot import configure_chinese_font

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def compare(baseline, candidate, output):
    directories = [Path(baseline), Path(candidate)]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reports = [json.loads((p/'metrics.json').read_text(encoding='utf-8')) for p in directories]
    if reports[0]['evaluation_data_metadata_sha256'] != reports[1]['evaluation_data_metadata_sha256']:
        raise ValueError('Paired comparison requires the exact same evaluation trajectory')
    tracks = []
    for p in directories:
        with np.load(p/'evaluation-tracks.npz') as archive:
            tracks.append({key: archive[key] for key in archive.files})
    np.testing.assert_array_equal(tracks[0]['frame'], tracks[1]['frame'])
    np.testing.assert_array_equal(tracks[0]['phase'], tracks[1]['phase'])
    common, a, b = np.intersect1d(tracks[0]['selected'], tracks[1]['selected'], return_indices=True)
    # Same feature identity AND same observed pixel: compare against one geometry target.
    same_pixel = (np.rint(tracks[0]['uv'][:, a]) == np.rint(tracks[1]['uv'][:, b])).all(-1)
    valid = tracks[0]['on_body'][:, a] & tracks[1]['on_body'][:, b] & same_pixel
    geometric = tracks[0]['geometric'][:, a]
    errors = [np.sum((t['prediction'][:, ids]-geometric)**2, axis=-1)
              for t, ids in zip(tracks, (a, b))]
    paired = {}
    for phase in ('test', 'hold', 'blind'):
        mask = valid & (tracks[0]['phase'] == phase)[:, None]
        paired[phase] = dict(observations=int(mask.sum()),
                            baseline_rmse_mm=float(np.sqrt(errors[0][mask].mean())*1000) if mask.any() else None,
                            candidate_rmse_mm=float(np.sqrt(errors[1][mask].mean())*1000) if mask.any() else None)
    result = dict(evaluation_seed=reports[0]['evaluation_seed'], common_feature_id_count=len(common),
                  criterion='Same feature ID, valid body observation in both methods, same rounded image pixel; common geometry target',
                  paired_geometry=paired,
                  models=[dict(name=name, point_count=r['point_count'], summary=r['summary'],
                               checkpoint_sha256=r['checkpoint_sha256']) for name, r in zip(('E1', 'E2'), reports)])
    (output/'comparison.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    with (output/'per-frame.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'frame', 'phase', 'time_s', 'valid_points', 'body_points',
                         'stereo_target_rmse_mm', 'body_geometry_rmse_mm', 'status'])
        for name, report in zip(('E1', 'E2'), reports):
            for r in report['per_frame']:
                writer.writerow([name, r['frame'], r['phase'], r['time'], r['valid_features'], r['body_features'],
                                 np.sqrt(r['squared_error_sum']/r['valid_features'])*1000 if r['valid_features'] else '',
                                 np.sqrt(r['body_geometric_error_sum']/r['body_features'])*1000 if r['body_features'] else '', r['status']])
    configure_chinese_font()
    colors, labels = ['#879bb3', '#149b86'], ['E1 原模型', 'E2 改进测量后重训']
    phases, phase_names = ['test', 'hold', 'blind'], ['新动作', '停止', '相机中断']
    fig, axs = plt.subplots(2, 2, figsize=(13.5, 8.5), layout='constrained')
    fig.suptitle(f'冻结模型同轨迹对比 · 新采集 {len(reports[0]["per_frame"])} 帧 · 种子 {reports[0]["evaluation_seed"]}', fontsize=17)
    x = np.arange(3)
    for i, (name, color) in enumerate(zip(labels, colors)):
        values = [paired[p]['baseline_rmse_mm' if i == 0 else 'candidate_rmse_mm'] for p in phases]
        bars = axs[0, 0].bar(x+(i-.5)*.37, values, .35, label=name, color=color)
        axs[0, 0].bar_label(bars, fmt='%.2f', padding=3)
    axs[0, 0].set(title='同一点、同一像素：对真实几何的三维误差', ylabel='RMSE / mm（越低越好）',
                  xticks=x, xticklabels=[f'{n}\n{paired[p]["observations"]} 次配对观测' for n, p in zip(phase_names, phases)])
    axs[0, 0].margins(y=.25)
    axs[0, 0].legend(fontsize=9)
    for name, color, report in zip(labels, colors, reports):
        path = recorded_path(report['model_directory'])/'history.json'
        history = json.loads(path.read_text())
        axs[0, 1].plot([r['epoch'] for r in history], [np.sqrt(r['validation_squared_m'])*1000 for r in history], color=color, label=name+' 验证')
        axs[0, 1].plot([r['epoch'] for r in history], [np.sqrt(r['train_squared_m'])*1000 for r in history], color=color, linestyle=':', alpha=.8)
    axs[0, 1].set(title='学习曲线：实线验证，虚线训练', xlabel='训练轮次', ylabel='对各自双目观测 RMSE / mm')
    axs[0, 1].legend(fontsize=9)
    axs[0, 1].grid(alpha=.2)
    blind = tracks[0]['phase'] == 'blind'
    times = np.array([r['time'] for r in reports[0]['per_frame']])
    for name, color, err in zip(labels, colors, errors):
        count = valid[blind].sum(1)
        value = np.sqrt(np.divide((err*valid)[blind].sum(1), count, out=np.full(count.shape, np.nan), where=count > 0))*1000
        axs[1, 0].plot(times[blind]-times[blind][0], value, color=color, label=name)
    axs[1, 0].set(title='相机关闭后：同一批身体点的空间预测', xlabel='相机中断时长 / s', ylabel='对真实几何 RMSE / mm')
    axs[1, 0].legend(fontsize=9)
    axs[1, 0].grid(alpha=.2)
    for i, (name, color, report) in enumerate(zip(labels, colors, reports)):
        bars = axs[1, 1].bar(x+(i-.5)*.37, [report['summary'][p]['mean_body_features_per_frame'] for p in phases], .35,
                            label=f'{name}（共 {report["point_count"]} 点）', color=color)
        axs[1, 1].bar_label(bars, fmt='%.1f', padding=3)
    axs[1, 1].set(title='覆盖代价：每帧可验证的身体点数量', ylabel='有效且落在身体像素内的点 / 帧', xticks=x, xticklabels=phase_names)
    axs[1, 1].margins(y=.25)
    axs[1, 1].legend(fontsize=9)
    fig.savefig(output/'comparison.png', dpi=160)
    fig.canvas.draw()
    for index, ax in enumerate(axs.flat, 1):
        extent = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(output/f'panel-{index}.png', dpi=160, bbox_inches=extent.expanded(1.03, 1.06))
    plt.close(fig)

    report = reports[1]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5), layout='constrained')
    fig.suptitle('E2 反馈依赖检查 · 误差参考为双目轨迹，不能当成真实空间精度', fontsize=15)
    for offset, key, label, color in [(-.23, 'model_rmse_m', '正确反馈', '#149b86'),
                                     (0., 'fixed_rmse_m', '固定位置', '#879bb3'),
                                     (.23, 'shuffled_q_rmse_m', '打乱当前反馈', '#d58b4b')]:
        bars = axs[0].bar(x+offset, [report['summary'][p][key]*1000 for p in phases], .22, label=label, color=color)
        axs[0].bar_label(bars, fmt='%.1f', padding=3, fontsize=9)
    axs[0].set(xticks=x, xticklabels=phase_names, ylabel='三维 RMSE / mm', title='冻结模型是否依赖实际反馈')
    axs[0].margins(y=.25)
    axs[0].legend(fontsize=9)
    rows = [r for r in report['per_frame'] if r['phase'] == 'blind']
    for field, label in [('squared_error_sum', '反馈更新预测'), ('last_seen_squared_error_sum', '保持最后所见位置')]:
        axs[1].plot([r['time']-rows[0]['time'] for r in rows],
                    [np.sqrt(r[field]/r['valid_features'])*1000 if r['valid_features'] else np.nan for r in rows], label=label)
    axs[1].set(xlabel='相机中断时长 / s', ylabel='对隐藏双目参考 RMSE / mm', title='关闭相机并继续运动')
    axs[1].legend(fontsize=9)
    axs[1].grid(alpha=.2)
    fig.savefig(output/'feedback-controls.png', dpi=160)
    plt.close(fig)
    print(json.dumps(paired, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output', required=True)
    compare(**vars(parser.parse_args()))
