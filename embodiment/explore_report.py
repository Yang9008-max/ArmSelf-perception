"""Freeze-first evaluation and concrete plots for autonomous exploration."""
from __future__ import annotations

import argparse
import base64
import csv
from io import BytesIO
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image

from .experiment import write_json
from .exploration import ExplorationBelief
from .vision import StereoTracks


def evaluate(runs, data):
    runs, data = Path(runs), Path(data)
    output = runs/'report'
    output.mkdir(exist_ok=True)
    if (output/'evaluation.json').exists():
        raise FileExistsError('Preserving the frozen exploration evaluation')
    config = json.loads((runs/'config.json').read_text())
    if 'all_models_frozen_utc' not in config:
        raise ValueError('Complete and freeze all acquisition runs before evaluating')
    prior = Path(config['prior_directory'])
    selected = np.array(json.loads((prior/'discovery.json').read_text())['selected'])
    metadata = json.loads((data/'metadata.json').read_text())
    with np.load(prior/'reference.npz') as reference:
        tracker = StereoTracks(reference['left'], metadata['calibration'], 'refined11')
        np.testing.assert_allclose(tracker.reference_points, reference['points'])
    samples = []
    for frame, row in enumerate(metadata['frames']):
        with np.load(data/'observations'/row['file']) as obs:
            q = obs['q'].copy()
            images = (obs['left'], obs['right']) if bool(obs['camera_available']) else None
        with np.load(data/'evaluation'/row['file']) as truth:
            if images is None:
                images = (truth['oracle_left'], truth['oracle_right'])
            tracks = tracker.observe(*images)
            uv, valid = tracks['uv'][selected], tracks['valid'][selected]
            pixels = np.rint(uv).astype(int)
            h, w = truth['body_mask'].shape
            pixels[:, 0] = np.clip(pixels[:, 0], 0, w-1)
            pixels[:, 1] = np.clip(pixels[:, 1], 0, h-1)
            body = truth['body_mask'][pixels[:, 1], pixels[:, 0]] & valid
            z = truth['depth'][pixels[:, 1], pixels[:, 0]]
            c = metadata['calibration']
            geometry = np.column_stack(((uv[:, 0]-c['cx'])*z/c['focal_px'],
                                        (uv[:, 1]-c['cy'])*z/c['focal_px'], z))
            samples.append(dict(q=q, xyz=tracks['xyz'][selected], valid=valid, body=body,
                                geometry=geometry, phase=row['phase'], frame=frame))
    arrays = {key: np.stack([r[key] for r in samples]) for key in samples[0]}
    checkpoints = []
    common_known = np.ones(len(selected), dtype=bool)
    for seed in config['seeds']:
        bootstrap = []
        for policy in config['policies']:
            directory = runs/f'{seed}-{policy}'
            trace = json.loads((directory/'decisions.json').read_text())
            assert all(r['observed_before_choice'] == r['step'] for r in trace)
            with np.load(directory/'tracks.npz') as train:
                bootstrap.append(train['q'][:16].copy())
            freeze = json.loads((directory/'freeze.json').read_text())
            for budget in config['checkpoints']:
                name = f'belief-{budget:03d}.npz'
                assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == freeze['checkpoints'][name]
                with np.load(directory/name) as archive:
                    belief = ExplorationBelief.from_state({key: archive[key] for key in archive.files})
                prediction = belief.predict(arrays['q'])[0]
                known = belief.count >= 4
                common_known &= known
                checkpoints.append(dict(seed=seed, policy=policy, budget=budget,
                                        prediction=prediction, known=known, belief=belief))
        np.testing.assert_array_equal(bootstrap[0], bootstrap[1])
        np.testing.assert_array_equal(bootstrap[0], bootstrap[2])
    common_mask = arrays['body'] & common_known[None]
    scores = []
    for checkpoint in checkpoints:
        prediction, known = checkpoint['prediction'], checkpoint['known']
        geometric_error = np.sum((prediction-arrays['geometry'])**2, axis=-1)
        observed_error = np.sum((prediction-arrays['xyz'])**2, axis=-1)
        all_valid = arrays['valid'] & known[None]
        row = {key: checkpoint[key] for key in ('seed', 'policy', 'budget')}
        row.update(common_body_geometry_rmse_mm=float(np.sqrt(geometric_error[common_mask].mean())*1000),
                   common_body_stereo_rmse_mm=float(np.sqrt(observed_error[common_mask].mean())*1000),
                   all_observation_rmse_mm=float(np.sqrt(observed_error[all_valid].mean())*1000),
                   known_features=int(known.sum()), common_body_observations=int(common_mask.sum()))
        for phase in ('test', 'hold', 'blind'):
            mask = common_mask & (arrays['phase'] == phase)[:, None]
            row[f'{phase}_body_geometry_rmse_mm'] = float(np.sqrt(geometric_error[mask].mean())*1000)
        scores.append(row)
    evaluation = dict(config=config, evaluation_seed=metadata['seed'],
                      evaluation_metadata_sha256=hashlib.sha256((data/'metadata.json').read_bytes()).hexdigest(),
                      evaluation_frames=len(samples), common_features=int(common_known.sum()),
                      common_body_observations=int(common_mask.sum()), scores=scores,
                      limitations=['E2-discovered feature IDs and reference image are prior knowledge',
                                   'Online quadratic belief and acquisition rule are additional to the E2 neural model; no RL',
                                   'Local motion envelope, ideal feedback, three acquisition seeds, one final trajectory',
                                   'Regression covariance is an uncertainty proxy; visibility and correspondence failures remain',
                                   'No complete shape, learned parts or persistent identity objective'])
    write_json(output/'evaluation.json', evaluation)
    with (output/'scores.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scores[0]))
        writer.writeheader()
        writer.writerows(scores)
    np.savez_compressed(output/'evaluation-observations.npz', selected=selected, common_known=common_known, **arrays)
    for policy in config['policies']:
        values = [r['common_body_geometry_rmse_mm'] for r in scores if r['policy'] == policy and r['budget'] == 112]
        print(policy, 'final common-body geometry mm:', np.round(values, 2), 'mean', np.mean(values), flush=True)
    return evaluation


def build(runs):
    runs = Path(runs)
    output = runs/'report'
    report = json.loads((output/'evaluation.json').read_text())
    config, scores = report['config'], report['scores']
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    colors = dict(active='#149b86', random='#d18a43', preset='#8499b2')
    labels = dict(active='模型选择动作', random='随机动作', preset='预设动作')
    traces = {}
    for seed in config['seeds']:
        for policy in config['policies']:
            traces[seed, policy] = json.loads((runs/f'{seed}-{policy}'/'decisions.json').read_text())
    fig, axs = plt.subplots(2, 2, figsize=(13.5, 8.5), layout='constrained')
    fig.suptitle('E3 自主探索对照 · 3 个采集种子 · 相同 112 帧预算 · 新轨迹冻结评价', fontsize=16)
    budgets = config['checkpoints']
    for policy in config['policies']:
        values = np.array([[r['common_body_geometry_rmse_mm'] for r in scores
                            if r['policy'] == policy and r['budget'] == b] for b in budgets])
        mean, std = values.mean(1), values.std(1)
        axs[0, 0].plot(budgets, mean, '-o', label=labels[policy], color=colors[policy])
        axs[0, 0].fill_between(budgets, mean-std, mean+std, color=colors[policy], alpha=.12)
    axs[0, 0].set(title='共同身体点的空间学习效率', xlabel='已获得的图像/反馈观测数', ylabel='对真实几何 RMSE / mm（越低越好）')
    axs[0, 0].legend(fontsize=9)
    axs[0, 0].grid(alpha=.2)
    for i, policy in enumerate(config['policies']):
        values = [r['common_body_geometry_rmse_mm'] for r in scores if r['policy'] == policy and r['budget'] == 112]
        bar = axs[0, 1].bar(i, np.mean(values), color=colors[policy], width=.6)
        axs[0, 1].scatter(i+np.linspace(-.12, .12, len(values)), values, s=28, edgecolor='white', color='#243d4a', zorder=3)
        axs[0, 1].bar_label(bar, fmt='%.2f', padding=4)
    axs[0, 1].set(title='最终误差：圆点为各采集种子', ylabel='共同身体点 RMSE / mm',
                  xticks=range(3), xticklabels=[labels[p] for p in config['policies']])
    axs[0, 1].margins(y=.2)
    for policy in config['policies']:
        visible = np.array([[r['visible_features'] for r in traces[s, policy]] for s in config['seeds']])
        axs[1, 0].plot(np.arange(112)+1, visible.mean(0), label=labels[policy], color=colors[policy])
    axs[1, 0].axvspan(1, 16, color='gray', alpha=.1)
    axs[1, 0].set(title='有效视觉对应数量（含轮廓歧义）', xlabel='采集步数；前 16 步为共同随机起步', ylabel='有效候选点 / 帧')
    axs[1, 0].legend(fontsize=9)
    trace = traces[config['seeds'][0], 'active']
    for joint in range(7):
        q = np.array([r['measured_q'] for r in trace])
        normalized = (q[:, joint]-config['center'][joint])/config['amplitude'][joint]
        axs[1, 1].plot(np.arange(112)+1, normalized, label=f'q{joint+1}', linewidth=1)
    axs[1, 1].axvline(16, color='gray', linestyle='--')
    axs[1, 1].set(title=f'模型实际选择后测到的运动 · 种子 {config["seeds"][0]}', xlabel='采集步数', ylabel='归一化实际关节反馈')
    axs[1, 1].legend(ncol=4, fontsize=8)
    fig.savefig(output/'exploration-comparison.png', dpi=160)
    fig.canvas.draw()
    for i, ax in enumerate(axs.flat, 1):
        extent = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(output/f'exploration-{i}.png', dpi=160, bbox_inches=extent.expanded(1.03, 1.06))
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), layout='constrained')
    for seed in config['seeds']:
        trace = traces[seed, 'active']
        axes[0].plot([r['step']+1 for r in trace], [r['score'] for r in trace], label=f'种子 {seed}')
    axes[0].axvline(16, color='gray', linestyle='--')
    axes[0].set(title='每步选择时的模型信息评分', xlabel='采集步数；第 17 步起使用模型评分', ylabel='回归不确定性评分（非奖励真值）')
    axes[0].legend(fontsize=9)
    movement = {}
    for i, policy in enumerate(config['policies']):
        distances = []
        for seed in config['seeds']:
            q = np.array([r['measured_q'] for r in traces[seed, policy]])
            distances.append(float(np.linalg.norm(np.diff(q, axis=0)/np.array(config['amplitude']), axis=-1).sum()))
        movement[policy] = distances
        bar = axes[1].bar(i, np.mean(distances), color=colors[policy], width=.6)
        axes[1].bar_label(bar, fmt='%.1f', padding=3)
    axes[1].set(title='探索代价：实际关节运动总量', ylabel='按各关节探索幅度归一化的累计路径长度',
                xticks=range(3), xticklabels=[labels[p] for p in config['policies']])
    axes[1].margins(y=.2)
    fig.savefig(output/'decision-scores.png', dpi=160)
    plt.close(fig)
    replay = {}
    seed = config['seeds'][0]
    for policy in config['policies']:
        replay[policy] = []
        for r in traces[seed, policy][::2]:
            with np.load(runs/f'{seed}-{policy}'/'observations'/f'{r["step"]:05d}.npz') as obs:
                buffer = BytesIO()
                Image.fromarray(obs['left']).resize((400, 300)).save(buffer, format='JPEG', quality=82)
            replay[policy].append(dict(step=r['step'], source=r['source'], score=r['score'],
                                       seen=r['observed_before_choice'], visible=r['visible_features'],
                                       q=r['measured_q'], target=r['target'],
                                       image='data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii')))
    means = {p: float(np.mean([r['common_body_geometry_rmse_mm'] for r in scores if r['policy'] == p and r['budget'] == 112]))
             for p in config['policies']}
    outcome = '，'.join(f'{labels[p]} {means[p]:.2f} mm' for p in config['policies'])
    if means['active'] < min(means['random'], means['preset']):
        conclusion = (f'本次模型选动作的平均误差较低，但相对随机只差 {means["random"]-means["active"]:.2f} mm。'
                      '三次采集与一条评价轨迹不足以证明稳定优于随机探索。')
    else:
        conclusion = '本次自主选择动作没有同时超过两个对照。闭环已运行，学习效率的优势尚未得到证明。'
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>E3 · 模型自主探索与实测对照</title>
<style>*{box-sizing:border-box}body{margin:0;background:#101820;color:#e6edf5;font:15px/1.65 system-ui,'Microsoft YaHei'}main{max-width:1240px;margin:auto;padding:24px}h1{font-size:26px}p{color:#bed0dd}a{color:#78dabb}.tag{color:#78dabb;letter-spacing:2px;font-size:12px}.note{background:#233442;padding:14px;border-left:3px solid #e5b260}.plots{display:grid;grid-template-columns:1fr 1fr;gap:16px}.plots img{width:100%;display:block;background:white}button{background:#284553;border:1px solid #587382;color:white;padding:9px 14px;border-radius:6px;cursor:pointer}.bar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:15px 0}input{flex:1;min-width:140px}.views{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.card{background:#192a35;padding:12px;border:1px solid #3d5362;border-radius:8px}.card img{width:100%;display:block}.card h3{margin:0 0 8px}.detail{font-size:12px;overflow-wrap:anywhere}.wide{width:100%}@media(max-width:760px){main{padding:14px}.plots,.views{grid-template-columns:1fr}h1{font-size:22px}}</style><main>
<div class="tag">E3 / CLOSED SENSOR–MODEL–ACTION LOOP</div><h1>模型开始决定下一步怎么动</h1>
<p>每步先用当前实测反馈和已经学到的预测模型选择动作，执行后读取双目图像与实际 q，再更新模型。动作日志逐步保存；没有提前读取下一帧或仿真身体真值。</p>
<p class="note">这是基于在线概率预测模型的主动探索。信息评分、局部运动范围和前 16 步随机起步由程序设定；当前没有强化学习策略。使用 E2 已发现的 41 个参考特征作为起点，尚未从零学习完整身体形态。</p>
<h2>相同预算下的实际结果</h2><p>__OUTCOME__。__CONCLUSION__</p>
<div class="plots">__PLOTS__</div>
<p>3 个采集种子，均采 112 帧、每帧物理运行 0.2 秒。共同起步后才切换策略。阴影表示三次采集的标准差；评价使用冻结后新采的 __EVAL_FRAMES__ 帧。比较限制在各次起步均已有至少 4 次观测的 __COMMON_FEATURES__ 个特征及其共同有效身体像素，累计 __BODY_OBS__ 次观测；这些不是独立实验次数。</p>
<a href="decision-scores.png" target="_blank"><img class="wide" src="decision-scores.png" alt="每步模型信息评分和实际运动代价"></a>
<p>预算只匹配观测数和仿真时间，没有匹配实际运动总量或能量。自主策略会走更多、也会让部分特征失去对应；信息评分并未完整考虑未来遮挡和可跟踪性。</p>
<h2>看实际探索过程</h2><p>下方为种子 __SEED__ 的同步录制。绿色策略从第 17 步起，根据前面得到的观测更新决策。每两步保留一个回放画面，原始逐步数据仍在本地。</p>
<div class="bar"><button id="play">播放探索</button><input id="timeline" aria-label="探索时间线" type="range" min="0" max="55" value="0"><span id="step"></span></div>
<div class="views">__VIEWS__</div>
<h2>距离完整实体感知还缺什么</h2>
<p>① 已运行：观测更新预测模型，模型选择下一动作。② 待验证：通过主动改变姿态补全可见形态，在未见过的姿态与遮挡上验证表面和空间范围。③ 待学习：从各区域随反馈共同变化的规律建立部件关系。④ 待学习：遮挡、停止、重新出现后保持并重新确认同一实体，而不是只依赖程序保留编号。</p>
<p>当前仍有轮廓/背景错误对应，回归不确定性也未标定。选择了高信息评分的动作不保证它真正改善身体感知，需看上面的独立数据。完整形态、部件关系和持续身份目前没有可靠完成时间。</p>
<p><a href="scores.csv" download>下载全部采样预算指标 CSV</a> · <a href="evaluation.json">评价与算法配置 JSON</a> · <a href="exploration-comparison.png" target="_blank">完整数据图</a> · <a href="../">返回 E2 空间模型</a></p>
</main><script>const replay=__REPLAY__;const labels={active:'模型选择动作',random:'随机动作',preset:'预设动作'};let index=0,playing=false,last=0;const slider=document.getElementById('timeline');function show(){slider.value=index;document.getElementById('step').textContent=`第 ${replay.active[index].step+1} / 112 步`;Object.keys(replay).forEach(p=>{const f=replay[p][index];document.getElementById(p+'-image').src=f.image;document.getElementById(p+'-info').textContent=`决策来源：${f.source} ｜ 已观察 ${f.seen} 帧 ｜ 信息评分 ${f.score.toFixed(3)} ｜ 有效特征 ${f.visible}`;document.getElementById(p+'-q').textContent='实测 q：'+f.q.map(v=>v.toFixed(3)).join(', ');});}slider.oninput=()=>{index=Number(slider.value);show()};document.getElementById('play').onclick=()=>{playing=!playing;document.getElementById('play').textContent=playing?'暂停探索':'播放探索'};function tick(t){if(playing&&t-last>250){index=(index+1)%replay.active.length;show();last=t}requestAnimationFrame(tick)}show();requestAnimationFrame(tick);</script></html>'''
    plots = ''.join(f'<a href="exploration-{i}.png" target="_blank"><img src="exploration-{i}.png" alt="自主探索实测图 {i}"></a>' for i in range(1, 5))
    views = ''.join(f'<section class="card"><h3>{labels[p]}</h3><img id="{p}-image" alt="{labels[p]}的真实相机输入"><p class="detail" id="{p}-info"></p><p class="detail" id="{p}-q"></p></section>' for p in config['policies'])
    for key, value in dict(OUTCOME=outcome, CONCLUSION=conclusion, PLOTS=plots, VIEWS=views,
                           EVAL_FRAMES=report['evaluation_frames'], COMMON_FEATURES=report['common_features'],
                           BODY_OBS=report['common_body_observations'], SEED=seed,
                           REPLAY=json.dumps(replay, ensure_ascii=False)).items():
        html = html.replace(f'__{key}__', str(value))
    (output/'index.html').write_text(html, encoding='utf-8')
    print(outcome, conclusion, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('evaluate', 'build'))
    parser.add_argument('--runs', required=True)
    parser.add_argument('--data')
    args = parser.parse_args()
    if args.mode == 'evaluate':
        if args.data is None:
            parser.error('--data is required for evaluation')
        evaluate(args.runs, args.data)
    else:
        build(args.runs)
