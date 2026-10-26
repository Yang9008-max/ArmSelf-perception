"""E7 report: learned source contributions and held-out failures."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .trust_cycle import ROOT, FINAL, DEV, CONDITIONS, MODES, source_for, load_npz
from .body_cycle import read_json, write_json, sha, utc
from .recovery_cycle import BODY, SEGMENTATION
from .action_cycle import BANK
from .vision import project

LABELS = dict(learned='学习动态权重', fixed_equal='固定等权', fixed_learned='训练得到的固定权重',
              action_only='仅动作递推', feedback_only='仅反馈', e6_rules='E6 手工规则／共用视觉前端')
CLABELS = dict(clean='正常观测', biased_feedback='反馈偏差', delayed_feedback='反馈延迟',
    camera_missing='相机中断', wrong_rgb='错误图像', bias_and_wrong_rgb='反馈偏差＋错误图像',
    bias_and_camera_missing='反馈偏差＋相机中断', no_feedback='反馈缺失')
SHORT = ('Clean','Encoder bias','Encoder lag','Camera missing','Wrong RGB','Bias + wrong RGB','Bias + missing RGB','No feedback')
COLORS = ('#4d8dba','#e3a34d','#39b497')


def summarize(output):
    per_seed = [read_json(output/'scores'/f'data-{s}.json') for s in FINAL]
    conditions = {}
    for c in CONDITIONS:
        conditions[c] = dict(modes={m: {k: float(np.sqrt(np.mean([r['conditions'][c]['modes'][m][k]**2 for r in per_seed])))
            for k in per_seed[0]['conditions'][c]['modes'][m]} for m in MODES},
            weights=np.mean([r['conditions'][c]['post_weights'] for r in per_seed], axis=0).tolist())
    checks = {}
    for c in CONDITIONS:
        checks[c] = {m: sum(r['conditions'][c]['modes']['learned']['body_rmse_mm'] < r['conditions'][c]['modes'][m]['body_rmse_mm']
            for r in per_seed) for m in MODES if m != 'learned'}
    criteria = dict(clean_within_half_mm_all_seeds=all(r['conditions']['clean']['modes']['learned']['body_rmse_mm']-
        r['conditions']['clean']['modes']['feedback_only']['body_rmse_mm'] <= .5 for r in per_seed),
        biased_beats_feedback_all_seeds=all(r['conditions']['biased_feedback']['modes']['learned']['body_rmse_mm'] <
            r['conditions']['biased_feedback']['modes']['feedback_only']['body_rmse_mm'] for r in per_seed),
        feedback_weight_lower_when_biased_all_seeds=all(r['conditions']['biased_feedback']['post_weights'][1] <
            r['conditions']['clean']['post_weights'][1] for r in per_seed),
        visual_weight_lower_for_wrong_rgb_all_seeds=all(r['conditions']['wrong_rgb']['post_weights'][2] <
            r['conditions']['clean']['post_weights'][2] for r in per_seed))
    return dict(completed_utc=utc(), seeds=list(FINAL), trajectories=3, frames=450, conditions=conditions,
                wins_out_of_three=checks, criteria=criteria, per_seed=per_seed)


def validate(output):
    protocol = read_json(output/'final-protocol.json')
    training = read_json(output/'training-freeze.json')
    assert not set(training['training_seeds']+[training['dev_seed']]) & set(FINAL)
    for name, digest in protocol['source_digests'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    for name, digest in protocol['models'].items():
        assert sha(output/f'{name}.pt') == digest
    for path, key in ((BODY,'body_sha256'),(SEGMENTATION,'segmentation_sha256'),(BANK,'bank_sha256')):
        assert sha(path) == protocol[key]
    count, before = 0, 0
    for seed in FINAL:
        source = source_for(output, seed)
        meta = read_json(source/'metadata.json')
        assert meta['prop_max_displacement_m'] == 0.
        assert meta['fixed_body_points'] == 28
        for name, digest in meta['before_action_prediction_hashes'].items():
            assert sha(source/'before'/name) == digest
            before += 1
        cache = output/'visual'/f'data-{seed}.npz'
        assert sha(cache) == read_json(cache.with_suffix('.json'))['sha256']
        for condition in CONDITIONS:
            path = output/'predictions'/f'data-{seed}-{condition}.npz'
            assert sha(path) == read_json(path.with_suffix('.json'))['prediction_sha256']
            a = load_npz(path)
            for mode in MODES:
                assert a[f'prediction_{mode}'].shape == (150,41,3)
                assert np.isfinite(a[f'prediction_{mode}']).all()
            np.testing.assert_allclose(a['weights'].sum(-1), 1, atol=2e-7)
            assert (a['weights'] >= 0).all()
            assert (a['weights'][:,1][~a['available'][:,0]] == 0).all()
            assert (a['weights'][:,2][~a['available'][:,1]] == 0).all()
            count += 1
        for condition in ('camera_missing','bias_and_camera_missing'):
            a = load_npz(output/'predictions'/f'data-{seed}-{condition}.npz')
            assert not a['weights'][40:95,2].any()
        clean = load_npz(output/'predictions'/f'data-{seed}-clean.npz')
        for c in CONDITIONS:
            a = load_npz(output/'predictions'/f'data-{seed}-{c}.npz')
            np.testing.assert_array_equal(clean['prediction_action_only'], a['prediction_action_only'])
            if c != 'no_feedback':
                np.testing.assert_array_equal(clean['q_learned'][:30], a['q_learned'][:30])
    result = dict(verified_utc=utc(), prospective_files_verified=before, prediction_files_verified=count,
        train_dev_test_seeds_disjoint=True, hashes_valid=True, all_41_points_finite=True,
        fixed_28_body_point_denominator=True, props_fixed=True,
        missing_sources_have_exact_zero_weights=True, action_baseline_independent_of_sensor_corruption=True,
        pre_intervention_outputs_identical=True,
        sensor_weights_are_not_calibrated_probabilities=True)
    write_json(output/'validation.json', result)
    return result


def figures(output, report, summary):
    values = np.array([[summary['conditions'][c]['modes'][m]['body_rmse_mm'] for c in CONDITIONS] for m in MODES])
    fig, ax = plt.subplots(figsize=(12,5), layout='constrained')
    im = ax.imshow(values, cmap='YlOrRd', vmin=9, vmax=max(30, values.max()))
    ax.set_xticks(range(8), SHORT, rotation=25, ha='right')
    ax.set_yticks(range(6), ['Learned dynamic','Equal fixed','Learned fixed','Action only','Feedback only','E6 rules / shared front end'])
    for row in range(6):
        for col in range(8):
            ax.text(col,row,f'{values[row,col]:.2f}',ha='center',va='center',fontsize=10,
                    color='white' if values[row,col] > 29 else 'black')
    ax.set_title('Held-out body RMSE (mm); all 28 physical anchors, all frames')
    fig.colorbar(im, ax=ax, label='mm')
    fig.savefig(report/'results.png', dpi=150)
    plt.close(fig)
    seed = FINAL[0]
    source = source_for(output, seed)
    meta = read_json(source/'metadata.json')
    truth = [load_npz(source/'evaluation'/r['file']) for r in meta['frames']]
    xyz = np.stack([x['anchor_xyz'] for x in truth])
    mask = np.stack([x['anchor_body'] for x in truth])
    fault = np.array([r['actuator_delay_s'] > 0 for r in meta['frames']])
    for c in CONDITIONS:
        a = load_npz(output/'predictions'/f'data-{seed}-{c}.npz')
        fig, axes = plt.subplots(2,1,figsize=(11,5.5),sharex=True,layout='constrained')
        axes[0].stackplot(np.arange(150), a['weights'].T, labels=['Action prior','Feedback','Stereo vision'], colors=COLORS)
        axes[0].set_ylim(0,1)
        axes[0].set_ylabel('Learned contribution')
        axes[0].legend(ncol=3,loc='upper right')
        axes[0].set_title(SHORT[CONDITIONS.index(c)]+' / one held-out trajectory (not calibrated confidence)')
        for mode, color, label in [('learned','#007f6e','Learned'),('action_only',COLORS[0],'Action only'),('feedback_only',COLORS[1],'Feedback only')]:
            err = np.sqrt((np.sum((a[f'prediction_{mode}']-xyz)**2,-1)*mask).sum(-1)/28)*1000
            axes[1].plot(err,label=label,color=color)
        axes[1].set_ylabel('Body RMSE (mm)')
        axes[1].set_xlabel('Causal update step')
        axes[1].legend(ncol=3)
        for ax in axes:
            ax.axvline(30,ls=':',color='#777777',lw=1)
            for lo, hi in ((50,75),(110,135)):
                ax.axvspan(lo,hi,color='#a0a0a0',alpha=.12)
        fig.savefig(report/f'weights-{c}.png',dpi=130)
        plt.close(fig)


def replay(output, report):
    seed = FINAL[0]
    source = source_for(output, seed)
    meta = read_json(source/'metadata.json')
    a = load_npz(output/'predictions'/f'data-{seed}-biased_feedback.npz')
    font_path = Path('C:/Windows/Fonts/msyh.ttc')
    font = ImageFont.truetype(str(font_path),18) if font_path.exists() else ImageFont.load_default()
    small = ImageFont.truetype(str(font_path),15) if font_path.exists() else font
    frames = []
    colors = {'feedback_only':'#e9a44d','action_only':'#5fa3d5','learned':'#40cfad'}
    for i in range(0,150,3):
        rgb = load_npz(source/'after'/meta['frames'][i]['file'])['left']
        canvas = Image.new('RGB',(1200,400),'#101f2a')
        draw = ImageDraw.Draw(canvas)
        for col, mode in enumerate(('feedback_only','action_only','learned')):
            view = Image.fromarray(rgb).resize((384,288))
            canvas.paste(view,(col*400+8,42))
            draw.text((col*400+12,10),LABELS[mode],fill=colors[mode],font=font)
            uv = project(a[f'prediction_{mode}'][i],meta['calibration'])*.6
            for u,v in uv:
                if 0 <= u < 384 and 0 <= v < 288:
                    x,y=col*400+8+float(u),42+float(v)
                    draw.ellipse((x-2,y-2,x+2,y+2),fill=colors[mode])
        draw.text((12,342),f'反馈偏差场景 · 步 {i+1}/150 · 点为模型输出，不是真值',fill='#dce8ee',font=small)
        weights=a['weights'][i]
        draw.text((12,371),'学习权重：动作 %.2f / 反馈 %.2f / 视觉 %.2f' % tuple(weights),fill='#81d9c0',font=small)
        frames.append(canvas)
    frames[0].save(report/'learned-trust.gif',save_all=True,append_images=frames[1:],duration=140,loop=0)


def build(output=ROOT):
    report=output/'report'
    report.mkdir(exist_ok=True)
    validation=validate(output)
    summary=summarize(output)
    write_json(report/'summary.json',summary)
    figures(output,report,summary)
    replay(output,report)
    with (report/'scores.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['seed','condition','mode','body_rmse_mm','actuator_fault_rmse_mm'])
        for row in summary['per_seed']:
            for c, result in row['conditions'].items():
                for m, score in result['modes'].items():
                    writer.writerow([row['seed'],c,m,score['body_rmse_mm'],score['actuator_fault_rmse_mm']])
    rows=''.join('<tr><th>'+LABELS[m]+'</th>'+''.join(f'<td>{summary["conditions"][c]["modes"][m]["body_rmse_mm"]:.2f}</td>' for c in CONDITIONS)+'</tr>' for m in MODES)
    weight_rows=''.join('<tr><th>'+CLABELS[c]+'</th>'+''.join(f'<td>{v:.3f}</td>' for v in summary['conditions'][c]['weights'])+'</tr>' for c in CONDITIONS)
    options=''.join(f'<option value="{c}">{CLABELS[c]}</option>' for c in CONDITIONS)
    learn=[summary['conditions'][c]['modes']['learned']['body_rmse_mm'] for c in CONDITIONS]
    fixed=[summary['conditions'][c]['modes']['fixed_learned']['body_rmse_mm'] for c in CONDITIONS]
    gain=sum(x<y for x,y in zip(learn,fixed))
    clean_difference=learn[0]-summary['conditions']['clean']['modes']['feedback_only']['body_rmse_mm']
    cw=summary['conditions']['clean']['weights']
    bw=summary['conditions']['biased_feedback']['weights']
    ww=summary['conditions']['wrong_rgb']['weights']
    def mm(condition,mode):
        return summary['conditions'][condition]['modes'][mode]['body_rmse_mm']
    tests=read_json(output/'test-result.json') if (output/'test-result.json').exists() else None
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>E7 · 学习信息可信度</title><style>body{{margin:0;background:#101e29;color:#deebf1;font:16px/1.8 system-ui,'Microsoft YaHei',sans-serif}}main{{max-width:1220px;margin:auto;padding:25px}}section{{background:#192d3a;border:1px solid #365064;border-radius:12px;padding:22px;margin:22px 0}}h1{{font-size:31px}}h2{{font-size:22px}}a{{color:#72d8ba}}img{{width:100%;height:auto}}table{{border-collapse:collapse;white-space:nowrap}}th,td{{padding:9px 13px;border-bottom:1px solid #38505f;text-align:right}}th{{text-align:left}}.scroll{{overflow:auto}}.note{{border-left:3px solid #e7ac57;padding:12px 18px;background:#2b3540}}select{{padding:10px;background:#223e50;color:white;border:1px solid #55778b;border-radius:6px}}small{{color:#a7bfcd}}</style>
<main><p><a href="/">全部实验</a> · <a href="/e6-action-body/report/">E6 对照</a></p><h1>E7：让模型学习怎样相信动作、反馈与视觉</h1>
<p>同一身体状态，新增 32 维 GRU 权重网络。各路信息的动态权重由训练得到；身体解码器和动作模型冻结。原有三件桌面物体固定。</p>
<section><h2>新测试的结果</h2><p>3 条新轨迹，共 450 个物理步骤 × 8 种观测条件。动态权重在 {gain}/8 个条件下优于训练得到的固定权重。正常条件相对仅反馈变化 {clean_difference:+.3f} mm。三个种子均满足正常退化不超过 0.5 mm：{'通过' if summary['criteria']['clean_within_half_mm_all_seeds'] else '未通过'}。</p>
<p>各条件复用同一物理轨迹；不是 8 组独立场景。表中为三个种子汇总平方误差再开方，所有 28 个物理身体点始终留在分母。单位 mm，越低越好。</p>
<div class="scroll"><table><tr><th>方法</th>{''.join('<th>'+CLABELS[c]+'</th>' for c in CONDITIONS)}</tr>{rows}</table></div><img src="results.png" alt="全部条件的空间误差"></section>
<section><h2>结论：反馈可靠性部分学到了，视觉故障判断仍未通过</h2>
<p>正常与偏差反馈条件下，干预后的反馈平均贡献由 {cw[1]:.1%} 降至 {bw[1]:.1%}，三个种子都出现这一变化。偏差反馈误差由仅反馈的 {mm('biased_feedback','feedback_only'):.2f} mm 降至学习融合的 {mm('biased_feedback','learned'):.2f} mm；正常退化在每个种子均不超过 0.5 mm。这是预定容差检查，不是统计非劣效性检验。</p>
<p class="note">错误图像下，视觉平均权重却从 {cw[2]:.1%} 增至 {ww[2]:.1%}，健康反馈权重降至 {ww[1]:.1%}。误差从正常 {mm('clean','learned'):.2f} 升至 {mm('wrong_rgb','learned'):.2f} mm。网络感知到了分歧，却没有可靠判断是谁出了错；不能据此声称三路可信度问题已经解决。</p>
<p>偏差反馈下，学习融合 {mm('biased_feedback','learned'):.2f} mm 仍差于仅动作 {mm('biased_feedback','action_only'):.2f} mm，也差于共用前端的 E6 规则 {mm('biased_feedback','e6_rules'):.2f} mm。相机中断和反馈缺失时亦未超过固定等权。动态权重有条件收益，还不是普遍优于简单方法的融合器。</p></section>
<section><h2>网络实际怎样分配权重</h2><p>选择条件查看第一条独立测试轨迹。上图是每一步学习到的三路贡献，下图是身体误差。灰色区间为未告知模型的执行延迟；竖线是传感器干预开始位置。</p>
<select id="condition">{options}</select><img id="weights" src="weights-clean.png" alt="时序权重及误差曲线">
<p class="note">权重表示对状态更新的贡献，并非“传感器正确概率”。本轮没有做概率校准。图中相机缺失时视觉权重为零，是输入可用性约束；有观测时如何分配权重才是学出来的。</p>
<div class="scroll"><table><tr><th>干预后平均权重（步 40–149）</th><th>动作预测</th><th>实际反馈</th><th>双目视觉</th></tr>{weight_rows}</table></div></section>
<section><h2>反馈偏差时的回放</h2><img src="learned-trust.gif" alt="仅反馈、仅动作和学习融合的对照回放"><p>三列为同一张当前相机图像，叠加各方案输出；没有把评价真值画成模型认知。图像是仿真渲染，融合推断是离线因果回放。</p></section>
<section><h2>怎样训练，怎样防止看答案</h2><p>E6 四条旧轨迹（包括旧测试）明确转作训练资料，另采两条含执行延迟的训练轨迹。训练使用 6 个种子的 780 帧中各取前 120 步，共 720 步，按不同传感器干预扩展为训练序列。独立开发种子 {DEV} 用于选择训练轮次。最终种子 {', '.join(map(str,FINAL))} 在权重、配置与源码冻结后采集。</p>
<p>监督标签是训练轨迹中干净的实际关节读数。输入只有动作预测、可能受损的反馈、RGB 产生的视觉候选和匹配质量；网络不接收干净标签、故障类别或仿真几何。损失比较融合状态与干净读数，并比较同一冻结身体网络的两种输出；另用候选在训练记录中的实际误差监督权重。开发选轮次依据全部条件的状态误差，并对正常条件额外赋三倍权重。它不是无监督的可信度发现。</p>
<p>视觉前端用独立于融合状态的动作递推寻找真实 RGB 对应，并通过冻结身体解码器求视觉状态候选。它仍受同一动作模型先验影响，动作与视觉可能相关出错，不能当成统计独立证据。视觉匹配、最少支持数、逆解器和缺失通道屏蔽仍是显式算法；学习的是可用候选之间的动态权重。</p>
<p>固定等权、训练得到的固定权重以及 E6 式手工规则使用同一视觉候选，检验动态权重是否有额外价值。这里的 E6 规则对照共用新前端，不能拿它的数值冒充原 E6 完全相同的复现。“仅反馈”在反馈缺失条件退回独立动作预测。</p></section>
<section><h2>结果边界</h2><p>固定相机、同一机械臂、局部动作范围，训练和测试故障来自相关的人工干预族。即使新符号、幅度或组合有效，也不足以证明能识别任意传感器故障。三路信息共同出错或缺乏独立证据时，权重网络可能作出错误判断。</p>
<p>本轮没有新增完整身体形态、持续身份、实机或已校准的不确定性，也没有声称学会了自我意识。判断成功应同时看正常退化、受损条件收益、是否优于固定权重与仅动作，以及视觉是否实际有用。</p>
<p>已验证 {validation['prospective_files_verified']} 份动作前预测、{validation['prediction_files_verified']} 份融合输出、冻结哈希、全部有限输出、缺失通道零贡献、干预前输出一致及固定干扰物。{str(tests['tests'])+' 项回归测试通过。' if tests else '回归测试记录尚未写入。'}</p>
<p><a href="summary.json">完整指标</a> · <a href="scores.csv">逐种子 CSV</a> · <a href="../final-protocol.json">冻结测试协议</a> · <a href="../training-history.json">训练曲线数据</a> · <a href="../validation.json">自动审计</a></p></section></main>
<script>document.getElementById('condition').addEventListener('change',e=>{{document.getElementById('weights').src='weights-'+e.target.value+'.png'}});</script></html>'''
    (report/'index.html').write_text(text,encoding='utf-8')
    print('report',report.resolve(),flush=True)
    return summary


if __name__ == '__main__':
    build()
