"""Build E8's audited report from frozen predictions; never tune the estimator."""
from __future__ import annotations

import csv
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from . import trust_cycle as e7
from .reliable_cycle import ROOT,TRAIN,DEV,CALIBRATION,FINAL,CONDITIONS,PROFILES,source_for,verify_freeze
from .body_cycle import read_json,write_json,sha,utc
from .vision import project
from .action_state import ActionTransition
from .body_memory import BodyModel
from .trust_state import ReliabilityGate,LearnedBodyState
from .reliable_state import JointReliability,ReliableBodyState

CLABELS = dict(clean='正常',biased_feedback='反馈偏差',delayed_feedback='反馈延迟',
    camera_missing='相机缺失',wrong_rgb='错误图像',bias_and_wrong_rgb='偏差＋错图',
    bias_and_camera_missing='偏差＋缺图',no_feedback='反馈缺失')
SHORT = ('Clean','Biased F','Delayed F','No RGB','Wrong RGB','Bias + wrong RGB','Bias + no RGB','No F')
LABELS = dict(per_joint='E8 逐关节／无偏差状态',bias_state='E8 逐关节＋偏差状态',
    e7_learned='冻结 E7',e7_fixed_equal='固定等权',e7_fixed_learned='E7 训练固定权重',
    e7_action_only='仅动作',e7_feedback_only='仅反馈／缺失时回退动作',e7_e6_rules='E6 规则／共用前端')


def benchmark(output,selected):
    import torch
    torch.set_num_threads(4)
    source=source_for(output,FINAL[0])
    cache=output/'visual'/f'data-{FINAL[0]}.npz'
    sensors=e7.sensor_stream(source,cache,'clean',training=False)
    initial=e7.load_npz(source/'initial.npz')['q']
    transition,body=ActionTransition.load(output/'action.pt'),BodyModel(e7.BODY)
    gates=dict(e7=ReliabilityGate.load(output/'gate.pt'),e8=JointReliability.load(output/f'{selected}.pt'))
    results={}
    for name,gate in gates.items():
        durations=[]
        cls=LearnedBodyState if name=='e7' else ReliableBodyState
        for repeat in range(4):
            state=cls(transition,gate,body,initial)
            for i in range(len(sensors['duration'])):
                started=perf_counter()
                state.forecast(sensors['target'][i],float(sensors['duration'][i]))
                f_ok,v_ok=sensors['available'][i]
                state.finish(sensors['feedback'][i] if f_ok else None,sensors['visual'][i],sensors['quality'][i],v_ok)
                elapsed=(perf_counter()-started)*1000
                if repeat:
                    durations.append(elapsed)
        results[name]=dict(constructed_parameters=sum(p.numel() for p in gate.parameters()),
            median_step_ms=float(np.median(durations)),p95_step_ms=float(np.quantile(durations,.95)),
            measurements=len(durations))
    write_json(output/'benchmark.json',dict(created_utc=utc(),results=results,torch_cpu_threads=4,
        scope='forecast plus state update and body decoding; cached visual proposals; excludes image matching, rendering and I/O',
        purpose='descriptive CPU timing only, not a real-time end-to-end guarantee'))
    return results


def audit(output):
    protocol = verify_freeze(output)
    prospective,predictions = 0,0
    for seed in FINAL:
        source = source_for(output,seed)
        metadata = read_json(source/'metadata.json')
        assert metadata['prop_max_displacement_m']==0.
        for filename,digest in metadata['before_action_prediction_hashes'].items():
            assert sha(source/'before'/filename)==digest
            prospective+=1
        clean = e7.load_npz(output/'e8-predictions'/f'data-{seed}-clean.npz')
        action = e7.load_npz(output/'predictions'/f'data-{seed}-clean.npz')['prediction_action_only']
        for c in CONDITIONS:
            for directory,key in (('predictions','prediction_sha256'),('e8-predictions','sha256')):
                path = output/directory/f'data-{seed}-{c}.npz'
                assert sha(path)==read_json(path.with_suffix('.json'))[key]
                predictions+=1
            a = e7.load_npz(output/'e8-predictions'/f'data-{seed}-{c}.npz')
            baseline = e7.load_npz(output/'predictions'/f'data-{seed}-{c}.npz')
            np.testing.assert_array_equal(action,baseline['prediction_action_only'])
            for p in PROFILES:
                assert a[f'xyz_{p}'].shape==(150,41,3)
                assert np.isfinite(a[f'xyz_{p}']).all()
                assert np.isfinite(a[f'sigma_normalized_{p}']).all()
                assert (a[f'sigma_normalized_{p}']>0).all()
                weights=a[f'weights_{p}']
                np.testing.assert_allclose(weights.sum(-1),1,atol=2e-7)
                assert (weights[~a['available'][:,0],:,1]==0).all()
                assert (weights[~a['available'][:,1],:,2]==0).all()
                if c!='no_feedback':
                    np.testing.assert_array_equal(clean[f'q_{p}'][:30],a[f'q_{p}'][:30])
                    np.testing.assert_array_equal(clean[f'weights_{p}'][:30],weights[:30])
    assert set(TRAIN).isdisjoint((*DEV,CALIBRATION,*FINAL))
    assert set(DEV).isdisjoint((CALIBRATION,*FINAL))
    assert len(set((*DEV,CALIBRATION,*FINAL)))==6
    for name in ('gate','fixed','action'):
        assert sha(output/f'{name}.pt')==sha(e7.ROOT/f'{name}.pt')
    result=dict(verified_utc=utc(),selected=protocol['selected'],prospective_files=prospective,
        prediction_files=predictions,splits_disjoint=True,original_e7_models_unchanged=True,
        frozen_sources_valid=True,props_fixed=True,all_41_points_finite=True,
        fixed_body_denominator=28,missing_sources_exact_zero=True,positive_finite_joint_uncertainty=True,
        pre_intervention_predictions_identical=True,action_baseline_corruption_independent=True)
    write_json(output/'validation.json',result)
    return result


def summarize(output):
    protocol=read_json(output/'final-protocol.json')
    selected=protocol['selected']
    rows=[read_json(output/'scores'/f'data-{s}.json') for s in FINAL]
    conditions={}
    for c in CONDITIONS:
        modes={}
        for mode in rows[0]['conditions'][c]['modes']:
            scores=[r['conditions'][c]['modes'][mode] for r in rows]
            merged={k:float(np.sqrt(np.mean([s[k]**2 for s in scores]))) for k in
                    ('body_rmse_mm','after_intervention_rmse_mm','actuator_fault_rmse_mm')}
            if mode in PROFILES:
                for key in ('joint_box_coverage','joint_marginal_coverage','interval_mean_width_normalized'):
                    merged[key]=float(np.mean([s[key] for s in scores]))
                for key in ('post_weights','post_weights_per_joint','predicted_bias_normalized','interval_width_per_joint'):
                    merged[key]=np.mean([s[key] for s in scores],0).tolist()
            modes[mode]=merged
        ours,old=modes[selected]['body_rmse_mm'],modes['e7_learned']['body_rmse_mm']
        wins=sum(r['conditions'][c]['modes'][selected]['body_rmse_mm']<r['conditions'][c]['modes']['e7_learned']['body_rmse_mm'] for r in rows)
        deltas=[r['conditions'][c]['modes'][selected]['body_rmse_mm']-r['conditions'][c]['modes']['e7_learned']['body_rmse_mm'] for r in rows]
        conditions[c]=dict(modes=modes,delta_vs_e7_mm=ours-old,relative_gain_vs_e7=1-ours/old,wins_vs_e7=wins,
                           seed_deltas_vs_e7_mm=deltas,max_seed_regression_mm=max(deltas))
    criteria=dict(
        wrong_rgb_gain_at_least_10_percent=conditions['wrong_rgb']['relative_gain_vs_e7']>=.1,
        biased_feedback_gain_at_least_5_percent=conditions['biased_feedback']['relative_gain_vs_e7']>=.05,
        clean_vs_feedback_within_half_mm_all_seeds=all(r['conditions']['clean']['modes'][selected]['body_rmse_mm']-r['conditions']['clean']['modes']['e7_feedback_only']['body_rmse_mm']<=.5 for r in rows),
        clean_vs_e7_within_point3_mm_all_seeds=all(r['conditions']['clean']['modes'][selected]['body_rmse_mm']-r['conditions']['clean']['modes']['e7_learned']['body_rmse_mm']<=.3 for r in rows),
        regressions_over_half_mm=[c for c in CONDITIONS if conditions[c]['delta_vs_e7_mm']>.5],
        conditions_with_any_seed_regression_over_half_mm=[c for c in CONDITIONS if conditions[c]['max_seed_regression_mm']>.5])
    diagnostics={}
    for c in ('clean','wrong_rgb','biased_feedback'):
        old_weights=[]
        for seed in FINAL:
            a=e7.load_npz(output/'predictions'/f'data-{seed}-{c}.npz')
            old_weights.append(a['weights'][40:].mean(0))
        diagnostics[c]=dict(e7_post_weights=np.mean(old_weights,0).tolist(),
                            e8_post_weights=conditions[c]['modes'][selected]['post_weights'])
    return dict(created_utc=utc(),selected=selected,final_seeds=list(FINAL),physical_steps=450,
                conditions=conditions,criteria=criteria,contribution_diagnostics=diagnostics,per_seed=rows)


def figures(output,report,summary):
    selected=summary['selected']
    other=next(p for p in PROFILES if p!=selected)
    fig,ax=plt.subplots(figsize=(12,5.5),layout='constrained')
    x=np.arange(8)
    for offset,mode,color,label in ((-.25,'e7_learned','#d99044','Frozen E7'),(0,selected,'#168f7a','E8 selected'),(.25,other,'#6888b7','E8 other profile')):
        ax.bar(x+offset,[summary['conditions'][c]['modes'][mode]['body_rmse_mm'] for c in CONDITIONS],.24,label=label,color=color)
    ax.set_xticks(x,SHORT,rotation=20,ha='right')
    ax.set_ylabel('Body RMSE (mm) / all 28 fixed body anchors')
    ax.set_title('Three new held-out trajectories / same inputs and evaluation points')
    ax.legend()
    fig.savefig(report/'comparison.png',dpi=150)
    plt.close(fig)
    history=read_json(output/'training-history.json')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for p,trace in history.items():
        axes[0].plot([x['epoch'] for x in trace],[x['selection'] for x in trace],label=p)
        axes[1].plot([x['epoch'] for x in trace],[x['clean'] for x in trace],label=p)
    for ax,title in zip(axes,('Development selection loss','Clean development state/decoder loss')):
        ax.set_yscale('log')
        ax.set_xlabel('Epoch')
        ax.set_title(title)
        ax.legend()
    fig.savefig(report/'training.png',dpi=150)
    plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,7),layout='constrained')
    for p in PROFILES:
        axes[0].plot(x,[summary['conditions'][c]['modes'][p]['joint_box_coverage'] for c in CONDITIONS],marker='o',label=p)
        axes[1].plot(x,[summary['conditions'][c]['modes'][p]['interval_mean_width_normalized'] for c in CONDITIONS],marker='o',label=p)
    axes[0].axhline(.9,color='#333333',ls='--',label='Nominal 90% joint-state box')
    axes[0].set_ylim(0,1.02)
    axes[0].set_ylabel('All 7 joints covered / empirical fraction')
    axes[1].set_ylabel('Mean box width / normalized joint units')
    for ax in axes:
        ax.set_xticks(x,SHORT,rotation=15,ha='right')
        ax.legend()
    fig.savefig(report/'uncertainty.png',dpi=150)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,c in zip(axes,('clean','biased_feedback','wrong_rgb')):
        weights=np.asarray(summary['conditions'][c]['modes'][selected]['post_weights_per_joint'])
        ax.imshow(weights,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
        ax.set_xticks(range(3),['Action','Feedback','Vision'])
        ax.set_yticks(range(7),[f'Joint {i+1}' for i in range(7)])
        ax.set_title(c)
        for i in range(7):
            for j in range(3):
                ax.text(j,i,f'{weights[i,j]:.2f}',ha='center',va='center',color='white' if weights[i,j]>.55 else 'black')
    fig.savefig(report/'joint-weights.png',dpi=150)
    plt.close(fig)
    seed=FINAL[0]
    source=source_for(output,seed)
    meta=read_json(source/'metadata.json')
    truths=[e7.load_npz(source/'evaluation'/r['file']) for r in meta['frames']]
    xyz,mask=np.stack([t['anchor_xyz'] for t in truths]),np.stack([t['anchor_body'] for t in truths])
    for c in ('clean','biased_feedback','wrong_rgb','bias_and_wrong_rgb'):
        a=e7.load_npz(output/'e8-predictions'/f'data-{seed}-{c}.npz')
        b=e7.load_npz(output/'predictions'/f'data-{seed}-{c}.npz')
        fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True,layout='constrained')
        axes[0].stackplot(np.arange(150),a[f'weights_{selected}'].mean(1).T,labels=['Action','Feedback','Vision'],colors=['#6888b7','#d99044','#168f7a'])
        axes[0].set_ylim(0,1)
        axes[0].set_ylabel('Mean of per-joint contributions')
        axes[0].legend(ncol=3)
        for predicted,label,color in ((a[f'xyz_{selected}'],'E8','#168f7a'),(b['prediction_learned'],'E7','#d99044'),(b['prediction_feedback_only'],'Feedback only','#6888b7')):
            err=np.sqrt((np.sum((predicted-xyz)**2,-1)*mask).sum(-1)/28)*1000
            axes[1].plot(err,label=label,color=color)
        axes[1].legend(ncol=3)
        axes[1].set_xlabel('Causal update step')
        axes[1].set_ylabel('Body RMSE (mm)')
        axes[0].set_title(f'{c} / first held-out trajectory')
        for ax in axes:
            ax.axvline(30,ls=':',color='#777777')
            for lo,hi in ((50,75),(110,135)):
                ax.axvspan(lo,hi,alpha=.12,color='#888888')
        fig.savefig(report/f'timeline-{c}.png',dpi=140)
        plt.close(fig)


def replay(output,report,selected,condition):
    seed=FINAL[0]
    source=source_for(output,seed)
    meta=read_json(source/'metadata.json')
    a=e7.load_npz(output/'e8-predictions'/f'data-{seed}-{condition}.npz')
    b=e7.load_npz(output/'predictions'/f'data-{seed}-{condition}.npz')
    font_path=Path('C:/Windows/Fonts/msyh.ttc')
    font=ImageFont.truetype(str(font_path),18) if font_path.exists() else ImageFont.load_default()
    frames=[]
    for i in range(0,150,3):
        rgb=e7.load_npz(source/'after'/meta['frames'][i]['file'])['left']
        canvas=Image.new('RGB',(1200,410),'#101f2a')
        draw=ImageDraw.Draw(canvas)
        panels=((b['prediction_feedback_only'][i],'仅反馈／动作回退','#72a5d9'),
                (b['prediction_learned'][i],'冻结 E7','#e6a652'),
                (a[f'xyz_{selected}'][i],'E8 开发集所选版本','#49d1ad'))
        for col,(xyz,label,color) in enumerate(panels):
            canvas.paste(Image.fromarray(rgb).resize((384,288)),(col*400+8,43))
            draw.text((col*400+12,10),label,fill=color,font=font)
            uv=project(xyz,meta['calibration'])*.6
            for u,v in uv:
                if 0<=u<384 and 0<=v<288:
                    x,y=col*400+8+float(u),43+float(v)
                    draw.ellipse((x-2,y-2,x+2,y+2),fill=color)
        draw.text((12,342),f'{CLABELS[condition]} · 步 {i+1}/150 · 展示当前物理画面及模型输出点',fill='#dce8ee',font=font)
        draw.text((12,374),'三列使用同一当前画面；错误图像分支实际输入为过去 RGB 回放。',fill='#acc6d1',font=font)
        frames.append(canvas)
    frames[0].save(report/f'e8-{condition}.gif',save_all=True,append_images=frames[1:],duration=140,loop=0)
    if condition=='wrong_rgb':
        frames[22].save(report/'preview.jpg',quality=90)


def build(output=ROOT):
    report=output/'report'
    report.mkdir(exist_ok=True)
    verified=audit(output)
    summary=summarize(output)
    selected=summary['selected']
    performance=benchmark(output,selected)
    write_json(report/'summary.json',summary)
    figures(output,report,summary)
    for c in ('wrong_rgb','biased_feedback'):
        replay(output,report,selected,c)
    with (report/'scores.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['seed','condition','mode','body_rmse_mm','after_intervention_rmse_mm','actuator_fault_rmse_mm','joint_box_coverage','interval_mean_width_normalized'])
        for row in summary['per_seed']:
            for c,result in row['conditions'].items():
                for mode,metrics in result['modes'].items():
                    writer.writerow([row['seed'],c,mode,metrics['body_rmse_mm'],metrics['after_intervention_rmse_mm'],metrics['actuator_fault_rmse_mm'],metrics.get('joint_box_coverage',''),metrics.get('interval_mean_width_normalized','')])
    protocol=read_json(output/'final-protocol.json')
    training=read_json(output/'training-freeze.json')
    calibration=read_json(output/'calibration.json')
    tests=read_json(output/'test-result.json') if (output/'test-result.json').exists() else None
    condition_rows=''.join('<tr><th>'+CLABELS[c]+'</th>'+''.join(f'<td>{summary["conditions"][c]["modes"][m]["body_rmse_mm"]:.3f}</td>' for m in (selected,'e7_learned','e7_fixed_learned','e7_action_only','e7_feedback_only','e7_e6_rules',next(p for p in PROFILES if p!=selected)))+f'<td>{summary["conditions"][c]["relative_gain_vs_e7"]:+.1%}</td><td>{summary["conditions"][c]["wins_vs_e7"]}/3</td></tr>' for c in CONDITIONS)
    md_rows='\n'.join('| '+CLABELS[c]+' | '+' | '.join(f'{summary["conditions"][c]["modes"][m]["body_rmse_mm"]:.3f}' for m in (selected,'e7_learned','e7_action_only','e7_feedback_only',next(p for p in PROFILES if p!=selected)))+f' | {summary["conditions"][c]["relative_gain_vs_e7"]:+.1%} | {summary["conditions"][c]["wins_vs_e7"]}/3 |' for c in CONDITIONS)
    criterion_rows=''.join(f'<li>{key}：{value}</li>' for key,value in summary['criteria'].items())
    def criterion(key):
        return '通过' if summary['criteria'][key] else '未通过'
    normal=summary['conditions']['clean']['modes'][selected]
    box_mean=np.mean([summary['conditions'][c]['modes'][selected]['joint_box_coverage'] for c in CONDITIONS])
    worst_coverage=min(CONDITIONS,key=lambda c:summary['conditions'][c]['modes'][selected]['joint_box_coverage'])
    d=summary['contribution_diagnostics']
    explanation=(f"正常／错误图像下，E7 的反馈贡献分别为 {d['clean']['e7_post_weights'][1]:.1%}／{d['wrong_rgb']['e7_post_weights'][1]:.1%}，"
                 f"E8 七关节平均反馈贡献为 {d['clean']['e8_post_weights'][1]:.1%}／{d['wrong_rgb']['e8_post_weights'][1]:.1%}。"
                 f"错误图像下视觉贡献为 E7 {d['wrong_rgb']['e7_post_weights'][2]:.1%}、E8 {d['wrong_rgb']['e8_post_weights'][2]:.1%}；"
                 f"反馈偏差下 E8 平均反馈贡献为 {d['biased_feedback']['e8_post_weights'][1]:.1%}。"
                 "这些是状态更新的贡献诊断，不能单独当作正确识别故障来源的证明；还要看误差、对照和每个关节的变化。")
    widths=''.join('<tr><th>'+CLABELS[c]+'</th>'+f'<td>{summary["conditions"][c]["modes"][selected]["joint_box_coverage"]:.1%}</td><td>{summary["conditions"][c]["modes"][selected]["joint_marginal_coverage"]:.1%}</td><td>{summary["conditions"][c]["modes"][selected]["interval_mean_width_normalized"]:.3f}</td></tr>' for c in CONDITIONS)
    profile_rows=''.join(f'<tr><th>{LABELS[p]}</th><td>{s["parameters"]:,}</td><td>{s["epoch"]}</td><td>{s["selection"]:.6f}</td><td>{calibration["profiles"][p]["factor"]:.3f}</td></tr>' for p,s in training['profiles'].items())
    markdown=f'''# E8 详细优化报告

## 本轮做了什么

针对 E7 错误视觉会抑制健康反馈、整路权重无法处理局部反馈偏差的问题，训练逐关节融合及加入反馈偏差状态的两个候选。开发集选定版本为 **{LABELS[selected]}（{selected}）**，最终测试不参与选择。原有干扰物固定；动作模型、身体解码器、分割、RGB 匹配与逆解均冻结。

偏差状态候选开发分数较高，本轮主结果未启用它；它的失败和独立测试表现作为消融保留。组合优化相对 E7 的比较不能确定每一个超参数的单独贡献，本轮只对偏差分支做了训练消融。

## 网络与参数

- 时序特征从 E7 的 60 维扩展为 99 维：增加反馈—视觉差及绝对差、历史偏差状态和逐关节历史权重；缺失视觉的数值和质量在进入记忆前清零。
- GRU 隐状态从 32 增至 64；输出从 3 个整路贡献变为 7×3 个逐关节贡献。softmax 权重仍是贡献，不是正确概率。
- 偏差分支输出七个归一化偏差提案及七个更新率，更新后反馈为 `f - bias*scale`；提案采用 tanh 限幅 ±0.8 个状态尺度。更新率初始化 sigmoid(-1)，约 0.269，随后学习。无反馈时偏差状态保持。
- 初始权重 logits 为每关节 `[0,2,-2]`；偏差输出头从零修正开始，不在开始时人为注入偏移。
- 辅助权重监督从 E7 的整路候选平均误差变为逐关节候选误差；权重系数从 0.2 改为 0.03，温度从 0.00001 改为 0.0001。附加反馈偏差监督系数 0.05、归一化损失尺度 0.2。
- 训练 {training['configuration']['epochs']} 轮，Adam 学习率 0.002、梯度裁剪 2、每 20 步截断反向传播。状态损失沿用归一化关节 MSE 与冻结身体解码器输出一致性；选模分数为开发平均状态损失＋3×正常损失。
- 不确定性分支预测七个正标准差，归一化 log 标准差限于 [-7,1]；分支输入从时序状态分离梯度，使用误差分离的高斯 NLL，系数 0.001，避免其标签改变状态估计路径。
- 参数量、最优轮次和开发分数详见 [training-freeze.json](../training-freeze.json)。`per_joint` 是去掉偏差状态的训练消融，和 `bias_state` 使用相同数据、训练轮数及选模规则。未使用的偏差头参数仍计入该消融的构造参数量。

E7 构造参数量为 {performance['e7']['constructed_parameters']:,}，E8 为 {performance['e8']['constructed_parameters']:,}。固定四个 CPU 线程、缓存视觉候选下，动作预测＋状态更新＋身体解码的每步中位耗时为 E7 {performance['e7']['median_step_ms']:.3f} ms、E8 {performance['e8']['median_step_ms']:.3f} ms，P95 分别为 {performance['e7']['p95_step_ms']:.3f} 和 {performance['e8']['p95_step_ms']:.3f} ms。该测量不包含 RGB 匹配、渲染或读盘，不能当作端到端实时性能。

## 数据与防止看答案

训练沿用 E7 六条训练轨迹，共 720 个不同物理步骤，增强为 {training['training_sequences']} 条序列。干净实际读数只作为训练教师，偏差标签只用于损失；故障类别、干净教师和几何真值不进入模型。两个开发种子为 {DEV}，独立校准种子为 {CALIBRATION}。E7 旧最终数据未用于训练或选模。

模型选定后才在校准轨迹上拟合一个尺度因子，构造名义 90% 的同时七关节状态区间。随后冻结源码、权重、校准和协议，再采集三个新最终种子 {FINAL}，共 450 个步骤，每条复用八种观测干预。所有基线和 E8 全部预测先保存并哈希，再打开几何评分。相同 28 个身体锚点始终在分母，41 个候选都必须输出有限值。

## 独立测试结果

单位 mm，越低越好；三个等长轨迹先汇总平方误差再开方。改善相对同批数据上的冻结 E7；消融列为另一个开发候选，不能用最终成绩重新选部署模型。

| 条件 | E8 所选 | 冻结 E7 | 仅动作 | 仅反馈／动作回退 | 另一 E8 候选 | 相对 E7 改善 | 种子胜出 |
|---|---:|---:|---:|---:|---:|---:|---:|
{md_rows}

预定错误图像改善 ≥10%：**{criterion('wrong_rgb_gain_at_least_10_percent')}**；反馈偏差改善 ≥5%：**{criterion('biased_feedback_gain_at_least_5_percent')}**。正常相对仅反馈 ≤+0.5 mm、相对 E7 ≤+0.3 mm 的逐种子保护检查分别为 **{criterion('clean_vs_feedback_within_half_mm_all_seeds')}**、**{criterion('clean_vs_e7_within_point3_mm_all_seeds')}**。汇总退化超过 0.5 mm 的条件：{summary['criteria']['regressions_over_half_mm']}；至少一个种子退化超过 0.5 mm 的条件：{summary['criteria']['conditions_with_any_seed_regression_over_half_mm']}。这些是事前容差检查，不是统计非劣效性证明。

完整的固定等权、训练固定权重、E6 规则，以及故障后／隐藏执行延迟区间结果见 [scores.csv](scores.csv) 和 [summary.json](summary.json)。不能把 E7 原报告与本报告的绝对误差直接拼接；这里的 E7 已在同批新数据上重跑。

{explanation}

## 不确定性实际效果

所选模型最终八条件等权汇总的七关节同时覆盖为 **{box_mean:.1%}**，名义目标 90%。原始预测标准差需乘独立校准因子 **{calibration['profiles'][selected]['factor']:.3f}** 才构成此次名义区间，标准差本身不是 90% 边界。逐条件覆盖和区间宽度见 [uncertainty.png](uncertainty.png) 与 summary。校准只针对关节状态误差，既不涵盖身体解码器的固定几何误差，也不是每路传感器正确概率。轨迹帧与同轨迹故障分支相关，不能把 3,600 个回放时步解释为独立样本或严格的任意分布保证。

覆盖最低条件为 **{CLABELS[worst_coverage]}**，实际同时覆盖 **{summary['conditions'][worst_coverage]['modes'][selected]['joint_box_coverage']:.1%}**。汇总覆盖接近目标也不能掩盖某个条件覆盖不足。

## 审计、复现与限制

审计通过 {verified['prospective_files']} 份动作前预测、{verified['prediction_files']} 份最终融合输出，验证冻结哈希、种子隔离、固定干扰物、共同分母、全部有限输出、缺失通道零贡献、干预前一致和动作基线对传感器干预独立。回归测试记录：{str(tests['tests'])+' 项通过' if tests else '尚未写入'}。

本轮仍是固定相机、同一形态、局部动作范围、合成故障下的离线因果融合。动作与视觉共享动作先验，可能相关出错；三路证据共同错误时不保证可辨识。没有新增身体形态、完整位姿、主动探索或实机证据。偏差状态与关节坐标是显式结构，其中的估计参数通过监督学习，不能称为完全无监督自我发现。

复现时使用新的输出目录，依次执行 `python -m embodiment.reliable_cycle train`、`calibrate`、`freeze`、`final`，再执行 `python -m embodiment.reliable_report`。冻结后的目录不得覆盖或回改；训练数据来自原 E6/E7 存档，默认种子和配置在 reliable_cycle 中。
'''
    (report/'详细优化报告.md').write_text(markdown,encoding='utf-8')
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>E8 · 逐关节可靠更新</title>
<style>body{{margin:0;background:#101e29;color:#deebf1;font:16px/1.8 system-ui,'Microsoft YaHei',sans-serif}}main{{max-width:1240px;margin:auto;padding:26px}}section{{background:#192d3a;border:1px solid #365064;border-radius:12px;padding:22px;margin:22px 0}}h1{{font-size:30px}}h2{{font-size:22px}}a{{color:#72d8ba}}img{{width:100%;height:auto}}table{{border-collapse:collapse;white-space:nowrap}}th,td{{padding:9px 12px;border-bottom:1px solid #38505f;text-align:right}}th{{text-align:left}}.scroll{{overflow:auto}}.note{{border-left:3px solid #e7ac57;padding:12px 18px;background:#2b3540}}</style>
<main><p><a href="/">全部实验</a> · <a href="/e7-learned-trust/report/">原 E7</a> · <a href="详细优化报告.md">完整文字报告</a></p><h1>E8：逐关节融合、反馈偏差状态与不确定性</h1>
<p>从训练中学习可靠更新同一个身体状态。部署版本由开发集选择：<b>{LABELS[selected]}（{selected}）</b>。偏差状态候选未被选为主结果，完整消融保留。原有三件干扰物固定，身体／动作模型与视觉前端冻结。</p>
<section><h2>三个新测试种子的结果</h2><p>450 个物理步骤，每步回放八种观测条件。所有方法使用同一视觉候选和 28 个固定身体锚点。单位 mm，越低越好；E7 在这批新数据上重跑。</p>
<div class="scroll"><table><tr><th>条件</th><th>E8 所选</th><th>冻结 E7</th><th>训练固定权重</th><th>仅动作</th><th>仅反馈*</th><th>E6 规则</th><th>另一 E8 候选</th><th>相对 E7 改善</th><th>种子胜出</th></tr>{condition_rows}</table></div>
<p>* 反馈缺失时，反馈基线按原协议回退到独立动作预测。八种条件复用相同物理轨迹，不是八个独立场景。</p><img src="comparison.png" alt="冻结E7和两个E8模型的同批测试比较"></section>
<section><h2>提前规定的目标是否实现</h2><p>错误图像改善 ≥10%：<b>{criterion('wrong_rgb_gain_at_least_10_percent')}</b>；反馈偏差改善 ≥5%：<b>{criterion('biased_feedback_gain_at_least_5_percent')}</b>。</p><p>正常相对仅反馈退化 ≤0.5 mm：{criterion('clean_vs_feedback_within_half_mm_all_seeds')}；正常相对 E7 退化 ≤0.3 mm：{criterion('clean_vs_e7_within_point3_mm_all_seeds')}。这两项按每个种子检查，属于预定容差而非统计非劣效性证明。</p><p class="note">相对 E7 汇总退化超过 0.5 mm 的条件：{summary['criteria']['regressions_over_half_mm']}；至少一个种子退化超过 0.5 mm 的条件：{summary['criteria']['conditions_with_any_seed_regression_over_half_mm']}。完整负面结果保留；不能因部分条件改善就称三路可靠性已解决。</p></section>
<section><h2>改了哪些模型参数</h2><p>输入 60→99 维，GRU 隐状态 32→64；整路 3 个权重→每关节 3 个，共 21 个贡献。新增反馈偏差提案与更新率，用持续状态补偿反馈，而非只能丢弃整路读数。不确定性分支输出七个状态标准差。</p><p>Adam 学习率 0.002，训练 {training['configuration']['epochs']} 轮，20 步截断，梯度裁剪 2。辅助贡献监督系数 0.2→0.03、温度 1e-5→1e-4；偏差监督系数 0.05，不确定性 NLL 系数 0.001。数据量仍为 720 个不同训练物理步骤。</p>
<div class="scroll"><table><tr><th>训练版本</th><th>构造参数量</th><th>开发最优轮次</th><th>开发选择分数</th><th>90% 区间校准因子</th></tr>{profile_rows}</table></div><p>无偏差版本仍构造但不使用偏差头，其参数计入总数。两个版本按同一开发规则选择，最终测试没有改变部署选择。贡献权重不是可信度概率；组合相对 E7 的收益不能归因于某一个单独超参数。</p><p>构造参数 E7 {performance['e7']['constructed_parameters']:,}→E8 {performance['e8']['constructed_parameters']:,}；缓存视觉候选下，每步预测／更新／解码中位 CPU 耗时 {performance['e7']['median_step_ms']:.3f}→{performance['e8']['median_step_ms']:.3f} ms。不含图像匹配和渲染，详见 <a href="../benchmark.json">计时记录</a>。</p><img src="training.png" alt="两个版本的开发损失曲线"><img src="joint-weights.png" alt="逐关节三路平均贡献"></section>
<section><h2>正常和受损条件的更新过程</h2><p>{explanation}</p><img src="timeline-clean.png" alt="正常时序"><img src="timeline-wrong_rgb.png" alt="错误图像时序"><img src="timeline-biased_feedback.png" alt="反馈偏差时序"><img src="timeline-bias_and_wrong_rgb.png" alt="反馈和视觉同时受损时序"><p>上图是七关节贡献的均值，下图是身体误差。竖线为传感器干预起点，灰区为未输入模型的执行延迟。</p></section>
<section><h2>回放：错误图像和偏差反馈</h2><img src="e8-wrong_rgb.gif" alt="错误图像条件同画面对照"><img src="e8-biased_feedback.gif" alt="偏差反馈条件同画面对照"><p>点为各模型的输出；三列展示同一当前物理画面。错误图像分支实际接收过去 RGB，当前画面只用于报告展示。不是把仿真真值作为模型预测。</p></section>
<section><h2>模型知道自己有多不确定吗</h2><p>用独立校准种子 {CALIBRATION} 确定状态区间尺度。名义 90% 的同时七关节区间，最终八条件汇总覆盖为 <b>{box_mean:.1%}</b>；最低条件为 {CLABELS[worst_coverage]}，覆盖 {summary['conditions'][worst_coverage]['modes'][selected]['joint_box_coverage']:.1%}。同时报告宽度，避免用无限放宽区间制造高覆盖。</p><img src="uncertainty.png" alt="实际覆盖率和区间宽度"><div class="scroll"><table><tr><th>条件</th><th>七关节同时覆盖</th><th>单关节平均覆盖</th><th>平均归一化宽度</th></tr>{widths}</table></div><p class="note">这是关节状态误差区间，未包含身体解码器的几何误差。有限合成故障及相关轨迹帧下的经验校准，不能解释为任意新场景的严格覆盖保证。</p></section>
<section><h2>证据和适用边界</h2><p>训练六条轨迹各前120步；两个开发轨迹用于选择；独立校准后冻结，再采三条新测试轨迹。教师和故障类别只用于训练损失／构造观测，不进入模型；评价几何不参与训练与选模。所有最终预测保存哈希后才评分。</p><p>审计 {verified['prospective_files']} 份动作前预测、{verified['prediction_files']} 份融合文件，验证源代码／模型哈希、固定物体、共同分母、有限输出、缺失通道零贡献与干预前一致。{str(tests['tests'])+' 项回归测试通过。' if tests else '回归记录尚未写入。'}</p><p>仍是固定形态、固定相机、局部动作和合成故障下的离线因果融合。视觉匹配依赖动作先验，两路可能相关错误。本轮不增加完整身体形态、朝向、持续身份、主动探索或实机证据。</p><p><a href="summary.json">完整逐种子结果</a> · <a href="scores.csv">CSV</a> · <a href="../final-protocol.json">冻结协议</a> · <a href="../training-freeze.json">训练与参数</a> · <a href="../calibration.json">校准记录</a> · <a href="../validation.json">审计</a></p></section></main></html>'''
    (report/'index.html').write_text(html,encoding='utf-8')
    print('Report',report.resolve(),flush=True)
    return summary


if __name__=='__main__':
    build()
