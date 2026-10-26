"""E13 paired comparisons, failure disclosure, figures and interactive replay."""
from __future__ import annotations
import csv
import argparse
import html
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from . import evidence_cycle as run,trust_cycle as e7,extent_cycle_v2 as e9
from .body_cycle import read_json,write_json,sha,utc
from .branch_report import CN as OLD_CN
from .surface_extent import SurfaceAtlas
from .vision import project

CN={**OLD_CN,'hold_no_feedback':'受阻＋反馈缺失'}
LABEL=dict(evidence='E13 视觉选择',e12='E12 同增强观测',e12_original='E12 原搜索预算',
    no_visual='关闭视觉选择',e8='E8B',single_surface='同表面单状态',action='仅动作',feedback='仅反馈')
GOAL_LABEL=dict(accuracy='交叉故障收益与正常保护',ambiguity='竞争真实状态覆盖与集合宽度',
    actual_updates='视觉实际更新与消融收益',recovery='仍受阻时新证据恢复')


def summarize(output):
    score=read_json(output/'scores.json');frames=read_json(output/'frame-scores.json');events=read_json(output/'recovery-events.json')
    by={}
    for c in run.CONDITIONS:
        by[c]={}
        for mode in run.MODES:
            rows=[x for x in score['rows'] if x['condition']==c and x['mode']==mode]
            item={k:float(np.sqrt(np.mean([r[k]**2 for r in rows]))) for k in ('rmse_mm','fault_rmse_mm')}
            item['hold_rmse_mm']=float(np.sqrt(np.mean([r['hold_rmse_mm']**2 for r in rows]))) if c in run.HELD else None
            item['off_surface']=float(np.mean([r['off_surface'] for r in rows]))
            if mode=='evidence':
                for k in ('unknown','coverage','narrow_coverage','width_normalized','visual_updates'):
                    item[k]=float(np.mean([r[k] for r in rows]))
            ev=[x for x in events if x['condition']==c and x['mode']==mode and 'held_absolute' in x]
            if ev:
                times=[x['held_absolute']['time_s'] for x in ev if x['held_absolute']['recovered']]
                item['held_absolute']=dict(events=len(ev),recovered=len(times),within_1s=sum(t<=1.00001 for t in times),
                    median_success_s=float(np.median(times)) if times else None)
            by[c][mode]=item
    improvements={c:1-by[c]['evidence']['fault_rmse_mm']/by[c]['e12']['fault_rmse_mm'] for c in ('hold_feedback_bias','hold_no_feedback')}
    guards=[]
    for seed in run.FINAL:
        guard=dict(seed=seed)
        for c in ('clean','execution_hold'):
            key='hold_rmse_mm' if c=='execution_hold' else 'rmse_mm'
            r={x['mode']:x[key] for x in score['rows'] if x['condition']==c and x['seed']==seed}
            guard[c+'_regression_mm']=r['evidence']-r['e12']
        healthy=next(x for x in score['rows'] if x['condition']=='clean' and x['seed']==seed and x['mode']=='evidence')
        guard['normal_unknown']=healthy['unknown'];guard['normal_width']=healthy['width_normalized']
        guards.append(guard)
    macro={m:float(np.mean([by[c][m]['fault_rmse_mm'] for c in ('both','coherent_replay')])) for m in run.MODES}
    deterioration=macro['evidence']/macro['e12']-1
    twins=score['twins'];n=sum(t['distinguishable_physical_frames'] for t in twins)
    coverage=sum(t['evidence_both_covered'] for t in twins)/n
    narrow=sum(t['narrow_both_covered'] for t in twins)/n
    old=sum(t['e12_both_covered'] for t in twins)/n
    unknown=sum(t['unknown_frames'] for t in twins)/n
    calibration=read_json(output/'calibration.json');radii=np.array(calibration['radius_normalized'])
    audit=read_json(output/'audit.json');absolute=by['twin_held']['evidence']['held_absolute']
    goals=dict(accuracy=dict(passed=bool(all(v>=.5 for v in improvements.values()) and deterioration<=.03 and
        all(r[c+'_regression_mm']<=.5 for r in guards for c in ('clean','execution_hold'))),
        improvements=improvements,normal_guards=guards,common_fault_deterioration=deterioration),
        ambiguity=dict(passed=bool(all(t['unknown_rate']>=.9 and t['evidence_coverage']>=.9 and t['evidence_width']<=.8 for t in twins)
            and all(r['normal_unknown']<=.1 and r['normal_width']<=.35 for r in guards)
            and np.max(radii.mean(-1))<=.25),frames=n,simultaneous_coverage=coverage,narrow_same_bank_coverage=narrow,
            matched_e12_coverage=old,unknown_rate=unknown,normal_unknown=by['clean']['evidence']['unknown'],
            normal_width=by['clean']['evidence']['width_normalized'],largest_mean_branch_radius=float(radii.mean(-1).max()),twins=twins),
        actual_updates=dict(passed=bool(audit['visual_selections_differing_from_ablation']>0 and
            audit['visually_corrected_future_differs_from_open_action']>0 and
            all(by[c]['evidence']['fault_rmse_mm']<by[c]['no_visual']['fault_rmse_mm'] for c in improvements)),
            selections=audit['visual_selections'],changes_vs_ablation=audit['visual_selections_differing_from_ablation'],
            changed_future=audit['visually_corrected_future_differs_from_open_action']),
        recovery=dict(passed=absolute['within_1s']==6,**absolute))
    result=dict(created_utc=utc(),conditions=by,goals=goals,macro_errors=macro,final_seeds=run.FINAL,
        configuration=read_json(output/'development-selection.json'),calibration=calibration,audit=audit,
        network_retrained=False,candidate_sets_are_not_probabilities=True)
    if (output/'diagnostics.json').exists():result['diagnostics']=read_json(output/'diagnostics.json')
    write_json(output/'report'/'summary.json',result)
    return result,frames,events


def figures(report,s,rows):
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False})
    fig,axes=plt.subplots(1,2,figsize=(16,5.7))
    for ax,conditions in zip(axes,(run.CONDITIONS[:7],run.CONDITIONS[7:])):
        modes=('evidence','e12','single_surface','action') if ax is axes[0] else ('evidence','e12','e12_original','feedback')
        for j,m in enumerate(modes):
            ax.bar(np.arange(len(conditions))+(j-1.5)*.19,[s['conditions'][c][m]['fault_rmse_mm'] for c in conditions],.19,label=LABEL[m])
        ax.set_xticks(np.arange(len(conditions)),[CN[c] for c in conditions],rotation=32,ha='right',fontsize=8)
        if ax is axes[0]:
            values=[s['conditions'][c][m]['fault_rmse_mm'] for c in conditions for m in modes]
            ax.set_ylim(max(0,min(values)-1),max(values)+1)
        ax.set(ylabel='传感故障窗口固定锚点RMSE / mm');ax.legend(fontsize=8)
        ax.set_title('正常物理轨迹条件（纵轴局部放大）' if ax is axes[0] else '持续受阻与同输入配对条件')
    fig.tight_layout();fig.savefig(report/'errors.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True)
    for m in ('evidence','e12','single_surface','feedback'):
        selected=[r for r in rows if r['seed']==run.FINAL[0] and r['condition']=='hold_no_feedback' and r['mode']==m]
        axes[0].plot(np.arange(180)*.1,[r['error_mm'] for r in selected],label=LABEL[m])
    selected=[r for r in rows if r['seed']==run.FINAL[0] and r['condition']=='hold_no_feedback' and r['mode']=='evidence']
    axes[1].step(np.arange(180)*.1,[r['unknown'] for r in selected],label='未知')
    axes[1].step(np.arange(180)*.1,[r['visual_selected'] for r in selected],label='新视觉支持',alpha=.7)
    axes[2].plot(np.arange(180)*.1,[r['width_normalized'] for r in selected],label='候选包络半宽')
    for ax in axes:
        ax.axvspan(3,15,color='#8bbbe0',alpha=.14)
        for lo,hi in ((3,7),(11,14)):ax.axvspan(lo,hi,color='orange',alpha=.17)
        ax.axvline(7,color='green',ls=':');ax.axvline(14,color='green',ls=':');ax.legend(fontsize=8)
    axes[0].set(ylabel='实体锚点误差 / mm');axes[2].set(xlabel='时间 / s',ylabel='归一化半宽')
    fig.tight_layout();fig.savefig(report/'timeline.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    ts=s['goals']['ambiguity']['twins']
    for j,(k,label) in enumerate((('e12_coverage','E12 原余量'),('narrow_coverage','E13候选＋原余量'),('evidence_coverage','E13分支校准'))):
        axes[0].bar(np.arange(3)+(j-1)*.25,[t[k]*100 for t in ts],.25,label=label)
    axes[0].axhline(90,ls='--',color='gray');axes[0].set_xticks(range(3),[str(t['seed']) for t in ts]);axes[0].set(ylabel='两真实构型同时覆盖 / %');axes[0].legend(fontsize=8)
    axes[1].bar(range(6),np.array(s['calibration']['radius_normalized']).mean(-1))
    axes[1].axhline(.25,ls='--',color='gray');axes[1].set_xticks(range(6),['开环','局部运动','维持命令','衰减停止','反馈','已解析主状态']);axes[1].set(ylabel='平均归一化半径；非概率区间')
    fig.tight_layout();fig.savefig(report/'sets.png',dpi=150);plt.close(fig)


def document(output,report,s):
    g=s['goals'];cfg=s['configuration'];profile=cfg['profiles'][cfg['selected']]
    a=s['audit'];lines=['# E13：视觉选择身体状态与分支误差校准','',
        '完成日期：2026-10-25。旧网络权重未重训；优化对象是视觉搜索、显式状态推断和候选误差范围。原干扰物、屏和相机保持固定。','',
        '[交互回放](http://127.0.0.1:8767/e13-visual-evidence/report/) · [执行前协议](body-evidence-e13.md)','',
        '## 本轮结果','','下列为三个独立测试种子的均方汇总RMSE；十三条件共享六条物理轨迹。','',
        '| 条件／故障窗口 | E12同增强观测/mm | E13/mm | 改善 |','|---|---:|---:|---:|']
    for c,v in g['accuracy']['improvements'].items():
        lines.append(f"| {CN[c]} | {s['conditions'][c]['e12']['fault_rmse_mm']:.3f} | {s['conditions'][c]['evidence']['fault_rmse_mm']:.3f} | {v*100:.2f}% |")
    lines+=['',f"首个不可辨识窗口有{g['ambiguity']['frames']}个隐藏身体分歧>15mm的帧：新包络同时覆盖两真状态 **{g['ambiguity']['simultaneous_coverage']*100:.2f}%**；同一E13候选使用旧余量 **{g['ambiguity']['narrow_same_bank_coverage']*100:.2f}%**；同批增强观测E12 **{g['ambiguity']['matched_e12_coverage']*100:.2f}%**。这轮覆盖对照使用全新种子，不能直接拿历史E12的29.36%相减。",'',
        f"正常未知 **{g['ambiguity']['normal_unknown']*100:.2f}%**，正常平均归一化半宽 **{g['ambiguity']['normal_width']:.4f}**。共同受损／协同回放宏误差：E12 **{s['macro_errors']['e12']:.3f}** → E13 **{s['macro_errors']['evidence']:.3f}mm**。",'',
        '| 预先验收 | 结果 |','|---|---|']
    for k,v in g.items():lines.append(f"| {GOAL_LABEL[k]} | {'通过' if v['passed'] else '未通过'} |")
    lines+=['','## 怎样改的','',
        'E12的第一处缺口是：当前反馈不可信时，即使图像更支持停止历史，也没有独立视觉选择主状态的通路。第二处是：动作和反馈都错时，图像匹配仍只围绕这两个位置搜索，健康图像可能根本没有产生足够的正确对应。第三处是：健康融合状态的误差余量，不能覆盖开环运动历史的累计动力学误差。','',
        'E13沿用冻结的动作、身体、图谱和逐关节融合网络。一个冻结E12中立观察器在当前观测之前预测停止历史，用这两个位置增加图像搜索起点。82片段仍各自只保留一个按图像光度选择的真实双目匹配；多次搜索不是多份证据。每个条件的测量预先生成，E13与E12及单状态表面对照使用同一批测量。','',
        '实验先记录物理轨迹，再按因果顺序离线回放：每步先用历史和当前命令forecast，再读该步反馈与视觉finish。当前及未来观测不进入这一步先验；这不等于已经部署了实机在线控制。评价几何始终与推理输入分开，全部最终预测封存后才由评分阶段读取。','',
        '视觉选择在同一个学习身体坐标系中比较四条事先预测的运动历史。两组固定图像片段分别要求足够支持、低残差、对明显不同的运动解释有残差优势，且连续多帧成立。反馈与图像形成更好的共同解释时拒绝强行确认旧物理分支。两片段子集共享相机，不能被描述成两个独立传感器或可靠性概率。当前通路选择已有历史支持的构型，没有直接从图像估计任意七关节姿态。','',
        '只有图像时也可选择停止历史、实际更新主状态及未来预测，独立开环动作历史仍保留。只有一套可能受损的图像证据时不解除来源不确定性；未知和候选继续保留。图像随后中断时传播已选历史，但不伪造新视觉观测。状态纠错和确定故障来源是不同结论。','',
        '额外停止搜索将最多初始化次数由2增至4，计算预算确有增加，不能宣称整个系统计算预算相同。原两次搜索E12对照单列；同增强观测E12和关闭视觉选择对照用于判断推理贡献。','',
        '## 参数、开发选择与独立校准','',
        f"旧开发2026103211仅作工程检查；正式开发 **{run.DEV}**，独立校准 **{run.CAL}**，最终 **{run.FINAL}**。仅四组视觉参数，选择 **{cfg['selected']}**：连续 **{profile['visual_confirm']}帧**，有界表面纠错增益 **{profile['visual_gain']}**。E12反馈确认2帧、关联倍率1.5保持冻结。",'',
        '| 视觉参数 | 开发选择损失 | 连续确认 | 表面增益 |','|---|---:|---:|---:|']
    for name,p in cfg['profiles'].items():lines.append(f"| {name} | {p['selection']:.8f} | {p['visual_confirm']} | {p['visual_gain']} |")
    t=s['calibration']['thresholds']
    lines+=['',f"基础空间关联{t['association_m']*1000:.3f}mm、关节关联{t['association_q']:.5f}，均乘1.5；表面残差≤{t['surface_m']*1000:.3f}mm。上述三门限沿用E12开发校准，新校准种子只估计分支误差半径。两组各≥6匹配，组胜者与总胜者身体距离<12mm；竞争分支距离>15mm，两组各有≥3mm残差优势。反馈比候选更贴合图像1mm以上时否决视觉选择。",'',
        '开发选择使用记录的实际关节读数作教师，不读取CAD评价几何。损失权重：正常、健康受阻、受阻反馈偏差及受阻反馈缺失各8；共同受损和协同回放各4，其余1。不同条件共享轨迹，因此这些项不是独立样本。','',
        '误差半径只用单独校准轨迹：运动误差来自正常开环历史，停止误差来自受阻配对事先停止历史，主状态误差来自健康正常主状态。每关节95%误差分位与0.02取大后，再以“七关节最大标准化误差”的99.5%分位乘固定1.25安全因子。局部运动沿用开环半径；反馈固定0.02。全部在最终采集前哈希冻结。','',
        '| 候选 | 七关节归一化半径（无量纲） | 平均 |','|---|---|---:|']
    for name,rs in zip(('开环命令','局部响应','维持旧命令','衰减停止','反馈候选','主状态'),s['calibration']['radius_normalized']):
        lines.append(f"| {name} | {', '.join(f'{v:.5f}' for v in rs)} | {np.mean(rs):.5f} |")
    lines+=['','包络是逐候选加半径后的矩形外包络，不是概率置信区间；覆盖率的改善包含扩大误差范围的作用。矩形内部不保证每个构型都可实现；一个校准种子、相关时间帧也没有有限样本覆盖保证。必须同时看候选距离、宽度、正常可用性及定位误差。','',
        '## 完整同批故障窗口结果 / mm','','| 条件 | E13 | E12增强 | E12原预算 | 关视觉 | E8B | 单状态表面 | 仅动作 | 仅反馈 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c in run.CONDITIONS:lines.append('| '+CN[c]+' | '+' | '.join(f"{s['conditions'][c][m]['fault_rmse_mm']:.3f}" for m in run.MODES)+' |')
    lines+=['','主表统一为30–69、110–139帧故障窗口。健康持续受阻完整30–149帧的E13/E12误差分别为 '+f"{s['conditions']['execution_hold']['evidence']['hold_rmse_mm']:.3f}/{s['conditions']['execution_hold']['e12']['hold_rmse_mm']:.3f}mm。整体、表面离体率及逐种子结果全部保留在seed-scores.json；434表面点没有因难度被删除。",'',
        '| 交叉条件 | E12全180帧离体比例 | E13全180帧离体比例 |','|---|---:|---:|']
    for c in ('hold_feedback_bias','hold_no_feedback'):
        lines.append(f"| {CN[c]} | {s['conditions'][c]['e12']['off_surface']*100:.3f}% | {s['conditions'][c]['evidence']['off_surface']*100:.3f}% |")
    lines+=['','离体定义为434预测片段到真实几何表面距离>15mm。图谱本身没有增加，改善来自相同身体表示的状态更准确；不能将离体减少解释为发现了更多身体。','',
        '| 种子 | 受阻反馈偏差E12/mm | E13/mm | 受阻反馈缺失E12/mm | E13/mm |','|---|---:|---:|---:|---:|']
    seedrows=read_json(output/'scores.json')['rows']
    for seed in run.FINAL:
        values=[next(r['fault_rmse_mm'] for r in seedrows if r['seed']==seed and r['condition']==c and r['mode']==m)
            for c in ('hold_feedback_bias','hold_no_feedback') for m in ('e12','evidence')]
        lines.append('| '+str(seed)+' | '+' | '.join(f'{v:.3f}' for v in values)+' |')
    lines+=['','## 正常保护与竞争状态覆盖','','| 种子 | 正常回归/mm | 健康受阻30–149帧回归/mm | 正常未知 | 正常半宽 |','|---|---:|---:|---:|---:|']
    for r in g['accuracy']['normal_guards']:lines.append(f"| {r['seed']} | {r['clean_regression_mm']:+.4f} | {r['execution_hold_regression_mm']:+.4f} | {r['normal_unknown']*100:.2f}% | {r['normal_width']:.4f} |")
    lines+=['','| 种子 | 隐藏身体分歧帧 | 未知 | E12两状态覆盖 | E13旧余量覆盖 | E13新余量覆盖 | 新包络半宽 |','|---|---:|---:|---:|---:|---:|---:|']
    for r in g['ambiguity']['twins']:
        lines.append(f"| {r['seed']} | {r['distinguishable_physical_frames']} | {r['unknown_rate']*100:.1f}% | {r['e12_coverage']*100:.1f}% | {r['narrow_coverage']*100:.1f}% | {r['evidence_coverage']*100:.1f}% | {r['evidence_width']:.4f} |")
    lines+=['','| 种子 | 运动最近候选中位/mm | 95%/mm | 停止最近候选中位/mm | 95%/mm |','|---|---:|---:|---:|---:|']
    for r in a['nearest_motion_candidates']:
        lines.append('| '+str(r['seed'])+' | '+' | '.join(f'{r[k]:.3f}' for k in ('moving_nearest_median_mm','moving_nearest_p95_mm','held_nearest_median_mm','held_nearest_p95_mm'))+' |')
    if 'diagnostics' in s:
        diagnostics=s['diagnostics']['rows'];moving=[r for r in diagnostics if r['physical']=='moving'];held=[r for r in diagnostics if r['physical']=='held']
        lost=[i for r in moving for i in r['failed_frames']]
        counts=np.sum([r['failed_joint_counts'] for r in moving],axis=0);worst=int(counts.argmax())
        lines+=['','## 未过线的原因：运动历史的延迟误差仍欠覆盖','',
            f"停止真实状态覆盖{sum(r['covered'] for r in held)}/{sum(r['frames'] for r in held)}帧；运动状态覆盖{sum(r['covered'] for r in moving)}/{sum(r['frames'] for r in moving)}帧。运动遗漏{len(lost)}帧，其中{sum(50<=i<75 for i in lost)}帧位于50–74帧实际60ms执行延迟窗口内。索引{worst}（第{worst+1}控制通道）越界{int(counts[worst])}帧。",'',
            '单独校准轨迹同样含60ms延迟，但其动作路径未覆盖三个新轨迹的全部动力学残差。按运动／停止分组有改善，仍不足以表达动作幅度、执行延迟和时间相关误差。这不是停止解释被删除，也不能通过报告未知率100%来掩盖。',
            '', '下面为最终封存后的失败诊断，不改变半径、种子或验收目标。下一步若研究延迟条件下的预测误差，需要新开发与新独立测试，不能在本批最终数据上扩大半径后宣称达标。','',
            '| 种子 | 实际运动状态覆盖 | 停止状态覆盖 | 运动遗漏帧 | 各关节越界帧数0–6 |','|---|---:|---:|---|---|']
        for m,h in zip(moving,held):
            lines.append(f"| {m['seed']} | {m['covered']}/{m['frames']} | {h['covered']}/{h['frames']} | {m['failed_frames']} | {m['failed_joint_counts']} |")
    lines+=['','## 新证据、实际提交与恢复','',
        f"实际视觉选择{a['visual_selections']}次，其中{a['visual_selections_differing_from_ablation']}次主状态与无视觉选择消融不同，{a['visually_corrected_future_differs_from_open_action']}次后续预测与独立开环动作不同。无视觉选择消融与同增强观测E12的状态轨迹逐位完全一致，说明收益不是更改基线参数制造的。",'',
        f"其中{a['visual_selections_inside_fault_window']}次视觉选择发生在传感故障尚未解除的窗口内。下表是封存后的时序诊断，不是另设的新验收门限；第一次选择仍需之后的误差评分验证，不能等同于绝对恢复。",'',
        '| 种子 | 交叉条件 | 故障开始帧 | 窗口内视觉更新数 | 第一次更新延迟/s |','|---|---|---:|---:|---:|']
    for ev in a['visual_acquisition_events']:
        delay=ev['first_update_delay_s']
        lines.append(f"| {ev['seed']} | {CN[ev['condition']]} | {ev['start']} | {ev['visual_selections']} | {'—' if delay is None else f'{delay:.2f}'} |")
    lines+=['',
        f"保留了{a['prediction_files']}份最终预测、{a['command_files']}份执行前命令；同输入配对前70帧全部输入、测量与所有方法预测逐字段一致。原两起点观测在{a['original_observer_exact_comparisons']}帧实际图像上与冻结实现逐字段相等；额外停止搜索赢得{a['stop_seed_winning_patch_frames']}/{a['valid_patch_frames']}个有效片段帧的光度竞争。这是观测来源记录，不是独立证据数量。",'',
        '| 新证据返回条件 | 方法 | 绝对恢复 | 1秒内 | 成功中位/s |','|---|---|---:|---:|---:|']
    for c in ('twin_held','hold_feedback_bias','hold_no_feedback','hold_no_camera'):
        for m in ('evidence','e12','single_surface','feedback'):
            r=s['conditions'][c][m]['held_absolute'];v=r['median_success_s']
            lines.append(f"| {CN[c]} | {LABEL[m]} | {r['recovered']}/{r['events']} | {r['within_1s']} | {'—' if v is None else f'{v:.2f}'} |")
    lines+=['','恢复检查在70–109、140–149帧进行，身体仍持续受阻。要求与实际状态的学习身体距离≤5mm、实体锚点误差≤实际反馈解码误差+2mm并连续5帧。不是仅凭回到健康估计就算成功。','',
        '## 未解决的事情与结论边界','',
        '仅相机证据的选择仍可能被回放、重复外观、错误匹配诱导；两个空间子集和连续帧不能保证来源独立。当前来源冲突处理仍是显式门限与历史假设，没有训练完整联合故障模型，也没有校准故障概率。','',
        f"受阻叠加相机缺失仍为 **{s['conditions']['hold_no_camera']['evidence']['fault_rmse_mm']:.3f}mm**，仅反馈 **{s['conditions']['hold_no_camera']['feedback']['fault_rmse_mm']:.3f}mm**。缺少新图像时保守规则无法确认新的停止状态；优化视觉通路并未解决无视觉受阻。协同回放宏误差E13 {s['macro_errors']['evidence']:.3f}mm，独立仅动作 {s['macro_errors']['action']:.3f}mm，仍需用强对照判断净收益。",'',
        '只校准一个轨迹种子，不能推论任意时延、任意受阻时间或外力移动下仍可靠。停止历史在首次冲突锚定；本实验故障和受阻起始同步，未知期间晚些时候才发生受阻尚未验证。三最终种子仅六条物理轨迹，十三条件共享轨迹，不是39个独立物理重复。','',
        '本轮没有扩大形体图谱、验证完整连杆朝向、重新发现身体归属或迁移实机。28实体锚点与434表面点只评价当前冻结身体映射的一部分；健康反馈的解码误差仍是明显下限。结论应限定为：当前受控仿真中，视觉可以在反馈不可靠时参与同一身体状态的纠错，并可以对保留运动解释的误差范围做更合适的经验校准。','',
        '## 复现','','```powershell',r'.\.venv\Scripts\python.exe -B -X utf8 -m embodiment.evidence_cycle develop',r'.\.venv\Scripts\python.exe -B -X utf8 -m embodiment.evidence_cycle freeze',r'.\.venv\Scripts\python.exe -B -X utf8 -m embodiment.evidence_cycle final',r'.\.venv\Scripts\python.exe -B -X utf8 -m embodiment.evidence_cycle score',r'.\.venv\Scripts\python.exe -B -X utf8 -m embodiment.evidence_audit',r'.\.venv\Scripts\python.exe -B -X utf8 -m embodiment.evidence_diagnostics',r'.\.venv\Scripts\python.exe -B -X utf8 -m embodiment.evidence_report','```','',
        '完整重新采集须为各阶段传入新的 --output 目录；冻结数据和评价分数拒绝覆盖。']
    if (output/'test-result.json').exists():
        tests=read_json(output/'test-result.json');lines+=['',f"回归验证：{tests['tests']}项通过，本轮新增{tests['new_tests']}项。"]
    text='\n'.join(lines)+'\n';(report/'详细报告.md').write_text(text,encoding='utf8');Path('docs/body-evidence-e13-summary.md').write_text(text,encoding='utf8')


def replay(output,report,rows):
    seed=run.FINAL[0];atlas=SurfaceAtlas.load(e9.ROOT/'atlas-multi.npz')
    ids=np.linspace(0,len(atlas.weights)-1,100,dtype=int)
    cal=read_json(run.source_for(output,seed,'clean')/'metadata.json')['calibration']
    folder=report/'frames';folder.mkdir(exist_ok=True)
    for kind,c in (('normal','clean'),('held','execution_hold')):
        source=run.source_for(output,seed,c)
        for i in sorted(set(range(0,180,2))|set(range(20))):
            rgb=e7.load_npz(source/'rgb'/f'{i:05d}.npz')
            Image.fromarray(rgb['left']).resize((480,360)).save(folder/f'{kind}-{i:03d}.jpg',quality=83)
    lookup={(r['condition'],r['frame']):r for r in rows if r['seed']==seed and r['mode']=='evidence'}
    def uv(q):
        return np.clip(np.nan_to_num(project(atlas.predict(q)[ids],cal)*.75),-1500,1500).round(1).tolist()
    data=dict(seed=seed,labels=CN,modes=LABEL,conditions={})
    for c in run.CONDITIONS:
        p=e7.load_npz(output/'predictions'/f'{seed}-{c}.npz');a=e7.load_npz(output/'sensors'/f'{seed}-{c}.npz')
        kind='held' if c in run.HELD else 'normal';items=[]
        for i in range(0,180,2):
            path=Path(a['rgb_paths'][i]) if a['rgb_paths'][i] else None;r=lookup[c,i]
            items.append(dict(i=i,current=f'frames/{kind}-{i:03d}.jpg',
                image=None if path is None else f'frames/{"held" if "-held" in str(path) else "normal"}-{int(path.stem):03d}.jpg',
                unknown=r['unknown'],covered=r['covered'],width=r['width_normalized'],error=r['error_mm'],
                override=r['override_action'],visual=r['visual_selected'],support=r['support'],
                points={m:uv(p[f'q_{m}'][i]) for m in run.MODES},
                alternatives=[uv(q) for q,v in zip(p['bank'][i,:4],p['bank_valid'][i,:4]) if v]))
        data['conditions'][c]=items
    (report/'replay.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')),encoding='utf8')
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17);animation=[]
    for row in data['conditions']['hold_no_feedback']:
        im=Image.new('RGB',(960,430),'#101b29');d=ImageDraw.Draw(im)
        im.paste(Image.open(report/row['current']),(0,38))
        if row['image']:im.paste(Image.open(report/row['image']),(480,38))
        d.text((12,8),'实际身体＋E13主状态与保留解释',font=font,fill='white')
        d.text((492,8),'模型收到的相机',font=font,fill='white')
        for points,color in list(zip(row['alternatives'],['#ffbd63','#62bfff','#c294ff','#ed91b2']))+[(row['points']['evidence'],'#55ed9c')]:
            for x,y in points:
                if 0<=x<480 and 0<=y<360:d.ellipse((x-1.5,y+36.5,x+1.5,y+39.5),fill=color)
        d.text((12,402),f"t={row['i']*.1:.1f}s  未知={row['unknown']}  新视觉支持={row['visual']}  实体误差={row['error']:.2f}mm",font=font,fill='white')
        animation.append(im)
    animation[0].save(report/'body-evidence.gif',save_all=True,append_images=animation[1:],duration=160,loop=0)
    animation[30].save(report/'preview.jpg',quality=90)


def webpage(report,s):
    goals=''.join(f'<li>{GOAL_LABEL[k]}：{"通过" if v["passed"] else "未通过"}</li>' for k,v in s['goals'].items())
    table=''.join('<tr><td>'+CN[c]+'</td>'+''.join(f"<td>{s['conditions'][c][m]['fault_rmse_mm']:.2f}</td>" for m in run.MODES)+'</tr>' for c in run.CONDITIONS)
    intro='；'.join(f"{CN[c]}：{s['conditions'][c]['e12']['fault_rmse_mm']:.2f} → {s['conditions'][c]['evidence']['fault_rmse_mm']:.2f}mm" for c in ('hold_feedback_bias','hold_no_feedback'))
    intro+=f"；两真状态同时包络覆盖{s['goals']['ambiguity']['simultaneous_coverage']*100:.1f}%。"
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>E13 · 视觉证据与身体状态纠错</title><style>*{box-sizing:border-box}body{margin:0;background:#101b29;color:#e8f0f6;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1240px;margin:auto;padding:26px 20px 60px}section{padding:20px;margin:22px 0;background:#192b3d;border:1px solid #345069;border-radius:12px}a{color:#75dac3}p{color:#c5d3e2}button,select{background:#28435a;color:white;border:1px solid #608198;border-radius:6px;padding:9px;margin:4px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}canvas,img{width:100%;border-radius:8px;background:#080e16}input[type=range]{width:100%}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;border-bottom:1px solid #3e5368;text-align:left}.scroll{overflow-x:auto}#stats{white-space:pre-line}@media(max-width:750px){.grid{grid-template-columns:1fr}main{padding:15px}}</style><main><p><a href="/">← 实验入口</a> · <a href="详细报告.md">完整报告</a> · <a href="summary.json">全部指标</a> · <a href="body-evidence-e13.md">执行前协议</a> · <a href="/e12-branch-body/report/">历史E12</a></p><h1>反馈不可靠时，视觉能改变身体状态吗？</h1><p>旧网络冻结 · 新开发与单独校准 · 3个新测试种子／6条物理轨迹 · 13种共享条件</p><section><h2>本轮结果</h2><p>INTRO</p><ul>GOALS</ul><p>视觉选择可以更新同一个身体状态，但单一图像来源不能消除故障来源歧义。误差包络是经验候选集合，不是校准概率或完整实体。</p></section><section><h2>实际身体与收到的观测</h2><select id="condition"></select><select id="mode"></select><button id="play">暂停</button><label><input type="checkbox" id="branches" checked>显示竞争运动解释</label><div class="grid"><div><p>实际身体＋所选预测表面</p><canvas id="actual" width="480" height="360"></canvas></div><div><p>模型收到的相机</p><canvas id="sensor" width="480" height="360"></canvas></div></div><input id="frame" type="range" min="0" max="89" value="0"><p id="stats">加载中…</p><p>绿色为所选方法，其他颜色为未排除的运动解释。显示100点，实体评分保持28锚点和全部434表面点。3–15秒持续受阻；7–11秒和14–15秒新证据已恢复，身体仍受阻。</p></section><section><h2>故障窗口实体误差 / mm</h2><div class="scroll"><table><tr><th>条件</th>HEADERS</tr>TABLE</table></div><img src="errors.png"></section><section><h2>反馈缺失时，视觉选择与未知</h2><img src="timeline.png"></section><section><h2>集合校准的收益与代价</h2><img src="sets.png"></section><section><h2>方法、来源与限制</h2><p>同一图像每片段只保留一个匹配；增加停止搜索使初始化预算2→4，并单列原预算E12。增强观测E12及关闭视觉选择使用相同测量，检验推理贡献。相机缺失下的新停止识别、任意受阻起点、实机泛化和完整形体仍未解决。</p><p><a href="seed-scores.json">逐种子全部结果</a> · <a href="frame-scores.csv">逐帧分数</a> · <a href="recovery-events.csv">恢复事件</a> · <a href="audit.json">因果与来源审计</a> · <a href="protocol-freeze.json">冻结哈希</a> · <a href="test-result.json">回归测试</a> · <a href="body-evidence.gif">动画</a></p></section></main><script>let data,busy=false,playing=true;const $=id=>document.getElementById(id),cache=new Map();function load(path){if(!path)return Promise.resolve(null);if(cache.has(path))return cache.get(path);const p=new Promise(resolve=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=()=>resolve(null);im.src=path});cache.set(path,p);return p}async function draw(){if(!data||busy)return;busy=true;try{const r=data.conditions[$('condition').value][+$('frame').value],mode=$('mode').value;const imgs=await Promise.all([load(r.current),load(r.image)]);['actual','sensor'].forEach((id,j)=>{const c=$(id).getContext('2d');c.fillStyle='#080e16';c.fillRect(0,0,480,360);if(imgs[j])c.drawImage(imgs[j],0,0,480,360);else{c.fillStyle='white';c.font='20px sans-serif';c.fillText('相机缺失',180,180)}});const c=$('actual').getContext('2d');function dots(points,color){c.fillStyle=color;for(const [x,y] of points){c.beginPath();c.arc(x,y,1.8,0,Math.PI*2);c.fill()}}if($('branches').checked)r.alternatives.forEach((p,j)=>dots(p,['#ffbd63','#62bfff','#c294ff','#ed91b2'][j]));dots(r.points[mode],'#55ed9c');$('stats').textContent=`t=${(r.i*.1).toFixed(1)}s · ${r.unknown?'未知：保留竞争解释':'已选择观测支持的状态'}\nE13实体锚点误差 ${r.error.toFixed(2)}mm；新视觉支持=${r.visual}；表面匹配=${r.support}/82\n包络覆盖实际构型=${r.covered}；归一化半宽=${r.width.toFixed(4)}。切换方法仅改变绿色投影，上述读数始终为E13。`}finally{busy=false}}fetch('replay.json').then(r=>r.json()).then(x=>{data=x;for(const [k,v] of Object.entries(x.labels))$('condition').add(new Option(v,k));for(const [k,v] of Object.entries(x.modes))$('mode').add(new Option(v,k));$('condition').value='hold_no_feedback';draw()}).catch(e=>$('stats').textContent=e.message);for(const id of ['condition','mode','frame','branches'])$(id).addEventListener('input',draw);$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'暂停':'播放'};setInterval(()=>{if(data&&playing&&!busy){$('frame').value=(+$('frame').value+1)%90;draw()}},180);</script></html>'''
    page=page.replace('INTRO',html.escape(intro)).replace('GOALS',goals).replace('TABLE',table)
    page=page.replace('HEADERS',''.join(f'<th>{v}</th>' for v in LABEL.values()))
    page=page.replace('<a href="audit.json">因果与来源审计</a>',
        '<a href="audit.json">因果与来源审计</a> · <a href="diagnostics.json">未过线原因</a>')
    if not (report/'test-result.json').exists():
        page=page.replace(' · <a href="test-result.json">回归测试</a>','')
    (report/'index.html').write_text(page,encoding='utf8')


def main(output=run.ROOT):
    run.verify(output);report=output/'report';report.mkdir(exist_ok=True)
    s,rows,events=summarize(output)
    for source,target in (('scores.json','seed-scores.json'),('audit.json','audit.json'),('protocol-freeze.json','protocol-freeze.json'),('test-result.json','test-result.json'),('diagnostics.json','diagnostics.json')):
        if (output/source).exists():write_json(report/target,read_json(output/source))
    (report/'body-evidence-e13.md').write_text(run.PROTOCOL.read_text(encoding='utf8'),encoding='utf8')
    for name,records in (('frame-scores.csv',rows),('recovery-events.csv',events)):
        flat=[{(f'{k}_{sub}' if sub else k):v for k,value in row.items() for sub,v in (value.items() if isinstance(value,dict) else [('',value)])} for row in records]
        keys=list(dict.fromkeys(k for row in flat for k in row))
        with (report/name).open('w',newline='',encoding='utf-8-sig') as handle:
            writer=csv.DictWriter(handle,keys);writer.writeheader();writer.writerows(flat)
    figures(report,s,rows);replay(output,report,rows)
    # Add the new visual-selection indicator without changing the frozen E12 renderer.
    data=read_json(report/'replay.json');lookup={(r['condition'],r['frame']):r for r in rows if r['seed']==run.FINAL[0] and r['mode']=='evidence'}
    for c,items in data['conditions'].items():
        for r in items:r['visual']=lookup[c,r['i']]['visual_selected']
    (report/'replay.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')),encoding='utf8')
    document(output,report,s);webpage(report,s)
    write_json(output/'report-artifacts.json',dict(created_utc=utc(),generator_sha256=sha(Path(__file__)),
        hashes={str(p):sha(p) for p in report.rglob('*') if p.is_file()}))
    print('E13 REPORT',{k:v['passed'] for k,v in s['goals'].items()},flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=run.ROOT)
    main(parser.parse_args().output)
