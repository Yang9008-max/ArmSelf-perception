"""E11 report: actual updates, competing surfaces and observational twins."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .body_cycle import read_json, write_json, sha, utc
from . import trust_cycle as e7
from . import extent_cycle_v2 as e9
from .hypothesis_cycle import ROOT, FINAL, CONDITIONS, MODES, PROFILES, verify, source_for
from .surface_extent import SurfaceAtlas
from .vision import project

CN=dict(clean='传感器正常',feedback_bias='反馈偏差',stale_rgb='过去图像',both='偏差＋过去图像',
        no_feedback='反馈缺失',no_camera='相机缺失',coherent_replay='协同回放',
        execution_hold='执行受阻／观测正常',twin_moving='同输入／身体运动',twin_held='同输入／身体受阻')
LABEL=dict(competing='竞争解释＋后续纠错',e8='原 E8B',single_surface='单状态表面纠错',
           naive_vote='相容多数候选',action='仅动作',feedback='仅反馈／缺失时递推')


def summarize(output):
    scored=read_json(output/'scores.json')
    events=read_json(output/'recovery-events.json')
    rows=read_json(output/'frame-scores.json')
    conditions={}
    for condition in CONDITIONS:
        conditions[condition]={}
        for mode in MODES:
            chosen=[r for r in scored['rows'] if r['condition']==condition and r['mode']==mode]
            item=dict(rmse_mm=float(np.sqrt(np.mean([r['rmse_mm']**2 for r in chosen]))),
                fault_rmse_mm=float(np.sqrt(np.mean([r['fault_rmse_mm']**2 for r in chosen]))),
                off_surface=float(np.mean([r['off_surface'] for r in chosen])))
            if mode=='competing':
                for k in ('unknown','coverage','width_normalized','commits','surface_corrections'):
                    item[k]=float(np.mean([r[k] for r in chosen]))
                returns=[r for r in rows if r['condition']==condition and r['mode']==mode
                         and (80<=r['frame']<90 or 150<=r['frame']<160)]
                item['late_return_unknown']=float(np.mean([r['unknown'] for r in returns]))
            selected=[r for r in events if r['condition']==condition and r['mode']==mode]
            for kind in ('convergence','actual_error'):
                times=[r[kind]['time_s'] for r in selected if r[kind]['recovered']]
                item[kind]=dict(events=len(selected),recovered=len(times),censored=len(selected)-len(times),
                    median_success_s=float(np.median(times)) if times else None,
                    restricted_mean_s=float(np.mean([r[kind]['time_s'] if r[kind]['recovered'] else 4. for r in selected])) if selected else None)
            conditions[condition][mode]=item
    normal=[]
    for seed in FINAL:
        chosen={r['mode']:r for r in scored['rows'] if r['seed']==seed and r['condition']=='clean'}
        delta=chosen['competing']['rmse_mm']-chosen['e8']['rmse_mm']
        normal.append(dict(seed=seed,regression_mm=delta,passed=delta<=.5))
    affected=('both','coherent_replay')
    macro={m:float(np.mean([conditions[c][m]['fault_rmse_mm'] for c in affected])) for m in MODES}
    improvement=1-macro['competing']/macro['e8']
    recovery={c:(conditions[c]['competing']['convergence']['recovered']>=5 and
                 conditions[c]['competing']['convergence']['median_success_s'] is not None and
                 conditions[c]['competing']['convergence']['median_success_s']<=1.) for c in CONDITIONS
                 if c not in ('clean','execution_hold')}
    twins=scored['twins']
    unknown_ok=all(t['distinguishable_physical_frames']>0 and t['unknown_rate']>=.9 and
                   t['simultaneous_coverage']>=.9 and t['output_prefix_identical'] for t in twins)
    updates=sum(r['commits'] for r in scored['rows'] if r['mode']=='competing')
    corrections=sum(r['surface_corrections'] for r in scored['rows'] if r['mode']=='competing')
    goals=dict(state=dict(passed=bool(improvement>=.05 and all(x['passed'] for x in normal)),
        relative_fault_improvement=improvement,normal_guards=normal),
        recovery=dict(passed=all(recovery.values()),conditions=recovery),
        ambiguity=dict(passed=unknown_ok,twins=twins),
        actual_updates=dict(passed=updates>0,commits=updates,surface_corrections=corrections))
    result=dict(created_utc=utc(),final_seeds=FINAL,conditions=conditions,goals=goals,macro_errors=macro,
        configuration=read_json(output/'development-selection.json'),
        calibration=read_json(output/'calibration.json'),mechanism='explicit inference with frozen learned models',
        candidate_sets_are_not_probabilities=True,duplicates_are_not_independent_votes=True)
    total=sum(t['distinguishable_physical_frames'] for t in twins)
    result['assessment']=dict(twin_frames=total,twin_both_covered=sum(t['both_states_covered'] for t in twins),
        twin_joint_coverage=sum(t['both_states_covered'] for t in twins)/total if total else None,
        normal_unknown=conditions['clean']['competing']['unknown'],
        execution_hold_rmse_mm=conditions['execution_hold']['competing']['fault_rmse_mm'],
        execution_hold_feedback_mm=conditions['execution_hold']['feedback']['fault_rmse_mm'])
    if (output/'audit.json').exists():
        result['audit']=read_json(output/'audit.json')
    if (output/'diagnostics.json').exists():
        result['diagnostics']=read_json(output/'diagnostics.json')
    write_json(output/'report'/'summary.json',result)
    return result,rows,events


def figures(report,s,rows):
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False})
    x=np.arange(len(CONDITIONS))
    fig,axes=plt.subplots(1,2,figsize=(14,5.7),gridspec_kw={'width_ratios':[7,4]})
    for ax,conditions,title in zip(axes,(CONDITIONS[:7],CONDITIONS[7:]),('原七条件','执行受阻与观测配对（纵轴单独刻度）')):
        xx=np.arange(len(conditions))
        modes=('competing','e8','single_surface','action') if ax is axes[0] else ('competing','e8','single_surface','action','feedback')
        width=.76/len(modes)
        for j,m in enumerate(modes):
            ax.bar(xx+(j-(len(modes)-1)/2)*width,[s['conditions'][c][m]['fault_rmse_mm'] for c in conditions],width,label=LABEL[m])
        ax.set_xticks(xx,[CN[c] for c in conditions],rotation=25,ha='right',fontsize=9)
        ax.set(ylabel='故障窗口共同锚点 RMSE / mm',title=title)
    axes[0].legend(ncol=2,fontsize=8)
    handles,labels=axes[1].get_legend_handles_labels();axes[1].legend(handles[-1:],labels[-1:],loc='upper center',fontsize=8)
    fig.tight_layout();fig.savefig(report/'errors.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.9))
    nonclean=[c for c in CONDITIONS if c not in ('clean','execution_hold')]
    for j,m in enumerate(('competing','e8','single_surface')):
        axes[0].bar(np.arange(len(nonclean))+(j-1)*.25,[s['conditions'][c][m]['convergence']['recovered'] for c in nonclean],.25,label=LABEL[m])
        axes[1].plot(np.arange(len(nonclean)),[s['conditions'][c][m]['convergence']['restricted_mean_s'] for c in nonclean],'o-',label=LABEL[m])
    axes[0].set(ylabel='收敛恢复事件 / 6',ylim=(0,6.9));axes[1].set(ylabel='受限平均确认时间 / s；未恢复计4s',ylim=(0,4.1))
    for ax in axes:
        ax.set_xticks(np.arange(len(nonclean)),[CN[c] for c in nonclean],rotation=25,ha='right',fontsize=8);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(report/'recovery.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.8))
    axes[0].bar(x,[s['conditions'][c]['competing']['unknown']*100 for c in CONDITIONS],label='全段未知')
    axes[0].plot(x,[s['conditions'][c]['competing']['late_return_unknown']*100 for c in CONDITIONS],'o-',color='orange',label='恢复后10–19帧')
    axes[0].set(ylabel='未知比例 / %',ylim=(0,105));axes[0].legend(fontsize=8)
    axes[1].bar(x,[s['conditions'][c]['competing']['coverage']*100 for c in CONDITIONS]);axes[1].set(ylabel='同时七关节集合覆盖 / %',ylim=(0,105))
    axes[2].bar(x,[s['conditions'][c]['competing']['width_normalized'] for c in CONDITIONS]);axes[2].set(ylabel='平均归一化半宽；非概率区间')
    for ax in axes:
        ax.set_xticks(x,[CN[c] for c in CONDITIONS],rotation=45,ha='right',fontsize=7)
    fig.suptitle('保留未知须与宽度、正常可用性和恢复后的收缩同时检查')
    fig.tight_layout();fig.savefig(report/'sets.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True)
    for c in ('clean','coherent_replay','execution_hold','twin_held'):
        r=[r for r in rows if r['seed']==FINAL[0] and r['condition']==c and r['mode']=='competing']
        axes[0].plot(np.arange(180)*.1,[a['error_mm'] for a in r],label=CN[c])
        axes[1].plot(np.arange(180)*.1,[a['hypotheses'] for a in r],label=CN[c])
        axes[2].plot(np.arange(180)*.1,[a['width_normalized'] for a in r],label=CN[c])
    for ax in axes:
        for start,stop in ((30,70),(110,140)):
            ax.axvspan(start*.1,stop*.1,color='orange',alpha=.13)
        ax.legend(fontsize=8)
    axes[0].set(ylabel='身体误差 / mm');axes[1].set(ylabel='保留解释数');axes[2].set(ylabel='集合归一化半宽',xlabel='时间 / s')
    fig.tight_layout();fig.savefig(report/'timeline.png',dpi=150);plt.close(fig)


def replay(output,report,rows):
    seed=FINAL[0]
    atlas=SurfaceAtlas.load(e9.ROOT/'atlas-multi.npz')
    ids=np.linspace(0,len(atlas.weights)-1,100,dtype=int)
    cal=read_json(source_for(output,seed,'clean')/'metadata.json')['calibration']
    folder=report/'frames';folder.mkdir(exist_ok=True)
    frame_ids=sorted(set(range(0,180,2))|set(range(20)))
    for physical in ('normal','held'):
        source=source_for(output,seed,'clean' if physical=='normal' else 'execution_hold')
        for i in frame_ids:
            a=e7.load_npz(source/'rgb'/f'{i:05d}.npz')
            Image.fromarray(a['left']).resize((480,360)).save(folder/f'{physical}-{i:03d}.jpg',quality=82)
    lookup={(r['condition'],r['frame']):r for r in rows if r['seed']==seed and r['mode']=='competing'}
    def uv(q):
        return np.clip(np.nan_to_num(project(atlas.predict(q)[ids],cal)*.75),-1500,1500).round(1).tolist()
    data=dict(seed=seed,labels=CN,modes=LABEL,conditions={})
    for c in CONDITIONS:
        p=e7.load_npz(output/'predictions'/f'{seed}-{c}.npz')
        a=e7.load_npz(output/'sensors'/f'{seed}-{c}.npz')
        physical='held' if c in ('execution_hold','twin_held') else 'normal'
        frames=[]
        for i in range(0,180,2):
            path=Path(a['rgb_paths'][i]) if a['rgb_paths'][i] else None
            image=None if path is None else f'frames/{"held" if "-held" in str(path) else "normal"}-{int(path.stem):03d}.jpg'
            r=lookup[c,i]
            frames.append(dict(i=i,current=f'frames/{physical}-{i:03d}.jpg',image=image,
                alternate=f'frames/{"normal" if physical=="held" else "held"}-{i:03d}.jpg',
                unknown=r['unknown'],error=r['error_mm'],width=r['width_normalized'],covered=r['covered'],
                committed=r['committed'],corrected=r['surface_corrected'],support=r['support'],count=r['hypotheses'],
                points={m:uv(p[f'q_{m}'][i]) for m in MODES},
                alternatives=[uv(p['bank'][i,j]) for j in (0,1,3) if p['bank_valid'][i,j]]))
        data['conditions'][c]=frames
    (report/'replay.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
    animation=[]
    frames=data['conditions']['execution_hold']
    for row in frames:
        image=Image.new('RGB',(960,430),'#101b29');draw=ImageDraw.Draw(image)
        image.paste(Image.open(report/row['current']),(0,38))
        if row['image']:
            image.paste(Image.open(report/row['image']),(480,38))
        draw.text((10,7),'当前真实身体＋保留的可能身体',font=font,fill='white')
        draw.text((490,7),'模型收到的相机',font=font,fill='white')
        layers=list(zip(row['alternatives'],['#ffbd63','#62bfff','#d79bff']))+[(row['points']['competing'],'#55ed9c')]
        for points,color in layers:
            for x,y in points:
                if 0<=x<480 and 0<=y<360:
                    draw.ellipse((x-1.5,y+36.5,x+1.5,y+39.5),fill=color)
        draw.text((10,403),f"t={row['i']*.1:.1f}s  {'未知：保留竞争解释' if row['unknown'] else '候选已收敛'}  锚点误差 {row['error']:.2f}mm",font=small,fill='white')
        animation.append(image)
    animation[0].save(report/'body-hypotheses.gif',save_all=True,append_images=animation[1:],duration=160,loop=0)
    animation[25].save(report/'preview.jpg',quality=90)


def document(output,report,s):
    selected=s['configuration']['selected'];p=s['configuration']['profiles'][selected]
    assessment=s['assessment'];co=s['conditions']['coherent_replay']
    lines=['# E11：相依证据、竞争身体解释与后续纠错', '', '完成：2026-10-25（Asia/Shanghai）。', '',
        '**本轮优化了显式推断机制；旧神经网络权重冻结，没有宣称模型新学会了来源故障概率。**', '',
        '[交互回放](http://127.0.0.1:8767/e11-body-hypotheses/report/) · [执行前协议](body-hypotheses-e11.md)', '',
        '**结论：恢复速度验收通过，故障准确性与保留真实竞争状态验收失败。** 旧网络冻结，新增来源去重、候选集合与后续提交均已实现，但这轮没有得到更可靠的整体身体感知。', '',
        f"共同故障宏平均误差{s['macro_errors']['e8']:.3f}→{s['macro_errors']['competing']:.3f}mm，升高{-s['goals']['state']['relative_fault_improvement']*100:.2f}%；配对{assessment['twin_frames']}个身体分歧帧全部未知，仅{assessment['twin_both_covered']}帧同时容纳两种真实状态；健康反馈下执行受阻仍为{assessment['execution_hold_rmse_mm']:.3f}mm，仅反馈为{assessment['execution_hold_feedback_mm']:.3f}mm。", '',
        f"同批协同回放恢复成功中位时间由E8B的{co['e8']['convergence']['median_success_s']:.2f}s降至{co['competing']['convergence']['median_success_s']:.2f}s。八种恢复条件每种6/6，但原E8B在本批也都6/6，不能用这次通过声称旧失败已被普遍解决；48个条件事件共享6条物理轨迹，且同输入运动分支与协同回放重复，不是48个独立试验。", '',
        '## 方法与变化', '',
        'E10 新增的一致性审计只作诊断；原E8B仍负责状态更新。E11 在其外保留受保护动作路径、当前反馈、递推状态、冲突前保持、视觉拟合和临时主状态。动作与由其初始化的视觉不能形成独立多数票；同一RGB上不同初始化得到的同一片段仅保留一个光度代价更小的实际双目观测。', '',
        '从冻结 E9B 图谱的82个空间分散片段取得当前三维测量；有界最小二乘拟合身体构型。候选冲突时保留集合和未知，不强行合成唯一身体或把中间空间填为实体。新观测在反馈、表面残差、动作预测及跨时间响应上连续一致后，提交状态并重置受污染的时序记忆；这些改变会进入下一次动作预测。', '',
        '单状态表面纠错对照获得相同82片段与同一拟合器，防止将额外视觉预算当作多解释机制收益。原E8B、相容多数候选、仅动作和仅反馈均保留。', '',
        '## 参数与开发选择', '',
        f'开发种子2026103111，两条物理轨迹；最终三个全新种子{FINAL}，每种子正常与执行受阻两条轨迹，各180帧。十条件共享相应物理轨迹。原有干扰物保持固定。', '',
        f"选中 **{selected}**：表面修正增益 **{p['gain']}**；恢复连续确认 **{p['confirm']}帧**。开发只比较增益0/0.2、确认3/5帧四组，按状态教师误差与正常权重选取。没有用最终数据选择。", '',
        '实际选择的增益为0：未启用日常几何坐标修正，不能宣称直接表面拟合优化成功。E9B实际匹配仍参与恢复准入检查；连续证据通过后提交实际反馈支持的状态并重置递推记忆。单状态直接表面纠错对照独立固定为0.2增益，避免对照随主方案的零增益退化。', '',
        '| 开发候选 | 增益 | 确认帧 | 选择损失 |','|---|---:|---:|---:|']
    for name,v in s['configuration']['profiles'].items():
        lines.append(f"| {name} | {v['gain']} | {v['confirm']} | {v['selection']:.8f} |")
    t=s['calibration']['thresholds']
    lines+=['',f"正常开发99%分位门限：动作—反馈 **{t['action_feedback']*1000:.3f}mm**；表面残差 **{t['surface']*1000:.3f}mm**；反馈动态残差 **{t['dynamics']*1000:.3f}mm**。拟合至少12个匹配点，3轮迭代、归一化单步≤0.04、累计≤0.12；几何变化≤6mm，回归阻尼0.001。", '',
        '集合余量取开发正常状态误差各关节99%分位与0.02归一化尺度的较大值。它是保守集合余量，不具有90%概率含义。', '',
        '新增依赖处理是来源去重和避免硬多数排除，并没有拟合完整联合似然或条件协方差。E8B临时点估计仍来自原融合网络；冲突时主状态未必是候选中最正确的一个。若集合宽度很大，只能说明尚未排除，并不能证明有多个高质量、带概率的身体解释。', '',
        '## 故障窗口固定锚点RMSE / mm', '',
        '| 条件 | E11 | 原E8B | 单状态同表面 | 相容多数 | 仅动作 | 仅反馈 |','|---|---:|---:|---:|---:|---:|---:|']
    for c in CONDITIONS:
        lines.append('| '+CN[c]+' | '+' | '.join(f"{s['conditions'][c][m]['fault_rmse_mm']:.3f}" for m in MODES)+' |')
    gain=s['goals']['state']['relative_fault_improvement']
    lines+=['',f"共同受损与协同回放两条件宏平均，E11相对原E8B变化为 **{gain*100:+.2f}%改善**。强对照宏平均："+
        '；'.join(f"{LABEL[m]} {v:.3f}mm" for m,v in s['macro_errors'].items())+'。正改善表示误差降低，负数表示变差。', '',
        '## 恢复、未知与集合宽度', '',
        '| 条件 | E11恢复/6 | 原E8B恢复/6 | E11成功中位/s | E8B成功中位/s | 全段未知 | 恢复后10–19帧未知 | 集合覆盖 | 归一化半宽 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c in CONDITIONS:
        a=s['conditions'][c]['competing'];b=s['conditions'][c]['e8'];r=a['convergence']
        time='—' if r['median_success_s'] is None else f"{r['median_success_s']:.2f}"
        bt='—' if b['convergence']['median_success_s'] is None else f"{b['convergence']['median_success_s']:.2f}"
        lines.append(f"| {CN[c]} | {r['recovered']}/{r['events']} | {b['convergence']['recovered']}/{b['convergence']['events']} | {time} | {bt} | {a['unknown']*100:.2f}% | {a['late_return_unknown']*100:.2f}% | {a['coverage']*100:.2f}% | {a['width_normalized']:.4f} |")
    lines+=['', '恢复定义为相对同物理轨迹健康观测分支的身体预测距离连续五帧≤2mm。最快确认0.5s，未恢复保留为4s删失；同时保留实际锚点误差相对健康分支≤2mm的恢复。回到健康分支不等于绝对几何正确。执行受阻本身就是对应健康分支，故不伪造该条件的故障解除事件。', '',
        '## 输入相同、身体不同的配对', '',
        '两条物理轨迹在相同动作命令下分别正常运动和受阻；第一故障窗口内输入相同的过去RGB与反馈，因此第一恢复之前完整输入前缀相同。任何因果模型都不能凭这些输入唯一判断隐藏身体。本轮要求模型保持未知、保留两种真实状态，并在新的不同观测到来后更新。此配对是合成验证，不等同全部自然故障。', '',
        '| 种子 | 身体分歧>15mm的帧 | 未知率 | 同时容纳两真实状态 | 输出前缀一致 |','|---|---:|---:|---:|---|']
    for a in s['goals']['ambiguity']['twins']:
        unknown='—' if a['unknown_rate'] is None else f"{a['unknown_rate']*100:.2f}%"
        cover='—' if a['simultaneous_coverage'] is None else f"{a['simultaneous_coverage']*100:.2f}%"
        lines.append(f"| {a['seed']} | {a['distinguishable_physical_frames']} | {unknown} | {cover} | {a['output_prefix_identical']} |")
    lines+=['', '## 验收与真实状态更新', '', '| 项目 | 结果 |','|---|---|']
    for k,label in [('state','共同故障至少改善5%且正常每种子退化≤0.5mm'),('recovery','各条件至少5/6事件，成功中位≤1s'),
                    ('ambiguity','不可辨识配对保留未知及两种真实状态'),('actual_updates','实际提交状态更新')]:
        lines.append(f"| {label} | {'通过' if s['goals'][k]['passed'] else '未通过'} |")
    updates=s['goals']['actual_updates']
    lines+=['',f"最终全条件共 **{updates['commits']}次恢复提交**、**{updates['surface_corrections']}次日常表面修正**。计数包含多个传感分支，不能当作独立物理试验次数。提交是否发生只是实现验收；有益与否须结合误差和对照。", '',
        '## 边界与失败的解释', '',
        '未知和候选记忆为显式算法；增益与门限为开发选择，身体图谱及动作、融合网络来自既有学习。没有新训练神经网络。RGB拟合仍有初始化偏差和错配，不能把跟踪通过当作物理真实的概率。', '',
        f"正常条件仍有{assessment['normal_unknown']*100:.2f}%帧未知，存在明显可用性代价；恢复后10–19帧未知已归零。执行受阻虽然候选包络覆盖100%，主状态仍接近错误动作分支，误差约98mm，说明保留正确反馈候选并不等于让它纠正主状态。配对几乎未保留两种真状态，表明仅列出六个快照不足以保存正确的竞争历史。", '',
        '从代码与结果可推断，当前恢复准入要求动作预测也与观测一致；动作／执行模型失效时，这个条件阻碍实际反馈推翻动作解释。受保护路径未被两路一致观测删除，却同时变得难以被可信反证推翻。冲突前停止候选也只保存一次last_safe快照，过早冲突会使其偏离真正停止前状态。该解释属于事后诊断，未据此修改最终模型。', '',
        '恢复时间只检验干预结束后回到健康观测分支；本批恢复时执行也已恢复，并不证明动作解释仍错误时模型能主动纠正。后续必须单独检验“执行持续受阻、反馈与视觉正常”期间的主状态纠错，避免等待动作再次正确就算成功。', '',
        '下一轮应给“按命令运动、执行受阻、反馈失真、图像陈旧”等解释各自的状态递推与观测预测，评价后续新证据在各解释下是否合理，并允许证据否决动作路径。需要与仅反馈、仅动作及同预算单状态方法比较；不能只扩大包络或放宽恢复门槛。当前不在同一最终集上继续调参。', '',
        '同一学习解码器的状态距离与固定物理锚点误差对象不同，分别报告；仍是部分表面，没有完整形态、部件身份或实机证据。配对正确保留未知证明对不可辨识性的处理，不证明模型可以凭空知道实际隐藏状态。', '',
        '## 复现与证据', '',
        '输出目录：`outputs/local-check-20261024/e11-body-hypotheses/`。开发选择、门限、代码和模型在最终采集前冻结；30份预测全保存封存后才读取特权几何评分。逐帧、逐事件、输入权限和审计记录均保留。', '',
        '最终采集前有一次对照澄清：将单状态表面纠错固定为0.2，而不是沿用主方案选中的0增益。修订前源码、协议和冻结记录保存在 `pre-final-baseline-clarification/`，未存在任何最终数据，四组开发选择和主方案状态代码均未改变。', '',
        '另有一次输入预检修订：两个独立渲染上下文的少数像素相差1灰度级，故成对实验改为共享前30帧完整记录，随后共享受控回放，以保证前70帧输入完全相同。旧开发与种子2026103131的部分预测未评分，作为预检保留；更换最终种子为2026103161–3163，重新开发选择和冻结后采集。修订记录见`docs/body-hypotheses-e11-revision.json`及`pre-shared-prefix-revision/`，原始物理记录未修改。', '',
        '```powershell',r'.\.venv\Scripts\python.exe -X utf8 -m embodiment.hypothesis_cycle prepare',
        r'.\.venv\Scripts\python.exe -X utf8 -m embodiment.hypothesis_cycle final',
        r'.\.venv\Scripts\python.exe -X utf8 -m embodiment.hypothesis_cycle score',
        r'.\.venv\Scripts\python.exe -X utf8 -m embodiment.hypothesis_audit',
        r'.\.venv\Scripts\python.exe -X utf8 -m embodiment.hypothesis_report','```', '',
        '冻结或完成阶段拒绝覆盖；复现使用新的 `--output` 路径。','']
    if 'audit' in s:
        a=s['audit']
        lines+=['## 实现与交付校验','',
            f"123项回归测试通过；审计验证{a['prediction_files']}份预测、{a['command_files']}份执行前命令、固定干扰物、相同28身体锚点和434表面点分母。配对前70帧所有输入、测量和预测字段完全一致。",
            '',f"{a['commits']}次恢复提交中，{a['commits_changing_point_estimate']}次相对原提案实际改变主状态；{a['commits_changing_future_forecast']}次实际改变下一步受保护预测。此校验验证更新生效，性能收益仍以误差与恢复对照为准。",'']
    if 'diagnostics' in s:
        d=s['diagnostics']
        lines+=['## 补充诊断：包络内不等于候选贴近','',
            '以下为封存后的描述性诊断，不改变验收或选择参数。逐关节包络是松弛外包；它容纳真值不代表某个候选准确。六个候选也不是六条各自有独立动力学和概率的完整历史。最近候选误差为同一学习图谱上相对实际构型的身体距离。','',
            '| 种子 | 正常运动真实状态最近候选中位/mm | 95%分位/mm | 受阻真实状态最近候选中位/mm | 95%分位/mm |',
            '|---|---:|---:|---:|---:|']
        for x in d['nearest_candidates']:
            lines.append('| '+str(x['seed'])+' | '+' | '.join('—' if x[k] is None else f'{x[k]:.3f}' for k in
                ('nearest_moving_median_mm','nearest_moving_p95_mm','nearest_held_median_mm','nearest_held_p95_mm'))+' |')
        instant='—' if d['median_instant_improvement_mm'] is None else f"{d['median_instant_improvement_mm']:.4f}mm"
        lines+=['',f"恢复提交的当帧固定锚点误差，相对提交前提案：{d['commits_helped_gt_001mm']}次降低超过0.01mm，{d['commits_hurt_gt_001mm']}次升高超过0.01mm，中位改善{instant}。它只衡量显式提交的瞬时效果，不等于整个时序机制的净收益。",'']
    text='\n'.join(lines)
    (report/'详细报告.md').write_text(text,encoding='utf-8')
    Path('docs/body-hypotheses-e11-summary.md').write_text(text,encoding='utf-8')


def webpage(report,s):
    rows=''.join('<tr><td>'+CN[c]+'</td>'+''.join(f"<td>{s['conditions'][c][m]['fault_rmse_mm']:.2f}</td>" for m in MODES)+'</tr>' for c in CONDITIONS)
    goals=''.join(f'<li>{html.escape(k)}：<b>{"通过" if v["passed"] else "未通过"}</b></li>' for k,v in s['goals'].items())
    a=s['assessment']
    finding=(f'恢复速度验收通过；共同故障准确性与竞争真状态保留失败。共同故障宏平均误差 '
        f'{s["macro_errors"]["e8"]:.3f} → {s["macro_errors"]["competing"]:.3f} mm；'
        f'配对 {a["twin_frames"]} 个身体分歧帧全部未知，但只有 {a["twin_both_covered"]} 帧同时覆盖两真实状态。'
        f'执行受阻、反馈健康时主状态误差 {a["execution_hold_rmse_mm"]:.2f} mm，仅反馈 {a["execution_hold_feedback_mm"]:.2f} mm。'
        f'正常条件未知率 {a["normal_unknown"]*100:.2f}%。')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>E11 · 竞争身体解释与纠错</title><style>*{box-sizing:border-box}body{margin:0;background:#101b29;color:#e8f0f6;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1180px;margin:auto;padding:28px 20px 65px}h1{font-size:32px}h2{font-size:23px}section{padding:20px;margin:22px 0;background:#192b3d;border:1px solid #345069;border-radius:12px}a{color:#75dac3}p{color:#c5d3e2}button,select{background:#28435a;color:white;border:1px solid #608198;border-radius:6px;padding:9px;margin:4px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}canvas,img{width:100%;border-radius:8px;background:#080e16}.note{border-left:4px solid #edb75b;padding-left:14px}input[type=range]{width:100%}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #3e5368;text-align:left}.scroll{overflow-x:auto}#stats{white-space:pre-line}small{color:#aec3d6}@media(max-width:750px){.grid{grid-template-columns:1fr}main{padding:15px}h1{font-size:26px}}</style><main>
<p><a href="/">← 实验入口</a> · <a href="详细报告.md">详细报告</a> · <a href="summary.json">全部指标</a> · <a href="body-hypotheses-e11.md">冻结协议</a></p><h1>保留可能的身体，再用新证据纠错</h1><p>3个新种子 · 正常与受阻两条物理轨迹 · 10种条件 · 82片段观测连接434片段身体图谱</p>
<p class="note">这是显式时序推断的优化，旧神经网络冻结。未知表示当前证据不足以排除竞争解释；候选集合不是概率区间，同一图像的多次匹配不算独立投票。</p>
<section><h2>状态与竞争解释回放</h2><select id="condition"></select><select id="mode"></select><button id="play">暂停</button><label><input type="checkbox" id="alternatives" checked>显示竞争身体</label><div class="grid"><div><p>实际身体＋当前状态与保留候选</p><canvas id="actual" width="480" height="360"></canvas></div><div><p>输入给模型的相机</p><canvas id="sensor" width="480" height="360"></canvas></div></div><input id="frame" type="range" min="0" max="89" value="0"><p id="stats">加载…</p><small>绿色：所选状态。橙／蓝／紫：动作、观测、停止解释的表面投影，未知时保留。展示每身体100点，评分保留全部434点。主读数为E11。多候选之间的空间不被填为实体。</small><details><summary>查看另一条隐藏物理轨迹（仅作配对评价，不给模型）</summary><canvas id="alternate" width="480" height="360" style="max-width:480px"></canvas><p>同输入配对在第一恢复前具有相同完整输入前缀，模型无法唯一判定是哪条身体轨迹。</p></details></section>
<section><h2>冻结验收</h2><ul>GOALS</ul><p>实际更新计数与性能收益分别检验；保留全部未知、宽度、失败和未恢复事件。</p></section>
<section><h2>故障窗口锚点误差 / mm</h2><div class="scroll"><table><tr><th>条件</th><th>E11</th><th>E8B</th><th>单状态同表面</th><th>多数候选</th><th>仅动作</th><th>仅反馈</th></tr>ROWS</table></div><img src="errors.png"></section><section><h2>恢复与未恢复</h2><img src="recovery.png"></section><section><h2>未知、集合覆盖与代价</h2><img src="sets.png"></section><section><h2>状态、候选与范围的时间过程</h2><img src="timeline.png"></section><section><h2>适用范围</h2><p>当前仿真机械臂、已知控制通道、标定双目和干净初始化。候选筛选、恢复确认和表面拟合为显式算法。未证明任意故障识别、完整身体身份或实机泛化。</p><p><a href="frame-scores.csv">逐帧分数</a> · <a href="recovery-events.csv">所有恢复事件</a> · <a href="body-hypotheses.gif">动画</a></p></section></main>
<script>let data,busy=false,playing=true;const $=id=>document.getElementById(id),cache=new Map();function load(path){if(!path)return Promise.resolve(null);if(cache.has(path))return cache.get(path);const p=new Promise(resolve=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=()=>resolve(null);im.src=path;});cache.set(path,p);return p;}async function draw(){if(!data||busy)return;busy=true;try{const r=data.conditions[$('condition').value][+$('frame').value],m=$('mode').value;const imgs=await Promise.all([load(r.current),load(r.image),load(r.alternate)]);for(const [j,id] of ['actual','sensor','alternate'].entries()){const ctx=$(id).getContext('2d');ctx.fillStyle='#080e16';ctx.fillRect(0,0,480,360);if(imgs[j])ctx.drawImage(imgs[j],0,0,480,360);else{ctx.fillStyle='white';ctx.font='20px sans-serif';ctx.fillText('相机缺失',180,180);}}const ctx=$('actual').getContext('2d');function dots(points,color){ctx.fillStyle=color;for(const [x,y] of points){ctx.beginPath();ctx.arc(x,y,1.8,0,Math.PI*2);ctx.fill();}}dots(r.points[m],'#55ed9c');if($('alternatives').checked)r.alternatives.forEach((p,j)=>dots(p,['#ffbd63','#62bfff','#d79bff'][j%3]));$('stats').textContent=`t=${(r.i*.1).toFixed(1)}s · ${r.unknown?'未知：保留竞争解释':'候选已收敛'} · 保留 ${r.count} 个解释\nE11锚点误差 ${r.error.toFixed(2)}mm；集合是否覆盖真实状态：${r.covered?'是':'否'}；归一化半宽 ${r.width.toFixed(4)}\n当前表面匹配 ${r.support}/82；恢复提交：${r.committed?'是':'否'}；日常表面修正：${r.corrected?'是':'否'}。`;}finally{busy=false;}}fetch('replay.json').then(r=>r.json()).then(x=>{data=x;for(const [k,v] of Object.entries(x.labels))$('condition').add(new Option(v,k));for(const [k,v] of Object.entries(x.modes))$('mode').add(new Option(v,k));$('condition').value='twin_moving';draw();}).catch(e=>$('stats').textContent=e.message);for(const id of ['condition','mode','frame','alternatives'])$(id).addEventListener('input',draw);$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'暂停':'播放';};setInterval(()=>{if(data&&playing&&!busy){$('frame').value=(+$('frame').value+1)%90;draw();}},180);</script></html>'''
    page=page.replace('<section><h2>状态与竞争解释回放</h2>',
        '<section class="note"><h2>本轮结论：能恢复，还不能可靠选对身体</h2><p>'+html.escape(finding)+
        '</p><p>原E8B在本批也全部恢复；协同回放成功中位由1.15s降至0.90s。默认回放展示执行受阻的失败，所有条件可切换。</p></section><section><h2>状态与竞争解释回放</h2>')
    page=page.replace("$('condition').value='twin_moving'","$('condition').value='execution_hold'")
    page=page.replace("dots(r.points[m],'#55ed9c');if($('alternatives').checked)r.alternatives.forEach((p,j)=>dots(p,['#ffbd63','#62bfff','#d79bff'][j%3]));",
        "if($('alternatives').checked)r.alternatives.forEach((p,j)=>dots(p,['#ffbd63','#62bfff','#d79bff'][j%3]));dots(r.points[m],'#55ed9c');")
    page=page.replace('<a href="frame-scores.csv">逐帧分数</a>',
        '<a href="seed-scores.json">逐种子全部分数</a> · <a href="audit.json">状态更新审计</a> · <a href="diagnostics.json">提交与候选诊断</a> · <a href="frame-scores.csv">逐帧分数</a>')
    (report/'index.html').write_text(page.replace('GOALS',goals).replace('ROWS',rows),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT)
    args=parser.parse_args();verify(args.output)
    report=args.output/'report';report.mkdir(exist_ok=True)
    (report/'body-hypotheses-e11.md').write_text(Path('docs/body-hypotheses-e11.md').read_text(encoding='utf-8'),encoding='utf-8')
    s,rows,events=summarize(args.output)
    for source,target in [('scores.json','seed-scores.json'),('audit.json','audit.json'),
                          ('diagnostics.json','diagnostics.json'),('protocol-freeze.json','protocol-freeze.json')]:
        write_json(report/target,read_json(args.output/source))
    for name,records in [('frame-scores.csv',rows),('recovery-events.csv',events)]:
        records=[{(f'{k}_{subkey}' if subkey else k):subvalue for k,v in row.items()
                  for subkey,subvalue in (v.items() if isinstance(v,dict) else [('',v)])}
                 for row in records] if name=='recovery-events.csv' else records
        keys=list(dict.fromkeys(k for row in records for k in row))
        with (report/name).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,keys);w.writeheader();w.writerows(records)
    figures(report,s,rows);replay(args.output,report,rows);document(args.output,report,s);webpage(report,s)
    write_json(args.output/'report-artifacts.json',dict(created_utc=utc(),generator_sha256=sha(Path(__file__)),
        hashes={str(p):sha(p) for p in report.rglob('*') if p.is_file()}))
    print('E11 report complete',{k:v['passed'] for k,v in s['goals'].items()},flush=True)


if __name__=='__main__':
    main()
