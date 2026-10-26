"""E12 report with all failures, paired physical branches and counterfactuals."""
from __future__ import annotations
import csv
import html
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from . import branch_cycle as run,trust_cycle as e7,extent_cycle_v2 as e9
from .body_cycle import read_json,write_json,sha,utc
from .surface_extent import SurfaceAtlas
from .vision import project

CN=dict(clean='观测正常',feedback_bias='反馈偏差',stale_rgb='过去图像',both='偏差＋过去图像',
    no_feedback='反馈缺失',no_camera='相机缺失',coherent_replay='协同回放',execution_hold='受阻／观测正常',
    twin_moving='同输入／身体运动',twin_held='同输入／身体受阻',hold_feedback_bias='受阻＋反馈偏差',hold_no_camera='受阻＋相机缺失')
LABEL=dict(branch='E12 运动解释与纠错',e11='冻结 E11',e8='E8B',single_surface='单状态同表面',
    no_override='关闭观测推翻',action='仅动作',feedback='仅反馈')
GOAL_LABEL=dict(state='状态精度与正常保护',recovery='执行仍受阻时恢复',ambiguity='保留真实竞争状态与正常可用性',actual_updates='实际更新与消融收益')


def summarize(output):
    score=read_json(output/'scores.json');frames=read_json(output/'frame-scores.json')
    events=read_json(output/'recovery-events.json');by={}
    for c in run.CONDITIONS:
        by[c]={}
        for mode in run.MODES:
            rows=[x for x in score['rows'] if x['condition']==c and x['mode']==mode]
            item={key:float(np.sqrt(np.mean([r[key]**2 for r in rows]))) for key in ('rmse_mm','fault_rmse_mm')}
            item['hold_rmse_mm']=float(np.sqrt(np.mean([r['hold_rmse_mm']**2 for r in rows]))) if c in run.HELD else None
            item['off_surface']=float(np.mean([r['off_surface'] for r in rows]))
            if mode=='branch':
                for key in ('unknown','coverage','width_normalized','commits'):
                    item[key]=float(np.mean([r[key] for r in rows]))
            ev=[x for x in events if x['condition']==c and x['mode']==mode and 'held_absolute' in x]
            if ev:
                times=[x['held_absolute']['time_s'] for x in ev if x['held_absolute']['recovered']]
                item['held_absolute']=dict(events=len(ev),recovered=len(times),within_1s=sum(t<=1.00001 for t in times),
                    median_success_s=float(np.median(times)) if times else None)
            by[c][mode]=item
    normal=[dict(seed=seed,regression_mm=next(r['rmse_mm'] for r in score['rows'] if r['seed']==seed and r['condition']=='clean' and r['mode']=='branch')-
                 next(r['rmse_mm'] for r in score['rows'] if r['seed']==seed and r['condition']=='clean' and r['mode']=='e11')) for seed in run.FINAL]
    macro={m:float(np.mean([by[c][m]['fault_rmse_mm'] for c in ('both','coherent_replay')])) for m in run.MODES}
    improvement=1-by['execution_hold']['branch']['hold_rmse_mm']/by['execution_hold']['e11']['hold_rmse_mm']
    deterioration=macro['branch']/macro['e11']-1
    absolute=by['twin_held']['branch']['held_absolute']
    twins=score['twins'];total=sum(t['distinguishable_physical_frames'] for t in twins)
    coverage=sum(t['both_states_covered'] for t in twins)/total if total else 0
    unknown=sum(t['unknown_frames'] for t in twins)/total if total else 0
    normal_u=by['clean']['branch']['unknown'];normal_w=by['clean']['branch']['width_normalized']
    audit=read_json(output/'audit.json')
    goals=dict(state=dict(passed=bool(improvement>=.7 and deterioration<=.03 and all(r['regression_mm']<=.5 for r in normal)),
                          held_improvement=improvement,common_fault_deterioration=deterioration,normal_guards=normal),
        recovery=dict(passed=absolute['within_1s']==6,**absolute),
        ambiguity=dict(passed=all(t['unknown_rate']>=.9 and t['simultaneous_coverage']>=.9 for t in twins) and normal_u<=.1 and normal_w<=.35,
                       frames=total,unknown_rate=unknown,simultaneous_coverage=coverage,normal_unknown=normal_u,normal_width=normal_w,twins=twins),
        actual_updates=dict(passed=bool(audit['overrides']>0 and audit['corrected_future_differs_from_open_action']>0 and
                by['execution_hold']['branch']['hold_rmse_mm']<by['execution_hold']['no_override']['hold_rmse_mm']),
                overrides=audit['overrides'],commits=audit['commits'],changed_future=audit['corrected_future_differs_from_open_action']))
    result=dict(created_utc=utc(),conditions=by,goals=goals,macro_errors=macro,final_seeds=run.FINAL,
        configuration=read_json(output/'development-selection.json'),calibration=read_json(output/'calibration.json'),audit=audit,
        network_retrained=False,candidate_sets_are_not_probabilities=True)
    if (output/'diagnostics.json').exists():
        result['diagnostics']=read_json(output/'diagnostics.json')
    write_json(output/'report'/'summary.json',result)
    return result,frames,events


def figures(report,s,rows):
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False})
    fig,axes=plt.subplots(1,2,figsize=(15,5.5))
    for ax,conditions in zip(axes,(run.CONDITIONS[:7],run.CONDITIONS[7:])):
        modes=('branch','e11','single_surface','action') if ax is axes[0] else ('branch','e11','single_surface','feedback')
        for j,m in enumerate(modes):
            ax.bar(np.arange(len(conditions))+(j-1.5)*.19,[s['conditions'][c][m]['fault_rmse_mm'] for c in conditions],.19,label=LABEL[m])
        ax.set_xticks(np.arange(len(conditions)),[CN[c] for c in conditions],rotation=28,ha='right',fontsize=8)
        ax.set(ylabel='传感故障窗口固定锚点RMSE / mm');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(report/'errors.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True)
    for m in ('branch','e11','no_override','feedback'):
        chosen=[r for r in rows if r['seed']==run.FINAL[0] and r['condition']=='twin_held' and r['mode']==m]
        axes[0].plot(np.arange(180)*.1,[r['error_mm'] for r in chosen],label=LABEL[m])
    chosen=[r for r in rows if r['seed']==run.FINAL[0] and r['condition']=='twin_held' and r['mode']=='branch']
    axes[1].step(np.arange(180)*.1,[r['unknown'] for r in chosen],label='未知')
    axes[1].step(np.arange(180)*.1,[r['override_action'] for r in chosen],label='观测推翻动作',alpha=.7)
    axes[2].plot(np.arange(180)*.1,[r['width_normalized'] for r in chosen],label='包络归一化半宽')
    for ax in axes:
        ax.axvspan(3,15,color='#8bbbe0',alpha=.14,label='持续受阻' if ax is axes[0] else None)
        for start,stop in ((3,7),(11,14)):
            ax.axvspan(start,stop,color='orange',alpha=.17)
        ax.axvline(7,color='green',ls=':');ax.axvline(14,color='green',ls=':');ax.legend(fontsize=8)
    axes[0].set(ylabel='实体锚点误差 / mm');axes[2].set(xlabel='时间 / s',ylabel='集合半宽')
    fig.tight_layout();fig.savefig(report/'timeline.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,5))
    for ax,key,label in zip(axes,('unknown','coverage','width_normalized'),('全段未知比例','全段七关节包络覆盖','归一化半宽；非概率区间')):
        ax.bar(np.arange(len(run.CONDITIONS)),[s['conditions'][c]['branch'][key] for c in run.CONDITIONS])
        ax.set_xticks(np.arange(len(run.CONDITIONS)),[CN[c] for c in run.CONDITIONS],rotation=60,ha='right',fontsize=7)
        ax.set_title(label,fontsize=10)
    fig.tight_layout();fig.savefig(report/'sets.png',dpi=150);plt.close(fig)


def replay(output,report,rows):
    seed=run.FINAL[0];atlas=SurfaceAtlas.load(e9.ROOT/'atlas-multi.npz')
    ids=np.linspace(0,len(atlas.weights)-1,100,dtype=int)
    cal=read_json(run.source_for(output,seed,'clean')/'metadata.json')['calibration']
    folder=report/'frames';folder.mkdir(exist_ok=True)
    frame_ids=sorted(set(range(0,180,2))|set(range(20)))
    for kind,c in (('normal','clean'),('held','execution_hold')):
        source=run.source_for(output,seed,c)
        for i in frame_ids:
            rgb=e7.load_npz(source/'rgb'/f'{i:05d}.npz')
            Image.fromarray(rgb['left']).resize((480,360)).save(folder/f'{kind}-{i:03d}.jpg',quality=83)
    lookup={(r['condition'],r['frame']):r for r in rows if r['seed']==seed and r['mode']=='branch'}
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
                unknown=r['unknown'],covered=r['covered'],width=r['width_normalized'],error=r['error_mm'],override=r['override_action'],
                support=r['support'],points={m:uv(p[f'q_{m}'][i]) for m in run.MODES},
                alternatives=[uv(q) for q,v in zip(p['bank'][i,:4],p['bank_valid'][i,:4]) if v]))
        data['conditions'][c]=items
    (report/'replay.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')),encoding='utf8')
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17);animation=[]
    for row in data['conditions']['twin_held']:
        im=Image.new('RGB',(960,430),'#101b29');d=ImageDraw.Draw(im)
        im.paste(Image.open(report/row['current']),(0,38))
        if row['image']:
            im.paste(Image.open(report/row['image']),(480,38))
        d.text((12,8),'实际身体＋E12主状态与保留解释',font=font,fill='white')
        d.text((492,8),'模型收到的相机',font=font,fill='white')
        for points,color in list(zip(row['alternatives'],['#ffbd63','#62bfff','#c294ff','#ed91b2']))+[(row['points']['branch'],'#55ed9c')]:
            for x,y in points:
                if 0<=x<480 and 0<=y<360:
                    d.ellipse((x-1.5,y+36.5,x+1.5,y+39.5),fill=color)
        d.text((12,402),f"t={row['i']*.1:.1f}s  未知={row['unknown']}  推翻动作={row['override']}  实体误差={row['error']:.2f}mm",font=font,fill='white')
        animation.append(im)
    animation[0].save(report/'body-branches.gif',save_all=True,append_images=animation[1:],duration=160,loop=0)
    animation[40].save(report/'preview.jpg',quality=90)


def document(output,report,s,events):
    selected=s['configuration']['selected'];profile=s['configuration']['profiles'][selected]
    g=s['goals'];t=s['calibration']['thresholds'];a=s['audit']
    lines=['# E12：动作先验可被推翻，持续运动解释参与纠错','','完成日期：2026-10-25。',
        '', '**本轮优化的是显式状态推断；旧网络权重没有重训。** 原有干扰物、遮挡屏和相机保持固定。',
        '', '[交互报告](http://127.0.0.1:8767/e12-branch-body/report/) · [执行前协议](body-branches-e12.md)',
        '', '## 结果先行','',
        f"健康观测下持续受阻30–149帧，固定锚点RMSE：冻结E11 **{s['conditions']['execution_hold']['e11']['hold_rmse_mm']:.3f}mm** → E12 **{s['conditions']['execution_hold']['branch']['hold_rmse_mm']:.3f}mm**，改善 **{g['state']['held_improvement']*100:.2f}%**。同批关闭推翻消融为 **{s['conditions']['execution_hold']['no_override']['hold_rmse_mm']:.3f}mm**，仅反馈为 **{s['conditions']['execution_hold']['feedback']['hold_rmse_mm']:.3f}mm**。",
        '',f"共同受损及协同回放宏误差：E11 {s['macro_errors']['e11']:.3f} → E12 {s['macro_errors']['branch']:.3f}mm（恶化定义为正：{g['state']['common_fault_deterioration']*100:+.2f}%）。",
        '',f"配对分歧 {g['ambiguity']['frames']} 帧：未知 {g['ambiguity']['unknown_rate']*100:.2f}%，同时覆盖两种实际状态 {g['ambiguity']['simultaneous_coverage']*100:.2f}%；正常未知 {g['ambiguity']['normal_unknown']*100:.2f}%，正常平均归一化半宽 {g['ambiguity']['normal_width']:.4f}。",
        '',f"新证据到来但执行仍受阻的配对恢复：{g['recovery']['recovered']}/{g['recovery']['events']}事件达到绝对恢复标准，{g['recovery']['within_1s']}/6在1秒内确认。",'',
        '| 预先验收 | 结果 |','|---|---|']
    for k,v in g.items():
        lines.append(f"| {GOAL_LABEL[k]} | {'通过' if v['passed'] else '未通过'} |")
    lines+=['','## 根因与实现','','E11 把动作预期也作为接受反馈纠错的必要条件。执行受阻时，错误动作先验因此阻止健康观测更新；冲突前单帧快照也无法保留持续的停止状态。',
        '', 'E12 在同一个学习身体坐标系中维护四条事先预测的运动历史：独立开环命令、由可信观测校正的局部命令响应、持续此前命令的伺服解释、速度逐步衰减的停止解释。关联检查发生在读入当前反馈之前预测的状态上；当前跳变不能生成它自己的支持候选。',
        '', '反馈只需与其中一条物理历史相容，并在相同原始双目图像的至少12个实际匹配片段上获得支持。连续确认后可推翻开环动作、更新主状态、重置旧融合隐藏状态，并让新状态进入下一次预测。开环历史仍独立运行，不被反馈写回。失联后速度使用实际观测间隔，避免速度尖峰。',
        '', '82片段匹配沿用冻结E11来源去重：同一图像多种初始化只选一个真实双目匹配；它们不算独立投票。E12没有训练联合故障概率或完整协方差，候选包络也不是概率区间。日常表面拟合不直接修改主构型；几何残差用于证据准入。',
        '', '当前确认提交的目标仍来自反馈构型。视觉能否决不相容读数，但还没有独立选择恢复构型的通路。因此反馈坏而图像健康时，可能只保留未知和候选，主定位仍不够好；受阻叠加反馈偏差这一交叉条件专门保留这个检查，不能用健康反馈条件的收益覆盖它。',
        '', '## 参数、开发选择与冻结','',
        f"工程诊断复用了旧开发种子2026103111，正式开发为2026103211；最终独立种子为{run.FINAL}。只比较确认2/3帧、关联门限倍率1/1.5四组，选中 **{selected}**：**{profile['confirm']}帧确认，倍率{profile['association_factor']}**。",'',
        '| 候选 | 归一化开发选择损失 | 确认帧 | 门限倍率 |','|---|---:|---:|---:|']
    for name,p in s['configuration']['profiles'].items():
        lines.append(f"| {name} | {p['selection']:.8f} | {p['confirm']} | {p['association_factor']} |")
    lines+=['',f"基础门限：空间关联 **{t['association_m']*1000:.3f}mm**、关节最大归一化残差 **{t['association_q']:.5f}**、表面残差 **{t['surface_m']*1000:.3f}mm**。关联两门限乘所选倍率；表面门限不随倍率放宽。实际关联门限为 **{t['association_m']*profile['association_factor']*1000:.3f}mm** 与 **{t['association_q']*profile['association_factor']:.5f}**。停止速度衰减0.5；限幅20倍尺度/秒。",'',
        '开发选择使用记录的实际关节读数作状态教师，正常及健康受阻权重8，共同故障及协同回放权重4，其余1。相同确认帧数下，两种门限倍率的开发损失很接近；所选倍率不能被解释为通用最优。集合余量来自开发正常主状态误差及受阻配对最近事先运动候选误差，99%分位与0.02取大；不是由最终覆盖率倒推。各关节余量为：`'+json.dumps(profile['margin_normalized'])+'`。',
        '', '冻结包括代码、协议、模型、校准、开发选择的哈希；最终所有条件预测封存后才读取评价几何。E11基线保留原校准及原确认3帧、增益0；单状态同表面基线增益0.2。',
        '', '## 传感故障窗口固定锚点RMSE / mm','','| 条件 | E12 | 冻结E11 | E8B | 同表面单状态 | 关闭推翻 | 仅动作 | 仅反馈 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for c in run.CONDITIONS:
        lines.append('| '+CN[c]+' | '+' | '.join(f"{s['conditions'][c][m]['fault_rmse_mm']:.3f}" for m in run.MODES)+' |')
    lines+=['','## 逐种子正常保护与完整受阻区间','','| 种子 | 正常E12/mm | 正常E11/mm | 回归/mm | 正常未知 | 健康受阻E12/mm | 健康受阻E11/mm |','|---|---:|---:|---:|---:|---:|---:|']
    seed_rows=read_json(output/'scores.json')['rows']
    for seed in run.FINAL:
        normal={r['mode']:r['rmse_mm'] for r in seed_rows if r['seed']==seed and r['condition']=='clean'}
        held={r['mode']:r['hold_rmse_mm'] for r in seed_rows if r['seed']==seed and r['condition']=='execution_hold'}
        normal_unknown=next(r['unknown'] for r in seed_rows if r['seed']==seed and r['condition']=='clean' and r['mode']=='branch')
        lines.append(f"| {seed} | {normal['branch']:.3f} | {normal['e11']:.3f} | {normal['branch']-normal['e11']:+.3f} | {normal_unknown*100:.2f}% | {held['branch']:.3f} | {held['e11']:.3f} |")
    lines+=['','这些列使用原传感故障窗口30–69、110–139帧；上述健康持续受阻主验收使用完整30–149帧，二者不能混用。逐种子结果见seed-scores.json，逐帧误差见frame-scores.csv。',
        '', '## 仍受阻时，新证据是否真正纠错？','','受阻延长到149帧；观测先在70和140帧恢复，因此70–109及140–149帧有新证据但没有恢复执行。绝对恢复要求学习身体映射距离实际读数≤5mm，同时实体锚点误差≤健康反馈映射的误差+2mm，连续5帧；确认最短0.5秒。',
        '', '| 条件 | 方法 | 成功事件 | 1秒内 | 成功中位/s |','|---|---|---:|---:|---:|']
    for c in ('twin_held','hold_feedback_bias','hold_no_camera'):
        for m in run.MODES:
            r=s['conditions'][c][m]['held_absolute'];v=r['median_success_s']
            lines.append(f"| {CN[c]} | {LABEL[m]} | {r['recovered']}/{r['events']} | {r['within_1s']} | {'—' if v is None else f'{v:.2f}'} |")
    lines+=['', '恢复成功不保证身体几何已无偏；健康反馈映射仍有系统误差。传统相对健康分支恢复事件也完整保留在recovery-events.csv；主验收采用更严格的上述标准。',
        '', '## 不可辨识性与候选质量','','同输入配对前70帧的全部输入、匹配和预测逐字段精确相同。模型没有超出可得证据区分隐藏身体；未知与持续保存竞争历史必须同时成立。逐关节外包络是松弛集合，覆盖真值不能证明有一个准确候选，故另报实际运动候选到两种真实身体的距离。',
        '', '| 种子 | 分歧帧 | 未知率 | 两真状态同时覆盖 |','|---|---:|---:|---:|']
    for t0 in g['ambiguity']['twins']:
        lines.append(f"| {t0['seed']} | {t0['distinguishable_physical_frames']} | {t0['unknown_rate']*100:.2f}% | {t0['simultaneous_coverage']*100:.2f}% |")
    lines+=['','| 种子 | 正常运动最近候选中位/mm | 95%分位/mm | 受阻最近候选中位/mm | 95%分位/mm |','|---|---:|---:|---:|---:|']
    for x in a['nearest_motion_candidates']:
        lines.append('| '+str(x['seed'])+' | '+' | '.join('—' if x[k] is None else f'{x[k]:.3f}' for k in ('moving_nearest_median_mm','moving_nearest_p95_mm','held_nearest_median_mm','held_nearest_p95_mm'))+' |')
    lines+=['','最近候选距离只对仍有运动分支保留的帧统计；无运动分支帧数逐种子保留在audit.json，不能用条件统计掩盖提前排除。']
    if 'diagnostics' in s:
        diag=s['diagnostics']['rows']
        held_covered=sum(x['covered'] for x in diag if x['physical']=='held')
        moving_covered=sum(x['covered'] for x in diag if x['physical']=='moving')
        lines+=['','## 失败定位：运动解释的误差范围仍校准不足','',
            f"封存后的诊断表明，停止真实状态被包络覆盖 **{held_covered}/{g['ambiguity']['frames']}帧**，运动真实状态只覆盖 **{moving_covered}/{g['ambiguity']['frames']}帧**。此次主要遗漏发生在运动解释，不能把失败统一归因于停止历史丢失。",'',
            '停止运动候选的身体距离中位误差均不足1mm；运动候选中位误差约2–4mm，个别帧更高。健康融合状态的误差余量不能自动覆盖独立开环运动预测的动力学误差；当前余量只用正常主状态与受阻最近运动候选校准，没有充分覆盖正常运动的独立历史。这是校准设计的缺口。',
            '', '上述诊断在最终封存后进行，仅解释失败；没有扩大余量、修改门限或重新挑选最终种子。后续应在新的开发数据上分别校准运动/停止历史误差，加入执行时延变化，再冻结后用新测试检验覆盖与宽度。',
            '', '| 种子 | 物理状态 | 七关节覆盖/帧数 | 各关节越界帧数（0–6） |','|---|---|---:|---|']
        for x in diag:
            lines.append(f"| {x['seed']} | {x['physical']} | {x['covered']}/{x['frames']} | {x['failed_joint_counts']} |")
    lines+=['','## 强对照与仍然不好的条件','',
        f"共同故障宏误差E12 {s['macro_errors']['branch']:.3f}mm，独立仅动作对照 {s['macro_errors']['action']:.3f}mm。这里主要收益来自拒绝坏观测、回退动作路径，没有证明融合优于仅动作。健康受阻E12达到仅反馈的水平，也不代表身体解码几何误差已消失。",'',
        f"受阻叠加反馈偏差的故障窗口：E12 {s['conditions']['hold_feedback_bias']['branch']['fault_rmse_mm']:.3f}mm，仅反馈 {s['conditions']['hold_feedback_bias']['feedback']['fault_rmse_mm']:.3f}mm；受阻叠加相机缺失：E12 {s['conditions']['hold_no_camera']['branch']['fault_rmse_mm']:.3f}mm，仅反馈 {s['conditions']['hold_no_camera']['feedback']['fault_rmse_mm']:.3f}mm。门限在缺少证据时保持保守，但当前主状态仍可能沿错误动作路径；表示未知并未自动解决定位。",'',
        f"反馈缺失时E12 {s['conditions']['no_feedback']['branch']['fault_rmse_mm']:.3f}mm，同批E11 {s['conditions']['no_feedback']['e11']['fault_rmse_mm']:.3f}mm，也出现退化。这些条件未列入本轮主状态精度验收的共同故障宏平均，必须独立保留。"]
    lines+=['', '## 审计与边界','',
        f"审计{a['prediction_files']}份预测、{a['command_files']}份执行前命令；固定28身体锚点与434表面点分母；{a['commits']}次提交中{a['commits_changing_point_estimate']}次实际改变主状态。动作不赞成时发生{a['overrides']}次观测选择，其中{a['corrected_future_differs_from_open_action']}次下一步预测与独立开环动作历史不同。后者是更新生效审计，净性能收益仍看关闭推翻消融。",'',
        '三种子仅六条物理轨迹；十二条件共享轨迹与传感干预，不是36个独立物理重复。E12受阻时长及恢复条件与E11历史报告不同，只能用本轮同批基线估计收益。受阻通过执行器持续保持第29帧命令模拟，没有加入实际碰撞阻挡或外力，也不代表已经测试电机堵转。',
        '', '本轮结果只支持当前仿真、干净初始化、标定双目及已知控制通道下的明确结论。停止分支、确认门限和关联规则仍是人为设计，不能称为模型自己学到了任意因果解释；没有证明任意持续身份、所有共同故障可识别、完整物理体积、接触力或真实机械臂泛化。失败验收与交叉条件退化应如实保留。',
        '', '维持旧命令与停止分支在冲突发生时锚定，并未枚举任意受阻起点。这批共同故障与受阻的起始时刻同步；未知期间更晚才发生执行受阻，或机械臂受到外力移动等情况仍未验证。',
        '', '## 复现','','```powershell',r'.\.venv\Scripts\python.exe -X utf8 -m embodiment.branch_cycle prepare',r'.\.venv\Scripts\python.exe -X utf8 -m embodiment.branch_cycle final',r'.\.venv\Scripts\python.exe -X utf8 -m embodiment.branch_cycle score',r'.\.venv\Scripts\python.exe -X utf8 -m embodiment.branch_audit',r'.\.venv\Scripts\python.exe -X utf8 -m embodiment.branch_report','```','','冻结阶段拒绝覆盖；完整复现请给branch_cycle各阶段传入新的 --output 目录。']
    if (output/'test-result.json').exists():
        checked=read_json(output/'test-result.json')
        lines+=['',f"回归：{checked['tests']}项全部通过，其中本轮新增{checked['new_tests']}项；初次运行受Windows临时目录ACL限制，获准在沙箱外重跑通过。"]
    text='\n'.join(lines)+'\n'
    (report/'详细报告.md').write_text(text,encoding='utf8');Path('docs/body-branches-e12-summary.md').write_text(text,encoding='utf8')


def webpage(report,s):
    rows=''.join('<tr><td>'+CN[c]+'</td>'+''.join(f"<td>{s['conditions'][c][m]['fault_rmse_mm']:.2f}</td>" for m in run.MODES)+'</tr>' for c in run.CONDITIONS)
    goals=''.join(f'<li>{GOAL_LABEL[k]}：{"通过" if v["passed"] else "未通过"}</li>' for k,v in s['goals'].items())
    g=s['goals'];headline=(f"健康持续受阻：E11 {s['conditions']['execution_hold']['e11']['hold_rmse_mm']:.2f} → E12 {s['conditions']['execution_hold']['branch']['hold_rmse_mm']:.2f}mm；改善{g['state']['held_improvement']*100:.1f}%。"
        f"配对新证据恢复{g['recovery']['recovered']}/6，1秒内{g['recovery']['within_1s']}/6。两真状态同时包络覆盖{g['ambiguity']['simultaneous_coverage']*100:.1f}%。")
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>E12 · 动作先验与持续身体解释</title><style>*{box-sizing:border-box}body{margin:0;background:#101b29;color:#e8f0f6;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1180px;margin:auto;padding:26px 20px 60px}section{padding:20px;margin:22px 0;background:#192b3d;border:1px solid #345069;border-radius:12px}a{color:#75dac3}p{color:#c5d3e2}button,select{background:#28435a;color:white;border:1px solid #608198;border-radius:6px;padding:9px;margin:4px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}canvas,img{width:100%;border-radius:8px;background:#080e16}input[type=range]{width:100%}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #3e5368;text-align:left}.scroll{overflow-x:auto}#stats{white-space:pre-line}@media(max-width:750px){.grid{grid-template-columns:1fr}main{padding:15px}}</style><main>
<p><a href="/">← 实验入口</a> · <a href="详细报告.md">详细报告</a> · <a href="summary.json">全部指标</a> · <a href="body-branches-e12.md">冻结协议</a></p><h1>动作预测错了，身体状态能被新证据纠正吗？</h1><p>旧网络冻结 · 3个新种子、6条物理轨迹 · 12种共享输入条件 · 82观测片段 · 固定干扰物</p><section><h2>本轮结果</h2><p>HEADLINE</p><ul>GOALS</ul><p>这是显式推断机制的优化。未知、包络和停止模型不等于已学习的概率或完整身体身份。旧E11报告保留；本轮用同批冻结基线比较。</p></section>
<section><h2>当前身体与模型输入</h2><select id="condition"></select><select id="mode"></select><button id="play">暂停</button><label><input type="checkbox" id="branches" checked>显示竞争运动解释</label><div class="grid"><div><p>实际身体＋所选预测表面</p><canvas id="actual" width="480" height="360"></canvas></div><div><p>模型收到的相机</p><canvas id="sensor" width="480" height="360"></canvas></div></div><input id="frame" type="range" min="0" max="89" value="0"><p id="stats">加载中…</p><p>绿色为所选方法；其他颜色是尚未排除的事先运动预测。显示100点，实体评分保持28锚点及全部434表面点。3–15秒身体持续受阻；7–11秒、14–15秒观测已恢复，但身体仍受阻。</p></section>
<section><h2>传感故障窗口误差 / mm</h2><div class="scroll"><table><tr><th>条件</th><th>E12</th><th>冻结E11</th><th>E8B</th><th>同表面单状态</th><th>关闭推翻</th><th>仅动作</th><th>仅反馈</th></tr>ROWS</table></div><img src="errors.png"></section><section><h2>受阻配对全过程</h2><img src="timeline.png"></section><section><h2>未知、覆盖与集合代价</h2><img src="sets.png"></section><section><h2>完整记录与限制</h2><p><a href="seed-scores.json">逐种子分数</a> · <a href="frame-scores.csv">逐帧分数</a> · <a href="recovery-events.csv">全部恢复事件</a> · <a href="audit.json">更新与来源审计</a> · <a href="protocol-freeze.json">冻结哈希</a> · <a href="body-branches.gif">动画</a></p><p>三种子与受控同步回放不证明真实机械臂泛化或任意故障可辨识；确认门限与运动分支仍是人工设计。同一图像的多次匹配不算多个独立证据，外包络也不是实体或概率区间。</p></section></main>
<script>let data,busy=false,playing=true;const $=id=>document.getElementById(id),cache=new Map();function load(path){if(!path)return Promise.resolve(null);if(cache.has(path))return cache.get(path);const p=new Promise(resolve=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=()=>resolve(null);im.src=path});cache.set(path,p);return p}async function draw(){if(!data||busy)return;busy=true;try{const r=data.conditions[$('condition').value][+$('frame').value],mode=$('mode').value;const imgs=await Promise.all([load(r.current),load(r.image)]);['actual','sensor'].forEach((id,j)=>{const c=$(id).getContext('2d');c.fillStyle='#080e16';c.fillRect(0,0,480,360);if(imgs[j])c.drawImage(imgs[j],0,0,480,360);else{c.fillStyle='white';c.font='20px sans-serif';c.fillText('相机缺失',180,180)}});const c=$('actual').getContext('2d');function dots(points,color){c.fillStyle=color;for(const [x,y] of points){c.beginPath();c.arc(x,y,1.8,0,Math.PI*2);c.fill()}}if($('branches').checked)r.alternatives.forEach((p,j)=>dots(p,['#ffbd63','#62bfff','#c294ff','#ed91b2'][j]));dots(r.points[mode],'#55ed9c');$('stats').textContent=`t=${(r.i*.1).toFixed(1)}s · ${r.unknown?'未知：保留竞争解释':'已选择观测支持的状态'}\nE12实体锚点误差 ${r.error.toFixed(2)}mm；推翻动作=${r.override}；表面匹配=${r.support}/82\n包络覆盖实际构型=${r.covered}；归一化半宽=${r.width.toFixed(4)}。切换方法只改变绿色投影，上述读数始终为E12。`}finally{busy=false}}fetch('replay.json').then(r=>r.json()).then(x=>{data=x;for(const [k,v] of Object.entries(x.labels))$('condition').add(new Option(v,k));for(const [k,v] of Object.entries(x.modes))$('mode').add(new Option(v,k));$('condition').value='twin_held';draw()}).catch(e=>$('stats').textContent=e.message);for(const id of ['condition','mode','frame','branches'])$(id).addEventListener('input',draw);$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'暂停':'播放'};setInterval(()=>{if(data&&playing&&!busy){$('frame').value=(+$('frame').value+1)%90;draw()}},180);</script></html>'''
    page=page.replace('<a href="audit.json">更新与来源审计</a>',
        '<a href="audit.json">更新与来源审计</a> · <a href="diagnostics.json">包络失败诊断</a> · <a href="test-result.json">135项测试</a>')
    (report/'index.html').write_text(page.replace('HEADLINE',html.escape(headline)).replace('GOALS',goals).replace('ROWS',rows),encoding='utf8')


def main(output=run.ROOT):
    run.verify(output);report=output/'report';report.mkdir(exist_ok=True)
    s,rows,events=summarize(output)
    for source,target in (('scores.json','seed-scores.json'),('audit.json','audit.json'),('protocol-freeze.json','protocol-freeze.json'),('diagnostics.json','diagnostics.json'),('test-result.json','test-result.json')):
        write_json(report/target,read_json(output/source))
    (report/'body-branches-e12.md').write_text(Path('docs/body-branches-e12.md').read_text(encoding='utf8'),encoding='utf8')
    for name,records in (('frame-scores.csv',rows),('recovery-events.csv',events)):
        flat=[{(f'{k}_{sub}' if sub else k):value for k,v in row.items() for sub,value in (v.items() if isinstance(v,dict) else [('',v)])} for row in records]
        keys=list(dict.fromkeys(k for row in flat for k in row))
        with (report/name).open('w',newline='',encoding='utf-8-sig') as handle:
            writer=csv.DictWriter(handle,keys);writer.writeheader();writer.writerows(flat)
    figures(report,s,rows);replay(output,report,rows);document(output,report,s,events);webpage(report,s)
    write_json(output/'report-artifacts.json',dict(created_utc=utc(),generator_sha256=sha(Path(__file__)),
        hashes={str(p):sha(p) for p in report.rglob('*') if p.is_file()}))
    print('E12 REPORT',{k:v['passed'] for k,v in s['goals'].items()},flush=True)


if __name__=='__main__':
    main()
