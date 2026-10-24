"""Standalone inspection of learned 3D entity predictions and sensor loss."""
from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def build(data, output, comparison=None, version='E1'):
    data, output = Path(data), Path(output)
    report = json.loads((output/'metrics.json').read_text(encoding='utf-8'))
    model_output = Path(report.get('model_directory', str(output)))
    history = json.loads((model_output/'history.json').read_text(encoding='utf-8'))
    metadata = json.loads((data/'metadata.json').read_text(encoding='utf-8'))
    controls = json.loads((model_output/'feedback-discovery-controls.json').read_text(encoding='utf-8'))
    dashboard = ''
    if comparison:
        comparison = Path(comparison)
        comparison_data = json.loads((comparison/'comparison.json').read_text(encoding='utf-8'))
        paired = comparison_data['paired_geometry']['test']
        dashboard = ('<section class="dashboard"><h2>本轮实测数据</h2><p>两个模型在冻结后新采集的同一条轨迹上比较。'
                     f'新动作中的配对身体点误差：<strong>{paired["baseline_rmse_mm"]:.2f} → {paired["candidate_rmse_mm"]:.2f} mm</strong>。'
                     '这是同一点、同一像素的比较；不是整个机械臂的重建精度。</p><div class="plot-grid">')
        for index in range(1, 5):
            filename = f'panel-{index}.png'
            (output/filename).write_bytes((comparison/filename).read_bytes())
            dashboard += f'<a href="{filename}" target="_blank"><img src="{filename}" alt="实测数据图 {index}，点击查看原图"></a>'
        dashboard += '</div>'
        for filename in ('feedback-controls.png', 'comparison.png', 'comparison.json', 'per-frame.csv'):
            (output/filename).write_bytes((comparison/filename).read_bytes())
        error = report['summary']['test']['model_rmse_m']*1000
        dashboard += (f'<p class="note">仍有失败：少数特征会错跟到背景。包含全部有效特征时，新动作预测对双目轨迹的误差为 {error:.2f} mm，'
                      '高于上面的身体点几何误差。模型的“视觉支持”按中位误差判定，会掩盖少数异常对应；它不表示每个点都被正确确认。</p>'
                      '<a href="feedback-controls.png" target="_blank"><img src="feedback-controls.png" alt="正确反馈、固定位置、错配反馈与相机中断对照"></a>'
                      '<p><a href="comparison.png" target="_blank">完整对比图</a> · <a href="per-frame.csv" download>逐帧数据 CSV</a> · '
                      '<a href="comparison.json">对比指标 JSON</a> · <a href="metrics.json">本模型完整指标</a></p></section>')
        if (comparison/'failure-example.png').exists():
            (output/'failure-example.png').write_bytes((comparison/'failure-example.png').read_bytes())
            dashboard += ('<details><summary>展开具体失败：特征错跟到背景</summary>'
                          '<img src="failure-example.png" alt="特征 302 从机械臂附近错跟到桌面，观测与预测相差 821 mm">'
                          '<p>红色为图像跟踪位置，黄色为模型投影。这个异常保留在整体误差中，未因最终测试结果而删除或重新训练。</p></details>')
    frames = []
    for index, row in enumerate(report['per_frame']):
        if index % 2:
            continue
        entry = {key: row[key] for key in ('frame', 'phase', 'time', 'identity', 'status', 'points',
                                          'uncertainty_radius_m', 'measured_q')}
        with np.load(data/'observations'/metadata['frames'][row['frame']]['file']) as observation:
            buffer = BytesIO()
            Image.fromarray(observation['left']).resize((480, 360)).save(buffer, format='JPEG', quality=85)
        entry['image'] = 'data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii')
        frames.append(entry)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for key, label in [('train_squared_m', 'Training'), ('validation_squared_m', 'Validation')]:
        axes[0].plot([r['epoch'] for r in history], [np.sqrt(r[key])*1000 for r in history], label=label)
    axes[0].set(xlabel='Epoch', ylabel='3D error vs stereo tracks (mm)', title='Observation-only learning')
    axes[0].legend()
    for field, label in [('squared_error_sum', 'Learned entity'), ('fixed_squared_error_sum', 'Fixed learned position')]:
        axes[1].plot([r['time'] for r in report['per_frame']],
                     [np.sqrt(r[field]/max(r['valid_features'], 1))*1000 for r in report['per_frame']], label=label)
    blind_start = next(r['time'] for r in report['per_frame'] if r['phase'] == 'blind')
    axes[1].axvspan(blind_start, report['per_frame'][-1]['time'], alpha=.15, color='orange', label='Camera unavailable')
    axes[1].set(xlabel='Simulation time (s)', ylabel='3D error vs stereo tracks (mm)', title='Frozen unseen trajectories')
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(output/'learning-and-prediction.png', dpi=150)
    plt.close(fig)
    table = ''
    for phase, label in [('test', '新动作'), ('hold', '停止阶段'), ('blind', '相机中断')]:
        row = report['summary'][phase]
        table += '<tr><td>'+label+'</td>'+''.join('<td>'+f'{row[k]*1000:.2f}'+'</td>' for k in
            ('model_rmse_m', 'shuffled_q_rmse_m', 'body_geometric_rmse_m'))+'</tr>'
    chart = base64.b64encode((output/'learning-and-prediction.png').read_bytes()).decode('ascii')
    document = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>机械臂实体空间模型 · __VERSION__</title>
<style>*{box-sizing:border-box}body{margin:0;background:#101820;color:#e6edf5;font:15px/1.65 system-ui,'Microsoft YaHei',sans-serif}main{max-width:1220px;margin:auto;padding:24px}h1{font-size:25px;margin:4px 0}h2{font-size:19px}p{color:#b9c9d7}a{color:#76d4c2}.tag{color:#72d8bb;font-size:12px;letter-spacing:2px}.note{background:#223241;border-left:3px solid #f4ba65;padding:12px 16px}.views{display:grid;grid-template-columns:1fr 1fr;gap:16px}.card{background:#18242e;border:1px solid #334552;border-radius:10px;overflow:hidden}.card h2{margin:12px 16px}.card p{margin:10px 16px;font-size:13px}canvas{display:block;width:100%;aspect-ratio:4/3;background:#0b1118;touch-action:none}.bar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:16px 0}button{border:1px solid #435765;border-radius:6px;color:#e6edf5;background:#263d49;padding:9px 13px;cursor:pointer}button.active{background:#286b5a}input{flex:1;min-width:160px}#status{padding:12px;background:#20362f;border-radius:8px}#detail{font-size:13px;color:#b7c9d5}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:10px;border-bottom:1px solid #384a57}img{max-width:100%;height:auto}code{overflow-wrap:anywhere}@media(max-width:720px){main{padding:14px}.views{grid-template-columns:1fr}h1{font-size:21px}}</style></head><body><main>
<style>.plot-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.plot-grid img{display:block;width:100%;background:white}.table-scroll{overflow-x:auto}strong{color:#7ee3c1}@media(max-width:720px){.plot-grid{grid-template-columns:1fr}}</style>
<div class="tag">BODY ENTITY / EXPERIMENT __VERSION__</div><h1>实体空间预测 · 实测数据与回放</h1>
<p>通过双目 RGB 与实际关节反馈，发现随自身状态稳定变化的表面特征，并学习预测它们的三维位置。这里展示冻结模型在未参与学习的连续动作上的结果。</p>
__DASHBOARD__
<div class="note">这是初步的实体候选模型：__POINT_COUNT__ 个稀疏特征，尚未重建完整表面和实体内部。测试中严格落在机械臂像素内的特征观测为 __PURITY__；轮廓跟踪和双目深度有偏差。三维点的存在不等于已经完成实体理解。</div>
<div class="bar"><button id="play">播放轨迹</button><button data-phase="test">新动作</button><button data-phase="hold">停止阶段</button><button data-phase="blind">相机中断</button></div>
<div class="views"><section class="card"><h2>机器人眼睛 · 原始输入</h2><canvas id="eye" width="640" height="480"></canvas><p>黄色标记是实体模型预测点的投影。相机中断时此处没有隐藏图像，预测仍使用实测关节反馈。</p></section>
<section class="card"><h2>模型预测的三维实体特征</h2><canvas id="space" width="640" height="480" aria-label="可拖动旋转的三维实体点"></canvas><p>拖动旋转观察。绿色点是学习到的位置预测；线框仅包围已学点，不表示实体的完整体积。坐标单位为米。</p></section></div>
<div class="bar"><span id="time"></span><input id="timeline" type="range" min="0" step="1" aria-label="实验时间线"></div>
<div id="status"></div><p id="detail"></p>
<h2>反馈是否真正发挥作用</h2><p>使用正确反馈时序发现 __POINT_COUNT__ 个候选点；打乱反馈时序为 __SHUFFLED_COUNT__ 个，固定反馈为 __CONSTANT_COUNT__ 个。冻结的三维模型只输入实测 q，故相机中断时可以继续给出空间预测；时序身份与不确定性管理目前由显式状态机制实现。</p>
<div class="table-scroll"><table><tr><th>阶段</th><th>预测对双目轨迹 RMSE / mm</th><th>打乱当前 q / mm</th><th>身体点对仿真几何 RMSE / mm</th></tr>__TABLE__</table></div>
<p>最后一列只评价严格落在身体上的特征；本次测试的仿真几何仅用于冻结后的评分。新动作阶段双目本身的身体点几何误差为 __STEREO_ERROR__ mm，不能把拟合观测的精度当成真实物理精度。相机中断时，保持最后所见位置的误差为 __LAST_SEEN__ mm，模型对同一双目参考为 __BLIND_ERROR__ mm。</p>
<h2>训练与持续预测</h2><img src="data:image/png;base64,__CHART__" alt="训练曲线和相机中断期间的空间预测误差">
<p>此次评价 __EVAL_COUNT__ 帧：__TEST_COUNT__ 新动作、__HOLD_COUNT__ 停止、__BLIND_COUNT__ 相机中断。训练和特征发现均不使用 CAD、身体身份或仿真深度。相机标定提供焦距、6 cm 双目基线与光学坐标系。深度学习部分是 7 → 64 → 64 → __OUTPUT_COUNT__ 的位置网络；双目测量、特征筛选和实体状态管理仍是显式算法。</p>
<p>当前限于一个局部动作范围；仍需学习更完整的实体形态、部件关系、接触与占据空间约束，检验移动视角和未知工具。实体记忆不会把“暂时看不见”直接改成“不存在”，但目前的保持规则也不等于已经学习了物体恒存。</p>
<p>模型 SHA-256：<code>__HASH__</code>。这是可复查的录制实验，原 MuJoCo 实时相机仍在 <a href="http://127.0.0.1:8765/camera">原始相机页面</a>运行。</p>
</main><script>
const frames=__FRAMES__,cal=__CALIBRATION__;
const eye=document.getElementById('eye'),space=document.getElementById('space'),slider=document.getElementById('timeline');
const ec=eye.getContext('2d'),sc=space.getContext('2d');let index=0,yaw=.5,pitch=-.2,playing=false,last=0,requestId=0;
slider.max=frames.length-1;const center=[0,0,0];frames[0].points.forEach(p=>p.forEach((v,i)=>center[i]+=v/frames[0].points.length));
const rotate=p=>{let x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2];let a=x*Math.cos(yaw)+z*Math.sin(yaw),b=-x*Math.sin(yaw)+z*Math.cos(yaw);return [320+a*780,240+(y*Math.cos(pitch)-b*Math.sin(pitch))*780,y*Math.sin(pitch)+b*Math.cos(pitch)];};
function drawSpace(){sc.fillStyle='#0b1118';sc.fillRect(0,0,640,480);const f=frames[index];let lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];f.points.forEach(p=>p.forEach((v,i)=>{lo[i]=Math.min(lo[i],v);hi[i]=Math.max(hi[i],v)}));const corners=Array.from({length:8},(_,i)=>[0,1,2].map(k=>i&(1<<k)?hi[k]:lo[k]));sc.strokeStyle='#2d4654';sc.lineWidth=1;corners.forEach((p,i)=>[1,2,4].forEach(bit=>{if(!(i&bit)){const a=rotate(p),b=rotate(corners[i|bit]);sc.beginPath();sc.moveTo(a[0],a[1]);sc.lineTo(b[0],b[1]);sc.stroke();}}));
f.points.map(rotate).sort((a,b)=>b[2]-a[2]).forEach(p=>{sc.beginPath();sc.arc(p[0],p[1],4.5,0,Math.PI*2);sc.fillStyle=f.phase==='blind'?'#f5bd68':'#58dfae';sc.fill();});
sc.font='16px system-ui';sc.fillStyle='#a9bfcd';sc.fillText(f.phase==='blind'?'CAMERA OFF · PREDICTED FROM FEEDBACK':'LEARNED 3D SURFACE FEATURES',20,28);sc.fillText('__POINT_COUNT__ features · drag to rotate',20,455);sc.strokeStyle='#adc1cf';sc.beginPath();sc.moveTo(530,440);sc.lineTo(608,440);sc.stroke();sc.fillText('0.10 m',540,430);}
function show(){const f=frames[index];slider.value=index;document.getElementById('time').textContent=`t = ${f.time.toFixed(1)} s`;document.getElementById('status').textContent=({visually_supported:'视觉支持当前实体预测',predicted_unobserved:'相机不可用 · 保留实体并根据实测反馈预测',visual_contradiction:'视觉与实体预测不一致 · 需要重新观察'})[f.status];document.getElementById('detail').textContent=`实体标识：${f.identity}　｜　预测误差提示尺度：${(f.uncertainty_radius_m*1000).toFixed(0)} mm（工程规则，非存在概率）`;drawSpace();let token=++requestId;const im=new Image();im.onload=()=>{if(token!==requestId)return;ec.drawImage(im,0,0,640,480);if(f.phase==='blind'){ec.fillStyle='#121d26';ec.fillRect(0,0,640,480);ec.fillStyle='#f5bd68';ec.font='26px system-ui';ec.fillText('CAMERA INPUT UNAVAILABLE',105,235);ec.font='18px system-ui';ec.fillStyle='#9eb4c2';ec.fillText('Only measured joint feedback is available',115,275);}else{ec.fillStyle='#ffd279';f.points.forEach(p=>{const x=p[0]/p[2]*cal.focal_px+cal.cx,y=p[1]/p[2]*cal.focal_px+cal.cy;ec.beginPath();ec.arc(x,y,3,0,Math.PI*2);ec.fill();});}};im.src=f.image;document.querySelectorAll('[data-phase]').forEach(b=>b.classList.toggle('active',b.dataset.phase===f.phase));}
document.getElementById('play').onclick=()=>{playing=!playing;document.getElementById('play').textContent=playing?'暂停播放':'播放轨迹';};slider.oninput=()=>{index=Number(slider.value);show();};document.querySelectorAll('[data-phase]').forEach(b=>b.onclick=()=>{index=frames.findIndex(f=>f.phase===b.dataset.phase);show();});
let dragging=null;space.onpointerdown=e=>{dragging=[e.clientX,e.clientY];space.setPointerCapture(e.pointerId);};space.onpointermove=e=>{if(!dragging)return;yaw+=(e.clientX-dragging[0])*.008;pitch+=(e.clientY-dragging[1])*.008;dragging=[e.clientX,e.clientY];drawSpace();};space.onpointerup=()=>dragging=null;
function tick(t){if(playing&&t-last>200){index=(index+1)%frames.length;show();last=t;}requestAnimationFrame(tick);}show();requestAnimationFrame(tick);
</script></body></html>'''
    document = (document.replace('__TABLE__', table).replace('__CHART__', chart)
                .replace('__VERSION__', version).replace('__DASHBOARD__', dashboard)
                .replace('__EVAL_COUNT__', str(len(report['per_frame'])))
                .replace('__TEST_COUNT__', str(report['summary']['test']['frames']))
                .replace('__HOLD_COUNT__', str(report['summary']['hold']['frames']))
                .replace('__BLIND_COUNT__', str(report['summary']['blind']['frames']))
                .replace('__OUTPUT_COUNT__', str(report['point_count']*3))
                .replace('__STEREO_ERROR__', f"{1000*report['summary']['test']['stereo_body_geometric_rmse_m']:.2f}")
                .replace('__HASH__', report['checkpoint_sha256'])
                .replace('__FRAMES__', json.dumps(frames, ensure_ascii=False))
                .replace('__CALIBRATION__', json.dumps(report['calibration']))
                .replace('__POINT_COUNT__', str(report['point_count']))
                .replace('__PURITY__', f"{100*report['summary']['test']['body_purity']:.1f}%")
                .replace('__SHUFFLED_COUNT__', str(controls['shuffled_time_q']['selected_count']))
                .replace('__CONSTANT_COUNT__', str(controls['constant_q']['selected_count']))
                .replace('__LAST_SEEN__', f"{1000*report['summary']['blind']['last_seen_rmse_m']:.2f}")
                .replace('__BLIND_ERROR__', f"{1000*report['summary']['blind']['model_rmse_m']:.2f}"))
    (output/'index.html').write_text(document, encoding='utf-8')
    print('Saved', output/'index.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--comparison')
    parser.add_argument('--version', default='E1')
    build(**vars(parser.parse_args()))
