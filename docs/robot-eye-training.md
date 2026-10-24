# V6：机器人眼位的重新训练与验证

2026-10-24。本轮将肩侧上方的机器人眼位用于真正的新训练，并将 V6 接入实时页面。输入仍为 RGB 和七路实际关节位置，输出当前可见的活动机械臂及夹爪区域；桌面、固定底座和外物视为环境。训练不读取身体真值、几何身份、CAD 或深度。

## 同一批新姿态的结果

先冻结权重与 0.5 阈值，再用 seed `20261202` 采集 50 个新姿态。V5 与 V6 使用完全相同的图像和实际反馈。指标为逐姿态平均，杂物误判率按可见外物像素汇总。

| 指标 | 原 V5 | 新 V6 |
|---|---:|---:|
| 初始帧身体 IoU | 17.94% | **84.43%** |
| 运动后停止帧身体 IoU | 18.16% | **84.56%** |
| 初始帧精确率 | 97.63% | 91.44% |
| 初始帧召回率 | 18.01% | **91.60%** |
| 外物像素被判为身体 | 0% | 3.33% |

V5 的外物误报很低，同时漏掉约 82% 的身体。不能把低误报率单独当作模型更好的证据。这也是本轮必须在新视角重新测量的原因；历史地面场景的 V5 分数不能移用。

另有 100 帧机械臂保持、环境继续变化的对照：V6 身体 IoU **84.38%**，外物误判率 **4.19%**。运动后停止帧来自新的实际姿态，不是复制初始帧。保持识别只说明逐帧模型在这些静止图像上有效，尚不是学习到时间记忆。

| 最终环境配置 | 姿态数 | V6 身体 IoU | 外物误判率 |
|---|---:|---:|---:|
| clean | 10 | 83.46% | 无外物 |
| static | 10 | 86.92% | 0% |
| moving | 10 | 85.02% | 3.36% |
| occlusion 配置 | 10 | 82.57% | 5.93% |
| lighting | 10 | 84.20% | 无外物 |

**遮挡验证未成立。** 本轮最终集全部姿态的实际身体遮挡覆盖率为零，包括名为 `occlusion` 的配置。桌面小物的世界位置范围没有针对新眼位的投影重新设计，物体在桌上并不保证它挡住手臂。因此该组只能当作另一种外物配置，不能宣称抗遮挡已验证。保留这份失败记录；下一轮需要先测量投影覆盖再构造遮挡测试，本轮不修改最终集或重新调参。

## 图像与反馈究竟学到了什么

运动区域的 RGB 光流预测 EPE：正确实际 Δq 为 **2.596 px**，零运动预测为 **4.481 px**，打乱 Δq 对应关系为 **5.025 px**。这支持学到的运动响应与反馈变化之间有对应关系。目标是图像估计光流，包含独立外物运动，并不是真实几何运动真值。

固定 RGB 与正确 Δq，只打乱当前 q：身体 IoU 从 **84.43% 变成 84.30%**，约 **1.48%** 的预测身体并集像素发生变化。当前分割仍主要依赖视觉，尚未证明即时反馈融合的收益。这个诊断衡量冻结模型对输入的依赖，也不能替代重新训练图像单模态模型的消融实验。

因此本轮完成的是：**反馈参与监督的身体区域学习，在当前眼位下取得可用的视觉分割结果。** 距离用户希望的完整多模态身体感知仍有差距。

## 配方、过程与经验

- 新采集 100 训练、25 验证、20 开发姿态，seed `20261201`，五种环境配置均衡。训练背景参考只用训练 RGB 拟合，最终测试缓存复用冻结参考。
- 从头训练运动分支 20 轮、身体分支 30 轮，CPU 四线程、batch 4。RGB 光流、实测反馈的正反向探测、背景参考、GrabCut 弱标签及静止对照构成监督；静止证据权重为 0.5。
- 身体分支阶段的运动损失权重为 0，最终组合独立保存的运动分支，保留运动学习。新视角不再套用旧 V5 候选掩膜，以免旧视角漏检限制新模型。组合后约 25.5 万参数，权重约 1.06 MB。
- 最佳轮次按不读取真值的验证损失选择。运动验证目标从 1.0667 降至 0.6286，身体目标从 0.4020 降至 0.0543；这是弱监督目标，不是身体准确率。总训练含缓存约 378 秒。
- 开发集 IoU：V5 为 17.35%，V6 为 84.81%。检查最低、中位、最高样本后采用第一套配方，没有搜索阈值或额外配方；冻结后另采最终集，未再改动权重。

**图像和反馈要同一时刻。** MuJoCo 步进后，派生几何位置需要刷新。collector 和 demo 现在在 RGB 渲染前执行 `mj_forward`，与当前实测 q 对齐；旧结果按此前代码版本保留，不能将本次改善全部归因于网络训练技巧。

**画面条件也是数据的一部分。** 实时页面使用 -45° 灯光和阴影，本轮将这些灯光写入独立场景快照，并显式记录 `--shadows`。影子区分仍未单独验证；这轮只是把阴影加入真实输入条件。新增检查确认阴影开关改变 RGB，但不改变 q 或评估身体几何。

**换视角会改变整个任务。** 同一手臂在近距离眼位中的位置、尺度和可见表面均发生变化。不能依靠旧模型分数或仅凭肉眼挑图判断泛化，也不能把新旧场景分数直接比较成某一种算法的贡献。

**场景名字不是验证证据。** 实际覆盖统计发现遮挡为零，避免了将“配置了遮挡物”误写成“通过抗遮挡实验”。下次测试设计必须检查干扰是否真正作用在画面和身体上。

## 连续演示与实时接入

另外运行 90 帧、15 FPS 的确定性连续轨迹，外物独立运动，手臂最后保持非初始姿态。运动阶段平均 IoU **84.05%**，稳定保持阶段 **83.66%**。只是一条演示轨迹，不应把 90 帧视作 90 个独立测试姿态。

实时首页默认加载 `outputs/perception/v6-eye/model/model.pt`，显示 V6 和训练场景来源；原始 RGB 与绿色预测始终来自同一仿真帧。独立 `/camera` 页面保留纯相机画面。网络实际输入为 192 × 144 RGB 与七路实测位置，显示图为 640 × 480；物体位置、动作目标和评估掩膜不进入推理。

41 项测试通过，在线 smoke test 通过同步推理、关节与夹爪跟随、暂停、探索、无效输入拒绝、阴影和预测/真值分别保存等检查。该次在线推理约 **31 ms/帧**，随负载变化。两个已有标签页已刷新并保存实际截图。

本地成果均位于 `outputs/perception/v6-eye/`（不纳入 Git）：

| 文件 | 内容 |
|---|---|
| `comparison.html` / `comparison.png` | 配对结果、最低/中位/最高样本、曲线和连续演示 |
| `recipe.json` / `freeze.json` | 配方、代码/场景/权重哈希与冻结时间 |
| `data/` / `final-data/` | 开发阶段数据与冻结后新采集的最终数据 |
| `run-body/history.json` / `training.json` | 全部训练目标与超参数 |
| `model/model.pt` | 最终 V6 权重，含实时展示来源 |
| `final-v5/metrics.json` / `final-v6/metrics.json` | 新旧模型逐姿态评价 |
| `final-v6/feedback-ablation.json` | 当前 q 扰动诊断 |
| `final-v6/robot-eye-demo.gif` / `.json` | 连续推理与逐帧结果 |
| `browser-full.jpg` / `camera-browser.jpg` | 实时页面与独立原图页截图 |

权重 SHA-256：`3afd89ca2cdc2c0e221c32560a987de1a3dbbd538e6ffbba6e0673ca060e15a1`。

实验 XML 完整归档为 `sim/scenes/robot-eye-v6.xml`，SHA-256：`501616f64d0d87f5949a31c0495c7807304338122d185062d1eddd78ececf1f5`。实时场景和实验场景几何/相机一致，实验 XML 将运行时灯光写成静态配置，故 XML 哈希不同。原 V5 权重、数据、旧场景及报告保留。

## 复现入口

使用全新输出目录，以下命令不会覆盖当前 V6。完整实验应先完成开发训练/检查和冻结，再采最终集。

```powershell
.\.venv\Scripts\python.exe -m perception.collect --output outputs/perception/v6-eye-repro/data --scene sim/scenes/robot-eye-v6.xml --train 100 --val 25 --test 20 --environment mixed --shadows --seed 20261201
.\.venv\Scripts\python.exe -m perception.train --data outputs/perception/v6-eye-repro/data --cache outputs/perception/v6-eye-repro/cache --output outputs/perception/v6-eye-repro/run-body --motion-epochs 20 --segmentation-epochs 30 --batch-size 4 --threads 4 --seed 20261201 --refine-labels --use-null-evidence --motion-weight 0 --control-weight 0.5
.\.venv\Scripts\python.exe -m perception.compose --body-checkpoint outputs/perception/v6-eye-repro/run-body/model.pt --motion-checkpoint outputs/perception/v6-eye-repro/run-body/motion.pt --output outputs/perception/v6-eye-repro/model/model.pt
.\.venv\Scripts\python.exe -m perception.evaluate --data outputs/perception/v6-eye-repro/data --checkpoint outputs/perception/v6-eye-repro/model/model.pt --output outputs/perception/v6-eye-repro/development --cache outputs/perception/v6-eye-repro/cache --threads 4
```

冻结后采集最终集，先做不需要光流缓存的身体评价：

```powershell
.\.venv\Scripts\python.exe -m perception.collect --output outputs/perception/v6-eye-repro/final-data --scene sim/scenes/robot-eye-v6.xml --train 0 --val 0 --test 50 --environment mixed --shadows --seed 20261202
.\.venv\Scripts\python.exe -m perception.evaluate --data outputs/perception/v6-eye-repro/final-data --checkpoint outputs/perception/v6-eye-repro/model/model.pt --output outputs/perception/v6-eye-repro/final-v6 --threads 4
.\.venv\Scripts\python.exe -m perception.evaluate --data outputs/perception/v6-eye-repro/final-data --checkpoint outputs/perception/run-v5-supported/model.pt --output outputs/perception/v6-eye-repro/final-v5 --threads 4
```

运动/反馈诊断缓存必须使用训练集冻结参考，不能在最终图像上重新拟合背景：

```powershell
@'
from pathlib import Path
from perception.data import prepare_cache
p = Path('outputs/perception/v6-eye-repro')
prepare_cache(p/'final-data', p/'final-cache', splits=('test',), background_reference=p/'cache/background-reference.npz', refine_labels=True, use_null_evidence=True)
'@ | .\.venv\Scripts\python.exe -
.\.venv\Scripts\python.exe -m perception.ablate_feedback --data outputs/perception/v6-eye-repro/final-data --checkpoint outputs/perception/v6-eye-repro/model/model.pt --cache outputs/perception/v6-eye-repro/final-cache --output outputs/perception/v6-eye-repro/final-v6/feedback-ablation.json --threads 4
.\.venv\Scripts\python.exe -m perception.evaluate --data outputs/perception/v6-eye-repro/final-data --checkpoint outputs/perception/v6-eye-repro/model/model.pt --output outputs/perception/v6-eye-repro/final-v6 --cache outputs/perception/v6-eye-repro/final-cache --feedback-diagnostic outputs/perception/v6-eye-repro/final-v6/feedback-ablation.json --threads 4
.\.venv\Scripts\python.exe -m perception.demo --checkpoint outputs/perception/v6-eye-repro/model/model.pt --output outputs/perception/v6-eye-repro/final-v6/robot-eye-demo.gif --scene sim/scenes/robot-eye-v6.xml --shadows --environment moving --frames 90 --fps 15 --threads 4
```

当前选择的 V6 已含展示用 `deployment` 元数据；重训得到的权重数值和文件哈希不保证与本轮相同。要比较记录中的结果，应读取保留的原始指标与冻结文件。

回看旧 V5 可显式启动：

```powershell
.\.venv\Scripts\python.exe sim/server.py --port 8766 --checkpoint outputs/perception/run-v5-supported/model.pt
```

## 仍待解决

夹爪薄结构、近端边界和少量外物仍会出错。当前探索范围较小；手动滑块可超出训练范围，84.43% 不代表完整工作空间。未知工具、真实遮挡、相机移动、反馈噪声/延迟、影子区分与学习时间记忆均未验证。下一步研究重点应包括让反馈在推理中产生可测量收益，而非只继续提升同一视角的外观分割分数。
