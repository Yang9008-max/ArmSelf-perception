# V5：环境干扰与过程经验

2026-10-24。本轮已实际采集、训练、比较及运行连续 MuJoCo 推理。它是有限场景中的改进，尚未解决可靠的身体归属。过程中的失败也保留，见 [实验经验记录](learning-log.md)。

**场景版本说明：** 本报告全部训练、评价与已保存动画使用原地面场景。后续按用户要求将默认 `sim/scene.xml` 改为桌面场景，尚未在新场景重新训练或测量准确率。原 XML 已原样归档为 `sim/scenes/floor-v1.xml`，SHA-256 为 `a15c6f4d4d6613efdc2dca51a1c289e9f655e974993fdd167b575657b6d3c133`；归档原文及生成干扰场景的哈希均与 V5 数据元信息一致。复现本报告的数据采集须显式指定该归档路径。

## 最终结果

权重与规则冻结后另采 **50 个姿态**，随机种子 `20261102`，五类环境各 10 个。三个模型使用完全相同的 RGB/反馈输入及 0.5 阈值。身体 IoU 是逐姿态平均；杂物误判率是“预测为身体的可见杂物像素 / 全部可见杂物像素”，不是整幅背景误报率。

| 指标 | 原 V4 | 只加环境数据 + 候选约束 | V5：静止对照单独加权 + 同一约束 |
|---|---:|---:|---:|
| 初始姿态身体 IoU ↑ | 62.04% | 65.86% | **66.84%** |
| 运动后保持姿态 IoU ↑ | 62.49% | 66.36% | **68.14%** |
| 身体精确率 ↑ | 68.29% | 73.11% | **74.98%** |
| 身体召回率 ↑ | 87.47% | 87.37% | 86.21% |
| 初始帧杂物误判率 ↓ | 97.81% | 95.60% | **53.01%** |
| q 保持控制帧身体 IoU ↑，含全部环境 | 62.28% | 66.08% | **67.32%** |
| q 保持控制帧杂物误判率 ↓，含全部环境 | 96.35% | 96.04% | **49.16%** |
| 仅独立运动组：q 保持时杂物误判率 ↓ | 94.18% | 94.07% | **41.74%** |

相对 V4，48/50 个初始姿态 IoU 改善，平均增加 4.80 个百分点。相同约束下，相比仅增加环境数据，静止对照方案的 IoU 增益只有 **0.98 个百分点**；其主要收益是减少杂物误判。身体召回略降，不能省略这一代价。这里没有声称每类环境都优于只增加数据：后者在干净和光照组的 IoU 更高。

| 环境，初始帧 | V4 身体 IoU | V5 身体 IoU | V4 杂物误判 | V5 杂物误判 |
|---|---:|---:|---:|---:|
| 干净 | 70.25% | 71.46% | 无杂物 | 无杂物 |
| 同色静态杂物 | 59.17% | 68.19% | 97.93% | 33.92% |
| 独立运动杂物 | 58.41% | 65.19% | 96.74% | 51.70% |
| 前景遮挡配置 | 54.86% | 60.25% | 98.51% | **67.25%** |
| 光照及地面外观变化 | 67.51% | 69.09% | 无杂物 | 无杂物 |

前景组仍是明显失败区域。评价额外隐藏杂物渲染身体掩膜，确认该组 **8/10 个姿态存在实际遮挡，平均挡住 3.89% 的身体像素**，因此这里只验证了轻度遮挡；不能把“放了前景物体”写成已经通过严重遮挡测试。这些额外掩膜也只存于 `evaluation/`。

## 做了什么

新增 `perception/environment.py`，在固定相机中加入三个与机械臂颜色相近的胶囊、方块和球体。它们是无接触的可视干扰物，轨迹独立于机器人动作。静态组不动，运动/前景组每次探测重新随机方向与速度，范围 0.16–0.40 m/s。光照组在组间改变光强和地面颜色，组内不变；本轮继续关闭阴影。

训练数据为 80 组，验证 20 组，开发评价 20 组，种子 `20261101`。开发评价放在 `test/` 目录仅沿用文件格式，本轮始终将它视为开发数据。最终 50 组在规则冻结后重新采集，未据此继续调参。各组的环境类型平衡，形状类别及参数范围在训练与最终评价间相同；新种子不等于测试了未知物体类别。

每个姿态先物理稳定 0.8 秒，再从相同完整仿真状态独立执行 14 次正反向关节/夹爪探测，每次 0.35 秒，记录实际 q 与实际 Δq。执行顺序随机化，存储顺序仍按通道排列。额外执行两次保持原目标的对照，让外物继续运动。最终数据中，保持对照的实际关节变化最大约 `1.32e-12`（旋转通道弧度、夹爪通道米）；这是理想仿真的结果，不代表真实舵机也能如此稳定。

新增负证据只依据以下观测：

1. 两次对照中，每个实际关节变化都小于其探测尺度的 2.5%。
2. 两次 RGB 光流均有有效对应，幅度大于 0.35 像素，且对齐比原坐标更能解释 RGB 变化。
3. 两次都支持的像素才作为负种子；若与原动作正种子冲突，退回未知。

其余监督继续使用正向探测拟合、负向探测检查的图像响应种子，训练 RGB 拟合的冻结背景参考，以及逐图像 GrabCut 补全。真值、杂物身份、环境类型、深度、CAD 均不进入这些步骤。模型推理时不运行光流或 GrabCut。

## 为什么不是第一轮就成功

开发集上的实际尝试按顺序保留：

| 尝试 | 身体 IoU | 杂物误判率 | 观察 |
|---|---:|---:|---|
| 原 V4 | 61.83% | 98.49% | 相似外观几乎都被当作身体 |
| 只加环境数据 | 67.71% | 97.90% | 轮廓改善，外物归属几乎没改善 |
| 加对照，按普通负标签训练 | 67.68% | 89.82% | 可靠种子太少，在背景负标签中被淹没 |
| 对照负种子单独加权 | 57.64% | 54.98% | 外物减少，但轮廓膨胀，整体变差 |
| 加权分支与 V4 候选范围取交集 | 66.36% | 54.12% | 限制新增误报，仍有大量外物错误 |

运动/前景训练组每个姿态平均只有 22.5 / 35.44 个新增负种子，而普通背景负标签有两万多个。只把这些种子塞进同一个负类均值，并不能让模型重视它们。最终分割损失为原来的正负平衡 BCE，加 `0.5 ×` 可靠对照负种子的独立平均 BCE。

本轮将分割阶段的运动辅助损失权重设为 0：机器人 Δq 无法解释独立外物运动，不应强迫网络拟合整幅光流。原运动分支另行冻结保留，但没有因此证明它已适应了杂物场景。

单独加权又产生新的问题：开发集身体召回升到 92.53%，IoU 却下降，说明它增加了其他误报。最终使用冻结 V4 的身体图像预测作为候选上限，新分支只能排除，不能向外添加。具体取两个 logit 的最小值，在统一阈值下等价于身体掩膜取交集。对照比较中的“只加数据”模型也施加同样约束，使用同一套 V4 权重。

最终组合包含三套网络：新身体分支、冻结 V4 身体候选分支、冻结原运动分支，共 **382,269 参数**，检查点约 **1.59 MB**。代价是额外计算，以及无法补回 V4 已漏掉的身体像素；它是组合模型，不能当成单个网络已学会全部机制。

三个训练均从同一 V4 身体权重初始化，种子 `20261101`，CPU、4 线程、批量 4、25 轮。依据不含真值的验证损失，分别选中第 23、17、10 轮。加权版后期训练损失继续下降、验证损失变差，实际曲线见本地报告目录的 `training-curves.png`；不同损失的数值不能直接横向比较。

## 尚未解决的事

最终测试打乱当前 q 后，IoU 从 66.84% 变为 66.87%，变化像素约占预测身体并集的 1.39%。这依然没有证明即时反馈融合有效。反馈主要参与动作监督及显式 Δq 运动响应，当前分割仍主要依赖视觉。

最终弱标签正类精确率为 86.49%；虽然补全覆盖全部像素，仍有约 16.4% 的真实身体像素被标成背景。若以庞大负标签集合为分母，这一污染率仅约 0.41%，会掩盖身体上的损失。评价需要关心身体自身，而不能只看整图误差或标签覆盖率。

独立连续演示另使用未在采集探测中出现的有界正弦杂物轨迹：先 60 帧机械臂运动，再保持非初始姿态。除去初始 0.3 秒保持过渡，26 帧的身体平均 IoU 为 **62.79%**，杂物误判 **65.79%**，低于成组探测评价。它提醒我们，短时控制对照有效不等于长期连续感知已经稳定。模型逐帧推理，没有时间记忆。

本轮仍为通用六旋转关节加夹爪的七通道机构，未对齐 SO-ARM 的五关节加夹爪；无真实反馈噪声/延迟、相机移动、接触干扰、严重遮挡或未知工具验证。下一步值得检验的是跨时间的图像—实际反馈对应、持续身体状态，以及更强遮挡下的可见区域变化，而不是继续只堆外观样本。本轮最终数据不再用于参数搜索。

## 本地结果与复现入口

- 最终权重：`outputs/perception/run-v5-supported/model.pt`，SHA-256 `19a168ba57ca6fbe528d558d5ff429f0c992db0641dd8897b979d73816fc017c`。
- 三模型对照：`outputs/perception/v5-final-report/environment-comparison.html`，内嵌每类最低 IoU 示例。
- 完整指标：同目录 `metrics.json`、`report.html`、`feedback-ablation.json`。
- 连续推理：同目录 `environment-demo.gif` / `.json`，含逐帧实际 q、qvel 与杂物位置。
- 过程曲线：同目录 `training-curves.png`。
- 冻结记录：`outputs/perception/v5-final-freeze.json`；保留冻结时的模型与源代码哈希。
- 开发失败结果：`v5-development`、`v5-weighted-development`；最终比较未覆盖这些文件。

生成数据与权重不纳入 Git，代码、经验与实验说明保留在项目中。以下是本轮核心调用；已有输出会被保护，复现实验请换新输出目录。V4 初始化及原运动检查点来源见 [上一轮报告](model-optimization.md)。

```powershell
# 数据：这里的 test 20 是开发评价
.\.venv\Scripts\python.exe -m perception.collect --scene sim/scenes/floor-v1.xml --output outputs/perception/data-v5 --train 80 --val 20 --test 20 --environment mixed --seed 20261101

# 先运行不使用对照的分支
.\.venv\Scripts\python.exe -m perception.train --data outputs/perception/data-v5 --cache outputs/perception/cache-v5-base --output outputs/perception/run-v5-ablation-body --motion-epochs 0 --segmentation-epochs 25 --threads 4 --seed 20261101 --init-checkpoint outputs/perception/run-v4-body/model.pt --refine-labels --motion-weight 0

# 使用对照、单独加权；复用图像光流而不复用最终标签
.\.venv\Scripts\python.exe -m perception.train --data outputs/perception/data-v5 --cache outputs/perception/cache-v5-null --reuse-cache outputs/perception/cache-v5-base --output outputs/perception/run-v5-weighted-body --motion-epochs 0 --segmentation-epochs 25 --threads 4 --seed 20261101 --init-checkpoint outputs/perception/run-v4-body/model.pt --refine-labels --use-null-evidence --motion-weight 0 --control-weight 0.5

# 最终组合；无对照分支同样使用 --support-checkpoint 以作比较
.\.venv\Scripts\python.exe -m perception.compose --body-checkpoint outputs/perception/run-v5-weighted-body/model.pt --motion-checkpoint outputs/perception/run-v1/motion-normalized.pt --support-checkpoint outputs/perception/run-v4-body/model.pt --output outputs/perception/run-v5-supported/model.pt

# 冻结后另采最终测试，只评价、不继续调参
.\.venv\Scripts\python.exe -m perception.collect --scene sim/scenes/floor-v1.xml --output outputs/perception/data-v5-final --train 0 --val 0 --test 50 --environment mixed --seed 20261102
.\.venv\Scripts\python.exe -m perception.evaluate --data outputs/perception/data-v5-final --checkpoint outputs/perception/run-v5-supported/model.pt --output outputs/perception/new-v5-report --threads 4
# 连续动画现有入口使用当前默认场景；原地面场景动画请查看已保存的 environment-demo.gif。
# 如运行下面命令，结果属于新桌面场景，不能当作复现本报告的连续指标。
.\.venv\Scripts\python.exe -m perception.demo --checkpoint outputs/perception/run-v5-supported/model.pt --output outputs/perception/new-v5-report/tabletop-environment-demo.gif --environment moving --frames 90 --fps 15 --threads 4
```

基础评价无需光流缓存。若计算最终弱标签/运动指标，必须通过 `prepare_cache(..., splits=('test',), background_reference=训练缓存中的背景参考)` 创建测试缓存，禁止在最终测试图像上重新拟合背景。三模型报告由 `python -m perception.environment_report --baseline <V4 metrics.json> --exposure <仅数据分支 metrics.json> --controls <对照分支 metrics.json> --output <目录>` 生成。

验证：`python -m unittest discover -s tests` 共 **34 项通过**，包括实际保持反馈、无接触环境干扰、观测/真值隔离、静止对照证据、稀疏负证据梯度、冻结运动响应及候选范围约束。另实际运行了全部训练、最终评价与动态演示。
