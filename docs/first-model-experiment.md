# 第一版身体感知模型：实现与 MuJoCo 验证

实验日期：2026-10-24。已完成自动采集、图像与反馈监督、PyTorch 模型训练、独立姿态测试和连续运动推理。

本文保留 V3 阶段的实验记录。后续已完成 [V4 优化与新姿态配对测试](model-optimization.md)，继续使用本轮 seed 20261026 排错后，另以 seed 20261027 作为新的最终测试。

冻结后的模型在 **32 个新随机姿态**上取得 **53.71% 的可见身体 IoU**；运动后保持的姿态为 **54.69%**。模型能找出大致活动臂区域，但边界偏粗，靠近底座的活动部分及夹爪细节仍有遗漏。这是受控仿真中的第一版结果，没有完成真实 SO-ARM 验证，也不能据此宣称模型已经建立一般性的身体概念。

## 1. 实际交付

| 内容 | 位置 |
|---|---|
| 成组动作探测与实际反馈采集 | `perception/collect.py` |
| RGB 光流、正反向响应检查 | `perception/flow.py` |
| 训练图像背景参考与部分标签 | `perception/background.py`、`perception/data.py` |
| 神经网络与损失 | `perception/model.py` |
| 分阶段训练与检查点 | `perception/train.py` |
| 真值隔离的评价与 HTML 报告 | `perception/evaluate.py` |
| 连续运动和停止推理演示 | `perception/demo.py` |
| 当前反馈输入扰动诊断 | `perception/ablate_feedback.py` |
| 一条命令完成采集、训练、评价 | `perception/run.py` |

已生成的本地文件保留在 Git 忽略的 `outputs/perception/` 下：

- 权重：`run-v3/model.pt`，127,423 个参数，约 0.53 MB。
- 最终独立测试：`confirmation-report/report.html`、`metrics.json`、`examples.png`。
- 运动演示：`confirmation-report/motion-demo.gif`，附逐帧反馈与指标 `motion-demo.json`。
- 输入扰动诊断：`confirmation-report/feedback-ablation.json`。
- 失败实验：`final-report/` 是早期 V2 的**诊断结果**，目录名称不代表最终结果。

最终权重 SHA-256：

```text
edb641906cd3496cefc2dd77dcf59ef86faf2259022e139f8c5dd64309c2d19f
```

## 2. 模型到底学什么

网络输入是当前 `192×144` RGB 图像与实际关节位置 `q`，输出身体概率图和各反馈通道的二维像素运动响应 `J`。U-Net 的通道宽度为 8/16/32/64，反馈经过 MLP 后调节中间特征。没有输入未来图像、仿真运动学雅可比、CAD 或部件身份。

```text
当前 RGB ──图像编码器──┐
                      ├──解码器──身体概率图
实际位置 q ──反馈编码器─┘         └─像素响应 J
                                     │
实际变化 Δq ──────────────────────────┴─预测视觉位移 JΔq
                                              │
两帧 RGB 估计的光流 ──────────────────────────监督
```

`Δq` 来自实际物理步进前后的关节位置差，除以对应通道的探测尺度；它在网络**外部**与 `J` 相乘。动作指令单独保存，不能代替反馈。七个通道是六个旋转关节及一个夹爪执行器，旋转位置单位为弧度，夹爪行程为米。

运动训练使用可靠光流上的 Smooth L1，并分别平均运动区域和静止区域。身体分割使用正、负标签分别平均的 BCE；未知像素完全忽略。分割阶段保留 `0.1 × 运动损失`。身体概率图不用于遮掉运动损失，避免模型通过“所有像素都不是身体”逃避监督。

阈值固定为身体概率 `0.5`；运动区域由 RGB 光流幅度大于 `0.35` 像素定义。检查点只按观测数据上的验证损失选取，没有用真值 IoU 选模型或调阈值。

## 3. 身体标签从哪里来

实际实现对原讨论方案作了一个明确调整：**部分标签由局部线性响应教师生成，没有让尚未成熟的神经网络自己给自己作证。**

每个初始姿态从相同的完整仿真状态出发，对七个执行器分别做正、负两个方向的小幅探测，共 14 次。由当前图像和探测后图像估计光流，并检查前后向一致性、边界和光度误差。只用七个正向探测拟合像素局部线性响应，再用七个负向探测检验预测是否成立。神经网络的运动训练也使用正向探测，验证运动损失使用另一些姿态的负向探测。

正标签要求正反方向上都出现可靠位移，线性响应能解释另一方向的变化，且按光流对齐确实比不对齐更能解释图像变化。后一个条件用于排除光流平滑把相邻背景“拖动”的现象。负标签先来自可靠静止的有纹理区域，其余保持未知。

第一轮分割暴露了无纹理天空缺少负标签的问题，因此加入另一类观测证据：仅用 **64 个训练初始图像**拟合每像素背景中位数及重复出现比例；当前像素必须与该参考相符、在全部正反探测中外观不变，且避开已有正标签邻域，才补为负标签。验证和测试只能应用冻结参考，不能重新拟合。

这增加了“固定相机、光照和背景，探索姿态使背景反复可见”的假设。持续占据同一位置、外观变化不明显的身体表面仍可能被错标为背景，因此“观测不变”不是普遍的非身体判据。背景参考仅用于训练标签生成，实际推理只需当前 RGB 和 `q`，不运行背景减除。

训练标签平均覆盖：正标签 **1.05%**、负标签 **95.87%**、未知 **3.07%**。这是整幅图的比例，不是身体召回率。

## 4. 数据隔离与实验过程

| 数据用途 | 种子 | 初始姿态组数 | 是否影响最终模型 |
|---|---:|---:|---|
| 训练 | 20261024 的 train 子随机流 | 64 | 优化权重、拟合背景参考 |
| 验证 | 20261024 的 val 子随机流 | 16 | 仅按观测损失选权重 |
| 初始预留测试 | 20261024 的 test 子随机流 | 16 | 不参与训练和权重选择 |
| V2 失败诊断 | 20261025 的 test 子随机流 | 32 | 查看失败后修改了方法，故不再视作最终测试 |
| 最终独立测试 | 20261026 的 test 子随机流 | 32 | 配置及权重冻结后评价，不再据此调参 |

同一姿态下的探测共享起点，不能把 14 次探测当作 14 个独立样本。采集器从保存的完整物理状态恢复后执行探测，保存实际 `dq`；最终测试全部探测矩阵的秩为 7。相机与外观在各集合间保持相同，所以这里检验的是相同分布内的新姿态，并非新环境泛化。

观测和真值物理分开保存：

```text
data/
  metadata.json
  observations/{train,val,test}/*.npz  RGB、实际 q/dq、指令及停止观测
  evaluation/{train,val,test}/*.npz    可见身体掩膜，仅用于评价
```

训练和标签生成只读取 `observations/` 的白名单字段。评价器先完成预测，再读取对应真值。仿真几何分组用于生成评价掩膜：固定底座按环境，活动臂与夹爪按身体；这些分组不进入学习程序。

使用 CPU PyTorch 2.14.1+cpu、四线程、batch size 4。首先训练运动响应 20 轮，再从选出的运动权重训练分割 30 轮。V3 分割阶段沿用 V1 的纯运动检查点，未继承失败的分割权重。运动阶段连同当时的缓存准备约 209 秒；V3 分割阶段连同缓存复用约 232 秒。总探索还包括失败轮次、重新采集和评价，不能将这两个时间之和视为整个研发耗时。

原始 `run-v1/history.json` 包含随后中止的一轮分割日志，但保存的 `motion.pt` 早于该轮。实际复用文件 `motion-normalized.pt` 为同一运动权重补齐了归一化元数据。最终训练记录保留初始化文件哈希和来源历史。

## 5. 实际测量结果

以下为 32 个最终测试姿态逐组平均，非只选好的示例：

| 评价图像 | IoU | 精确率 | 召回率 |
|---|---:|---:|---:|
| 新初始姿态 | 53.71% | 65.16% | 75.42% |
| 运动后保持的姿态 | 54.69% | 65.41% | 77.00% |

初始姿态 IoU 最低 **48.37%**、中位 **53.85%**、最高 **56.85%**。报告自动选取最低、中位和最高样本，展示 RGB、预测及评价真值。预测形状偏粗，部分活动臂被漏掉，夹爪也没有精确分开。

停止图像来自正向肩关节探测后的实际状态，继续保持目标 0.5 秒后重新采集。它与初始姿态不同，因此两个分数之差不是纯粹的“停止效应”。连续 6 秒演示另包含 60 帧运动与 30 帧保持：运动平均 IoU **54.47%**；排除最初 0.3 秒稳定过程后的保持帧平均 **54.97%**。这仅是一条确定轨迹，不能替代独立样本测试。模型逐帧推理，没有学习时间记忆。

负向探测的视觉位移误差 EPE，单位为图像像素，越低越好：

| 方法 | 运动区域 | 静止区域 | 全部有效区域 |
|---|---:|---:|---:|
| 学习响应 + 正确实际 Δq | 0.8771 | 0.3210 | 0.3415 |
| 全部预测不动 | 2.3652 | 0.0045 | 0.0916 |
| 学习响应 + 配对打乱的 Δq | 2.6437 | 0.3585 | 0.4429 |

正确反馈对应的运动预测明显好于不动和错误配对，说明模型学到了可用于解释运动的响应。但它在静止背景上弱于零运动基线，导致整图误差也更高；不能只报告运动区域的优势。该目标是 **RGB 估计光流**，不是仿真提供的真实物理光流。打乱 `Δq` 的结果不能单独证明网络充分利用了当前 `q`，两者要分开检验。

对同一冻结模型再做当前反馈输入扰动，保持 RGB 和用于运动预测的实际 `Δq` 不变：

| 当前位置输入 | 身体 IoU | 运动区域 EPE |
|---|---:|---:|
| 正确 `q` | 53.7113% | 0.8771 |
| 跨姿态打乱 `q` | 53.7214% | 0.8930 |
| 固定为训练归一化中心 | 53.7122% | 0.8790 |

打乱 `q` 只改变整幅图 **0.0392%** 的预测像素，约占两种预测身体并集的 **1.36%**。因此，**当前分割对即时位置反馈的依赖很弱，主要使用已学到的视觉信息**。这与反馈变化参与运动监督并不矛盾，但不能据此宣称已实现充分的视觉与本体反馈融合。这里是一次推理输入扰动，不是重新训练的图像单模态对照，也不能说明反馈在其他任务或场景中没有价值。

仅在最终评价时对部分标签检查：正标签精确率 **91.44%**；负标签中的身体像素比例 **0.38%**。由于背景占绝大多数，后一个小数字仍对应 **3,211 个身体像素，占真实身体像素约 14.7%**。这些被错误排除的表面会限制完整身体恢复。上述真值质量不用于产生标签或调阈值。

## 6. 保留的失败与尚未证明的部分

V2 曾取得很低的部分标签验证损失，但在 seed 20261025 的诊断数据上 IoU 只有 **5.88%**：大面积天空处于未知标签区域，模型把它们判为身体。这个失败说明“已知区域上的损失低”不代表整幅图的身体分割正确。增加训练图像的背景证据后冻结 V3，再换 seed 20261026 做最终评价。两个轮次的测试姿态不同，不能把两组均值当作严格配对的消融实验。

当前结论受以下边界限制：

- 通用六关节加夹爪，共七个反馈通道；真实 SO-ARM100/101 是五关节加夹爪，尚未对齐机构和标定。
- 输入为理想仿真位置，无传感器噪声、延迟和漏读；不等于已验证真实舵机反馈。
- 训练姿态围绕初始姿态作有限范围探索，测试来自相同范围；未验证大范围姿态、相机移动、材质变化和新背景。
- 无投影阴影，未验证影子区分、工具接入和身体边界扩展；这些仍后置。
- 部分标签依赖人工设定的通用光流和置信规则；不是零先验、端到端发现身体。
- 当前 `q` 的推理输入扰动只检验已训练模型的依赖程度，不等价于重新训练的图像单模态消融。虽然训练入口提供 `--image-only`，本次尚未完成配对的重新训练对照。
- 停止时保留预测不等于具有长期身体记忆，推理仍可能主要利用已学到的视觉外观。

## 7. 复现

在项目根目录，先按 README 建立 `.venv` 并安装 `requirements-model.txt`。需要可用的 MuJoCo OpenGL 渲染。数据、缓存和输出目录均使用新名称以保护已有实验。

最短流程（新实验，不保证与已有权重逐字节一致）：

```powershell
.\.venv\Scripts\python.exe -m perception.run --output outputs/perception/new-run --train 64 --val 16 --test 16 --motion-epochs 20 --segmentation-epochs 30 --threads 4
```

复现本次分阶段训练及独立最终测试的步骤如下。第二阶段重新初始化数据打乱随机流，与本次 V3 的启动方式一致：

```powershell
.\.venv\Scripts\python.exe -m perception.collect --output outputs/perception/reproduce-data --train 64 --val 16 --test 16 --seed 20261024
.\.venv\Scripts\python.exe -m perception.train --data outputs/perception/reproduce-data --cache outputs/perception/reproduce-cache --output outputs/perception/reproduce-motion --motion-epochs 20 --segmentation-epochs 0 --threads 4 --seed 20261024
.\.venv\Scripts\python.exe -m perception.train --data outputs/perception/reproduce-data --cache outputs/perception/reproduce-cache --output outputs/perception/reproduce-body --motion-epochs 0 --segmentation-epochs 30 --threads 4 --seed 20261024 --init-checkpoint outputs/perception/reproduce-motion/motion.pt
.\.venv\Scripts\python.exe -m perception.collect --output outputs/perception/reproduce-test --train 0 --val 0 --test 32 --seed 20261026
.\.venv\Scripts\python.exe -c "from perception.data import prepare_cache; prepare_cache('outputs/perception/reproduce-test', 'outputs/perception/reproduce-test-cache', splits=('test',), background_reference='outputs/perception/reproduce-cache/background-reference.npz')"
.\.venv\Scripts\python.exe -m perception.evaluate --data outputs/perception/reproduce-test --cache outputs/perception/reproduce-test-cache --checkpoint outputs/perception/reproduce-body/model.pt --output outputs/perception/reproduce-report --threads 4
.\.venv\Scripts\python.exe -m perception.demo --checkpoint outputs/perception/reproduce-body/model.pt --output outputs/perception/reproduce-report/motion-demo.gif --frames 90 --fps 15 --threads 4
.\.venv\Scripts\python.exe -m perception.ablate_feedback --data outputs/perception/reproduce-test --cache outputs/perception/reproduce-test-cache --checkpoint outputs/perception/reproduce-body/model.pt --output outputs/perception/reproduce-report/feedback-ablation.json --threads 4
```

若只查看已训练模型的连续推理：

```powershell
.\.venv\Scripts\python.exe -m perception.demo --checkpoint outputs/perception/run-v3/model.pt --output outputs/perception/new-demo.gif
```

测试入口：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

已执行光流、反馈配对、未知标签梯度、加载器隔离真值、背景参考和评价指标测试，也运行了更新后的一条命令采集—训练—评价检查（2 个训练姿态、1 个验证姿态、1 个测试姿态，各训练一轮）。小检查只证明流程能运行，其指标不用于本实验结论。
