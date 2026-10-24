# ArmSelf-perception

让模型结合**摄像头图像与舵机实际反馈**，通过自己的活动感知机械臂这一物理实体：它在哪里、如何随反馈变化，以及暂时停止或看不见时如何继续预测它。

这是身体自感知研究原型。先研究感知，抓取后置；活动手臂与夹爪属于身体，始终静止的固定底座按环境处理。完整身体形态、部件关系和学习得到的持续身份**尚未实现**。

**换平台入口：[迁移与交接说明](docs/migration.md) · [模型、数据和报告下载](https://github.com/Yang9008-max/ArmSelf-perception/releases/tag/handoff-2026-10-24) · [实验经验日志](docs/learning-log.md)**

## 当前做到哪里

| 阶段 | 实际实现 | 不能据此宣称的能力 |
|---|---|---|
| E1 / E2 实体空间原型，当前主线 | 双目 RGB 提供空间观测；用实际关节反馈发现可预测的活动特征；小型神经网络学习反馈到三维位置的映射；相机中断时继续更新位置 | 完整形状、实体体积、部件结构、学出来的物体恒存 |
| E3 自主探索，当前实验 | 根据已观察的图像与反馈更新在线概率回归模型，再选择下一步动作；与随机、预设动作进行同预算比较 | 强化学习策略、端到端动作学习、从零发现整个身体 |
| V1–V6 图像分割，历史基线 | 图像与实测反馈生成弱监督，训练身体区域与运动响应；V6 接在实时页面 | 仅凭绿色区域或较高 IoU 就证明模型理解实体存在 |

E2 的位置网络是 **7 → 64 → 64 → 123** 的 MLP，约 **1.27 万参数**，对应 41 个稀疏特征；图像跟踪、双目匹配、特征筛选与实体状态规则仍由程序实现。视觉产生训练目标，网络推理位置时主要输入实际 q，不能说已经实现了完整的多模态时序感知。

E3 使用在线二阶概率回归和显式信息评分，不使用强化学习奖励或策略训练。它从 E2 继承参考图像和 41 个特征身份，不加载 E2 的三维网络权重；前 16 帧共同随机起步，之后再按经验选动作。

学习输入与仿真真值分开保存：`observations/` 放 RGB 和实际反馈，`evaluation/` 放身体身份、深度等评价信息。E2 开发阶段确实使用验证真值诊断了双目误差，最终模型冻结后才采新的测试轨迹；详细过程见实验记录。

## 已有数据图与结果

E2 在冻结后新采的同一条 162 帧轨迹上，与 E1 使用同特征、同像素的身体观测配对比较：

| 阶段 | E1 几何 RMSE | E2 几何 RMSE |
|---|---:|---:|
| 新动作 | 14.30 mm | **9.81 mm** |
| 停止保持 | 15.17 mm | **10.90 mm** |
| 相机中断 | 15.11 mm | **10.68 mm** |

![E2 配对误差、训练曲线、相机中断和覆盖](docs/assets/e2-comparison.png)

这些是被验证的稀疏身体点误差，**不是整个机械臂重建精度**。E2 每帧可验证身体点由 26.0 降至 23.2，身体点纯度约 65.1%；仍有背景错配，全部有效候选对双目观测的误差为 25.43 mm。一个 821 mm 的错误对应也保留在报告中，未因测试结果而删除。[E2 方法、原始指标和失败分析](docs/entity-stereo-refinement.md)。

E3 三种策略各跑三个采集种子，每次 112 帧，在另一条冻结后新轨迹上评价共同身体点：

| 观察预算 | 模型选动作 | 随机动作 | 预设动作 |
|---|---:|---:|---:|
| 16 帧 | 39.89 mm | 39.89 mm | 39.89 mm |
| 32 帧 | 27.87 mm | 29.74 mm | 26.78 mm |
| 64 帧 | **14.76 mm** | 20.98 mm | 22.58 mm |
| 112 帧 | **11.55 mm** | 11.71 mm | 16.81 mm |

![E3 同预算探索比较与实际反馈](docs/assets/e3-exploration.png)

最终相对随机只改善约 0.16 mm，三个种子中有一次更差，尚无稳定优势。自主策略可见特征更少，运动路径也更长；本次匹配了图像数与仿真时间，没有匹配能耗或运动总量。E3 的预测器与评价轨迹不同，数值不能直接与 E2 排名。[E3 自主探索、决策过程与局限](docs/active-exploration.md)。

## 安装与恢复

推荐 **Python 3.13**。本次实际运行环境为 Windows x64 / Python 3.13.5，主要依赖版本锁定在 `requirements*.txt`；训练和测试使用 CPU。Linux/macOS 尚未在本次交接中实机验证；Linux 无显示环境配置见[迁移说明](docs/migration.md#linux-无显示环境)。

```bash
git clone https://github.com/Yang9008-max/ArmSelf-perception.git
cd ArmSelf-perception
python -m venv .venv
```

激活环境，Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS：

```bash
source .venv/bin/activate
```

然后安装。Windows/Linux 如需 CPU 版 PyTorch，先执行第一条；macOS 省略第一条，直接安装 requirements：

```bash
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-model.txt
```

**只克隆 Git 仓库不会下载 `outputs/`。** 接着选择恢复资料；脚本只依赖 Python 标准库，会校验归档及每个文件的 SHA-256，并拒绝覆盖内容不同的本地实验。

```bash
# 小型预览包：权重、图表、回放、决策、元数据；没有原始 RGB 和评价数组
python scripts/artifacts.py fetch --bundle runtime

# 换平台继续研究：完整当前实验 + 历史原始实验资料
python scripts/artifacts.py fetch --bundle all
python scripts/artifacts.py verify --bundle all
```

压缩大小：runtime 约 **30.5 MiB**，current 约 **801.5 MiB**，legacy 约 **248.9 MiB**。也可只下载当前 E1/E2/E3：`python scripts/artifacts.py fetch --bundle current`。离线恢复、所含资料与缓存重建方法见[迁移说明](docs/migration.md)。已经安装的相同文件会跳过，先装 runtime 再装 all 也可以。

## 先看已有结果

恢复 runtime 或 current 后，从仓库根目录运行：

```bash
python -m http.server 8767 --bind 127.0.0.1 --directory outputs/embodiment/entity-v2/final-e2
```

打开 [E2 数据图与空间预测回放](http://127.0.0.1:8767/)；[E3 三种探索策略的同步相机回放](http://127.0.0.1:8767/exploration/)。这些页面展示保存的实验，不是在线 E3 推理服务。图表旁保留 CSV、JSON 与失败示例；同目录的 `metrics.json` 给出完整逐帧指标。

## 运行 MuJoCo 与独立摄像头

```bash
python sim/server.py --port 8765
```

打开 [桌面仿真与 V6 实时预测](http://127.0.0.1:8765/) 和 [独立原始摄像头](http://127.0.0.1:8765/camera)。Windows 也可双击 `start.cmd`；自动启动脚本只安装基础仿真依赖，模型依赖仍需按上面安装。

实时页面目前加载 **V6 分割基线** `outputs/perception/v6-eye/model/model.pt`，E2/E3 未替换这个在线模型。页面里的“自动探索”是早期预设关节探测；E3 的模型选动作由 `embodiment.explore_run` 单独执行。页面不会把预设动作显示成已学策略。缺少权重时原始摄像头仍能运行，页面会显示模型不可用。

场景是桌上机械臂、夹爪和球/方块/长条物体。相机固定在机械臂肩部侧上方，世界位置 `(-0.44, -0.16, 1.56)` m，距桌面约 0.80 m，垂直视野 68°；E1/E2/E3 增加一台平行相机，双目基线 6 cm，RGB 为 640 × 480。当前演示是六个旋转关节加一通道夹爪，**尚未按 SO-ARM100/101 的五关节加夹爪结构对齐**；反馈也是理想仿真 q，未模拟真实舵机噪声与延迟。

![桌面机械臂与物体](docs/assets/tabletop-scene.png)

## 继续训练和探索

以下命令均在仓库根目录执行，并使用新目录，避免覆盖冻结实验。恢复 current 后，仅重跑 E3：

```bash
python -m embodiment.explore_run --output outputs/embodiment/repro-active
python -m embodiment.collect --output outputs/embodiment/repro-active/final-data --evaluation-only --seed 202610244
python -m embodiment.explore_report evaluate --runs outputs/embodiment/repro-active --data outputs/embodiment/repro-active/final-data
python -m embodiment.explore_report build --runs outputs/embodiment/repro-active
```

默认是 active/random/preset × 410/411/412，每次 112 帧，共 1,008 次训练观测；模型全部冻结后才采最终评价集。它使用恢复的 E2 参考图像与特征先验。若要从新的图像采集重新建立 E2：

```bash
python -m embodiment.collect --output outputs/embodiment/repro-entity/data --seed 20261210
python -m embodiment.experiment fit --data outputs/embodiment/repro-entity/data --output outputs/embodiment/repro-entity/model --tracker-mode refined11
python -m embodiment.experiment controls --data outputs/embodiment/repro-entity/data --output outputs/embodiment/repro-entity/model
python -m embodiment.collect --output outputs/embodiment/repro-entity/final-data --evaluation-only --seed 202610241
python -m embodiment.experiment evaluate --data outputs/embodiment/repro-entity/final-data --output outputs/embodiment/repro-entity/model --evaluation-output outputs/embodiment/repro-entity/review
python -m embodiment.review --data outputs/embodiment/repro-entity/final-data --output outputs/embodiment/repro-entity/review --version E2
```

再运行 E3 时可指定 `--prior outputs/embodiment/repro-entity/model --calibration-data outputs/embodiment/repro-entity/data`。随机种子、浮点运算和渲染环境会影响新实验；已发布指标对应归档权重与原始数据，重训不保证产生完全相同的文件哈希。历史 V6 的完整训练配方见[机器人眼位训练](docs/robot-eye-training.md)。

## 检查与源码导航

```bash
python -m unittest discover -s tests
```

在线检查需另一个终端保持仿真服务运行：

```bash
python sim/smoke_test.py --url http://127.0.0.1:8765 --require-model
```

| 路径 | 内容 |
|---|---|
| `embodiment/collect.py`、`vision.py` | 双目采集、实际 q、RGB 对应和三维观测 |
| `embodiment/model.py`、`experiment.py` | 候选发现、神经位置模型、冻结与独立评价 |
| `embodiment/exploration.py`、`explore_run.py` | 在线信念、动作评分、MuJoCo 探索和决策日志 |
| `embodiment/compare.py`、`review.py`、`explore_report.py` | 配对指标、数据图和可交互回放 |
| `perception/` | 历史视觉分割、光流弱监督与反馈诊断 |
| `sim/` | 场景 XML、桌面试验台、独立摄像头页面 |
| `scripts/artifacts.py`、`artifacts/manifest.json` | 跨平台恢复、逐文件哈希和 Release 清单 |
| `tests/` | 数据边界、几何、反馈、探索、部署与迁移检查 |

## 研究记录与下一步

- [明确的目标与边界](docs/research-scope.md)、[早期讨论总结](docs/discussion-summary.md)。
- [E1 实体空间原型](docs/body-entity-model.md)、[E2 测量优化](docs/entity-stereo-refinement.md)、[E3 自主探索](docs/active-exploration.md)。
- [第一版模型](docs/first-model-experiment.md)、[V4 优化](docs/model-optimization.md)、[V5 环境干扰](docs/environment-interference.md)、[V6 机器人眼位](docs/robot-eye-training.md)。
- [过程、失败和取舍](docs/learning-log.md)、[Seeed SO-ARM 官方资料索引](docs/seeed_soarm/README.md)、[相机与反馈约定](docs/seeed_soarm/02-camera-and-model-inputs.md)。

下一阶段需要支持探索中新显露区域的注册、跨姿态表面对应与空间范围、从响应中学习部件关系，以及停止/遮挡/重现时的身份恢复。当前固定的 41 个参考点和人为实体状态保持不足以实现这些能力。影子与工具归属问题仍保留，真实机械臂部署也尚未验证。
