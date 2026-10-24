# E2：改善空间测量，并把真实进度画出来

2026-10-24。继续优化“机械臂实体存在”的初步空间模型。用户要求看到具体数据图，不能只听指标描述。本轮保留 E1，在 `codex/body-existence` 上修改观测算法、重新训练、冻结，再采集新轨迹比较。

## 现在是否使用深度学习

是混合方案。E2 的神经网络是 `7 → 64 → 64 → 123`，两层 SiLU，12,667 个可训练参数；输入七路实测关节位置，输出 41 个候选特征的三维位置。训练目标由 RGB 双目测量产生，不来自机械臂 CAD 或仿真真值。当前相机图像在预测时用于检查对应关系，位置解码器本身只输入实际 q。

双目测量、候选点筛选、固定身份编号和相机缺失时的状态保持仍是显式算法。本轮没有把这些规则变成学习得到的物体恒存，也没有学到完整表面、占据体积或接触性质。

## 排查与取舍

E1 对双目轨迹只有约 3–5 mm 的误差，但双目参考自身在身体上的几何误差约 14–16 mm。先改善“用来学的观测”，避免网络只是更准确地拟合有偏数据。

在原验证段每隔 5 帧采样、保持相同的时间跟踪像素，比较双目 LK 图像块：

| 双目方法 | 有效身体点观测数 | 各自有效点的几何 RMSE | 同一批 378 次观测的 RMSE |
|---|---:|---:|---:|
| 原 21 × 21 | 435 | 15.26 mm | 14.63 mm |
| 11 × 11 | 378 | 10.60 mm | 10.60 mm |
| 21 × 21 初值再用 11 × 11 细化 | 380 | 10.85 mm | 10.61 mm |

另检查 7 × 7：保留 358 次观测，自身 RMSE 10.25 mm。采用 11 × 11，兼顾观测数量和减少轮廓块中前景/背景混合；不再叠加收益很小的粗到细步骤。时间跟踪仍使用 21 × 21，双目仍要求极线、前后往返、匹配误差及深度有效性检查。

**这里确实查看了开发验证段的仿真几何来诊断、比较观测方法。** 不能说整个开发过程从未看过真值。几何和身体身份没有成为训练目标、候选点筛选输入，也没有被用于删除特定身体外特征。新的最终评价数据在 E2 冻结之后才采集，未用于本轮调参。

额外修正：下一帧跟踪初值独立复制，避免重置无效初值时顺带改写本帧返回的 UV。匹配方法写入检查点；旧检查点仍使用原 21 × 21 方法。新模型保存原参考图像，跨轨迹评价不能重新检测特征并把新的序号冒充原身份。

## 训练与冻结

- 继续使用 E1 的 334 帧训练、80 帧验证；候选筛选和 400 轮训练配方不变。未训练旧测试/停止/中断段。
- 原方法发现 46 个候选，E2 发现 41 个；正确/打乱/恒定反馈的发现数量分别为 **41 / 0 / 0**。
- 验证 RMSE 为 **5.45 mm**，E1 为 4.94 mm。两者拟合不同测量目标，不能仅用这个数字判定真实空间精度。
- E2 于 `2026-10-24T06:51:58.545868+00:00` 冻结；权重 SHA-256：`ee75c98fdd0520d2f871a2475e0fb7fdcacadd69c15c2cc86969f0c596cb9d2e`。
- 随后使用新种子 `202610241` 采集 162 帧：90 新动作（含 10 帧初始 home）、24 保持、48 相机中断且继续运动。桌面干扰物布局/运动也随种子变化。仍是同一眼位和局部动作家族，不是新机器人或全工作空间泛化。
- 两个冻结模型使用各自原参考图像，在同一条新轨迹上独立评价。新模型不因结果而再训练或删除异常点。

## 可直接比较的真实几何误差

配对条件：原参考特征身份相同，两种方法都有效且落在身体上，并且四舍五入后的图像像素相同；对照同一个仿真几何目标。每次特征观测不是独立样本，不能把数量当作独立实验次数。

| 阶段 | 配对观测数 | E1 真实几何 RMSE | E2 真实几何 RMSE |
|---|---:|---:|---:|
| 新动作 | 1,991 | 14.30 mm | **9.81 mm** |
| 停止 | 553 | 15.17 mm | **10.90 mm** |
| 相机中断 | 1,087 | 15.11 mm | **10.68 mm** |

新动作配对误差降低约 **31.4%**。这个结论只适用于上述共同可验证的身体点，不能改写成“完整机械臂空间精度 9.81 mm”。

## 数量代价与失败不能省略

| 阶段 | E1 身体点/帧 | E2 身体点/帧 | E1 全部候选的有效观测率 | E2 有效观测率 |
|---|---:|---:|---:|---:|
| 新动作 | 26.00 | 23.16 | 90.3% | 86.7% |
| 停止 | 27.50 | 24.79 | 98.7% | 97.6% |
| 相机中断 | 26.75 | 24.25 | 94.9% | 91.8% |

E2 新动作中严格位于身体像素内的有效观测为 **65.1%**，仍有约三分之一位于轮廓外。身体点数量下降约 10.9%，候选总数从 46 降至 41。这些数量不是完整表面覆盖率。

**新的轨迹暴露了错误对应。** E2 特征 302 在第 98 帧（t = 10.7 s）错跟到桌面附近：模型投影约 `(171.49, 305.13)`，图像跟踪约 `(205.09, 301.45)`；双目深度为 1.217 m，模型预测为 0.434 m，观测残差达 **821 mm**。该像素的仿真表面深度为 1.144 m，也支持这是背景对应问题。这是整个感知流程的失败，不能删掉后只报身体上的好成绩。

包含所有有效候选观测，对各自双目参考的整体误差如下。这些数字同时包含网络预测误差和错误对应，不能当成已经正确确认的物理点误差：

| 阶段 | E1 整体观测 RMSE | E2 整体观测 RMSE | E2 打乱当前 q |
|---|---:|---:|---:|
| 新动作 | 73.14 mm | 25.43 mm | 75.30 mm |
| 停止 | 94.88 mm | 51.22 mm | 68.62 mm |
| 相机中断 | 76.35 mm | 33.26 mm | 69.25 mm |

相机中断时保持最后所见位置的 E2 对照为 46.19 mm。反馈更新有帮助，但错误视觉对应已污染最后观测对照。这也解释了新轨迹的整体观测误差远高于 E1 原连续记录上的 3–5 mm。

**状态判定还有缺口。** 当前“视觉支持”使用有效点的中位误差；多数点一致时，少数极端错误仍会被掩盖。页面已说明它不表示每个点都已确认。下一轮应把对应关系失效、特征级不确定性和实体形态约束纳入模型，而不只提高点坐标拟合精度。本轮没有在最终测试上继续修改规则。

## 数据图与复现

本地 `outputs/embodiment/entity-v2/` 保留：

- `run-v2/`：权重、原参考图像、缓存、训练曲线、筛选与冻结记录。
- `final-data/`：新轨迹；中断阶段观测 RGB 全零，隐藏 RGB 仅供评价。
- `final-e1/`、`final-e2/`：两个模型的逐帧指标和可配对特征记录。
- `report/comparison.png`：空间精度、训练曲线、中断预测和数量代价四张图。
- `report/feedback-controls.png`：正确反馈、固定位置、打乱反馈、保持最后所见的对照。
- `report/per-frame.csv`、`comparison.json`：可复查原始数值。
- `report/failure-example.png`：第 98 帧错误对应实例。
- `final-e2/index.html`：图表、CSV 下载、三维点与原始相机交互回放。8767 页面现在展示 E2；8765 仍是历史实时分割演示。

```powershell
.\.venv\Scripts\python.exe -m embodiment.experiment fit --data outputs/embodiment/entity-v1/data-v1 --output outputs/embodiment/repro-e2/model --tracker-mode refined11
.\.venv\Scripts\python.exe -m embodiment.experiment controls --data outputs/embodiment/entity-v1/data-v1 --output outputs/embodiment/repro-e2/model
.\.venv\Scripts\python.exe -m embodiment.collect --output outputs/embodiment/repro-e2/data --evaluation-only --seed 202610241
.\.venv\Scripts\python.exe -m embodiment.experiment evaluate --data outputs/embodiment/repro-e2/data --output outputs/embodiment/entity-v1/run-v1 --reference-data outputs/embodiment/entity-v1/data-v1 --evaluation-output outputs/embodiment/repro-e2/e1
.\.venv\Scripts\python.exe -m embodiment.experiment evaluate --data outputs/embodiment/repro-e2/data --output outputs/embodiment/repro-e2/model --evaluation-output outputs/embodiment/repro-e2/e2
.\.venv\Scripts\python.exe -m embodiment.compare --baseline outputs/embodiment/repro-e2/e1 --candidate outputs/embodiment/repro-e2/e2 --output outputs/embodiment/repro-e2/report
.\.venv\Scripts\python.exe -m embodiment.review --data outputs/embodiment/repro-e2/data --output outputs/embodiment/repro-e2/e2 --comparison outputs/embodiment/repro-e2/report --version E2
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

49 项测试通过，包括已知深度的 RGB 双目恢复、无纹理拒绝、跨轨迹参考身份保护和中断阶段继续运动。仅有一条新的最终轨迹，尚无多随机种子统计或真实硬件验证。
