# 机械臂自感知：本次检索与可用开源

检索日期：2026-10-24。以下均已核对作者项目页或官方仓库；适配判断是对当前项目的分析。

| 研究 / 开源 | 已公开内容 | 与当前项目的关系与接入判断 |
| --- | --- | --- |
| [Full-Body Visual Self-Modeling of Robot Morphologies](https://arxiv.org/abs/2111.06389)，[官方代码](https://github.com/BoyuanChen/visual-selfmodeling) | 关节状态条件下的连续空间占据查询；训练、仿真数据生成、网格评估代码；仓库 MIT | 与完整身体形状自感知最直接相关。可把目前的稀疏身体点扩展为连续三维表示。官方复现环境是 Ubuntu / Python 3.6 / CUDA 11，并使用表面点云；当前 RGB 弱监督数据还不能直接替代这些输入。适合作为下一阶段三维路线。 |
| [RoboEngine / Robo-SAM](https://roboengine.github.io/)，[官方代码](https://github.com/michaelyuancb/roboengine)，[论文](https://arxiv.org/abs/2503.18738) | 机械臂分割、物体分割、背景增强工具；RoboSeg 数据集及 Hugging Face 权重链接；仓库 MIT | 可作为跨外观分割基线或离线伪标签教师。其预训练语义知识不同于通过自身动作学习身体，接入时应单独报告。当前优先完成 CPU 可验证的模型微调和 RGB 边界细化；本轮未下载或运行其大模型权重。 |
| [CoTracker3](https://cotracker3.github.io/)，[官方代码](https://github.com/facebookresearch/co-tracker) | 在线 / 离线点跟踪、可下载权重、遮挡点跟踪实现 | 可替换当前 E2/E3 的传统点跟踪，减少身体点身份漂移；需要另测三维定位与动作因果一致性。官方建议 GPU，小任务可用 CPU；仓库许可为 CC BY-NC 4.0。本轮未运行其权重。 |
| [SemanticSelfModel](https://github.com/samavmoore/SemanticSelfModel) | 四足机器人关节条件 SDF、身体连续描述子及逆向点跟踪的研究代码 | 提供从“哪里是身体”到“这是哪一块身体”的实现参考。平台与当前机械臂不同；作为扩展阅读，不宣称已复现。 |

本轮实际采用的是 OpenCV 已有的 [GrabCut](https://docs.opencv.org/4.13.0/d8/d83/tutorial_py_grabcut.html)：由网络高置信度身体内部和低置信度范围外自动生成种子，用当前 RGB 修正不确定边界。没有人工涂抹，没有预设机械臂颜色，没有将仿真几何或评估真值提供给推理。

实际训练试验：从 V6 身体分支继续训练，采集 100 组静态干扰物训练姿态和 25 组验证姿态，使用动作 / 反馈导出的弱标签及原有 GrabCut 标签扩展；24 轮、学习率 0.00025，添加 0.12 幅度的亮度扰动。按不读取身体真值的验证损失选择权重，保留 V6 独立冻结的运动响应分支。微调后开发集原始 IoU 为 0.810，低于原始 V6 的 0.814；微调后再细化为 0.921，也低于 V6 加细化的 0.956。因此部署的 V7 沿用 V6 神经网络权重，提升来自 RGB 边界细化；微调候选只作为消融记录保留。

边界细化的两个预设仅在 20 组开发姿态上比较。最终候选和细化参数先冻结，再用不同随机种子采集最终测试集；V6、仅细化 V6、仅微调 V7 与完整 V7 对同一批画面评估。最终指标和逐姿态记录见 `outputs/perception/v7-static/`。

当前优先级是：先稳定可见身体分割，再验证 CoTracker 是否改善身体点身份，最后建立关节条件的三维占据 / SDF。后三个开源研究的指标与当前场景不同，不能直接拿论文结果作为本项目成绩。

最终独立测试使用冻结后的新种子 `202610248`，共 80 组姿态：原始 V6 IoU 0.8441，部署版 0.9540；运动后保持姿态分别为 0.8382 和 0.9428。80 组初始姿态全部改善，平均提升的配对 bootstrap 95% 区间为 [0.0976, 0.1224]。但可见杂物像素误分割率从 4.31% 升至 4.90%，仍有同色物体混淆。最终测试不能支持“完全排除杂物”或真实硬件泛化的结论。

复现本轮已有权重的静态演示：

```powershell
python sim/server.py --checkpoint outputs/perception/v7-static/model/model.pt
python -m perception.demo --checkpoint outputs/perception/v7-static/model/model.pt --environment static --scene sim/scenes/robot-eye-v6.xml --shadows --output outputs/perception/v7-static/replay.gif
```

完整消融评估与结果页重建：`python -m perception.static_report`。只重建已存在指标的结果页：`python -m perception.static_report --build-only`。新的部署权重只在本地 `outputs/` 中，未推送到 GitHub。
