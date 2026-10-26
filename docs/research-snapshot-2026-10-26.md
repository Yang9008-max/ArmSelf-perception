# 研究快照与交接：2026-10-26

本次整理覆盖 E4—E13、V7 静态分割辅助路线、对应代码与测试，以及未实施的 R1 真机先导方案。保留较早的 E1—E3 交接版本，不覆盖历史 Release 或冻结实验。

- [研究定义、三个核心问题及证据边界](research-definition.md)
- [E13 方法、参数、对照和全部失败](body-evidence-e13-summary.md)
- [首轮真机实验方案 R1，尚未实施](real-robot-pilot-r1.md)
- [本次 GitHub Release](https://github.com/Yang9008-max/ArmSelf-perception/releases/tag/research-2026-10-26)
- [原始 E1—E3 交接 Release](https://github.com/Yang9008-max/ArmSelf-perception/releases/tag/handoff-2026-10-24)

## 研究判断

当前系统是在受控仿真中，将学习身体映射与动作预测、实际反馈和视觉证据连接起来的研究原型。最近几轮主要改进已有身体模型上的显式推断，不等于神经网络自主学会完整身体身份、形态或故障概率。

E13 在三个新种子、六条物理轨迹上，受阻并叠加反馈偏差或缺失时，固定身体锚点故障窗口误差从同批 E12 的 75.252 降至 11.216 mm，改善 85.10%。关闭视觉选择通路后，状态与同增强观测 E12 一致。收益来自视觉真正更新状态和未来预测。

限制也保留：两真实构型同时覆盖 82.88%，未达到预设 90%；同一组候选用旧余量只有 20.72%，覆盖改善主要来自重新校准误差范围。宽度限制通过，失败的是同时覆盖要求。受阻且相机缺失仍为 74.332 mm，仅反馈为 10.560 mm。十三条件共享六条物理轨迹，不是三十九次独立物理重复。

下一步优先补真实身体发现和动作后果预测的证据。硬件未确定，不预设型号、不采购、不连接或下发真机动作。普通 RGB 路线先验证二维问题；三维能力需对应的测量和独立评价条件。真机接通不等于模型有效。

## 源码与文档索引

| 内容 | 入口 |
|---|---|
| 固定干扰物、分割与边界细化 | `perception/environment.py`、`perception/boundary.py`、`sim/server.py` |
| E4 共同表示联合评价 | [协议与结果](body-cycle-e4.md)、`embodiment/body_cycle.py` |
| E5 视觉记忆与反馈纠错 | [方法与失败](body-recovery-e5.md)、`embodiment/recovery_cycle.py` |
| E6 动作、反馈与视觉 | [方法与对照](action-body-e6.md)、`embodiment/action_state.py` |
| E7 学习融合贡献 | [训练与边界](learned-trust-e7.md)、`embodiment/trust_state.py` |
| E8 / E8B 逐关节可靠更新 | [两轮总结](model-optimization-e8-summary.md)、`embodiment/reliable_cycle_v2.py` |
| E9 / E9B 部分身体表面 | [两轮总结](body-extent-e9-summary.md)、`embodiment/surface_extent.py` |
| E10 冲突与未知 | [结果与负面发现](body-conflict-e10-summary.md)、`embodiment/conflict_state.py` |
| E11 竞争候选 | [方法与失败](body-hypotheses-e11-summary.md)、`embodiment/hypothesis_state.py` |
| E12 可反驳动作先验 | [方法与结果](body-branches-e12-summary.md)、`embodiment/branch_state.py` |
| E13 视觉选择与候选校准 | [完整报告](body-evidence-e13-summary.md)、`embodiment/evidence_state.py` |
| 真机补充研究 | [R1 草案](real-robot-pilot-r1.md)，没有真机数据或能力结论 |

## 下载与查看

安装环境见根目录 README。只下载可查看的回放、图表、权重和元数据：

```powershell
python scripts/artifacts.py fetch --manifest artifacts/research-2026-10-26.json --bundle runtime
python scripts/artifacts.py verify --manifest artifacts/research-2026-10-26.json --bundle runtime
python -m http.server 8767 --bind 127.0.0.1 --directory outputs/local-check-20261024
```

浏览器打开 `http://127.0.0.1:8767/`，或 `http://127.0.0.1:8767/e13-visual-evidence/report/`。需要启动 HTTP 服务；GitHub 仓库页面不会直接执行交互 HTML。runtime 不包含 NPZ/NPY 原始数组，不能据此重新评分或训练。

完整继续研究需要先恢复原始交接依赖，再恢复本次全部阶段：

```powershell
python scripts/artifacts.py fetch --bundle all
python scripts/artifacts.py fetch --manifest artifacts/research-2026-10-26.json --bundle all
python scripts/artifacts.py verify --manifest artifacts/research-2026-10-26.json --bundle all
```

本次新增文件原始大小约 9.8 GiB，共 43,648 个文件，下载压缩大小以 manifest 为准。下载缓存与解压内容分别占用空间。按阶段单独恢复可把 `all` 换成 `e13-visual-evidence` 等名称；单阶段通常还有旧模型和前序阶段依赖，不代表独立重算环境。`supporting-data` 另含 V7 部署权重、原始静态训练／最终数据、冻结背景参考和六份本机场景快照；完整恢复会自动包含。列出全部包：

```powershell
python scripts/artifacts.py list --manifest artifacts/research-2026-10-26.json --bundle all
```

离线时把 Release ZIP 下载到同一目录，然后将 `fetch` 改为 `restore --archive-dir <附件目录>`。原脚本默认的 `artifacts/manifest.json` 仍指向历史交接，不改变其内容。

较大的阶段拆为约 100 MiB 的独立 ZIP，原始成员没有修改。`--bundle e13-visual-evidence` 等阶段名称会自动选择该阶段的所有小包；`all` 选择完整快照。无需手动拼接 ZIP，也不重复下载已校验的分包。同一次恢复共用 Release 元数据查询，避免每个小包重复请求清单。

## 保存与复现边界

源码、协议、报告和测试进入 Git；大文件由 Release 保存。各阶段原始数据、模型、评分、开发试验和失败均保留；排除 Python 字节码、字体缓存及可从原图再生成的逐样本光流缓存，冻结背景参考与缓存元数据保留。runtime 与完整阶段包有相同文件，恢复器会校验后跳过相同内容，拒绝覆盖不同内容。

本次回归：150 项通过，包含新增的分包恢复测试。测试未重新训练模型或重新采集最终实验。各阶段原有的 73、84、148 等测试数是阶段历史记录，不改写为当前测试数。

Release 另附 `source-working-tree.zip`，保存发布时源码与文档的原始字节，以便核对冻结实验的源文件哈希。Git 的换行转换可能影响冻结哈希，不能修改历史冻结记录来掩盖差异。

部分历史 JSON、工程辅助脚本和后期实验代码仍含采集机的绝对路径。原始文件保留原样，不宣称所有后期重算入口已经跨平台验证；新的机器先恢复文件并查看保存的报告。若重算遇到路径问题，应单独适配路径解析、输出到新目录，不能改写冻结输入或覆盖旧结果。Windows 已验证，Linux/macOS 全流程尚未验证。

设备相关 USB 标定、虚拟环境、本机下载缓存和第三方原始资料不作为自有源码上传。根目录外的本地阅读入口属于工作区导航，不是另一份独立研究结果。
