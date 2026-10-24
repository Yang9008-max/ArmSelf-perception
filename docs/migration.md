# 换平台运行与研究交接

本次交接日期：2026-10-24。仓库：<https://github.com/Yang9008-max/ArmSelf-perception>。固定资料版本：[handoff-2026-10-24](https://github.com/Yang9008-max/ArmSelf-perception/releases/tag/handoff-2026-10-24)。源码、研究讨论、失败经验和代表图在 Git；大文件在同名 Release，不能只下载 GitHub 的 Source code ZIP 就认为已经有权重和数据。

## 带走什么

| 下载项 | 内容 | 用途 |
|---|---|---|
| `runtime` | E1/E2 权重、参考、图表、交互回放、E3 快照与决策、数据元信息、V6 实时权重 | 快速检查已完成结果；没有原始图像和评价数组 |
| `current` | 完整 E1/E2/E3，包括原始双目、实际反馈、评价真值、权重、训练轨迹、决策、报告；加 V6 实时权重 | 继续当前实体感知和探索研究 |
| `legacy` | V1–V6 原始观测、训练权重、曲线、历史评估、动画、环境与视角实验、场景快照；冻结背景参考与缓存元信息 | 回看早期路线和重建分割实验 |
| `all` | 下载 `current` + `legacy`，不再重复下载 runtime | 完整研究交接，推荐 |

压缩后 runtime 为 **30.5 MiB**、current 为 **801.5 MiB**、legacy 为 **248.9 MiB**；all 合计约 **1.03 GiB**、4,406 个文件。运行 `python scripts/artifacts.py list --bundle all` 查看清单；[manifest](../artifacts/manifest.json) 逐文件记录路径、长度与 SHA-256，同时列出未打包的项目。

省略的是可重新生成的逐样本光流/弱标签缓存和进程日志。**原始观测、评价数据、冻结背景参考、权重与指标均保留。** `.venv`、Git 凭证、个人原始报告、第三方下载的规格书也不上传；第三方资料的官方链接与研究笔记在 [Seeed SO-ARM 索引](seeed_soarm/README.md)。

## 新平台的最短路径

1. 安装 Git、Python 3.13；克隆仓库、建立并激活 `.venv`，按 [README](../README.md#安装与恢复) 安装模型依赖。
2. 在仓库根目录执行 `python scripts/artifacts.py fetch --bundle all`。
3. 执行 `python scripts/artifacts.py verify --bundle all` 校验已安装文件。
4. 执行 `python -m unittest discover -s tests`；有图形环境后启动 `python sim/server.py`。
5. 打开数据图回放，确认记录中的 E2/E3 结果与预期一致，再使用新目录开始训练。

只看回放无需先安装 PyTorch 或 MuJoCo：标准库下载 runtime 后，用 `python -m http.server 8767 --bind 127.0.0.1 --directory outputs/embodiment/entity-v2/final-e2` 即可。`http://127.0.0.1:8767/` 是 E2，`/exploration/` 是 E3。

Windows 不方便激活虚拟环境时，将命令里的 `python` 换成 `.\.venv\Scripts\python.exe`；Linux/macOS 可直接使用 `.venv/bin/python`。

## 网络不便时离线恢复

从 Release 下载所需 ZIP、`manifest.json` 和 `SHA256SUMS.txt`。将 ZIP 放在一个目录，例如 `downloads`，然后在克隆仓库内执行：

```bash
python scripts/artifacts.py restore --bundle all --archive-dir downloads
python scripts/artifacts.py verify --bundle all
```

脚本使用仓库中的 manifest，先校验 ZIP 的 SHA-256，再检查成员路径、文件长度与逐文件哈希。不同内容的现有文件会导致明确报错，不会覆盖新实验；最简单的处理是在另一个干净克隆中恢复。相同文件可以重复恢复。

自动下载缓存默认在 `.artifacts/`，支持未完成下载的 HTTP Range 重试。下载器优先通过公开 GitHub 附件 API 取得文件，API 查询不可用时回退网页下载地址，无需登录。交接时本机连接 `github.com` 网页下载入口曾超时，而 API 到附件存储的路径可用，因此建议克隆 `main` 使用最新下载器；冻结数据仍使用同一个 Release。解压后如需节省空间可以自行清理下载缓存，保留 `outputs/`。建议为完整资料、下载缓存和后续训练预留至少 10 GB 可用空间。

## Linux 无显示环境

Python 推理和图表回放不依赖显示服务器，但 MuJoCo 的 RGB 采集需要 OpenGL 上下文。绑定包已自带 MuJoCo 库，无需另装 `mujoco-py`。[官方 Python 安装说明](https://mujoco.readthedocs.io/en/stable/python.html#installation)。

Linux 有可用 GPU/EGL 驱动时，在导入 MuJoCo 前设置：

```bash
export MUJOCO_GL=egl
python -m embodiment.collect --output outputs/embodiment/linux-check --evaluation-only --seed 901
```

无 GPU 的 Debian/Ubuntu 环境可尝试软件渲染，先安装系统库：

```bash
sudo apt-get update
sudo apt-get install -y libosmesa6 libgl1 fonts-noto-cjk
export MUJOCO_GL=osmesa
python -m embodiment.collect --output outputs/embodiment/linux-check --evaluation-only --seed 901
```

EGL/OSMesa 选择来自 MuJoCo 的[官方上下文实现](https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/rendering/classic/gl_context.py)。是否可用取决于目标平台驱动与容器权限；**本次只在 Windows 完成运行验证，未声称已跑过 Linux/macOS**。CPU 软件渲染不等于无法做模型训练，但采图速度可能较慢。不要在 Windows 设置 `MUJOCO_GL=egl/osmesa`。

新生成的中文图优先使用系统 Noto Sans CJK、思源黑体或 Windows 微软雅黑。归档图已包含正确文字，不需要字体即可查看。缺字体只会影响重绘文字，不应因此修改实验数值。

在远程服务器上，服务默认只监听本地回环。可在自己的电脑建立 SSH 端口转发：

```bash
ssh -L 8765:127.0.0.1:8765 -L 8767:127.0.0.1:8767 user@server
```

再在服务器运行实时仿真或报告 HTTP 服务，在本机浏览对应端口。Notebook 平台可使用其提供的端口预览功能；本仓库没有绑定某个云平台的专用 API。

## 原始成果在哪里

| 路径 | 核心内容 |
|---|---|
| `outputs/embodiment/entity-v1/data-v1/` | 566 帧 E1/E2 初始双目数据，训练/验证/新动作/保持/相机中断 |
| `outputs/embodiment/entity-v1/run-v1/` | E1 的 46 点权重、训练轨迹、控制实验和原始评价 |
| `outputs/embodiment/entity-v2/run-v2/` | E2 的 41 点权重、固定参考 RGB、发现记录、冻结哈希 |
| `outputs/embodiment/entity-v2/final-data/` | E1/E2 配对的最终 162 帧，seed 202610241 |
| `outputs/embodiment/entity-v2/final-e1/`、`final-e2/` | 冻结模型在同一新轨迹上的逐点/逐帧结果；后者有整合页面 |
| `outputs/embodiment/entity-v2/report/` | 同点同像素比较、学习曲线、反馈对照、CSV、失败图 |
| `outputs/embodiment/active-v1/410-active/` 等九个目录 | 三种策略 × 三种采集种子；每次 112 帧原始双目、实际 q、目标命令、决策与四个快照 |
| `outputs/embodiment/active-v1/final-data/` | E3 冻结后新采 162 帧，seed 202610244 |
| `outputs/embodiment/active-v1/report/` | 分预算/分种子指标、实际路径代价、图表与三策略同步回放 |
| `outputs/perception/v6-eye/` | 历史实时分割模型、新眼位训练、最终配对评价和反馈消融 |

重要权重 SHA-256：

```text
E1  79a28e31eeca40c2d8d18a9e5de444ea403cf0a1fe13a14c7be267bf0d878806
E2  ee75c98fdd0520d2f871a2475e0fb7fdcacadd69c15c2cc86969f0c596cb9d2e
V6  3afd89ca2cdc2c0e221c32560a987de1a3dbbd538e6ffbba6e0673ca060e15a1
```

所有归档观测、指标、权重保持原始字节。旧 JSON 中的 Windows 绝对路径没有批量改写；报告读取器通过 `project_paths.py` 将其中 `outputs/...` 映射到当前克隆，新实验尽可能记录相对路径。历史代码哈希对应当时的工作树与换行格式；本次迁移有路径和字体适配，不能宣称当前源码哈希与旧冻结记录全部相同。原实验源码保留在 `6177fa8` 及更早提交。

## 如何重新评价归档模型

先恢复 current。下列命令只生成新的评价目录，不改写已发布模型或原始指标：

```bash
python -m embodiment.experiment evaluate --data outputs/embodiment/entity-v2/final-data --output outputs/embodiment/entity-v2/run-v2 --evaluation-output outputs/embodiment/migration-e2-check
python -m embodiment.experiment evaluate --data outputs/embodiment/entity-v2/final-data --output outputs/embodiment/entity-v1/run-v1 --reference-data outputs/embodiment/entity-v1/data-v1 --evaluation-output outputs/embodiment/migration-e1-check
python -m embodiment.compare --baseline outputs/embodiment/migration-e1-check --candidate outputs/embodiment/migration-e2-check --output outputs/embodiment/migration-comparison
```

E1 没有独立保存参考图像，必须指定原始 `--reference-data`；随意拿一张新图重新检测的特征不能当成原身份。E2 已保存 `reference.npz`，恢复时校验其哈希。

E3 原 `report/evaluation.json` 故意不允许直接覆盖。重新跑完整探索请使用 README 中的新目录命令；只重画已保存报告可运行 `python -m embodiment.explore_report build --runs outputs/embodiment/active-v1`，它不重新训练模型。

## 历史光流缓存怎么处理

原始 `observations/*.npz` 和每个缓存目录的 `background-reference.npz`、元信息都在 legacy 包。训练入口 `perception.train` 会根据原图自动生成新缓存；更细的步骤在各历史实验文档中。

**最终测试不能重新拟合背景。** 评价集的缓存必须复用训练集冻结的背景参考。示例（恢复 legacy 后，Bash；Windows 用等价 Python 脚本执行）：

```bash
python - <<'PY'
from perception.data import prepare_cache
prepare_cache('outputs/perception/v6-eye/final-data',
              'outputs/perception/v6-eye/rebuilt-final-cache',
              splits=('test',),
              background_reference='outputs/perception/v6-eye/cache/background-reference.npz',
              refine_labels=True, use_null_evidence=True)
PY
```

这会重新计算光流和弱标签，耗时大于直接看冻结报告。新缓存随 OpenCV/代码版本可能有差别；归档原始指标才是已发表本轮结果的依据。

## 交给下一平台的研究重点

目标是感知机械臂的实体存在。单纯继续追求分割 IoU 或固定 41 点的拟合误差，会再次偏离目标。当前 E3 的最终随机对照优势很小，不应把它写成强化学习或完整自主身体发现已经完成。

下一阶段应明确验收：探索中新身体区域能加入表示；跨姿态能预测可观察表面与未覆盖范围；从局部动作响应学习部件关联；停止、遮挡、重现及相似外物出现时能恢复同一实体。真实舵机反馈、噪声/延迟、未知工具和真正遮挡仍需验证。每次新实验继续记录假设、失败、代价、原始数据与独立测试，而不是只保存最高分。
