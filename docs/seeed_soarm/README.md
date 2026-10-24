# Seeed SO-ARM 资料索引

整理日期：2026-10-23（Asia/Shanghai）。根据用户所说的“seed 的 soarm”，按 Seeed Studio 的 SO-ARM100 / SO-ARM101 整理。具体套件型号、Standard/Pro 版本和摄像头型号尚未由用户确认。

本目录包含官方资料的中文研究摘记和来源链接。网页摘记是针对本项目的概括，不是网页全文镜像。原厂规格书已下载供本地查阅，公开仓库使用官方文件链接，并保留版本与下载校验记录。

## 本地资料

- [官方硬件、传感器与采集接口摘记](01-official-notes.md)
- [摄像头位置与模型输入约定](02-camera-and-model-inputs.md)
- [STS3215-C001 原厂规格书 PDF](https://files.seeedstudio.com/products/Feetech/108090023_STS3215-C001_Datasheet.pdf)：Seeed 官方文件站托管，8 页，A/1，2023-06-23。第 4 页可查磁编码器和反馈类型。该文件针对所列舵机，不自动代表所有 Pro 版本舵机。本地下载路径为 `originals/STS3215-C001_Datasheet.pdf`，不随仓库分发。
- [来源与下载校验记录](sources.json)
- [本项目已确认的研究目标](../research-scope.md)

## 官方网页入口

| 内容 | 来源 |
|---|---|
| SO-ARM100/101、套件与开源项目 | [Seeed 第 2 章](https://wiki.seeedstudio.com/robotics/soarm_physical_ai_course_chapter_2/) |
| 关节与总线舵机 | [Seeed 第 4 章](https://wiki.seeedstudio.com/robotics/soarm_physical_ai_course_chapter_4/) |
| 舵机状态读取 | [Seeed 第 7 章](https://wiki.seeedstudio.com/robotics/soarm_physical_ai_course_chapter_7/) |
| 摄像头位置及采集 | [Seeed 第 12 章](https://wiki.seeedstudio.com/robotics/soarm_physical_ai_course_chapter_12/) |
| 图像、状态与动作数据组织 | [Seeed 第 13 章](https://wiki.seeedstudio.com/robotics/soarm_physical_ai_course_chapter_13/) |
| STS3215 反馈量含义 | [飞特原厂说明](https://www.feetechrc.com/20210430-56680.html) |
| 默认观测与动作接口 | [LeRobot SOFollower 源码](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py) |
| 位置单位配置 | [LeRobot SOFollower 配置](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/config_so_follower.py) |
| 原始机械设计 | [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) |

Seeed 教程使用其验证过的 LeRobot 分支；上游 `main` 会变化。本次代码核查对应查阅日，未锁定提交 SHA，不能把当前配置默认值套用到所有历史版本。本文没有安装或运行 LeRobot，也没有连接真实硬件。
