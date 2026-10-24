# 官方资料摘记：硬件能反馈什么，软件实际读取什么

查阅日期：2026-10-23。以下将官方事实与本项目的解释分开记录。

## 1. SO-ARM 的控制通道

SO-ARM100/101 每条臂有五个手臂旋转关节及一个夹爪控制通道，合计六个 STS3215 总线舵机。控制板连接 USB 与舵机串行总线。SO100/101 和 Standard/Pro 存在版本差异，不能只凭“SO-ARM”确定全部电机参数。来源：[Seeed 第 2 章](https://wiki.seeedstudio.com/robotics/soarm_physical_ai_course_chapter_2/)、[第 4 章](https://wiki.seeedstudio.com/robotics/soarm_physical_ai_course_chapter_4/)。

## 2. 舵机反馈

[STS3215-C001 规格书](https://files.seeedstudio.com/products/Feetech/108090023_STS3215-C001_Datasheet.pdf) 第 4 页 §6-6 说明使用 12 位磁编码器，§7-10 列出位置、速度、负载、输入电压、工作电流、工作温度六类反馈。它们属于舵机状态读回，并不都来自独立的机械传感器。

[Seeed 第 7 章 §7.6](https://wiki.seeedstudio.com/robotics/soarm_physical_ai_course_chapter_7/) 的调试界面还显示运动状态和异常状态。因此“感”可以包含身体内部状态，不能只理解成相机图像。

**不要把 Load 直接当作测得的关节扭矩或末端接触力。** [飞特 2021 年原厂说明](https://www.feetechrc.com/20210430-56680.html) 将 Load 解释为电机驱动电压的占空比。电流和 Load 可以提供负载相关线索，但其物理解释、换算和精度需要匹配具体型号与固件；仅凭这两个字段不能声称已有力觉或触觉。

## 3. 默认 LeRobot 观测不包含全部反馈

核查 [SOFollower 当前实现](https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/robots/so_follower/so_follower.py)：关节依次命名为 `shoulder_pan`、`shoulder_lift`、`elbow_flex`、`wrist_flex`、`wrist_roll`、`gripper`。`get_observation()` 读取 `Present_Position` 及已配置相机帧；受支持的深度相机可另返回深度。`send_action()` 写入 `Goal_Position`，返回下发目标，并非测得位置。速度、电流等需要另行扩展采集；位置读取与相机取帧也不是硬件同步曝光。

[当前配置源码](https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/robots/so_follower/config_so_follower.py) 的 `use_degrees` 默认值为 `True`。结合上述归一化模式，身体关节可使用校准后的度数，夹爪使用 0–100 的归一化值。夹爪值不是毫米。关闭 `use_degrees` 时身体关节使用另一套归一化范围，必须随数据保存配置和校准信息。

本项目的处理建议：单独记录真实读取值、动作目标、单位、时间戳与读数有效性；不要把“下发 30 度”写成“测得 30 度”。未采集的反馈保留为缺失，不用常数零伪造。

## 4. 官方相机布置

[Seeed 第 12 章](https://wiki.seeedstudio.com/robotics/soarm_physical_ai_course_chapter_12/) 给出两类布置：工作台上方主视角搭配夹爪附近视角，或搭配前侧方视角；该课程采用前方加侧方两个视角。这是该课程的数据采集配置，不是本项目第一阶段必须使用两台相机的结论。

本项目解释：全臂自我感知优先让主相机看全活动机械臂。腕部相机可以补充末端画面，但不应默认它能看清整条手臂。两台普通 RGB 相机也不自动等于已标定的双目深度系统。

## 5. 数据组织

[Seeed 第 13 章](https://wiki.seeedstudio.com/robotics/soarm_physical_ai_course_chapter_13/) 将图像流与包含状态、动作的表格数据分别组织，并在元数据中记录帧率和特征定义；相机键名需要在采集、训练和使用时一致。

本项目需要借鉴这些采集习惯，但当前目标是学习身体感知。官方课程中的抓取示范、ACT 策略训练、VLA 工作流不自动成为本项目目标，也不要求先做抓取。
