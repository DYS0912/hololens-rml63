# HoloLens2 → RML63 手势遥操作项目

HoloLens2 右手捏合手势遥操作 RealMan RML63 六轴机械臂。当前阶段：**单次手势触发运动与停止已完成实机验证；三维连续实时跟随尚未完成**，对应入口保持硬锁定（`--enable-motion` 报错退出）。

> 本仓库为交接整理版（复制整理，原工作目录 `/home/dys/rml63_robot_project` 与桌面程序保持原样、仍可运行）。
> 详细文件来源、版本依据与验证状态见 [docs/文件清单.md](docs/文件清单.md)。

## 一、系统组成与数据流

```
┌────────────┐   UDP 5005 (直连, rml63.hand_pose.v1)   ┌──────────────────────────┐
│ HoloLens2  │ ─────────────────────────────────────► │ Ubuntu (172.20.10.2)     │
│ Unity 应用 │  右手位姿/valid/tracked/control_enabled │  hololens_receiver.py    │
│ (Windows)  │                                        │   → 校验/落盘 packets    │
└────────────┘                                        │  hololens_target_        │
        ▲ 头显状态回传：未实现（ROBOT: UNKNOWN 属正常） │    converter.py（离线）  │
        │                                             │  hololens_live_preview.py│
        │                                             │  hololens_robot_control  │
        │                                             │    .py（运动锁锁定）     │
        │                                             │  hololens_hand_step_5cm  │
        │                                             │    .py（单次，已验证）   │
        │                                             └────────────┬─────────────┘
        │                                       ROS2 Humble topic ─┤  /rm_driver/movel_cmd
        │                                                          │  /rm_driver/move_stop_cmd
        │                                              ┌───────────▼───────────┐
        │                                              │ rm_driver（官方 SDK，  │
        │                                              │ 未收录本仓库）         │
        └──────────────── 未实现 ──────────────────────│ /rm_driver/udp_arm_    │
                                                        │   position（≈200Hz）  │
                                                        │ /rm_driver/udp_joint_ │
                                                        │   speed（6轴, RPM）   │
                                                        └───────────┬───────────┘
                                                                    ▼
                                                               RML63 机械臂
```

- 头显端：Unity + MRTK 手部追踪，右手位姿经 UDP 直连发送（协议 `rml63.hand_pose.v1`，Unity tracking space，米，四元数 xyzw）。**Unity 源码位于 Windows（`E:\UnityProjects\RML63_HoloLens2`），本次未收录，见 [docs/文件清单.md](docs/文件清单.md) 未收录说明。**
- Ubuntu 端：接收/校验/落盘 → 离线目标转换 → 只读预览 / 单次运动 / （未完成）连续控制。
- ROS2 层：官方 `rm_driver` 订阅 `/rm_driver/movel_cmd`（Movel，阻塞模式）与 `/rm_driver/move_stop_cmd`，发布 `/rm_driver/udp_arm_position`（TCP 位姿，约 200 Hz）与 `/rm_driver/udp_joint_speed`（六轴速度，单位 RPM，精度 0.02 RPM）。

## 二、实际环境版本（以现场核实为准）

| 组件 | 版本/配置 |
|---|---|
| Ubuntu | Linux 6.8.0-149-generic；IP 172.20.10.2（与头显直连） |
| HoloLens2 | 172.20.10.3；Windows Unity 工程（待补充源码） |
| Python | 系统 3.10（接收/转换/控制/Dora 录制）；miniconda `lerobot` 环境 3.12（verify_dataset） |
| ROS2 | Humble；`source /opt/ros/humble/setup.bash` + `/home/dys/DYS_ws/install/setup.bash` |
| rm_driver | 官方 SDK `ros2_rm_robot-humble`（lib v1.1.3；包内 license 字段为 TODO，未收录本仓库） |
| Dora-rs | ≥0.5（系统 Python，`pip install 'dora-rs>=0.5.0'`） |
| LeRobot | 0.6.2 副本（仅自研 RML63 适配器收录，见环境文档） |

## 三、已完成 / 未完成（与代码、日志逐条核对）

**已完成并验证**
- HoloLens2 右手追踪、捏合识别、UDP 位姿发送（协议 v1）。
- Ubuntu 接收、逐包校验（schema/类型/维度/有限性/四元数范数）、按来源 IP 过滤、落盘与统计（`hololens_receiver.py`，97 项既有离线测试覆盖）。
- 离线目标转换、间隔重锚定、未标定预览（`hololens_target_converter.py`）。
- Dora 离线回放链路与逐帧比较（`hololens_replay_dora.py`，真实记录 708 帧 / 292 有效）。
- **单次手势触发运动与停止实验**（日志见 examples/）：
  - `hololens_hand_step_5cm.py` 成功完成单次 Base +Z 运动：位移比例 1:1，单次目标上限 0.05 m，保持工具姿态，每次启动最多发送一个运动目标，到位误差约 0.39 mm（日志 `hand_step_5cm_1794900795537975634.jsonl`，用户亲眼确认机械臂移动）。
  - `hololens_single_trigger.py` 单次触发 + 150 ms 后请求停止实验（日志 `single_trigger_1794895213957199167.jsonl`，movel_result 与 stop_result 均收到；注意停止回执延迟 ≠ 实际制动时间）。
- 停止确认机制（离线验证）：`move_stop_result` 时间门控 + 位置/关节速度双通道连续静止窗口；正常运动回执 2s 期限、停止请求后单独等待在途回执 15s，互不混用（`hololens_robot_control.py`，92 项新增离线测试覆盖）。

**尚未完成（不要视为已实现）**
- 三维连续实时跟随的完整驱动集成（`experimental/follow_target_core.py`、`follow_command_gate.py` 仅为离线研发模块，**不是**可实机使用的控制器）。
- 松手可靠停止后，重新锚定并继续跟随的完整实机验证。
- 实时发送周期、端到端延迟、工作空间边界的实机验证。
- 头显接收机器人状态回传（头显显示 ROBOT: UNKNOWN 属预期）。

## 四、目录结构与推荐阅读顺序

```
scripts/            当前程序入口（含实机验证过的单次运动程序）
ros2_ws/src/rml63_bridge/   自研 bridge + Dora 节点 + dataflow（保留原相对路径依赖）
lerobot/src/lerobot/robots/rml63/  自研 LeRobot RML63 适配器
tests/              离线测试（全部模拟对象，不接硬件）
docs/               说明文档（文件清单/实验进度/交接说明/环境与部署）
experimental/       未完成的三维跟随研发代码（离线，禁止实机）
legacy/             历史实验版本（被 scripts/ 取代，保留备查）
examples/           经检查的小数据样例与实验证据摘录
datasets/hololens_checks/run_20261112_101842/  真实记录小样例（测试与回放依赖）
```

**师兄推荐阅读顺序**
1. [docs/交接说明.md](docs/交接说明.md) —— 先看哪些文件、当前哪个程序真的运行过。
2. [docs/实验进度.md](docs/实验进度.md) —— 带日志证据的进度与未验证项。
3. [docs/文件清单.md](docs/文件清单.md) —— 每个文件的来源、版本依据、验证状态。
4. `scripts/hololens_hand_step_5cm.py` —— 唯一完成实机运动验证的入口（只读模式可安全运行）。
5. `scripts/hololens_robot_control.py` —— 连续控制的当前状态（运动锁锁定）与停止状态机设计。
6. [docs/环境与部署.md](docs/环境与部署.md) —— 依赖与安装方式。

## 五、运行入口与模式

| 程序 | 只读模式 | 真实运动 | 说明 |
|---|---|---|---|
| `hololens_hand_step_5cm.py` | 默认（不创建运动/停止发布器） | `--enable-motion`（单次目标，最多一个） | 实机验证过的单次运动入口 |
| `hololens_robot_control.py` | 默认只读，不发布任何消息 | **`--enable-motion` 被硬性禁用**（报错退出，exit 3）：driver 订阅队列在 block=1 下堆积、停止后恢复执行的风险未实机核实 | 连续控制的研发状态机 |
| `hololens_live_preview.py` | 仅订阅机器人状态 + 终端预览 | 无 | 实时只读预览 |
| `hololens_receiver.py` | 仅接收 UDP 落盘 | 无 | 数据采集 |
| `hololens_target_converter.py` / `hololens_replay_dora.py` / `prepare_hololens_calibration.py` | 离线 | 无 | 离线管线 |

运行前请阅读各脚本头部 docstring（含安全边界）。**真实运动实验必须两人在场、急停可用，且须按 [docs/交接说明.md](docs/交接说明.md) 的实机前置条件逐项核实。**

## 六、关键约定

- **单位**：所有位置/位移为**米**（代码与日志）；仅显示输出可能以 mm 呈现（如 `%.1fmm`）。
- **四元数**：顺序 xyzw；发送端 `quaternion_order: "xyzw"` 已随协议固定。
- **坐标系**：头显数据为 **Unity tracking space**（未标定，不是 RML63 基坐标）；机器人目标为 **RML63 Base 坐标**；两者尚未建立真实标定（见 `docs/hololens_calibration.md`，目前只有虚拟测试）。轴映射通过 `--axis-signs`（±1）与 `--axis-order`（x,y,z / x,z,y）配置（如单次实验中 Unity +Y → Base +Z）；**当前版本未验证坐标映射精度**，仅作近似方向映射。
- **关节速度**：`/rm_driver/udp_joint_speed` 为 `float32[] joint_speed`，6 轴，单位 RPM、精度 0.02 RPM，需控制器 UDP 上报 `joint_speed_enable=true`。
- **停止语义**：`move_stop_result=true` 仅表示 `rm_set_arm_stop` 接口返回 0（控制器接受停止指令），**不等于机械臂已静止**；实际静止必须由反馈确认（位置 + 关节速度连续静止窗口）。停止回执延迟不是实际制动时间。
- **目标限幅 ≠ 位置保证**：`clamp_radius`/`limit_step` 限制的是**发送的目标**，不是机械臂实际位置绝不会越界。

## 七、已知限制与下一步工作

1. **队列风险**（连续控制被锁定的直接原因）：rm_driver 的 `movel_cmd` 订阅队列（ParametersQoS 深度 10）在 block=1 阻塞回调期间会堆积；停止后已排队命令是否被丢弃/恢复执行，SDK 为编译库、驱动源码无清队逻辑，未实机核实。需核实：新 movel 是否替换在途轨迹；`move_stop` 后已排队命令是否继续执行；`joint_speed_enable` 运行时确实使能。
2. 连续控制已实现"一次最多一个在途命令"纪律（离线验证）：上一命令 `movel_result` 未明前不发下一条，失败/超时/未知即进程级锁定。
3. 静止阈值（位置 0.5 mm、速度 0.1 RPM）仅为静止噪声记录（最大 0.05 RPM）下的下界参考，最终值须结合运动样本与实机验收确定。
4. Unity 端源码待补充（Windows 工程）；头显状态回传未实现。
5. SIGKILL 无进程内兜底，唯一保障是人工急停。

## 八、测试

```
python3 -m unittest discover -s tests     # 189 项离线测试（97 既有 + 92 新增），全部模拟对象
python3 tests/test_hololens_control_logic.py   # 控制逻辑 92 项
```

离线测试通过**不代表**实机验证通过；实机验证状态以 [docs/实验进度.md](docs/实验进度.md) 的日志证据为准。
