# RML63 机器人项目（统一目录）

RealMan RML63 机械臂 + Intel RealSense D435i + Dora 数据采集 + LeRobot 0.6.2
（ACT 训练目标）。当前阶段：**仅离线 mock 验证**——不启动 rm_driver /
realsense2_camera，不连接 RML63 / D435i / 绑扎枪，不发送任何控制指令，
不训练 ACT。

## 目录结构

```
rml63_robot_project/
├── ros2_ws/
│   ├── src/
│   │   ├── realman_ros2/        # 睿尔曼官方 ROS2 SDK（rm_driver/rm_control/
│   │   │                        #   rm_description/rm_bringup/rm_ros_interfaces，
│   │   │                        #   零修改；未复制 gazebo/moveit2/示例包）
│   │   └── rml63_bridge/        # 自定义只读桥接 + Dora 数据采集层
│   ├── build/ install/ log/     # colcon 构建产物（build 后产生）
├── lerobot/                     # LeRobot 0.6.2 副本（含自定义 RML63 适配器
│   │                            #   src/lerobot/robots/rml63/）
├── datasets/
│   ├── runs/<run_id>/           # 每次录制一个独立目录
│   │   ├── stage/               # 离线 staging（episode parquet + PNG）
│   │   ├── lerobot/             # LeRobot 0.6.2 输出（meta/ data/ images/）
│   │   ├── dataflow.yml         # 本次运行实际使用的 dataflow
│   │   └── control.ctl          # episode 控制文件
│   └── hololens_checks/<run_id>/  # HoloLens 2 手部数据接收日志
│       ├── packets.jsonl        #   通过校验的包（含 valid=false）
│       └── summary.json         #   运行参数与统计
├── scripts/                     # 三个入口 + 路径重写工具 + 手部数据接收
│   ├── build_ros2_ws.sh
│   ├── record_dora.py
│   ├── verify_dataset.py
│   ├── rewrite_image_paths.py
│   └── hololens_receiver.py     # HoloLens 2 手部数据接收（仅接收）
├── tests/test_hololens_receiver.py  # 接收工具离线测试（仅回环，不占 5005）
├── docs/hololens_receiver.md    # 接收工具使用说明
└── README.md
```

## 环境分工

| 用途 | 环境 | Python |
|---|---|---|
| 构建 ROS2 工作空间 / 跑测试 / Dora 录制 | 系统 Python + /opt/ros/humble | 3.10（已装 dora-rs 0.5.0、pyarrow） |
| LeRobot 读回验证 / 训练（未来） | conda env `lerobot`（/home/dys/miniconda3/envs/lerobot） | 3.12 |

## 快速开始（三条命令，均在项目根目录执行）

### 1. 构建 ROS2 工作空间（系统 Python 3.10）

```bash
cd /home/dys/rml63_robot_project
bash scripts/build_ros2_ws.sh
source ros2_ws/install/setup.bash     # 之后需要 ros2 命令时
```

### 2. 离线 mock 录制（真实 Dora runtime，系统 Python 3.10）

```bash
cd /home/dys/rml63_robot_project
python3 scripts/record_dora.py --seconds 4
# 输出在 datasets/runs/<run_id>/lerobot/
```

### 3. 用 LeRobotDataset 读回验证（conda lerobot env，Python 3.12）

```bash
cd /home/dys/rml63_robot_project
/home/dys/miniconda3/envs/lerobot/bin/python scripts/verify_dataset.py
# 默认验证最新一次运行；可指定: --dataset datasets/runs/<run_id>/lerobot
```

## 项目迁移到新位置后

图像列存**绝对路径**是 LeRobot 0.6.2 的要求（PIL 直接 open，不做相对解析）。
整个项目搬到新目录后有两种方式修复：

1. **重新生成（推荐）**：在新位置重新执行入口 2（record_dora.py 从自身
   位置推导项目根，dataflow 路径全部按新位置重写，输出全新绝对路径）；
2. **重建路径**：图像文件已随项目搬移时（parquet 本身没变，只是根路径变了）：
   ```bash
   python3 scripts/rewrite_image_paths.py \
       --dataset datasets/runs/<run_id>/lerobot \
       --new-dataset-root /新位置/rml63_robot_project/datasets/runs/<run_id>/lerobot
   # 默认 dry-run 只打印计划；确认后加 --write
   ```

## HoloLens 2 手部数据接收（仅接收，不控制）

仅 UDP 接收 + 日志 + 统计：不发送任何消息、不建控制通道、不控制机械臂、
不启动 ROS2、不接 Dora、不生成训练数据。详细说明（参数、显示、输出文件、
统计口径、离线测试）见 [docs/hololens_receiver.md](docs/hololens_receiver.md)。

```bash
cd /home/dys/rml63_robot_project
python3 scripts/hololens_receiver.py --duration 60   # 戴好头显后手动运行
# 输出在 datasets/hololens_checks/run_<时间戳>/（packets.jsonl + summary.json）
# 按 Ctrl+C 或等待 --duration 到期即正常收尾；离线测试: python3 tests/test_hololens_receiver.py
```

## 约束（当前阶段）

- 仅离线：STATE_INPUT_MODE/CAMERA_INPUT_MODE/ACTION_MODE 均为 mock；
- 模拟 action 仅用于数据格式验证，不代表真实示范动作，不用于研究结论，
  不得用于训练 ACT；
- 官方 SDK 与 LeRobot 源码零修改（自定义代码全部在 rml63_bridge 与
  scripts/ 中）；
- 旧目录 /home/dys/ros2_rm_robot-humble1.6.0 与 /home/dys/lerobot-main
  保持原样（含旧 conda editable 指向），本工程验证脚本通过限定
  sys.path/PYTHONPATH 使用新副本，不改动旧环境。
