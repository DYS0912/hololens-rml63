# HoloLens 2 手部数据接收工具（仅接收）

## 用途与边界

- 只做：UDP 接收、逐包校验、日志保存、统计、终端显示。
- 不做：不发送任何消息、不建控制通道、不控制机械臂、不启动 ROS2 节点、
  不接 Dora、不生成训练数据集、不把手部坐标当作机械臂 action。
- 头显程序已安装，无需修改；本工具不需要重新打包头显应用。
- 依赖：仅 Python 3 标准库（系统 Python 3.10 即可），不安装任何包。

## 网络约定

| 端 | 地址 |
|---|---|
| Ubuntu（本机） | 172.20.10.2 |
| HoloLens 2 | 172.20.10.3 |
| UDP 接收端口 | 5005 |

HoloLens → Ubuntu 直连已实测成功。

## 快速开始（60 秒采集）

戴好头显后，在项目根目录运行：

```bash
cd /home/dys/rml63_robot_project
python3 scripts/hololens_receiver.py --duration 60
```

结束（到达 60 秒或按 Ctrl+C）后自动生成统计文件。

## 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--bind` | `172.20.10.2` | 本机监听 IP |
| `--port` | `5005` | UDP 接收端口 |
| `--peer` | `172.20.10.3` | 期望的头显来源 IP；其他来源只计数不记录 |
| `--duration` | `0` | 运行秒数，**从监听成功起计时**；`0` 表示持续运行直到 Ctrl+C |
| `--no-data-timeout` | `5.0` | 超过多少秒没有“通过校验”的数据就显示“等待头显数据……” |
| `--log-root` | 项目内 `datasets/hololens_checks` | 运行日志根目录 |

## 终端显示（每秒一行）

```
[08:59:45] 右手追踪正常 | 位置：[0.123, -0.456, 0.789] | 最近1秒: 30包 | 累计: 有效 120 / 无效 3 / 格式错误 1 / 来源不符 0
[08:59:46] 右手姿态不可用 | 无坐标 | 最近1秒: 28包 | 累计: 有效 120 / 无效 31 / 格式错误 1 / 来源不符 0
[09:00:10] 等待头显数据……（5.0s 无有效数据） | 最近1秒: 0包 | 累计: ...
```

- **右手追踪正常**：最近一个通过校验的包 `valid=true`，显示原始坐标（显示仅取 3 位小数，不修改数据）。
- **右手姿态不可用 | 无坐标**：最近包 `valid=false`。这是正常的“姿态不可用”记录，
  不是畸形数据；原始空数组/null 与 `invalid_reason` 原样保存，不做零填充。
- **等待头显数据……**：超过 `--no-data-timeout`（默认 5 秒）没有收到“通过校验”的
  数据。此时**不会沿用旧的有效坐标**。“没有收到包”和“收到 valid=false”是
  区分的：前者累计无效数不增长且显示等待行。
- **最近1秒: N包**：最近 1 秒实际收到的、通过校验的包数。

## 停止

- **Ctrl+C（推荐）**或到达 `--duration`：关闭 socket、刷新文件、保存 summary.json。
- SIGTERM 也会正常收尾。
- **`kill -9` 强制结束不保证生成最终 summary.json**（已刷新的 packets.jsonl 部分仍在）。

## 输出文件

每次运行创建独立目录 `datasets/hololens_checks/run_<年月日_时分秒>/`，
同名自动追加 `_2`、`_3`，绝不覆盖旧记录：

```
datasets/hololens_checks/run_20261112_085944/
├── packets.jsonl   每个通过校验的包一行（含 valid=false）
└── summary.json    运行参数与统计（正常结束时生成）
```

### packets.jsonl

每行 = 发送端原始协议字段（完整原样保留） + 附加对象 `_receiver`：

| 附加字段 | 含义 |
|---|---|
| `_receiver.wall_time_ms` | 接收端墙钟时间（unix 毫秒） |
| `_receiver.monotonic_s` | 接收端单调时钟（秒，用于频率/间隔统计） |
| `_receiver.from_ip` | 来源 IP |
| `_receiver.from_port` | 来源端口 |

附加字段**绝不覆盖**发送端原始字段（若发送端恰好有 `_receiver` 键，接收端改用
`__receiver`）。valid=false 的 `position`/`rotation` 保留原始 `[]`/`null`，
不零填充，不伪造坐标。

### summary.json 主要字段

| 字段 | 说明 |
|---|---|
| `run.*` | 运行参数（bind/port/peer/duration/超时阈值）、开始与结束时间、结束原因（`duration_reached` / `keyboard_interrupt` / `sigterm` / `disk_write_error` 等）、运行目录 |
| `timing.wait_before_first_packet_s` | 首包前等待时间（从监听成功到首个通过校验的包）；零包时为 null 并附原因 |
| `packets.total_received` | 收到的 UDP 包总数 |
| `packets.wrong_source` | 来源 IP 不符的包数（不进入记录） |
| `packets.malformed` / `malformed_reasons` | 格式错误的包数及按原因分类（如 `not_json`、`schema_mismatch`、`sequence_invalid`、`position_invalid`、`quaternion_norm_out_of_tolerance`…） |
| `packets.validated` | 通过校验的包总数（= 有效 + 无效） |
| `packets.valid_pose` / `invalid_pose` / `valid_ratio` | 有效/无效姿态数及有效比例（基于**全部**通过校验的包，与每秒屏幕采样无关） |
| `timing.first/last_validated_*` | 首/末个通过校验包的接收时间（墙钟 ISO + 单调秒） |
| `timing.avg_receive_rate_hz` | 平均接收频率 = (N-1)/(末包单调时间-首包单调时间)，N 为通过校验的包数；少于 2 包或时间差为 0 时为 null 并附原因 |
| `timing.interval_mean_s` / `interval_p95_s` / `interval_max_s` | 接收间隔的均值、P95（最近秩法）、最大值 |
| `sessions.<session_id>.*` | 各会话序号统计：首/最大序号、包数、有效/无效数、重复数、乱序数、迟到补齐数、结束时仍未补齐的缺失序号 |

不可计算的指标一律写 `null` 并附 `*_reason` 说明，不做除零。

## 校验与统计口径（重要）

1. 处理顺序：**来源 IP 过滤 → JSON/字段校验 → 落盘**。三类分别计数：
   来源不符（`wrong_source`）、格式错误（`malformed`）、有效/无效姿态。
2. 校验内容：JSON 可解析（拒绝 NaN/Infinity）；`schema` 必须为
   `rml63.hand_pose.v1`；`valid`/`tracked` 必须是布尔值；`sequence` 必须是非负
   整数（布尔不算整数）；`session_id` 必须是非空字符串；时间字段必须是有限数值；
   `valid=true` 时 `position` 必须 3 维、`rotation` 必须 4 维且全部为有限数值，
   四元数范数须满足 `|norm-1| <= 1e-3`（容差为常量，已记录在 summary）。
3. 单个错误数据包不会导致程序退出。
4. 序号按 `session_id` **分别**跟踪：新会话序号从头开始不视为丢包；区分
   重复、乱序、暂时缺口与迟到补齐（迟到包补齐缺口会更新统计）。
5. `missing_sequences` 只表示“某些序号未被本次接收记录覆盖”（未收到，或收到但
   未通过校验被剔除），**不宣称网络丢包**。
6. 两端时钟未同步：不要用发送时间戳与接收墙钟时间之差估算单向延迟；
   接收间隔/抖动也不是端到端控制延迟。summary 中的 `notes` 也重复声明了这几点。
7. `--duration` 从监听（bind）成功开始计时；平均频率排除首包前的等待时间。
8. 磁盘写入失败会明确报错并停止记录，不假装成功（退出码非 0）。

## 端口被占用怎么办

如果 5005 被占用（例如旧的临时脚本 `~/rml63_hand_receiver.py` 还在运行）：

1. **回到旧终端按 Ctrl+C 停止旧脚本**；
2. 不要自行结束不明进程；
3. 然后重新运行本工具。

## 离线测试

```bash
cd /home/dys/rml63_robot_project
python3 tests/test_hololens_receiver.py
# 或
python3 -m unittest discover -s tests -v
```

覆盖：有效姿态、valid=false 的 `[]`/`null`、JSON/字段类型/维度/非有限值错误、
来源过滤、会话切换、重复/乱序/缺口/迟到补齐、零包与单包统计、首包前等待不影响
平均频率、日志读回与独立运行目录、接收超时不沿用旧有效坐标。
socket 集成测试**只使用 127.0.0.1 回环与临时空闲端口，不占用真实采集端口 5005**。

## 协议字段参考

`schema = rml63.hand_pose.v1`，`source = hololens2`，`hand = right`，
`pose_type = openxr_grip`，`coordinate_frame = unity_tracking_space`，
`position_unit = m`，`quaternion_order = xyzw`；另有 `session_id`、`sequence`、
`timestamp_unix_ms`、`monotonic_seconds`、`tracked`、`valid`、`invalid_reason`、
`position`（3 维）、`rotation`（4 维）。
