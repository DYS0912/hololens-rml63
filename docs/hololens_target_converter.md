# 离线手部轨迹 → 未标定虚拟目标预览转换器

## 用途与边界

把 `scripts/hololens_receiver.py` 采集的 `packets.jsonl` 逐行转换成带
`preview_target` 的离线轨迹预览：

- **只做**：离线逐行转换、统计、落盘。**不做**：不监听 UDP、不发送任何控制
  指令、不启动 ROS2、不连接任何硬件、不接 Dora、不控制机械臂。
- **不修改**输入 `packets.jsonl`（只读打开，输出写独立文件）。
- `preview_target` 是**未标定的虚拟位置预览**：坐标轴沿用 Unity tracking
  space，**不是 RML63 基坐标，也不是 LeRobot action**，不用于训练、不用于
  机械臂控制。
- 当前**只做平移映射，不做旋转映射**：`target_rotation` 恒为 `null`，注明
  “旋转映射未实现”，不伪造单位四元数。

## 快速开始

```bash
cd /home/dys/rml63_robot_project
python3 scripts/hololens_target_converter.py \
    --input datasets/hololens_checks/run_20261112_101842/packets.jsonl \
    --output datasets/hololens_checks/run_20261112_101842/target_packets.jsonl
```

输出文件已存在时默认拒绝覆盖（另选文件名，或用 `--overwrite` 显式允许）；
输入与输出指向同一文件时始终拒绝。

## 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input` | 必填 | 输入 packets.jsonl 路径 |
| `--output` | 必填 | 输出 JSONL 路径（默认拒绝覆盖） |
| `--scale` | `0.2` | 平移缩放系数，必须为**有限正数** |
| `--interval-threshold` | `0.2` | 锚定期间的接收间隔阈值（秒）。**这是离线记录间隔检查，不是实时断网看门狗** |
| `--overwrite` | 关 | 允许覆盖已存在的输出文件 |

## 算法（捏合 → 虚拟目标状态机）

- 虚拟目标初始位置 = `[0,0,0]`。这是**明确设置的预览原点**，不是用零填充
  缺失数据。
- 仅当 `valid` 与 `control_enabled` 都**严格为布尔 true**，且 `position`
  为 3 个**有限数值**时，才允许更新目标。`control_enabled` 缺失时**默认不启用**。
- **开始捏合**：`hand_anchor` = 当前手位置，`target_anchor` = 上一有效虚拟
  目标；该帧输出 `target_anchor`，**不产生跳变**。
- **持续捏合**：`target_position = target_anchor + scale × (position - hand_anchor)`。
- **松开 / 姿态无效 / 字段错误**：`target_valid=false`、`target_position=null`
  并写明原因；内部保留上一虚拟目标供下次重新锚定，**不把旧目标冒充有效输出**。
- **会话切换**：`session_id` 改变时重新开始一段，虚拟目标回到预览原点。
- **序号**：同一会话内**重复或倒退**的 `sequence` 不更新目标，并**中断当前
  锚定**；前进方向上的序号缺口不单独处理，由接收间隔检查兜底。
- **接收间隔**：锚定期间相邻记录间隔超过阈值（或时间倒退、时间戳缺失无法
  校验）时中断锚定并重新锚定（无跳变）。时间来源**优先 `_receiver.monotonic_s`**，
  其次 `monotonic_seconds`，再次 `_receiver.wall_time_ms/1000`。

## 输出格式

每行 = 原始记录字段（原样保留）+ 独立字段 `preview_target`：

| 字段 | 说明 |
|---|---|
| `target_valid` | 该帧是否输出有效虚拟目标（bool） |
| `target_position` | 虚拟目标位置 `[x,y,z]`；无效帧为 `null`（不零填充） |
| `target_rotation` | 恒为 `null`（旋转映射未实现） |
| `reason` | 本帧原因（中文，如“捏合持续映射”“松开捏合…”“接收间隔 … 超过阈值…”“重复序号…”） |
| `frame_class` | `valid_output` / `paused` / `abnormal`，用于统计 |
| `rotation_note` | “旋转映射未实现（target_rotation 恒为 null，不伪造单位四元数）” |
| `coordinate_note` | “未标定的虚拟位置预览：坐标轴沿用 Unity tracking space，不是 RML63 基坐标，也不是 LeRobot action” |

若原始记录已含 `preview_target` 键，转换结果改写入 `_preview_target`，
绝不覆盖原始字段。无法解析为 JSON 对象的行写成
`{"_raw_input_line": <原文>, "preview_target": {...abnormal...}}`，计为异常帧。

## 统计

转换结束在终端输出（可复制为 JSON）：

- `valid_output`：有效输出帧数（捏合开始/重新锚定 + 持续映射）
- `paused`：暂停帧数（松开捏合、姿态不可用 valid=false）
- `abnormal`：异常帧数（字段缺失/类型错误、非有限位置、重复/倒退序号、
  无法解析的行等）
- `reanchor_count`：重新锚定次数（各会话内除首次外的捏合开始次数，含松开、
  序号异常、间隔过大、时间戳缺失后的重新锚定）
- `sessions`：各会话的记录数、有效输出数、重新锚定数、首/末序号

## 离线测试

```bash
cd /home/dys/rml63_robot_project
python3 tests/test_hololens_target_converter.py
# 或
python3 -m unittest discover -s tests -v
```

覆盖：起点（预览原点）、缩放、松开、重新捏合不跳变、缺失/非有限位置、
缺失启用字段、会话切换、重复/乱序序号、接收间隔过大、时间来源优先级、
原始文件不变、拒绝覆盖/同一文件、`--scale` 校验、不可解析行。

## 注意事项

- 转换结果不接 Dora、不生成训练数据集、不用于机械臂控制。
- 本工具不校验四元数范数、不做旋转映射；`rotation` 原始字段原样保留在
  输出记录中，仅供后续参考。
