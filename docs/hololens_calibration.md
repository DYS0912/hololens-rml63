# HoloLens 坐标标定准备（离线）

## 用途与边界

为以后建立 **HoloLens 坐标系 → RML63 基坐标系** 的映射做准备：
从 `target_packets.jsonl` 中选出用于标定的样本，保存为
`calibration_samples.jsonl`。

- 本次**只处理已有数据**：不启动 ROS2、不连接机械臂、不发送任何控制
  指令、不修改任何原始文件（packets.jsonl / target_packets.jsonl /
  已有数据集均只读）；
- **当前仍未完成真实坐标标定**：没有 `robot_tcp_position`（未连接机械臂、
  未采集机器人 TCP），因此**不能计算 HoloLens 到 RML63 的真实变换**；
- `hand_position` / `target_position` 均为**未标定的 Unity tracking
  space 坐标**，不是 RML63 基坐标，也不是 LeRobot action；每条样本标记
  `coordinate_status = "uncalibrated_unity_tracking_space"`。

## 使用

```bash
cd /home/dys/rml63_robot_project
python3 scripts/prepare_hololens_calibration.py \
  --input datasets/hololens_checks/run_<id>/target_packets.jsonl \
  --output datasets/hololens_checks/run_<id>/calibration_samples.jsonl
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input` | 必填 | 输入 target_packets.jsonl |
| `--output` | 必填 | 输出 calibration_samples.jsonl（默认拒绝覆盖，`--overwrite` 显式允许；输入输出同一文件始终拒绝） |
| `--min-distance` | `0.02` | 相邻样本 `hand_position` 最小欧氏距离（米）；`0` 关闭去重 |
| `--max-samples` | `0` | 最多保存样本数（`0` = 不限制） |

仅 Python 标准库，无第三方依赖。

## 选取规则

1. 只选 **target_valid 严格为 true**、**control_enabled 严格为 true**、
   且 **target_position 为 3 个有限数值** 的帧（有效目标帧）；
2. 与上一个选中样本的 `hand_position` 欧氏距离小于 `--min-distance`
   的帧跳过（防止保存大量重复点）。**按 hand_position 去重**而非
   target_position：target_position 是 0.2 倍缩放的平移预览，捏合开始帧
   target 不动而手会大幅移动，按 target 去重会误删有效配对点；
3. `--max-samples` 限制总数；按输入顺序输出。

## 样本字段

每条样本（JSONL 一行）：

| 字段 | 说明 |
|---|---|
| `session_id` / `sequence` / `timestamp_unix_ms` | 原始记录标识与采集时间 |
| `hand_position` | 手位置（Unity tracking space，未标定） |
| `target_position` | 虚拟目标预览位置（未标定，**不是 RML63 坐标**） |
| `target_rotation` | 恒为 `null`（旋转映射未实现） |
| `coordinate_frame` | `unity_tracking_space` |
| `coordinate_status` | `uncalibrated_unity_tracking_space` |
| `robot_tcp_position` | `null`（当前没有机器人 TCP 数据） |

## 统计

运行结束打印并输出 JSON 统计：输入总帧数、有效目标帧数、选中标定样本数
（附按距离跳过/超上限/手位置异常排除的明细）、坐标系名称、
`robot_calibrated`（当前恒为 `false`）、以及说明性 notes。

## 刚体变换估计（仅供虚拟测试）

`estimate_rigid_transform(source, target)`（Umeyama，纯标准库实现，
含 3×3 Jacobi 特征分解）在给定成对点时估计旋转 R（det=+1）、平移 t 与
拟合误差 rms；点数不足 3 对或点退化（共线/重合）时返回 None。

**本函数当前只用于人工点虚拟测试**（`tests/test_prepare_hololens_calibration.py`
中的已知旋转/随机旋转/噪声/反射/退化用例，人工点仅验证算法，不写成真实
实验数据）。真实标定需要成对的 (hand_position, robot_tcp_position)，
待采集到机器人 TCP 数据后才能进行。

## 测试

```bash
cd /home/dys/rml63_robot_project
python3 tests/test_prepare_hololens_calibration.py
python3 -m unittest discover -s tests -v
```
