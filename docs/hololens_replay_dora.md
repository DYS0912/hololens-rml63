# 手部记录 → Dora 回放 → 虚拟目标转换（真实 Dora runtime，仅离线）

## 用途与边界

把已保存的 `packets.jsonl` 通过**真实 Dora runtime** 走一遍：
`回放节点 → 虚拟目标转换节点 → 保存节点`，输出与
`scripts/hololens_target_converter.py` 逐帧一致的结果。

- 转换节点**复用** `hololens_target_converter.py` 的状态机（不重写算法）；
- **不监听实时 UDP**、不启动 ROS2、不连接头显/机械臂/相机、不发送控制
  指令、不修改官方 SDK、不安装新依赖（仅用已验证的 dora Python API）；
- 输出保持**未标定的虚拟位置预览**标识（Unity tracking space，非 RML63
  基坐标，非 LeRobot action），不生成 action、不转换成训练数据。

## 一条命令（项目根目录）

```bash
cd /home/dys/rml63_robot_project
python3 scripts/hololens_replay_dora.py
```

- `--input` / `--expected` 缺省时自动取 `datasets/hololens_checks/` 下最新
  `run_*` 目录的 `packets.jsonl` 与 `target_packets.jsonl`；
- 输出目录自动选择 `datasets/hololens_checks/dora_replay_<时间戳>/`
  （重名自动加 `_2`，不覆盖任何已有文件），运行结束打印结果路径。

可选参数：`--scale 0.2`（平移缩放，有限正数）、`--interval-threshold 0.2`
（接收间隔阈值秒，离线记录间隔检查，不是实时断网看门狗）、
`--chunk-size 10000`（回放批大小）、`--timeout 120`（stop_after 兜底上界，
触达说明流程异常）。

## 数据流与终止

```
hololens_replay ──records──> hololens_target_converter ──converted──> hololens_saver
```

- 回放节点按**文件顺序**回放全部记录，每条附加 `_replay` 诊断字段
  （`index`、`wall_time_ms` 回放侧墙钟），**不替换**原始时间戳、
  session_id、sequence、valid、control_enabled 与 `_receiver` 等原始字段；
- 发完全部记录后发送 `{"_eos": true, ...}` 结束标记（**明确通知下游**）；
- 转换节点逐帧调用 `TargetConverter`（间隔判定使用记录内原始时间，
  `_receiver.monotonic_s` 优先，**回放速度不影响结果**），EOS 原样转发；
- 保存节点逐行写 `replay_output.jsonl`（**含有效帧、暂停帧与无效原因**，
  不只保存捏合帧），收到 EOS 后 flush + 关闭文件、写完成标记
  `replay_done.json`，然后退出；
- `dora.run` 在**全部节点退出后自行返回**（实测），不靠固定等待强制终止；
  `--timeout` 仅是 stop_after 兜底上界。

### 传输可靠性（实测结论）

dora 0.5 的通道在“逐条快速发送”与“多批快发（100 条 × 8 批）”时都会
**丢失中间消息**（实测分别只剩 1/6 与 408/708）；**单条大消息
（500KB+）实测 100% 送达**。因此默认 `--chunk-size 10000`，让 708 条记录
落在一条消息内。若未来处理更大的文件且必须分批，请用实测验证批数上限，
不要直接调小批大小。

## 输出与核对

运行目录 `datasets/hololens_checks/dora_replay_<时间戳>/`：

| 文件 | 说明 |
|---|---|
| `replay_output.jsonl` | 真实 Dora 输出：原始字段 + `_replay` + `preview_target` |
| `replay_done.json` | 保存节点完成标记（`state=completed`、记录数、分类计数） |
| `compare_report.json` | 与 `target_packets.jsonl` 的逐帧比较报告 |
| `dataflow.yml` | 本次运行实际使用的 dataflow（路径已重写） |
| `out/` | dora 会话日志（runtime 自动生成） |

逐帧比较内容：数量、顺序（行序）、session_id、sequence、target_valid、
target_position（浮点容差 1e-9）、target_rotation（双方均为 null）、
frame_class（暂停/异常分类）；**排除** `_replay`、`_raw_input_line` 等
回放诊断字段。

验收预期（2026-11-12 的 708 条真实记录）：总 708 | 有效输出 292 |
暂停 416 | 异常 0。不一致时运行器报错并输出前 10 处不匹配，定位原因，
不调整结果凑数。

## 测试

```bash
cd /home/dys/rml63_robot_project
python3 tests/test_hololens_replay_dora.py          # 回放链路离线测试
python3 -m unittest discover -s tests -v            # 全部离线测试（80 个）
```

离线测试不含 dora runtime（核心类直连）；真实 runtime 验收由
`scripts/hololens_replay_dora.py` 完成。
