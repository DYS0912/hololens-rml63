# examples/ —— 经检查的数据样例与实验证据摘录

| 文件 | 内容 | 来源 |
|---|---|---|
| hand_step_5cm_success_events.jsonl | 5cm 单次运动成功运行的关键事件（configuration/gesture_triggered/selected_target/motion_sent/movel_result/completed）+ 发送与完成时刻附近的位姿/速度样例 | 完整日志 /mnt/hgfs/vmshare/hand_step_5cm_1794900795537975634.jsonl（2.3M，保留本地） |
| single_trigger_stop_events.jsonl | 单次触发+停止实验关键事件（motion_sent/stop_requested/movel_result/stop_result/diagnostic_end） | 完整日志 /mnt/hgfs/vmshare/single_trigger_1794895213957199167.jsonl（1.2M，保留本地） |
| packets_sample.jsonl | 真实接收记录前 20 行（协议 rml63.hand_pose.v1） | datasets/hololens_checks/run_20261112_101842/packets.jsonl |
| target_packets_sample.jsonl | 目标转换记录前 20 行（preview_target 字段示例） | 同上 target_packets.jsonl |
| joint_speed_stationary_sample.yaml | 静止关节速度前 200 帧（float32[6]，单位 RPM；记录无时间戳） | datasets/hololens_checks/joint_speed_stationary_20261115_165154.yaml（5812 帧，保留本地） |

完整数据集与日志一律保留本地，未上传（见 docs/文件清单.md 第七节）。
