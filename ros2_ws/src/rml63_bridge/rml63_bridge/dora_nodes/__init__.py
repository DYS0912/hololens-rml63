"""Dora 数据采集层节点（离线可验证，不依赖 dora 安装）。

本子包包含三个 Dora 节点脚本 + 一个离线模拟 action 源 + 一个 dataflow
离线模拟器：

  rml63_state_dora_node          状态节点（未来 ROS2 输入 -> 7 个输出通道）
  d435i_camera_dora_node         相机节点（未来 D435i 输入 -> 8 个输出通道）
  rml63_dataset_recorder_dora_node  录制节点（同步 + episode 管理 + LeRobot 0.6.2）
  mock_action_dora_node          离线模拟 action 源（仅数据格式验证，不用于训练）
  simulate_dataflow              无 dora 环境下的全链路离线模拟 CLI

设计约束（与整个 rml63_bridge 项目一致）：
  - 所有节点模块可被直接 import，不要求 dora/pyarrow 已安装（惰性导入）；
  - 核心逻辑（*Core 类）只接受纯 Python/numpy 值，可脱离 dora 事件循环测试；
  - 图像只以 numpy 数组 / pyarrow 数组传输，绝不进入 JSON 字符串；
  - 不启动任何 ROS2 节点、不连接任何硬件、不发送任何控制指令；
  - action 缺失时拒绝生成训练数据，绝不用 observation.state 冒充 action。
"""
