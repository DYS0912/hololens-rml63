# experimental/ —— 未完成的三维实时跟随研发代码

**本目录全部为离线研发代码：未接入 ROS/驱动，未实机验证，不是可直接用于实机的控制器。**

| 文件 | 说明 |
|---|---|
| follow_target_core.py | 离线平移规划器（工作空间边界需在 Base 坐标下实机勘测，不能凭空设定） |
| follow_command_gate.py | 离线命令门参考实现（epoch/停止确认纪律；必须运行在最终命令消费侧，发布侧使用不能使 driver 已有订阅队列失效） |
| collect_follow_build_inputs.py | 只读收集本地 SDK/构建输入（不启动驱动、不接触机器人） |
| collect_follow_interface.py | 只读 ROS/源码清单收集（无发布者、无服务、无连接） |

状态：仅完成离线设计；三维连续跟随的驱动集成、松手停止后重锚定实机验证均未完成。
集成前必须先核实 rm_driver 订阅队列在 block=1 下的堆积与停止后恢复执行行为
（见根 README 第七节与 docs/交接说明.md）。
