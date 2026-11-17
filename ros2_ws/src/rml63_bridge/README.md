# rml63_bridge — 只读 RML63 ROS2 bridge 原型

只订阅睿尔曼 RML63 状态话题、在本地组装 observation packet 的只读原型。
**不含任何控制能力**：无 publisher、无 action client、无 service client、
不订阅任何控制话题。

## 状态

- 阶段: 原型（未接真机、未启动 rm_driver）
- 实现: 订阅 + 帧组装 + validity 判定 + log 输出
- 未实现: `send_action()`、LeRobot (Python 3.12) 连接、ZMQ/共享内存传输、
  rm_driver 断连自动重连联动、真机对齐验证、Intel RealSense D435i 相机接入
  （仅文档预留，见下文）

## 官方 SDK 的 RML63 配置文件

实际机械臂型号：**RML63**
官方 arm_type：**"RML_63"**（rm_driver/config/rm_63_config.yaml）
官方 ROS2 文件名：**rm_63_\***（保持官方命名，不修改）

本桥接只订阅 rm_driver 状态话题，与具体型号无关；真机启动时使用官方 SDK
中**真实存在**的 63 系文件（均已确认存在）:

- `rm_driver/config/rm_63_config.yaml`（arm_type: "RML_63"、arm_dof: 6；
  arm_ip/udp_ip 等需现场修改）
- `rm_driver/launch/rm_63_driver.launch.py`
- `rm_control/launch/rm_63_control.launch.py`
- `rm_bringup/launch/rm_63_bringup.launch.py`
- `rm_description/launch/rm_63_display.launch.py`

注意: 官方 arm_type 字符串是 **"RML_63"**（不要写成 "RM63" 或 "RML63"，
以 SDK 源码/配置为准）。官方包名 rm_driver / rm_control / rm_description /
rm_bringup / rm_ros_interfaces 保持不变。

## 运行（当前禁止：未接真机时不要运行节点）

```bash
source /opt/ros/humble/setup.bash
source <sdk>/install/setup.bash
ros2 run rml63_bridge rml63_bridge_node
```

参数:
- `stale_timeout`（默认 0.1 s）: 状态数据超时阈值，需真机验证后调整
- `require_all_enabled`（默认 true）: 是否要求 6 关节全部使能
- `transport_type`（默认 "log"）: 输出通道，"log" 或 "zmq"
- `transport_address`（默认 tcp://127.0.0.1:5557）: ZMQ 地址

环境变量（参数未设置时生效）:
- `RML63_BRIDGE_TRANSPORT=log|zmq`（未设置时默认 log，绝不默认开启 ZMQ）
- `RML63_BRIDGE_ZMQ_ADDRESS=tcp://127.0.0.1:5557`

## ZMQ 传输（只读、单向）

ZMQ 模式用 PUB socket 单向发送 observation（ROS2 bridge → LeRobot），
JSON 序列化，字段固定为 observation 的 10 个字段（无 action/command/
target_joint/trigger 等控制字段）。**没有接收通道**：transport 类没有
recv/receive 方法，LeRobot 无法通过该 socket 向 bridge 发指令。

启动方式（接真机时才允许，当前禁止实际启动）:

```bash
# 方式一：环境变量
RML63_BRIDGE_TRANSPORT=zmq \
RML63_BRIDGE_ZMQ_ADDRESS=tcp://127.0.0.1:5557 \
ros2 run rml63_bridge rml63_bridge_node

# 方式二：ROS2 参数
ros2 run rml63_bridge rml63_bridge_node \
  --ros-args -p transport_type:=zmq \
  -p transport_address:=tcp://127.0.0.1:5557
```

需要 pyzmq（用户级安装: `pip install --user pyzmq`）。未安装 pyzmq 时
LogTransport 与 ObservationAssembler 不受影响，ZMQ socket 测试自动跳过。
注意 PUB socket 的 slow-joiner 行为: 订阅端必须先 connect 再开始录制，
否则首帧可能被丢弃。

## 订阅话题（全部只读）

/joint_states、/rm_driver/udp_joint_speed、/rm_driver/udp_joint_current、
/rm_driver/udp_arm_position、/rm_driver/udp_joint_en_flag、
/rm_driver/udp_rm_err、/rm_driver/udp_joint_error_code、
/rm_driver/udp_arm_current_status（可选，rm_driver 默认关闭）、
/rm_driver/udp_rm_plus_state（可选，rm_driver 默认关闭）

## Observation 字段

| 字段 | 含义 | 单位/格式 |
|---|---|---|
| timestamp | joint_states.header.stamp（接收时刻） | 秒 |
| joint_position[6] | 关节位置 | rad |
| joint_velocity[6] | 关节速度（RPM 换算） | rad/s |
| joint_current[6] | 关节电流（保持官方定义） | mA |
| tcp_pose[7] | 末端位姿 | [x, y, z, qw, qx, qy, qz] |
| arm_status | 状态码（9=急停） | 0~14 或 None |
| joint_enabled[6] | 使能状态 | bool |
| tool_state | RM Plus 末端设备状态 | Rmplusstate 或 None |
| valid | 帧有效性 | bool |

## D435i 相机预留输入（未实现，仅文档）

实际设备相机型号为 **Intel RealSense D435i**（含 IMU）。当前未安装驱动、
未订阅任何相机话题，以下输入全部为**预留设计**，接入后走独立的图像/
IMU 通道（ZMQ 侧新增字段或独立端口），**不并入当前 25 维 RML63 observation，
Observation 数据类与 LeRobot action/observation 维度保持不变**。

预留输入（未来对应的 realsense2_camera 常见话题）:
- color image          — `/camera/camera/color/image_raw`
- depth image          — `/camera/camera/depth/image_rect_raw`
- aligned depth image  — `/camera/camera/aligned_depth_to_color/image_raw`
- gyro/sample          — `/camera/camera/gyro/sample`
- accel/sample         — `/camera/camera/accel/sample`
- imu                  — `/camera/camera/imu`

注明:
1. 以上全部是**预留接口**，当前未实现任何相机订阅；
2. **话题名称、帧率、IMU 是否启用都需要实机验证**（以真实相机连接后
   `ros2 topic list` 实测为准）；
3. D435i 的 RGB、深度和 IMU **暂时不并入 RML63 的 25 维 observation**；
4. 图像后续应**单独保存**（如 LeRobot 数据集图像文件）或通过**独立图像
   通道**传输；
5. **不要把完整图像数组放进当前 JSON observation**。

离线数据结构（已实现，见 `rml63_bridge/camera_packet.py`）:
- `CameraPacket`: timestamp / camera_name / rgb(HxWx3 uint8) /
  depth(HxW uint16) / aligned_depth(HxW uint16) / gyro[3] rad/s /
  accel[3] m/s² / frame_id / valid / invalid_reasons；
- `make_mock_camera_packet()`: 离线生成 640×480 模拟帧（含 valid/invalid）；
- `validate_camera_packet(packet, require_imu=False)`: 校验缺失/形状/类型，
  IMU 是否必需可配置；缺失数据一律为 None，绝不用零填充；
- `camera_packet_to_metadata()`: 元数据只含图像 shape/dtype 摘要，
  不含原始像素数组。

注意:
1. 通用模块名 realsense2_camera / RealSense 保持不变；
2. D435i 相机接入前不要启动任何相机节点、不要安装驱动。

## 对齐说明（重要）

udp_* 话题无时间戳。本原型把"最新收到的 udp_* 值"当作当前
`joint_states.header.stamp` 帧的数据 —— 依据是 rm_driver 在同一 UDP 定时器
回调内、用同一 UDP 包的数据顺序发布这些话题。**这是源码层面的近似推断，
必须在真机上验证**（DDS 乱序、UDP 丢包、回调抖动）。

## 有效性规则

valid=true 当且仅当:
- position / velocity / current 长度均为 6；
- 无超时（joint_states 时间戳与所有必选 udp_* 接收时间均新于 stale_timeout）；
- 关节错误码全 0（长度 6）；
- rm 错误为空；
- arm status 存在且新鲜时不是急停（9）；
- 使能标志长度 6 且（require_all_enabled 时）全为 true。

任何缺失/超时/错误 → valid=false，缺失字段为 None，**绝不用零填充**。
零值是合法数据（回零位、零电流），不作为错误判据。

## 测试

```bash
source /opt/ros/humble/setup.bash
source <sdk>/install/setup.bash
python3 -m pytest test/ -v   # 纯静态单元测试，不启动任何 ROS2 节点
```
