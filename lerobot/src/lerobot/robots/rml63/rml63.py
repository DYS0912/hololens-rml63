#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from functools import cached_property

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_rml63 import RML63Config

logger = logging.getLogger(__name__)


class RML63(Robot):
    """LeRobot adapter for the RealMan RML63 6-axis arm.

    Skeleton implementation: intentionally does NOT import `rclpy` and does NOT
    talk to the real robot yet. It only provides the required :class:`Robot`
    interface so the rest of LeRobot can be exercised offline.

    Observation features follow the RealMan driver topics:
        - ``joint{i}.pos``     (rad, from ``/joint_states``; position only,
                               degrees are converted to radians by rm_driver)
        - ``joint{i}.vel``     (RPM, from ``/rm_driver/udp_joint_speed``; the
                               raw unit is RPM and must be converted to rad/s
                               or deg/s by the bridge, NOT used as deg/s)
        - ``joint{i}.current`` (mA, from ``/rm_driver/udp_joint_current``; kept
                               in mA per the official message definition;
                               this message has no header timestamp)
        - ``tcp.*``            (m + quaternion, from ``/rm_driver/udp_arm_position``)

    Alignment note: the ``udp_*`` messages carry no header timestamp, so using
    ``joint_states.header.stamp`` as the frame timestamp is only an approximation
    based on both messages being published back-to-back in the same UDP timer
    callback; it must be validated on the real machine before relying on it.

    TODO(rml63): add a ROS2 bridge (subprocess or middleware transport, since
    LeRobot runs on Python 3.12 while ROS2 Humble ships Python 3.10) and replace
    the stub bodies of ``get_observation`` / ``send_action``.
    """

    config_class = RML63Config
    name = "rml63"

    _num_joints = 6

    def __init__(self, config: RML63Config):
        super().__init__(config)
        self.config = config
        self._connected = False

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        features: dict[str, type | tuple] = {}
        for i in range(1, self._num_joints + 1):
            features[f"joint{i}.pos"] = float
            features[f"joint{i}.vel"] = float
            features[f"joint{i}.current"] = float
        features.update(
            {
                "tcp.x": float,
                "tcp.y": float,
                "tcp.z": float,
                "tcp.qx": float,
                "tcp.qy": float,
                "tcp.qz": float,
                "tcp.qw": float,
            }
        )
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"joint{i}.pos": float for i in range(1, self._num_joints + 1)}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        # Stub kept on purpose - no real hardware control in this step.
        # TODO: 需要通过 ROS2 bridge 连接 ROS2 Humble (Python 3.10) 与
        # LeRobot (Python 3.12)，再由 bridge 接入 rm_driver 话题。
        self._connected = True
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        # The RML63 controller handles its own calibration; nothing to do here.
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        # Stub kept on purpose - no ROS2 subscription yet.
        # TODO: 需要通过 ROS2 bridge 读取 /joint_states、/rm_driver/udp_joint_speed、
        # /rm_driver/udp_joint_current、/rm_driver/udp_arm_position。
        # 单位处理: pos=rad; vel=RPM(需换算); current=mA(保持官方定义)。
        # 时间戳: 若用 joint_states.header.stamp 对齐，仅是基于同一 UDP 回调顺序
        # 的近似对齐，必须真机验证。
        obs: RobotObservation = {key: 0.0 for key in self.observation_features}
        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        # Stub kept on purpose - never send commands to the real robot.
        # TODO: 需要通过 ROS2 bridge 将 joint{i}.pos 映射到官方接口
        # (如 /rm_driver/movej_canfd_cmd，类型 rm_ros_interfaces/msg/Jointpos)，
        # 接口名称保持官方 ROS2 命名，不得修改。
        return action

    @check_if_not_connected
    def disconnect(self) -> None:
        self._connected = False
        logger.info(f"{self} disconnected.")
