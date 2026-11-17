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

from dataclasses import dataclass

from ..config import RobotConfig


@RobotConfig.register_subclass("rml63")
@dataclass(kw_only=True)
class RML63Config(RobotConfig):
    """Configuration class for the RealMan RML63 6-axis arm.

    Skeleton config: the adapter does not connect to the real robot yet, so only
    the base `id` and `calibration_dir` fields are used at this stage. ROS2
    parameters (e.g. topic names, UDP port, joint order) will be added here when
    the rclpy-based bridge is introduced. The official ROS2 package/topic names
    (rm_driver, rm_control, rm_ros_interfaces, /rm_driver/... topics) must not be
    changed by this adapter.
    """
