"""Dora 节点 1：RML63 状态节点（rml63_state_dora_node）。

未来输入（ROS2 话题，经 relay 进入 Dora 通道）：
    joint_states              <- /joint_states（帧触发 + 时间戳来源）
    udp_joint_speed           <- /rm_driver/udp_joint_speed（RPM）
    udp_joint_current         <- /rm_driver/udp_joint_current（mA）
    udp_arm_position          <- /rm_driver/udp_arm_position（Pose）
    udp_joint_en_flag         <- /rm_driver/udp_joint_en_flag
    udp_rm_err                <- /rm_driver/udp_rm_err
    udp_joint_error_code      <- /rm_driver/udp_joint_error_code
    udp_arm_current_status    <- /rm_driver/udp_arm_current_status（可选）

输出（7 个通道，缺失为不发送/None，绝不零填充）：
    state          float32[25] = joint_position[6]+joint_velocity[6]
                                +joint_current[6]+tcp_pose[7]
    tcp_pose       float32[7]
    joint_current  float32[6]
    joint_velocity float32[6]
    status         int（arm_status）
    timestamp      float（joint_states.header.stamp 秒）
    valid          bool

离线模式（STATE_INPUT_MODE=mock，默认）：tick 触发时用 mock_data 生成
确定性 Observation（不依赖 ROS2 消息包、不依赖 rclpy）。
真实模式（ros2）：由外部 relay 把话题数据送入输入通道，joint_states 通道
到达时组装并输出一帧 —— 该 relay 需在连接真机阶段实现（见计划报告）。
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

import numpy as np

# 注意: dora 以普通脚本方式运行本文件（python3 <path>），必须使用绝对导入；
# dataflow env 中的 PYTHONPATH 指向 rml63_bridge 包根目录。
from rml63_bridge.mock_data import DEFAULT_FPS, DEFAULT_INVALID_EVERY, make_mock_observation
from rml63_bridge.observation import Observation, ObservationAssembler
from rml63_bridge.dora_nodes.dora_compat import (
    dora_available,
    describe_node,
    send_named_outputs,
    value_to_py,
)

STATE_OUTPUTS = (
    "state", "tcp_pose", "joint_current", "joint_velocity",
    "status", "timestamp", "valid",
)
STATE_DIM = 25


def build_state_25(obs: Optional[Observation]) -> Optional[np.ndarray]:
    """25 维状态 = pos[6]+vel[6]+current[6]+tcp[7]；任一部分缺失 -> None。"""
    if obs is None:
        return None
    if None in (obs.joint_position, obs.joint_velocity, obs.joint_current, obs.tcp_pose):
        return None
    state = (
        list(obs.joint_position)
        + list(obs.joint_velocity)
        + list(obs.joint_current)
        + list(obs.tcp_pose)
    )
    if len(state) != STATE_DIM:
        return None
    return np.asarray(state, dtype=np.float32)


class StateNodeCore:
    """状态节点核心逻辑（无 dora、无 rclpy，可离线测试）。

    通道名 -> ObservationAssembler 的映射见 ``CHANNEL_HANDLERS``；
    帧触发通道为 ``joint_states``（唯一带时间戳的消息）。
    """

    CHANNEL_HANDLERS = {
        "joint_states": "on_joint_states",
        "udp_joint_speed": "on_udp_joint_speed",
        "udp_joint_current": "on_udp_joint_current",
        "udp_arm_position": "on_udp_arm_position",
        "udp_joint_en_flag": "on_udp_joint_en_flag",
        "udp_rm_err": "on_udp_rm_err",
        "udp_joint_error_code": "on_udp_joint_error_code",
        "udp_arm_current_status": "on_udp_arm_current_status",
    }

    def __init__(
        self,
        input_mode: str = "mock",
        mock_fps: float = DEFAULT_FPS,
        mock_invalid_every: int = DEFAULT_INVALID_EVERY,
        start_timestamp: float = 0.0,
        stale_timeout: float = 0.1,
        require_all_enabled: bool = True,
    ):
        if input_mode not in ("mock", "ros2"):
            raise ValueError(f"input_mode 仅支持 mock/ros2，收到 {input_mode!r}")
        self.input_mode = input_mode
        self.mock_fps = mock_fps
        self.mock_invalid_every = mock_invalid_every
        self.start_timestamp = start_timestamp
        self._assembler = ObservationAssembler(
            stale_timeout=stale_timeout, require_all_enabled=require_all_enabled
        )
        self._mock_index = 0
        self._latest: Optional[Observation] = None
        self.frames_produced = 0

    # ------------------------------------------------------------ 输入
    def on_tick(self) -> dict:
        """tick 心跳。mock 模式：生成下一帧模拟 Observation 并输出。"""
        if self.input_mode != "mock":
            return {}
        obs = make_mock_observation(
            self._mock_index,
            fps=self.mock_fps,
            invalid_every=self.mock_invalid_every,
            start_timestamp=self.start_timestamp,
        )
        self._mock_index += 1
        self._latest = obs
        self.frames_produced += 1
        return self.outputs()

    def on_message(self, channel: str, msg: Any) -> None:
        """真实模式输入通道：把消息交给 ObservationAssembler。

        ``joint_states`` 通道触发组装并把结果存为最新帧。
        """
        handler = self.CHANNEL_HANDLERS.get(channel)
        if handler is None:
            return
        result = getattr(self._assembler, handler)(msg)
        if channel == "joint_states":
            self._latest = result
            self.frames_produced += 1

    # ------------------------------------------------------------ 输出
    def outputs(self) -> dict:
        """最新帧的 7 通道输出。缺失字段为 None（调用方跳过发送）。"""
        obs = self._latest
        if obs is None:
            return {}
        return {
            "state": build_state_25(obs),
            "tcp_pose": (
                np.asarray(obs.tcp_pose, dtype=np.float32)
                if obs.tcp_pose is not None
                else None
            ),
            "joint_current": (
                np.asarray(obs.joint_current, dtype=np.float32)
                if obs.joint_current is not None
                else None
            ),
            "joint_velocity": (
                np.asarray(obs.joint_velocity, dtype=np.float32)
                if obs.joint_velocity is not None
                else None
            ),
            "status": obs.arm_status,
            "timestamp": obs.timestamp,
            "valid": bool(obs.valid),
        }

    @property
    def latest_observation(self) -> Optional[Observation]:
        return self._latest


def _env_inputs() -> dict:
    return {
        "tick": "dora/timer/millis/33",
        "joint_states": "/joint_states (未来 ROS2 relay)",
        "udp_joint_speed": "/rm_driver/udp_joint_speed (未来)",
        "udp_joint_current": "/rm_driver/udp_joint_current (未来)",
        "udp_arm_position": "/rm_driver/udp_arm_position (未来)",
        "udp_joint_en_flag": "/rm_driver/udp_joint_en_flag (未来)",
        "udp_rm_err": "/rm_driver/udp_rm_err (未来)",
        "udp_joint_error_code": "/rm_driver/udp_joint_error_code (未来)",
        "udp_arm_current_status": "/rm_driver/udp_arm_current_status (未来, 可选)",
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="rml63_state_dora_node")
    parser.add_argument("--describe", action="store_true", help="打印节点设计并退出（不需要 dora）")
    parser.add_argument("--check", action="store_true", help="离线自检并退出（不需要 dora）")
    args = parser.parse_args()

    if args.describe:
        print(describe_node("rml63_state", _env_inputs(), list(STATE_OUTPUTS)))
        return 0
    if args.check:
        core = StateNodeCore(input_mode="mock")
        out = core.on_tick()
        state = out.get("state")
        print(
            "rml63_state check: mock 帧输出 ok; state="
            + (f"float32[{list(state.shape)}]" if state is not None else "None")
            + f" valid={out.get('valid')} timestamp={out.get('timestamp')}"
        )
        return 0

    if not dora_available():
        print(
            "错误: dora-rs 未安装，无法作为 Dora 节点运行。\n"
            "离线验证请使用: python3 -m rml63_bridge.dora_nodes.simulate_dataflow\n"
            "或本脚本的 --describe / --check 模式。",
            file=sys.stderr,
        )
        return 1

    from dora import Node  # noqa: PLC0415（惰性导入）

    node = Node()
    core = StateNodeCore(
        input_mode=os.environ.get("STATE_INPUT_MODE", "mock"),
        mock_fps=float(os.environ.get("STATE_MOCK_FPS", str(DEFAULT_FPS))),
        mock_invalid_every=int(
            os.environ.get("STATE_MOCK_INVALID_EVERY", str(DEFAULT_INVALID_EVERY))
        ),
        stale_timeout=float(os.environ.get("STATE_STALE_TIMEOUT", "0.1")),
        require_all_enabled=os.environ.get("STATE_REQUIRE_ALL_ENABLED", "true").lower()
        == "true",
    )
    print(f"[rml63_state] input_mode={core.input_mode}", flush=True)

    for event in node:
        etype = event["type"]
        if etype == "STOP":
            break
        if etype != "INPUT":
            continue
        eid = event["id"]
        value = value_to_py(event.get("value"))
        if eid == "tick":
            send_named_outputs(node, core.on_tick(), STATE_OUTPUTS)
        elif eid in core.CHANNEL_HANDLERS:
            core.on_message(eid, value)
            if eid == "joint_states":
                send_named_outputs(node, core.outputs(), STATE_OUTPUTS)

    print(f"[rml63_state] stopped (frames={core.frames_produced})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
