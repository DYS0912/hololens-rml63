"""Dora 节点 2：D435i 相机节点（d435i_camera_dora_node）。

未来输入（ROS2 话题，经 relay 进入 Dora 通道）：
    color          <- /camera/camera/color/image_raw（帧触发 + 时间戳来源）
    depth          <- /camera/camera/depth/image_rect_raw
    aligned_depth  <- /camera/camera/aligned_depth_to_color/image_raw
    gyro           <- /camera/camera/gyro/sample（需真机验证是否启用）
    accel          <- /camera/camera/accel/sample（需真机验证是否启用）
    imu            <- /camera/camera/imu（合成 IMU 消息，需真机验证）

输出（8 个通道）：
    rgb               uint8 HxWx3（numpy 数组，dora 传输时 ravel 为数值数组）
    depth             uint16 HxW
    aligned_depth     uint16 HxW
    gyro              float32[3]（缺失为 None，不发送）
    accel             float32[3]（缺失为 None，不发送）
    camera_timestamp  float
    frame_id          int
    valid             bool

图像约束：只以 numpy / pyarrow 数值数组传输，**绝不进入 JSON 字符串**。
离线模式（CAMERA_INPUT_MODE=mock，默认）：tick 触发时用
make_mock_camera_packet 生成确定性帧（含 640x480 RGB + 16-bit 深度 +
模拟 IMU）。不依赖 OpenCV、pyrealsense2 与任何相机硬件。
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

import numpy as np

# 注意: dora 以普通脚本方式运行本文件（python3 <path>），必须使用绝对导入；
# dataflow env 中的 PYTHONPATH 指向 rml63_bridge 包根目录。
from rml63_bridge.camera_packet import (
    DEFAULT_FPS,
    CameraPacket,
    make_mock_camera_packet,
    validate_camera_packet,
)
from rml63_bridge.dora_nodes.dora_compat import (
    dora_available,
    describe_node,
    send_named_outputs,
    value_to_py,
)

CAMERA_OUTPUTS = (
    "rgb", "depth", "aligned_depth", "gyro", "accel",
    "camera_timestamp", "frame_id", "valid",
)

# 未来 ROS2 输入通道 -> CameraPacket 字段（离线 mock 模式不使用）
_CHANNEL_FIELDS = {
    "color": "rgb",
    "depth": "depth",
    "aligned_depth": "aligned_depth",
    "gyro": "gyro",
    "accel": "accel",
    "imu": "imu",  # 合成 IMU：包含 gyro+accel，需真机验证
}


class CameraNodeCore:
    """相机节点核心逻辑（无 dora、无相机硬件，可离线测试）。

    帧触发通道为 ``color``（带 header.stamp）；其他通道按到达缓冲，
    触发时组装 CameraPacket 并校验。
    """

    def __init__(
        self,
        input_mode: str = "mock",
        mock_fps: float = DEFAULT_FPS,
        mock_invalid_every: int = 5,
        camera_name: str = "d435i",
        require_imu: bool = False,  # D435i IMU 话题需真机验证后再开启
    ):
        if input_mode not in ("mock", "ros2"):
            raise ValueError(f"input_mode 仅支持 mock/ros2，收到 {input_mode!r}")
        self.input_mode = input_mode
        self.mock_fps = mock_fps
        self.mock_invalid_every = mock_invalid_every
        self.camera_name = camera_name
        self.require_imu = require_imu
        self._mock_index = 0
        self._buffers: dict = {}
        self._latest: Optional[CameraPacket] = None
        self.frames_produced = 0

    # ------------------------------------------------------------ 输入
    def on_tick(self) -> dict:
        """tick 心跳。mock 模式：生成下一帧模拟 CameraPacket 并输出。"""
        if self.input_mode != "mock":
            return {}
        packet = validate_camera_packet(
            make_mock_camera_packet(
                self._mock_index,
                fps=self.mock_fps,
                invalid_every=self.mock_invalid_every,
            ),
            require_imu=self.require_imu,
        )
        self._mock_index += 1
        self._latest = packet
        self.frames_produced += 1
        return self.outputs()

    def on_message(self, channel: str, msg: Any) -> None:
        """真实模式输入通道：缓冲消息；``color`` 触发组装。"""
        if channel not in _CHANNEL_FIELDS:
            return
        self._buffers[channel] = msg
        if channel != "color":
            return
        # 触发组装（timestamp 取自 color 消息 header.stamp；离线 mock 不经过此路径）
        stamp = None
        header = getattr(msg, "header", None)
        if header is not None:
            stamp = header.stamp.sec + header.stamp.nanosec * 1e-9
        self._assemble(stamp)

    def _assemble(self, stamp: Optional[float]) -> None:
        packet = CameraPacket(timestamp=stamp, camera_name=self.camera_name)
        for channel, field in _CHANNEL_FIELDS.items():
            if field == "imu" or channel not in self._buffers:
                continue
            setattr(packet, field, self._buffers.get(channel))
        imu = self._buffers.get("imu")
        if imu is not None:
            packet.gyro = getattr(imu, "gyro", None)
            packet.accel = getattr(imu, "accel", None)
        self._latest = validate_camera_packet(packet, require_imu=self.require_imu)
        self.frames_produced += 1

    # ------------------------------------------------------------ 输出
    def outputs(self) -> dict:
        """最新帧的 8 通道输出。图像为 numpy 数组；缺失为 None（不发送）。"""
        packet = self._latest
        if packet is None:
            return {}
        return {
            "rgb": packet.rgb,
            "depth": packet.depth,
            "aligned_depth": packet.aligned_depth,
            "gyro": (
                np.asarray(packet.gyro, dtype=np.float32)
                if packet.gyro is not None
                else None
            ),
            "accel": (
                np.asarray(packet.accel, dtype=np.float32)
                if packet.accel is not None
                else None
            ),
            "camera_timestamp": packet.timestamp,
            "frame_id": packet.frame_id,
            "valid": bool(packet.valid),
        }

    @property
    def latest_packet(self) -> Optional[CameraPacket]:
        return self._latest


def _env_inputs() -> dict:
    return {
        "tick": "dora/timer/millis/33",
        "color": "/camera/camera/color/image_raw (未来 ROS2 relay)",
        "depth": "/camera/camera/depth/image_rect_raw (未来)",
        "aligned_depth": "/camera/camera/aligned_depth_to_color/image_raw (未来)",
        "gyro": "/camera/camera/gyro/sample (未来, 需真机验证)",
        "accel": "/camera/camera/accel/sample (未来, 需真机验证)",
        "imu": "/camera/camera/imu (未来, 需真机验证)",
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="d435i_camera_dora_node")
    parser.add_argument("--describe", action="store_true", help="打印节点设计并退出（不需要 dora）")
    parser.add_argument("--check", action="store_true", help="离线自检并退出（不需要 dora）")
    args = parser.parse_args()

    if args.describe:
        print(describe_node("d435i_camera", _env_inputs(), list(CAMERA_OUTPUTS)))
        return 0
    if args.check:
        core = CameraNodeCore(input_mode="mock")
        out = core.on_tick()
        rgb, depth = out.get("rgb"), out.get("depth")
        print(
            "d435i_camera check: mock 帧输出 ok; "
            f"rgb={None if rgb is None else list(rgb.shape)}"
            f" depth={None if depth is None else list(depth.shape)}"
            f" valid={out.get('valid')}"
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
    core = CameraNodeCore(
        input_mode=os.environ.get("CAMERA_INPUT_MODE", "mock"),
        mock_fps=float(os.environ.get("CAMERA_MOCK_FPS", str(DEFAULT_FPS))),
        mock_invalid_every=int(os.environ.get("CAMERA_MOCK_INVALID_EVERY", "5")),
        camera_name=os.environ.get("CAMERA_NAME", "d435i"),
        require_imu=os.environ.get("CAMERA_REQUIRE_IMU", "false").lower() == "true",
    )
    print(f"[d435i_camera] input_mode={core.input_mode}", flush=True)

    for event in node:
        etype = event["type"]
        if etype == "STOP":
            break
        if etype != "INPUT":
            continue
        eid = event["id"]
        if eid == "tick":
            send_named_outputs(node, core.on_tick(), CAMERA_OUTPUTS)
        elif eid in _CHANNEL_FIELDS:
            core.on_message(eid, value_to_py(event.get("value")))
            if eid == "color":
                send_named_outputs(node, core.outputs(), CAMERA_OUTPUTS)

    print(f"[d435i_camera] stopped (frames={core.frames_produced})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
