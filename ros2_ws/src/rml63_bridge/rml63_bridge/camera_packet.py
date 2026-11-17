"""D435i offline camera data structure and mock generator.

No ROS2, no pyrealsense2, no real camera. This module defines the SEPARATE
camera data path:

    CameraPacket (images + IMU)  !=  Observation (25-dim RML63 state)

Images are NOT merged into the RML63 Observation; they are saved separately
or shipped on an independent image channel. Metadata contains only
shape/dtype summaries, never the raw arrays.

Missing data is always None and is never replaced by zeros. The real
D435i IMU topics must be verified after connecting the physical camera
(see README "D435i 相机预留输入").
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Optional

import numpy as np

DEFAULT_CAMERA_NAME = "d435i"
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 30.0
MOCK_CAMERA_INVALID_REASON = "mock: simulated camera frame drop (offline)"


@dataclass
class CameraPacket:
    """One camera frame from the (reserved) D435i input path."""

    timestamp: Optional[float] = None  # seconds
    camera_name: str = DEFAULT_CAMERA_NAME
    rgb: Any = None  # np.ndarray HxWx3, dtype=uint8, else None
    depth: Any = None  # np.ndarray HxW, dtype=uint16, else None
    aligned_depth: Any = None  # np.ndarray HxW, dtype=uint16, else None
    gyro: Optional[list] = None  # [3] rad/s
    accel: Optional[list] = None  # [3] m/s^2
    frame_id: Optional[int] = None
    valid: bool = False
    invalid_reasons: list = field(default_factory=list)


def _image_meta(arr: Any) -> dict:
    """Summarize one image as shape/dtype only (never the raw pixels)."""
    if arr is None:
        return {"present": False, "shape": None, "dtype": None}
    return {"present": True, "shape": list(arr.shape), "dtype": str(arr.dtype)}


def camera_packet_to_metadata(packet: CameraPacket) -> dict:
    """Metadata-only dict of a CameraPacket.

    Image arrays are summarized as shape/dtype — the raw pixel arrays are
    deliberately NOT included. gyro/accel (3 floats each) are small enough
    to be carried as plain lists.
    """
    return {
        "timestamp": packet.timestamp,
        "camera_name": packet.camera_name,
        "frame_id": packet.frame_id,
        "rgb": _image_meta(packet.rgb),
        "depth": _image_meta(packet.depth),
        "aligned_depth": _image_meta(packet.aligned_depth),
        "gyro": None if packet.gyro is None else [float(v) for v in packet.gyro],
        "accel": None if packet.accel is None else [float(v) for v in packet.accel],
        "valid": packet.valid,
        "invalid_reasons": list(packet.invalid_reasons),
    }


def validate_camera_packet(packet: CameraPacket, require_imu: bool = False) -> CameraPacket:
    """Return a copy with valid / invalid_reasons computed.

    Rules:
      - timestamp, rgb, depth are required (missing -> invalid);
      - aligned_depth is optional (alignment can be disabled on the camera);
      - gyro/accel are required only when require_imu=True; when present
        they must have length 3;
      - present arrays must match the documented shape/dtype.

    Missing data stays None in the returned packet — never zero-filled.
    """
    reasons: list = []

    if packet.timestamp is None:
        reasons.append("timestamp: 缺失")

    if packet.rgb is None:
        reasons.append("rgb: 缺失")
    elif not (
        isinstance(packet.rgb, np.ndarray)
        and packet.rgb.ndim == 3
        and packet.rgb.shape[2] == 3
        and packet.rgb.dtype == np.uint8
    ):
        reasons.append("rgb: 形状/类型错误 (需 HxWx3 uint8)")

    if packet.depth is None:
        reasons.append("depth: 缺失")
    elif not (
        isinstance(packet.depth, np.ndarray)
        and packet.depth.ndim == 2
        and packet.depth.dtype == np.uint16
    ):
        reasons.append("depth: 形状/类型错误 (需 HxW uint16)")

    if packet.aligned_depth is not None and not (
        isinstance(packet.aligned_depth, np.ndarray)
        and packet.aligned_depth.ndim == 2
        and packet.aligned_depth.dtype == np.uint16
    ):
        reasons.append("aligned_depth: 形状/类型错误 (需 HxW uint16)")

    if packet.gyro is not None and len(packet.gyro) != 3:
        reasons.append("gyro: 长度错误 (需 3 元素 rad/s)")
    if packet.accel is not None and len(packet.accel) != 3:
        reasons.append("accel: 长度错误 (需 3 元素 m/s^2)")
    if require_imu:
        if packet.gyro is None:
            reasons.append("gyro: 缺失 (require_imu)")
        if packet.accel is None:
            reasons.append("accel: 缺失 (require_imu)")

    return replace(packet, valid=not reasons, invalid_reasons=reasons)


def make_mock_camera_packet(
    index: int = 0,
    fps: float = DEFAULT_FPS,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    invalid_every: int = 5,
    include_imu: bool = True,
) -> CameraPacket:
    """One deterministic offline D435i frame (640x480 by default).

    timestamp = index / fps (default 30 Hz). Every ``invalid_every``-th
    frame (index > 0) simulates a full frame drop: rgb/depth/aligned_depth/
    gyro/accel are all None and valid=False. invalid_every <= 0 disables
    invalid frames. include_imu=False drops gyro/accel (for testing the
    require_imu validation path).
    """
    timestamp = index / fps
    is_invalid = invalid_every > 0 and index > 0 and index % invalid_every == 0
    if is_invalid:
        return CameraPacket(
            timestamp=timestamp,
            frame_id=index,
            valid=False,
            invalid_reasons=[MOCK_CAMERA_INVALID_REASON],
        )

    yy, xx = np.mgrid[0:height, 0:width]
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = ((xx * 255) // max(width - 1, 1)).astype(np.uint8)
    rgb[..., 1] = ((yy * 255) // max(height - 1, 1)).astype(np.uint8)
    rgb[..., 2] = ((xx + yy + 10 * index) % 256).astype(np.uint8)

    depth = (500 + 200 * np.sin(xx / 30.0 + index * 0.05) + 10 * (yy % 30)).astype(np.uint16)
    aligned_depth = (depth + 5).astype(np.uint16)  # deterministic tiny offset

    packet = CameraPacket(
        timestamp=timestamp,
        rgb=rgb,
        depth=depth,
        aligned_depth=aligned_depth,
        gyro=[0.01 * math.sin(index * 0.5), 0.02 * math.cos(index * 0.3), 0.005 * math.sin(index * 0.7)]
        if include_imu
        else None,
        accel=[0.01 * math.sin(index * 0.2), 0.02 * math.cos(index * 0.2), 9.81 + 0.05 * math.sin(index * 0.1)]
        if include_imu
        else None,
        frame_id=index,
    )
    return validate_camera_packet(packet)
