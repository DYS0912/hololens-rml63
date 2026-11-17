"""Offline multi-sensor synchronization: align RML63 Observation and D435i
CameraPacket by timestamp into unified training frames.

Rules (offline batch mode):
  - the RML63 observation timestamp is the frame's primary time;
  - for each observation the CameraPacket with the smallest |time_delta|
    inside max_time_delta (default 50 ms) is selected;
  - a camera frame whose timestamp lies more than max_time_delta in the
    FUTURE is never used ("未来过远"); on ties the past frame wins;
  - missing observation, missing camera match (when require_camera=True),
    invalid source frames, or IMU missing (when require_imu=True) all
    produce valid=False with reasons;
  - missing data stays None — never zero-filled; the original
    Observation and CameraPacket objects are kept on the frame;
  - metadata summaries never carry raw RGB/depth arrays, only shape /
    dtype / frame_id.

No ROS2, no hardware: this module only consumes in-memory streams.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional

from .camera_packet import CameraPacket, camera_packet_to_metadata, validate_camera_packet
from .observation import Observation

DEFAULT_MAX_TIME_DELTA = 0.05  # seconds (50 ms)
_TIME_EPS = 1e-9  # 浮点误差容差：恰好落在窗口边界上的帧不应被误拒


@dataclass
class SynchronizedFrame:
    """One unified offline training frame (observation x camera)."""

    timestamp: Optional[float] = None  # 主时间 = observation timestamp
    observation: Optional[Observation] = None  # 原始对象，缺失为 None
    camera_packet: Optional[CameraPacket] = None  # 原始对象，缺失为 None
    observation_timestamp: Optional[float] = None
    camera_timestamp: Optional[float] = None
    time_delta: Optional[float] = None  # camera_ts - obs_ts（秒）
    valid: bool = False
    invalid_reasons: list = field(default_factory=list)


def _pick_camera_packet(
    obs_ts: Optional[float], packets: List[CameraPacket], max_time_delta: float
) -> Optional[CameraPacket]:
    """Nearest camera packet with |dt| <= max_time_delta.

    A future frame (dt > 0) beyond +max_time_delta is rejected outright;
    past frames beyond -max_time_delta are rejected as well. On equal
    |dt| the past (non-future) frame is preferred.
    """
    if obs_ts is None:
        return None
    best: Optional[CameraPacket] = None
    best_abs: Optional[float] = None
    for p in packets:
        dt = p.timestamp - obs_ts
        if dt > max_time_delta + _TIME_EPS:
            continue  # 未来过远，不允许使用
        adt = abs(dt)
        if adt > max_time_delta + _TIME_EPS:
            continue  # 过去过远
        if best_abs is None or adt < best_abs or (math.isclose(adt, best_abs) and dt < 0):
            best = p
            best_abs = adt
    return best


def synchronize_observations(
    observations: Iterable[Optional[Observation]],
    camera_packets: Iterable[Optional[CameraPacket]],
    max_time_delta: float = DEFAULT_MAX_TIME_DELTA,
    require_camera: bool = True,
    require_imu: bool = False,
) -> List[SynchronizedFrame]:
    """Align observations and camera packets into SynchronizedFrames.

    - observations may contain None entries (dropped RML63 frames) -> invalid;
    - camera_packets may contain None entries (dropped camera frames); they
      are skipped during matching;
    - require_camera=False relaxes the camera requirement: frames without a
      camera match stay valid if the observation itself is valid;
    - require_imu=True additionally demands gyro/accel on the matched
      camera packet (D435i IMU topics must be verified on the real camera).
    """
    packets = [p for p in camera_packets if p is not None and p.timestamp is not None]
    frames: List[SynchronizedFrame] = []

    for obs in observations:
        frame = SynchronizedFrame()
        reasons: list = []

        if obs is None:
            reasons.append("observation: 缺失")
            frame.invalid_reasons = reasons
            frames.append(frame)
            continue

        frame.observation = obs
        frame.timestamp = obs.timestamp
        frame.observation_timestamp = obs.timestamp

        if obs.timestamp is None:
            reasons.append("observation: timestamp 缺失")
        if not obs.valid:
            reasons.append("observation: 原始帧无效")

        matched = _pick_camera_packet(obs.timestamp, packets, max_time_delta)
        if matched is None:
            if require_camera:
                reasons.append("camera: 无 max_time_delta 内的相机帧")
        else:
            frame.camera_packet = matched
            frame.camera_timestamp = matched.timestamp
            frame.time_delta = matched.timestamp - obs.timestamp
            checked = validate_camera_packet(matched, require_imu=require_imu)
            if not checked.valid:
                for r in checked.invalid_reasons:
                    reasons.append(f"camera: {r}")

        frame.valid = not reasons
        frame.invalid_reasons = reasons
        frames.append(frame)

    return frames


def _observation_meta(obs: Optional[Observation]) -> Optional[dict]:
    """Small-array observation summary (never contains image arrays)."""
    if obs is None:
        return None
    return {
        "timestamp": obs.timestamp,
        "joint_position": obs.joint_position,
        "joint_velocity": obs.joint_velocity,
        "joint_current": obs.joint_current,
        "tcp_pose": obs.tcp_pose,
        "arm_status": obs.arm_status,
        "joint_enabled": obs.joint_enabled,
        "tool_state": (
            None
            if obs.tool_state is None
            else {"present": True, "type": type(obs.tool_state).__name__}
        ),
        "valid": obs.valid,
        "invalid_reasons": list(obs.invalid_reasons),
    }


def synchronized_frame_to_metadata(frame: SynchronizedFrame) -> dict:
    """Metadata-only dict of a SynchronizedFrame.

    Observation carries its small numeric arrays; the camera side uses
    camera_packet_to_metadata (shape/dtype/frame_id only) — raw RGB/depth
    pixel arrays are NEVER serialized here.
    """
    return {
        "timestamp": frame.timestamp,
        "observation_timestamp": frame.observation_timestamp,
        "camera_timestamp": frame.camera_timestamp,
        "time_delta": frame.time_delta,
        "observation": _observation_meta(frame.observation),
        "camera": (
            camera_packet_to_metadata(frame.camera_packet)
            if frame.camera_packet is not None
            else None
        ),
        "valid": frame.valid,
        "invalid_reasons": list(frame.invalid_reasons),
    }
