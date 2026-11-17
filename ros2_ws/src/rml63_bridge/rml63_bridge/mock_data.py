"""Offline mock RML63 state generator.

Generates deterministic Observation frames WITHOUT ROS2, without the real
arm, and without any camera. Used to exercise the full offline pipeline:

    mock state -> ObservationAssembler-shaped data -> ZMQ PUB -> SUB -> LeRobot

Frame contents and units mirror the real rm_driver topics:
    joint_position : [6] rad        (as in /joint_states)
    joint_velocity : [6] rad/s      (already converted from RPM)
    joint_current  : [6] mA         (official unit, unconverted)
    tcp_pose       : [7] [x, y, z, qw, qx, qy, qz]

timestamp increments at 1/fps (default 30 Hz). The stream deliberately
contains both valid=True and valid=False frames (every ``invalid_every``-th
frame is invalid) so downstream filtering can be tested.
"""

from __future__ import annotations

import math
from typing import Iterator

from .observation import Observation, ObservationAssembler

DEFAULT_FPS = 30.0
DEFAULT_INVALID_EVERY = 5

MOCK_INVALID_REASON = "mock: simulated joint error (offline)"


def _quaternion_from_axis_angle(angle: float) -> list:
    """Unit quaternion around the z axis: [qw, qx, qy, qz]."""
    half = angle / 2.0
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def make_mock_observation(
    index: int,
    fps: float = DEFAULT_FPS,
    invalid_every: int = DEFAULT_INVALID_EVERY,
    start_timestamp: float = 0.0,
) -> Observation:
    """One deterministic mock frame.

    index=0 is always valid; frames with index % invalid_every == 0
    (index > 0) are invalid. invalid_every <= 0 disables invalid frames.
    """
    timestamp = start_timestamp + index / fps

    joint_position = [
        0.5 * math.sin(0.1 * index + 0.7 * j) for j in range(6)
    ]  # rad
    joint_velocity = [
        0.3 * math.cos(0.1 * index + 0.7 * j) for j in range(6)
    ]  # rad/s
    joint_current = [
        800.0 + 200.0 * math.sin(0.13 * index + 1.3 * j) for j in range(6)
    ]  # mA

    # TCP pose: small deterministic motion, unit quaternion
    tcp_angle = 0.1 * math.sin(0.05 * index)
    x = 0.30 + 0.05 * math.sin(0.1 * index)
    y = 0.10 + 0.05 * math.cos(0.1 * index)
    z = 0.40 + 0.02 * math.sin(0.2 * index)
    tcp_pose = [x, y, z] + _quaternion_from_axis_angle(tcp_angle)

    is_invalid = invalid_every > 0 and index > 0 and index % invalid_every == 0

    return Observation(
        timestamp=timestamp,
        joint_position=joint_position,
        joint_velocity=joint_velocity,
        joint_current=None if is_invalid else joint_current,
        tcp_pose=tcp_pose,
        arm_status=0,
        joint_enabled=[True] * 6,
        tool_state=None,
        valid=not is_invalid,
        invalid_reasons=[MOCK_INVALID_REASON] if is_invalid else [],
    )


def mock_observation_stream(
    count: int,
    fps: float = DEFAULT_FPS,
    invalid_every: int = DEFAULT_INVALID_EVERY,
    start_timestamp: float = 0.0,
) -> Iterator[Observation]:
    """Yield ``count`` deterministic mock frames (mixed valid/invalid)."""
    for index in range(count):
        yield make_mock_observation(
            index, fps=fps, invalid_every=invalid_every, start_timestamp=start_timestamp
        )


def feed_mock_frame(
    assembler: ObservationAssembler,
    index: int,
    fps: float = DEFAULT_FPS,
    invalid_every: int = DEFAULT_INVALID_EVERY,
    stamp: float = 0.0,
) -> Observation:
    """Feed one mock frame through an ObservationAssembler using mock ROS2
    message objects (no rclpy, no nodes). This exercises the real
    assembler path of the offline pipeline:

        mock RML63 state -> ObservationAssembler -> Observation

    Feed order mirrors rm_driver's same-callback publish order: the
    un-stamped udp_* topics first, joint_states (the frame trigger) last.
    For invalid frames the joint_current topic is omitted and a non-zero
    joint error code is injected, so the assembler itself produces
    valid=False with its own reasons.
    """
    from geometry_msgs.msg import Pose
    from rm_ros_interfaces.msg import (
        Jointcurrent,
        Jointenflag,
        Jointerrorcode,
        Jointspeed,
        Rmerr,
    )
    from sensor_msgs.msg import JointState

    direct = make_mock_observation(index, fps=fps, invalid_every=invalid_every)
    is_invalid = not direct.valid
    stamp = stamp if stamp is not None else direct.timestamp

    # joint speed: assembler converts RPM -> rad/s, so feed the rad/s mock
    # values expressed back in RPM to round-trip them.
    rpm = [v * 60.0 / (2.0 * math.pi) for v in direct.joint_velocity]
    speed = Jointspeed()
    speed.joint_speed = rpm

    current = Jointcurrent()
    if direct.joint_current is not None:
        current.joint_current = direct.joint_current  # ROS2 setter rejects None

    pose = Pose()
    x, y, z, qw, qx, qy, qz = direct.tcp_pose
    pose.position.x, pose.position.y, pose.position.z = x, y, z
    pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z = (
        qw,
        qx,
        qy,
        qz,
    )

    en_flag = Jointenflag()
    en_flag.joint_en_flag = [True] * 6

    rm_err = Rmerr()
    rm_err.err_len = 0
    rm_err.err = []

    joint_err = Jointerrorcode()
    if is_invalid:
        joint_err.joint_error = [0, 0, 3, 0, 0, 0]  # simulated joint 3 error
    else:
        joint_err.joint_error = [0] * 6
    joint_err.dof = 6

    # udp_* topics first (un-stamped), joint_states last (frame trigger);
    # the invalid frame omits joint_current entirely, like a lost topic.
    assembler.on_udp_joint_speed(speed)
    if not is_invalid:
        assembler.on_udp_joint_current(current)
    assembler.on_udp_arm_position(pose)
    assembler.on_udp_joint_en_flag(en_flag)
    assembler.on_udp_rm_err(rm_err)
    assembler.on_udp_joint_error_code(joint_err)

    js = JointState()
    js.header.stamp.sec = int(stamp)
    js.header.stamp.nanosec = int((stamp % 1) * 1e9)
    js.name = [f"joint{i}" for i in range(1, 7)]
    js.position = direct.joint_position
    return assembler.on_joint_states(js)
