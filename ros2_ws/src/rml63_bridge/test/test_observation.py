"""Unit tests for the RML63 observation assembler (pure logic, no ROS2 node).

No rclpy node is ever instantiated here: tests run entirely on mock message
objects so that nothing is published, subscribed, or commanded.

Run with the SDK sourced:
    python3 -m pytest test/ -v
"""

import math
import unittest

from geometry_msgs.msg import Pose
from rm_ros_interfaces.msg import (
    Armcurrentstatus,
    Jointcurrent,
    Jointenflag,
    Jointerrorcode,
    Jointspeed,
    Rmerr,
    Rmplusstate,
)
from sensor_msgs.msg import JointState

from rml63_bridge.observation import (
    EMERGENCY_STOP_STATUS,
    ObservationAssembler,
    rpm_to_rad_s,
)

JOINTS = [f"joint{i}" for i in range(1, 7)]


def make_joint_state(position, stamp_sec=10.0):
    msg = JointState()
    msg.header.stamp.sec = int(stamp_sec)
    msg.header.stamp.nanosec = int((stamp_sec % 1) * 1e9)
    msg.name = JOINTS[: len(position)]
    msg.position = [float(p) for p in position]
    return msg


def make_speed(rpm_values):
    msg = Jointspeed()
    msg.joint_speed = [float(v) for v in rpm_values]
    return msg


def make_current(milliamp_values):
    msg = Jointcurrent()
    msg.joint_current = [float(v) for v in milliamp_values]
    return msg


def make_pose(x=0.1, y=0.2, z=0.3, qw=1.0, qx=0.0, qy=0.0, qz=0.0):
    msg = Pose()
    msg.position.x, msg.position.y, msg.position.z = x, y, z
    msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z = qw, qx, qy, qz
    return msg


def make_en_flag(flags):
    msg = Jointenflag()
    msg.joint_en_flag = [bool(f) for f in flags]
    return msg


def make_rm_err(err_len=0, err=None):
    msg = Rmerr()
    msg.err_len = err_len
    msg.err = list(err or [])
    return msg


def make_joint_error(codes):
    msg = Jointerrorcode()
    msg.joint_error = [int(c) for c in codes]
    msg.dof = len(codes)
    return msg


def make_arm_status(status):
    msg = Armcurrentstatus()
    msg.arm_current_status = int(status)
    return msg


def make_tool_state():
    msg = Rmplusstate()
    msg.sys_state = 1
    return msg


class FakeClock:
    """Mutable clock: tests advance time to simulate staleness."""

    def __init__(self, t=10.0):
        self.t = t

    def __call__(self):
        return self.t


def feed_all(assembler, position=None, rpm=None, current=None, pose=None,
             flags=None, rm_err=None, joint_err=None, status=None, tool=None):
    """Feed a full fresh frame; returns the observation from on_joint_states."""
    position = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] if position is None else position
    rpm = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] if rpm is None else rpm
    current = [10.0] * 6 if current is None else current
    pose = make_pose() if pose is None else pose
    flags = [True] * 6 if flags is None else flags
    rm_err = make_rm_err() if rm_err is None else rm_err
    joint_err = make_joint_error([0] * 6) if joint_err is None else joint_err

    assembler.on_udp_joint_speed(make_speed(rpm))
    assembler.on_udp_joint_current(make_current(current))
    assembler.on_udp_arm_position(pose)
    assembler.on_udp_joint_en_flag(make_en_flag(flags))
    assembler.on_udp_rm_err(rm_err)
    assembler.on_udp_joint_error_code(joint_err)
    if status is not None:
        assembler.on_udp_arm_current_status(make_arm_status(status))
    if tool is not None:
        assembler.on_udp_rm_plus_state(tool)
    return assembler.on_joint_states(make_joint_state(position))


class TestRpmConversion(unittest.TestCase):
    def test_rpm_to_rad_s(self):
        result = rpm_to_rad_s([0.0, 60.0, 30.0, -60.0])
        self.assertEqual(result[0], 0.0)
        self.assertAlmostEqual(result[1], 2.0 * math.pi)
        self.assertAlmostEqual(result[2], math.pi)
        self.assertAlmostEqual(result[3], -2.0 * math.pi)


class TestValidFrames(unittest.TestCase):
    def test_normal_frame_valid(self):
        a = ObservationAssembler(now=FakeClock())
        obs = feed_all(a, rpm=[60.0] * 6, current=[12.5] * 6)
        self.assertTrue(obs.valid, obs.invalid_reasons)
        self.assertEqual(len(obs.joint_position), 6)
        self.assertEqual(len(obs.joint_velocity), 6)
        self.assertAlmostEqual(obs.joint_velocity[0], 2.0 * math.pi)  # RPM -> rad/s
        self.assertEqual(obs.joint_current[0], 12.5)  # mA, unconverted
        # tcp_pose order: [x, y, z, qw, qx, qy, qz]
        self.assertEqual(obs.tcp_pose, [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0])
        self.assertEqual(obs.arm_status, None)  # optional topic absent -> ok
        self.assertEqual(obs.joint_enabled, [True] * 6)
        self.assertAlmostEqual(obs.timestamp, 10.0)

    def test_zeros_are_valid_data_not_errors(self):
        """All-zero readings with healthy codes must be valid=True:
        zeros are legitimate data and must not be treated as missing."""
        a = ObservationAssembler(now=FakeClock())
        obs = feed_all(a, position=[0.0] * 6, rpm=[0.0] * 6, current=[0.0] * 6)
        self.assertTrue(obs.valid, obs.invalid_reasons)
        self.assertEqual(obs.joint_position, [0.0] * 6)
        self.assertEqual(obs.joint_current, [0.0] * 6)

    def test_optional_arm_status_and_tool_carried(self):
        a = ObservationAssembler(now=FakeClock())
        tool = make_tool_state()
        obs = feed_all(a, status=0, tool=tool)
        self.assertTrue(obs.valid, obs.invalid_reasons)
        self.assertEqual(obs.arm_status, 0)
        self.assertIs(obs.tool_state, tool)


class TestInvalidFrames(unittest.TestCase):
    def test_short_position_invalid(self):
        a = ObservationAssembler(now=FakeClock())
        obs = feed_all(a, position=[0.0] * 5)  # 5 joints instead of 6
        self.assertFalse(obs.valid)
        self.assertIsNone(obs.joint_position)  # not zero-filled
        self.assertTrue(any("position" in r for r in obs.invalid_reasons))

    def test_short_velocity_invalid(self):
        a = ObservationAssembler(now=FakeClock())
        obs = feed_all(a, rpm=[0.0] * 5)
        self.assertFalse(obs.valid)
        self.assertIsNone(obs.joint_velocity)

    def test_short_current_invalid(self):
        a = ObservationAssembler(now=FakeClock())
        obs = feed_all(a, current=[0.0] * 5)
        self.assertFalse(obs.valid)
        self.assertIsNone(obs.joint_current)

    def test_missing_topic_invalid_and_not_zero_filled(self):
        """A topic that never arrived must yield valid=False with field=None,
        never a fabricated zero array."""
        a = ObservationAssembler(now=FakeClock())
        a.on_udp_joint_speed(make_speed([0.0] * 6))
        a.on_udp_arm_position(make_pose())
        a.on_udp_joint_en_flag(make_en_flag([True] * 6))
        a.on_udp_rm_err(make_rm_err())
        a.on_udp_joint_error_code(make_joint_error([0] * 6))
        # joint_current intentionally never fed
        obs = a.on_joint_states(make_joint_state([0.1] * 6))
        self.assertFalse(obs.valid)
        self.assertIsNone(obs.joint_current)
        self.assertTrue(any("joint_current" in r for r in obs.invalid_reasons))

    def test_timeout_invalid(self):
        clock = FakeClock()
        a = ObservationAssembler(stale_timeout=0.1, now=clock)
        feed_all(a)  # everything fresh at t=10.0 -> valid
        clock.t = 10.5  # advance 0.5 s: udp topics now stale
        obs = a.on_joint_states(make_joint_state([0.1] * 6, stamp_sec=10.5))
        self.assertFalse(obs.valid)
        self.assertTrue(any("stale" in r for r in obs.invalid_reasons))

    def test_joint_states_stamp_timeout_invalid(self):
        clock = FakeClock()
        a = ObservationAssembler(stale_timeout=0.1, now=clock)
        feed_all(a)  # fresh at t=10.0
        clock.t = 10.5
        # re-feed udp topics at t=10.5 so only the stamp is old
        a.on_udp_joint_speed(make_speed([0.0] * 6))
        a.on_udp_joint_current(make_current([0.0] * 6))
        a.on_udp_arm_position(make_pose())
        a.on_udp_joint_en_flag(make_en_flag([True] * 6))
        a.on_udp_rm_err(make_rm_err())
        a.on_udp_joint_error_code(make_joint_error([0] * 6))
        obs = a.on_joint_states(make_joint_state([0.1] * 6, stamp_sec=9.0))
        self.assertFalse(obs.valid)
        self.assertTrue(any("时间戳超时" in r for r in obs.invalid_reasons))

    def test_joint_error_code_nonzero_invalid(self):
        a = ObservationAssembler(now=FakeClock())
        obs = feed_all(a, joint_err=make_joint_error([0, 0, 1, 0, 0, 0]))
        self.assertFalse(obs.valid)
        self.assertTrue(any("joint_error_code" in r for r in obs.invalid_reasons))

    def test_rm_err_invalid(self):
        a = ObservationAssembler(now=FakeClock())
        obs = feed_all(a, rm_err=make_rm_err(err_len=1, err=[100]))
        self.assertFalse(obs.valid)
        self.assertTrue(any("rm_err" in r for r in obs.invalid_reasons))

    def test_emergency_stop_invalid(self):
        a = ObservationAssembler(now=FakeClock())
        obs = feed_all(a, status=EMERGENCY_STOP_STATUS)
        self.assertFalse(obs.valid)
        self.assertTrue(any("急停" in r for r in obs.invalid_reasons))

    def test_disabled_joint_invalid(self):
        a = ObservationAssembler(now=FakeClock())
        obs = feed_all(a, flags=[True, True, True, True, True, False])
        self.assertFalse(obs.valid)
        self.assertTrue(any("使能" in r for r in obs.invalid_reasons))

    def test_no_joint_states_yet_invalid(self):
        a = ObservationAssembler(now=FakeClock())
        a.on_udp_joint_speed(make_speed([0.0] * 6))
        # first frame can only be emitted by on_joint_states; simulate via internal
        obs = a._assemble()
        self.assertFalse(obs.valid)


if __name__ == "__main__":
    unittest.main()
