"""Observation dataclass and frame-assembly logic for the RML63 arm (bridge).

Pure Python (no rclpy runtime dependency in this module) so the whole
validity logic can be unit-tested with mock ROS2 message objects.

Timestamp alignment note (must be validated on the real machine):
    The rm_driver publishes ``/joint_states`` (which carries a header stamp)
    and the un-stamped ``/rm_driver/udp_*`` topics back-to-back inside the
    same UDP timer callback, from data parsed out of the same UDP packet.
    Treating the latest udp_* values as the frame that belongs to the
    current ``joint_states.header.stamp`` is therefore an APPROXIMATION of
    "same callback order" alignment. It has NOT been validated on real
    hardware: DDS reordering, UDP loss and callback jitter must be measured
    before this alignment can be trusted for training data.

    ``joint_states.header.stamp`` itself is the host receive time
    (``this->now()`` in rm_driver), not the arm-side sampling instant.

Zero values are legitimate data (e.g. a joint at its zero position or zero
current): the assembler NEVER uses zeros to mask missing/error data, and it
NEVER substitutes zeros for missing fields. Missing fields are None.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# Official rm_driver constants (rm_ros_interfaces docs):
JOINT_COUNT = 6
RPM_TO_RAD_S = 2.0 * math.pi / 60.0  # rad/s = rpm * 2*pi/60
EMERGENCY_STOP_STATUS = 9  # RM_STOP_E in Armcurrentstatus

# Topic keys used internally (never published, never commanded):
TOPIC_JOINT_STATES = "joint_states"
TOPIC_UDP_JOINT_SPEED = "udp_joint_speed"
TOPIC_UDP_JOINT_CURRENT = "udp_joint_current"
TOPIC_UDP_ARM_POSITION = "udp_arm_position"
TOPIC_UDP_JOINT_EN_FLAG = "udp_joint_en_flag"
TOPIC_UDP_RM_ERR = "udp_rm_err"
TOPIC_UDP_JOINT_ERROR_CODE = "udp_joint_error_code"
TOPIC_UDP_ARM_CURRENT_STATUS = "udp_arm_current_status"  # optional (gated in rm_driver by default)
TOPIC_UDP_RM_PLUS_STATE = "udp_rm_plus_state"  # optional (gated in rm_driver by default)

REQUIRED_TOPICS = (
    TOPIC_JOINT_STATES,
    TOPIC_UDP_JOINT_SPEED,
    TOPIC_UDP_JOINT_CURRENT,
    TOPIC_UDP_ARM_POSITION,
    TOPIC_UDP_JOINT_EN_FLAG,
    TOPIC_UDP_RM_ERR,
    TOPIC_UDP_JOINT_ERROR_CODE,
)
OPTIONAL_TOPICS = (
    TOPIC_UDP_ARM_CURRENT_STATUS,
    TOPIC_UDP_RM_PLUS_STATE,
)


def rpm_to_rad_s(rpm_values):
    """Convert joint speeds from RPM (official unit of udp_joint_speed) to rad/s."""
    return [float(v) * RPM_TO_RAD_S for v in rpm_values]


@dataclass
class Observation:
    """One aligned observation frame assembled from rm_driver state topics.

    All *_data fields are ``None`` when the corresponding data is missing,
    stale, or structurally wrong. They are NEVER zero-filled.
    """

    timestamp: Optional[float] = None  # joint_states.header.stamp (host receive time, seconds)
    joint_position: Optional[list] = None  # [6] rad (from /joint_states)
    joint_velocity: Optional[list] = None  # [6] rad/s (converted from RPM)
    joint_current: Optional[list] = None  # [6] mA (official unit, unconverted)
    tcp_pose: Optional[list] = None  # [7] = [x, y, z, qw, qx, qy, qz] (m + quaternion)
    arm_status: Optional[int] = None  # Armcurrentstatus code (9 = emergency stop)
    joint_enabled: Optional[list] = None  # [6] bool enable flags
    tool_state: Any = None  # raw Rmplusstate message if fresh, else None
    valid: bool = False
    invalid_reasons: list = field(default_factory=list)


class ObservationAssembler:
    """Collects rm_driver state messages and assembles validity-checked frames.

    Frames are emitted from ``on_joint_states`` (the only stamped message).
    A frame is valid=True only if ALL of the following hold:
      - joint position length == 6;
      - velocity length == 6 (converted RPM -> rad/s);
      - current length == 6;
      - no timeout: joint_states stamp and every required udp_* topic are
        fresher than ``stale_timeout``;
      - all joint error codes == 0 (length 6);
      - rm error empty (err_len == 0 and err array empty);
      - arm status, if present and fresh, is not the emergency-stop code;
      - joint enable flags length == 6 and (if ``require_all_enabled``) all True.

    Any violation produces valid=False with the specific reasons recorded.
    Missing data is NEVER replaced by zeros.
    """

    def __init__(
        self,
        stale_timeout: float = 0.1,
        require_all_enabled: bool = True,
        now: Optional[Callable[[], float]] = None,
    ):
        self.stale_timeout = stale_timeout
        self.require_all_enabled = require_all_enabled
        self._now = now if now is not None else time.time
        self._latest: dict = {}  # topic -> (message, local receive time)

    # --- message intake -------------------------------------------------
    def on_joint_states(self, msg) -> Observation:
        self._latest[TOPIC_JOINT_STATES] = (msg, self._now())
        return self._assemble()

    def on_udp_joint_speed(self, msg) -> None:
        self._latest[TOPIC_UDP_JOINT_SPEED] = (msg, self._now())

    def on_udp_joint_current(self, msg) -> None:
        self._latest[TOPIC_UDP_JOINT_CURRENT] = (msg, self._now())

    def on_udp_arm_position(self, msg) -> None:
        self._latest[TOPIC_UDP_ARM_POSITION] = (msg, self._now())

    def on_udp_joint_en_flag(self, msg) -> None:
        self._latest[TOPIC_UDP_JOINT_EN_FLAG] = (msg, self._now())

    def on_udp_rm_err(self, msg) -> None:
        self._latest[TOPIC_UDP_RM_ERR] = (msg, self._now())

    def on_udp_joint_error_code(self, msg) -> None:
        self._latest[TOPIC_UDP_JOINT_ERROR_CODE] = (msg, self._now())

    def on_udp_arm_current_status(self, msg) -> None:
        self._latest[TOPIC_UDP_ARM_CURRENT_STATUS] = (msg, self._now())

    def on_udp_rm_plus_state(self, msg) -> None:
        self._latest[TOPIC_UDP_RM_PLUS_STATE] = (msg, self._now())

    # --- assembly -------------------------------------------------------
    def _assemble(self) -> Observation:
        now = self._now()
        reasons: list = []
        obs = Observation()

        entry = self._latest.get(TOPIC_JOINT_STATES)
        if entry is None:
            reasons.append("joint_states: 未收到")
            obs.invalid_reasons = reasons
            return obs

        js, js_recv = entry
        stamp = js.header.stamp.sec + js.header.stamp.nanosec * 1e-9
        obs.timestamp = stamp
        if now - stamp > self.stale_timeout:
            reasons.append("joint_states: 时间戳超时(stale)")
        if now - js_recv > self.stale_timeout:
            reasons.append("joint_states: 本地接收超时")

        def required(topic, label):
            e = self._latest.get(topic)
            if e is None:
                reasons.append(f"{label}: 未收到")
                return None
            msg, recv = e
            if now - recv > self.stale_timeout:
                reasons.append(f"{label}: 超时(stale)")
                return None  # stale data must NOT be used: treat as missing
            return msg

        speed = required(TOPIC_UDP_JOINT_SPEED, "joint_speed")
        current = required(TOPIC_UDP_JOINT_CURRENT, "joint_current")
        pose = required(TOPIC_UDP_ARM_POSITION, "arm_position")
        en_flag = required(TOPIC_UDP_JOINT_EN_FLAG, "joint_en_flag")
        rm_err = required(TOPIC_UDP_RM_ERR, "rm_err")
        joint_err = required(TOPIC_UDP_JOINT_ERROR_CODE, "joint_error_code")

        # joint position: length must be exactly 6
        pos = list(js.position)
        if len(pos) == JOINT_COUNT:
            obs.joint_position = pos
        else:
            reasons.append(f"joint_states.position 长度 {len(pos)} != {JOINT_COUNT}")
            obs.joint_position = None  # malformed data, not zero-filled

        # velocity: RPM -> rad/s, length must be exactly 6
        if speed is not None and len(speed.joint_speed) == JOINT_COUNT:
            obs.joint_velocity = rpm_to_rad_s(speed.joint_speed)
        else:
            reasons.append("joint_speed 长度错误或缺失")
            obs.joint_velocity = None

        # current: keep official unit mA, length must be exactly 6
        if current is not None and len(current.joint_current) == JOINT_COUNT:
            obs.joint_current = list(current.joint_current)
        else:
            reasons.append("joint_current 长度错误或缺失")
            obs.joint_current = None

        # TCP pose: [x, y, z, qw, qx, qy, qz]
        if pose is not None:
            p = pose.position
            q = pose.orientation
            obs.tcp_pose = [p.x, p.y, p.z, q.w, q.x, q.y, q.z]
        else:
            obs.tcp_pose = None

        # joint error codes: length 6 and all zero
        if joint_err is not None:
            codes = list(joint_err.joint_error)
            if len(codes) == JOINT_COUNT and all(c == 0 for c in codes):
                pass  # healthy
            else:
                reasons.append("joint_error_code: 非零或长度错误")
        # (joint_err missing/stale already added by required())

        # rm error: must be empty
        if rm_err is not None and (rm_err.err_len != 0 or len(rm_err.err) != 0):
            reasons.append("rm_err: 存在机械臂错误")

        # joint enable flags
        if en_flag is not None and len(en_flag.joint_en_flag) == JOINT_COUNT:
            flags = list(en_flag.joint_en_flag)
            obs.joint_enabled = flags
            if self.require_all_enabled and not all(flags):
                reasons.append("joint_en_flag: 存在未使能关节")
        else:
            reasons.append("joint_en_flag 长度错误或缺失")

        # optional: arm status (gated topic; if fresh and == 9 -> invalid)
        e = self._latest.get(TOPIC_UDP_ARM_CURRENT_STATUS)
        if e is not None:
            msg, recv = e
            if now - recv > self.stale_timeout:
                reasons.append("arm_status: 超时(stale)")
            else:
                obs.arm_status = int(msg.arm_current_status)
                if obs.arm_status == EMERGENCY_STOP_STATUS:
                    reasons.append("arm_status: 急停状态(RM_STOP_E)")

        # optional: tool state (gated topic; carried through when fresh)
        e = self._latest.get(TOPIC_UDP_RM_PLUS_STATE)
        if e is not None:
            msg, recv = e
            if now - recv > self.stale_timeout:
                reasons.append("rm_plus_state: 超时(stale)")
            else:
                obs.tool_state = msg

        obs.valid = not reasons
        obs.invalid_reasons = reasons
        return obs
