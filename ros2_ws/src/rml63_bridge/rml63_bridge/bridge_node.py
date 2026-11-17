"""Read-only ROS2 node prototype for the RML63 arm (bridge).

HARD CONSTRAINTS (by design, do not weaken):
  - subscriptions ONLY: this node creates NO publishers, NO action clients,
    NO service clients, and never sends any message;
  - it therefore cannot move the arm, stop the arm, or influence rm_driver
    in any way;
  - do not run this node against a real robot in this prototype stage;
    it is written for the mock/offline unit tests and for interface review.

Alignment note (mirrors observation.py):
  The udp_* topics carry no header timestamp. They are treated as the frame
  belonging to the current ``joint_states.header.stamp`` because rm_driver
  publishes all of them back-to-back in the same UDP timer callback from one
  UDP packet. This is an approximation that MUST be validated on the real
  machine (DDS ordering, UDP loss, callback jitter).

Frequency: state topics are emitted by rm_driver at the ``udp_cycle`` rate
(default 5 ms = 200 Hz). A frame is assembled on every joint_states arrival.
"""

from __future__ import annotations

import os

import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node
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

from .observation import Observation, ObservationAssembler
from .transport import (
    DEFAULT_ZMQ_ADDRESS,
    make_transport,
    resolve_transport_name,
)


class RML63BridgeNode(Node):
    """Subscribes to rm_driver state topics and emits observation packets."""

    def __init__(self):
        super().__init__("rml63_bridge_node")

        self.declare_parameter("stale_timeout", 0.1)
        self.declare_parameter("require_all_enabled", True)
        # transport_type: "log" (default) or "zmq"; empty value = use env var
        # RML63_BRIDGE_TRANSPORT; if that is unset too -> "log".
        self.declare_parameter("transport_type", "")
        self.declare_parameter("transport_address", "")

        stale_timeout = float(self.get_parameter("stale_timeout").value)
        require_all_enabled = bool(self.get_parameter("require_all_enabled").value)
        transport_name = resolve_transport_name(
            str(self.get_parameter("transport_type").value),
            os.environ.get("RML63_BRIDGE_TRANSPORT"),
        )
        address = str(self.get_parameter("transport_address").value) or os.environ.get(
            "RML63_BRIDGE_ZMQ_ADDRESS", DEFAULT_ZMQ_ADDRESS
        )

        self.assembler = ObservationAssembler(
            stale_timeout=stale_timeout,
            require_all_enabled=require_all_enabled,
            now=lambda: self.get_clock().now().nanoseconds * 1e-9,
        )
        self.transport = make_transport(transport_name, self.get_logger(), address)
        if hasattr(self.transport, "start"):
            self.transport.start()

        # --- subscriptions ONLY (no publishers, no actions, no services) ---
        self.create_subscription(
            JointState, "joint_states", self._on_joint_states, 10
        )
        self.create_subscription(
            Jointspeed, "/rm_driver/udp_joint_speed", self._on_udp_joint_speed, 10
        )
        self.create_subscription(
            Jointcurrent, "/rm_driver/udp_joint_current", self._on_udp_joint_current, 10
        )
        self.create_subscription(
            Pose, "/rm_driver/udp_arm_position", self._on_udp_arm_position, 10
        )
        self.create_subscription(
            Jointenflag, "/rm_driver/udp_joint_en_flag", self._on_udp_joint_en_flag, 10
        )
        self.create_subscription(
            Rmerr, "/rm_driver/udp_rm_err", self._on_udp_rm_err, 10
        )
        self.create_subscription(
            Jointerrorcode, "/rm_driver/udp_joint_error_code", self._on_udp_joint_error_code, 10
        )
        self.create_subscription(
            Armcurrentstatus, "/rm_driver/udp_arm_current_status", self._on_udp_arm_current_status, 10
        )
        self.create_subscription(
            Rmplusstate, "/rm_driver/udp_rm_plus_state", self._on_udp_rm_plus_state, 10
        )
        # NOTE: NO subscriptions to movej_cmd / movej_canfd_cmd / pause_cmd /
        # emergency_stop_cmd or any other control topic. NO action clients.

        self.get_logger().info(
            "rml63_bridge started (READ-ONLY): stale_timeout=%.3fs require_all_enabled=%s transport=%s",
            stale_timeout,
            require_all_enabled,
            transport_name,
        )

    # --- callbacks ------------------------------------------------------
    def _on_joint_states(self, msg: JointState) -> None:
        obs: Observation = self.assembler.on_joint_states(msg)
        self.transport.send(obs)

    def _on_udp_joint_speed(self, msg: Jointspeed) -> None:
        self.assembler.on_udp_joint_speed(msg)

    def _on_udp_joint_current(self, msg: Jointcurrent) -> None:
        self.assembler.on_udp_joint_current(msg)

    def _on_udp_arm_position(self, msg: Pose) -> None:
        self.assembler.on_udp_arm_position(msg)

    def _on_udp_joint_en_flag(self, msg: Jointenflag) -> None:
        self.assembler.on_udp_joint_en_flag(msg)

    def _on_udp_rm_err(self, msg: Rmerr) -> None:
        self.assembler.on_udp_rm_err(msg)

    def _on_udp_joint_error_code(self, msg: Jointerrorcode) -> None:
        self.assembler.on_udp_joint_error_code(msg)

    def _on_udp_arm_current_status(self, msg: Armcurrentstatus) -> None:
        self.assembler.on_udp_arm_current_status(msg)

    def _on_udp_rm_plus_state(self, msg: Rmplusstate) -> None:
        self.assembler.on_udp_rm_plus_state(msg)


def main(args=None):
    rclpy.init(args=args)
    node = RML63BridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(node, "transport") and hasattr(node.transport, "close"):
            try:
                node.transport.close()
            except Exception:
                pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
