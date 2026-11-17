"""Unit tests for the observation transport layer (pure Python, no rclpy node).

The serialization tests run without pyzmq. The actual ZMQ socket tests are
skipped via pytest.importorskip("zmq") when pyzmq is not installed.
No ROS2 node is ever instantiated and no socket sends anything over the
network (tests use inproc:// endpoints only).
"""

import json

import numpy as np
import pytest
from rm_ros_interfaces.msg import Rmplusstate

from rml63_bridge.observation import Observation
from rml63_bridge.transport import (
    DEFAULT_TRANSPORT,
    LogTransport,
    OBSERVATION_FIELDS,
    ZmqObservationTransport,
    make_transport,
    observation_to_dict,
    observation_to_json,
    resolve_transport_name,
)

FORBIDDEN_KEYS = ("action", "command", "target_joint", "trigger", "cmd")


def make_valid_observation():
    return Observation(
        timestamp=10.123456,
        joint_position=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        joint_velocity=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        joint_current=[12.5, 13.5, 14.5, 15.5, 16.5, 17.5],
        tcp_pose=[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
        arm_status=0,
        joint_enabled=[True] * 6,
        tool_state=None,
        valid=True,
        invalid_reasons=[],
    )


def make_invalid_observation():
    return Observation(
        timestamp=10.0,
        joint_position=None,
        joint_velocity=None,
        joint_current=None,
        tcp_pose=None,
        arm_status=None,
        joint_enabled=None,
        tool_state=None,
        valid=False,
        invalid_reasons=["joint_current 长度错误或缺失"],
    )


class DummyLogger:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


class TestSerialization:
    def test_valid_observation_serializes(self):
        obs = make_valid_observation()
        d = observation_to_dict(obs)
        assert set(d.keys()) == set(OBSERVATION_FIELDS)
        assert d["valid"] is True
        s = observation_to_json(obs)
        assert json.loads(s)["timestamp"] == pytest.approx(10.123456)

    def test_invalid_observation_serializes(self):
        obs = make_invalid_observation()
        d = observation_to_dict(obs)
        assert d["valid"] is False
        assert d["invalid_reasons"] == ["joint_current 长度错误或缺失"]

    def test_none_becomes_json_null(self):
        s = observation_to_json(make_invalid_observation())
        d = json.loads(s)
        assert d["joint_current"] is None
        assert d["tcp_pose"] is None
        # raw JSON text must contain null, not zeros
        assert '"joint_position": null' in s

    def test_numpy_values_serialize(self):
        obs = make_valid_observation()
        obs.joint_position = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
        obs.joint_current = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
        obs.timestamp = np.float32(9.5)
        d = observation_to_dict(obs)
        # numpy converted to plain Python types
        assert isinstance(d["timestamp"], float)
        assert all(isinstance(v, float) for v in d["joint_position"])
        assert all(isinstance(v, int) for v in d["joint_current"])
        assert d["joint_position"][0] == pytest.approx(0.1, rel=1e-5)

    def test_array_dimensions(self):
        d = observation_to_dict(make_valid_observation())
        assert len(d["joint_position"]) == 6
        assert len(d["joint_velocity"]) == 6
        assert len(d["joint_current"]) == 6
        assert len(d["tcp_pose"]) == 7
        assert len(d["joint_enabled"]) == 6

    def test_no_control_fields_in_json(self):
        obs = make_valid_observation()
        # attach a tool_state with an evil attribute to prove it is filtered
        tool = Rmplusstate()
        tool.sys_state = 1
        obs.tool_state = tool
        s = observation_to_json(obs)
        text_lower = s.lower()
        for key in FORBIDDEN_KEYS:
            assert key not in d_of(obs), f"dict 包含禁止字段 {key}"
            assert f'"{key}' not in text_lower, f"JSON 包含禁止字段 {key}"
        assert set(d_of(obs).keys()) == set(OBSERVATION_FIELDS)

    def test_ros_message_tool_state_serializes(self):
        tool = Rmplusstate()
        tool.sys_state = 1
        obs = make_valid_observation()
        obs.tool_state = tool
        d = observation_to_dict(obs)
        assert isinstance(d["tool_state"], dict)
        assert d["tool_state"]["sys_state"] == 1

    def test_default_transport_is_log(self):
        # unset parameter + unset env -> log
        assert resolve_transport_name("", None) == "log"
        assert resolve_transport_name("", "") == "log"
        assert DEFAULT_TRANSPORT == "log"
        t = make_transport("log", DummyLogger())
        assert isinstance(t, LogTransport)

    def test_env_var_selection(self):
        assert resolve_transport_name("", "zmq") == "zmq"
        assert resolve_transport_name("log", "zmq") == "log"  # param wins
        assert resolve_transport_name("zmq", None) == "zmq"

    def test_unknown_transport_raises(self):
        with pytest.raises(ValueError):
            make_transport("nope", DummyLogger())


def d_of(obs):
    return observation_to_dict(obs)


class TestZmqTransportInterface:
    """Interface-shape tests: run even when pyzmq is NOT installed."""

    def test_no_receive_method(self):
        t = ZmqObservationTransport("inproc://test-none")
        assert not hasattr(t, "recv")
        assert not hasattr(t, "receive")
        assert not hasattr(t, "send_action")

    def test_close_idempotent_without_socket(self):
        t = ZmqObservationTransport("inproc://test-close")
        t.close()  # never started
        t.close()  # again -> no crash
        assert True

    def test_no_socket_created_at_import(self):
        # the class constructor must not create any socket
        t = ZmqObservationTransport("inproc://test-lazy")
        assert t._socket is None


@pytest.fixture(scope="module")
def zmq():
    return pytest.importorskip("zmq")


class TestZmqSocket:
    """Actual socket tests: skipped when pyzmq is not installed."""

    def test_pub_sub_roundtrip(self, zmq):
        # NOTE: PUB and SUB must share ONE context for inproc:// endpoints
        # (cross-context inproc subscriptions do not propagate in libzmq
        # 4.3.5; verified empirically). The transport therefore receives the
        # test context via its `context=` argument.
        addr = "inproc://rml63_bridge_test"
        ctx = zmq.Context()
        transport = ZmqObservationTransport(addr, context=ctx).start()
        try:
            sub = ctx.socket(zmq.SUB)
            sub.connect(addr)
            sub.setsockopt_string(zmq.SUBSCRIBE, "")
            # slow joiner: PUB drops frames until the subscriber is known
            import time

            time.sleep(0.1)
            obs = make_valid_observation()
            transport.send(obs)  # warmup frame may be dropped
            transport.send(obs)
            if sub.poll(2000, zmq.POLLIN):
                raw = sub.recv_string()
                d = json.loads(raw)
                assert d["valid"] is True
                assert len(d["joint_position"]) == 6
                assert len(d["tcp_pose"]) == 7
                assert d["timestamp"] == pytest.approx(10.123456)
            else:
                pytest.fail("订阅端未收到任何帧")
            sub.close()
        finally:
            transport.close()
            ctx.destroy(linger=0)

    def test_send_after_close_reopens(self, zmq):
        addr = "inproc://rml63_bridge_test2"
        transport = ZmqObservationTransport(addr).start()
        transport.close()
        transport.send(make_valid_observation())  # must not crash (auto re-start)
        transport.close()

    def test_close_idempotent_with_socket(self, zmq):
        transport = ZmqObservationTransport("inproc://rml63_bridge_test3").start()
        transport.close()
        transport.close()  # repeated close must not crash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
