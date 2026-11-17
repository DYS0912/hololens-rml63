"""Unit tests for the offline mock data pipeline (no ROS2, no robot, no camera).

Covers: mock_data generation, frame_sequence filtering, ZMQ PUB/SUB
delivery, and the mock->ObservationAssembler->Observation path.
No rclpy node is ever instantiated.
"""

import json
import time

import pytest

from rml63_bridge.mock_data import (
    DEFAULT_FPS,
    MOCK_INVALID_REASON,
    feed_mock_frame,
    make_mock_observation,
    mock_observation_stream,
)
from rml63_bridge.observation import ObservationAssembler
from rml63_bridge.replay_observation import frame_sequence
from rml63_bridge.transport import (
    OBSERVATION_FIELDS,
    ZmqObservationTransport,
    observation_to_json,
)

FORBIDDEN_KEYS = ("action", "command", "target_joint", "trigger", "cmd")


class FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


class TestMockData:
    def test_timestamp_increments_at_fps(self):
        frames = list(mock_observation_stream(count=30, fps=30.0, invalid_every=0))
        stamps = [f.timestamp for f in frames]
        assert stamps == sorted(stamps)
        assert all(b > a for a, b in zip(stamps, stamps[1:]))
        diffs = [b - a for a, b in zip(stamps, stamps[1:])]
        for d in diffs:
            assert d == pytest.approx(1.0 / 30.0)

    def test_dimensions_6_6_6_7(self):
        obs = make_mock_observation(1, invalid_every=0)
        assert obs.valid is True
        assert len(obs.joint_position) == 6
        assert len(obs.joint_velocity) == 6
        assert len(obs.joint_current) == 6
        assert len(obs.tcp_pose) == 7
        assert len(obs.joint_enabled) == 6
        # quaternion is unit-length: qw^2+qx^2+qy^2+qz^2 == 1
        q = obs.tcp_pose[3:]
        assert sum(v * v for v in q) == pytest.approx(1.0)

    def test_valid_and_invalid_samples(self):
        frames = list(mock_observation_stream(count=10, invalid_every=5))
        for i, f in enumerate(frames):
            if i > 0 and i % 5 == 0:
                assert f.valid is False
                assert f.invalid_reasons == [MOCK_INVALID_REASON]
                assert f.joint_current is None
            else:
                assert f.valid is True
                assert f.invalid_reasons == []
                assert f.joint_current is not None
        assert sum(1 for f in frames if f.valid) == 9

    def test_none_to_json_null(self):
        obs = make_mock_observation(5, invalid_every=5)
        s = observation_to_json(obs)
        assert '"joint_current": null' in s

    def test_no_control_fields(self):
        for idx in (1, 5):
            s = observation_to_json(make_mock_observation(idx, invalid_every=5))
            low = s.lower()
            for key in FORBIDDEN_KEYS:
                assert f'"{key}' not in low
            d = json.loads(s)
            assert set(d.keys()) == set(OBSERVATION_FIELDS)


class TestFrameSequence:
    def test_drop_invalid_filters(self):
        frames = list(
            frame_sequence(count=10, fps=30.0, invalid_every=5, drop_invalid=True)
        )
        assert len(frames) == 9  # index 5 dropped
        assert all(f.valid for f in frames)

    def test_keep_invalid_when_not_dropping(self):
        frames = list(
            frame_sequence(count=10, fps=30.0, invalid_every=5, drop_invalid=False)
        )
        assert len(frames) == 10
        assert sum(1 for f in frames if not f.valid) == 1


class TestAssemblerPipeline:
    """mock RML63 state -> ObservationAssembler -> Observation."""

    def test_valid_frame_through_assembler(self):
        clock = FakeClock(10.0)
        assembler = ObservationAssembler(stale_timeout=0.1, now=clock)
        direct = make_mock_observation(1, invalid_every=0)
        obs = feed_mock_frame(assembler, 1, invalid_every=0, stamp=10.0)
        assert obs.valid is True, obs.invalid_reasons
        assert len(obs.joint_position) == 6
        assert len(obs.joint_velocity) == 6
        assert len(obs.joint_current) == 6
        assert len(obs.tcp_pose) == 7
        # velocity round-trips through the assembler's RPM -> rad/s conversion
        for got, want in zip(obs.joint_velocity, direct.joint_velocity):
            assert got == pytest.approx(want, rel=1e-6)
        # tcp_pose order: [x, y, z, qw, qx, qy, qz]
        assert obs.tcp_pose[0:3] == pytest.approx(direct.tcp_pose[0:3])
        assert obs.timestamp == pytest.approx(10.0)

    def test_invalid_frame_through_assembler(self):
        clock = FakeClock(10.0)
        assembler = ObservationAssembler(stale_timeout=0.1, now=clock)
        feed_mock_frame(assembler, 1, invalid_every=5, stamp=10.0)  # valid first
        clock.t = 10.5  # make the previous joint_current stale
        obs = feed_mock_frame(assembler, 5, invalid_every=5, stamp=10.5)
        assert obs.valid is False
        assert any("joint_current" in r for r in obs.invalid_reasons)
        assert any("joint_error_code" in r for r in obs.invalid_reasons)
        assert obs.joint_current is None  # never zero-filled / never stale-filled


@pytest.fixture(scope="module")
def zmq():
    return pytest.importorskip("zmq")


class TestZmqReplay:
    """Offline ZMQ PUB/SUB: replay -> SUB receives observation JSON."""

    def test_pub_sub_receives_mock_frames(self, zmq):
        addr = "inproc://rml63_mock_replay"
        ctx = zmq.Context()
        transport = ZmqObservationTransport(addr, context=ctx).start()
        try:
            sub = ctx.socket(zmq.SUB)
            sub.connect(addr)
            sub.setsockopt_string(zmq.SUBSCRIBE, "")
            time.sleep(0.1)  # slow joiner: let the subscription propagate

            frames = list(frame_sequence(count=6, fps=30.0, invalid_every=5))
            for f in frames:
                transport.send(f)

            received = []
            deadline = time.time() + 2.0
            while time.time() < deadline and len(received) < len(frames):
                if sub.poll(200, zmq.POLLIN):
                    received.append(json.loads(sub.recv_string()))
            assert len(received) >= 1, "SUB 未收到任何 mock 帧"

            for d in received:
                assert set(d.keys()) == set(OBSERVATION_FIELDS)
                if d["valid"]:
                    assert len(d["joint_position"]) == 6
                    assert len(d["joint_velocity"]) == 6
                    assert len(d["joint_current"]) == 6
                    assert len(d["tcp_pose"]) == 7
                else:
                    assert d["joint_current"] is None
                    assert d["invalid_reasons"] == [MOCK_INVALID_REASON]
        finally:
            transport.close()
            ctx.destroy(linger=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
