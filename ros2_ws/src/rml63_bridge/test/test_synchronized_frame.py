"""Unit tests for offline observation/camera synchronization (no ROS2,
no hardware)."""

import json
from dataclasses import replace

import pytest

from rml63_bridge.camera_packet import make_mock_camera_packet
from rml63_bridge.mock_data import make_mock_observation
from rml63_bridge.synchronized_frame import (
    synchronize_observations,
    synchronized_frame_to_metadata,
)


def obs_at(t, valid=True, index=1):
    o = make_mock_observation(index, invalid_every=0)
    o = replace(o, timestamp=t)
    if not valid:
        o = replace(o, valid=False, invalid_reasons=["mock obs invalid"])
    return o


def cam_at(t, include_imu=True, valid=True):
    c = make_mock_camera_packet(1, invalid_every=0, include_imu=include_imu)
    c = replace(c, timestamp=t)
    if not valid:
        c = replace(c, valid=False, invalid_reasons=["mock camera invalid"])
    return c


class TestSynchronization:
    def test_perfectly_synced_valid(self):
        frames = synchronize_observations([obs_at(1.0)], [cam_at(1.0)])
        assert frames[0].valid is True
        assert frames[0].time_delta == pytest.approx(0.0)

    def test_delta_under_50ms_valid(self):
        frames = synchronize_observations([obs_at(1.0)], [cam_at(1.03)])
        assert frames[0].valid is True
        assert frames[0].time_delta == pytest.approx(0.03)

    def test_delta_over_50ms_invalid(self):
        frames = synchronize_observations([obs_at(1.0)], [cam_at(1.06)])
        assert frames[0].valid is False
        assert any("camera" in r for r in frames[0].invalid_reasons)

    def test_missing_camera_invalid(self):
        frames = synchronize_observations([obs_at(1.0)], [])
        assert frames[0].valid is False
        assert frames[0].camera_packet is None

    def test_missing_observation_invalid(self):
        frames = synchronize_observations([None], [cam_at(1.0)])
        assert frames[0].valid is False
        assert frames[0].observation is None
        assert any("observation" in r for r in frames[0].invalid_reasons)

    def test_require_imu_true_missing_imu_invalid(self):
        frames = synchronize_observations(
            [obs_at(1.0)], [cam_at(1.0, include_imu=False)], require_imu=True
        )
        assert frames[0].valid is False
        assert any("gyro" in r or "accel" in r for r in frames[0].invalid_reasons)

    def test_require_imu_false_missing_imu_ok(self):
        frames = synchronize_observations(
            [obs_at(1.0)], [cam_at(1.0, include_imu=False)], require_imu=False
        )
        assert frames[0].valid is True, frames[0].invalid_reasons

    def test_nearest_camera_selected(self):
        frames = synchronize_observations(
            [obs_at(1.0)], [cam_at(0.96), cam_at(1.03)]
        )
        assert frames[0].valid is True
        assert frames[0].camera_timestamp == pytest.approx(1.03)

    def test_tie_prefers_past_frame(self):
        frames = synchronize_observations(
            [obs_at(1.0)], [cam_at(1.03), cam_at(0.97)]
        )
        assert frames[0].camera_timestamp == pytest.approx(0.97)

    def test_no_far_future_frame_used(self):
        # 只有超出 +50ms 的未来帧 -> invalid
        frames = synchronize_observations([obs_at(1.0)], [cam_at(1.07)])
        assert frames[0].valid is False
        # 存在 +70ms 未来帧与 -10ms 过去帧 -> 只能选过去帧
        frames = synchronize_observations(
            [obs_at(1.0)], [cam_at(0.99), cam_at(1.07)]
        )
        assert frames[0].valid is True
        assert frames[0].camera_timestamp == pytest.approx(0.99)

    def test_no_zero_fill(self):
        bad_obs = replace(obs_at(1.0, valid=True), joint_current=None)
        frames = synchronize_observations([bad_obs], [])
        f = frames[0]
        assert f.valid is False
        assert f.observation.joint_current is None  # 保持 None，不被填充为零
        assert f.camera_packet is None

    def test_metadata_has_no_raw_images(self):
        frames = synchronize_observations([obs_at(1.0)], [cam_at(1.0)])
        meta = synchronized_frame_to_metadata(frames[0])
        text = json.dumps(meta)
        assert len(text) < 10_000
        assert "ndarray" not in text
        assert meta["camera"]["rgb"]["shape"] == [480, 640, 3]
        assert meta["camera"]["depth"]["shape"] == [480, 640]
        assert meta["camera"]["rgb"]["dtype"] == "uint8"

    def test_timestamps_and_reasons_correct(self):
        bad_obs = replace(obs_at(1.0), valid=False, invalid_reasons=["mock obs invalid"])
        frames = synchronize_observations([bad_obs], [cam_at(1.02)])
        f = frames[0]
        assert f.timestamp == pytest.approx(1.0)
        assert f.observation_timestamp == pytest.approx(1.0)
        assert f.camera_timestamp == pytest.approx(1.02)
        assert f.time_delta == pytest.approx(0.02)
        assert f.valid is False
        assert "observation: 原始帧无效" in f.invalid_reasons
        assert "mock obs invalid" in f.observation.invalid_reasons


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
