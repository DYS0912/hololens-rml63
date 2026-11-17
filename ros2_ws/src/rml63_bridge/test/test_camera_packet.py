"""Unit tests for the D435i offline camera packet (no ROS2, no pyrealsense2,
no real camera)."""

import json
from dataclasses import fields

import numpy as np
import pytest

from rml63_bridge.camera_packet import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    CameraPacket,
    camera_packet_to_metadata,
    make_mock_camera_packet,
    validate_camera_packet,
)
from rml63_bridge.observation import Observation


class TestMockCameraPacket:
    def test_rgb_shape_and_dtype(self):
        p = make_mock_camera_packet(1, invalid_every=0)
        assert p.rgb.shape == (DEFAULT_HEIGHT, DEFAULT_WIDTH, 3)
        assert p.rgb.dtype == np.uint8

    def test_depth_shape_and_dtype(self):
        p = make_mock_camera_packet(1, invalid_every=0)
        assert p.depth.shape == (DEFAULT_HEIGHT, DEFAULT_WIDTH)
        assert p.depth.dtype == np.uint16

    def test_aligned_depth_shape_and_dtype(self):
        p = make_mock_camera_packet(1, invalid_every=0)
        assert p.aligned_depth.shape == (DEFAULT_HEIGHT, DEFAULT_WIDTH)
        assert p.aligned_depth.dtype == np.uint16

    def test_gyro_length_3(self):
        p = make_mock_camera_packet(2, invalid_every=0)
        assert len(p.gyro) == 3

    def test_accel_length_3(self):
        p = make_mock_camera_packet(2, invalid_every=0)
        assert len(p.accel) == 3

    def test_timestamp_correct(self):
        fps = 30.0
        p = make_mock_camera_packet(7, fps=fps, invalid_every=0)
        assert p.timestamp == pytest.approx(7 / fps)

    def test_none_not_converted_to_zeros(self):
        p = make_mock_camera_packet(5, invalid_every=5)  # invalid frame
        assert p.valid is False
        assert p.rgb is None  # stays None, NOT a zeros array
        assert p.depth is None
        assert p.aligned_depth is None
        assert p.gyro is None
        assert p.accel is None


class TestValidateCameraPacket:
    def test_missing_rgb_invalid(self):
        p = make_mock_camera_packet(1, invalid_every=0)
        p.rgb = None
        out = validate_camera_packet(p)
        assert out.valid is False
        assert any("rgb" in r for r in out.invalid_reasons)
        assert out.rgb is None  # never zero-filled

    def test_missing_depth_invalid(self):
        p = make_mock_camera_packet(1, invalid_every=0)
        p.depth = None
        out = validate_camera_packet(p)
        assert out.valid is False
        assert any("depth" in r for r in out.invalid_reasons)

    def test_imu_missing_configurable(self):
        p = make_mock_camera_packet(1, invalid_every=0, include_imu=False)
        assert p.gyro is None and p.accel is None

        # IMU not required -> frame stays valid, no IMU reason
        ok = validate_camera_packet(p, require_imu=False)
        assert ok.valid is True, ok.invalid_reasons
        assert not any("gyro" in r or "accel" in r for r in ok.invalid_reasons)

        # IMU required -> invalid with explicit reasons
        bad = validate_camera_packet(p, require_imu=True)
        assert bad.valid is False
        assert any("gyro" in r for r in bad.invalid_reasons)
        assert any("accel" in r for r in bad.invalid_reasons)


class TestMetadata:
    def test_metadata_has_no_raw_image_arrays(self):
        p = make_mock_camera_packet(3, invalid_every=0)
        meta = camera_packet_to_metadata(p)
        text = json.dumps(meta)
        # no raw pixels: only shape/dtype summaries for images
        assert meta["rgb"] == {"present": True, "shape": [480, 640, 3], "dtype": "uint8"}
        assert meta["depth"] == {"present": True, "shape": [480, 640], "dtype": "uint16"}
        assert meta["aligned_depth"] == {
            "present": True,
            "shape": [480, 640],
            "dtype": "uint16",
        }
        assert len(text) < 10_000, f"metadata 过大 ({len(text)} bytes)，疑似含原始图像"
        assert "ndarray" not in text

    def test_metadata_keeps_imu_values(self):
        p = make_mock_camera_packet(3, invalid_every=0)
        meta = camera_packet_to_metadata(p)
        assert meta["gyro"] == pytest.approx(p.gyro)
        assert meta["accel"] == pytest.approx(p.accel)


class TestObservationUnchanged:
    def test_observation_dataclass_unchanged(self):
        """Camera work must NOT touch the 25-dim RML63 Observation."""
        obs_fields = {f.name for f in fields(Observation)}
        assert obs_fields == {
            "timestamp",
            "joint_position",
            "joint_velocity",
            "joint_current",
            "tcp_pose",
            "arm_status",
            "joint_enabled",
            "tool_state",
            "valid",
            "invalid_reasons",
        }
        cam_fields = {f.name for f in fields(CameraPacket)}
        assert cam_fields == {
            "timestamp",
            "camera_name",
            "rgb",
            "depth",
            "aligned_depth",
            "gyro",
            "accel",
            "frame_id",
            "valid",
            "invalid_reasons",
        }
        # no camera fields leaked into Observation, no joint fields in CameraPacket
        assert not (cam_fields - {"timestamp", "valid", "invalid_reasons"}) & obs_fields

    def test_observation_still_25_dim(self):
        obs = Observation(
            joint_position=[0.0] * 6,
            joint_velocity=[0.0] * 6,
            joint_current=[0.0] * 6,
            tcp_pose=[0.0] * 7,
        )
        dims = (
            len(obs.joint_position)
            + len(obs.joint_velocity)
            + len(obs.joint_current)
            + len(obs.tcp_pose)
        )
        assert dims == 25


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
