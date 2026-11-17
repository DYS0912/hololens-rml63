"""Offline end-to-end multi-sensor simulation (no ROS2, no hardware).

Simulates: 30 Hz RML63 observations (with jitter, drops, invalid frames),
30 Hz D435i camera frames (jitter, drops, invalid frames), 200 Hz IMU-only
packets. Verifies unified synchronized frames, validity statistics,
monotonic timestamps, sync-error statistics, and a ZMQ metadata roundtrip.
"""

import json
import random
import time
from dataclasses import replace

import pytest

from rml63_bridge.camera_packet import CameraPacket, make_mock_camera_packet
from rml63_bridge.mock_data import make_mock_observation
from rml63_bridge.synchronized_frame import (
    synchronize_observations,
    synchronized_frame_to_metadata,
)

FORBIDDEN_KEYS = ("action", "command", "target_joint", "trigger", "cmd")
N_FRAMES = 100
OBS_FPS = 30.0
IMU_FPS = 200.0


def build_obs_stream(rng: random.Random):
    obs = []
    for i in range(N_FRAMES):
        if i % 13 == 12:  # 偶发缺失帧
            obs.append(None)
            continue
        o = make_mock_observation(i, invalid_every=5)  # 偶发 invalid 帧 (i%5==0)
        obs.append(replace(o, timestamp=i / OBS_FPS + rng.uniform(-0.004, 0.004)))
    return obs


def build_camera_stream(rng: random.Random):
    cams = []
    for i in range(N_FRAMES):
        if i % 17 == 16:  # 偶发丢帧
            cams.append(None)
            continue
        c = make_mock_camera_packet(i, fps=OBS_FPS, invalid_every=11)  # 偶发 invalid 帧
        cams.append(replace(c, timestamp=i / OBS_FPS + rng.uniform(-0.005, 0.005)))
    return cams


def build_imu_stream(rng: random.Random):
    """200 Hz IMU-only packets (rgb/depth 为 None，仅 gyro/accel)。"""
    imu = []
    for i in range(int(N_FRAMES / OBS_FPS * IMU_FPS)):
        imu.append(
            CameraPacket(
                timestamp=i / IMU_FPS + rng.uniform(-0.0004, 0.0004),
                gyro=[0.01 * (i % 7), 0.02 * (i % 5), 0.005 * (i % 3)],
                accel=[0.0, 0.0, 9.81 + 0.05 * (i % 4)],
                frame_id=i,
                valid=True,
            )
        )
    return imu


class TestOfflineE2E:
    def test_unified_frames_stats_and_errors(self):
        rng = random.Random(42)
        obs_list = build_obs_stream(rng)
        cam_list = build_camera_stream(rng)

        frames = synchronize_observations(
            obs_list, cam_list, max_time_delta=0.05, require_camera=True, require_imu=False
        )

        # 1. 能生成统一同步帧
        assert len(frames) == N_FRAMES
        valid = [f for f in frames if f.valid]
        invalid = [f for f in frames if not f.valid]
        print(f"\n  valid={len(valid)} invalid={len(invalid)}")

        # 2. valid/invalid 统计正确（确定性种子下的合理范围 + 结构性断言）
        assert 55 <= len(valid) <= 90
        assert all(f.invalid_reasons for f in invalid)  # invalid 必有原因
        assert all(f.camera_packet is not None for f in valid)
        assert all(abs(f.time_delta) <= 0.05 for f in valid)
        # 缺失的 observation/camera 绝不被零填充
        for f in frames:
            if not f.valid and f.observation is not None and f.camera_packet is None:
                assert f.camera_timestamp is None and f.time_delta is None

        # 3. timestamp 单调递增（过滤 observation 缺失帧）
        stamps = [f.timestamp for f in frames if f.timestamp is not None]
        assert stamps == sorted(stamps)
        assert all(b > a for a, b in zip(stamps, stamps[1:]))

        # 4. 同步误差可统计
        deltas = [abs(f.time_delta) for f in valid]
        mean_dt = sum(deltas) / len(deltas)
        print(
            f"  sync error: mean={mean_dt * 1000:.2f} ms "
            f"max={max(deltas) * 1000:.2f} ms (n={len(deltas)})"
        )
        assert max(deltas) <= 0.05
        assert mean_dt <= 0.01  # 两个 30Hz 流 + 轻微抖动，均值应在 ~3ms 量级

        # 5. 不包含控制字段
        for f in frames:
            text = json.dumps(synchronized_frame_to_metadata(f)).lower()
            for key in FORBIDDEN_KEYS:
                assert f'"{key}' not in text

        # require_camera=False: 放宽相机要求后 valid 只增不减
        relaxed = synchronize_observations(
            obs_list, cam_list, max_time_delta=0.05, require_camera=False, require_imu=False
        )
        assert sum(1 for f in relaxed if f.valid) >= len(valid)

    def test_imu_200hz_alignment(self):
        rng = random.Random(7)
        obs_list = build_obs_stream(rng)
        imu_list = build_imu_stream(rng)

        # 200 Hz IMU: 为每个有效 observation 手工做最近邻对齐并统计误差
        deltas = []
        for o in obs_list:
            if o is None or not o.valid:
                continue
            best = min(imu_list, key=lambda p: abs(p.timestamp - o.timestamp))
            dt = abs(best.timestamp - o.timestamp)
            assert len(best.gyro) == 3 and len(best.accel) == 3
            deltas.append(dt)
        assert deltas, "没有可对齐的 observation"
        mean_dt = sum(deltas) / len(deltas)
        print(
            f"\n  IMU(200Hz) alignment: mean={mean_dt * 1000:.2f} ms "
            f"max={max(deltas) * 1000:.2f} ms (n={len(deltas)})"
        )
        assert max(deltas) <= 0.004  # 5ms 网格 + ±0.4ms 抖动
        assert mean_dt <= 0.003

    @pytest.fixture(scope="class")
    def zmq(self):
        return pytest.importorskip("zmq")

    def test_zmq_metadata_roundtrip(self, zmq):
        rng = random.Random(42)
        obs_list = build_obs_stream(rng)
        cam_list = build_camera_stream(rng)
        frames = synchronize_observations(
            obs_list, cam_list, max_time_delta=0.05, require_camera=True
        )
        valid = [f for f in frames if f.valid]
        assert valid, "没有 valid 帧可供传输"

        addr = "inproc://rml63_sync_e2e"
        ctx = zmq.Context()
        pub = ctx.socket(zmq.PUB)
        pub.bind(addr)
        sub = ctx.socket(zmq.SUB)
        sub.connect(addr)
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        time.sleep(0.1)

        payload = json.dumps([synchronized_frame_to_metadata(f) for f in valid[:5]])
        pub.send_string(payload)
        if not sub.poll(2000, zmq.POLLIN):
            pytest.fail("SUB 未收到同步帧元数据")
        received = json.loads(sub.recv_string())

        assert len(received) == 5
        for meta in received:
            text = json.dumps(meta).lower()
            for key in FORBIDDEN_KEYS:
                assert f'"{key}' not in text
            assert "camera" in meta and meta["camera"] is not None
            assert meta["camera"]["rgb"]["shape"] == [480, 640, 3]  # 只有 shape/dtype 摘要
        pub.close()
        sub.close()
        ctx.destroy(linger=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
