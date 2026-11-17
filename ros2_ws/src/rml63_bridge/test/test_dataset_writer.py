"""Unit + offline E2E tests for the LeRobot v2-style dataset writer
(no ROS2, no hardware)."""

import json
from dataclasses import replace

import numpy as np
import pytest

from rml63_bridge.camera_packet import make_mock_camera_packet
from rml63_bridge.dataset_writer import STATE_DIM, DatasetWriter
from rml63_bridge.mock_data import make_mock_observation
from rml63_bridge.synchronized_frame import SynchronizedFrame, synchronize_observations


def make_frame(ts, valid=True, with_camera=True, with_imu=True, index=1):
    o = make_mock_observation(index, invalid_every=0)
    o = replace(o, timestamp=ts)
    c = None
    if with_camera:
        c = make_mock_camera_packet(index, invalid_every=0, include_imu=with_imu)
        c = replace(c, timestamp=ts)
    return SynchronizedFrame(
        timestamp=ts,
        observation=o,
        camera_packet=c,
        observation_timestamp=ts,
        camera_timestamp=ts if c is not None else None,
        time_delta=0.0 if c is not None else None,
        valid=valid,
        invalid_reasons=[] if valid else ["test: invalid frame"],
    )


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_rows(episode_path):
    """读回 episode 数据行（后端无关：jsonl 或 parquet）。

    dataset_writer 的后端取决于运行环境有无 pyarrow（dora-rs 依赖
    pyarrow，安装后 staging 自动切换为 parquet）。
    """
    jsonl = episode_path.with_suffix(".jsonl")
    if jsonl.is_file():
        return read_jsonl(jsonl)
    parquet = episode_path.with_suffix(".parquet")
    if parquet.is_file():
        pytest.importorskip("pyarrow", reason="读取 parquet 需要 pyarrow")
        import pyarrow.parquet as pq

        return pq.read_table(parquet).to_pylist()
    raise FileNotFoundError(f"未找到 episode 数据文件: {episode_path}.*")


class TestDatasetWriter:
    def test_one_valid_episode_saves(self, tmp_path):
        w = DatasetWriter(str(tmp_path))
        w.start_episode("task_a")
        for i in range(3):
            assert w.append_frame(make_frame(1.0 + i * 0.1, index=i)) is True
        w.finish_episode()
        rows = read_rows(tmp_path / "data" / "episode_000000")
        assert len(rows) == 3

    def test_multiple_episodes_index_correct(self, tmp_path):
        w = DatasetWriter(str(tmp_path))
        for ep in range(2):
            w.start_episode(f"task_{ep}")
            w.append_frame(make_frame(1.0 + ep))
            w.finish_episode()
        rows = read_rows(tmp_path / "data" / "episode_000000")
        rows += read_rows(tmp_path / "data" / "episode_000001")
        assert [r["episode_index"] for r in rows] == [0, 1]
        assert rows[0]["task_index"] != rows[1]["task_index"]

    def test_frame_index_consecutive_despite_skips(self, tmp_path):
        w = DatasetWriter(str(tmp_path))
        w.start_episode("task")
        assert w.append_frame(make_frame(1.0, valid=False)) is False  # 被过滤
        w.append_frame(make_frame(1.1))
        w.append_frame(make_frame(1.2))
        w.finish_episode()
        rows = read_rows(tmp_path / "data" / "episode_000000")
        assert [r["frame_index"] for r in rows] == [0, 1]

    def test_invalid_frames_not_written(self, tmp_path):
        w = DatasetWriter(str(tmp_path))
        w.start_episode("task")
        w.append_frame(make_frame(1.0))
        assert w.append_frame(make_frame(1.1, valid=False)) is False
        w.finish_episode()
        rows = read_rows(tmp_path / "data" / "episode_000000")
        assert len(rows) == 1
        assert all(r["valid"] is True for r in rows)
        # skipped 记录在 episode 元数据里
        w.write_metadata()
        info = json.loads((tmp_path / "meta_data" / "info.json").read_text())
        assert info["episodes"][0]["skipped_frames"] == 1
        assert info["episodes"][0]["skipped_reasons"] == {"frame_invalid": 1}

    def test_none_images_not_zero_filled(self, tmp_path):
        w = DatasetWriter(str(tmp_path))
        w.start_episode("task")
        w.append_frame(make_frame(1.0, with_camera=False))  # valid 但无相机
        w.finish_episode()
        rows = read_rows(tmp_path / "data" / "episode_000000")
        r = rows[0]
        assert r["observation.images.rgb.path"] is None
        assert r["observation.images.depth.path"] is None
        assert r["observation.images.aligned_depth.path"] is None
        # 没有生成任何全零图像文件
        image_files = list((tmp_path / "videos").rglob("*"))
        assert all(not f.is_file() for f in image_files) or image_files == []

    def test_image_paths_relative(self, tmp_path):
        w = DatasetWriter(str(tmp_path))
        w.start_episode("task")
        w.append_frame(make_frame(1.0))
        w.finish_episode()
        r = read_rows(tmp_path / "data" / "episode_000000")[0]
        for key in (
            "observation.images.rgb.path",
            "observation.images.depth.path",
            "observation.images.aligned_depth.path",
        ):
            assert r[key].startswith("videos/")
            assert not r[key].startswith("/")
            assert str(tmp_path) not in r[key]

    def test_state_25_dim_and_order(self, tmp_path):
        w = DatasetWriter(str(tmp_path))
        w.start_episode("task")
        w.append_frame(make_frame(1.0))
        w.finish_episode()
        r = read_rows(tmp_path / "data" / "episode_000000")[0]
        state = r["observation.state"]
        assert len(state) == STATE_DIM == 25
        obs = make_frame(1.0).observation
        assert state[0:6] == list(obs.joint_position)
        assert state[6:12] == list(obs.joint_velocity)
        assert state[12:18] == list(obs.joint_current)
        assert state[18:25] == list(obs.tcp_pose)

    def test_action_none_not_faked(self, tmp_path):
        w = DatasetWriter(str(tmp_path))
        w.start_episode("task")
        w.append_frame(make_frame(1.0), action=None)
        w.append_frame(make_frame(1.1), action=[0.1] * 6)
        w.finish_episode()
        rows = read_rows(tmp_path / "data" / "episode_000000")
        assert rows[0]["action"] is None  # 显式缺失 (JSON null)
        assert "action" in rows[0]  # 字段存在但为 null，不是被删除
        assert rows[0]["observation.state"] is not None
        assert rows[1]["action"] == [0.1] * 6  # 显式传入的 action 原样保存

    def test_stats_exclude_invalid(self, tmp_path):
        w = DatasetWriter(str(tmp_path))
        w.start_episode("task")
        w.append_frame(make_frame(1.0))
        w.append_frame(make_frame(1.1, valid=False))
        w.append_frame(make_frame(1.2, index=2))
        w.write_metadata()
        stats = json.loads((tmp_path / "meta_data" / "stats.json").read_text())
        assert stats["n_valid_frames"] == 2
        f0 = make_frame(1.0).observation
        f2 = make_frame(1.2, index=2).observation
        s0 = np.array(list(f0.joint_position) + list(f0.joint_velocity) + list(f0.joint_current) + list(f0.tcp_pose))
        s2 = np.array(list(f2.joint_position) + list(f2.joint_velocity) + list(f2.joint_current) + list(f2.tcp_pose))
        manual = np.stack([s0, s2])
        assert np.allclose(stats["observation.state"]["mean"], manual.mean(axis=0))
        assert np.allclose(stats["observation.state"]["std"], manual.std(axis=0))

    def test_episode_metadata_written(self, tmp_path):
        w = DatasetWriter(str(tmp_path))
        w.start_episode("rebar_tying")
        w.append_frame(make_frame(1.0))
        w.write_metadata()
        info = json.loads((tmp_path / "meta_data" / "info.json").read_text())
        ep = info["episodes"][0]
        assert ep["task_name"] == "rebar_tying"
        assert ep["task_index"] == 0
        assert ep["frame_count"] == 1
        assert ep["data_path"].startswith("data/episode_000000.")
        assert ep["backend"] in ("jsonl", "parquet")
        assert "task_index_map" in info
        imu_rows = read_jsonl(tmp_path / "imu" / "episode_000000.jsonl")
        assert len(imu_rows) == 1
        assert len(imu_rows[0]["gyro"]) == 3 and len(imu_rows[0]["accel"]) == 3

    def test_image_files_actually_created(self, tmp_path):
        w = DatasetWriter(str(tmp_path))
        w.start_episode("task")
        w.append_frame(make_frame(1.0))
        w.finish_episode()
        r = read_rows(tmp_path / "data" / "episode_000000")[0]
        for key in (
            "observation.images.rgb.path",
            "observation.images.depth.path",
            "observation.images.aligned_depth.path",
        ):
            p = tmp_path / r[key]
            assert p.is_file(), f"{p} 不存在"
        # 可读回；depth 为 16-bit 无损 PNG（PIL 读回为 int32 容器，
        # 数值范围 ≤ 65535，转回 uint16 后与源数组逐值一致）
        from PIL import Image

        rgb = np.array(Image.open(tmp_path / r["observation.images.rgb.path"]))
        assert rgb.shape == (480, 640, 3) and rgb.dtype == np.uint8
        depth_raw = np.array(Image.open(tmp_path / r["observation.images.depth.path"]))
        assert depth_raw.shape == (480, 640)
        assert depth_raw.max() <= 65535
        depth = depth_raw.astype(np.uint16)
        assert np.array_equal(depth, make_frame(1.0).camera_packet.depth)

    def test_timestamps_monotonic(self, tmp_path):
        w = DatasetWriter(str(tmp_path))
        w.start_episode("task")
        for i in range(10):
            w.append_frame(make_frame(1.0 + i * 0.05, index=i))
        w.write_metadata()
        rows = read_rows(tmp_path / "data" / "episode_000000")
        stamps = [r["timestamp"] for r in rows]
        assert stamps == sorted(stamps)
        assert all(b > a for a, b in zip(stamps, stamps[1:]))
        info = json.loads((tmp_path / "meta_data" / "info.json").read_text())
        assert info["episodes"][0]["timestamp_non_monotonic"] == 0


class TestOfflineDatasetE2E:
    def test_100_frames_one_episode(self, tmp_path):
        import random

        rng = random.Random(42)
        obs_list, cam_list = [], []
        for i in range(100):
            o = make_mock_observation(i, invalid_every=5)
            obs_list.append(replace(o, timestamp=i / 30.0 + rng.uniform(-0.004, 0.004)))
            if i % 17 == 16:
                cam_list.append(None)
                continue
            c = make_mock_camera_packet(i, fps=30.0, invalid_every=11)
            cam_list.append(replace(c, timestamp=i / 30.0 + rng.uniform(-0.005, 0.005)))

        frames = synchronize_observations(
            obs_list, cam_list, max_time_delta=0.05, require_camera=True
        )
        valid_frames = [f for f in frames if f.valid]
        assert valid_frames

        w = DatasetWriter(str(tmp_path))
        w.start_episode("rebar_tying")
        written = sum(1 for f in frames if w.append_frame(f, action=None))
        w.write_metadata()

        # 有效帧数量正确；invalid 帧被过滤
        assert written == len(valid_frames)
        info = json.loads((tmp_path / "meta_data" / "info.json").read_text())
        ep = info["episodes"][0]
        assert ep["frame_count"] == written
        assert ep["skipped_frames"] == len(frames) - written

        # 数据集可重新读取
        rows = read_rows(tmp_path / "data" / "episode_000000")
        assert len(rows) == written
        stamps = [r["timestamp"] for r in rows]
        assert stamps == sorted(stamps)
        for r in rows:
            assert len(r["observation.state"]) == 25
            assert r["valid"] is True
            assert r["action"] is None

        # 图像可读取（第一帧），depth 无损且数值一致
        first = rows[0]
        from PIL import Image

        rgb = np.array(Image.open(tmp_path / first["observation.images.rgb.path"]))
        assert rgb.shape == (480, 640, 3)
        depth_raw = np.array(Image.open(tmp_path / first["observation.images.depth.path"]))
        assert depth_raw.shape == (480, 640) and depth_raw.max() <= 65535
        assert np.array_equal(depth_raw.astype(np.uint16), valid_frames[0].camera_packet.depth)

        # metadata JSON 不含原始像素
        for meta in ("info.json", "stats.json"):
            text = (tmp_path / "meta_data" / meta).read_text()
            assert len(text) < 20_000
            assert "ndarray" not in text
        stats = json.loads((tmp_path / "meta_data" / "stats.json").read_text())
        assert stats["n_valid_frames"] == written
        assert len(stats["observation.state"]["mean"]) == 25


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
