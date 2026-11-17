"""Compatibility checker + converter tests (no ROS2, no training, no hardware)."""

import hashlib
import json
from dataclasses import replace

import pytest

from rml63_bridge.camera_packet import make_mock_camera_packet
from rml63_bridge.dataset_compat_check import (
    ACT_TRAINING_ACTION_MESSAGE,
    check_dataset,
)
from rml63_bridge.convert_to_lerobot import apply_conversion, plan_conversion
from rml63_bridge.dataset_writer import DatasetWriter
from rml63_bridge.mock_data import make_mock_observation
from rml63_bridge.synchronized_frame import SynchronizedFrame, synchronize_observations


def make_frame(ts, index=0, valid=True):
    o = make_mock_observation(index, invalid_every=0)
    o = replace(o, timestamp=ts)
    c = make_mock_camera_packet(index, invalid_every=0)
    c = replace(c, timestamp=ts)
    return SynchronizedFrame(
        timestamp=ts,
        observation=o,
        camera_packet=c,
        observation_timestamp=ts,
        camera_timestamp=ts,
        time_delta=0.0,
        valid=valid,
        invalid_reasons=[] if valid else ["test invalid"],
    )


def read_staging_rows(root):
    """读回 staging episode_000000 的行（jsonl/parquet 双后端）。"""
    jsonl = root / "data" / "episode_000000.jsonl"
    if jsonl.is_file():
        return [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    parquet = root / "data" / "episode_000000.parquet"
    if parquet.is_file():
        pytest.importorskip("pyarrow", reason="读取 parquet 需要 pyarrow")
        import pyarrow.parquet as pq

        return pq.read_table(parquet).to_pylist()
    raise FileNotFoundError(f"未找到 episode 数据文件: {root}/data/episode_000000.*")


def build_offline_dataset(tmp_path, n=6, action_dim=None, drop_image=False, with_invalid=False):
    """用 DatasetWriter 生成小数据集；action_dim=None -> action 全为 null。"""
    root = tmp_path / "ds"
    obs_list = [
        replace(make_mock_observation(i, invalid_every=0), timestamp=i / 30.0) for i in range(n)
    ]
    cam_list = [
        replace(make_mock_camera_packet(i, invalid_every=0), timestamp=i / 30.0) for i in range(n)
    ]
    frames = synchronize_observations(obs_list, cam_list, max_time_delta=0.05)
    if with_invalid:
        frames[2] = replace(frames[2], valid=False, invalid_reasons=["test invalid"])

    w = DatasetWriter(str(root))
    w.start_episode("task_a")
    for f in frames:
        action = None if action_dim is None else [0.01 * j for j in range(action_dim)]
        w.append_frame(f, action=action)
    w.write_metadata()

    if drop_image:
        r = read_staging_rows(root)[0]
        (root / r["observation.images.rgb.path"]).unlink()
    return root


class TestCompatChecker:
    def test_recognizes_jsonl_png_dataset(self, tmp_path):
        root = build_offline_dataset(tmp_path)
        report = check_dataset(str(root))
        # 后端取决于环境有无 pyarrow（dora-rs 依赖 pyarrow）：
        # jsonl 后端需提示 Parquet；parquet 后端仍需提示 meta/ 布局
        assert report.backend in (
            "jsonl+png (rml63_bridge 离线后端)",
            "parquet+png (rml63_bridge 离线后端)",
        )
        if report.backend.startswith("jsonl"):
            assert any(i.category == "backend" and "Parquet" in i.message for i in report.issues)
        assert any(i.category == "layout" and "meta/" in i.message for i in report.issues)

    def test_recognizes_25_dim_state(self, tmp_path):
        root = build_offline_dataset(tmp_path)
        report = check_dataset(str(root))
        assert report.state_dim == 25
        assert not any("state 维度" in i.message for i in report.issues)

    def test_action_null_reports_not_trainable(self, tmp_path):
        root = build_offline_dataset(tmp_path, action_dim=None)
        report = check_dataset(str(root))
        assert report.n_frames_action_null == 6
        assert report.n_frames_with_action == 0
        assert not report.trainable
        assert any(i.level == "critical" and i.message == ACT_TRAINING_ACTION_MESSAGE for i in report.issues)

    def test_wrong_action_dim_reports_error(self, tmp_path):
        root = build_offline_dataset(tmp_path, action_dim=5)
        report = check_dataset(str(root), action_dim=6)
        assert any(i.category == "action" and "维度 5 != 期望 6" in i.message for i in report.issues)
        assert report.trainable  # action 存在，只是维度不匹配 -> error 而非 critical

    def test_missing_image_file_reports_error(self, tmp_path):
        root = build_offline_dataset(tmp_path, action_dim=6, drop_image=True)
        report = check_dataset(str(root))
        assert any(i.category == "image" and "图像文件缺失" in i.message for i in report.issues)

    def test_stats_valid_invalid_correct(self, tmp_path):
        root = build_offline_dataset(tmp_path, action_dim=6, with_invalid=True)
        report = check_dataset(str(root))
        assert report.n_frames == 5  # invalid 帧未写入
        stats = json.loads((root / "meta_data" / "stats.json").read_text())
        assert stats["n_valid_frames"] == 5
        info = json.loads((root / "meta_data" / "info.json").read_text())
        assert info["episodes"][0]["skipped_frames"] == 1


class TestConverter:
    def test_dry_run_does_not_touch_input(self, tmp_path):
        root = build_offline_dataset(tmp_path, action_dim=6)
        out = tmp_path / "converted"

        def snapshot():
            hashes = {}
            for f in sorted(root.rglob("*")):
                if f.is_file():
                    hashes[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
            return hashes

        before = snapshot()
        plan = plan_conversion(str(root), str(out), action_dim=6, dry_run=True)
        assert plan.frames, plan.errors
        assert plan.skipped_action_null == 0
        errors = apply_conversion(plan)  # dry_run=True -> 不写任何文件
        assert errors == []
        assert not out.exists()  # 输出目录未创建
        assert before == snapshot()  # 输入数据逐字节未变

    def test_action_null_refused(self, tmp_path):
        root = build_offline_dataset(tmp_path, action_dim=None)
        plan = plan_conversion(str(root), str(tmp_path / "out"), action_dim=6, dry_run=True)
        assert plan.frames == []
        assert any(ACT_TRAINING_ACTION_MESSAGE in e for e in plan.errors)

    def test_no_action_faking(self, tmp_path):
        root = build_offline_dataset(tmp_path, action_dim=6)
        plan = plan_conversion(str(root), str(tmp_path / "out"), action_dim=6, dry_run=True)
        assert plan.frames
        src_rows = read_staging_rows(root)
        actions = [r["action"] for r in src_rows]
        # 转换后 action 与原始完全一致（未被 state 冒充）
        for converted, original in zip(plan.frames, src_rows):
            assert converted["action"] == original["action"]
            assert converted["observation.state"] == original["observation.state"]
        assert all(len(a) == 6 for a in actions)
        # 跳过帧统计: 全部有 action -> 无跳过
        assert plan.skipped_action_null == 0

    def test_mixed_null_actions_skipped(self, tmp_path):
        root = build_offline_dataset(tmp_path, action_dim=6)
        # 人为把第 2 帧 action 改为 null（写回原后端）
        rows = read_staging_rows(root)
        rows[2]["action"] = None
        jsonl = root / "data" / "episode_000000.jsonl"
        if jsonl.is_file():
            jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        else:
            pytest.importorskip("pyarrow", reason="写 parquet 需要 pyarrow")
            import pyarrow as pa
            import pyarrow.parquet as pq

            pq.write_table(
                pa.Table.from_pylist(rows), root / "data" / "episode_000000.parquet"
            )
        plan = plan_conversion(str(root), str(tmp_path / "out"), action_dim=6, dry_run=True)
        assert plan.skipped_action_null == 1
        assert len(plan.frames) == 5  # 其余 5 帧被转换
        assert all(f["action"] is not None for f in plan.frames)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
