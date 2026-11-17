"""LeRobot 0.6.2 Parquet 格式验证测试（不训练、不启动 ROS2、不连接硬件）。

parquet 相关测试需要 pyarrow（lerobot conda env 已安装），在无 pyarrow 的
环境下自动跳过；LeRobot 加载接口测试需要 datasets/pandas，缺失时跳过并
记录原因。

模拟 action 仅用于数据格式验证，不代表真实示范动作，不用于研究结论。
"""

import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from rml63_bridge.camera_packet import make_mock_camera_packet
from rml63_bridge.convert_to_lerobot import (
    ACT_TRAINING_ACTION_MESSAGE,
    MOCK_ACTION_NOTICE,
    apply_conversion,
    plan_conversion,
)
from rml63_bridge.dataset_writer import DatasetWriter
from rml63_bridge.mock_data import make_mock_observation
from rml63_bridge.synchronized_frame import synchronize_observations


def mock_action(i, dim=6):
    """模拟 action 仅用于数据格式验证，不代表真实示范动作，不用于研究结论。"""
    return [0.01 * j + 0.005 * math.sin(0.3 * i) for j in range(dim)]


def build_action_dataset(tmp_path, n=6, with_invalid=False, all_null_action=False):
    root = tmp_path / "src_ds"
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
    w.start_episode("rebar_tying")
    for i, f in enumerate(frames):
        action = None if all_null_action else mock_action(i)
        w.append_frame(f, action=action)
    w.write_metadata()
    return root


def snapshot_hashes(root: Path) -> dict:
    out = {}
    for f in sorted(root.rglob("*")):
        if f.is_file():
            out[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


@pytest.fixture(scope="module")
def pq():
    """pyarrow.parquet（无 pyarrow 时按测试跳过，不影响收集）。"""
    pytest.importorskip("pyarrow", reason="需要 pyarrow（lerobot conda env）")
    import pyarrow.parquet as parquet

    return parquet


@pytest.fixture(scope="class")
def converted(tmp_path_factory):
    pytest.importorskip("pyarrow", reason="需要 pyarrow（lerobot conda env）")
    tmp = tmp_path_factory.mktemp("fmt")
    src = build_action_dataset(tmp, n=6, with_invalid=True)
    out = tmp / "out_ds"
    plan = plan_conversion(str(src), str(out), action_dim=6, fps=30, dry_run=False)
    errors = apply_conversion(plan)
    assert errors == [], errors
    return src, out


class TestConvertedLayout:
    def test_directory_structure(self, converted):
        _, out = converted
        for rel in (
            "meta/info.json",
            "meta/stats.json",
            "meta/tasks.parquet",
            "meta/episodes/chunk-000/file-000.parquet",
            "data/chunk-000/file-000.parquet",
        ):
            assert (out / rel).is_file(), f"缺少 {rel}"
        for sub in ("rgb", "depth", "aligned_depth"):
            assert (out / "images" / sub).is_dir()

    def test_info_json_fields(self, converted):
        _, out = converted
        info = json.loads((out / "meta" / "info.json").read_text())
        for key in (
            "codebase_version",
            "fps",
            "features",
            "robot_type",
            "total_episodes",
            "total_frames",
            "total_tasks",
        ):
            assert key in info, f"info.json 缺少 {key}"
        assert info["total_episodes"] == 1
        assert info["total_frames"] == 5  # 6 帧中 1 帧 invalid 被过滤
        assert "action" in info["features"]
        assert info["features"]["observation.state"]["shape"] == [25]

    def test_tasks_parquet_readable(self, converted, pq):

        _, out = converted
        t = pq.read_table(out / "meta" / "tasks.parquet")
        assert t.num_rows >= 1
        assert "task_index" in t.column_names

    def test_episodes_parquet_readable(self, converted, pq):

        _, out = converted
        t = pq.read_table(out / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
        assert t.num_rows == 1
        row = t.to_pylist()[0]
        assert row["episode_index"] == 0
        assert row["length"] == 5
        assert "data/chunk_index" in row and "data/file_index" in row

    def test_data_parquet_readable(self, converted, pq):

        _, out = converted
        t = pq.read_table(out / "data" / "chunk-000" / "file-000.parquet")
        assert t.num_rows == 5
        for col in (
            "timestamp",
            "episode_index",
            "frame_index",
            "task_index",
            "index",
            "observation.state",
            "action",
            "observation.images.rgb",
            "observation.images.depth",
            "observation.images.aligned_depth",
        ):
            assert col in t.column_names, f"缺列 {col}"

    def test_observation_state_shape_25(self, converted, pq):

        _, out = converted
        t = pq.read_table(out / "data" / "chunk-000" / "file-000.parquet")
        for row in t["observation.state"].to_pylist():
            assert len(row) == 25

    def test_action_shape_6(self, converted, pq):

        _, out = converted
        t = pq.read_table(out / "data" / "chunk-000" / "file-000.parquet")
        for row in t["action"].to_pylist():
            assert len(row) == 6

    def test_action_dtype_float32(self, converted, pq):

        _, out = converted
        t = pq.read_table(out / "data" / "chunk-000" / "file-000.parquet")
        s = t.schema.field("observation.state").type
        a = t.schema.field("action").type
        assert s.value_type == "float", f"observation.state 不是 float32: {s}"
        assert a.value_type == "float", f"action 不是 float32: {a}"

    def test_invalid_frames_excluded(self, converted, pq):

        _, out = converted
        t = pq.read_table(out / "data" / "chunk-000" / "file-000.parquet")
        assert t.num_rows == 5  # 6 帧输入, 1 帧 invalid -> 不进入 Parquet

    def test_action_null_rejected(self, tmp_path):
        root = build_action_dataset(tmp_path, n=4, all_null_action=True)
        plan = plan_conversion(str(root), str(tmp_path / "out"), action_dim=6, dry_run=True)
        assert plan.frames == []
        assert any(ACT_TRAINING_ACTION_MESSAGE in e for e in plan.errors)

    def test_original_data_untouched(self, tmp_path):
        pytest.importorskip("pyarrow", reason="需要 pyarrow（lerobot conda env）")
        root = build_action_dataset(tmp_path, n=4)
        before = snapshot_hashes(root)
        out = tmp_path / "out_ds"
        plan = plan_conversion(str(root), str(out), action_dim=6, dry_run=False)
        assert apply_conversion(plan) == []
        assert before == snapshot_hashes(root)  # 原始 JSONL+PNG 逐字节未变

    def test_dry_run_produces_no_output(self, tmp_path):
        root = build_action_dataset(tmp_path, n=4)
        out = tmp_path / "dry_out"
        plan = plan_conversion(str(root), str(out), action_dim=6, dry_run=True)
        assert apply_conversion(plan) == []
        assert not out.exists()

    def test_image_paths_absolute_and_readable(self, converted, pq):
        # 注意: LeRobot 0.6.2 图像列存绝对路径（PIL 直接 open，不做相对解析）
        from PIL import Image

        _, out = converted
        t = pq.read_table(out / "data" / "chunk-000" / "file-000.parquet")
        for key in ("rgb", "depth", "aligned_depth"):
            p = t[f"observation.images.{key}"][0].as_py()
            assert p.startswith("/")
            assert Path(p).is_file()
            img = Image.open(p)
            assert img.size == (640, 480)

    def test_mock_action_differs_from_state(self, converted, pq):

        _, out = converted
        t = pq.read_table(out / "data" / "chunk-000" / "file-000.parquet")
        state = t["observation.state"][0].as_py()
        action = t["action"][0].as_py()
        assert action != state[:6]  # 模拟 action 不复制 observation.state
        assert len(action) == 6 and len(state) == 25

    def test_lerobot_loading_interface(self, converted):
        """尝试用 LeRobot 0.6.2 加载接口读取转换后的数据集。

        加载链路需要 datasets + pandas（lerobot/datasets/dataset_reader.py
        顶层 import datasets；io_utils 用 pandas 读 tasks.parquet）。当前
        lerobot conda env 未安装 datasets/pandas，按指示不擅自安装——
        记录缺失并跳过实际加载。
        """
        src_root, out = converted
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lerobot" / "src"))
        try:
            import datasets  # noqa: F401
            import pandas  # noqa: F401
        except ImportError as exc:
            pytest.skip(f"LeRobot 加载依赖缺失: {exc.name}（未安装 datasets/pandas，按指示不擅自安装）")
        from lerobot.datasets import LeRobotDataset

        ds = LeRobotDataset(str(out))
        assert len(ds) == 5
        features = ds.features
        assert "observation.state" in features and "action" in features
        item = ds[0]
        assert len(item["observation.state"]) == 25
        assert len(item["action"]) == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
