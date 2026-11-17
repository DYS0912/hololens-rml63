"""Dora 数据采集层离线测试（不安装 Dora、不启动 ROS2、不连接硬件）。

验证范围:
  - dataflow YAML 可解析且通道连接与节点输出一致；
  - mock state / mock CameraPacket 可以进入 recorder；
  - 时间同步、episode 控制协议、invalid 过滤、action 缺失拒绝；
  - 图像绝不进入 JSON；生成数据可被 LeRobot 0.6.2 读取（pyarrow 门控）；
  - 导入节点模块不触发 rclpy/dora/cv2 等真实运行依赖。

模拟 action 仅用于数据格式验证，不代表真实示范动作，不用于研究结论。
"""

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from rml63_bridge.camera_packet import make_mock_camera_packet
from rml63_bridge.convert_to_lerobot import ACT_TRAINING_ACTION_MESSAGE
from rml63_bridge.dora_nodes import (
    d435i_camera_dora_node,
    mock_action_dora_node,
    rml63_dataset_recorder_dora_node,
    rml63_state_dora_node,
)
from rml63_bridge.dora_nodes.dora_compat import value_to_py
from rml63_bridge.dora_nodes.d435i_camera_dora_node import CameraNodeCore
from rml63_bridge.dora_nodes.mock_action_dora_node import MockActionCore
from rml63_bridge.dora_nodes.rml63_dataset_recorder_dora_node import (
    RecorderNodeCore,
    recover_offline_dataset,
)
from rml63_bridge.dora_nodes.rml63_state_dora_node import StateNodeCore
from rml63_bridge.dora_nodes.simulate_dataflow import run_dataflow_simulation

PKG_ROOT = Path(__file__).resolve().parents[1]
DATAFLOW_FILE = PKG_ROOT / "dataflows" / "rml63_d435i_record.yml"

HAS_PYARROW = importlib.util.find_spec("pyarrow") is not None


@pytest.fixture(scope="module")
def pq():
    """pyarrow.parquet（无 pyarrow 时相关测试跳过，不影响收集）。"""
    pytest.importorskip("pyarrow", reason="需要 pyarrow（lerobot conda env）")
    import pyarrow.parquet as parquet

    return parquet


def read_staging_rows(dataset_root):
    """读回 staging episode 数据（jsonl 或 parquet 双后端）。"""
    rows = []
    files = sorted(Path(dataset_root).glob("data/episode_*"))
    assert files, f"staging 无 episode 数据文件: {dataset_root}"
    for f in files:
        if f.suffix == ".jsonl":
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        else:
            pytest.importorskip("pyarrow")
            import pyarrow.parquet as parquet

            rows.extend(parquet.read_table(f).to_pylist())
    return rows


# ---------------------------------------------------------------------------
# 1. dataflow YAML
# ---------------------------------------------------------------------------

class TestDataflowYaml:
    def test_yaml_parses_and_references_existing_paths(self):
        dataflow = yaml.safe_load(DATAFLOW_FILE.read_text(encoding="utf-8"))
        assert "nodes" in dataflow
        nodes = dataflow["nodes"]
        ids = [n["id"] for n in nodes]
        assert ids == ["rml63_state", "d435i_camera", "mock_action", "rml63_dataset_recorder"]
        assert len(set(ids)) == len(ids), "节点 id 重复"
        for n in nodes:
            assert Path(n["path"]).is_file(), f"节点脚本不存在: {n['path']}"
            assert isinstance(n["inputs"], dict) and isinstance(n["outputs"], list)

    def test_yaml_channel_wiring_matches_node_outputs(self):
        dataflow = yaml.safe_load(DATAFLOW_FILE.read_text(encoding="utf-8"))
        nodes = {n["id"]: n for n in dataflow["nodes"]}
        state_outputs = set(nodes["rml63_state"]["outputs"])
        camera_outputs = set(nodes["d435i_camera"]["outputs"])
        action_outputs = set(nodes["mock_action"]["outputs"])
        rec_inputs = dict(nodes["rml63_dataset_recorder"]["inputs"])

        assert {"state", "tcp_pose", "joint_current", "joint_velocity",
                "status", "timestamp", "valid"} <= state_outputs
        assert {"rgb", "depth", "aligned_depth", "gyro", "accel",
                "camera_timestamp", "frame_id", "valid"} <= camera_outputs
        assert action_outputs == {"action"}
        # recorder 的所有 state/camera/action 输入都来自对应节点
        assert rec_inputs["state"] == "rml63_state/state"
        assert rec_inputs["rgb"] == "d435i_camera/rgb"
        assert rec_inputs["action"] == "mock_action/action"
        assert rec_inputs["state_valid"] == "rml63_state/valid"
        assert rec_inputs["camera_valid"] == "d435i_camera/valid"

    def test_yaml_no_ros2_topics_wired_in_mock_mode(self):
        """离线模式下 dataflow 不引用任何真实 ROS2 话题（仅注释）。"""
        text = DATAFLOW_FILE.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert "/joint_states" not in stripped
            assert "/camera/camera/" not in stripped


# ---------------------------------------------------------------------------
# 2. 节点核心：mock 输入
# ---------------------------------------------------------------------------

class TestStateNodeCore:
    def test_mock_outputs_valid_frame(self):
        core = StateNodeCore(input_mode="mock")
        out = core.on_tick()
        assert out["valid"] is True
        state = out["state"]
        assert isinstance(state, np.ndarray) and state.shape == (25,)
        assert state.dtype == np.float32
        assert out["tcp_pose"].shape == (7,)
        assert out["joint_current"].shape == (6,)
        assert out["joint_velocity"].shape == (6,)
        assert isinstance(out["status"], int)
        assert isinstance(out["timestamp"], float)

    def test_mock_invalid_frame_never_zero_filled(self):
        core = StateNodeCore(input_mode="mock", mock_invalid_every=5)
        for _ in range(5):
            core.on_tick()
        out = core.on_tick()  # index 5 -> invalid
        assert out["valid"] is False
        assert out["state"] is None  # 缺失即 None，绝不零填充
        assert out["joint_current"] is None

    def test_state_layout_25(self):
        core = StateNodeCore(input_mode="mock")
        out = core.on_tick()
        obs = core.latest_observation
        state = out["state"]
        np.testing.assert_allclose(state[0:6], obs.joint_position)
        np.testing.assert_allclose(state[6:12], obs.joint_velocity)
        np.testing.assert_allclose(state[12:18], obs.joint_current)
        np.testing.assert_allclose(state[18:25], obs.tcp_pose)


class TestCameraNodeCore:
    def test_mock_outputs_images_as_arrays(self):
        core = CameraNodeCore(input_mode="mock")
        out = core.on_tick()
        assert out["valid"] is True
        assert isinstance(out["rgb"], np.ndarray)
        assert out["rgb"].shape == (480, 640, 3) and out["rgb"].dtype == np.uint8
        assert isinstance(out["depth"], np.ndarray)
        assert out["depth"].shape == (480, 640) and out["depth"].dtype == np.uint16
        assert out["aligned_depth"].dtype == np.uint16
        assert out["gyro"].shape == (3,)
        assert isinstance(out["frame_id"], int)
        assert isinstance(out["camera_timestamp"], float)

    def test_mock_invalid_frame(self):
        core = CameraNodeCore(input_mode="mock", mock_invalid_every=5)
        for _ in range(5):
            core.on_tick()
        out = core.on_tick()  # index 5 -> drop
        assert out["valid"] is False
        assert out["rgb"] is None

    def test_images_never_json_strings(self):
        """相机输出是数值数组；JSON 序列化不允许（像素绝不进 JSON 字符串）。"""
        core = CameraNodeCore(input_mode="mock")
        out = core.on_tick()
        for key in ("rgb", "depth", "aligned_depth"):
            value = out[key]
            assert isinstance(value, np.ndarray), f"{key} 不是 numpy 数组"
            with pytest.raises(TypeError):
                json.dumps({key: value})  # 数组不可直接 JSON 序列化


class TestMockActionCore:
    def test_mock_action_6d_deterministic(self):
        core = MockActionCore(mode="mock")
        a0 = core.on_tick()
        assert len(a0) == 6
        assert all(isinstance(x, float) for x in a0)
        core2 = MockActionCore(mode="mock")
        assert core2.on_tick() == a0

    def test_none_mode_outputs_nothing(self):
        core = MockActionCore(mode="none")
        assert core.on_tick() is None


# ---------------------------------------------------------------------------
# 3. recorder：同步 / episode 协议 / 过滤 / action 拒绝
# ---------------------------------------------------------------------------

def _mk_recorder(tmp_path, **kwargs):
    defaults = dict(
        dataset_root=str(tmp_path / "stage"),
        output_root=str(tmp_path / "out"),
        camera_start_timeout=0.5,
        require_action=False,
    )
    defaults.update(kwargs)
    return RecorderNodeCore(**defaults)


def _valid_state(timestamp):
    return np.linspace(0.1, 2.5, 25, dtype=np.float32)


def _camera_values(timestamp, index=0):
    p = make_mock_camera_packet(index)
    return {
        "rgb": p.rgb,
        "depth": p.depth,
        "aligned_depth": p.aligned_depth,
        "gyro": p.gyro,
        "accel": p.accel,
        "camera_timestamp": timestamp,
        "frame_id": index,
        "valid": True,
    }


class TestRecorderSyncAndFiltering:
    def test_mock_state_and_camera_enter_recorder(self, tmp_path):
        """完整链路: mock state + mock CameraPacket -> recorder -> staging 数据。"""
        result = run_dataflow_simulation(
            count=4, invalid_every=0, action_mode="mock",
            dataset_root=str(tmp_path / "stage"), output_root=str(tmp_path / "out"),
        )
        assert result.frames_written == 4
        rows = read_staging_rows(result.dataset_root)
        assert len(rows) == 4
        for r in rows:
            assert r["valid"] is True
            assert len(r["observation.state"]) == 25
            assert len(r["action"]) == 6  # 模拟 action 仅用于格式验证
            assert r["observation.images.rgb.path"].endswith(".png")
            assert r["action"] != r["observation.state"][:6]

    def test_time_sync_in_recorder(self, tmp_path):
        """超出 max_time_delta 的相机帧被拒绝; 窗口内最近帧被选中。"""
        rec = _mk_recorder(tmp_path, camera_start_timeout=0.1)
        rec.on_state(state=_valid_state(1.0), timestamp=1.0, valid=True, now=1.0)
        # 未来过远 (+0.10) -> 拒绝
        rec.on_camera(**_camera_values(1.10), now=1.0)
        rec.on_tick(now=1.0)
        assert rec.skipped["frame_invalid"] == 1
        assert rec.frames_written == 0
        # 过去 0.05s -> 窗口内, 选中
        rec.on_camera(**_camera_values(0.95, index=1), now=1.0)
        rec.on_tick(now=1.0)
        assert rec.frames_written == 1
        rec.finalize()
        rows = read_staging_rows(rec.dataset_root)
        assert len(rows) == 1
        assert abs(rows[0]["timestamp"] - 1.0) < 1e-6

    def test_invalid_frames_filtered(self, tmp_path):
        result = run_dataflow_simulation(
            count=12, invalid_every=5, action_mode="mock",
            dataset_root=str(tmp_path / "stage"), output_root=str(tmp_path / "out"),
        )
        # index 5 / 10 是 invalid（state 与 camera 同帧丢弃）
        assert result.frames_written == 10
        assert result.skipped["frame_invalid"] == 2
        rows = read_staging_rows(result.dataset_root)
        assert all(r["valid"] for r in rows)

    def test_recorder_reshapes_flat_transport_images(self, tmp_path):
        """回归: dora 传输的图像是 ravel 后的扁平数组（真实 dora runtime
        运行发现的 bug），recorder 必须按 camera_shapes 还原形状。"""
        rec = _mk_recorder(tmp_path, camera_shapes={"d435i": (480, 640)})
        p = make_mock_camera_packet(0)
        rec.on_state(state=_valid_state(0.0), timestamp=0.0, valid=True, now=0.0)
        rec.on_camera(
            rgb=p.rgb.ravel(),
            depth=p.depth.ravel(),
            aligned_depth=p.aligned_depth.ravel(),
            gyro=p.gyro,
            accel=p.accel,
            camera_timestamp=0.0,
            frame_id=0,
            valid=True,
            now=0.0,
        )
        rec.on_tick(now=0.0)
        assert rec.frames_written == 1, dict(rec.skipped)

    def test_camera_stale_detection(self, tmp_path):
        rec = _mk_recorder(tmp_path)
        rec.on_state(state=_valid_state(0.0), timestamp=0.0, valid=True, now=0.0)
        rec.on_camera(**_camera_values(0.0), now=0.0)
        rec.on_tick(now=4.0)  # 图像 4s 未更新 -> 陈旧, 跳过
        assert rec.skipped["camera_stale"] == 1
        assert rec.frames_written == 0

    def test_camera_start_timeout_fatal(self, tmp_path):
        rec = _mk_recorder(tmp_path, camera_start_timeout=0.5)
        rec.on_state(state=_valid_state(0.0), timestamp=0.0, valid=True, now=0.0)
        rec.on_tick(now=0.0)  # 相机未就绪, 开始等待
        out = rec.on_tick(now=0.6)  # 超时 -> 致命错误
        assert out.get("status") == "error"
        assert rec.fatal_error is not None
        assert "相机" in rec.fatal_error
        final = rec.finalize()
        assert final["status"] == "error"


class TestEpisodeLifecycle:
    def test_save_next_discard_stop(self, tmp_path):
        commands = {2: "save", 3: "next", 5: "discard", 6: "next"}
        result = run_dataflow_simulation(
            count=10, invalid_every=0, action_mode="mock",
            dataset_root=str(tmp_path / "stage"), output_root=str(tmp_path / "out"),
            control_commands=commands,
        )
        # 帧 0,1 -> episode 0 (save); 帧 3,4 -> 被 discard; 帧 6..9 -> 新 episode。
        # 注意: 每个 save/next/discard 后的第一个 tick 会被相机首帧等待跳过
        # （同 dorobot2 语义：场景复位后必须等各相机重发新帧才开始录制）——
        # 因此 next 后的帧 3 与帧 6 不入库。
        assert result.frames_written == 6
        assert result.episodes_committed == 2
        stage = Path(result.dataset_root)
        episode_files = sorted(stage.glob("data/episode_*"))
        assert len(episode_files) == 2, "discard 的 episode 不应写盘"
        names = {f.name for f in episode_files}
        assert "episode_000000.jsonl" in names or "episode_000000.parquet" in names
        assert "episode_000001.jsonl" in names or "episode_000001.parquet" in names
        # 图像目录: episode 0 有 2 帧; episode 1 目录被 discard 清空后由新
        # episode 重新写入 3 帧（帧 7,8,9；若 discard 未清理则会残留帧 4 的图）
        assert len(list((stage / "videos" / "rgb" / "episode_000000").glob("*.png"))) == 2
        assert len(list((stage / "videos" / "rgb" / "episode_000001").glob("*.png"))) == 3
        rows = read_staging_rows(str(stage))
        # 持久化行 = 2 (ep0) + 3 (ep1)；被 discard 的帧 4 已从数据与图像中移除
        assert len(rows) == 5
        assert {r["episode_index"] for r in rows} == {0, 1}
        assert sum(1 for r in rows if r["episode_index"] == 0) == 2
        assert sum(1 for r in rows if r["episode_index"] == 1) == 3

    def test_control_file_protocol(self, tmp_path):
        ctl = tmp_path / "record.ctl"
        rec = _mk_recorder(tmp_path, control_file=str(ctl), camera_start_timeout=0.1)
        rec.on_state(state=_valid_state(0.0), timestamp=0.0, valid=True, now=0.0)
        rec.on_camera(**_camera_values(0.0), now=0.0)

        def write_cmd(cmd, t):
            ctl.write_text(cmd)
            os.utime(ctl, ns=(int(t * 1e9), int(t * 1e9)))

        write_cmd("save", 1.0)
        assert rec.poll_control() == "save"
        assert rec.poll_control() is None  # 同一 mtime 不重复触发
        write_cmd("next", 2.0)
        assert rec.poll_control() == "next"
        write_cmd("stop", 3.0)
        assert rec.poll_control() == "stop"

    def test_stop_command_via_simulation(self, tmp_path):
        result = run_dataflow_simulation(
            count=20, invalid_every=0, action_mode="mock",
            dataset_root=str(tmp_path / "stage"), output_root=str(tmp_path / "out"),
            control_commands={7: "stop"},
        )
        assert result.frames_written == 7  # 帧 0..6, stop 于 tick 7 前
        assert result.final["status"] == "stopped"


class TestActionPolicy:
    def test_action_missing_refuses_training_data(self, tmp_path):
        out_root = tmp_path / "out"
        result = run_dataflow_simulation(
            count=6, invalid_every=0, action_mode="none",
            dataset_root=str(tmp_path / "stage"), output_root=str(out_root),
            require_action=True,
        )
        assert any(ACT_TRAINING_ACTION_MESSAGE in e for e in result.errors)
        assert not out_root.exists(), "action 全缺时必须拒绝生成训练数据"
        # staging 仍保留（action=null 显式标记），但绝不伪造
        rows = read_staging_rows(result.dataset_root)
        assert len(rows) == 6
        assert all(r["action"] is None for r in rows)
        assert all(len(r["observation.state"]) == 25 for r in rows)

    def test_action_never_copied_from_state(self, tmp_path):
        result = run_dataflow_simulation(
            count=4, invalid_every=0, action_mode="mock",
            dataset_root=str(tmp_path / "stage"), output_root=str(tmp_path / "out"),
        )
        rows = read_staging_rows(result.dataset_root)
        for r in rows:
            assert r["action"] != r["observation.state"][:6]


class TestImagesNeverInJson:
    def test_staging_files_contain_no_pixel_data(self, tmp_path):
        run_dataflow_simulation(
            count=3, invalid_every=0, action_mode="mock",
            dataset_root=str(tmp_path / "stage"), output_root=str(tmp_path / "out"),
        )
        for f in Path(tmp_path / "stage").rglob("*"):
            if not f.is_file() or f.suffix not in (".jsonl", ".json"):
                continue
            text = f.read_text(encoding="utf-8")
            assert "base64" not in text.lower()
            assert "data:image" not in text
        rows = read_staging_rows(str(tmp_path / "stage"))
        for r in rows:
            for key in ("observation.images.rgb.path",
                        "observation.images.depth.path",
                        "observation.images.aligned_depth.path"):
                v = r[key]
                assert isinstance(v, str) and len(v) < 300
                assert v.startswith("videos/")


class TestRecovery:
    def test_recover_after_interruption(self, tmp_path):
        stage = tmp_path / "stage"
        result = run_dataflow_simulation(
            count=8, invalid_every=0, action_mode="mock",
            dataset_root=str(stage), output_root=str(tmp_path / "out"),
            finalize=False,  # 模拟 dora destroy 杀掉 recorder
        )
        assert any("中断" in e for e in result.errors)
        assert not (stage / "meta_data" / "info.json").exists()

        recovered = recover_offline_dataset(str(stage))
        assert recovered["recovered"] is True
        assert recovered["episodes"] == 1
        assert recovered["frames"] == 8
        info = json.loads((stage / "meta_data" / "info.json").read_text(encoding="utf-8"))
        assert info.get("recovered") is True
        stats = json.loads((stage / "meta_data" / "stats.json").read_text(encoding="utf-8"))
        assert stats["n_valid_frames"] == 8
        assert len(stats["observation.state"]["mean"]) == 25

    def test_recover_no_data(self, tmp_path):
        empty = tmp_path / "empty"
        (empty / "data").mkdir(parents=True)
        recovered = recover_offline_dataset(str(empty))
        assert recovered["recovered"] is False
        assert "无数据可恢复" in recovered["error"]


class TestLeRobot062Output:
    def test_lerobot_layout_generated(self, tmp_path, pq):
        """转换产物为 LeRobot 0.6.2 v3.0 布局（pyarrow 门控）。"""
        out = tmp_path / "lerobot_out"
        result = run_dataflow_simulation(
            count=12, invalid_every=5, action_mode="mock",
            dataset_root=str(tmp_path / "stage"), output_root=str(out),
        )
        if not HAS_PYARROW:
            pytest.skip("需要 pyarrow（lerobot conda env）")
        assert result.errors == [], result.errors
        for rel in (
            "meta/info.json",
            "meta/stats.json",
            "meta/tasks.parquet",
            "meta/episodes/chunk-000/file-000.parquet",
            "data/chunk-000/file-000.parquet",
        ):
            assert (out / rel).is_file(), f"缺少 {rel}"
        info = json.loads((out / "meta" / "info.json").read_text(encoding="utf-8"))
        assert info["codebase_version"] == "v3.0"
        assert info["robot_type"] == "rml63"
        assert info["features"]["observation.state"]["shape"] == [25]
        t = pq.read_table(out / "data" / "chunk-000" / "file-000.parquet")
        assert t.num_rows == 10  # 12 帧 - 2 invalid
        assert "action" in t.column_names
        assert "observation.images.rgb" in t.column_names
        for sub in ("rgb", "depth", "aligned_depth"):
            p = t[f"observation.images.{sub}"][0].as_py()
            assert p.startswith("/"), "LeRobot 0.6.2 图像列须存绝对路径"
            assert Path(p).is_file()
        assert t.schema.field("action").type.value_type == "float"

    def test_stats_from_real_state_data(self, tmp_path, pq):
        """observation.state 统计来自真实数据（非默认假值）。"""
        pytest.importorskip("pyarrow", reason="需要 pyarrow（lerobot conda env）")
        out = tmp_path / "lerobot_out"
        run_dataflow_simulation(
            count=8, invalid_every=0, action_mode="mock",
            dataset_root=str(tmp_path / "stage"), output_root=str(out),
        )
        stats = json.loads((out / "meta" / "stats.json").read_text(encoding="utf-8"))
        state_stats = stats["observation.state"]
        assert state_stats["count"] == 8
        mean = np.asarray(state_stats["mean"])
        assert mean.shape == (25,)
        assert np.isfinite(mean).all()
        # 真实数据统计 ≠ 常量假值
        assert not np.allclose(mean, 0.5)

    def test_lerobot_dataset_loading(self, tmp_path, pq):
        """LeRobotDataset 实际加载（需 datasets+pandas，见 test_lerobot_format）。"""
        pytest.importorskip("pyarrow", reason="需要 pyarrow（lerobot conda env）")
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lerobot" / "src"))
        try:
            import datasets  # noqa: F401
            import pandas  # noqa: F401
        except ImportError as exc:
            pytest.skip(f"LeRobot 加载依赖缺失: {exc.name}")
        from lerobot.datasets import LeRobotDataset

        out = tmp_path / "lerobot_out"
        run_dataflow_simulation(
            count=12, invalid_every=5, action_mode="mock",
            dataset_root=str(tmp_path / "stage"), output_root=str(out),
        )
        ds = LeRobotDataset(str(out))
        assert len(ds) == 10
        item = ds[0]
        assert len(item["observation.state"]) == 25
        assert len(item["action"]) == 6
        assert item["action"] != item["observation.state"][:6]


class TestNoRos2NoHardware:
    def test_importing_nodes_does_not_import_rclpy_or_dora(self):
        """独立解释器验证：导入节点模块不加载 rclpy/dora/cv2/pyrealsense2/zmq。"""
        import subprocess

        code = (
            "import sys\n"
            "sys.path.insert(0, {pkg!r})\n"
            "from rml63_bridge.dora_nodes import (\n"
            "    rml63_state_dora_node, d435i_camera_dora_node,\n"
            "    rml63_dataset_recorder_dora_node, mock_action_dora_node,\n"
            ")\n"
            "bad = [m for m in ('rclpy', 'dora', 'cv2', 'pyrealsense2', 'zmq')\n"
            "       if m in sys.modules]\n"
            "print('BAD:' + ','.join(bad) if bad else 'CLEAN')\n"
        ).format(pkg=str(PKG_ROOT))
        proc = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True, text=True, timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        assert "CLEAN" in proc.stdout, proc.stdout

    def test_node_sources_have_no_control_calls(self):
        """节点源码不含任何控制指令（send_goal / publish / rm_movej 等）。"""
        for f in (PKG_ROOT / "rml63_bridge" / "dora_nodes").glob("*.py"):
            text = f.read_text(encoding="utf-8")
            assert "send_goal" not in text, f"{f.name} 含 send_goal"
            assert "create_publisher" not in text, f"{f.name} 含 create_publisher"
            assert "rm_movej" not in text, f"{f.name} 含 rm_movej"

    def test_value_to_py_scalar_unwrap(self):
        class FakeArrow:
            def to_numpy(self, zero_copy_only=True):
                return np.array([1.5])

        assert value_to_py(FakeArrow()) == 1.5
        assert value_to_py(None) is None
        assert value_to_py([1, 2]) == [1, 2]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
