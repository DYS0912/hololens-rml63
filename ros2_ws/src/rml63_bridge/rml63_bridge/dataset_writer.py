"""Offline LeRobot v2-style dataset writer for SynchronizedFrames.

Writes unified training frames to a local dataset directory WITHOUT ROS2,
without hardware, and without the ``datasets`` library:

    dataset_root/
      meta_data/
        info.json          # 数据集与 episode 元数据
        stats.json         # observation.state 统计（仅 valid 帧）
      data/
        episode_000000.jsonl   # 帧级数据（flat 列名，v2 风格；
                               #  安装 pyarrow 后自动切换为 .parquet）
      videos/
        rgb/ depth/ aligned_depth/   # 图像按 episode/frame 存 PNG
      imu/
        episode_000000.jsonl         # 每帧 gyro[3] rad/s, accel[3] m/s^2

Frame rows use flat v2-style keys:
    timestamp, episode_index, frame_index, task_index, valid,
    observation.state (25-dim), observation.images.<name>.path,
    observation.imu.gyro, observation.imu.accel, action

Rules:
  - only valid frames are written; invalid frames are counted as skipped
    and never enter the training data;
  - observation.state = [joint_position[6], joint_velocity[6],
    joint_current[6], tcp_pose[7]]  (25 dims);
  - action=None is stored explicitly as JSON null — the observation.state
    is NEVER used to fake an action. send_action is not implemented;
  - missing images stay None in the row (relative path) and no file is
    created — zero-filled placeholder images are never written;
  - IMU missing -> gyro/accel null; with require_imu=True such frames are
    skipped (not valid for this dataset);
  - raw image arrays never appear in any JSON metadata.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .synchronized_frame import SynchronizedFrame

STATE_DIM = 25
STATE_KEYS = ["joint_position[6]", "joint_velocity[6]", "joint_current[6]", "tcp_pose[7]"]
IMAGE_NAMES = ("rgb", "depth", "aligned_depth")
IMAGE_FORMATS = ("png", "jpg")  # depth/aligned_depth always PNG (lossless)


@dataclass
class EpisodeRecord:
    episode_index: int
    task_name: str
    task_index: int
    frame_count: int = 0  # 写入的帧数
    skipped_frames: int = 0  # 被过滤的帧数
    skipped_reasons: Dict[str, int] = field(default_factory=dict)
    timestamp_non_monotonic: int = 0
    data_path: str = ""
    backend: str = "jsonl"
    started_at: str = ""
    finished_at: str = ""


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class DatasetWriter:
    """Accumulates SynchronizedFrames into episodes and writes the dataset."""

    def __init__(
        self,
        dataset_root: str,
        require_imu: bool = False,
        image_format: str = "png",
    ):
        if image_format not in IMAGE_FORMATS:
            raise ValueError(f"image_format 仅支持 {IMAGE_FORMATS}")
        self.dataset_root = Path(dataset_root)
        self.require_imu = require_imu
        self.image_format = image_format

        self._task_index_map: Dict[str, int] = {}
        self._episodes: List[EpisodeRecord] = []
        self._current: Optional[EpisodeRecord] = None
        self._rows: List[dict] = []
        self._state_samples: List[np.ndarray] = []
        self._current_state_samples: List[np.ndarray] = []  # 仅当前 episode（discard 时可整体丢弃）
        self._frame_counter = 0
        self._last_timestamp: Optional[float] = None

        (self.dataset_root / "meta_data").mkdir(parents=True, exist_ok=True)
        (self.dataset_root / "data").mkdir(parents=True, exist_ok=True)
        for name in IMAGE_NAMES:
            (self.dataset_root / "videos" / name).mkdir(parents=True, exist_ok=True)
        (self.dataset_root / "imu").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ API
    def start_episode(self, task_name: str) -> None:
        """(Re)start an episode; finalizes any open episode first."""
        self.finish_episode()
        if task_name not in self._task_index_map:
            self._task_index_map[task_name] = len(self._task_index_map)
        self._current = EpisodeRecord(
            episode_index=len(self._episodes),
            task_name=task_name,
            task_index=self._task_index_map[task_name],
            started_at=_now_iso(),
        )
        self._rows = []
        self._current_state_samples = []
        self._frame_counter = 0
        self._last_timestamp = None

    def append_frame(self, frame: SynchronizedFrame, action: Any = None) -> bool:
        """Write one frame. Returns True if written, False if filtered out.

        Filtered (never written): invalid frames, frames missing their
        observation/state, frames missing IMU when require_imu=True.
        """
        if self._current is None:
            raise RuntimeError("append_frame 前必须先调用 start_episode()")

        def skip(reason: str) -> bool:
            self._current.skipped_frames += 1
            self._current.skipped_reasons[reason] = self._current.skipped_reasons.get(reason, 0) + 1
            return False

        if not frame.valid:
            return skip("frame_invalid")
        obs = frame.observation
        if obs is None:
            return skip("observation_missing")
        if None in (obs.joint_position, obs.joint_velocity, obs.joint_current, obs.tcp_pose):
            return skip("state_incomplete")
        state = (
            list(obs.joint_position)
            + list(obs.joint_velocity)
            + list(obs.joint_current)
            + list(obs.tcp_pose)
        )
        if len(state) != STATE_DIM:
            return skip(f"state_dim_{len(state)}")
        if frame.timestamp is None:
            return skip("timestamp_missing")

        cam = frame.camera_packet
        gyro = list(cam.gyro) if cam is not None and cam.gyro is not None else None
        accel = list(cam.accel) if cam is not None and cam.accel is not None else None
        if self.require_imu and (gyro is None or accel is None):
            return skip("imu_missing")

        if self._last_timestamp is not None and frame.timestamp <= self._last_timestamp:
            self._current.timestamp_non_monotonic += 1
        self._last_timestamp = frame.timestamp

        # 图像: 单独保存，行内只留相对路径；缺失保持 None，绝不写全零图像
        ep_name = f"episode_{self._current.episode_index:06d}"
        img_paths: Dict[str, Optional[str]] = {}
        for name in IMAGE_NAMES:
            arr = getattr(cam, name) if cam is not None else None
            if arr is None:
                img_paths[name] = None
                continue
            rel = f"videos/{name}/{ep_name}/frame_{self._frame_counter:06d}.{self._save_format(name)}"
            abs_path = self.dataset_root / rel
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_image(abs_path, arr, name)
            img_paths[name] = rel

        # IMU 行
        imu_line = {
            "episode_index": self._current.episode_index,
            "frame_index": self._frame_counter,
            "timestamp": frame.timestamp,
            "gyro": gyro,
            "accel": accel,
        }
        with (self.dataset_root / "imu" / f"{ep_name}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(imu_line) + "\n")

        row = {
            "timestamp": frame.timestamp,
            "episode_index": self._current.episode_index,
            "frame_index": self._frame_counter,
            "task_index": self._current.task_index,
            "valid": True,
            "observation.state": state,
            "observation.images.rgb.path": img_paths["rgb"],
            "observation.images.depth.path": img_paths["depth"],
            "observation.images.aligned_depth.path": img_paths["aligned_depth"],
            "observation.imu.gyro": gyro,
            "observation.imu.accel": accel,
            # action=None 是显式缺失标记（JSON null），绝不用 observation.state 冒充
            "action": action,
        }
        self._rows.append(row)
        self._current_state_samples.append(np.asarray(state, dtype=np.float64))
        self._frame_counter += 1
        return True

    def finish_episode(self) -> None:
        """Write the current episode's data file and close it."""
        if self._current is None:
            return
        ep_name = f"episode_{self._current.episode_index:06d}"
        backend = self._write_episode_data(ep_name, self._rows)
        self._current.frame_count = len(self._rows)
        self._current.backend = backend
        self._current.data_path = f"data/{ep_name}.{backend}"
        self._current.finished_at = _now_iso()
        self._episodes.append(self._current)
        self._state_samples.extend(self._current_state_samples)
        self._current = None
        self._rows = []
        self._current_state_samples = []

    def discard_episode(self) -> None:
        """Abort the current episode: drop all accumulated rows/samples and
        delete the episode's on-disk artifacts (PNG dirs + IMU jsonl).

        Used by the episode control protocol (discard). The episode does NOT
        enter ``_episodes``, so it never reaches the training data.
        """
        if self._current is None:
            return
        ep_name = f"episode_{self._current.episode_index:06d}"
        for name in IMAGE_NAMES:
            ep_dir = self.dataset_root / "videos" / name / ep_name
            if ep_dir.exists():
                shutil.rmtree(ep_dir, ignore_errors=True)
        imu_file = self.dataset_root / "imu" / f"{ep_name}.jsonl"
        if imu_file.exists():
            imu_file.unlink()
        self._current = None
        self._rows = []
        self._current_state_samples = []
        self._frame_counter = 0
        self._last_timestamp = None

    def write_metadata(self) -> None:
        """Write meta_data/info.json and meta_data/stats.json."""
        self.finish_episode()
        info = {
            "format_version": "v2-style-offline",
            "created_at": _now_iso(),
            "writer": "rml63_bridge.dataset_writer",
            "require_imu": self.require_imu,
            "image_format": self.image_format,
            "state_dim": STATE_DIM,
            "state_keys": STATE_KEYS,
            "action_policy": "action 允许为 None（显式缺失）；不使用 observation.state 冒充 action",
            "task_index_map": dict(self._task_index_map),
            "dirs": {
                "data": "data/",
                "images": "videos/{rgb,depth,aligned_depth}/",
                "imu": "imu/",
            },
            "episodes": [
                {
                    "episode_index": e.episode_index,
                    "task_name": e.task_name,
                    "task_index": e.task_index,
                    "frame_count": e.frame_count,
                    "skipped_frames": e.skipped_frames,
                    "skipped_reasons": dict(e.skipped_reasons),
                    "timestamp_non_monotonic": e.timestamp_non_monotonic,
                    "data_path": e.data_path,
                    "backend": e.backend,
                    "started_at": e.started_at,
                    "finished_at": e.finished_at,
                }
                for e in self._episodes
            ],
        }
        (self.dataset_root / "meta_data" / "info.json").write_text(
            json.dumps(info, indent=2), encoding="utf-8"
        )
        stats = self._compute_stats()
        (self.dataset_root / "meta_data" / "stats.json").write_text(
            json.dumps(stats, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------- helpers
    def _save_format(self, name: str) -> str:
        if name == "rgb":
            return self.image_format
        return "png"  # depth/aligned_depth 始终无损 PNG

    @staticmethod
    def _save_image(path: Path, arr: np.ndarray, name: str) -> None:
        """Save one image with PIL (lazy import). Depth stays uint16 PNG."""
        from PIL import Image

        if name == "rgb":
            img = Image.fromarray(np.asarray(arr, dtype=np.uint8), mode="RGB")
        else:
            img = Image.fromarray(np.asarray(arr, dtype=np.uint16), mode="I;16")
        img.save(path)

    def _write_episode_data(self, ep_name: str, rows: List[dict]) -> str:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            path = self.dataset_root / "data" / f"{ep_name}.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            return "jsonl"
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, self.dataset_root / "data" / f"{ep_name}.parquet")
        return "parquet"

    def _compute_stats(self) -> dict:
        samples = self._state_samples
        if not samples:
            return {
                "n_valid_frames": 0,
                "note": "统计仅包含已写入的 valid 帧；invalid 帧不计入",
                "observation.state": {"mean": [], "std": [], "min": [], "max": []},
            }
        arr = np.stack(samples)
        return {
            "n_valid_frames": int(len(samples)),
            "note": "统计仅包含已写入的 valid 帧；invalid 帧不计入",
            "observation.state": {
                "mean": arr.mean(axis=0).tolist(),
                "std": arr.std(axis=0).tolist(),
                "min": arr.min(axis=0).tolist(),
                "max": arr.max(axis=0).tolist(),
            },
        }
