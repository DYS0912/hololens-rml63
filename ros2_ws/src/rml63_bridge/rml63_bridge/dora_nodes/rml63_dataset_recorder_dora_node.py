"""Dora 节点 3：数据集录制节点（rml63_dataset_recorder_dora_node）。

接收通道（dataflow 中由 rml63_state / d435i_camera / mock_action 供给）：
    state, tcp_pose, joint_current, joint_velocity, status,
    state_timestamp, state_valid          <- rml63_state/*
    rgb, depth, aligned_depth, gyro, accel,
    camera_timestamp, frame_id, camera_valid  <- d435i_camera/*
    action                                 <- mock_action/action（离线模拟；未来真实 action 源）
    tool_state                             <- （预留）绑扎枪状态，未来通道
    tick                                   <- dora/timer/millis/33（帧处理心跳）

功能：
  - 多传感器时间同步：复用 synchronized_frame.synchronize_observations
    （max_time_delta=0.05，tie 偏好过去帧，拒绝未来过远帧）；
  - episode 生命周期：控制文件协议 save/next/discard/stop（mtime 轮询）；
  - 相机首帧等待超时 + 相机陈旧检测（复用 dorobot2 的工程实践）；
  - invalid 帧过滤（valid=False 与同步失败均不写入）；
  - 图像独立保存为 PNG（DatasetWriter），行内只存路径，像素绝不进 JSON；
  - stop 时生成 LeRobot 0.6.2 布局（meta/ data/ images/ tasks.parquet /
    episodes parquet，复用 convert_to_lerobot，真实 observation.state 统计）；
  - action 缺失：有 action 的帧才进入训练 Parquet；一帧 action 都没有时
    拒绝生成训练数据（绝不伪造 action）；
  - 录制被中断（dora destroy 等）后可用 recover_offline_dataset() 从
    已写盘的 episode 文件重建 meta 并继续转换。

输出通道: status（"stopped" / "error"）。

!! 约束 !! 不启动 ROS2、不连接硬件、不发送控制指令；离线可用
simulate_dataflow 或本模块核心类测试。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# 注意: dora 以普通脚本方式运行本文件（python3 <path>），必须使用绝对导入；
# dataflow env 中的 PYTHONPATH 指向 rml63_bridge 包根目录。
from rml63_bridge.camera_packet import CameraPacket, validate_camera_packet
from rml63_bridge.convert_to_lerobot import (
    ACT_TRAINING_ACTION_MESSAGE,
    MOCK_ACTION_NOTICE,
    apply_conversion,
    plan_conversion,
)
from rml63_bridge.dataset_writer import DatasetWriter
from rml63_bridge.observation import Observation
from rml63_bridge.synchronized_frame import (
    DEFAULT_MAX_TIME_DELTA,
    synchronize_observations,
)
from rml63_bridge.dora_nodes.dora_compat import dora_available, describe_node, value_to_py

DEFAULT_CAMERA_NAMES = ("d435i",)
RECORDER_OUTPUTS = ("status",)

# recorder 输入通道 id（dataflow YAML 中与上游输出一一对应）。
# 注意: dora 按通道独立投递，跨通道顺序不保证；本节点采用
# latest-value 缓冲 + valid 最后到达触发组装（见 main() 事件循环）。
STATE_PART_IDS = {
    "state", "tcp_pose", "joint_current", "joint_velocity",
    "status", "state_timestamp", "state_valid",
}
CAMERA_PART_IDS = {
    "rgb", "depth", "aligned_depth", "gyro", "accel",
    "camera_timestamp", "frame_id", "camera_valid",
}


class ControlFilePoller:
    """控制文件轮询：mtime 变化 -> 返回新命令（save/next/discard/stop）。

    复用 dorobot2 run_record.py 的 episode 控制协议设计（控制文件由外部
    CLI/脚本写入），但命令语义与数据集格式按本项目实现。
    """

    def __init__(self, path: Optional[str]):
        self.path = Path(path) if path else None
        self._last_mtime = 0.0

    def poll(self) -> Optional[str]:
        if self.path is None or not self.path.exists():
            return None
        mtime = self.path.stat().st_mtime
        if mtime <= self._last_mtime:
            return None
        self._last_mtime = mtime
        cmd = self.path.read_text(encoding="utf-8").strip()
        return cmd or None


class RecorderNodeCore:
    """录制节点核心逻辑（无 dora 依赖，可离线逐事件驱动）。

    tick 心跳驱动帧处理；输入通道方法（on_state/on_camera/on_action/
    on_tool_state）只缓冲最新数据；apply_command 执行 episode 控制。
    """

    def __init__(
        self,
        dataset_root: str,
        output_root: str,
        task: str = "rebar_tying",
        fps: float = 30.0,
        robot_type: str = "rml63",
        action_dim: int = 6,
        max_time_delta: float = DEFAULT_MAX_TIME_DELTA,
        camera_stale_timeout: float = 3.0,
        camera_start_timeout: float = 30.0,
        require_action: bool = True,
        control_file: Optional[str] = None,
        camera_names=DEFAULT_CAMERA_NAMES,
        camera_shapes: Optional[Dict[str, tuple]] = None,
    ):
        self.dataset_root = str(dataset_root)
        self.output_root = str(output_root)
        self.task = task
        self.fps = fps
        self.robot_type = robot_type
        self.action_dim = action_dim
        self.max_time_delta = max_time_delta
        self.camera_stale_timeout = camera_stale_timeout
        self.camera_start_timeout = camera_start_timeout
        self.require_action = require_action
        self.expected_cameras = set(camera_names)
        # dora 传输的图像是 ravel 后的扁平数组（发送端 to_output_array 展开），
        # 收到后必须按相机名还原为 (h, w, 3) / (h, w)。name -> (height, width)
        self._camera_shapes: Dict[str, tuple] = dict(camera_shapes or {})

        self.writer = DatasetWriter(self.dataset_root)
        self._control = ControlFilePoller(control_file)

        self._obs_buffer: deque = deque(maxlen=64)
        self._cam_buffer: deque = deque(maxlen=256)
        self._action_buffer: Optional[List[float]] = None
        self._tool_state_buffer: Any = None
        self._latest_cam_arrival: Dict[str, float] = {}
        self._cameras_seen: set = set()
        self._camera_wait_start: Optional[float] = None
        self._stale_warned: set = set()

        self._running = True
        self._paused = False
        self._episode_open = False
        self._episode_frames = 0
        self._current_episode_index = 0
        self._finalized_result: Optional[dict] = None

        self.episodes_committed = 0
        self.frames_written = 0
        self.action_frames = 0
        self.ticks_processed = 0
        self.skipped: Counter = Counter()
        self.fatal_error: Optional[str] = None

    # ------------------------------------------------------------ 输入通道
    def on_state(
        self,
        state,
        tcp_pose=None,
        joint_current=None,
        joint_velocity=None,
        status=None,
        timestamp=None,
        valid=False,
        tool_state=None,
        now=None,
    ) -> None:
        """rml63_state 节点的 7 通道 -> Observation 对象入缓冲。"""
        obs = Observation(timestamp=timestamp, tool_state=tool_state, valid=bool(valid))
        if state is not None:
            arr = np.asarray(state, dtype=np.float64)
            if arr.shape == (25,):
                obs.joint_position = arr[0:6].tolist()
                obs.joint_velocity = arr[6:12].tolist()
                obs.joint_current = arr[12:18].tolist()
                obs.tcp_pose = arr[18:25].tolist()
            else:
                obs.invalid_reasons.append(f"state 形状 {arr.shape} != (25,)")
                obs.valid = False
        else:
            obs.invalid_reasons.append("state 缺失")
            obs.valid = False
        if status is not None:
            obs.arm_status = int(status)
        if not obs.valid and not obs.invalid_reasons:
            obs.invalid_reasons.append("state 节点报告 invalid")
        self._obs_buffer.append(obs)

    def on_camera(
        self,
        rgb=None,
        depth=None,
        aligned_depth=None,
        gyro=None,
        accel=None,
        camera_timestamp=None,
        frame_id=None,
        valid=False,
        camera_name: str = "d435i",
        now=None,
    ) -> None:
        """d435i_camera 节点的 8 通道 -> CameraPacket 对象入缓冲。

        rgb/depth/aligned_depth 在 dora 传输中是扁平数组，按 camera_shapes
        还原；形状不匹配时保持原样，由 validate_camera_packet 报告错误。
        """
        rgb = self._reshape_image(rgb, camera_name, channels=3)
        depth = self._reshape_image(depth, camera_name)
        aligned_depth = self._reshape_image(aligned_depth, camera_name)
        packet = CameraPacket(
            timestamp=camera_timestamp,
            camera_name=camera_name,
            rgb=rgb,
            depth=depth,
            aligned_depth=aligned_depth,
            gyro=None if gyro is None else [float(x) for x in gyro],
            accel=None if accel is None else [float(x) for x in accel],
            frame_id=None if frame_id is None else int(frame_id),
            valid=bool(valid),
        )
        packet = validate_camera_packet(packet, require_imu=False)
        self._cam_buffer.append(packet)
        self._latest_cam_arrival[camera_name] = now if now is not None else time.time()
        self._cameras_seen.add(camera_name)

    def _reshape_image(self, arr, camera_name: str, channels: Optional[int] = None):
        if arr is None or camera_name not in self._camera_shapes:
            return arr
        h, w = self._camera_shapes[camera_name]
        target = (h, w, channels) if channels else (h, w)
        try:
            return np.asarray(arr).reshape(target)
        except ValueError:
            return arr  # 形状不匹配：交给 validate_camera_packet 报告

    def on_action(self, action, now=None) -> None:
        """最新 action（6 维）。action=None 表示该帧没有 action。"""
        if action is None:
            self._action_buffer = None
            return
        self._action_buffer = [float(x) for x in action]

    def on_tool_state(self, tool_state) -> None:
        """（预留）绑扎枪状态，随下一帧写入工具状态侧记文件。"""
        self._tool_state_buffer = tool_state

    # ------------------------------------------------------------ 控制协议
    def poll_control(self) -> Optional[str]:
        return self._control.poll()

    def apply_command(self, cmd: str) -> None:
        """save / next / discard / stop。"""
        cmd = (cmd or "").strip().lower()
        print(f"[recorder] CONTROL: {cmd!r}", flush=True)
        if cmd == "save":
            self._commit_current_episode()
            self._paused = True
            self._reset_frame_state()
        elif cmd == "next":
            self._paused = False
            self._reset_frame_state()
        elif cmd == "discard":
            self._discard_current_episode()
            self._paused = True
            self._reset_frame_state()
        elif cmd == "stop":
            self.finalize()
            self._running = False

    def _commit_current_episode(self) -> None:
        if not self._episode_open:
            return
        if self._episode_frames > 0:
            self.writer.finish_episode()
            self.episodes_committed += 1
        else:
            self.writer.discard_episode()
        self._episode_open = False
        self._episode_frames = 0

    def _discard_current_episode(self) -> None:
        if not self._episode_open:
            return
        self.writer.discard_episode()
        tool_file = self._tool_sidecar_path()
        if tool_file.exists():
            tool_file.unlink()
        self._episode_open = False
        self._episode_frames = 0

    def _reset_frame_state(self) -> None:
        """save/next/discard 后重置相机等待与告警状态（同 dorobot2 语义）。"""
        self._cameras_seen.clear()
        self._camera_wait_start = None
        self._stale_warned.clear()

    # ------------------------------------------------------------ 心跳
    def on_tick(self, now=None) -> dict:
        """帧处理心跳：控制轮询 -> 相机健全性 -> 时间同步 -> 写入。"""
        if not self._running:
            return {}
        now = now if now is not None else time.time()
        self.ticks_processed += 1

        cmd = self.poll_control()
        if cmd:
            self.apply_command(cmd)
        if not self._running:
            return {"status": "stopped"}
        if self._paused:
            return {}

        # 相机首帧等待（超时 -> 致命错误，同 dorobot2 语义）
        if not self._cameras_seen.issuperset(self.expected_cameras):
            if self._camera_wait_start is None:
                self._camera_wait_start = now
            elif now - self._camera_wait_start > self.camera_start_timeout:
                missing = self.expected_cameras - self._cameras_seen
                self.fatal_error = (
                    f"相机 {sorted(missing)} 在 {self.camera_start_timeout}s 内未发送首帧"
                )
                self._running = False
                return {"status": "error", "error": self.fatal_error}
            return {}
        self._camera_wait_start = None

        # 相机陈旧检测（任一相机过期 -> 跳过本帧并告警一次）
        for cam in self.expected_cameras:
            arrival = self._latest_cam_arrival.get(cam)
            if arrival is not None and now - arrival >= self.camera_stale_timeout:
                self.skipped["camera_stale"] += 1
                if cam not in self._stale_warned:
                    print(
                        f"[recorder] WARNING: {cam} 图像陈旧 "
                        f"({now - arrival:.1f}s) - 相机可能已断流",
                        flush=True,
                    )
                    self._stale_warned.add(cam)
                return {}

        # 多传感器时间同步（复用已验证的同步逻辑）
        obs = self._obs_buffer[-1] if self._obs_buffer else None
        frames = synchronize_observations(
            [obs],
            list(self._cam_buffer),
            max_time_delta=self.max_time_delta,
            require_camera=True,
            require_imu=False,
        )
        frame = frames[0]
        if not frame.valid:
            self.skipped["frame_invalid"] += 1
            if self.skipped["frame_invalid"] <= 3:  # 前 3 次打印原因，便于运行时排障
                print(
                    f"[recorder] DEBUG frame invalid: {frame.invalid_reasons}",
                    flush=True,
                )
            return {}

        # episode 按需开启（首帧触发）
        if not self._episode_open:
            self.writer.start_episode(self.task)
            self._episode_open = True
            self._current_episode_index = self.episodes_committed
            self._episode_frames = 0

        action = (
            [float(x) for x in self._action_buffer]
            if self._action_buffer is not None
            else None
        )
        if self.writer.append_frame(frame, action=action):
            self._episode_frames += 1
            self.frames_written += 1
            if action is not None:
                self.action_frames += 1
            self._write_tool_sidecar(frame)
        else:
            self.skipped["writer_filtered"] += 1
        return {}

    # ------------------------------------------------------------ 收尾
    @property
    def running(self) -> bool:
        return self._running

    def finalize(self) -> dict:
        """stop：收尾 staging 数据集并转换为 LeRobot 0.6.2 训练格式。

        幂等：重复调用返回首次结果（dora 停止事件与 stop 命令可能同时到达）。
        """
        if self._finalized_result is not None:
            return self._finalized_result
        if self._episode_open:
            if self._episode_frames > 0:
                self.writer.finish_episode()
                self.episodes_committed += 1
            else:
                self.writer.discard_episode()
            self._episode_open = False
            self._episode_frames = 0
        self.writer.write_metadata()

        errors: List[str] = []
        if self.require_action and self.action_frames == 0:
            errors.append(
                ACT_TRAINING_ACTION_MESSAGE
                + "；拒绝生成训练数据（不伪造 action）。"
                + " 离线 staging 数据保留在 "
                + self.dataset_root
            )
        else:
            try:
                plan = plan_conversion(
                    self.dataset_root,
                    self.output_root,
                    action_dim=self.action_dim,
                    fps=self.fps,
                    robot_type=self.robot_type,
                    dry_run=False,
                )
                errors = apply_conversion(plan)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"转换异常: {exc}")

        result = {
            "status": "stopped",
            "dataset_root": self.dataset_root,
            "output_root": self.output_root,
            "episodes_committed": self.episodes_committed,
            "frames_written": self.frames_written,
            "action_frames": self.action_frames,
            "ticks_processed": self.ticks_processed,
            "skipped": dict(self.skipped),
            "errors": errors,
        }
        if self.fatal_error:
            result["status"] = "error"
            result["errors"] = [self.fatal_error] + errors
        self._finalized_result = result
        return result

    # ------------------------------------------------------------ 内部
    def _tool_sidecar_path(self) -> Path:
        return (
            Path(self.dataset_root)
            / "tool_state"
            / f"episode_{self._current_episode_index:06d}.jsonl"
        )

    def _write_tool_sidecar(self, frame) -> None:
        if frame.observation is None or frame.observation.tool_state is None:
            return
        path = self._tool_sidecar_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            record = json.dumps(
                {
                    "episode_index": self._current_episode_index,
                    "timestamp": frame.timestamp,
                    "tool_state": frame.observation.tool_state,
                },
                default=str,
            )
        except (TypeError, ValueError):
            record = json.dumps(
                {
                    "episode_index": self._current_episode_index,
                    "timestamp": frame.timestamp,
                    "tool_state": repr(frame.observation.tool_state),
                }
            )
        with path.open("a", encoding="utf-8") as f:
            f.write(record + "\n")


# ---------------------------------------------------------------------------
# 元数据恢复（录制被 dora destroy / 进程被杀中断后）
# ---------------------------------------------------------------------------

def _collect_staging_rows(dataset_root: Path) -> List[dict]:
    """读回 staging 的 episode 数据文件（jsonl 或 parquet 双后端）。"""
    rows: List[dict] = []
    for f in sorted((dataset_root / "data").glob("episode_*")):
        if f.suffix == ".jsonl":
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        elif f.suffix == ".parquet":
            import pyarrow.parquet as pq

            rows.extend(pq.read_table(f).to_pylist())
    return rows


def recover_offline_dataset(dataset_root: str) -> dict:
    """从已写盘的 episode 文件重建 meta_data（info.json + stats.json）。

    对应 dorobot2 main.py 的 _generate_meta_fallback：recorder 在
    finalize 之前被杀时，已完成的 episode 数据仍在 staging 目录，
    本函数补写元数据后可继续 convert_to_lerobot 转换。
    绝不伪造 action：恢复的 info.json 记录 recovered=True。
    """
    root = Path(dataset_root)
    meta_dir = root / "meta_data"
    meta_dir.mkdir(parents=True, exist_ok=True)
    info_file = meta_dir / "info.json"

    rows = _collect_staging_rows(root)
    if not rows:
        return {
            "recovered": False,
            "error": f"无数据可恢复: {dataset_root}（data/episode_* 不存在）",
        }

    # 按 episode 分组（保持文件顺序）
    by_ep: Dict[int, List[dict]] = {}
    for r in rows:
        by_ep.setdefault(int(r["episode_index"]), []).append(r)

    episodes = []
    for ep_idx, ep_rows in sorted(by_ep.items()):
        episodes.append(
            {
                "episode_index": ep_idx,
                "frame_count": len(ep_rows),
                "data_path": f"data/episode_{ep_idx:06d}.*",
                "recovered": True,
            }
        )

    info = {
        "format_version": "v2-style-offline-recovered",
        "recovered": True,
        "note": "录制进程在 finalize 前被中断，本文件由 recover_offline_dataset 重建；"
        "action 缺失的帧在转换时会被拒绝进入训练数据",
        "state_dim": 25,
        "episodes": episodes,
    }
    info_file.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    # observation.state 统计（仅真实数据；缺失帧的 None 值跳过）
    samples = [
        np.asarray(r["observation.state"], dtype=np.float64)
        for r in rows
        if r.get("observation.state") is not None
    ]
    stats = {"n_valid_frames": len(samples), "note": "恢复统计（仅 observation.state）"}
    if samples:
        arr = np.stack(samples)
        stats["observation.state"] = {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
        }
    (meta_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return {
        "recovered": True,
        "episodes": len(episodes),
        "frames": len(rows),
        "dataset_root": str(root),
    }


# ---------------------------------------------------------------------------
# Dora 节点入口
# ---------------------------------------------------------------------------

def _env_inputs() -> dict:
    return {
        "tick": "dora/timer/millis/33",
        "state": "rml63_state/state",
        "tcp_pose": "rml63_state/tcp_pose",
        "joint_current": "rml63_state/joint_current",
        "joint_velocity": "rml63_state/joint_velocity",
        "status": "rml63_state/status",
        "state_timestamp": "rml63_state/timestamp",
        "state_valid": "rml63_state/valid",
        "rgb": "d435i_camera/rgb",
        "depth": "d435i_camera/depth",
        "aligned_depth": "d435i_camera/aligned_depth",
        "gyro": "d435i_camera/gyro",
        "accel": "d435i_camera/accel",
        "camera_timestamp": "d435i_camera/camera_timestamp",
        "frame_id": "d435i_camera/frame_id",
        "camera_valid": "d435i_camera/valid",
        "action": "mock_action/action（未来: 真实 action 源）",
        "tool_state": "（预留）绑扎枪状态",
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="rml63_dataset_recorder_dora_node")
    parser.add_argument("--describe", action="store_true", help="打印节点设计并退出（不需要 dora）")
    parser.add_argument("--check", action="store_true", help="离线自检并退出（不需要 dora）")
    args = parser.parse_args()

    if args.describe:
        print(describe_node("rml63_dataset_recorder", _env_inputs(), list(RECORDER_OUTPUTS)))
        print(f"注意: {MOCK_ACTION_NOTICE}")
        return 0
    if args.check:
        print("rml63_dataset_recorder check: 核心类可导入（RecorderNodeCore / "
              "ControlFilePoller / recover_offline_dataset）")
        print(f"注意: {MOCK_ACTION_NOTICE}")
        return 0

    if not dora_available():
        print(
            "错误: dora-rs 未安装，无法作为 Dora 节点运行。\n"
            "离线验证请使用: python3 -m rml63_bridge.dora_nodes.simulate_dataflow\n"
            "或本脚本的 --describe / --check 模式。",
            file=sys.stderr,
        )
        return 1

    from dora import Node  # noqa: PLC0415（惰性导入）

    camera_shapes = None
    shapes_raw = os.environ.get("RML63_DORA_CAMERA_SHAPES")
    if shapes_raw:
        camera_shapes = {
            c["name"]: (int(c["height"]), int(c["width"]))
            for c in json.loads(shapes_raw)
        }
    node = Node()
    core = RecorderNodeCore(
        dataset_root=os.environ.get("RML63_DORA_DATASET_ROOT", "/tmp/rml63_dora_stage"),
        output_root=os.environ.get("RML63_DORA_OUTPUT_ROOT", "/tmp/rml63_dora_lerobot"),
        task=os.environ.get("RML63_DORA_TASK", "rebar_tying"),
        fps=float(os.environ.get("RML63_DORA_FPS", "30")),
        robot_type=os.environ.get("RML63_DORA_ROBOT_TYPE", "rml63"),
        action_dim=int(os.environ.get("RML63_DORA_ACTION_DIM", "6")),
        max_time_delta=float(os.environ.get("RML63_DORA_MAX_TIME_DELTA", "0.05")),
        camera_stale_timeout=float(
            os.environ.get("RML63_DORA_CAMERA_STALE_TIMEOUT", "3.0")
        ),
        camera_start_timeout=float(
            os.environ.get("RML63_DORA_CAMERA_START_TIMEOUT", "30.0")
        ),
        require_action=os.environ.get("RML63_DORA_REQUIRE_ACTION", "true").lower()
        == "true",
        control_file=os.environ.get("RML63_DORA_CONTROL_FILE"),
        camera_names=tuple(
            c for c in os.environ.get("RML63_DORA_CAMERA_NAMES", "d435i").split(",") if c
        ),
        camera_shapes=camera_shapes,
    )
    print(
        f"[recorder] task={core.task} root={core.dataset_root} "
        f"control_file={core._control.path}",
        flush=True,
    )

    state_parts: Dict[str, Any] = {}
    camera_parts: Dict[str, Any] = {}
    event_counts: Dict[str, int] = {}
    trace: list = []
    debug_trace = os.environ.get("RML63_DORA_DEBUG_TRACE") == "true"
    _t0 = time.time()

    for event in node:
        etype = event["type"]
        if etype == "STOP":
            break
        if etype != "INPUT":
            continue
        eid = event["id"]
        event_counts[eid] = event_counts.get(eid, 0) + 1
        if debug_trace and len(trace) < 120:
            trace.append((round(time.time() - _t0, 3), eid))
        value = value_to_py(event.get("value"))

        if eid == "tick":
            out = core.on_tick()
            if out.get("status"):
                from rml63_bridge.dora_nodes.dora_compat import to_output_array

                node.send_output("status", to_output_array(str(out.get("status"))))
        elif eid == "action":
            core.on_action(value)
        elif eid == "tool_state":
            core.on_tool_state(value)
        elif eid in STATE_PART_IDS:
            # 逐通道 latest-value 缓冲（跨通道到达顺序可能被打乱，实测
            # 个别通道滞后约 1 tick；不 clear，缺失时用上一次的值，
            # 由 state_valid/camera_valid 的帧标志负责无效化）
            state_parts[eid] = value
            if eid == "state_valid":  # state 节点按序发送，valid 最后到达
                core.on_state(
                    state=state_parts.get("state"),
                    tcp_pose=state_parts.get("tcp_pose"),
                    joint_current=state_parts.get("joint_current"),
                    joint_velocity=state_parts.get("joint_velocity"),
                    status=state_parts.get("status"),
                    timestamp=state_parts.get("state_timestamp"),
                    valid=bool(state_parts.get("state_valid", False)),
                    tool_state=core._tool_state_buffer,
                )
        elif eid in CAMERA_PART_IDS:
            camera_parts[eid] = value
            if eid == "camera_valid":  # camera 节点按序发送，valid 最后到达
                core.on_camera(
                    rgb=camera_parts.get("rgb"),
                    depth=camera_parts.get("depth"),
                    aligned_depth=camera_parts.get("aligned_depth"),
                    gyro=camera_parts.get("gyro"),
                    accel=camera_parts.get("accel"),
                    camera_timestamp=camera_parts.get("camera_timestamp"),
                    frame_id=camera_parts.get("frame_id"),
                    valid=bool(camera_parts.get("camera_valid", False)),
                )

    result = core.finalize()
    result["events_seen"] = dict(sorted(event_counts.items()))
    result["buffers"] = {
        "observations": len(core._obs_buffer),
        "camera_packets": len(core._cam_buffer),
    }
    if debug_trace:
        print("[recorder] TRACE:", flush=True)
        for t, eid in trace:
            print(f"  {t:7.3f}  {eid}", flush=True)
    if result.get("status"):
        try:
            from rml63_bridge.dora_nodes.dora_compat import to_output_array

            node.send_output("status", to_output_array(str(result["status"])))
        except Exception:  # noqa: BLE001
            pass
    print(f"[recorder] stopped: {json.dumps(result, ensure_ascii=False)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
