"""JSONL+PNG -> LeRobot 0.6.2 布局转换器（dry-run 优先，不覆盖原始数据）。

输入后端: rml63_bridge dataset_writer 的离线数据集
（data/episode_*.jsonl 或 data/episode_*.parquet，取决于写盘环境有无 pyarrow）。
输出目标: LeRobot 0.6.2 "lerobot" 存储格式（静态审计结论）:

    <out>/
      meta/info.json, meta/stats.json, meta/tasks.parquet,
      meta/episodes/chunk-000/file-000.parquet
      data/chunk-000/file-000.parquet
      images/<key>/episode-XXXXXX/frame-YYYYYY.png

规则:
  - 只转换 action 不为空的帧；action 全部缺失时拒绝转换并说明原因；
  - 绝不使用 observation.state 冒充 action；
  - 保留 episode_index / frame_index / timestamp / task；
  - 保留 RGB、depth、aligned_depth 的相对路径（复制图像到新布局）；
  - 保留 25 维 observation.state；gyro/accel 作为普通 float32 特征列；
  - dry-run 模式只做规划与统计，不创建输出目录、不复制图像、不覆盖原始数据。

依赖说明: 写出 Parquet 需要 pyarrow（LeRobot 0.6.2 的 data/tasks/episodes
均为 Parquet）；当前系统 Python 无 pyarrow 时，dry-run 全部可用，实际写出
阶段会明确报错并给出安装提示（不自动安装）。

MOCK_ACTION_NOTICE:
    模拟 action 仅用于数据格式验证，不代表真实示范动作，不用于研究结论。
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .dataset_compat_check import (
    ACT_TRAINING_ACTION_MESSAGE,
    DEFAULT_STATE_DIM,
    LEROBOT_META_DIR,
)

# 模拟 action 仅用于数据格式验证，不代表真实示范动作，不用于研究结论。
MOCK_ACTION_NOTICE = "模拟 action 仅用于数据格式验证，不代表真实示范动作，不用于研究结论。"

DEFAULT_CODEBASE_VERSION = "v3.0"
CHUNK_FILE = "chunk-000/file-000"
IMAGE_KEYS = ("rgb", "depth", "aligned_depth")
IMU_KEYS = ("gyro", "accel")


@dataclass
class ConversionPlan:
    input_root: str
    output_root: str
    dry_run: bool
    action_dim: int
    fps: int
    robot_type: str
    episodes: List[dict] = field(default_factory=list)
    frames: List[dict] = field(default_factory=list)
    skipped_action_null: int = 0
    skipped_other: int = 0
    image_copies: List[tuple] = field(default_factory=list)  # (src, dst_rel)
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"输入: {self.input_root} (JSONL+PNG)",
            f"输出: {self.output_root} (LeRobot 0.6.2 布局) dry_run={self.dry_run}",
            f"转换 episodes={len(self.episodes)} frames={len(self.frames)} "
            f"action_dim={self.action_dim}",
            f"跳过: action 为空 {self.skipped_action_null}, 其他 {self.skipped_other}",
            f"图像复制: {len(self.image_copies)} 个文件",
        ]
        if self.frames:
            lines.append(f"声明: {MOCK_ACTION_NOTICE}")
        if self.errors:
            lines.append("错误:")
            lines += [f"  - {e}" for e in self.errors]
        return "\n".join(lines)


def _load_offline(root: Path) -> Dict[str, Any]:
    """读取 rml63_bridge 离线数据集。

    dataset_writer 的后端取决于运行环境的 pyarrow 是否可用:
    有 pyarrow -> data/episode_*.parquet；无 pyarrow -> data/episode_*.jsonl。
    两种后端都支持，优先读 Parquet。
    """
    info = json.loads((root / "meta_data" / "info.json").read_text(encoding="utf-8"))
    rows: List[dict] = []
    data_dir = root / "data"
    parquet_files = sorted(data_dir.glob("episode_*.parquet"))
    jsonl_files = sorted(data_dir.glob("episode_*.jsonl"))
    if parquet_files:
        import pyarrow.parquet as pq

        for f in parquet_files:
            rows.extend(pq.read_table(f).to_pylist())
    else:
        for f in jsonl_files:
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
    return {"info": info, "rows": rows}


def plan_conversion(
    input_root: str,
    output_root: str,
    action_dim: int = 6,
    fps: int = 30,
    robot_type: str = "rml63",
    dry_run: bool = True,
) -> ConversionPlan:
    """规划转换（纯 Python，不写文件）。

    - action=None 的帧被跳过并计数；
    - 若没有任何 action 非空的帧 -> errors 包含拒绝原因，frames 为空；
    - 绝不把 observation.state 当作 action。
    """
    in_root = Path(input_root)
    plan = ConversionPlan(
        input_root=input_root,
        output_root=output_root,
        dry_run=dry_run,
        action_dim=action_dim,
        fps=fps,
        robot_type=robot_type,
    )
    if not in_root.is_dir():
        plan.errors.append(f"输入目录不存在: {in_root}")
        return plan

    ds = _load_offline(in_root)
    rows = ds["rows"]
    if not rows:
        plan.errors.append("输入数据集没有帧数据")
        return plan

    # 按 episode 分组（保持原始顺序）
    by_ep: Dict[int, List[dict]] = {}
    for r in rows:
        by_ep.setdefault(r["episode_index"], []).append(r)

    new_episode_index = 0
    global_frame = 0
    for old_ep, ep_rows in sorted(by_ep.items()):
        converted = []
        episode_failed = False
        for r in ep_rows:
            action = r.get("action")
            state = r.get("observation.state")
            if action is None:
                plan.skipped_action_null += 1
                continue
            if state is None or len(state) != DEFAULT_STATE_DIM:
                plan.skipped_other += 1
                continue
            if len(action) != action_dim:
                plan.errors.append(
                    f"episode {old_ep} frame {r.get('frame_index')}: action 维度 "
                    f"{len(action)} != {action_dim}，已跳过该帧"
                )
                plan.skipped_other += 1
                continue

            img_paths = {}
            for key in IMAGE_KEYS:
                src_rel = r.get(f"observation.images.{key}.path")
                if src_rel is None:
                    img_paths[key] = None
                    continue
                src = in_root / src_rel
                if not src.is_file():
                    plan.errors.append(f"图像源文件缺失: {src_rel}")
                    plan.skipped_other += 1
                    episode_failed = True
                    break
                dst_rel = (
                    f"images/{key}/"
                    f"episode-{new_episode_index:06d}/frame-{global_frame:06d}.png"
                )
                img_paths[key] = dst_rel
                plan.image_copies.append((str(src), dst_rel))
            if episode_failed:
                plan.skipped_other += len(ep_rows) - (ep_rows.index(r) + 1)
                break

            converted.append(
                {
                    "timestamp": r["timestamp"],
                    "episode_index": new_episode_index,
                    "frame_index": global_frame,
                    "task_index": r.get("task_index", 0),
                    "index": global_frame,
                    "observation.state": state,
                    "action": action,
                    # 注意: LeRobot 0.6.2 的图像列存【绝对路径】(其 writer 存
                    # str(root / fpath)，读取时 PIL 直接 open 该字符串，不做
                    # 相对解析) —— 因此这里存输出目录下的绝对路径
                    "observation.images.rgb": (
                        str(Path(plan.output_root) / img_paths["rgb"])
                        if img_paths["rgb"] is not None
                        else None
                    ),
                    "observation.images.depth": (
                        str(Path(plan.output_root) / img_paths["depth"])
                        if img_paths["depth"] is not None
                        else None
                    ),
                    "observation.images.aligned_depth": (
                        str(Path(plan.output_root) / img_paths["aligned_depth"])
                        if img_paths["aligned_depth"] is not None
                        else None
                    ),
                    "observation.imu.gyro": r.get("observation.imu.gyro"),
                    "observation.imu.accel": r.get("observation.imu.accel"),
                }
            )
            global_frame += 1
        if converted:
            plan.episodes.append(
                {
                    "episode_index": new_episode_index,
                    "task_index": converted[0]["task_index"],
                    "length": len(converted),
                    "from_timestamp": converted[0]["timestamp"],
                    "to_timestamp": converted[-1]["timestamp"],
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                }
            )
            plan.frames.extend(converted)
            new_episode_index += 1

    if not plan.frames:
        plan.errors.append(ACT_TRAINING_ACTION_MESSAGE + "；拒绝转换（不伪造 action）")

    # dry-run 必须可用: 无 pyarrow 时只在实际写出阶段报告
    try:
        import pyarrow  # noqa: F401

        has_pyarrow = True
    except ImportError:
        has_pyarrow = False
    if not has_pyarrow and not dry_run:
        plan.errors.append(
            "写出 LeRobot Parquet (data/tasks/episodes) 需要 pyarrow；"
            "请先安装（如 pip install --user pyarrow）。dry-run 无需 pyarrow。"
        )
    return plan


def _features(action_dim: int, height: int, width: int) -> dict:
    return {
        "observation.state": {"dtype": "float32", "shape": [DEFAULT_STATE_DIM], "names": None},
        "action": {"dtype": "float32", "shape": [action_dim], "names": None},
        "observation.images.rgb": {
            "dtype": "image",
            "shape": [3, height, width],
            "names": ["r", "g", "b"],
        },
        "observation.images.depth": {
            "dtype": "image",
            "shape": [1, height, width],
            "names": ["depth"],
            "info": {"depth_unit": "mm"},
        },
        "observation.images.aligned_depth": {
            "dtype": "image",
            "shape": [1, height, width],
            "names": ["depth"],
            "info": {"depth_unit": "mm"},
        },
        "observation.imu.gyro": {"dtype": "float32", "shape": [3], "names": None},
        "observation.imu.accel": {"dtype": "float32", "shape": [3], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
    }


def _feature_stats(plan: ConversionPlan) -> dict:
    def stats_for(key, dim):
        vals = np.array([f[key] for f in plan.frames if f.get(key) is not None], dtype=np.float64)
        if vals.size == 0:
            return None
        return {
            "mean": vals.mean(axis=0).tolist(),
            "std": vals.std(axis=0).tolist(),
            "min": vals.min(axis=0).tolist(),
            "max": vals.max(axis=0).tolist(),
            "count": len(vals),
        }

    stats = {"observation.state": stats_for("observation.state", DEFAULT_STATE_DIM)}
    stats["action"] = stats_for("action", plan.action_dim)
    for key in IMU_KEYS:
        s = stats_for(f"observation.imu.{key}", 3)
        if s is not None:
            stats[f"observation.imu.{key}"] = s
    return {k: v for k, v in stats.items() if v is not None}


def _float_list_column(values: list, dtype=float) -> Any:
    """显式 float32 列表列（pyarrow 从 Python float 推断为 double，且
    高版本 pyarrow 不支持 list<double> -> list<float> 的 cast，故从源头
    用 numpy float32 构造）。"""
    arrays = [
        np.asarray(v, dtype=np.float32) if v is not None else None for v in values
    ]
    return arrays


def apply_conversion(plan: ConversionPlan) -> List[str]:
    """执行规划: dry_run 时不写任何文件；否则写出 LeRobot 布局。

    返回错误列表（空 = 成功）。绝不修改输入数据集。
    """
    if plan.dry_run:
        return []
    if plan.errors:
        return list(plan.errors)
    out = Path(plan.output_root)
    errors: List[str] = []

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return [
            "写出 LeRobot Parquet 需要 pyarrow；请先安装（如 pip install --user pyarrow）"
        ]

    try:
        (out / LEROBOT_META_DIR).mkdir(parents=True, exist_ok=True)
        (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        (out / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)

        # 图像复制
        for src, dst_rel in plan.image_copies:
            dst = out / dst_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        # 帧数据 Parquet: observation.state/action/imu/timestamp 显式 float32，
        # 索引列 int64
        float_list_cols = {
            "observation.state",
            "action",
            "observation.imu.gyro",
            "observation.imu.accel",
        }
        int_cols = {"episode_index", "frame_index", "task_index", "index"}
        frame_fields: List[Any] = []
        frame_arrays: List[Any] = []
        ordered_names = list(plan.frames[0].keys()) if plan.frames else []
        for name in ordered_names:
            values = [r.get(name) for r in plan.frames]
            if name in float_list_cols:
                frame_fields.append(pa.field(name, pa.list_(pa.float32())))
                frame_arrays.append(pa.array(_float_list_column(values), type=pa.list_(pa.float32())))
            elif name in int_cols:
                frame_fields.append(pa.field(name, pa.int64()))
                frame_arrays.append(pa.array(values, type=pa.int64()))
            elif name == "timestamp":
                frame_fields.append(pa.field(name, pa.float32()))
                frame_arrays.append(
                    pa.array(np.asarray(values, dtype=np.float32), type=pa.float32())
                )
            else:
                frame_fields.append(pa.field(name, pa.string()))
                frame_arrays.append(pa.array([v if isinstance(v, str) else None for v in values], type=pa.string()))
        pq.write_table(
            pa.Table.from_arrays(frame_arrays, schema=pa.schema(frame_fields)),
            out / "data" / f"{CHUNK_FILE}.parquet",
        )

        # episodes Parquet: 时间戳 float32，其余 int64
        ep_fields = [
            pa.field("episode_index", pa.int64()),
            pa.field("task_index", pa.int64()),
            pa.field("length", pa.int64()),
            pa.field("from_timestamp", pa.float32()),
            pa.field("to_timestamp", pa.float32()),
            pa.field("data/chunk_index", pa.int64()),
            pa.field("data/file_index", pa.int64()),
        ]
        ep_arrays = [
            pa.array([e["episode_index"] for e in plan.episodes], type=pa.int64()),
            pa.array([e["task_index"] for e in plan.episodes], type=pa.int64()),
            pa.array([e["length"] for e in plan.episodes], type=pa.int64()),
            pa.array(
                np.asarray([e["from_timestamp"] for e in plan.episodes], dtype=np.float32),
                type=pa.float32(),
            ),
            pa.array(
                np.asarray([e["to_timestamp"] for e in plan.episodes], dtype=np.float32),
                type=pa.float32(),
            ),
            pa.array([e["data/chunk_index"] for e in plan.episodes], type=pa.int64()),
            pa.array([e["data/file_index"] for e in plan.episodes], type=pa.int64()),
        ]
        pq.write_table(
            pa.Table.from_arrays(ep_arrays, schema=pa.schema(ep_fields)),
            out / "meta" / "episodes" / f"{CHUNK_FILE}.parquet",
        )
        tasks = [
            {"task_index": 0, "task": plan.robot_type, "task_description": ""}
        ]
        pq.write_table(pa.Table.from_pylist(tasks), out / "meta" / "tasks.parquet")

        info = {
            "codebase_version": DEFAULT_CODEBASE_VERSION,
            "robot_type": plan.robot_type,
            "fps": plan.fps,
            "total_episodes": len(plan.episodes),
            "total_frames": len(plan.frames),
            "total_tasks": 1,
            "chunks_size": 1000,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": None,
            "storage_format": "lerobot",
            "splits": {"train": f"0:{len(plan.episodes)}"},
            "features": _features(plan.action_dim, 480, 640),
        }
        (out / LEROBOT_META_DIR / "info.json").write_text(
            json.dumps(info, indent=2), encoding="utf-8"
        )
        (out / LEROBOT_META_DIR / "stats.json").write_text(
            json.dumps(_feature_stats(plan), indent=2), encoding="utf-8"
        )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"写出失败: {exc}")
    return errors


def main(args=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="将 rml63_bridge JSONL+PNG 数据集转换为 LeRobot 0.6.2 布局"
    )
    parser.add_argument("--input", required=True, help="输入数据集根目录 (JSONL+PNG)")
    parser.add_argument("--output", required=True, help="输出数据集根目录 (LeRobot 布局)")
    parser.add_argument("--action-dim", type=int, default=6)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--robot-type", default="rml63")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="只做规划与统计，不写任何文件（默认开启）",
    )
    parser.add_argument("--write", action="store_true", help="实际写出（默认 dry-run）")
    args = parser.parse_args(args)
    dry_run = not args.write

    plan = plan_conversion(
        args.input, args.output, action_dim=args.action_dim, fps=args.fps,
        robot_type=args.robot_type, dry_run=dry_run,
    )
    print(plan.summary())
    if plan.errors:
        print("拒绝转换或存在错误（未修改任何数据）")
        return 1
    if dry_run:
        print("dry-run 完成：未创建输出目录、未复制图像、未覆盖原始数据。")
        return 0
    errors = apply_conversion(plan)
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return 1
    print(f"转换完成: {plan.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
