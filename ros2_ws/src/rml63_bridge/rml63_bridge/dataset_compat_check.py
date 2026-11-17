"""LeRobot 0.6.2 兼容性检查器（静态检查，不启动 ROS2、不训练）。

期望格式来自对 LeRobot 0.6.2 源码的静态审计（2026-11-03）:
  - 目录: meta/ (info.json, stats.json, tasks.parquet, episodes/...),
    data/chunk-XXX/file-YYY.parquet, images/<key>/episode-XXXXXX/frame-YYYYYY.png
    （深度帧默认 .tiff）或 videos/<key>/chunk-XXX/file-YYY.mp4
  - info.json 必需字段: codebase_version (str), fps (int), features (dict)
  - stats.json 可选: {feature: {mean, std, min, max, count, percentiles...}}
  - ACT 训练要求 features 含 "action" 且行数据 action 非空
    （policies/act/configuration_act.py: "action" is required as an output key）
  - observation.state: float32 向量 [state_dim]；action: float32 向量 [action_dim]

本检查器只读，绝不伪造 action，绝不把 observation.state 当作 action。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# ---- LeRobot 0.6.2 期望布局（静态审计结论） --------------------------------
LEROBOT_META_DIR = "meta"  # 注意: 不是 meta_data
LEROBOT_EXPECTED_FILES = {
    "info": "meta/info.json",
    "stats": "meta/stats.json",  # 加载时可选，但训练通常需要
    "tasks": "meta/tasks.parquet",
    "episodes": "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    "data": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    "image": "images/{image_key}/episode-{episode_index:06d}/frame-{frame_index:06d}.png",
    "depth": "images/{image_key}/episode-{episode_index:06d}/frame-{frame_index:06d}.tiff",
}
LEROBOT_REQUIRED_INFO_FIELDS = ("codebase_version", "fps", "features")
ACT_TRAINING_ACTION_MESSAGE = "当前数据集缺少训练动作，不能用于 ACT 训练"
DEFAULT_STATE_DIM = 25

LEVELS = ("critical", "error", "warning", "info")


@dataclass
class CompatIssue:
    level: str  # critical / error / warning / info
    category: str
    message: str

    def to_text(self) -> str:
        return f"[{self.level.upper()}] {self.category}: {self.message}"


@dataclass
class CompatReport:
    dataset_root: str
    backend: str = "unknown"
    issues: List[CompatIssue] = field(default_factory=list)
    n_episodes: int = 0
    n_frames: int = 0
    n_frames_with_action: int = 0
    n_frames_action_null: int = 0
    state_dim: Optional[int] = None

    @property
    def trainable(self) -> bool:
        return not any(i.level == "critical" for i in self.issues)

    def to_text(self) -> str:
        lines = [
            f"数据集根目录: {self.dataset_root}",
            f"识别后端: {self.backend}",
            f"episodes={self.n_episodes} frames={self.n_frames} "
            f"(action 非空 {self.n_frames_with_action}, action=null {self.n_frames_action_null})",
            f"observation.state 维度: {self.state_dim}",
            f"可训练(ACT): {'是' if self.trainable else '否'}",
        ]
        for issue in self.issues:
            lines.append("  " + issue.to_text())
        return "\n".join(lines)


def _read_offline_info(root: Path) -> Optional[dict]:
    p = root / "meta_data" / "info.json"
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    return None


def _read_offline_rows(root: Path) -> List[dict]:
    """读回离线数据集的 episode 数据（jsonl 或 parquet 双后端）。

    dataset_writer 的后端取决于运行环境有无 pyarrow：有 pyarrow ->
    data/episode_*.parquet；无 pyarrow -> data/episode_*.jsonl。
    """
    rows: List[dict] = []
    data_dir = root / "data"
    if not data_dir.is_dir():
        return rows
    for f in sorted(data_dir.glob("episode_*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    for f in sorted(data_dir.glob("episode_*.parquet")):
        import pyarrow.parquet as pq

        rows.extend(pq.read_table(f).to_pylist())
    return rows


def check_dataset(
    dataset_root: str,
    expected_state_dim: int = DEFAULT_STATE_DIM,
    action_dim: Optional[int] = None,
) -> CompatReport:
    """对照 LeRobot 0.6.2 期望格式检查数据集，输出兼容性报告。"""
    root = Path(dataset_root)
    report = CompatReport(dataset_root=str(root))

    if not root.is_dir():
        report.issues.append(CompatIssue("critical", "root", f"目录不存在: {root}"))
        return report

    # 1. 后端识别 + 目录结构
    jsonl_files = sorted((root / "data").glob("episode_*.jsonl")) if (root / "data").is_dir() else []
    parquet_files = (
        sorted((root / "data").rglob("*.parquet")) if (root / "data").is_dir() else []
    )
    if jsonl_files and not parquet_files:
        report.backend = "jsonl+png (rml63_bridge 离线后端)"
        report.issues.append(
            CompatIssue(
                "error",
                "backend",
                "LeRobot 0.6.2 的 data/tasks/episodes 必须是 Parquet；JSONL 不能直接训练",
            )
        )
    elif parquet_files:
        episode_parquet = sorted((root / "data").glob("episode_*.parquet"))
        report.backend = (
            "parquet+png (rml63_bridge 离线后端)"
            if episode_parquet
            else "lerobot-native parquet"
        )
    else:
        report.issues.append(CompatIssue("critical", "data", "data/ 目录为空或不存在"))

    if (root / "meta_data").is_dir() and not (root / LEROBOT_META_DIR).is_dir():
        report.issues.append(
            CompatIssue(
                "error",
                "layout",
                f"检测到 meta_data/ 目录，但 LeRobot 0.6.2 要求 {LEROBOT_META_DIR}/"
                "（info.json、stats.json、tasks.parquet、episodes/ 均在 meta/ 下）",
            )
        )

    # 2. info.json
    info = _read_offline_info(root)
    if info is None:
        report.issues.append(
            CompatIssue("error", "info.json", f"缺少 {LEROBOT_META_DIR}/info.json 或无法解析")
        )
    else:
        missing = [k for k in LEROBOT_REQUIRED_INFO_FIELDS if k not in info]
        if missing:
            report.issues.append(
                CompatIssue(
                    "error",
                    "info.json",
                    f"缺少 LeRobot 必需字段: {missing}（需要 codebase_version/fps/features）",
                )
            )
        features = info.get("features")
        if isinstance(features, dict) and "action" not in features:
            report.issues.append(
                CompatIssue(
                    "error",
                    "info.json",
                    "features 中没有 action 特征；ACT 训练要求 action 作为输出特征",
                )
            )

    # 3. stats.json
    stats_path = root / LEROBOT_META_DIR / "stats.json"
    offline_stats = root / "meta_data" / "stats.json"
    if not stats_path.is_file() and not offline_stats.is_file():
        report.issues.append(
            CompatIssue("warning", "stats.json", "缺少 stats 文件（加载时可选，训练建议提供）")
        )

    # 4. episode 行数据
    rows = _read_offline_rows(root)
    report.n_episodes = len({r.get("episode_index") for r in rows})
    report.n_frames = len(rows)
    if not rows:
        report.issues.append(CompatIssue("critical", "data", "没有可读取的帧数据"))
        return report

    action_null = 0
    action_present = 0
    prev_ts = None
    for r in rows:
        # observation.state
        state = r.get("observation.state")
        if state is None:
            report.issues.append(
                CompatIssue("error", "observation.state", f"episode {r.get('episode_index')} frame {r.get('frame_index')} 缺 observation.state")
            )
        else:
            if report.state_dim is None:
                report.state_dim = len(state)
            if len(state) != expected_state_dim:
                report.issues.append(
                    CompatIssue(
                        "error",
                        "observation.state",
                        f"state 维度 {len(state)} != 期望 {expected_state_dim}",
                    )
                )
        # action
        if "action" not in r:
            report.issues.append(
                CompatIssue(
                    "error",
                    "action",
                    f"episode {r.get('episode_index')} frame {r.get('frame_index')} 缺少 action 字段",
                )
            )
        elif r["action"] is None:
            action_null += 1
        else:
            action_present += 1
            if action_dim is not None and len(r["action"]) != action_dim:
                report.issues.append(
                    CompatIssue(
                        "error",
                        "action",
                        f"action 维度 {len(r['action'])} != 期望 {action_dim}",
                    )
                )
        # 图像路径
        for key in ("rgb", "depth", "aligned_depth"):
            p = r.get(f"observation.images.{key}.path")
            if p is None:
                continue  # 缺失图像: 保留 None（本检查器只报告文件级缺失）
            if not (root / p).is_file():
                report.issues.append(
                    CompatIssue(
                        "error",
                        "image",
                        f"图像文件缺失: {p} (episode {r.get('episode_index')} frame {r.get('frame_index')})",
                    )
                )
        # 时间戳单调
        ts = r.get("timestamp")
        if ts is not None and prev_ts is not None and ts <= prev_ts:
            report.issues.append(CompatIssue("warning", "timestamp", "时间戳非严格递增"))
        if ts is not None:
            prev_ts = ts

    report.n_frames_with_action = action_present
    report.n_frames_action_null = action_null
    if action_present == 0:
        report.issues.append(CompatIssue("critical", "action", ACT_TRAINING_ACTION_MESSAGE))

    return report


def main(args=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="检查数据集是否符合 LeRobot 0.6.2 格式（只读，不伪造 action）"
    )
    parser.add_argument("--dataset-root", required=True, help="数据集根目录")
    parser.add_argument("--state-dim", type=int, default=DEFAULT_STATE_DIM)
    parser.add_argument("--action-dim", type=int, default=None)
    args = parser.parse_args(args)

    report = check_dataset(args.dataset_root, args.state_dim, args.action_dim)
    print(report.to_text())
    return 0 if report.trainable else 1


if __name__ == "__main__":
    raise SystemExit(main())
