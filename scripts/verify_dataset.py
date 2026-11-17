#!/usr/bin/env python3
"""入口 3：用本项目的 LeRobot 源码读回录制的数据集（lerobot conda env）。

用法（lerobot conda env，Python 3.12）:
    /home/dys/miniconda3/envs/lerobot/bin/python scripts/verify_dataset.py \
        [--dataset datasets/runs/<run_id>/lerobot]

不修改旧 conda 环境的 editable 指向：本脚本把 <项目根>/lerobot/src 置于
sys.path 最前（限定源码路径），并打印实际导入的 lerobot 文件路径。

检查项:
  - 帧数、observation.state 25 维 float32、action 6 维 float32；
  - action != observation.state[:6]（未伪造）；
  - RGB 与深度图像实际解码（torch 张量 -> numpy）；
  - 图像列路径全部位于本项目内（不引用 /tmp 或旧目录），文件均存在。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEROBOT_SRC = PROJECT_ROOT / "lerobot" / "src"

# 限定源码路径：优先本项目副本，不动 conda env 的 editable 安装
sys.path.insert(0, str(LEROBOT_SRC))

import numpy as np  # noqa: E402


def latest_run_dir() -> Path:
    runs = sorted((PROJECT_ROOT / "datasets" / "runs").glob("run_*"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise SystemExit(f"未找到任何录制输出: {PROJECT_ROOT}/datasets/runs/")
    return runs[0] / "lerobot"


def check_image_paths(dataset: Path) -> tuple[int, int]:
    """返回 (检查的图像数, 失败数)；路径必须在本项目内且文件存在。"""
    import pyarrow.parquet as pq

    bad = 0
    checked = 0
    for f in sorted((dataset / "data").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(f)
        for col in table.column_names:
            if not col.startswith("observation.images."):
                continue
            for v in table[col].to_pylist():
                checked += 1
                if not isinstance(v, str) or not v.startswith(f"{PROJECT_ROOT}/"):
                    print(f"  [BAD] 路径不在项目内: {v!r}")
                    bad += 1
                elif not Path(v).is_file():
                    print(f"  [BAD] 文件不存在: {v}")
                    bad += 1
    return checked, bad


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="LeRobotDataset 读回验证")
    parser.add_argument("--dataset", default=None,
                        help="数据集目录（默认: 最新一次运行）")
    args = parser.parse_args(argv)

    dataset = Path(args.dataset) if args.dataset else latest_run_dir()
    if not dataset.is_dir():
        raise SystemExit(f"数据集不存在: {dataset}")

    import lerobot
    from lerobot.datasets import LeRobotDataset

    print(f"项目根:        {PROJECT_ROOT}")
    print(f"数据集:        {dataset}")
    print(f"lerobot 源码:  {Path(lerobot.__file__).resolve()}")
    print(f"lerobot 版本:  {lerobot.__version__}")

    ds = LeRobotDataset(str(dataset))
    item = ds[0]
    state = item["observation.state"]
    action = item["action"]
    rgb = item["observation.images.rgb"]
    depth = item["observation.images.depth"]

    print(f"帧数:          {len(ds)}")
    print(f"state:         {list(state.shape)} {state.dtype}")
    print(f"action:        {list(action.shape)} {action.dtype}")
    print(f"action≠state:  {not (state[:6] == action).all()}")
    print(f"rgb 张量:      {list(rgb.shape)} | depth 张量: {list(depth.shape)}")

    # 实际解码为 numpy
    rgb_np = rgb.numpy()
    depth_np = depth.numpy()
    print(f"rgb 解码:      {rgb_np.shape} dtype={rgb_np.dtype} "
          f"min={rgb_np.min()} max={rgb_np.max()}")
    print(f"depth 解码:    {depth_np.shape} dtype={depth_np.dtype} "
          f"min={depth_np.min()} max={depth_np.max()} (≤65535: {depth_np.max() <= 65535})")

    info = json.loads((dataset / "meta" / "info.json").read_text(encoding="utf-8"))
    print(f"codebase:      {info.get('codebase_version')} | robot_type: "
          f"{info.get('robot_type')}")

    # 图像路径检查
    checked, bad = check_image_paths(dataset)
    print(f"图像路径检查:  {checked} 条, 失败 {bad}")
    if checked == 0:
        print("  [BAD] 未检查到任何图像列")
        bad += 1

    # 断言汇总
    import torch

    failures = []
    if not (state.shape == (25,) and state.dtype == torch.float32):
        failures.append(f"state 形状/类型错误: {state.shape} {state.dtype}")
    if not (action.shape == (6,) and action.dtype == torch.float32):
        failures.append(f"action 形状/类型错误: {action.shape} {action.dtype}")
    if (state[:6] == action).all():
        failures.append("action 与 state[:6] 相同（伪造嫌疑）")
    if rgb_np.max() > 255:
        failures.append("rgb 超出 uint8 范围")
    if depth_np.max() > 65535:
        failures.append("depth 超出 uint16 范围")
    if bad:
        failures.append(f"{bad} 条图像路径异常")
    if failures:
        print("结论: 验证失败")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("结论: 验证通过（帧数 / 25 维状态 / 6 维动作 / RGB 与深度解码 / 图像路径）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
