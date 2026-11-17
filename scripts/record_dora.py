#!/usr/bin/env python3
"""入口 2：用真实 Dora runtime 做一次离线 mock 录制（无 ROS2、无硬件）。

用法（系统 Python 3.10，需已安装 dora-rs）:
    python3 scripts/record_dora.py [--seconds 4] [--task rebar_tying]

流程:
  1. 从脚本自身位置推导项目根目录（可整体迁移）；
  2. 以 dataflows/rml63_d435i_record.yml 为模板，生成本次运行的
     dataflow（节点路径 / PYTHONPATH / 输出路径全部重写为项目内路径）；
  3. 每个运行写入 datasets/runs/<run_id>/：
       stage/    离线 staging（episode parquet + PNG + meta_data）
       lerobot/  LeRobot 0.6.2 输出（meta/ data/ images/）
       dataflow.yml  本次实际使用的 dataflow
       control.ctl    episode 控制文件
  4. dora.run 启动真实 runtime，后台线程在 --seconds 后写 stop 控制命令；
  5. recorder 收尾并转换 LeRobot 0.6.2 格式。

约束: 仅离线 mock 录制。不启动 rm_driver / realsense2_camera，不连接硬件，
不发送控制指令，不训练 ACT。模拟 action 仅用于数据格式验证。
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (
    PROJECT_ROOT / "ros2_ws" / "src" / "rml63_bridge"
    / "dataflows" / "rml63_d435i_record.yml"
)
BRIDGE_SRC = PROJECT_ROOT / "ros2_ws" / "src" / "rml63_bridge"


def render_dataflow(run_dir: Path, task: str, action_mode: str,
                    invalid_every: int) -> Path:
    """模板 -> 本次运行的 dataflow YAML（路径全部按项目根重写）。"""
    if not TEMPLATE.is_file():
        raise FileNotFoundError(f"dataflow 模板不存在: {TEMPLATE}")
    dataflow = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))

    stage = run_dir / "stage"
    out = run_dir / "lerobot"
    ctl = run_dir / "control.ctl"

    for node in dataflow["nodes"]:
        # 节点脚本路径 + PYTHONPATH 一律按当前项目根重写
        rel_script = Path(node["path"]).name
        node["path"] = str(BRIDGE_SRC / "rml63_bridge" / "dora_nodes" / rel_script)
        node["env"]["PYTHONPATH"] = str(BRIDGE_SRC)
        if node["id"] == "rml63_dataset_recorder":
            node["env"].update(
                {
                    "RML63_DORA_DATASET_ROOT": str(stage),
                    "RML63_DORA_OUTPUT_ROOT": str(out),
                    "RML63_DORA_CONTROL_FILE": str(ctl),
                    "RML63_DORA_TASK": task,
                }
            )
        if node["id"] == "mock_action":
            node["env"]["ACTION_MODE"] = action_mode
        for mock_node in ("rml63_state", "d435i_camera"):
            if node["id"] == mock_node:
                node["env"][
                    "STATE_MOCK_INVALID_EVERY" if mock_node == "rml63_state"
                    else "CAMERA_MOCK_INVALID_EVERY"
                ] = str(invalid_every)

    yaml_path = run_dir / "dataflow.yml"
    yaml_path.write_text(
        yaml.safe_dump(dataflow, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return yaml_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="离线 mock 录制（真实 Dora runtime）")
    parser.add_argument("--seconds", type=float, default=4.0,
                        help="录制时长（秒），到点自动写 stop 控制命令")
    parser.add_argument("--task", default="rebar_tying")
    parser.add_argument("--action-mode", choices=["mock", "none"], default="mock",
                        help="none = 全部帧无 action（验证拒绝训练数据）")
    parser.add_argument("--invalid-every", type=int, default=5,
                        help="mock 每 N 帧注入 1 帧 invalid（<=0 关闭）")
    args = parser.parse_args(argv)

    try:
        import dora  # noqa: F401
    except ImportError:
        print("错误: 未安装 dora-rs（系统 Python 需要: "
              "python3 -m pip install --user 'dora-rs>=0.5.0'）", file=sys.stderr)
        return 1

    run_id = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "datasets" / "runs" / run_id
    (run_dir / "stage").mkdir(parents=True, exist_ok=True)

    yaml_path = render_dataflow(run_dir, args.task, args.action_mode, args.invalid_every)
    ctl = run_dir / "control.ctl"
    if ctl.exists():
        ctl.unlink()

    print(f"项目根:   {PROJECT_ROOT}")
    print(f"运行目录: {run_dir}")
    print(f"dataflow: {yaml_path}")
    print(f"任务: {args.task} | 时长: {args.seconds}s | action 模式: {args.action_mode}")
    print("模拟 action 仅用于数据格式验证，不代表真实示范动作，不用于研究结论。")
    print("启动 dora runtime ...", flush=True)

    # 注意: dora.run() 是持 GIL 的阻塞 Rust 调用，Python 后台线程在录制期间
    # 无法执行（实测 sleep 4s 的线程在 dora.run 返回后才写文件）。控制文件
    # 必须由独立进程写入（bash subshell 进程不受 GIL 影响）。
    import subprocess

    subprocess.Popen(
        ["bash", "-c", f"sleep {args.seconds}; echo stop > {ctl}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"[ctl] 将在 {args.seconds}s 后由独立进程写入 stop", flush=True)

    dora.run(str(yaml_path), stop_after=args.seconds + 10)

    info = run_dir / "lerobot" / "meta" / "info.json"
    if info.is_file():
        import json

        meta = json.loads(info.read_text(encoding="utf-8"))
        print(f"录制完成: episodes={meta.get('total_episodes')} "
              f"frames={meta.get('total_frames')}")
    else:
        print(f"警告: 未找到输出 {info}（录制可能被拒绝或失败）", file=sys.stderr)

    print(f"验证命令（lerobot conda env）:\n"
          f"  {sys.executable if 'miniconda' in sys.executable else '/home/dys/miniconda3/envs/lerobot/bin/python'} "
          f"scripts/verify_dataset.py --dataset {run_dir / 'lerobot'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
