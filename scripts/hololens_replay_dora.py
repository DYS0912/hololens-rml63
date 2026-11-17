#!/usr/bin/env python3
"""入口：已保存手部记录 → Dora 回放节点 → 虚拟目标转换节点 → 保存节点。

用真实 Dora runtime 运行（dora.run），不是普通 Python 模拟。

用法（项目根目录，系统 Python 3.10，已安装 dora-rs）:
    python3 scripts/hololens_replay_dora.py
        [--input datasets/hololens_checks/run_20261112_101842/packets.jsonl]
        [--expected datasets/hololens_checks/run_20261112_101842/target_packets.jsonl]
        [--scale 0.2] [--interval-threshold 0.2] [--chunk-size 100]
        [--timeout 120]

默认值: --input 取 datasets/hololens_checks 下最新 run_* 的 packets.jsonl，
--expected 取同一目录的 target_packets.jsonl。

流程:
  1. 自动选择新输出目录 datasets/hololens_checks/dora_replay_<时间戳>/
     （重名自动加 _2，不覆盖任何已有文件）；
  2. 按模板生成 dataflow.yml（节点/输入/输出/标记路径全部重写为项目内路径）；
  3. dora.run 启动真实 runtime：
       - 回放节点：首个定时 tick 后按文件顺序分批发送全部记录，最后发送
         {"_eos": true} 结束标记并退出；
       - 转换节点：复用 scripts/hololens_target_converter.py 的状态机逐帧
         转换（未标定虚拟位置预览、仅平移、旋转映射未实现），EOS 原样转发；
         间隔判定使用记录内原始时间，回放速度不影响结果；
       - 保存节点：逐行写 replay_output.jsonl（含有效帧、暂停帧、异常帧
         与无效原因），收到 EOS 后 flush + 关闭文件、写完成标记
         replay_done.json，然后退出；
       - dora.run 在所有节点退出后自行返回（实测），不靠固定等待强制终止；
         --timeout 只是 stop_after 兜底上界，触达即说明流程异常；
  4. 与 --expected 逐帧比较（数量、顺序、session_id、sequence、
     target_valid、target_position 浮点容差、target_rotation 的 null、
     暂停/异常分类），排除 _replay 等回放诊断字段；比较报告写入
     compare_report.json，打印输出路径。

限制: 不监听实时 UDP，不启动 ROS2，不连接头显/机械臂/相机，不发送控制
指令，不修改官方 SDK，不安装新依赖。输出保持“未标定的虚拟位置预览”标识，
不生成 action，不转换成训练数据。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (
    PROJECT_ROOT / "ros2_ws" / "src" / "rml63_bridge"
    / "dataflows" / "hololens_replay_offline.yml"
)
BRIDGE_SRC = PROJECT_ROOT / "ros2_ws" / "src" / "rml63_bridge"
DORA_NODES_DIR = BRIDGE_SRC / "rml63_bridge" / "dora_nodes"
CHECKS_ROOT = PROJECT_ROOT / "datasets" / "hololens_checks"

COMPARE_FIELDS = ("session_id", "sequence", "target_valid",
                  "target_position", "target_rotation", "frame_class")
EXCLUDED_FIELDS = ("_replay", "_raw_input_line")


def _preview(rec: dict):
    return rec.get("preview_target") or rec.get("_preview_target")


def _counts(path: Path) -> dict:
    counts = {"total": 0, "valid_output": 0, "paused": 0, "abnormal": 0,
              "target_valid": 0}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            counts["total"] += 1
            pt = _preview(rec)
            cls = (pt or {}).get("frame_class")
            if cls in counts:
                counts[cls] += 1
            if (pt or {}).get("target_valid"):
                counts["target_valid"] += 1
    return counts


def compare_target_streams(actual_path, expected_path,
                           position_tol: float = 1e-9,
                           max_mismatches: int = 50):
    """逐帧比较真实 Dora 输出与已有 target_packets.jsonl。

    比较: 数量、顺序（行序）、session_id、sequence、target_valid、
    target_position（浮点容差）、target_rotation（双方均为 null）、
    frame_class（暂停/异常分类）。排除 _replay 等回放诊断字段。

    返回 (ok: bool, report: dict)。
    """
    def load(path):
        records = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    records.append(json.loads(line))
        return records

    actual = load(actual_path)
    expected = load(expected_path)
    report = {
        "actual_path": str(actual_path),
        "expected_path": str(expected_path),
        "position_tolerance": position_tol,
        "compared_fields": list(COMPARE_FIELDS),
        "excluded_fields": list(EXCLUDED_FIELDS),
        "actual_lines": len(actual),
        "expected_lines": len(expected),
        "actual_counts": _counts(actual_path),
        "expected_counts": _counts(expected_path),
        "mismatch_count": 0,
        "mismatches": [],
        "ok": False,
    }
    mismatches = []
    n = min(len(actual), len(expected))
    for i in range(n):
        a, e = actual[i], expected[i]
        pa_, pe_ = _preview(a), _preview(e)
        row = i + 1
        if a.get("session_id") != e.get("session_id"):
            mismatches.append({"line": row, "field": "session_id",
                               "expected": e.get("session_id"),
                               "actual": a.get("session_id")})
        if a.get("sequence") != e.get("sequence"):
            mismatches.append({"line": row, "field": "sequence",
                               "expected": e.get("sequence"),
                               "actual": a.get("sequence")})
        if (pa_ or {}).get("target_valid") != (pe_ or {}).get("target_valid"):
            mismatches.append({"line": row, "field": "target_valid",
                               "expected": (pe_ or {}).get("target_valid"),
                               "actual": (pa_ or {}).get("target_valid")})
        ap, ep = (pa_ or {}).get("target_position"), (pe_ or {}).get(
            "target_position")
        if (ap is None) != (ep is None) or (
                ap is not None and len(ap) == len(ep) and any(
                    abs(x - y) > position_tol for x, y in zip(ap, ep))):
            mismatches.append({"line": row, "field": "target_position",
                               "expected": ep, "actual": ap})
        if (pa_ or {}).get("target_rotation") != (pe_ or {}).get(
                "target_rotation"):
            mismatches.append({"line": row, "field": "target_rotation",
                               "expected": (pe_ or {}).get("target_rotation"),
                               "actual": (pa_ or {}).get("target_rotation")})
        if (pa_ or {}).get("frame_class") != (pe_ or {}).get("frame_class"):
            mismatches.append({"line": row, "field": "frame_class",
                               "expected": (pe_ or {}).get("frame_class"),
                               "actual": (pa_ or {}).get("frame_class")})
    if len(actual) != len(expected):
        mismatches.append({"line": n + 1, "field": "count",
                           "expected": len(expected), "actual": len(actual)})
    report["mismatch_count"] = len(mismatches)
    report["mismatches"] = mismatches[:max_mismatches]
    report["ok"] = not mismatches
    return report["ok"], report


def latest_check_run() -> Path:
    """datasets/hololens_checks 下最新的 run_* 目录。"""
    runs = sorted(CHECKS_ROOT.glob("run_*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"未找到采集运行目录: {CHECKS_ROOT}")
    return runs[-1]


def make_run_dir(root: Path, prefix: str = "dora_replay_") -> Path:
    """创建独立运行目录；重名自动加 _2，绝不覆盖。"""
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i in range(1000):
        suffix = "" if i == 0 else f"_{i + 1}"
        run_dir = root / f"{prefix}{stamp}{suffix}"
        try:
            run_dir.mkdir(exist_ok=False)
            return run_dir
        except FileExistsError:
            continue
    raise RuntimeError(f"无法在 {root} 创建独立运行目录")


def render_dataflow(run_dir: Path, args, input_path: Path) -> Path:
    """模板 -> 本次运行的 dataflow.yml（路径全部按项目根重写）。"""
    if not TEMPLATE.is_file():
        raise FileNotFoundError(f"dataflow 模板不存在: {TEMPLATE}")
    dataflow = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))

    output_path = run_dir / "replay_output.jsonl"
    marker_path = run_dir / "replay_done.json"
    bridge_pythonpath = str(BRIDGE_SRC)
    all_pythonpath = bridge_pythonpath + ":" + str(PROJECT_ROOT / "scripts")

    for node in dataflow["nodes"]:
        rel_script = Path(node["path"]).name
        node["path"] = str(DORA_NODES_DIR / rel_script)
        env = node["env"]
        env["PYTHONPATH"] = (
            all_pythonpath if node["id"] == "hololens_target_converter"
            else bridge_pythonpath)
        if node["id"] == "hololens_replay":
            env["RML63_HL_REPLAY_INPUT"] = str(input_path.resolve())
            env["RML63_HL_CHUNK_SIZE"] = str(args.chunk_size)
        elif node["id"] == "hololens_target_converter":
            env["RML63_HL_SCALE"] = str(args.scale)
            env["RML63_HL_INTERVAL_THRESHOLD"] = str(args.interval_threshold)
        elif node["id"] == "hololens_saver":
            env["RML63_HL_OUTPUT"] = str(output_path)
            env["RML63_HL_DONE_MARKER"] = str(marker_path)

    yaml_path = run_dir / "dataflow.yml"
    yaml_path.write_text(
        yaml.safe_dump(dataflow, allow_unicode=True, sort_keys=False),
        encoding="utf-8")
    return yaml_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="已保存手部记录 → Dora 回放 → 虚拟目标转换 → 保存"
                    "（真实 Dora runtime，仅离线）")
    parser.add_argument("--input", default=None,
                        help="输入 packets.jsonl（默认最新 run_* 的 "
                             "packets.jsonl）")
    parser.add_argument("--expected", default=None,
                        help="用于逐帧核对的 target_packets.jsonl"
                             "（默认同目录 target_packets.jsonl）")
    parser.add_argument("--scale", type=float, default=0.2,
                        help="平移缩放系数（有限正数，默认 0.2）")
    parser.add_argument("--interval-threshold", type=float, default=0.2,
                        help="接收间隔阈值秒（默认 0.2，离线记录间隔检查）")
    parser.add_argument("--chunk-size", type=int, default=10000,
                        help="回放批大小（默认 10000=单条大消息。dora 0.5 "
                             "实测多批快发会丢失中间批次，单条消息 100%% 送达；"
                             "转换结果与批大小无关）")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="dora.run stop_after 兜底上界秒（默认 120；"
                             "触达说明流程异常，正常靠节点退出自行结束）")
    args = parser.parse_args(argv)

    try:
        import dora  # noqa: F401
    except ImportError:
        print("错误: 未安装 dora-rs（系统 Python 需要: "
              "python3 -m pip install --user 'dora-rs>=0.5.0'）",
              file=sys.stderr)
        return 1

    if args.input is None:
        run_dir_src = latest_check_run()
        input_path = run_dir_src / "packets.jsonl"
    else:
        input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        print(f"[错误] 输入文件不存在: {input_path}", file=sys.stderr)
        return 2
    if args.expected is None:
        expected_path = input_path.parent / "target_packets.jsonl"
    else:
        expected_path = Path(args.expected).expanduser().resolve()
    if not expected_path.is_file():
        print(f"[错误] 核对文件不存在: {expected_path}", file=sys.stderr)
        return 2
    if not 0 < args.chunk_size:
        parser.error("--chunk-size 必须为正整数")
    if not (math.isfinite(args.scale) and args.scale > 0):
        parser.error("--scale 必须为有限正数")
    if not (math.isfinite(args.interval_threshold)
            and args.interval_threshold > 0):
        parser.error("--interval-threshold 必须为有限正数")
    if not (math.isfinite(args.timeout) and args.timeout > 0):
        parser.error("--timeout 必须为有限正数")

    output_dir = make_run_dir(CHECKS_ROOT)
    yaml_path = render_dataflow(output_dir, args, input_path)
    output_path = output_dir / "replay_output.jsonl"
    marker_path = output_dir / "replay_done.json"

    print(f"项目根:   {PROJECT_ROOT}")
    print(f"输入:     {input_path}")
    print(f"核对:     {expected_path}")
    print(f"运行目录: {output_dir}")
    print(f"dataflow: {yaml_path}")
    print("启动 dora runtime（回放 -> 转换 -> 保存；节点退出后自行结束）...",
          flush=True)

    t0 = time.monotonic()
    dora.run(str(yaml_path), stop_after=args.timeout)
    elapsed = time.monotonic() - t0
    print(f"dora.run 返回，耗时 {elapsed:.2f}s（stop_after={args.timeout}s "
          f"兜底）", flush=True)

    if not marker_path.is_file():
        print(f"[错误] 未生成完成标记 {marker_path}：流程可能在保存节点"
              f"写完前被 stop_after 终止。", file=sys.stderr)
        return 1
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("state") != "completed":
        print(f"[错误] 保存节点未正常收尾: {marker}", file=sys.stderr)
        return 1
    if not marker.get("closed"):
        print(f"[错误] 保存节点报告文件未关闭: {marker}", file=sys.stderr)
        return 1
    print(f"[保存节点] {marker['state']} | "
          f"records={marker['records_written']} counts={marker['counts']}")

    ok, report = compare_target_streams(output_path, expected_path)
    report_path = output_dir / "compare_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    ac = report["actual_counts"]
    ec = report["expected_counts"]
    print(f"[核对] Dora 输出: 总 {ac['total']} | 有效 {ac['valid_output']} | "
          f"暂停 {ac['paused']} | 异常 {ac['abnormal']}")
    print(f"[核对] 已有结果: 总 {ec['total']} | 有效 {ec['valid_output']} | "
          f"暂停 {ec['paused']} | 异常 {ec['abnormal']}")
    if report["ok"]:
        print("[核对] 逐帧比较一致（含数量/顺序/session_id/sequence/"
              "target_valid/target_position/target_rotation/分类）", flush=True)
    else:
        print(f"[核对] 不一致: 共 {report['mismatch_count']} 处不匹配"
              f"（前 {len(report['mismatches'])} 处见 compare_report.json）",
              file=sys.stderr)
        for m in report["mismatches"][:10]:
            print(f"  行 {m['line']} {m['field']}: "
                  f"期望 {m['expected']!r}，实际 {m['actual']!r}",
                  file=sys.stderr)

    print(f"[结果] 输出: {output_path}")
    print(f"[结果] 完成标记: {marker_path}")
    print(f"[结果] 比较报告: {report_path}")
    print(f"[结果] 运行目录: {output_dir}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
