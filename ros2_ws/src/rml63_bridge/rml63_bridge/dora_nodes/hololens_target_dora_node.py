#!/usr/bin/env python3
"""Dora 转换节点：复用 scripts/hololens_target_converter.py 的状态机。

输入:  records（回放节点批消息：JSON 数组字符串，或 EOS 标记）
输出:  converted（转换后批消息：JSON 数组字符串；EOS 原样转发）

说明:
  - 转换逻辑全部来自 hololens_target_converter.TargetConverter（不重写算法）：
    未标定虚拟位置预览（Unity tracking space），仅平移映射，旋转映射未实现；
  - 时间间隔判定使用记录中的原始时间（_receiver.monotonic_s 优先），
    与回放速度无关；
  - 有效帧、暂停帧与异常帧（含原因）全部保留并转发，不只保留捏合帧；
  - EOS 标记原样转发后本节点退出。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# 复用项目内转换器（scripts/hololens_target_converter.py），按自身位置定位项目根
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from hololens_target_converter import TargetConverter  # noqa: E402

import time  # noqa: E402

POST_SEND_FLUSH_S = 0.5  # EOS 转发后的冲刷余量（实测：转发后立即退出会丢 EOS）


class ConvertCore:
    """转换核心逻辑（无 dora，可离线测试）。"""

    def __init__(self, scale: float = 0.2, interval_threshold_s: float = 0.2):
        self.converter = TargetConverter(scale=scale,
                                         interval_threshold_s=interval_threshold_s)

    def on_message(self, msg):
        """处理一条消息。EOS dict 原样返回；列表逐条转换（保持顺序）。"""
        if isinstance(msg, dict) and msg.get("_eos"):
            return msg
        items = msg if isinstance(msg, list) else [msg]
        out = []
        for rec in items:
            if isinstance(rec, dict) and "_raw_input_line" in rec:
                out.append(self.converter.convert_line(
                    rec["_raw_input_line"]))
            elif isinstance(rec, dict):
                out.append(self.converter.convert_record(rec))
            else:
                out.append(self.converter.convert_line(json.dumps(rec)))
        return out

    @property
    def stats(self):
        return self.converter.stats


def _describe() -> str:
    return "\n".join([
        "node_id: hololens_target_converter",
        "inputs: records <- hololens_replay/records（JSON 数组批消息）",
        "outputs: converted（转换后批消息；EOS 原样转发）",
        "env: RML63_HL_SCALE（平移缩放，默认 0.2）",
        "     RML63_HL_INTERVAL_THRESHOLD（间隔阈值秒，默认 0.2）",
        "复用 scripts/hololens_target_converter.py 的状态机（不重写算法）。",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="hololens_target_dora_node")
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.describe:
        print(_describe())
        return 0
    if args.check:
        core = ConvertCore()
        out = core.on_message([{
            "session_id": "check", "sequence": 0, "valid": True,
            "control_enabled": True, "position": [1.0, 1.0, 1.0],
            "_receiver": {"monotonic_s": 1.0},
        }])
        pt = out[0].get("preview_target") or out[0].get("_preview_target")
        print(f"hololens_target_dora_node check: 转换 ok，首帧 "
              f"target_valid={pt['target_valid']} "
              f"target_position={pt['target_position']}")
        return 0

    try:
        from dora import Node  # noqa: PLC0415 惰性导入
        import pyarrow as pa  # noqa: PLC0415
    except ImportError:
        print("错误: dora-rs 未安装，无法作为 Dora 节点运行。"
              "离线验证请使用 --describe / --check。", file=sys.stderr)
        return 1

    core = ConvertCore(
        scale=float(os.environ.get("RML63_HL_SCALE", "0.2")),
        interval_threshold_s=float(
            os.environ.get("RML63_HL_INTERVAL_THRESHOLD", "0.2")))
    node = Node()
    eos_received = False
    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT":
            continue
        value = event.get("value")
        if value is None:
            continue
        raw = value.to_pylist() if hasattr(value, "to_pylist") else value
        if isinstance(raw, list) and len(raw) == 1:
            raw = raw[0]
        out = core.on_message(json.loads(raw))
        node.send_output("converted",
                         pa.array([json.dumps(out, ensure_ascii=False)]))
        if isinstance(out, dict) and out.get("_eos"):
            eos_received = True
            time.sleep(POST_SEND_FLUSH_S)  # 让 runtime 刷出 EOS 后再退出
            break
    stats = core.stats
    print(f"[hololens_target_converter] stopped eos_received={eos_received} "
          f"total={stats['total_records']} "
          f"valid_output={stats['valid_output']} "
          f"paused={stats['paused']} abnormal={stats['abnormal']} "
          f"reanchor={stats['reanchor_count']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
