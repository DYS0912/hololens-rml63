#!/usr/bin/env python3
"""Dora 保存节点：逐行写回放转换结果，收到 EOS 后关闭文件并写完成标记。

env:
  RML63_HL_OUTPUT       输出 JSONL 路径（必填）
  RML63_HL_DONE_MARKER  完成标记 JSON 路径（必填）

行为:
  - 保留有效帧、暂停帧与异常帧（含原因），不只保存捏合帧；
  - 收到 EOS（回放结束通知）后：flush + close 输出文件，写完成标记
    {"state": "completed", ...}，然后退出；未收到 EOS 被 STOP 时写
    {"state": "interrupted", ...}；
  - 本节点退出后 dora runtime 在全部节点退出时自行结束（实测），
    不靠固定等待强制终止。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def _preview(rec: dict):
    return rec.get("preview_target") or rec.get("_preview_target")


class SaverCore:
    """保存核心逻辑（无 dora，可离线测试）。"""

    def __init__(self, output_path, done_marker):
        self.output_path = Path(output_path)
        self.done_marker = Path(done_marker)
        self._file = None
        self.records_written = 0
        self.counts = {"valid_output": 0, "paused": 0, "abnormal": 0}

    def _open(self):
        if self._file is None:
            self._file = open(self.output_path, "w", encoding="utf-8")

    def write_items(self, items) -> int:
        self._open()
        for rec in items:
            self._file.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.records_written += 1
            pt = _preview(rec)
            cls = (pt or {}).get("frame_class")
            if cls in self.counts:
                self.counts[cls] += 1
        return self.records_written

    def finish(self, eos: bool) -> dict:
        """关闭文件并写完成标记；返回标记内容。"""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
        marker = {
            "state": "completed" if eos else "interrupted",
            "eos": eos,
            "output": str(self.output_path),
            "records_written": self.records_written,
            "counts": dict(self.counts),
            "closed": self._file is None,
            "wall_time": datetime.now().astimezone().isoformat(
                timespec="milliseconds"),
        }
        with open(self.done_marker, "w", encoding="utf-8") as fh:
            json.dump(marker, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        return marker


def _describe() -> str:
    return "\n".join([
        "node_id: hololens_saver",
        "inputs: converted <- hololens_target_converter/converted",
        "env: RML63_HL_OUTPUT（输出 JSONL）",
        "     RML63_HL_DONE_MARKER（完成标记 JSON）",
        "收到 EOS 后关闭文件并写完成标记，然后退出。",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description="hololens_saver_dora_node")
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.describe:
        print(_describe())
        return 0
    if args.check:
        print("hololens_saver check: SaverCore 可导入（write_items/finish）")
        return 0

    try:
        from dora import Node  # noqa: PLC0415 惰性导入
    except ImportError:
        print("错误: dora-rs 未安装，无法作为 Dora 节点运行。"
              "离线验证请使用 --describe / --check。", file=sys.stderr)
        return 1

    core = SaverCore(os.environ["RML63_HL_OUTPUT"],
                     os.environ["RML63_HL_DONE_MARKER"])
    node = Node()
    eos = False
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
        msg = json.loads(raw)
        if isinstance(msg, dict) and msg.get("_eos"):
            eos = True
            break
        core.write_items(msg)
    marker = core.finish(eos=eos)
    print(f"[hololens_saver] {marker['state']} "
          f"records={marker['records_written']} counts={marker['counts']}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
