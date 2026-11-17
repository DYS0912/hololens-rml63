#!/usr/bin/env python3
"""Dora 回放节点：按文件顺序回放已保存的 HoloLens 手部记录（仅离线）。

env:
  RML63_HL_REPLAY_INPUT  输入 packets.jsonl 路径（必填）
  RML63_HL_CHUNK_SIZE    每批记录数（默认 100）

输出通道: records —— 每条 Dora 消息是一个 JSON 数组（一批记录）；发完全部
记录后发送 {"_eos": true, "total": N, "unparseable": M}（回放结束的明确通知）。

说明:
  - 每个记录附加 _replay 诊断字段 {index, wall_time_ms}（回放侧信息），
    不替换原始采集时间戳、session_id、sequence、valid、control_enabled
    与 _receiver 等原始字段；
  - 采用“批消息 + 短间隔”发送：dora 0.5 实测逐条快速发送、以及多批快发
    （100 条/批 × 8 批）都会丢失中间消息，单条大消息（500KB+）实测 100%
    送达；默认 CHUNK_SIZE=10000 使 708 条记录落在一条消息内；
    回放速度不影响转换结果（间隔判定由转换节点使用记录内原始时间，
    _receiver.monotonic_s 优先）；
  - 首个定时 tick 后再发送，等待 runtime 把通道接线完成。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterator, Optional

DEFAULT_CHUNK_SIZE = 10000  # 默认单条大消息（dora 0.5 实测：多批快发会丢失中间批次）
PAUSE_BETWEEN_MESSAGES_S = 0.02   # 批消息间隔（dora 0.5 逐条快发会丢消息）
POST_SEND_FLUSH_S = 0.5           # 发送完后的冲刷余量（不是终止机制）


class ReplayCore:
    """回放核心逻辑（无 dora，可离线测试）。"""

    def __init__(self, input_path, chunk_size: int = DEFAULT_CHUNK_SIZE):
        self.input_path = Path(input_path)
        self.chunk_size = chunk_size
        self.total = 0
        self.unparseable = 0
        self._records = []

    def load(self) -> list:
        """读取全部记录；不可解析的行保留原文放入 _raw_input_line。"""
        records = []
        with open(self.input_path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    records.append({"_raw_input_line": line.rstrip("\n")})
                    self.unparseable += 1
                    continue
                if not isinstance(record, dict):
                    records.append({"_raw_input_line": line.rstrip("\n")})
                    self.unparseable += 1
                    continue
                records.append(record)
        self.total = len(records)
        self._records = records
        return records

    def iter_messages(self) -> Iterator[dict]:
        """按文件顺序分批产出消息；最后一条是 EOS 标记。"""
        records = self._records if self._records else self.load()
        for start in range(0, len(records), self.chunk_size):
            chunk = []
            for i, record in enumerate(records[start:start + self.chunk_size]):
                item = dict(record)
                key = "_replay"
                while key in item:
                    key = "_" + key  # 绝不覆盖发送端原始字段
                item[key] = {
                    "index": start + i,               # 文件顺序下标
                    "wall_time_ms": time.time() * 1000.0,  # 回放侧墙钟
                }
                chunk.append(item)
            yield chunk
        yield {"_eos": True, "total": len(records),
               "unparseable": self.unparseable}


def _describe() -> str:
    return "\n".join([
        "node_id: hololens_replay",
        "inputs: start <- dora/timer/millis/500（等待 runtime 就绪）",
        "outputs: records（JSON 数组批消息 + EOS 标记）",
        "env: RML63_HL_REPLAY_INPUT（输入 packets.jsonl）",
        "     RML63_HL_CHUNK_SIZE（批大小，默认 100）",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description="hololens_replay_dora_node")
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.describe:
        print(_describe())
        return 0
    if args.check:
        core = ReplayCore(os.environ.get(
            "RML63_HL_REPLAY_INPUT", "packets.jsonl"))
        core.load()
        print(f"hololens_replay check: total={core.total} "
              f"unparseable={core.unparseable}")
        return 0

    try:
        from dora import Node  # noqa: PLC0415 惰性导入
        import pyarrow as pa  # noqa: PLC0415
    except ImportError:
        print("错误: dora-rs 未安装，无法作为 Dora 节点运行。"
              "离线验证请使用 --describe / --check。", file=sys.stderr)
        return 1

    core = ReplayCore(
        os.environ["RML63_HL_REPLAY_INPUT"],
        chunk_size=int(os.environ.get("RML63_HL_CHUNK_SIZE",
                                      str(DEFAULT_CHUNK_SIZE))))
    node = Node()
    sent = 0
    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT":
            continue
        # 首个定时 tick 后发送
        for msg in core.iter_messages():
            node.send_output(
                "records",
                pa.array([json.dumps(msg, ensure_ascii=False)]))
            sent += 1
            time.sleep(PAUSE_BETWEEN_MESSAGES_S)
        time.sleep(POST_SEND_FLUSH_S)  # 让 runtime 刷出队列后再退出
        break
    print(f"[hololens_replay] sent={sent} total={core.total} "
          f"unparseable={core.unparseable}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
