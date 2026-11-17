"""离线模拟 action 源（mock_action_dora_node）—— 仅数据格式验证。

输出:
    action   float32[6]（确定性正弦序列）

!! 重要 !!
    模拟 action 仅用于数据格式验证，不代表真实示范动作，不用于研究结论。
    不得使用模拟 action 数据训练 ACT。
    接真机后本节点必须被真实 action 源替换（拖拽示教 / 脚本 / 绑扎枪遥操作）。

ACTION_MODE=mock: 每 tick 输出 6 维确定性 action；
ACTION_MODE=none: 不输出任何 action（用于验证 "action 缺失时拒绝生成训练数据"）。
"""

from __future__ import annotations

import math
import os
import sys
from typing import Optional

# 注意: dora 以普通脚本方式运行本文件（python3 <path>），必须使用绝对导入；
# dataflow env 中的 PYTHONPATH 指向 rml63_bridge 包根目录。
from rml63_bridge.convert_to_lerobot import MOCK_ACTION_NOTICE
from rml63_bridge.dora_nodes.dora_compat import dora_available, describe_node, send_named_outputs

ACTION_OUTPUTS = ("action",)
DEFAULT_ACTION_DIM = 6


def make_mock_action(index: int, dim: int = DEFAULT_ACTION_DIM) -> list:
    """确定性 6 维模拟 action（仅数据格式验证，不用于训练）。"""
    return [0.01 * j + 0.005 * math.sin(0.3 * index) for j in range(dim)]


class MockActionCore:
    """模拟 action 源核心（无 dora 依赖）。"""

    def __init__(self, mode: str = "mock", dim: int = DEFAULT_ACTION_DIM):
        if mode not in ("mock", "none"):
            raise ValueError(f"mode 仅支持 mock/none，收到 {mode!r}")
        self.mode = mode
        self.dim = dim
        self._index = 0
        self.actions_produced = 0

    def on_tick(self) -> Optional[list]:
        """mock 模式输出下一 action；none 模式输出 None（表示缺失）。"""
        if self.mode == "none":
            return None
        action = make_mock_action(self._index, self.dim)
        self._index += 1
        self.actions_produced += 1
        return action


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="mock_action_dora_node (离线专用)")
    parser.add_argument("--describe", action="store_true", help="打印节点设计并退出（不需要 dora）")
    parser.add_argument("--check", action="store_true", help="离线自检并退出（不需要 dora）")
    args = parser.parse_args()

    if args.describe:
        print(describe_node("mock_action", {"tick": "dora/timer/millis/33"}, list(ACTION_OUTPUTS)))
        print(f"注意: {MOCK_ACTION_NOTICE}")
        return 0
    if args.check:
        core = MockActionCore(mode="mock")
        print(f"mock_action check: action={core.on_tick()}")
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

    node = Node()
    core = MockActionCore(
        mode=os.environ.get("ACTION_MODE", "mock"),
        dim=int(os.environ.get("ACTION_DIM", str(DEFAULT_ACTION_DIM))),
    )
    print(f"[mock_action] mode={core.mode}（{MOCK_ACTION_NOTICE}）", flush=True)

    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT":
            continue
        if event["id"] == "tick":
            action = core.on_tick()
            if action is not None:
                send_named_outputs(node, {"action": action}, ACTION_OUTPUTS)

    print(f"[mock_action] stopped (actions={core.actions_produced})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
