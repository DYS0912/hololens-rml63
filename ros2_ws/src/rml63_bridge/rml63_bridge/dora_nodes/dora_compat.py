"""Dora 兼容层：dora 与 pyarrow 全部惰性导入。

本模块本身只依赖标准库 + numpy，可被任意环境 import。真正需要 dora /
pyarrow 的只有节点脚本的 ``main()``（运行时）—— 离线测试与模拟器
（simulate_dataflow）不需要它们。
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def dora_available() -> bool:
    """当前 Python 环境是否安装了 dora-rs。"""
    try:
        import dora  # noqa: F401

        return True
    except ImportError:
        return False


def import_pyarrow():
    """惰性导入 pyarrow（dora 运行环境自带；离线环境可能没有）。"""
    import pyarrow as pa

    return pa


def value_to_py(value: Any) -> Any:
    """dora 事件值 -> 纯 Python / numpy 值。

    - pyarrow 数组: 单元素解包为标量，多元素转 numpy 数组；
    - numpy 数组: 原样返回；
    - 其余类型: 原样返回。
    """
    if value is None:
        return None
    if hasattr(value, "to_numpy"):
        # dora 管道可能对数组做切片/拼接，允许拷贝（zero_copy_only=False）
        arr = value.to_numpy(zero_copy_only=False)
        if getattr(arr, "size", None) == 1:
            return arr.item()
        return arr
    if hasattr(value, "to_pylist"):
        lst = value.to_pylist()
        if isinstance(lst, list) and len(lst) == 1:
            return lst[0]
        return lst
    return value


def to_output_array(value: Any):
    """纯 Python / numpy 值 -> pyarrow 数组（供 dora ``send_output`` 使用）。

    图像（numpy 数组）以数值数组传输（ravel 后仍是数值，不是 JSON/字符串）；
    None 不允许发送，调用方跳过该通道（缺失数据绝不零填充）。
    """
    pa = import_pyarrow()
    if value is None:
        raise ValueError("None 不能作为 dora 输出；请跳过该通道（缺失数据不零填充）")
    if isinstance(value, np.ndarray):
        return pa.array(np.ascontiguousarray(value).ravel())
    if isinstance(value, (list, tuple)):
        return pa.array([float(x) if isinstance(x, (int, float)) else x for x in value])
    if isinstance(value, bool):
        return pa.array([value])
    if isinstance(value, int):
        return pa.array([value], type=pa.int64())
    if isinstance(value, float):
        return pa.array([value], type=pa.float32())
    if isinstance(value, str):
        return pa.array([value])
    raise TypeError(f"不支持发送的类型: {type(value)!r}")


def send_named_outputs(node, outputs: dict, names: tuple) -> int:
    """按 ``names`` 顺序发送非 None 输出；返回发送的通道数。

    用于节点脚本的 ``main()`` 事件循环；离线测试不经过此函数。
    """
    sent = 0
    for name in names:
        value = outputs.get(name)
        if value is None:
            continue
        node.send_output(name, to_output_array(value))
        sent += 1
    return sent


def describe_node(node_id: str, inputs: dict, outputs: list, env: Optional[dict] = None) -> str:
    """节点设计摘要（供 --describe 干跑输出，不需要 dora）。"""
    lines = [f"node_id: {node_id}"]
    lines.append("inputs:")
    for k, v in (inputs or {}).items():
        lines.append(f"  {k} <- {v}")
    lines.append("outputs: " + ", ".join(outputs or []))
    if env:
        lines.append("env:")
        for k, v in env.items():
            lines.append(f"  {k} = {v}")
    return "\n".join(lines)
