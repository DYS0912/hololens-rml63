#!/usr/bin/env python3
"""离线手部轨迹 → 未标定虚拟目标预览转换器（仅平移映射，不接硬件、不控制）。

输入: scripts/hololens_receiver.py 生成的 packets.jsonl（协议 rml63.hand_pose.v1）。
输出: 每行 = 原始记录原样保留 + 独立字段 preview_target 写入转换结果。
不修改输入文件，不监听 UDP，不发送任何控制指令，不启动 ROS2，不接 Dora。

预览语义（务必明确）:
  - preview_target 是"未标定的虚拟位置预览"：坐标轴沿用 Unity tracking space，
    不是 RML63 基坐标，不是 LeRobot action；
  - 虚拟目标初始位置 = [0,0,0]，这是明确设置的预览原点，不是零填充缺失数据；
  - 仅平移映射：target_rotation 恒为 null，注明"旋转映射未实现"，不伪造
    单位四元数。

算法:
  - 仅当 valid 与 control_enabled 都严格为布尔 true，且 position 为 3 个有限
    数值时，才允许更新目标；control_enabled 缺失时默认不启用；
  - 每次开始捏合: hand_anchor=当前手位置，target_anchor=上一有效虚拟目标，
    该帧输出 target_anchor（不产生跳变）；
  - 持续捏合: target_position = target_anchor + scale × (position - hand_anchor)；
  - 松开 / 姿态无效 / 字段错误: target_valid=false, target_position=null，
    写明原因；内部保留上一虚拟目标供下次重新锚定，不把旧目标冒充有效输出；
  - session_id 改变时重新开始一段，虚拟目标回到预览原点；
  - 同一会话内重复或倒退的 sequence 不更新目标，并中断当前锚定；
  - 锚定期间相邻接收时间间隔超过 --interval-threshold（默认 0.2 秒）时中断
    锚定并重新锚定（无跳变）。时间来源优先 _receiver.monotonic_s，其次
    monotonic_seconds，再次 _receiver.wall_time_ms/1000。这是离线记录间隔
    检查，不是实时断网看门狗。

用法（项目根目录，仅 Python 标准库）:
    python3 scripts/hololens_target_converter.py \
        --input datasets/hololens_checks/<run_id>/packets.jsonl \
        --output datasets/hololens_checks/<run_id>/target_packets.jsonl \
        [--scale 0.2] [--interval-threshold 0.2]
    默认拒绝覆盖已存在的输出文件（--overwrite 显式允许）；
    输入与输出指向同一文件时始终拒绝。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

COORDINATE_NOTE = ("未标定的虚拟位置预览：坐标轴沿用 Unity tracking space，"
                   "不是 RML63 基坐标，也不是 LeRobot action")
ROTATION_NOTE = "旋转映射未实现（target_rotation 恒为 null，不伪造单位四元数）"
DEFAULT_SCALE = 0.2
DEFAULT_INTERVAL_THRESHOLD_S = 0.2
PREVIEW_ORIGIN = [0.0, 0.0, 0.0]


class _NonFiniteJsonConstant(ValueError):
    """JSON 文本中出现 NaN/Infinity 等非有限常量。"""


def _reject_constant(value: str):
    raise _NonFiniteJsonConstant(value)


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_finite_number(x) -> bool:
    return _is_number(x) and math.isfinite(x)


def _is_finite_vector(x, dim: int) -> bool:
    return (isinstance(x, list) and len(x) == dim
            and all(_is_finite_number(e) for e in x))


class TargetConverter:
    """逐条记录做捏合 → 虚拟目标预览转换；状态按 session_id 隔离。"""

    def __init__(self, scale: float = DEFAULT_SCALE,
                 interval_threshold_s: float = DEFAULT_INTERVAL_THRESHOLD_S):
        if not _is_finite_number(scale) or scale <= 0:
            raise ValueError("scale 必须为有限正数")
        if not _is_finite_number(interval_threshold_s) \
                or interval_threshold_s <= 0:
            raise ValueError("interval_threshold_s 必须为有限正数")
        self.scale = float(scale)
        self.interval_threshold_s = float(interval_threshold_s)

        self._session = None
        self._last_seq = None
        self._pinching = False
        self._anchor_hand = None        # 捏合开始时的手位置
        self._anchor_target = None      # 捏合开始时的虚拟目标
        self._last_virtual_target = list(PREVIEW_ORIGIN)
        self._prev_time = None          # 上一记录时间（仅当前会话内）
        self._session_anchored = False  # 当前会话是否曾锚定过

        self.stats = {
            "tool": "rml63_hololens_target_converter",
            "scale": self.scale,
            "interval_threshold_s": self.interval_threshold_s,
            "interval_time_source": ("优先 _receiver.monotonic_s，其次 "
                                     "monotonic_seconds，再次 "
                                     "_receiver.wall_time_ms/1000"),
            "note": ("离线记录间隔检查，不是实时断网看门狗；preview_target 为"
                     "未标定的虚拟位置预览，坐标轴沿用 Unity tracking space，"
                     "不是 RML63 基坐标，不是 LeRobot action；仅平移映射，"
                     "旋转映射未实现（target_rotation 恒为 null）。"),
            "anchor_note": ("每次捏合开始：hand_anchor=当前手位置，"
                            "target_anchor=上一有效虚拟目标，该帧输出 "
                            "target_anchor（无跳变）。重新锚定次数=各会话内"
                            "除首次外的捏合开始次数（含松开、序号异常、间隔"
                            "过大、时间戳缺失后的重新锚定）。"),
            "total_records": 0,
            "blank_lines_skipped": 0,
            "output_records": 0,
            "valid_output": 0,
            "paused": 0,
            "abnormal": 0,
            "reanchor_count": 0,
            "session_count": 0,
            "sessions": {},
        }

    # ---- 对外接口 ----

    def convert_line(self, raw: str):
        """处理一行 JSONL。返回输出记录 dict；空白行返回 None。"""
        if not raw.strip():
            return None
        try:
            record = json.loads(raw, parse_constant=_reject_constant)
        except _NonFiniteJsonConstant:
            return self._fallback(raw, "行含非有限 JSON 常量（NaN/Infinity）")
        except (ValueError, UnicodeDecodeError):
            return self._fallback(raw, "行无法解析为 JSON")
        if not isinstance(record, dict):
            return self._fallback(raw, "记录不是 JSON 对象")
        return self.convert_record(record)

    def convert_record(self, record: dict) -> dict:
        """处理一条已解析的记录。返回 原始记录副本 + preview_target。"""
        out = dict(record)  # 保留原始记录，不修改原对象
        self.stats["total_records"] += 1
        pt = self._compute_preview(record)
        self.stats[pt["frame_class"]] += 1
        sid = record.get("session_id")
        if isinstance(sid, str) and sid != "":
            session = self.stats["sessions"][sid]
            session["records"] += 1
            if pt["target_valid"]:
                session["valid_outputs"] += 1
        key = "preview_target"
        while key in out:
            key = "_" + key  # 原始记录若已有该键，绝不覆盖
        out[key] = pt
        return out

    def _fallback(self, raw: str, reason: str) -> dict:
        self.stats["total_records"] += 1
        self.stats["abnormal"] += 1
        self._break_anchor()
        # 不推进 prev_time：垃圾行不参与间隔校验，下一有效记录会看到完整间隔
        pt = self._new_preview()
        pt["reason"] = reason
        return {"_raw_input_line": raw, "preview_target": pt}

    def begin_new_pinch(self):
        """实时路径专用：新捏合从预览原点起算。

        清空保留的虚拟目标并标记本会话未锚定，使下一帧的捏合开始以
        [0,0,0] 为初始虚拟目标（首目标不携带上一次捏合的累计虚拟位移）。
        仅在被显式调用时生效，不改变离线转换的既有语义。
        """
        self._last_virtual_target = list(PREVIEW_ORIGIN)
        self._session_anchored = False

    # ---- 核心状态机 ----

    def _new_preview(self) -> dict:
        return {
            "target_valid": False,
            "target_position": None,
            "target_rotation": None,
            "reason": None,
            "frame_class": "abnormal",
            "rotation_note": ROTATION_NOTE,
            "coordinate_note": COORDINATE_NOTE,
        }

    def _compute_preview(self, record: dict) -> dict:
        pt = self._new_preview()
        cur_time = self._frame_time(record)

        sid = record.get("session_id")
        if not (isinstance(sid, str) and sid != ""):
            pt["reason"] = "session_id 缺失或类型错误（需非空字符串）"
            self._break_anchor()
            self._prev_time = cur_time
            return pt
        if sid != self._session:
            # 会话切换：重新开始一段，虚拟目标回到预览原点
            self._session = sid
            self._last_seq = None
            self._pinching = False
            self._anchor_hand = None
            self._anchor_target = None
            self._last_virtual_target = list(PREVIEW_ORIGIN)
            self._prev_time = None
            self._session_anchored = False
            self.stats["sessions"][sid] = {
                "records": 0, "valid_outputs": 0, "reanchors": 0,
                "first_sequence": None, "last_sequence": None,
            }
            self.stats["session_count"] += 1

        seq = record.get("sequence")
        if not (isinstance(seq, int) and not isinstance(seq, bool)
                and seq >= 0):
            pt["reason"] = "sequence 缺失或类型错误（需非负整数）"
            self._break_anchor()
            self._prev_time = cur_time
            return pt
        session = self.stats["sessions"][sid]
        if session["first_sequence"] is None:
            session["first_sequence"] = seq
        session["last_sequence"] = seq
        if self._last_seq is not None and seq <= self._last_seq:
            pt["reason"] = ("重复序号：不更新目标并中断锚定"
                            if seq == self._last_seq
                            else "倒退序号：不更新目标并中断锚定")
            self._break_anchor()
            self._prev_time = cur_time
            return pt
        self._last_seq = seq

        valid = record.get("valid")
        if valid is not True:
            if valid is False:
                pt["reason"] = "姿态不可用（valid=false），不更新目标"
                pt["frame_class"] = "paused"
            else:
                pt["reason"] = "valid 字段缺失或类型错误（需严格布尔 true）"
            self._break_anchor()
            self._prev_time = cur_time
            return pt

        enabled = record.get("control_enabled")
        if enabled is not True:
            if enabled is False:
                pt["reason"] = "松开捏合（control_enabled=false），不更新目标"
                pt["frame_class"] = "paused"
            elif enabled is None:
                pt["reason"] = "control_enabled 缺失（默认不启用），不更新目标"
            else:
                pt["reason"] = ("control_enabled 类型错误（需严格布尔 true "
                                "才启用），不更新目标")
            self._break_anchor()
            self._prev_time = cur_time
            return pt

        pos = record.get("position")
        if not _is_finite_vector(pos, 3):
            pt["reason"] = "position 缺失、维度错误或含非有限数值"
            self._break_anchor()
            self._prev_time = cur_time
            return pt

        # 锚定期间的接收间隔校验（离线记录间隔检查，不是实时断网看门狗）
        if self._pinching and self._prev_time is not None:
            if cur_time is None:
                self._start_anchor(pos, pt, reason=(
                    "时间戳缺失，无法校验接收间隔，中断锚定并重新锚定（无跳变）"))
                self._prev_time = cur_time
                return pt
            delta = cur_time - self._prev_time
            if delta < 0:
                self._start_anchor(pos, pt, reason=(
                    f"接收时间倒退 {delta:.3f}s，中断锚定并重新锚定（无跳变）"))
                self._prev_time = cur_time
                return pt
            if delta > self.interval_threshold_s:
                self._start_anchor(pos, pt, reason=(
                    f"接收间隔 {delta:.3f}s 超过阈值 "
                    f"{self.interval_threshold_s}s，"
                    "中断锚定并重新锚定（无跳变）"))
                self._prev_time = cur_time
                return pt

        if self._pinching:
            target = [self._anchor_target[i]
                      + self.scale * (pos[i] - self._anchor_hand[i])
                      for i in range(3)]
            self._last_virtual_target = target
            pt.update(target_valid=True, target_position=list(target),
                      frame_class="valid_output",
                      reason=(f"捏合持续映射：target_position = "
                              f"target_anchor + {self.scale}×"
                              f"(position - hand_anchor)，仅平移"))
        else:
            self._start_anchor(pos, pt)
        self._prev_time = cur_time
        return pt

    def _start_anchor(self, pos, pt: dict, reason=None):
        first = not self._session_anchored
        self._anchor_hand = [float(v) for v in pos]
        self._anchor_target = list(self._last_virtual_target)
        self._pinching = True
        self._session_anchored = True
        if not first:
            self.stats["reanchor_count"] += 1
            session = self.stats["sessions"].get(self._session)
            if session is not None:
                session["reanchors"] += 1
        if reason is None:
            reason = ("捏合开始：以预览原点 [0,0,0] 为初始虚拟目标（无跳变）"
                      if first else "重新锚定：输出上一有效虚拟目标（无跳变）")
        pt.update(target_valid=True,
                  target_position=list(self._anchor_target),
                  frame_class="valid_output",
                  reason=reason)

    def _break_anchor(self):
        """中断锚定；保留上一虚拟目标供下次重新锚定。"""
        if self._pinching:
            self._pinching = False
            self._anchor_hand = None
            self._anchor_target = None

    @staticmethod
    def _frame_time(record: dict):
        """取接收时间（秒）。优先 _receiver.monotonic_s。"""
        rx = record.get("_receiver")
        if isinstance(rx, dict):
            t = rx.get("monotonic_s")
            if _is_finite_number(t):
                return float(t)
        t = record.get("monotonic_seconds")
        if _is_finite_number(t):
            return float(t)
        if isinstance(rx, dict):
            t = rx.get("wall_time_ms")
            if _is_finite_number(t):
                return float(t) / 1000.0
        return None


def _print_stats(converter: TargetConverter):
    stats = converter.stats
    print(f"[转换完成] 输入 {stats['input']}（{stats['total_records']} 条记录，"
          f"跳过空行 {stats['blank_lines_skipped']}）", flush=True)
    print(f"[转换完成] 输出 {stats['output']}（{stats['output_records']} 行）",
          flush=True)
    print(f"[统计] 有效输出 {stats['valid_output']} | "
          f"暂停帧 {stats['paused']} | 异常帧 {stats['abnormal']} | "
          f"重新锚定 {stats['reanchor_count']} | "
          f"会话 {stats['session_count']}", flush=True)
    print("[统计] 各会话: "
          + json.dumps(stats["sessions"], ensure_ascii=False), flush=True)
    print("[统计JSON]", flush=True)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="离线手部轨迹 → 未标定虚拟目标预览转换（仅平移；"
                    "不接硬件、不控制、不监听 UDP）")
    parser.add_argument("--input", required=True,
                        help="输入 packets.jsonl 路径")
    parser.add_argument("--output", required=True,
                        help="输出 JSONL 路径（默认拒绝覆盖已存在文件）")
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE,
                        help=f"平移缩放系数，须为有限正数（默认 {DEFAULT_SCALE}）")
    parser.add_argument("--interval-threshold", type=float,
                        default=DEFAULT_INTERVAL_THRESHOLD_S,
                        help="锚定期间的接收间隔阈值（秒，默认 0.2）。"
                             "这是离线记录间隔检查，不是实时断网看门狗")
    parser.add_argument("--overwrite", action="store_true",
                        help="允许覆盖已存在的输出文件")
    args = parser.parse_args(argv)
    if not (math.isfinite(args.scale) and args.scale > 0):
        parser.error("--scale 必须为有限正数")
    if not (math.isfinite(args.interval_threshold)
            and args.interval_threshold > 0):
        parser.error("--interval-threshold 必须为有限正数（秒）")

    in_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    if not in_path.is_file():
        print(f"[错误] 输入文件不存在或不是文件: {in_path}", file=sys.stderr)
        return 2
    if in_path == out_path:
        print("[错误] 输入与输出不能指向同一文件", file=sys.stderr)
        return 2
    if out_path.exists() and not args.overwrite:
        print(f"[错误] 输出文件已存在，拒绝覆盖（请另选文件名，"
              f"或用 --overwrite 显式允许）: {out_path}", file=sys.stderr)
        return 2

    converter = TargetConverter(scale=args.scale,
                                interval_threshold_s=args.interval_threshold)
    tmp_path = out_path.with_name(out_path.name + f".tmp{os.getpid()}")
    written = 0
    blank = 0
    try:
        with open(in_path, "r", encoding="utf-8") as fin, \
                open(tmp_path, "w", encoding="utf-8") as fout:
            for line in fin:
                out_record = converter.convert_line(line)
                if out_record is None:
                    blank += 1
                    continue
                fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                written += 1
        os.replace(tmp_path, out_path)
    except OSError as exc:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        print(f"[错误] 读写失败: {exc}", file=sys.stderr)
        return 1

    converter.stats["input"] = str(in_path)
    converter.stats["output"] = str(out_path)
    converter.stats["output_records"] = written
    converter.stats["blank_lines_skipped"] = blank
    _print_stats(converter)
    return 0


if __name__ == "__main__":
    sys.exit(main())
