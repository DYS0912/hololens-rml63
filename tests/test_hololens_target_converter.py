#!/usr/bin/env python3
"""hololens_target_converter.py 离线测试（仅标准库 unittest）。

覆盖: 起点（预览原点）、缩放、松开、重新捏合不跳变、缺失/非有限位置、
缺失启用字段、会话切换、重复/乱序序号、接收间隔过大、原始文件不变、
CLI 拒绝覆盖/同一文件、--scale 校验、不可解析行。

运行（项目根目录）:
    python3 tests/test_hololens_target_converter.py
    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "hololens_target_converter",
    PROJECT_ROOT / "scripts" / "hololens_target_converter.py")
tc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tc)


def make_record(seq=0, session="s1", valid=True, control=True,
                pos=(0.1, 0.2, 0.3), mono=1.0, **over) -> dict:
    record = {
        "schema": "rml63.hand_pose.v1",
        "session_id": session,
        "sequence": seq,
        "timestamp_unix_ms": 1700000000000,
        "monotonic_seconds": mono,
        "tracked": True,
        "valid": valid,
        "control_enabled": control,
        "position": list(pos),
        "rotation": [0.0, 0.0, 0.0, 1.0],
        "invalid_reason": None,
        "_receiver": {"wall_time_ms": mono * 1000.0,
                      "monotonic_s": mono,
                      "from_ip": "172.20.10.3",
                      "from_port": 6060},
    }
    record.update(over)
    return record


def _pt(conv, record):
    return conv.convert_record(record)["preview_target"]


class PinchMappingTests(unittest.TestCase):
    """起点、缩放、松开、重新捏合不跳变。"""

    def setUp(self):
        self.conv = tc.TargetConverter()

    def test_first_pinch_outputs_preview_origin(self):
        pt = _pt(self.conv, make_record(seq=0, pos=(0.5, 0.5, 0.5)))
        self.assertTrue(pt["target_valid"])
        self.assertEqual(pt["target_position"], [0.0, 0.0, 0.0])
        self.assertEqual(pt["frame_class"], "valid_output")
        self.assertIsNone(pt["target_rotation"])  # 不伪造单位四元数
        self.assertIn("旋转映射未实现", pt["rotation_note"])
        self.assertIn("Unity tracking space", pt["coordinate_note"])
        self.assertIn("捏合开始", pt["reason"])

    def test_continued_pinch_applies_default_scale(self):
        self.conv.convert_record(make_record(seq=0, pos=(1, 1, 1)))
        pt1 = _pt(self.conv, make_record(seq=1, pos=(2, 1, 1)))
        for got, want in zip(pt1["target_position"], [0.2, 0.0, 0.0]):
            self.assertAlmostEqual(got, want, places=12)
        self.assertIn("捏合持续映射", pt1["reason"])
        pt2 = _pt(self.conv, make_record(seq=2, pos=(2, 3, 1)))
        # 锚点仍是捏合起点的 (1,1,1)：0.2×((2,3,1)-(1,1,1)) = [0.2, 0.4, 0]
        for got, want in zip(pt2["target_position"], [0.2, 0.4, 0.0]):
            self.assertAlmostEqual(got, want, places=12)

    def test_custom_scale(self):
        conv = tc.TargetConverter(scale=0.5)
        conv.convert_record(make_record(seq=0, pos=(1, 1, 1)))
        pt = _pt(conv, make_record(seq=1, pos=(3, 1, 1)))
        for got, want in zip(pt["target_position"], [1.0, 0.0, 0.0]):
            self.assertAlmostEqual(got, want, places=12)

    def test_release_then_repinch_no_jump(self):
        self.conv.convert_record(make_record(seq=0, pos=(1, 1, 1)))
        pt1 = _pt(self.conv, make_record(seq=1, pos=(2, 1, 1)))
        for got, want in zip(pt1["target_position"], [0.2, 0.0, 0.0]):
            self.assertAlmostEqual(got, want, places=12)
        # 松开
        pt_rel = _pt(self.conv, make_record(seq=2, control=False,
                                            pos=(3, 3, 3)))
        self.assertFalse(pt_rel["target_valid"])
        self.assertIsNone(pt_rel["target_position"])
        self.assertEqual(pt_rel["frame_class"], "paused")
        self.assertIn("松开", pt_rel["reason"])
        # 重新捏合：手已移到远处，目标输出上一有效目标，无跳变
        pt3 = _pt(self.conv, make_record(seq=3, pos=(5, 5, 5)))
        self.assertTrue(pt3["target_valid"])
        for got, want in zip(pt3["target_position"], [0.2, 0.0, 0.0]):
            self.assertAlmostEqual(got, want, places=12)
        self.assertIn("重新锚定", pt3["reason"])
        self.assertEqual(self.conv.stats["reanchor_count"], 1)
        # 新锚点下继续映射
        pt4 = _pt(self.conv, make_record(seq=4, pos=(6, 5, 5)))
        for got, want in zip(pt4["target_position"], [0.4, 0.0, 0.0]):
            self.assertAlmostEqual(got, want, places=12)


class FieldErrorTests(unittest.TestCase):
    """缺失/非有限位置、缺失启用字段、valid/control 类型错误。"""

    def setUp(self):
        self.conv = tc.TargetConverter()

    def test_missing_position_breaks_anchor(self):
        self.conv.convert_record(make_record(seq=0, pos=(1, 1, 1)))
        self.conv.convert_record(make_record(seq=1, pos=(2, 1, 1)))
        bad = make_record(seq=2, pos=(3, 1, 1))
        bad.pop("position")
        pt = _pt(self.conv, bad)
        self.assertFalse(pt["target_valid"])
        self.assertIsNone(pt["target_position"])
        self.assertEqual(pt["frame_class"], "abnormal")
        self.assertIn("position", pt["reason"])
        # 锚定已中断：下一帧重新锚定，输出上一目标（无跳变）
        pt3 = _pt(self.conv, make_record(seq=3, pos=(3, 1, 1)))
        for got, want in zip(pt3["target_position"], [0.2, 0.0, 0.0]):
            self.assertAlmostEqual(got, want, places=12)

    def test_nonfinite_and_wrong_dimension_position(self):
        for bad_pos in ([0.0, float("nan"), 0.0],
                        [0.0, 0.0, float("inf")],
                        [1.0, 2.0],          # 2 维
                        "nope"):
            pt = _pt(self.conv, make_record(seq=0, pos=bad_pos))
            self.assertFalse(pt["target_valid"], bad_pos)
            self.assertIsNone(pt["target_position"], bad_pos)
            self.assertEqual(pt["frame_class"], "abnormal", bad_pos)

    def test_missing_control_enabled_defaults_disabled(self):
        bad = make_record(seq=0, control=True)
        bad.pop("control_enabled")
        pt = _pt(self.conv, bad)
        self.assertFalse(pt["target_valid"])
        self.assertIsNone(pt["target_position"])
        self.assertEqual(pt["frame_class"], "abnormal")
        self.assertIn("缺失", pt["reason"])
        # 之后正常捏合帧从预览原点开始
        pt1 = _pt(self.conv, make_record(seq=1, control=True,
                                         pos=(2, 2, 2)))
        self.assertEqual(pt1["target_position"], [0.0, 0.0, 0.0])

    def test_control_enabled_wrong_type(self):
        pt = _pt(self.conv, make_record(seq=0, control=1))
        self.assertFalse(pt["target_valid"])
        self.assertIn("类型错误", pt["reason"])

    def test_valid_wrong_type_rejected(self):
        for bad in ("true", 1, 0.0):
            pt = _pt(self.conv, make_record(seq=0, valid=bad))
            self.assertFalse(pt["target_valid"], bad)
            self.assertEqual(pt["frame_class"], "abnormal", bad)

    def test_valid_false_is_paused(self):
        pt = _pt(self.conv, make_record(seq=0, valid=False, control=False,
                                        pos=[]))
        self.assertFalse(pt["target_valid"])
        self.assertIsNone(pt["target_position"])
        self.assertEqual(pt["frame_class"], "paused")
        self.assertIn("姿态不可用", pt["reason"])


class SessionAndSequenceTests(unittest.TestCase):
    """会话切换回原点、重复/乱序/前进缺口序号。"""

    def setUp(self):
        self.conv = tc.TargetConverter()

    def test_session_switch_resets_target_to_origin(self):
        self.conv.convert_record(make_record(seq=0, session="A",
                                             pos=(1, 1, 1)))
        pt1 = _pt(self.conv, make_record(seq=1, session="A",
                                         pos=(2, 1, 1)))
        for got, want in zip(pt1["target_position"], [0.2, 0.0, 0.0]):
            self.assertAlmostEqual(got, want, places=12)
        # 新会话：即使上会话目标为 [0.2,0,0]，也回到预览原点
        pt_b = _pt(self.conv, make_record(seq=0, session="B",
                                          pos=(9, 9, 9), mono=100.0))
        self.assertTrue(pt_b["target_valid"])
        self.assertEqual(pt_b["target_position"], [0.0, 0.0, 0.0])
        self.assertEqual(self.conv.stats["session_count"], 2)
        self.assertEqual(set(self.conv.stats["sessions"]), {"A", "B"})

    def test_duplicate_sequence_no_update_breaks_anchor(self):
        self.conv.convert_record(make_record(seq=0, pos=(1, 1, 1)))
        self.conv.convert_record(make_record(seq=1, pos=(2, 1, 1)))
        pt = _pt(self.conv, make_record(seq=1, pos=(2, 1, 1)))  # 重复
        self.assertFalse(pt["target_valid"])
        self.assertIsNone(pt["target_position"])
        self.assertEqual(pt["frame_class"], "abnormal")
        self.assertIn("重复序号", pt["reason"])
        # 锚定已中断：下一帧重新锚定（无跳变）
        pt2 = _pt(self.conv, make_record(seq=2, pos=(3, 1, 1)))
        for got, want in zip(pt2["target_position"], [0.2, 0.0, 0.0]):
            self.assertAlmostEqual(got, want, places=12)
        self.assertEqual(self.conv.stats["reanchor_count"], 1)

    def test_backward_sequence_no_update_breaks_anchor(self):
        self.conv.convert_record(make_record(seq=0, pos=(1, 1, 1)))
        self.conv.convert_record(make_record(seq=1, pos=(2, 1, 1)))
        self.conv.convert_record(make_record(seq=5, pos=(2, 1, 1)))
        pt = _pt(self.conv, make_record(seq=4, pos=(2, 1, 1)))  # 倒退
        self.assertFalse(pt["target_valid"])
        self.assertIn("倒退序号", pt["reason"])
        # 下一帧重新锚定
        pt2 = _pt(self.conv, make_record(seq=6, pos=(3, 1, 1)))
        self.assertTrue(pt2["target_valid"])
        self.assertIn("重新锚定", pt2["reason"])

    def test_forward_sequence_gap_allowed_if_interval_ok(self):
        self.conv.convert_record(make_record(seq=0, pos=(1, 1, 1),
                                             mono=1.00))
        self.conv.convert_record(make_record(seq=1, pos=(2, 1, 1),
                                             mono=1.05))
        pt = _pt(self.conv, make_record(seq=3, pos=(2, 2, 1), mono=1.10))
        self.assertTrue(pt["target_valid"])
        self.assertIn("捏合持续映射", pt["reason"])
        for got, want in zip(pt["target_position"], [0.2, 0.2, 0.0]):
            self.assertAlmostEqual(got, want, places=12)


class IntervalTests(unittest.TestCase):
    """接收间隔过大 / 时间来源优先级（离线记录间隔检查）。"""

    def test_interval_too_large_reanchors_no_jump(self):
        conv = tc.TargetConverter(interval_threshold_s=0.2)
        conv.convert_record(make_record(seq=0, pos=(1, 1, 1), mono=1.00))
        pt1 = _pt(conv, make_record(seq=1, pos=(2, 1, 1), mono=1.05))
        for got, want in zip(pt1["target_position"], [0.2, 0.0, 0.0]):
            self.assertAlmostEqual(got, want, places=12)
        # 间隔 0.55s > 0.2s：中断锚定并重新锚定，输出上一目标（无跳变）
        pt2 = _pt(conv, make_record(seq=2, pos=(3, 1, 1), mono=1.60))
        self.assertTrue(pt2["target_valid"])
        for got, want in zip(pt2["target_position"], [0.2, 0.0, 0.0]):
            self.assertAlmostEqual(got, want, places=12)
        self.assertIn("间隔", pt2["reason"])
        self.assertEqual(conv.stats["reanchor_count"], 1)
        # 新锚点下继续映射
        pt3 = _pt(conv, make_record(seq=3, pos=(4, 1, 1), mono=1.65))
        for got, want in zip(pt3["target_position"], [0.4, 0.0, 0.0]):
            self.assertAlmostEqual(got, want, places=12)

    def test_interval_prefers_receiver_monotonic_s(self):
        conv = tc.TargetConverter(interval_threshold_s=0.2)
        conv.convert_record(make_record(seq=0, pos=(1, 1, 1), mono=1.00))
        # _receiver.monotonic_s=1.60 与 monotonic_seconds=1.05 冲突：
        # 必须优先用 _receiver.monotonic_s（1.60 → 间隔 0.60 > 0.2 触发）
        rec = make_record(seq=1, pos=(2, 1, 1), mono=1.05)
        rec["_receiver"]["monotonic_s"] = 1.60
        pt = _pt(conv, rec)
        self.assertIn("间隔", pt["reason"])
        self.assertEqual(pt["target_position"], [0.0, 0.0, 0.0])

    def test_interval_fallback_to_monotonic_seconds(self):
        conv = tc.TargetConverter(interval_threshold_s=0.2)
        conv.convert_record(make_record(seq=0, pos=(1, 1, 1), mono=1.00))
        rec = make_record(seq=1, pos=(2, 1, 1), mono=1.50)
        rec.pop("_receiver")  # 无接收端时间，回退到 monotonic_seconds
        pt = _pt(conv, rec)
        self.assertIn("间隔", pt["reason"])

    def test_interval_not_checked_when_released(self):
        conv = tc.TargetConverter(interval_threshold_s=0.2)
        conv.convert_record(make_record(seq=0, control=False, mono=1.0))
        pt = _pt(conv, make_record(seq=1, control=True, pos=(2, 2, 2),
                                   mono=100.0))
        self.assertTrue(pt["target_valid"])
        self.assertIn("捏合开始", pt["reason"])  # 正常锚定，不提间隔
        self.assertNotIn("间隔", pt["reason"])

    def test_interval_check_skipped_for_first_session_frame(self):
        conv = tc.TargetConverter(interval_threshold_s=0.2)
        conv.convert_record(make_record(seq=0, session="A", mono=1.0))
        pt = _pt(conv, make_record(seq=0, session="B", pos=(2, 2, 2),
                                   mono=500.0))
        self.assertTrue(pt["target_valid"])
        self.assertEqual(pt["target_position"], [0.0, 0.0, 0.0])


class StatsAndPreservationTests(unittest.TestCase):
    """统计计数、原始记录保留、键冲突不覆盖。"""

    def test_stats_counters(self):
        conv = tc.TargetConverter()
        conv.convert_record(make_record(seq=0, pos=(1, 1, 1)))      # 锚定
        conv.convert_record(make_record(seq=1, pos=(2, 1, 1)))      # 持续
        conv.convert_record(make_record(seq=2, control=False))      # 松开
        conv.convert_record(make_record(seq=2, control=True))       # 重复序号
        conv.convert_record(make_record(seq=3, pos=(3, 1, 1)))      # 重新锚定
        stats = conv.stats
        self.assertEqual(stats["total_records"], 5)
        self.assertEqual(stats["valid_output"], 3)
        self.assertEqual(stats["paused"], 1)
        self.assertEqual(stats["abnormal"], 1)
        self.assertEqual(stats["reanchor_count"], 1)
        self.assertEqual(stats["session_count"], 1)
        sess = stats["sessions"]["s1"]
        self.assertEqual(sess["records"], 5)
        self.assertEqual(sess["valid_outputs"], 3)
        self.assertEqual(sess["reanchors"], 1)
        self.assertEqual(sess["first_sequence"], 0)
        self.assertEqual(sess["last_sequence"], 3)

    def test_original_record_preserved_not_mutated(self):
        rec = make_record(seq=0, pos=(1, 2, 3))
        snapshot = dict(rec)
        conv = tc.TargetConverter()
        out = conv.convert_record(rec)
        self.assertNotIn("preview_target", rec)  # 原对象未被修改
        for key, value in snapshot.items():
            self.assertEqual(out[key], value)
        self.assertIn("preview_target", out)

    def test_preview_target_key_collision_not_overwritten(self):
        rec = make_record(seq=0)
        rec["preview_target"] = {"keep": True}
        out = tc.TargetConverter().convert_record(rec)
        self.assertEqual(out["preview_target"], {"keep": True})
        self.assertIn("_preview_target", out)
        self.assertIn("target_valid", out["_preview_target"])


class CliTests(unittest.TestCase):
    """main() 端到端：原始文件不变、拒绝覆盖、同一文件拒绝、参数校验。"""

    def _write_input(self, tmp: Path) -> Path:
        path = tmp / "in.jsonl"
        lines = [
            json.dumps(make_record(seq=0, pos=(1, 1, 1), mono=1.00)),
            json.dumps(make_record(seq=1, pos=(2, 1, 1), mono=1.05)),
            json.dumps(make_record(seq=2, control=False, mono=1.10)),
            "",  # 空行应被跳过
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_main_end_to_end_and_input_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            inp = self._write_input(tmp)
            before = inp.read_bytes()
            out = tmp / "out.jsonl"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = tc.main(["--input", str(inp), "--output", str(out)])
            self.assertEqual(rc, 0)
            self.assertEqual(inp.read_bytes(), before)  # 原始文件不变
            lines = out.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 3)
            for line in lines:
                self.assertIn("preview_target", json.loads(line))
            text = buf.getvalue()
            self.assertIn("有效输出", text)
            self.assertIn("暂停帧", text)
            self.assertIn("异常帧", text)
            self.assertIn("重新锚定", text)

    def test_main_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            inp = self._write_input(tmp)
            out = tmp / "out.jsonl"
            out.write_text("keep", encoding="utf-8")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = tc.main(["--input", str(inp), "--output", str(out)])
            self.assertEqual(rc, 2)
            self.assertEqual(out.read_text(encoding="utf-8"), "keep")

    def test_main_overwrite_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            inp = self._write_input(tmp)
            out = tmp / "out.jsonl"
            out.write_text("old", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                rc = tc.main(["--input", str(inp), "--output", str(out),
                              "--overwrite"])
            self.assertEqual(rc, 0)
            lines = out.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 3)

    def test_main_refuses_same_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            inp = self._write_input(tmp)
            before = inp.read_bytes()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = tc.main(["--input", str(inp), "--output", str(inp)])
            self.assertEqual(rc, 2)
            self.assertEqual(inp.read_bytes(), before)

    def test_main_scale_validation(self):
        for bad in ("0", "-1", "nan", "inf"):
            with self.assertRaises(SystemExit):
                with contextlib.redirect_stderr(io.StringIO()):
                    tc.main(["--input", "x.jsonl", "--output", "y.jsonl",
                             "--scale", bad])

    def test_unparseable_line_abnormal(self):
        conv = tc.TargetConverter()
        self.assertIsNone(conv.convert_line(""))        # 空白行跳过
        self.assertIsNone(conv.convert_line("   "))
        out = conv.convert_line("not json")
        self.assertEqual(out["_raw_input_line"], "not json")
        self.assertEqual(out["preview_target"]["frame_class"], "abnormal")
        self.assertIn("无法解析", out["preview_target"]["reason"])
        out2 = conv.convert_line("[1, 2]")
        self.assertIn("不是 JSON 对象", out2["preview_target"]["reason"])
        self.assertEqual(conv.stats["abnormal"], 2)

    def test_nonfinite_json_constant_rejected(self):
        conv = tc.TargetConverter()
        raw = json.dumps({"a": float("nan")})
        out = conv.convert_line(raw)
        self.assertEqual(out["preview_target"]["frame_class"], "abnormal")
        self.assertIn("非有限", out["preview_target"]["reason"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
