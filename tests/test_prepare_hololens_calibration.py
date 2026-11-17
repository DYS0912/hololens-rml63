#!/usr/bin/env python3
"""prepare_hololens_calibration.py 离线测试（仅标准库 unittest）。

覆盖: 选取规则（target_valid/control_enabled/有限三维位置）、样本字段、
按 hand_position 距离去重、--max-samples 上限、统计、CLI 拒绝覆盖/同一
文件/参数校验、输入文件不变、刚体变换估计（Umeyama）虚拟测试、真实记录
选取预检。

注意: 本文件中的“人工点”仅用于验证刚体变换算法（随机旋转/平移/噪声/
反射/退化），不写入任何真实实验数据文件。
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import random
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "prepare_hololens_calibration",
    PROJECT_ROOT / "scripts" / "prepare_hololens_calibration.py")
cal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cal)

REAL_TARGET = (PROJECT_ROOT / "datasets" / "hololens_checks"
               / "run_20261112_101842" / "target_packets.jsonl")


def make_tp_record(seq=0, session="s1", control=True, target_valid=True,
                   target_pos=(0.1, 0.2, 0.3), hand_pos=(0.5, 0.5, 0.5),
                   **over) -> dict:
    """构造 target_packets.jsonl 风格的记录。"""
    record = {
        "schema": "rml63.hand_pose.v1",
        "session_id": session,
        "sequence": seq,
        "timestamp_unix_ms": 1700000000000 + seq,
        "control_enabled": control,
        "valid": True,
        "position": list(hand_pos),
        "coordinate_frame": "unity_tracking_space",
        "_receiver": {"monotonic_s": 1.0 + seq * 0.03},
        "preview_target": {
            "target_valid": target_valid,
            "target_position": list(target_pos) if target_valid else None,
            "target_rotation": None,
            "reason": "x",
            "frame_class": "valid_output" if target_valid else "paused",
            "coordinate_note": "未标定的虚拟位置预览",
        },
    }
    record.update(over)
    return record


class SelectionTests(unittest.TestCase):
    def _run(self, records, **kw):
        kw.setdefault("min_distance", 0.0)  # 先关闭去重看选取本身
        return cal.select_calibration_samples(records, **kw)

    def test_selects_only_valid_enabled_finite(self):
        records = [
            make_tp_record(0, control=True, target_valid=True,
                           target_pos=(0, 0, 0)),          # 选
            make_tp_record(1, control=False, target_valid=True,
                           target_pos=(1, 1, 1)),          # 松开，不选
            make_tp_record(2, control=True, target_valid=False),  # 无效目标
            make_tp_record(3, control=True, target_valid=True,
                           target_pos=[0, float("nan"), 0]),      # 非有限
            make_tp_record(4, control=True, target_valid=True,
                           target_pos=[0, 0]),              # 维度错误
            make_tp_record(5, control=True, target_valid=True,
                           target_pos=(2, 2, 2)),          # 选
        ]
        samples, stats = self._run(records)
        self.assertEqual(stats["input_total_frames"], 6)
        self.assertEqual(stats["valid_target_frames"], 2)
        self.assertEqual(stats["selected_calibration_samples"], 2)
        self.assertEqual([s["sequence"] for s in samples], [0, 5])

    def test_control_enabled_must_be_strict_bool_true(self):
        for bad in (1, "true", None):
            records = [make_tp_record(0, control=True, target_valid=True,
                                      target_pos=(0, 0, 0))]
            records[0]["control_enabled"] = bad
            samples, stats = self._run(records)
            self.assertEqual(stats["valid_target_frames"], 0, bad)
            self.assertEqual(samples, [])

    def test_sample_fields(self):
        records = [make_tp_record(7, session="sess-a", control=True,
                                  target_valid=True,
                                  target_pos=(0.1, 0.2, 0.3),
                                  hand_pos=(0.9, 0.8, 0.7))]
        samples, _ = self._run(records)
        s = samples[0]
        self.assertEqual(s["session_id"], "sess-a")
        self.assertEqual(s["sequence"], 7)
        self.assertEqual(s["timestamp_unix_ms"], 1700000000007)
        self.assertEqual(s["hand_position"], [0.9, 0.8, 0.7])
        self.assertEqual(s["target_position"], [0.1, 0.2, 0.3])
        self.assertIsNone(s["target_rotation"])   # 旋转映射未实现
        self.assertEqual(s["coordinate_frame"], "unity_tracking_space")
        self.assertEqual(s["coordinate_status"],
                         "uncalibrated_unity_tracking_space")
        self.assertIsNone(s["robot_tcp_position"])  # 没有机器人 TCP 数据

    def test_min_distance_dedup_on_hand_position(self):
        # 手位置间距 0.011m、min-distance=0.02：应每隔一个选中（0,2,4,6,8），
        # 相邻选中样本间距 0.022m >= 0.02（避免浮点边界）
        records = []
        for i in range(10):
            records.append(make_tp_record(
                i, control=True, target_valid=True,
                hand_pos=(0.011 * i, 0.0, 0.0),
                target_pos=(0.1 * i, 0.0, 0.0)))
        samples, stats = cal.select_calibration_samples(
            records, min_distance=0.02)
        self.assertEqual(stats["valid_target_frames"], 10)
        self.assertEqual(len(samples), 5)  # 0,2,4,6,8
        self.assertEqual([s["sequence"] for s in samples],
                         [0, 2, 4, 6, 8])
        self.assertEqual(stats["skipped_by_distance"], 5)
        # 相邻选中样本的手位置距离 >= min-distance
        for a, b in zip(samples, samples[1:]):
            self.assertGreaterEqual(cal._euclid(a["hand_position"],
                                                b["hand_position"]), 0.02 - 1e-12)

    def test_min_distance_zero_disables_dedup(self):
        records = [make_tp_record(i, control=True, target_valid=True,
                                  hand_pos=(0.001 * i, 0, 0))
                   for i in range(5)]
        samples, stats = cal.select_calibration_samples(
            records, min_distance=0.0)
        self.assertEqual(len(samples), 5)
        self.assertEqual(stats["skipped_by_distance"], 0)

    def test_max_samples_cap(self):
        records = [make_tp_record(i, control=True, target_valid=True,
                                  hand_pos=(0.1 * i, 0, 0))
                   for i in range(10)]
        samples, stats = cal.select_calibration_samples(
            records, min_distance=0.0, max_samples=3)
        self.assertEqual(len(samples), 3)
        self.assertEqual([s["sequence"] for s in samples], [0, 1, 2])
        self.assertEqual(stats["skipped_by_max_samples"], 7)

    def test_stats_robot_calibrated_false(self):
        records = [make_tp_record(0)]
        _, stats = self._run(records)
        self.assertFalse(stats["robot_calibrated"])
        self.assertEqual(stats["coordinate_frame"], "unity_tracking_space")
        self.assertTrue(any("robot_tcp_position" in n for n in stats["notes"]))


class CliTests(unittest.TestCase):
    def _write_input(self, path: Path, n=4):
        records = [make_tp_record(i, control=True, target_valid=True,
                                  hand_pos=(0.05 * i, 0, 0))
                   for i in range(n)]
        with open(path, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")
        return records

    def test_end_to_end_and_input_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            inp = tmp / "in.jsonl"
            self._write_input(inp)
            before = inp.read_bytes()
            out = tmp / "samples.jsonl"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = cal.main(["--input", str(inp), "--output", str(out),
                               "--min-distance", "0.0"])
            self.assertEqual(rc, 0)
            self.assertEqual(inp.read_bytes(), before)  # 原始文件不变
            lines = out.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 4)
            first = json.loads(lines[0])
            self.assertEqual(first["coordinate_status"],
                             "uncalibrated_unity_tracking_space")
            text = buf.getvalue()
            self.assertIn("输入总帧数", text)
            self.assertIn("有效目标帧数", text)
            self.assertIn("选中标定样本数", text)
            self.assertIn("已完成真实机器人标定", text)

    def test_refuses_existing_output_and_same_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            inp = tmp / "in.jsonl"
            self._write_input(inp)
            out = tmp / "samples.jsonl"
            out.write_text("keep", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                rc = cal.main(["--input", str(inp), "--output", str(out)])
            self.assertEqual(rc, 2)
            self.assertEqual(out.read_text(encoding="utf-8"), "keep")
            before = inp.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                rc = cal.main(["--input", str(inp), "--output", str(inp)])
            self.assertEqual(rc, 2)
            self.assertEqual(inp.read_bytes(), before)

    def test_overwrite_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            inp = tmp / "in.jsonl"
            self._write_input(inp)
            out = tmp / "samples.jsonl"
            out.write_text("old", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                rc = cal.main(["--input", str(inp), "--output", str(out),
                               "--overwrite", "--min-distance", "0.0"])
            self.assertEqual(rc, 0)
            self.assertEqual(len(out.read_text().strip().splitlines()), 4)

    def test_arg_validation(self):
        for bad in ("nan", "-1"):
            with self.assertRaises(SystemExit):
                with contextlib.redirect_stderr(io.StringIO()):
                    cal.main(["--input", "x", "--output", "y",
                              "--min-distance", bad])
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                cal.main(["--input", "x", "--output", "y",
                          "--max-samples", "-2"])


class RigidTransformTests(unittest.TestCase):
    """刚体变换估计（Umeyama）虚拟测试：全部使用人工构造点。"""

    @staticmethod
    def _rodrigues(axis, angle):
        x, y, z = axis
        c, s = math.cos(angle), math.sin(angle)
        k = [[0, -z, y], [z, 0, -x], [-y, x, 0]]
        k2 = cal._matmul(k, k)
        return [[(1 if i == j else 0) + s * k[i][j] + (1 - c) * k2[i][j]
                 for j in range(3)] for i in range(3)]

    @staticmethod
    def _apply(r, t, points):
        return [[sum(r[i][k] * p[k] for k in range(3)) + t[i]
                 for i in range(3)] for p in points]

    def _assert_close_matrix(self, got, want, tol):
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(got[i][j], want[i][j], delta=tol)

    def test_known_rotation_90deg_about_z(self):
        r_true = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        t_true = [1.5, -2.0, 0.5]
        source = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0.5)]
        target = self._apply(r_true, t_true, source)
        result = cal.estimate_rigid_transform(source, target)
        self.assertIsNotNone(result)
        r, t, rms = result
        self._assert_close_matrix(r, r_true, 1e-6)
        for i in range(3):
            self.assertAlmostEqual(t[i], t_true[i], delta=1e-6)
        self.assertLess(rms, 1e-6)
        self.assertAlmostEqual(cal._det3(r), 1.0, delta=1e-6)

    def test_random_rotation_recovered(self):
        rng = random.Random(42)
        for _ in range(5):
            axis = [rng.uniform(-1, 1) for _ in range(3)]
            norm = math.sqrt(sum(a * a for a in axis))
            axis = [a / norm for a in axis]
            angle = rng.uniform(-math.pi, math.pi)
            r_true = self._rodrigues(axis, angle)
            t_true = [rng.uniform(-2, 2) for _ in range(3)]
            source = [[rng.uniform(-1, 1) for _ in range(3)]
                      for _ in range(8)]
            target = self._apply(r_true, t_true, source)
            r, t, rms = cal.estimate_rigid_transform(source, target)
            self._assert_close_matrix(r, r_true, 1e-6)
            for i in range(3):
                self.assertAlmostEqual(t[i], t_true[i], delta=1e-6)
            self.assertLess(rms, 1e-6)

    def test_noisy_points_small_rms(self):
        rng = random.Random(7)
        r_true = self._rodrigues([0, 0, 1], 0.3)
        t_true = [0.1, -0.2, 0.3]
        source = [[rng.uniform(-1, 1) for _ in range(3)] for _ in range(20)]
        target = []
        for p in self._apply(r_true, t_true, source):
            target.append([q + rng.gauss(0, 0.002) for q in p])
        r, t, rms = cal.estimate_rigid_transform(source, target)
        self.assertGreater(rms, 0.0)
        self.assertLess(rms, 0.01)
        self._assert_close_matrix(r, r_true, 0.05)

    def test_degenerate_inputs_return_none(self):
        # 点数不足
        self.assertIsNone(cal.estimate_rigid_transform([(0, 0, 0), (1, 0, 0)],
                                                       [(0, 0, 0), (1, 0, 0)]))
        # 长度不一致
        self.assertIsNone(cal.estimate_rigid_transform(
            [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
            [(0, 0, 0), (1, 0, 0)]))
        # 共线点（旋转不可确定）
        collinear = [(i, 0.0, 0.0) for i in range(4)]
        self.assertIsNone(cal.estimate_rigid_transform(collinear, collinear))
        # 重合点
        same = [(1.0, 2.0, 3.0)] * 5
        self.assertIsNone(cal.estimate_rigid_transform(same, same))
        # 非有限输入
        self.assertIsNone(cal.estimate_rigid_transform(
            [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
            [(0, 0, 0), (1, 0, float("nan")), (0, 1, 0)]))

    def test_reflection_input_yields_proper_rotation(self):
        # 目标点经反射变换（det=-1）生成：函数应返回 det=+1 的最近旋转
        r_ref = [[1, 0, 0], [0, 1, 0], [0, 0, -1]]
        source = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]
        target = self._apply(r_ref, [0.5, 0.5, 0.5], source)
        r, t, rms = cal.estimate_rigid_transform(source, target)
        self.assertAlmostEqual(cal._det3(r), 1.0, delta=1e-6)
        self.assertGreater(rms, 0.0)  # 反射无法用旋转精确表示


class RealDataTests(unittest.TestCase):
    """真实 target_packets.jsonl 的选取预检（不写任何文件）。"""

    @unittest.skipUnless(REAL_TARGET.is_file(), "真实 target_packets.jsonl 不存在")
    def test_real_selection_counts(self):
        records = [json.loads(line) for line in
                   REAL_TARGET.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        samples, stats = cal.select_calibration_samples(records)
        self.assertEqual(stats["input_total_frames"], 708)
        self.assertEqual(stats["valid_target_frames"], 292)
        self.assertGreater(stats["selected_calibration_samples"], 0)
        self.assertLessEqual(stats["selected_calibration_samples"], 292)
        self.assertFalse(stats["robot_calibrated"])
        for s in samples:
            self.assertEqual(s["coordinate_status"],
                             "uncalibrated_unity_tracking_space")
            self.assertIsNone(s["robot_tcp_position"])
            self.assertIsNone(s["target_rotation"])
            self.assertEqual(len(s["hand_position"]), 3)
            self.assertEqual(len(s["target_position"]), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
