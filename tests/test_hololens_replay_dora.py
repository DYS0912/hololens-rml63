#!/usr/bin/env python3
"""Dora 回放链路的离线测试（不启动 dora runtime，不监听 UDP）。

覆盖: 回放批消息与 EOS、原始字段保留、_replay 诊断字段不覆盖原始字段、
转换核心复用 TargetConverter（与直接调用结果一致）、EOS 转发、不可解析行、
保存核心落盘与完成标记、逐帧比较（诊断字段排除 + 各类不匹配检出）、
真实记录离线全链路预检（ReplayCore→ConvertCore→SaverCore 直连）。

运行（项目根目录）:
    python3 tests/test_hololens_replay_dora.py
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DORA_NODES_DIR = (PROJECT_ROOT / "ros2_ws" / "src" / "rml63_bridge"
                  / "rml63_bridge" / "dora_nodes")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


replay_node = _load("hololens_replay_node",
                    DORA_NODES_DIR / "hololens_replay_dora_node.py")
conv_node = _load("hololens_target_node",
                  DORA_NODES_DIR / "hololens_target_dora_node.py")
saver_node = _load("hololens_saver_node",
                   DORA_NODES_DIR / "hololens_saver_dora_node.py")
runner = _load("hololens_replay_dora_runner",
               PROJECT_ROOT / "scripts" / "hololens_replay_dora.py")

REAL_RUN_DIR = PROJECT_ROOT / "datasets" / "hololens_checks" \
    / "run_20261112_101842"
REAL_INPUT = REAL_RUN_DIR / "packets.jsonl"
REAL_TARGET = REAL_RUN_DIR / "target_packets.jsonl"


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


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


class ReplayCoreTests(unittest.TestCase):
    def test_chunks_order_and_eos(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = [make_record(seq=i, mono=1.0 + i * 0.03)
                   for i in range(250)]
            path = Path(tmp) / "in.jsonl"
            _write_jsonl(path, src)
            core = replay_node.ReplayCore(path, chunk_size=100)
            messages = list(core.iter_messages())
            self.assertEqual(len(messages), 4)  # 3 批 + 1 EOS
            self.assertEqual([len(m) for m in messages[:3]], [100, 100, 50])
            eos = messages[-1]
            self.assertTrue(eos["_eos"])
            self.assertEqual(eos["total"], 250)
            self.assertEqual(eos["unparseable"], 0)
            # 顺序与原始字段保留
            flat = [r for m in messages[:3] for r in m]
            for i, rec in enumerate(flat):
                self.assertEqual(rec["sequence"], i)
                self.assertEqual(rec["session_id"], src[i]["session_id"])
                self.assertEqual(rec["timestamp_unix_ms"],
                                 src[i]["timestamp_unix_ms"])
                self.assertEqual(rec["valid"], src[i]["valid"])
                self.assertEqual(rec["control_enabled"],
                                 src[i]["control_enabled"])
                self.assertEqual(rec["_receiver"]["monotonic_s"],
                                 src[i]["_receiver"]["monotonic_s"])
                self.assertEqual(rec["_replay"]["index"], i)
                self.assertIn("wall_time_ms", rec["_replay"])

    def test_blank_lines_skipped_and_unparseable_wrapped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "in.jsonl"
            path.write_text(
                json.dumps(make_record(seq=0)) + "\n\nnot json\n"
                + "[1, 2]\n" + json.dumps(make_record(seq=1)) + "\n",
                encoding="utf-8")
            core = replay_node.ReplayCore(path)
            messages = list(core.iter_messages())
            eos = messages[-1]
            self.assertEqual(eos["total"], 4)   # 空行不计
            self.assertEqual(eos["unparseable"], 2)
            flat = [r for m in messages[:-1] for r in m]
            self.assertEqual(flat[0]["sequence"], 0)
            self.assertEqual(flat[1]["_raw_input_line"], "not json")
            self.assertEqual(flat[2]["_raw_input_line"], "[1, 2]")
            self.assertEqual(flat[3]["sequence"], 1)

    def test_replay_field_never_overwrites_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = make_record(seq=0)
            src["_replay"] = {"keep": 1}  # 原始记录恰好有该键
            path = Path(tmp) / "in.jsonl"
            _write_jsonl(path, [src])
            core = replay_node.ReplayCore(path)
            flat = [r for m in core.iter_messages() for r in m]
            self.assertEqual(flat[0]["_replay"], {"keep": 1})  # 原始保留
            self.assertIn("__replay", flat[0])                # 诊断字段换键
            self.assertEqual(flat[0]["__replay"]["index"], 0)


class ConvertCoreTests(unittest.TestCase):
    def test_matches_direct_target_converter(self):
        # 与直接调用 TargetConverter 的 preview_target 完全一致
        from hololens_target_converter import TargetConverter  # noqa
        records = [
            make_record(seq=0, pos=(1, 1, 1), mono=1.00),
            make_record(seq=1, pos=(2, 1, 1), mono=1.05),
            make_record(seq=2, control=False, pos=(2, 1, 1), mono=1.10),
            make_record(seq=3, pos=(5, 5, 5), mono=1.15),   # 重新捏合
            make_record(seq=3, pos=(5, 5, 5), mono=1.20),   # 重复序号
            make_record(seq=4, pos=(6, 5, 5), mono=1.25),
            make_record(seq=0, session="s2", pos=(9, 9, 9), mono=1.30),
        ]
        core = conv_node.ConvertCore()
        direct = TargetConverter()
        out = []
        for rec in records:
            out.extend(core.on_message([rec]))
        expected = [direct.convert_record(rec) for rec in records]
        self.assertEqual(len(out), len(expected))
        for got, want in zip(out, expected):
            self.assertEqual(got.get("preview_target"),
                             want.get("preview_target"))
        self.assertEqual(core.stats["valid_output"],
                         direct.stats["valid_output"])
        self.assertEqual(core.stats["reanchor_count"],
                         direct.stats["reanchor_count"])

    def test_eos_passthrough(self):
        core = conv_node.ConvertCore()
        eos = {"_eos": True, "total": 3, "unparseable": 0}
        self.assertEqual(core.on_message(eos), eos)
        self.assertEqual(core.stats["total_records"], 0)

    def test_raw_input_line_fallback(self):
        core = conv_node.ConvertCore()
        out = core.on_message([{"_raw_input_line": "not json"}])
        pt = out[0]["preview_target"]
        self.assertEqual(pt["frame_class"], "abnormal")
        self.assertIn("无法解析", pt["reason"])
        self.assertEqual(core.stats["abnormal"], 1)

    def test_chunk_order_preserved(self):
        core = conv_node.ConvertCore()
        records = [make_record(seq=i, pos=(i, 0, 0), mono=1.0 + i * 0.03)
                   for i in range(5)]
        out = core.on_message(records)
        self.assertEqual([r["sequence"] for r in out], list(range(5)))


class SaverCoreTests(unittest.TestCase):
    def _converted(self, seq, frame_class, target_valid=True, pos=(0, 0, 0)):
        return {
            "session_id": "s1", "sequence": seq,
            "preview_target": {
                "target_valid": target_valid,
                "target_position": list(pos) if target_valid else None,
                "target_rotation": None,
                "reason": "x", "frame_class": frame_class,
            },
        }

    def test_write_finish_completed_writes_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            out_path = tmp / "out.jsonl"
            marker_path = tmp / "done.json"
            core = saver_node.SaverCore(out_path, marker_path)
            core.write_items([
                self._converted(0, "valid_output"),
                self._converted(1, "paused", target_valid=False),
                self._converted(2, "abnormal", target_valid=False),
            ])
            marker = core.finish(eos=True)
            self.assertEqual(marker["state"], "completed")
            self.assertTrue(marker["closed"])
            self.assertEqual(marker["records_written"], 3)
            self.assertEqual(marker["counts"],
                             {"valid_output": 1, "paused": 1, "abnormal": 1})
            lines = out_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 3)
            self.assertEqual(json.loads(lines[1])["preview_target"]
                             ["frame_class"], "paused")
            saved = json.loads(marker_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["state"], "completed")

    def test_finish_interrupted_without_eos(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            core = saver_node.SaverCore(tmp / "o.jsonl", tmp / "m.json")
            core.write_items([self._converted(0, "valid_output")])
            marker = core.finish(eos=False)
            self.assertEqual(marker["state"], "interrupted")
            self.assertEqual(marker["records_written"], 1)


class CompareTests(unittest.TestCase):
    def _rec(self, seq, valid, cls, pos):
        return {"session_id": "s1", "sequence": seq,
                "preview_target": {"target_valid": valid,
                                   "target_position": pos,
                                   "target_rotation": None,
                                   "frame_class": cls,
                                   "reason": "r"}}

    def test_identical_with_replay_diagnostics_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            expected = [self._rec(0, True, "valid_output", [0.0, 0.0, 0.0]),
                        self._rec(1, False, "paused", None),
                        self._rec(2, False, "abnormal", None)]
            actual = []
            for i, rec in enumerate(expected):
                copy = json.loads(json.dumps(rec))  # 深拷贝
                copy["_replay"] = {"index": i, "wall_time_ms": 1234.5}
                actual.append(copy)
            e_path = tmp / "e.jsonl"
            a_path = tmp / "a.jsonl"
            _write_jsonl(e_path, expected)
            _write_jsonl(a_path, actual)
            ok, report = runner.compare_target_streams(a_path, e_path)
            self.assertTrue(ok, report)
            self.assertEqual(report["mismatch_count"], 0)
            self.assertEqual(report["actual_lines"], 3)
            self.assertEqual(report["actual_counts"]["total"], 3)
            self.assertIn("_replay", report["excluded_fields"])

    def test_count_mismatch_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            e_path = tmp / "e.jsonl"
            a_path = tmp / "a.jsonl"
            _write_jsonl(e_path, [self._rec(0, True, "valid_output", [0, 0, 0])])
            _write_jsonl(a_path, [self._rec(0, True, "valid_output", [0, 0, 0]),
                                  self._rec(1, False, "paused", None)])
            ok, report = runner.compare_target_streams(a_path, e_path)
            self.assertFalse(ok)
            self.assertTrue(any(m["field"] == "count"
                                for m in report["mismatches"]))

    def test_field_mismatches_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            expected = [self._rec(0, True, "valid_output", [0.2, 0.0, 0.0]),
                        self._rec(1, False, "paused", None)]
            actual = [
                {  # 顺序/sequence 不符
                    "session_id": "s1", "sequence": 1,
                    "preview_target": {"target_valid": True,
                                       "target_position": [0.2, 0.0, 0.0],
                                       "target_rotation": None,
                                       "frame_class": "valid_output"}},
                {  # target_valid / 位置 / 旋转 / 分类均不符
                    "session_id": "s1", "sequence": 2,
                    "preview_target": {"target_valid": True,
                                       "target_position": [0.3, 0.0, 0.0],
                                       "target_rotation": [0, 0, 0, 1],
                                       "frame_class": "valid_output"}},
            ]
            e_path = tmp / "e.jsonl"
            a_path = tmp / "a.jsonl"
            _write_jsonl(e_path, expected)
            _write_jsonl(a_path, actual)
            ok, report = runner.compare_target_streams(a_path, e_path)
            self.assertFalse(ok)
            fields = {m["field"] for m in report["mismatches"]}
            self.assertIn("sequence", fields)
            self.assertIn("target_valid", fields)
            self.assertIn("target_position", fields)
            self.assertIn("target_rotation", fields)
            self.assertIn("frame_class", fields)


class OfflinePipelineTests(unittest.TestCase):
    """真实记录离线全链路预检（不启动 dora runtime）。"""

    @unittest.skipUnless(REAL_INPUT.is_file() and REAL_TARGET.is_file(),
                         "真实记录/已有结果不存在")
    def test_full_pipeline_matches_existing_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            out_path = tmp / "replay_output.jsonl"
            marker_path = tmp / "done.json"
            replay = replay_node.ReplayCore(REAL_INPUT)
            conv = conv_node.ConvertCore()
            saver = saver_node.SaverCore(out_path, marker_path)
            for msg in replay.iter_messages():
                if isinstance(msg, dict) and msg.get("_eos"):
                    marker = saver.finish(eos=True)
                    break
                saver.write_items(conv.on_message(msg))
            self.assertEqual(marker["state"], "completed")
            self.assertEqual(marker["records_written"], 708)
            self.assertEqual(marker["counts"],
                             {"valid_output": 292, "paused": 416,
                              "abnormal": 0})
            ok, report = runner.compare_target_streams(out_path, REAL_TARGET)
            self.assertTrue(ok, report["mismatches"][:5])
            self.assertEqual(report["actual_counts"],
                             {"total": 708, "valid_output": 292,
                              "paused": 416, "abnormal": 0,
                              "target_valid": 292})


if __name__ == "__main__":
    unittest.main(verbosity=2)
