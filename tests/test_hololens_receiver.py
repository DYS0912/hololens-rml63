#!/usr/bin/env python3
"""hololens_receiver.py 离线测试（仅标准库 unittest）。

运行（项目根目录）:
    python3 tests/test_hololens_receiver.py
    python3 -m unittest discover -s tests -v

注意: socket 集成测试只使用本地回环 127.0.0.1 与临时空闲端口，
绝不占用真实采集端口 5005。
"""

from __future__ import annotations

import importlib.util
import json
import math
import socket
import tempfile
import threading
import time
import unittest
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECEIVER_PATH = PROJECT_ROOT / "scripts" / "hololens_receiver.py"

_spec = importlib.util.spec_from_file_location("hololens_receiver",
                                               RECEIVER_PATH)
hr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hr)


def make_msg(**overrides) -> dict:
    """构造一个符合协议的完整消息。"""
    msg = {
        "schema": "rml63.hand_pose.v1",
        "source": "hololens2",
        "hand": "right",
        "pose_type": "openxr_grip",
        "coordinate_frame": "unity_tracking_space",
        "position_unit": "m",
        "quaternion_order": "xyzw",
        "session_id": "sess-test",
        "sequence": 0,
        "timestamp_unix_ms": 1700000000000,
        "monotonic_seconds": 1.25,
        "tracked": True,
        "valid": True,
        "position": [0.1, 0.2, 0.3],
        "rotation": [0.0, 0.0, 0.0, 1.0],
        "invalid_reason": None,
    }
    msg.update(overrides)
    return msg


class ValidationTests(unittest.TestCase):
    """字段/维度/有限性/四元数范数校验。"""

    def test_valid_pose_accepted(self):
        self.assertIsNone(hr.validate_message(make_msg()))

    def test_invalid_pose_empty_null_and_missing_accepted(self):
        cases = [(None, None), ([], []), (None, []), ([], None)]
        for pos, rot in cases:
            msg = make_msg(valid=False, tracked=False, position=pos,
                           rotation=rot, invalid_reason="out_of_view")
            self.assertIsNone(hr.validate_message(msg), msg)
        # 缺省字段视同 null
        msg = make_msg(valid=False, tracked=False, invalid_reason="lost")
        msg.pop("position")
        msg.pop("rotation")
        self.assertIsNone(hr.validate_message(msg))

    def test_invalid_pose_allows_but_ignores_wellformed_vector(self):
        msg = make_msg(valid=False, tracked=False,
                       position=[1.0, 2.0, 3.0], rotation=[0, 0, 0, 1])
        self.assertIsNone(hr.validate_message(msg))

    def test_invalid_pose_wrong_field_types_rejected(self):
        cases = [
            {"position": "nope"},
            {"position": [1, 2]},             # 2 维
            {"rotation": [0, 0, 0]},          # 3 维
            {"position": [1, "x", 3]},        # 元素非数值
            {"position": {"x": 1}},           # 非数组
        ]
        for over in cases:
            msg = make_msg(valid=False, tracked=False, **over)
            self.assertIsNotNone(hr.validate_message(msg), over)

    def test_valid_not_bool_rejected(self):
        for bad in (1, 0, "true", None):
            self.assertEqual(hr.validate_message(make_msg(valid=bad)),
                             "valid_not_bool")

    def test_tracked_not_bool_rejected(self):
        for bad in (1, "yes", None):
            self.assertEqual(hr.validate_message(make_msg(tracked=bad)),
                             "tracked_not_bool")

    def test_sequence_must_be_non_negative_int_not_bool(self):
        self.assertEqual(hr.validate_message(make_msg(sequence=True)),
                         "sequence_invalid")   # 布尔不能当整数
        self.assertEqual(hr.validate_message(make_msg(sequence=-1)),
                         "sequence_invalid")
        self.assertEqual(hr.validate_message(make_msg(sequence=1.5)),
                         "sequence_invalid")
        self.assertEqual(hr.validate_message(make_msg(sequence="1")),
                         "sequence_invalid")

    def test_session_id_must_be_non_empty_string(self):
        self.assertEqual(hr.validate_message(make_msg(session_id="")),
                         "session_id_invalid")
        self.assertEqual(hr.validate_message(make_msg(session_id=42)),
                         "session_id_invalid")
        msg = make_msg()
        msg.pop("session_id")
        self.assertEqual(hr.validate_message(msg), "session_id_invalid")

    def test_time_fields_must_be_finite_numbers(self):
        self.assertEqual(
            hr.validate_message(make_msg(timestamp_unix_ms=float("nan"))),
            "timestamp_unix_ms_invalid")
        self.assertEqual(
            hr.validate_message(make_msg(monotonic_seconds=float("inf"))),
            "monotonic_seconds_invalid")
        self.assertEqual(
            hr.validate_message(make_msg(timestamp_unix_ms="1700")),
            "timestamp_unix_ms_invalid")

    def test_position_rotation_dimension_and_finite(self):
        self.assertEqual(hr.validate_message(make_msg(position=[0, 0])),
                         "position_invalid")
        self.assertEqual(hr.validate_message(make_msg(rotation=[0, 0, 0])),
                         "rotation_invalid")
        self.assertEqual(
            hr.validate_message(make_msg(position=[0, 0, float("inf")])),
            "position_invalid")
        self.assertEqual(
            hr.validate_message(make_msg(rotation=[0, 0, float("nan"), 1])),
            "rotation_invalid")
        self.assertEqual(hr.validate_message(make_msg(position=(0, 0, 0))),
                         "position_invalid")  # 必须是 JSON 数组而非元组

    def test_quaternion_norm_tolerance_documented_1e_3(self):
        # 容差 1e-3 内接受
        self.assertIsNone(
            hr.validate_message(make_msg(rotation=[0, 0, 0, 1.0005])))
        # 超出容差拒绝
        self.assertEqual(
            hr.validate_message(make_msg(rotation=[0, 0, 0, 1.002])),
            "quaternion_norm_out_of_tolerance")
        self.assertEqual(
            hr.validate_message(make_msg(rotation=[1, 1, 1, 1])),
            "quaternion_norm_out_of_tolerance")

    def test_schema_and_object_checks(self):
        self.assertEqual(hr.validate_message(make_msg(schema="other.v1")),
                         "schema_mismatch")
        self.assertEqual(hr.validate_message([1, 2, 3]), "not_json_object")


class DatagramProcessingTests(unittest.TestCase):
    """整包处理：来源过滤、JSON 错误、计数、落盘。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.recv = hr.HoloLensReceiver(bind_ip="172.20.10.2", port=5005,
                                        peer_ip="172.20.10.3", duration=0.0,
                                        log_root=self._tmp.name)
        self.recv.open_run_files()

    def _send(self, payload: bytes) -> str:
        return self.recv.process_datagram(payload, ("172.20.10.3", 6060))

    def test_wrong_source_counted_but_not_recorded(self):
        result = self.recv.process_datagram(b"{}", ("192.168.1.99", 1234))
        self.assertEqual(result, "wrong_source")
        self.assertEqual(self.recv.wrong_source, 1)
        self.assertEqual(self.recv.validated, 0)
        self.assertEqual(self.recv.total_received, 1)
        self.recv.packets_file.flush()
        self.assertEqual(
            self.recv.packets_path.read_text(encoding="utf-8"), "")

    def test_json_errors_do_not_exit_and_are_counted(self):
        for payload in (b"not json at all", b"", b'{"valid": true}'):
            result = self._send(payload)
            self.assertEqual(result, "malformed")
        self.assertEqual(self.recv.malformed, 3)
        self.assertIn("not_json", self.recv.malformed_reasons)
        self.assertIn("schema_mismatch", self.recv.malformed_reasons)
        self.assertEqual(self.recv.validated, 0)

    def test_non_finite_json_constant_rejected(self):
        raw = json.dumps(make_msg(timestamp_unix_ms=float("nan")))
        result = self._send(raw.encode())
        self.assertEqual(result, "malformed")
        self.assertIn("non_finite_json_constant",
                      self.recv.malformed_reasons)

    def test_valid_and_invalid_saved_with_original_fields_and_rx_meta(self):
        for i in range(3):
            self.assertEqual(self._send(
                json.dumps(make_msg(sequence=i)).encode()), "saved_valid")
        invalid = make_msg(sequence=3, valid=False, tracked=False,
                           position=None, rotation=[],
                           invalid_reason="occluded")
        self.assertEqual(self._send(json.dumps(invalid).encode()),
                         "saved_invalid")
        self.recv.packets_file.flush()
        lines = self.recv.packets_path.read_text(
            encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 4)
        rec = json.loads(lines[0])
        self.assertIs(rec["valid"], True)
        self.assertEqual(rec["position"], [0.1, 0.2, 0.3])  # 原始值未改
        rx = rec["_receiver"]
        self.assertIn("wall_time_ms", rx)
        self.assertIn("monotonic_s", rx)
        self.assertEqual(rx["from_ip"], "172.20.10.3")
        self.assertEqual(rx["from_port"], 6060)
        rec4 = json.loads(lines[3])
        self.assertIsNone(rec4["position"])      # 保留原始 null，不零填充
        self.assertEqual(rec4["rotation"], [])   # 保留原始空数组
        self.assertEqual(rec4["invalid_reason"], "occluded")
        self.assertEqual(self.recv.valid_count, 3)
        self.assertEqual(self.recv.invalid_count, 1)
        self.assertEqual(self.recv.validated, 4)

    def test_sender_fields_never_overwritten(self):
        msg = make_msg(sequence=0)
        msg["_receiver"] = {"keep": 1}  # 发送端恰好占用附加键名
        self._send(json.dumps(msg).encode())
        self.recv.packets_file.flush()
        rec = json.loads(
            self.recv.packets_path.read_text(encoding="utf-8").strip())
        self.assertEqual(rec["_receiver"], {"keep": 1})  # 原始字段保留
        self.assertEqual(rec["__receiver"]["from_ip"], "172.20.10.3")

    def test_malformed_does_not_mix_into_validated_counts(self):
        self._send(b"garbage")
        self._send(json.dumps(make_msg(sequence=0)).encode())
        self.assertEqual(self.recv.malformed, 1)
        self.assertEqual(self.recv.validated, 1)


class SequenceTrackingTests(unittest.TestCase):
    """按 session_id 的序号统计。"""

    def _track(self, packets):
        tracker = hr.SessionTracker()
        for seq, valid in packets:
            tracker.add(seq, valid)
        return tracker

    def test_session_switch_resets_sequence(self):
        t1 = self._track([(0, True), (1, True), (2, True)])
        self.assertEqual(t1.missing_sequences, [])
        t2 = self._track([(0, True), (1, True)])
        self.assertEqual(t2.missing_sequences, [])  # 新会话从 0 开始不算丢包
        self.assertEqual(t2.first_sequence, 0)

    def test_first_packet_high_sequence_creates_no_retroactive_gap(self):
        tracker = self._track([(100, True), (101, True)])
        self.assertEqual(tracker.missing_sequences, [])

    def test_duplicate_out_of_order_gap_and_late_fill(self):
        tracker = self._track([
            (0, True), (1, True), (2, True), (4, True), (5, True),
            (3, True), (3, True), (8, False)])
        self.assertEqual(tracker.packet_count, 8)
        self.assertEqual(tracker.duplicate_count, 1)
        self.assertEqual(tracker.out_of_order_count, 1)   # 3 晚于 5 到达
        self.assertEqual(tracker.late_fill_count, 1)      # 3 补齐了缺口
        self.assertEqual(tracker.missing_sequences, [6, 7])  # 结束仍未补齐
        self.assertEqual(tracker.max_sequence, 8)
        self.assertEqual(tracker.valid_count, 7)
        self.assertEqual(tracker.invalid_count, 1)

    def test_per_session_isolation_in_receiver(self):
        with tempfile.TemporaryDirectory() as tmp:
            recv = hr.HoloLensReceiver(bind_ip="b", port=0, peer_ip="p",
                                       log_root=tmp)
            recv.open_run_files()
            for i in range(3):
                recv.process_datagram(
                    json.dumps(make_msg(session_id="A",
                                        sequence=i)).encode(), ("p", 1))
            recv.process_datagram(
                json.dumps(make_msg(session_id="B",
                                    sequence=0)).encode(), ("p", 1))
            self.assertEqual(recv.sessions["A"].missing_sequences, [])
            self.assertEqual(recv.sessions["B"].missing_sequences, [])
            self.assertEqual(recv.sessions["B"].first_sequence, 0)


class SummaryTests(unittest.TestCase):
    """summary.json 的零包/单包/频率口径。"""

    def _bare(self, **kw):
        kw.setdefault("bind_ip", "b")
        kw.setdefault("port", 5005)
        kw.setdefault("peer_ip", "p")
        kw.setdefault("log_root", "/tmp/not-used")
        return hr.HoloLensReceiver(**kw)

    def test_zero_packet_summary_all_null_with_reasons(self):
        recv = self._bare()
        summary = recv.build_summary("duration_reached", datetime.now(),
                                     123.0)
        self.assertEqual(summary["packets"]["validated"], 0)
        self.assertIsNone(summary["packets"]["valid_ratio"])
        self.assertIsNotNone(summary["packets"]["valid_ratio_reason"])
        self.assertIsNone(summary["timing"]["avg_receive_rate_hz"])
        self.assertIsNotNone(summary["timing"]["avg_receive_rate_reason"])
        self.assertIsNone(summary["timing"]["wait_before_first_packet_s"])
        self.assertIsNotNone(
            summary["timing"]["wait_before_first_packet_reason"])
        self.assertIsNone(summary["timing"]["interval_mean_s"])
        self.assertIsNone(summary["timing"]["interval_p95_s"])
        self.assertIsNone(summary["timing"]["interval_max_s"])
        self.assertIsNotNone(summary["timing"]["interval_stats_reason"])
        self.assertEqual(summary["run"]["end_reason"], "duration_reached")
        self.assertEqual(summary["sessions"], {})

    def test_single_packet_summary_no_rate(self):
        recv = self._bare()
        recv.listen_start_mono = 100.0
        recv.first_validated_mono = 105.0
        recv.last_validated_mono = 105.0
        recv.validated = 1
        recv.valid_count = 1
        summary = recv.build_summary("duration_reached", datetime.now(),
                                     110.0)
        self.assertIsNone(summary["timing"]["avg_receive_rate_hz"])
        self.assertIn("少于 2", summary["timing"]["avg_receive_rate_reason"])
        self.assertEqual(summary["timing"]["wait_before_first_packet_s"],
                         5.0)
        self.assertEqual(summary["packets"]["valid_ratio"], 1.0)

    def test_first_packet_wait_excluded_from_rate(self):
        recv = self._bare()
        recv.listen_start_mono = 0.0
        recv.first_validated_mono = 10.0   # 首包前等了 10 秒
        recv.last_validated_mono = 11.0
        recv.validated = 10
        recv.valid_count = 10
        summary = recv.build_summary("duration_reached", datetime.now(),
                                     20.0)
        self.assertAlmostEqual(summary["timing"]["avg_receive_rate_hz"],
                               9.0)  # (10-1)/(11-10)，不含首包前等待
        self.assertEqual(summary["timing"]["wait_before_first_packet_s"],
                         10.0)

    def test_zero_span_no_division(self):
        recv = self._bare()
        recv.listen_start_mono = 0.0
        recv.first_validated_mono = 5.0
        recv.last_validated_mono = 5.0   # 首末同刻
        recv.validated = 5
        recv.valid_count = 5
        summary = recv.build_summary("duration_reached", datetime.now(),
                                     10.0)
        self.assertIsNone(summary["timing"]["avg_receive_rate_hz"])
        self.assertIn("时间差为 0",
                      summary["timing"]["avg_receive_rate_reason"])

    def test_valid_ratio_over_all_records(self):
        recv = self._bare()
        recv.validated = 10
        recv.valid_count = 7
        recv.invalid_count = 3
        summary = recv.build_summary("duration_reached", datetime.now(),
                                     0.0)
        self.assertAlmostEqual(summary["packets"]["valid_ratio"], 0.7)

    def test_interval_stats_mean_p95_max(self):
        recv = self._bare()
        recv.intervals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                          10.0]
        summary = recv.build_summary("duration_reached", datetime.now(),
                                     0.0)
        self.assertAlmostEqual(summary["timing"]["interval_mean_s"], 5.5)
        self.assertEqual(summary["timing"]["interval_max_s"], 10.0)
        self.assertEqual(summary["timing"]["interval_p95_s"],
                         10.0)  # 最近秩法: 第10个
        self.assertEqual(summary["timing"]["p95_method"], "nearest-rank")


class DisplayTests(unittest.TestCase):
    """终端显示：超时后不得沿用旧有效坐标。"""

    def _bare(self, **kw):
        kw.setdefault("bind_ip", "b")
        kw.setdefault("port", 1)
        kw.setdefault("peer_ip", "p")
        kw.setdefault("log_root", "/tmp/not-used")
        return hr.HoloLensReceiver(**kw)

    def test_valid_line_shows_position(self):
        recv = self._bare()
        recv.last_validated_mono = 100.0
        recv.last_status = "valid"
        recv.last_position = [1.25, -0.5, 0.0]
        line = recv.status_line(101.0)
        self.assertIn("右手追踪正常", line)
        self.assertIn("[1.250, -0.500, 0.000]", line)

    def test_invalid_line_never_shows_coordinates(self):
        recv = self._bare()
        recv.last_validated_mono = 100.0
        recv.last_status = "invalid"
        recv.last_position = [9.0, 9.0, 9.0]  # 即使残留也不显示
        line = recv.status_line(101.0)
        self.assertIn("右手姿态不可用", line)
        self.assertIn("无坐标", line)
        self.assertNotIn("9.000", line)

    def test_timeout_does_not_keep_showing_old_pose(self):
        recv = self._bare()
        recv.no_data_timeout = 5.0
        recv.last_validated_mono = 100.0
        recv.last_status = "valid"
        recv.last_position = [0.1, 0.2, 0.3]
        line = recv.status_line(107.0)
        self.assertIn("等待头显数据", line)
        self.assertNotIn("右手追踪正常", line)
        self.assertNotIn("0.100", line)  # 旧坐标不再显示

    def test_within_timeout_still_shows_pose(self):
        recv = self._bare()
        recv.no_data_timeout = 5.0
        recv.last_validated_mono = 100.0
        recv.last_status = "valid"
        recv.last_position = [0.1, 0.2, 0.3]
        line = recv.status_line(104.9)
        self.assertIn("右手追踪正常", line)

    def test_no_packets_yet_is_waiting(self):
        recv = self._bare()
        line = recv.status_line(50.0)
        self.assertIn("等待头显数据", line)

    def test_recent_second_packet_count_and_counts(self):
        recv = self._bare()
        recv.no_data_timeout = 5.0
        recv.recent_times.extend([95.0, 99.5, 100.2])
        recv.last_validated_mono = 100.2
        recv.last_status = "invalid"
        recv.valid_count = 7
        recv.invalid_count = 3
        line = recv.status_line(101.0)
        self.assertIn("最近1秒: 1包", line)   # 只有 100.2 在最近 1 秒内
        self.assertIn("累计: 有效 7 / 无效 3", line)


class LogAndDirTests(unittest.TestCase):
    """日志读回、独立运行目录、正常收尾。"""

    def test_run_dirs_unique_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            d1 = hr.make_run_dir(Path(tmp))
            d2 = hr.make_run_dir(Path(tmp))
            self.assertNotEqual(d1, d2)
            self.assertTrue(d1.is_dir() and d2.is_dir())

    def test_finish_writes_parsable_summary_and_closes_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            recv = hr.HoloLensReceiver(bind_ip="127.0.0.1", port=0,
                                       peer_ip="127.0.0.1", log_root=tmp)
            recv.open_run_files()
            recv.process_datagram(
                json.dumps(make_msg(sequence=0)).encode(),
                ("127.0.0.1", 1))
            recv.process_datagram(
                json.dumps(make_msg(sequence=1, valid=False, tracked=False,
                                    position=[], rotation=None)).encode(),
                ("127.0.0.1", 1))
            rc = recv._finish("duration_reached")
            self.assertEqual(rc, 0)
            self.assertTrue(recv.summary_path.is_file())
            summary = json.loads(
                recv.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["packets"]["validated"], 2)
            self.assertEqual(summary["packets"]["valid_pose"], 1)
            self.assertEqual(summary["packets"]["invalid_pose"], 1)
            self.assertIsInstance(summary["timing"]["avg_receive_rate_hz"],
                                  float)
            self.assertEqual(summary["run"]["end_reason"],
                             "duration_reached")
            lines = recv.packets_path.read_text(
                encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            last = json.loads(lines[1])
            self.assertEqual(last["position"], [])   # 原始空数组保留
            self.assertIsNone(last["rotation"])      # 原始 null 保留


class SocketIntegrationTests(unittest.TestCase):
    """真实 UDP socket 集成测试：只走 127.0.0.1 回环与临时端口。"""

    @staticmethod
    def _free_port() -> int:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        return port

    def test_loopback_receive_summary_and_packets(self):
        port = self._free_port()   # 绝不使用 5005
        with tempfile.TemporaryDirectory() as tmp:
            recv = hr.HoloLensReceiver(bind_ip="127.0.0.1", port=port,
                                       peer_ip="127.0.0.1", duration=1.5,
                                       log_root=tmp, no_data_timeout=30.0)
            recv.start()
            thread = threading.Thread(target=recv.run, daemon=True)
            thread.start()
            client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            client.settimeout(1.0)
            target = ("127.0.0.1", recv.actual_port)
            try:
                for i in range(5):
                    client.sendto(
                        json.dumps(make_msg(sequence=i)).encode(), target)
                    time.sleep(0.05)
                client.sendto(json.dumps(
                    make_msg(sequence=5, valid=False, tracked=False,
                             position=None, rotation=[],
                             invalid_reason="lost")).encode(), target)
                client.sendto(b"garbage", target)
            finally:
                client.close()
            thread.join(timeout=10.0)
            self.assertFalse(thread.is_alive(),
                             "接收线程应在 duration 后自行退出")
            summary = json.loads(
                recv.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["run"]["end_reason"],
                             "duration_reached")
            self.assertEqual(summary["packets"]["total_received"], 7)
            self.assertEqual(summary["packets"]["wrong_source"], 0)
            self.assertEqual(summary["packets"]["malformed"], 1)
            self.assertIn("not_json",
                          summary["packets"]["malformed_reasons"])
            self.assertEqual(summary["packets"]["validated"], 6)
            self.assertEqual(summary["packets"]["valid_pose"], 5)
            self.assertEqual(summary["packets"]["invalid_pose"], 1)
            self.assertAlmostEqual(summary["packets"]["valid_ratio"],
                                   5.0 / 6.0)
            self.assertIsNotNone(summary["timing"]["avg_receive_rate_hz"])
            self.assertGreaterEqual(
                summary["timing"]["wait_before_first_packet_s"], 0.0)
            self.assertIsNone(summary["timing"]["interval_stats_reason"])
            lines = recv.packets_path.read_text(
                encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 6)
            last = json.loads(lines[-1])
            self.assertIsNone(last["position"])   # valid=false 保留 null

    def test_duration_counts_from_listen_start_not_first_packet(self):
        port = self._free_port()
        with tempfile.TemporaryDirectory() as tmp:
            recv = hr.HoloLensReceiver(bind_ip="127.0.0.1", port=port,
                                       peer_ip="127.0.0.1", duration=1.0,
                                       log_root=tmp, no_data_timeout=30.0)
            recv.start()
            t0 = time.monotonic()
            thread = threading.Thread(target=recv.run, daemon=True)
            thread.start()
            time.sleep(0.8)   # 首包前故意等待 0.8 秒
            client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            client.settimeout(1.0)
            client.sendto(json.dumps(make_msg(sequence=0)).encode(),
                          ("127.0.0.1", recv.actual_port))
            client.close()
            thread.join(timeout=10.0)
            elapsed = time.monotonic() - t0
            # 若从首包起计时会约 0.8+1.0=1.8s 才结束；从监听起计时应 ~1.0s
            self.assertLess(elapsed, 1.6,
                            f"duration 应从监听成功起计时，实际 {elapsed:.2f}s")
            summary = json.loads(
                recv.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["run"]["end_reason"],
                             "duration_reached")
            self.assertEqual(summary["packets"]["validated"], 1)
            self.assertIsNone(summary["timing"]["avg_receive_rate_hz"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
