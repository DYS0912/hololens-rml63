#!/usr/bin/env python3
"""hololens_robot_control.py 与 hololens_live_preview.py 的离线逻辑测试。

全部使用模拟对象：不导入/启动 rclpy，不监听 UDP，不发布任何 ROS 话题，
不连接任何硬件。覆盖:
  - 几何限制：clamp_radius / limit_step 空基准拒绝 / 目标-反馈偏差；
  - 每捏合从零增量：首目标==当前有效机械臂反馈锚点；不叠加旧累计位移；
  - 同批次"松开→重新捏合"：锚点与步长基准及时重置；
  - 停止状态机：请求后禁发新命令；旧 stop_result 无效；失败/反馈过期锁定；
    确认需 结果+位置+关节速度 齐备；恢复必须重新看到有效松开；
  - 只读模式绝不发布运动/停止消息；
  - SIGINT/SIGTERM 退出请求、有界停止确认、上下文失效明确报告无法确认停止。

运行（项目根目录）:
    python3 tests/test_hololens_control_logic.py
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import hololens_live_preview as lp
import hololens_robot_control as rc


def make_packet(seq, control=True, valid=True, pos=(0.5, 0.0, 0.0),
                session="s1", **over):
    packet = {
        "schema": "rml63.hand_pose.v1",
        "source": "hololens2",
        "hand": "right",
        "pose_type": "openxr_grip",
        "coordinate_frame": "unity_tracking_space",
        "position_unit": "m",
        "quaternion_order": "xyzw",
        "session_id": session,
        "sequence": seq,
        "timestamp_unix_ms": 1700000000000 + seq,
        "monotonic_seconds": 1.0 + seq * 0.03,
        "tracked": True,
        "valid": valid,
        "control_enabled": control,
        "position": list(pos),
        "rotation": [0.0, 0.0, 0.0, 1.0],
        "invalid_reason": None,
    }
    packet.update(over)
    return packet


def send(state, pkt, t):
    return state.receive(json.dumps(pkt).encode("utf-8"),
                         ("172.20.10.3", 6060), t)


def set_robot(state, pos, t):
    state.robot_pose(list(pos), [0.0, 0.0, 0.0, 1.0], t)


def make_state():
    return lp.Preview(peer="172.20.10.3", scale=0.2,
                      hand_timeout=0.2, robot_timeout=0.5)


def make_cfg(**over):
    cfg = SimpleNamespace(max_offset=0.05, max_step=0.003,
                          max_feedback_deviation=0.06, period=0.0)
    for key, value in over.items():
        setattr(cfg, key, value)
    return cfg


def make_planner():
    return {"last_sent": None, "last_send": 0.0, "reference": None,
            "release_count_seen": 0}


def make_snap(**over):
    snap = {
        "active": True,
        "release_required": True,
        "release_count": 0,
        "reason": "预览更新中（未标定，不发送运动命令）",
        "robot_reference_at_pinch": {"position_m": [0.3, 0.0, 0.4],
                                     "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]},
        "robot_feedback": {"position_m": [0.3, 0.0, 0.4],
                           "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]},
        "robot_target_preview_base_m": [0.3, 0.0, 0.4],
        "robot_target_preview_rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    snap.update(over)
    return snap


class FakeIO:
    def __init__(self, enable_motion=False):
        self.enable_motion = bool(enable_motion)
        self.motion_published = []
        self.stop_publish_count = 0
        self.stop_result = None      # (t, ok) 或 None
        self.movel_result = None     # (t, ok) 或 None，无命令编号
        self.pose = None             # (t, pos)
        self.speed = None            # (t, [v...])
        self.pose_seq = 0            # 位置回调序号（observe 只消费新样本）
        self.speed_seq = 0           # 速度回调序号（observe 只消费新样本）
        self.context_valid = True

    def publish_motion(self, msg):
        self.motion_published.append(msg)

    def publish_stop(self):
        self.stop_publish_count += 1

    def context_ok(self):
        return self.context_valid

    def set_pose(self, t, pos):
        self.pose = (t, list(pos))
        self.pose_seq += 1

    def set_speed(self, t, values):
        self.speed = (t, list(values))
        self.speed_seq += 1


class FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def make_sm(dry_run=False, **over):
    kw = dict(still_window_s=0.2, still_pos_epsilon_m=5e-4,
              still_speed_rpm=0.1, stop_confirm_timeout_s=2.0,
              feedback_max_age_s=0.5, max_sample_gap_s=0.05,
              dry_run=dry_run)
    kw.update(over)
    return rc.StopStateMachine(**kw)


def feed(sm, t, pos=(0.3, 0.0, 0.4), speed=(0.0,) * 6):
    """同时喂入位置与速度两路样本（各自保留时间戳）。"""
    sm.add_position_sample(t, list(pos))
    sm.add_speed_sample(t, None if speed is None else list(speed))


class GeometryLimitTests(unittest.TestCase):
    def test_clamp_radius_inside_returns_unchanged(self):
        self.assertEqual(rc.clamp_radius([0.31, 0, 0.4], [0.3, 0, 0.4], 0.05),
                         [0.31, 0.0, 0.4])

    def test_clamp_radius_clamps_to_sphere(self):
        got = rc.clamp_radius([0.4, 0.0, 0.4], [0.3, 0.0, 0.4], 0.05)
        self.assertAlmostEqual(got[0], 0.35, places=12)
        self.assertEqual(got[1:], [0.0, 0.4])

    def test_clamp_radius_zero_delta_unchanged(self):
        self.assertEqual(rc.clamp_radius([0.3, 0, 0.4], [0.3, 0, 0.4], 0.05),
                         [0.3, 0.0, 0.4])

    def test_clamp_radius_rejects_nonfinite_inputs(self):
        for args in (
            ([float("nan"), 0, 0], [0.3, 0, 0.4], 0.05),
            ([0.3, 0, 0.4], None, 0.05),
            ([0.3, 0, 0.4], [0.3, 0, 0.4], 0),
            ([0.3, 0, 0.4], [0.3, 0, 0.4], -1),
            ([0.3, 0, 0.4], [0.3, 0, 0.4], float("inf")),
        ):
            with self.assertRaises(ValueError, msg=args):
                rc.clamp_radius(*args)

    def test_limit_step_rejects_none_baseline(self):
        # 空基准绝不允许：拒绝绕过限制。
        with self.assertRaises(ValueError):
            rc.limit_step([0.3, 0, 0.4], None, 0.003)

    def test_limit_step_clamps(self):
        got = rc.limit_step([0.4, 0.0, 0.4], [0.3, 0.0, 0.4], 0.003)
        self.assertAlmostEqual(got[0], 0.303, places=12)
        self.assertEqual(got[1:], [0.0, 0.4])

    def test_limit_step_within_unchanged(self):
        self.assertEqual(rc.limit_step([0.301, 0, 0.4], [0.3, 0, 0.4], 0.003),
                         [0.301, 0.0, 0.4])

    def test_limit_step_rejects_nonfinite_inputs(self):
        for args in (
            ([float("inf"), 0, 0], [0.3, 0, 0.4], 0.003),
            ([0.3, 0, 0.4], [float("nan"), 0, 0.4], 0.003),
            ([0.3, 0, 0.4], [0.3, 0, 0.4], 0),
        ):
            with self.assertRaises(ValueError, msg=args):
                rc.limit_step(*args)

    def test_check_target_deviation(self):
        self.assertTrue(rc.check_target_deviation([0.31, 0, 0.4],
                                                  [0.3, 0, 0.4], 0.02))
        self.assertFalse(rc.check_target_deviation([0.33, 0, 0.4],
                                                   [0.3, 0, 0.4], 0.02))
        with self.assertRaises(ValueError):
            rc.check_target_deviation([0.3, 0, 0.4], [0.3, 0, float("nan")],
                                      0.02)


class PreviewPinchAnchorTests(unittest.TestCase):
    """实时预览层：每捏合从零增量、首目标==机械臂反馈锚点。"""

    def test_first_pinch_target_equals_robot_feedback(self):
        state = make_state()
        set_robot(state, [0.3, 0.0, 0.4], 10.0)
        send(state, make_packet(0, control=False), 10.0)
        send(state, make_packet(1, control=True, pos=(0.5, 0.0, 0.0)), 10.1)
        snap = state.snapshot(10.15)
        self.assertTrue(snap["active"])
        self.assertEqual(snap["preview_unity_m"], [0.0, 0.0, 0.0])
        self.assertEqual(snap["robot_target_preview_base_m"], [0.3, 0.0, 0.4])
        self.assertEqual(snap["robot_reference_at_pinch"]["position_m"],
                         [0.3, 0.0, 0.4])

    def test_repinch_does_not_carry_previous_displacement(self):
        state = make_state()
        set_robot(state, [0.3, 0.0, 0.4], 10.0)
        send(state, make_packet(0, control=False), 10.0)
        send(state, make_packet(1, control=True, pos=(0.5, 0.0, 0.0)), 10.1)
        send(state, make_packet(2, control=True, pos=(2.5, 0.0, 0.0)), 10.15)
        snap = state.snapshot(10.2)
        self.assertAlmostEqual(snap["preview_unity_m"][0], 0.4, places=12)
        # 松开后机械臂移动，重新捏合：首帧必须等于新反馈，不带旧位移。
        send(state, make_packet(3, control=False), 10.3)
        set_robot(state, [0.32, 0.0, 0.4], 10.4)
        send(state, make_packet(4, control=True, pos=(0.5, 0.0, 0.0)), 10.5)
        snap = state.snapshot(10.55)
        self.assertTrue(snap["active"])
        self.assertEqual(snap["preview_unity_m"], [0.0, 0.0, 0.0])
        self.assertEqual(snap["robot_target_preview_base_m"], [0.32, 0.0, 0.4])

    def test_same_batch_release_repinch_resets_anchor(self):
        state = make_state()
        set_robot(state, [0.3, 0.0, 0.4], 10.0)
        send(state, make_packet(0, control=False), 10.0)
        send(state, make_packet(1, control=True, pos=(0.5, 0.0, 0.0)), 10.1)
        send(state, make_packet(2, control=True, pos=(2.5, 0.0, 0.0)), 10.15)
        set_robot(state, [0.32, 0.0, 0.4], 10.2)
        # 同一批内：松开 + 重新捏合
        send(state, make_packet(3, control=False), 10.25)
        send(state, make_packet(4, control=True, pos=(0.5, 0.0, 0.0)), 10.26)
        snap = state.snapshot(10.27)
        self.assertTrue(snap["active"])
        self.assertEqual(snap["robot_target_preview_base_m"], [0.32, 0.0, 0.4])
        # 新锚点生效：手部增量以新锚点 (0.5,0,0) 起算
        send(state, make_packet(5, control=True, pos=(1.5, 0.0, 0.0)), 10.3)
        snap = state.snapshot(10.35)
        self.assertAlmostEqual(snap["preview_unity_m"][0], 0.2, places=12)

    def test_mid_pinch_gap_triggers_preview_pause(self):
        # 实时路径：手部数据间隔超过 hand_timeout(0.2s) 先触发 Preview 暂停
        # （更严格：必须重新松开）。离线转换器的"间隔重锚定无跳变"行为由
        # tests/test_hololens_target_converter.py::IntervalTests 覆盖。
        state = make_state()
        set_robot(state, [0.3, 0.0, 0.4], 10.0)
        send(state, make_packet(0, control=False), 10.0)
        send(state, make_packet(1, control=True, pos=(0.5, 0.0, 0.0)), 10.1)
        send(state, make_packet(2, control=True, pos=(2.5, 0.0, 0.0)), 10.15)
        # 保持机械臂反馈持续新鲜，仅让手部数据出现 0.85s 间隔
        set_robot(state, [0.3, 0.0, 0.4], 10.45)
        set_robot(state, [0.3, 0.0, 0.4], 10.9)
        send(state, make_packet(3, control=True, pos=(2.6, 0.0, 0.0)), 11.0)
        snap = state.snapshot(11.05)
        self.assertFalse(snap["active"])
        self.assertIn("松开", snap["reason"])

    def test_robot_feedback_stale_pauses(self):
        state = make_state()
        set_robot(state, [0.3, 0.0, 0.4], 10.0)
        send(state, make_packet(0, control=False), 10.0)
        send(state, make_packet(1, control=True, pos=(0.5, 0.0, 0.0)), 10.1)
        self.assertTrue(state.snapshot(10.15)["active"])
        snap = state.snapshot(11.0)  # 反馈年龄 0.9s > 0.5s
        self.assertFalse(snap["active"])

    def test_pinch_without_release_blocked(self):
        state = make_state()
        set_robot(state, [0.3, 0.0, 0.4], 10.0)
        send(state, make_packet(1, control=True, pos=(0.5, 0.0, 0.0)), 10.1)
        snap = state.snapshot(10.15)
        self.assertFalse(snap["active"])
        self.assertIn("松开", snap["reason"])


class ConverterPinchResetTests(unittest.TestCase):
    """转换器层：begin_new_pinch 从零起算、重锚定无跳变、离线语义不受影响。"""

    def _conv(self):
        return lp.LiveConverter(scale=0.2, interval_threshold_s=0.2)

    @staticmethod
    def make_record(seq=0, control=True, valid=True, pos=(0.1, 0.2, 0.3),
                    mono=1.0, **over):
        record = {
            "schema": "rml63.hand_pose.v1",
            "session_id": "s1",
            "sequence": seq,
            "timestamp_unix_ms": 1700000000000 + seq,
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

    @staticmethod
    def _pt(conv, record):
        return conv.convert_record(record)["preview_target"]

    def test_begin_new_pinch_restarts_from_origin(self):
        conv = self._conv()
        self._pt(conv, self.make_record(seq=0, pos=(1, 1, 1)))
        pt = self._pt(conv, self.make_record(seq=1, pos=(2, 1, 1), mono=1.05))
        self.assertAlmostEqual(pt["target_position"][0], 0.2, places=12)
        # 松开
        pt = self._pt(conv, self.make_record(seq=2, control=False, mono=1.1))
        self.assertFalse(pt["target_valid"])
        # 新捏合：begin_new_pinch 后首帧必须回到原点，不带旧累计位移
        conv.begin_new_pinch()
        pt = self._pt(conv, self.make_record(seq=3, pos=(5, 5, 5), mono=1.2))
        self.assertTrue(pt["target_valid"])
        self.assertEqual(pt["target_position"], [0.0, 0.0, 0.0])
        # 新锚点下继续映射：只用本次增量
        pt = self._pt(conv, self.make_record(seq=4, pos=(6, 5, 5), mono=1.25))
        self.assertAlmostEqual(pt["target_position"][0], 0.2, places=12)

    def test_begin_new_pinch_then_gap_reanchor_no_jump(self):
        conv = self._conv()
        conv.begin_new_pinch()
        self._pt(conv, self.make_record(seq=0, pos=(1, 1, 1), mono=1.00))
        pt = self._pt(conv, self.make_record(seq=1, pos=(2, 1, 1), mono=1.05))
        self.assertAlmostEqual(pt["target_position"][0], 0.2, places=12)
        # 间隔 0.55s > 0.2s：中断锚定并重新锚定，输出上一目标（无跳变）
        pt = self._pt(conv, self.make_record(seq=2, pos=(3, 1, 1), mono=1.60))
        self.assertTrue(pt["target_valid"])
        self.assertAlmostEqual(pt["target_position"][0], 0.2, places=12)
        self.assertIn("间隔", pt["reason"])

    def test_offline_semantics_unchanged_without_explicit_call(self):
        # 不调用 begin_new_pinch 时，重新捏合仍输出上一虚拟目标（离线语义）
        conv = self._conv()
        self._pt(conv, self.make_record(seq=0, pos=(1, 1, 1)))
        self._pt(conv, self.make_record(seq=1, pos=(2, 1, 1), mono=1.05))
        self._pt(conv, self.make_record(seq=2, control=False, mono=1.1))
        pt = self._pt(conv, self.make_record(seq=3, pos=(5, 5, 5), mono=1.2))
        self.assertTrue(pt["target_valid"])
        self.assertAlmostEqual(pt["target_position"][0], 0.2, places=12)


class StopStateMachineTests(unittest.TestCase):
    def test_request_stop_publishes_once(self):
        sm = make_sm()
        self.assertTrue(sm.request_stop(10.0, "松开"))
        self.assertEqual(sm.state, sm.STOP_REQUESTED)
        self.assertFalse(sm.request_stop(10.1, "again"))

    def test_old_stop_result_ignored(self):
        sm = make_sm()
        sm.request_stop(10.0, "x")
        sm.add_stop_result(9.5, True)      # 请求之前的旧结果
        self.assertIsNone(sm.stop_result)
        sm.add_stop_result(10.0, True)     # 恰在请求时刻：不接受
        self.assertIsNone(sm.stop_result)
        sm.add_stop_result(10.1, True)     # 本次请求之后：接受
        self.assertEqual(sm.stop_result, (10.1, True))

    def test_stop_result_false_locks(self):
        sm = make_sm()
        sm.request_stop(10.0, "x")
        sm.add_stop_result(10.1, False)
        self.assertEqual(sm.state, sm.LOCKED)
        self.assertIn("false", sm.last_reason)

    def test_confirmation_requires_result_and_still_feedback(self):
        sm = make_sm()
        sm.request_stop(10.0, "x")
        sm.add_stop_result(10.05, True)    # 有结果但无反馈：不确认
        sm.evaluate(10.5)
        self.assertEqual(sm.state, sm.STOP_REQUESTED)
        t = 10.1
        while t <= 10.6:
            feed(sm, t)
            t += 0.05
        sm.evaluate(10.65)
        self.assertEqual(sm.state, sm.STOPPED)
        self.assertIn("已确认", sm.last_reason)

    def test_moving_speed_blocks_confirmation_then_locks(self):
        sm = make_sm()
        sm.request_stop(10.0, "x")
        sm.add_stop_result(10.05, True)
        t = 10.1
        while t <= 10.6:
            feed(sm, t, speed=(0.5,) * 6)  # 0.5 RPM > 0.1
            t += 0.05
        sm.evaluate(10.65)
        self.assertEqual(sm.state, sm.STOP_REQUESTED)
        sm.evaluate(12.2)
        self.assertEqual(sm.state, sm.LOCKED)
        self.assertIn("静止条件", sm.last_reason)

    def test_missing_speed_data_locks_with_speed_reason(self):
        sm = make_sm()
        sm.request_stop(10.0, "x")
        sm.add_stop_result(10.05, True)
        t = 10.1
        while t <= 10.6:
            feed(sm, t, speed=None)
            t += 0.05
        sm.evaluate(12.2)
        self.assertEqual(sm.state, sm.LOCKED)
        self.assertIn("关节速度", sm.last_reason)

    def test_no_feedback_locks(self):
        sm = make_sm()
        sm.request_stop(10.0, "x")
        sm.add_stop_result(10.05, True)
        sm.evaluate(12.2)
        self.assertEqual(sm.state, sm.LOCKED)
        self.assertIn("无新鲜反馈", sm.last_reason)

    def test_stale_feedback_not_confirmed(self):
        sm = make_sm()
        sm.request_stop(10.0, "x")
        sm.add_stop_result(10.05, True)
        t = 10.1
        while t <= 10.6:
            feed(sm, t)
            t += 0.05
        sm.evaluate(11.2)  # 反馈已过期 0.6s > 0.5s：不确认
        self.assertEqual(sm.state, sm.STOP_REQUESTED)
        sm.evaluate(12.2)
        self.assertEqual(sm.state, sm.LOCKED)

    def test_feedback_arriving_before_request_ignored(self):
        sm = make_sm()
        feed(sm, 9.0)
        sm.request_stop(10.0, "x")
        feed(sm, 9.5)
        self.assertEqual(len(sm.pos_samples), 0)
        self.assertEqual(len(sm.spd_samples), 0)

    def test_dry_run_stops_without_confirmation(self):
        sm = make_sm(dry_run=True)
        sm.request_stop(10.0, "松开")
        self.assertEqual(sm.state, sm.STOPPED)
        self.assertIn("只读模式", sm.last_reason)

    def test_nonfinite_feedback_ignored(self):
        sm = make_sm()
        sm.request_stop(10.0, "x")
        sm.add_position_sample(10.1, [0.3, float("nan"), 0.4])  # 位置非有限：忽略
        sm.add_speed_sample(10.1, [0.0] * 6)
        self.assertEqual(len(sm.pos_samples), 0)
        self.assertEqual(len(sm.spd_samples), 1)
        sm.add_position_sample(10.2, [0.3, 0.0, 0.4])
        sm.add_speed_sample(10.2, [float("inf")] * 6)  # 速度非有限：记缺失
        self.assertEqual(len(sm.pos_samples), 1)
        self.assertEqual(len(sm.spd_samples), 2)
        self.assertIsNone(sm.spd_samples[-1][1])


class PlanTickTests(unittest.TestCase):
    """节拍规划：首帧限制、同批次换锚、停止禁发、偏差检查、在途纪律。"""

    def _setup(self, dry_run=True, enable_motion=False):
        sm = make_sm(dry_run=dry_run)
        io = FakeIO(enable_motion=enable_motion)
        state = make_state()
        cfg = make_cfg()
        planner = make_planner()
        tracker = rc.InFlightTracker(result_timeout_s=2.0)
        return sm, io, state, cfg, planner, tracker

    def test_first_frame_target_equals_feedback_exactly(self):
        sm, io, state, cfg, planner, tracker = self._setup()
        set_robot(state, [0.3, 0.0, 0.4], 10.0)
        send(state, make_packet(0, control=False), 10.0)
        send(state, make_packet(1, control=True, pos=(0.5, 0.0, 0.0)), 10.1)
        r = rc.plan_tick(sm, io, state.snapshot(10.15), 10.15, cfg, planner, tracker)
        self.assertEqual(sm.state, sm.ACTIVE)
        self.assertEqual(r["msg"]["target"], [0.3, 0.0, 0.4])
        self.assertFalse(r["stop_publish"])

    def test_first_frame_limit_enforced_when_feedback_moved(self):
        sm, io, state, cfg, planner, tracker = self._setup()
        set_robot(state, [0.3, 0.0, 0.4], 10.0)
        send(state, make_packet(0, control=False), 10.0)
        send(state, make_packet(1, control=True, pos=(0.5, 0.0, 0.0)), 10.1)
        set_robot(state, [0.31, 0.0, 0.4], 10.2)  # 反馈在捏合后移动 1cm
        r = rc.plan_tick(sm, io, state.snapshot(10.2), 10.2, cfg, planner, tracker)
        # 空基准用反馈播种：目标 = 反馈向锚点方向走 max_step，绝不跳变
        self.assertAlmostEqual(r["msg"]["target"][0], 0.307, places=12)
        self.assertEqual(r["msg"]["target"][1:], [0.0, 0.4])

    def test_continued_frames_clamped_and_step_limited(self):
        sm, io, state, cfg, planner, tracker = self._setup()
        set_robot(state, [0.3, 0.0, 0.4], 10.0)
        send(state, make_packet(0, control=False), 10.0)
        send(state, make_packet(1, control=True, pos=(0.5, 0.0, 0.0)), 10.1)
        rc.plan_tick(sm, io, state.snapshot(10.15), 10.15, cfg, planner, tracker)
        send(state, make_packet(2, control=True, pos=(2.5, 0.0, 0.0)), 10.2)
        r = rc.plan_tick(sm, io, state.snapshot(10.25), 10.25, cfg, planner, tracker)
        # 原始 0.1m 位移：先被 clamp_radius 压到 0.05，再被 limit_step 压到 0.003
        self.assertAlmostEqual(r["msg"]["target"][0], 0.303, places=12)
        self.assertEqual(r["msg"]["target"][1:], [0.0, 0.4])

    def test_same_batch_release_repinch_triggers_stop_then_recover(self):
        sm, io, state, cfg, planner, tracker = self._setup()
        set_robot(state, [0.3, 0.0, 0.4], 10.0)
        send(state, make_packet(0, control=False), 10.0)
        send(state, make_packet(1, control=True, pos=(0.5, 0.0, 0.0)), 10.1)
        rc.plan_tick(sm, io, state.snapshot(10.15), 10.15, cfg, planner, tracker)
        send(state, make_packet(2, control=True, pos=(2.5, 0.0, 0.0)), 10.2)
        rc.plan_tick(sm, io, state.snapshot(10.25), 10.25, cfg, planner, tracker)
        set_robot(state, [0.32, 0.0, 0.4], 10.3)
        # 同一批内：松开 + 重新捏合 → 松开不得被随后的捏合覆盖，必须进入停止流程
        send(state, make_packet(3, control=False), 10.3)
        send(state, make_packet(4, control=True, pos=(0.5, 0.0, 0.0)), 10.32)
        r = rc.plan_tick(sm, io, state.snapshot(10.35), 10.35, cfg, planner, tracker)
        self.assertTrue(r["stop_publish"])
        self.assertIsNone(r["msg"])
        self.assertIsNotNone(r["force_pause"])
        self.assertEqual(sm.state, sm.STOPPED)  # dry-run 直接确认
        state.pause(r["force_pause"])
        # 停止流程完成后，重新松开才能恢复
        set_robot(state, [0.32, 0.0, 0.4], 10.45)
        send(state, make_packet(5, control=False), 10.45)
        r = rc.plan_tick(sm, io, state.snapshot(10.5), 10.5, cfg, planner, tracker)
        self.assertEqual(sm.state, sm.IDLE)
        # 再捏合：首目标 == 新反馈
        set_robot(state, [0.32, 0.0, 0.4], 10.55)
        send(state, make_packet(6, control=True, pos=(0.5, 0.0, 0.0)), 10.55)
        r = rc.plan_tick(sm, io, state.snapshot(10.6), 10.6, cfg, planner, tracker)
        self.assertEqual(sm.state, sm.ACTIVE)
        self.assertEqual(r["msg"]["target"], [0.32, 0.0, 0.4])

    def test_cross_tick_release_stops_and_requires_new_release(self):
        sm, io, state, cfg, planner, tracker = self._setup()
        set_robot(state, [0.3, 0.0, 0.4], 10.0)
        send(state, make_packet(0, control=False), 10.0)
        send(state, make_packet(1, control=True, pos=(0.5, 0.0, 0.0)), 10.1)
        rc.plan_tick(sm, io, state.snapshot(10.15), 10.15, cfg, planner, tracker)
        # 跨拍松开：触发停止（dry-run 直接 STOPPED）
        send(state, make_packet(2, control=False), 10.2)
        r = rc.plan_tick(sm, io, state.snapshot(10.25), 10.25, cfg, planner, tracker)
        self.assertTrue(r["stop_publish"])
        self.assertIsNone(r["msg"])
        self.assertEqual(sm.state, sm.STOPPED)
        # 未重新松开就捏合：禁止
        set_robot(state, [0.3, 0.0, 0.4], 10.3)
        send(state, make_packet(3, control=True, pos=(0.5, 0.0, 0.0)), 10.3)
        r = rc.plan_tick(sm, io, state.snapshot(10.35), 10.35, cfg, planner, tracker)
        self.assertIsNone(r["msg"])
        self.assertEqual(sm.state, sm.STOPPED)
        # 重新松开 → 就绪
        set_robot(state, [0.3, 0.0, 0.4], 10.4)
        send(state, make_packet(4, control=False), 10.4)
        r = rc.plan_tick(sm, io, state.snapshot(10.45), 10.45, cfg, planner, tracker)
        self.assertIsNone(r["msg"])
        self.assertEqual(sm.state, sm.IDLE)
        # 再捏合 → 恢复，首目标 == 反馈（持续刷新反馈，避免 robot 间隔暂停）
        set_robot(state, [0.3, 0.0, 0.4], 10.5)
        send(state, make_packet(5, control=True, pos=(0.5, 0.0, 0.0)), 10.5)
        r = rc.plan_tick(sm, io, state.snapshot(10.55), 10.55, cfg, planner, tracker)
        self.assertEqual(sm.state, sm.ACTIVE)
        self.assertEqual(r["msg"]["target"], [0.3, 0.0, 0.4])

    def test_motion_forbidden_during_stop_requested(self):
        sm, io, state, cfg, planner, tracker = self._setup(dry_run=False)
        set_robot(state, [0.3, 0.0, 0.4], 10.0)
        send(state, make_packet(0, control=False), 10.0)
        send(state, make_packet(1, control=True, pos=(0.5, 0.0, 0.0)), 10.1)
        rc.plan_tick(sm, io, state.snapshot(10.15), 10.15, cfg, planner, tracker)
        send(state, make_packet(2, control=False), 10.2)
        r = rc.plan_tick(sm, io, state.snapshot(10.25), 10.25, cfg, planner, tracker)
        self.assertTrue(r["stop_publish"])
        self.assertEqual(sm.state, sm.STOP_REQUESTED)
        # 停止未确认期间，即使头显侧再次激活也禁止命令
        send(state, make_packet(3, control=True, pos=(0.5, 0.0, 0.0)), 10.3)
        r = rc.plan_tick(sm, io, state.snapshot(10.35), 10.35, cfg, planner, tracker)
        self.assertIsNone(r["msg"])
        self.assertEqual(sm.state, sm.STOP_REQUESTED)

    def test_deviation_exceeds_stops_and_pauses(self):
        sm, io, state, cfg, planner, tracker = self._setup()
        cfg.max_feedback_deviation = 1e-6
        set_robot(state, [0.3, 0.0, 0.4], 10.0)
        send(state, make_packet(0, control=False), 10.0)
        send(state, make_packet(1, control=True, pos=(0.5, 0.0, 0.0)), 10.1)
        rc.plan_tick(sm, io, state.snapshot(10.15), 10.15, cfg, planner, tracker)
        send(state, make_packet(2, control=True, pos=(2.5, 0.0, 0.0)), 10.2)
        r = rc.plan_tick(sm, io, state.snapshot(10.25), 10.25, cfg, planner, tracker)
        self.assertIsNone(r["msg"])
        self.assertTrue(r["stop_publish"])
        self.assertIsNotNone(r["force_pause"])
        self.assertEqual(sm.state, sm.STOPPED)  # dry-run 直接确认

    def test_nonfinite_target_stops(self):
        sm, io, state, cfg, planner, tracker = self._setup()
        sm.state = sm.ACTIVE
        snap = make_snap(robot_target_preview_base_m=[float("nan"), 0.0, 0.0])
        r = rc.plan_tick(sm, io, snap, 10.0, cfg, planner, tracker)
        self.assertIsNone(r["msg"])
        self.assertTrue(r["stop_publish"])
        self.assertIsNotNone(r["force_pause"])

    def test_nonfinite_quaternion_stops(self):
        sm, io, state, cfg, planner, tracker = self._setup()
        sm.state = sm.ACTIVE
        snap = make_snap(robot_target_preview_rotation_xyzw=[0.0, 0.0, 0.0,
                                                            float("inf")])
        r = rc.plan_tick(sm, io, snap, 10.0, cfg, planner, tracker)
        self.assertIsNone(r["msg"])
        self.assertTrue(r["stop_publish"])

    def test_missing_feedback_stops(self):
        sm, io, state, cfg, planner, tracker = self._setup()
        sm.state = sm.ACTIVE
        snap = make_snap(robot_feedback=None)
        r = rc.plan_tick(sm, io, snap, 10.0, cfg, planner, tracker)
        self.assertIsNone(r["msg"])
        self.assertTrue(r["stop_publish"])

    def test_rate_limit_skips_within_period(self):
        sm, io, state, cfg, planner, tracker = self._setup()
        cfg.period = 10.0
        set_robot(state, [0.3, 0.0, 0.4], 10.0)
        send(state, make_packet(0, control=False), 10.0)
        send(state, make_packet(1, control=True, pos=(0.5, 0.0, 0.0)), 10.1)
        r1 = rc.plan_tick(sm, io, state.snapshot(10.15), 10.15, cfg, planner, tracker)
        self.assertIsNotNone(r1["msg"])
        send(state, make_packet(2, control=True, pos=(2.5, 0.0, 0.0)), 10.2)
        r2 = rc.plan_tick(sm, io, state.snapshot(10.25), 10.25, cfg, planner, tracker)
        self.assertIsNone(r2["msg"])
        self.assertFalse(r2["stop_publish"])
        self.assertEqual(sm.state, sm.ACTIVE)

    def test_plan_tick_rejects_none_baseline_via_limit_step(self):
        # 空基准路径在任何时刻都不该到达 limit_step；直接调用必须抛错。
        with self.assertRaises(ValueError):
            rc.limit_step([0.3, 0.0, 0.4], None, 0.003)


class ExitHandlingTests(unittest.TestCase):
    def setUp(self):
        self._orig_signal = rc.signal
        self.fake_signal = FakeSignalModule()
        rc.signal = self.fake_signal

    def tearDown(self):
        rc.signal = self._orig_signal

    def test_signal_handlers_registered_and_set_flag(self):
        ctrl = rc.ExitController()
        rc.install_exit_handlers(ctrl)
        self.assertIn(self.fake_signal.SIGINT, self.fake_signal.handlers)
        self.assertIn(self.fake_signal.SIGTERM, self.fake_signal.handlers)
        self.assertFalse(ctrl.requested)
        self.fake_signal.handlers[self.fake_signal.SIGTERM](15, None)
        self.assertTrue(ctrl.requested)
        self.assertEqual(ctrl.signum, 15)

    def test_finalize_dry_run_never_publishes(self):
        sm = make_sm(dry_run=True)
        sm.note_motion_published()
        io = FakeIO(enable_motion=True)
        clock = FakeClock(10.0)
        lines = rc.finalize_stop(sm, io, now_fn=clock, spin_once=lambda: None,
                                 stop_confirm_timeout_s=2.0)
        self.assertEqual(io.stop_publish_count, 0)
        self.assertTrue(any("只读模式" in line for line in lines))

    def test_finalize_never_moved_no_stop(self):
        sm = make_sm()
        io = FakeIO(enable_motion=True)
        clock = FakeClock(10.0)
        lines = rc.finalize_stop(sm, io, now_fn=clock, spin_once=lambda: None,
                                 stop_confirm_timeout_s=2.0)
        self.assertEqual(io.stop_publish_count, 0)
        self.assertTrue(any("未发布过运动命令" in line for line in lines))

    def test_finalize_context_invalid_reports_unable(self):
        sm = make_sm()
        sm.note_motion_published()
        sm.state = sm.ACTIVE
        io = FakeIO(enable_motion=True)
        io.context_valid = False
        clock = FakeClock(10.0)
        lines = rc.finalize_stop(sm, io, now_fn=clock, spin_once=lambda: None,
                                 stop_confirm_timeout_s=2.0)
        self.assertEqual(io.stop_publish_count, 0)
        self.assertTrue(any("无法确认停止" in line for line in lines))

    def test_finalize_publishes_stop_and_confirms(self):
        sm = make_sm()
        sm.note_motion_published()
        sm.state = sm.ACTIVE
        io = FakeIO(enable_motion=True)
        clock = FakeClock(10.0)

        def spin():
            clock.advance(0.05)
            now = clock()
            io.stop_result = (now, True)
            io.set_pose(now, [0.3, 0.0, 0.4])
            io.set_speed(now, [0.0] * 6)

        lines = rc.finalize_stop(sm, io, now_fn=clock, spin_once=spin,
                                 stop_confirm_timeout_s=2.0)
        self.assertEqual(io.stop_publish_count, 1)
        self.assertEqual(sm.state, sm.STOPPED)
        self.assertTrue(any("停止已确认" in line for line in lines))

    def test_finalize_stop_result_false_reports_unable(self):
        sm = make_sm()
        sm.note_motion_published()
        sm.state = sm.ACTIVE
        io = FakeIO(enable_motion=True)
        clock = FakeClock(10.0)

        def spin():
            clock.advance(0.05)
            io.stop_result = (clock(), False)

        lines = rc.finalize_stop(sm, io, now_fn=clock, spin_once=spin,
                                 stop_confirm_timeout_s=2.0)
        self.assertEqual(io.stop_publish_count, 1)
        self.assertEqual(sm.state, sm.LOCKED)
        self.assertTrue(any("无法确认停止" in line for line in lines))

    def test_finalize_timeout_reports_unable(self):
        sm = make_sm()
        sm.note_motion_published()
        sm.state = sm.ACTIVE
        io = FakeIO(enable_motion=True)
        clock = FakeClock(10.0)

        def spin():
            clock.advance(0.1)  # 无任何反馈

        lines = rc.finalize_stop(sm, io, now_fn=clock, spin_once=spin,
                                 stop_confirm_timeout_s=0.3)
        self.assertEqual(io.stop_publish_count, 1)
        self.assertEqual(sm.state, sm.LOCKED)
        self.assertTrue(any("无法确认" in line for line in lines))

    def test_finalize_spin_exception_reports_unable(self):
        sm = make_sm()
        sm.note_motion_published()
        sm.state = sm.ACTIVE
        io = FakeIO(enable_motion=True)
        clock = FakeClock(10.0)

        def spin():
            raise RuntimeError("context died")

        lines = rc.finalize_stop(sm, io, now_fn=clock, spin_once=spin,
                                 stop_confirm_timeout_s=2.0)
        self.assertEqual(io.stop_publish_count, 1)
        self.assertTrue(any("无法确认停止" in line for line in lines))


class FakeSignalModule:
    SIGINT = 2
    SIGTERM = 15

    def __init__(self):
        self.handlers = {}

    def signal(self, signum, handler):
        self.handlers[signum] = handler


class ReadOnlyGateTests(unittest.TestCase):
    def test_gated_motion_skips_readonly(self):
        io = FakeIO(enable_motion=False)
        self.assertEqual(rc.gated_publish_motion(io, {"target": [0, 0, 0]}),
                         "skipped_readonly")
        self.assertEqual(io.motion_published, [])

    def test_gated_stop_skips_readonly(self):
        io = FakeIO(enable_motion=False)
        self.assertEqual(rc.gated_publish_stop(io), "skipped_readonly")
        self.assertEqual(io.stop_publish_count, 0)

    def test_gated_publish_when_enabled(self):
        io = FakeIO(enable_motion=True)
        self.assertEqual(rc.gated_publish_motion(io, {"target": [0, 0, 0]}),
                         "published")
        self.assertEqual(len(io.motion_published), 1)
        self.assertEqual(rc.gated_publish_stop(io), "published")
        self.assertEqual(io.stop_publish_count, 1)

    def test_readonly_full_flow_never_publishes(self):
        sm = make_sm(dry_run=True)
        io = FakeIO(enable_motion=False)
        state = make_state()
        cfg = make_cfg()
        planner = make_planner()
        tracker = rc.InFlightTracker(result_timeout_s=2.0)
        set_robot(state, [0.3, 0.0, 0.4], 10.0)
        send(state, make_packet(0, control=False), 10.0)
        send(state, make_packet(1, control=True, pos=(0.5, 0.0, 0.0)), 10.1)
        r = rc.plan_tick(sm, io, state.snapshot(10.15), 10.15, cfg, planner, tracker)
        self.assertIsNotNone(r["msg"])
        self.assertEqual(rc.gated_publish_motion(io, r["msg"]),
                         "skipped_readonly")
        send(state, make_packet(2, control=False), 10.2)
        r = rc.plan_tick(sm, io, state.snapshot(10.25), 10.25, cfg, planner, tracker)
        self.assertTrue(r["stop_publish"])
        self.assertEqual(rc.gated_publish_stop(io), "skipped_readonly")
        self.assertEqual(io.motion_published, [])
        self.assertEqual(io.stop_publish_count, 0)


class InFlightTrackerTests(unittest.TestCase):
    """在途纪律：一次最多一个在途运动命令（全部离线，无真实发布）。"""

    def test_single_in_flight_enforced(self):
        tr = rc.InFlightTracker(2.0)
        tr.on_publish(10.0)
        with self.assertRaises(RuntimeError):
            tr.on_publish(10.1)

    def test_old_result_not_bound(self):
        tr = rc.InFlightTracker(2.0)
        tr.on_publish(10.0)
        tr.on_result(9.5, True)      # 早于发布：忽略
        self.assertTrue(tr.in_flight)
        self.assertIsNone(tr.last_outcome)
        tr.on_result(10.0, True)     # 恰在发布时刻：忽略
        self.assertTrue(tr.in_flight)

    def test_reached_result_unblocks(self):
        tr = rc.InFlightTracker(2.0)
        tr.on_publish(10.0)
        tr.on_result(10.1, True)
        self.assertFalse(tr.in_flight)
        self.assertFalse(tr.blocked)
        self.assertEqual(tr.last_outcome, "reached")
        tr.on_publish(10.2)          # 状态明确后才允许下一条

    def test_false_result_blocks(self):
        tr = rc.InFlightTracker(2.0)
        tr.on_publish(10.0)
        tr.on_result(10.1, False)
        self.assertTrue(tr.blocked)
        self.assertEqual(tr.last_outcome, "failed_or_interrupted")
        with self.assertRaises(RuntimeError):
            tr.on_publish(10.2)      # 禁止自动恢复

    def test_timeout_unknown_blocks(self):
        tr = rc.InFlightTracker(2.0)
        tr.on_publish(10.0)
        tr.check(12.0)
        self.assertTrue(tr.blocked)
        self.assertEqual(tr.last_outcome, "unknown")

    def test_late_result_after_timeout_not_accepted(self):
        tr = rc.InFlightTracker(2.0)
        tr.on_publish(10.0)
        tr.check(12.0)               # unknown → 锁定
        tr.on_result(12.1, True)     # 在途已解除：迟到的结果不被绑定
        self.assertTrue(tr.blocked)

    def test_blocked_is_permanent_no_reset_path(self):
        # 故障锁定为进程级：无 reset 接口，松开→再捏合不能解除
        tr = rc.InFlightTracker(2.0)
        tr.on_publish(10.0)
        tr.check(12.0)
        self.assertTrue(tr.blocked)
        self.assertFalse(hasattr(tr, "reset"))
        with self.assertRaises(RuntimeError):
            tr.on_publish(12.1)


class MotionGateTests(unittest.TestCase):
    """真实运动入口硬性门控：--enable-motion 报错退出且不创建任何发布器。"""

    def test_entry_allowed_rejects_enable_motion(self):
        allowed, reason = rc.motion_entry_allowed(True)
        self.assertFalse(allowed)
        self.assertIn("锁定", reason)

    def test_entry_allowed_readonly(self):
        allowed, _ = rc.motion_entry_allowed(False)
        self.assertTrue(allowed)

    def test_main_enable_motion_errors_before_ros(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = rc.main(["--enable-motion"])
        self.assertEqual(code, 3)
        self.assertIn("锁定", stderr.getvalue())
        # 本测试环境无 rclpy/rm_ros_interfaces：若门控之后才导入 ROS 会走
        # ImportError 分支返回 1；返回 3 证明在创建节点/发布器之前已退出，
        # 该路径不会创建运动发布器或发送任何消息。


class PlanTickInFlightTests(unittest.TestCase):
    """plan_tick 与在途纪律的联动（模拟对象，无真实发布）。"""

    def _setup(self, enable_motion=True):
        sm = make_sm(dry_run=False)
        io = FakeIO(enable_motion=enable_motion)
        state = make_state()
        cfg = make_cfg()
        planner = make_planner()
        tracker = rc.InFlightTracker(result_timeout_s=2.0)
        return sm, io, state, cfg, planner, tracker

    def test_in_flight_blocks_next_command(self):
        sm, io, state, cfg, planner, tracker = self._setup()
        sm.state = sm.ACTIVE
        tracker.on_publish(10.0)
        r = rc.plan_tick(sm, io, make_snap(), 10.1, cfg, planner, tracker)
        self.assertIsNone(r["msg"])          # 上一条未明确前不发布下一条
        self.assertFalse(r["stop_publish"])

    def test_result_reached_allows_next_command(self):
        sm, io, state, cfg, planner, tracker = self._setup()
        sm.state = sm.ACTIVE
        tracker.on_publish(10.0)
        io.movel_result = (10.1, True)
        r = rc.plan_tick(sm, io, make_snap(), 10.2, cfg, planner, tracker)
        self.assertIsNotNone(r["msg"])       # 到达明确后允许下一条

    def test_unknown_result_stops_and_locks(self):
        sm, io, state, cfg, planner, tracker = self._setup()
        sm.state = sm.ACTIVE
        tracker.on_publish(10.0)
        r = rc.plan_tick(sm, io, make_snap(), 13.0, cfg, planner, tracker)
        self.assertIsNone(r["msg"])
        self.assertTrue(r["stop_publish"])
        self.assertIsNotNone(r["force_pause"])
        self.assertEqual(sm.state, sm.STOP_REQUESTED)

    def test_false_result_blocks_next_command(self):
        sm, io, state, cfg, planner, tracker = self._setup()
        sm.state = sm.ACTIVE
        tracker.on_publish(10.0)
        io.movel_result = (10.1, False)      # 被停止中断/失败
        r = rc.plan_tick(sm, io, make_snap(), 10.2, cfg, planner, tracker)
        self.assertIsNone(r["msg"])
        self.assertTrue(r["stop_publish"])
        self.assertIsNotNone(r["force_pause"])

    def test_release_pinch_does_not_reset_tracker(self):
        sm, io, state, cfg, planner, tracker = self._setup()
        sm.state = sm.ACTIVE
        tracker.on_publish(10.0)
        r = rc.plan_tick(sm, io, make_snap(), 13.0, cfg, planner, tracker)
        self.assertTrue(r["stop_publish"])
        self.assertTrue(tracker.blocked)
        self.assertEqual(sm.state, sm.STOP_REQUESTED)
        # 确认停止：结果 + 每拍更新两路反馈（只消费新样本）
        io.stop_result = (13.05, True)
        t = 13.1
        while t <= 13.4:
            io.set_pose(t, [0.3, 0.0, 0.4])
            io.set_speed(t, [0.0] * 6)
            rc.plan_tick(sm, io, make_snap(active=False, release_required=True),
                         t, cfg, planner, tracker)
            t += 0.05
        self.assertEqual(sm.state, sm.LOCKED)  # STOPPED 后旧在途结果未知 → 转 LOCKED
        self.assertIn("锁定", sm.last_reason)
        # 显式松开边沿：故障锁定不得被手势解除
        r = rc.plan_tick(sm, io, make_snap(active=False, release_required=True,
                                           release_count=1),
                         13.45, cfg, planner, tracker)
        self.assertEqual(sm.state, sm.LOCKED)
        # 再捏合：仍锁定，不产生目标
        r = rc.plan_tick(sm, io, make_snap(release_count=1), 13.5,
                         cfg, planner, tracker)
        self.assertEqual(sm.state, sm.LOCKED)
        self.assertIsNone(r["msg"])
        self.assertTrue(tracker.blocked)
        self.assertIn("锁定", sm.last_reason)

    def test_delayed_result_does_not_unlock_or_trigger_next(self):
        sm, io, state, cfg, planner, tracker = self._setup()
        sm.state = sm.ACTIVE
        tracker.on_publish(10.0)
        rc.plan_tick(sm, io, make_snap(), 13.0, cfg, planner, tracker)  # 超时锁定
        self.assertTrue(tracker.blocked)
        io.movel_result = (13.1, True)   # 延迟到达的成功结果
        r = rc.plan_tick(sm, io, make_snap(), 13.2, cfg, planner, tracker)
        self.assertIsNone(r["msg"])      # 不得解锁，不得触发下一条
        self.assertTrue(tracker.blocked)
        self.assertEqual(sm.state, sm.STOP_REQUESTED)

    def test_false_result_lock_survives_release_pinch(self):
        sm, io, state, cfg, planner, tracker = self._setup()
        sm.state = sm.ACTIVE
        tracker.on_publish(10.0)
        io.movel_result = (10.1, False)  # 被停止中断/失败
        r = rc.plan_tick(sm, io, make_snap(), 10.2, cfg, planner, tracker)
        self.assertTrue(r["stop_publish"])
        self.assertTrue(tracker.blocked)
        # 确认停止：结果 + 每拍更新两路反馈（只消费新样本）
        io.stop_result = (10.3, True)
        t = 10.35
        while t <= 10.65:
            io.set_pose(t, [0.3, 0.0, 0.4])
            io.set_speed(t, [0.0] * 6)
            rc.plan_tick(sm, io, make_snap(active=False, release_required=True),
                         t, cfg, planner, tracker)
            t += 0.05
        self.assertEqual(sm.state, sm.LOCKED)  # STOPPED 后旧在途结果失败 → 转 LOCKED
        # 松开→再捏合：故障锁定仍然保持
        rc.plan_tick(sm, io, make_snap(active=False, release_required=True,
                                       release_count=1),
                     10.7, cfg, planner, tracker)
        self.assertEqual(sm.state, sm.LOCKED)
        r = rc.plan_tick(sm, io, make_snap(release_count=1), 10.75,
                         cfg, planner, tracker)
        self.assertIsNone(r["msg"])
        self.assertEqual(sm.state, sm.LOCKED)
        self.assertTrue(tracker.blocked)
        self.assertIn("锁定", sm.last_reason)


class StopWindowContinuityTests(unittest.TestCase):
    """静止窗口的持续时长与连续性（而非仅凑足帧数）。"""

    def test_gap_in_samples_blocks_confirmation(self):
        sm = make_sm()
        sm.request_stop(10.0, "x")
        sm.add_stop_result(10.05, True)
        for t in (10.1, 10.15, 10.2):
            feed(sm, t)
        feed(sm, 10.4)   # 0.2s 断层
        feed(sm, 10.45)
        sm.evaluate(10.5)   # 跨度 0.35s 达标，但窗口不连续：不确认
        self.assertEqual(sm.state, sm.STOP_REQUESTED)
        sm.evaluate(12.2)   # 超时锁定，注明反馈不连续
        self.assertEqual(sm.state, sm.LOCKED)
        self.assertIn("不连续", sm.last_reason)

    def test_window_duration_not_frame_count(self):
        sm = make_sm(still_window_s=0.2)
        sm.request_stop(10.0, "x")
        sm.add_stop_result(10.05, True)
        # 3 帧凑够数量但跨度只有 0.08s：不确认
        for t in (10.1, 10.14, 10.18):
            feed(sm, t)
        sm.evaluate(10.25)
        self.assertEqual(sm.state, sm.STOP_REQUESTED)

    def test_delayed_old_result_limitation_documented(self):
        # 已知局限：无命令编号的 Bool 无法与上一周期的延迟结果区分；
        # 时间门控只能排除请求之前的结果，此用例固化当前行为（请求之后
        # 到达即被接受为辅助证据），实际确认以反馈静止为主。
        sm = make_sm()
        sm.request_stop(10.0, "x")
        sm.add_stop_result(10.05, True)   # 也可能是上一周期的延迟结果
        self.assertEqual(sm.stop_result, (10.05, True))


class RobotIOPublishTests(unittest.TestCase):
    """直接测试真实 RobotIO 方法（模拟发布器，不使用 FakeIO 替身）。"""

    class FakePublisher:
        def __init__(self):
            self.published = []

        def publish(self, msg):
            self.published.append(msg)

    class FakeEmpty:
        pass

    def _io(self, enable=True):
        # 绕过 __init__（其中导入 rclpy），仅注入方法所需属性，
        # 使被测代码是真实的 publish_stop/publish_motion 方法体。
        io = rc.RobotIO.__new__(rc.RobotIO)
        io.enable_motion = enable
        io.stop_pub = self.FakePublisher()
        io.motion_pub = self.FakePublisher()
        io._Empty = self.FakeEmpty
        return io

    def test_publish_stop_uses_stored_empty_class(self):
        # 回归：Empty 曾只在 __init__ 内局部导入，publish_stop 引用时 NameError
        io = self._io()
        io.publish_stop()
        self.assertEqual(len(io.stop_pub.published), 1)
        self.assertIsInstance(io.stop_pub.published[0], self.FakeEmpty)

    def test_publish_motion_forwards(self):
        io = self._io()
        msg = object()
        io.publish_motion(msg)
        self.assertEqual(io.motion_pub.published, [msg])
        self.assertEqual(io.stop_pub.published, [])

    def test_publish_stop_raises_when_readonly(self):
        io = self._io(enable=False)
        with self.assertRaises(RuntimeError):
            io.publish_stop()
        self.assertEqual(io.stop_pub.published, [])

    def test_publish_motion_raises_when_readonly(self):
        io = self._io(enable=False)
        with self.assertRaises(RuntimeError):
            io.publish_motion(object())
        self.assertEqual(io.motion_pub.published, [])

    def test_init_binds_empty_class(self):
        # 回归防护：__init__ 必须把 Empty 绑定到实例属性供 publish_stop 使用
        import inspect
        src = inspect.getsource(rc.RobotIO.__init__)
        self.assertIn("self._Empty = Empty", src)


class RobotIOSpeedCallbackTests(unittest.TestCase):
    """RobotIO._on_speed 的真实方法体：严格 6 轴校验。"""

    class FakeMsg:
        pass

    def _io(self):
        io = rc.RobotIO.__new__(rc.RobotIO)
        io.speed = None
        io.speed_seq = 0
        return io

    def test_six_axes_accepted(self):
        io = self._io()
        msg = self.FakeMsg()
        msg.joint_speed = [0.0, 0.01, 0.0, 0.0, 0.0, 0.05]
        io._on_speed(msg)
        self.assertIsNotNone(io.speed)
        self.assertEqual(len(io.speed[1]), 6)
        self.assertEqual(io.speed_seq, 1)

    def test_five_axes_rejected(self):
        io = self._io()
        msg = self.FakeMsg()
        msg.joint_speed = [0.0] * 5
        io._on_speed(msg)
        self.assertIsNone(io.speed)

    def test_seven_axes_rejected(self):
        io = self._io()
        msg = self.FakeMsg()
        msg.joint_speed = [0.0] * 7
        io._on_speed(msg)
        self.assertIsNone(io.speed)

    def test_nonfinite_rejected(self):
        io = self._io()
        msg = self.FakeMsg()
        msg.joint_speed = [0.0, float("nan"), 0.0, 0.0, 0.0, 0.0]
        io._on_speed(msg)
        self.assertIsNone(io.speed)


class StopObserveRegressionTests(unittest.TestCase):
    """observe 的旧错误回归：缓存反馈重复当作新样本、请求前样本、双通道独立。"""

    def test_cached_feedback_not_repeated_as_new_samples(self):
        # 复现报告：仅停止请求前的一帧零速度+位置，循环 observe 会错误确认。
        sm = make_sm()
        io = FakeIO()
        io.set_pose(9.5, [0.3, 0.0, 0.4])
        io.set_speed(9.5, [0.0] * 6)
        io.stop_result = (10.05, True)
        sm.request_stop(10.0, "x")
        for i in range(50):
            sm.observe(io)                 # 主循环反复调用，缓存无新样本
            sm.evaluate(10.1 + i * 0.02)
        self.assertEqual(sm.state, sm.STOP_REQUESTED)  # 绝不误确认
        sm.evaluate(12.2)
        self.assertEqual(sm.state, sm.LOCKED)
        self.assertIn("无新鲜反馈", sm.last_reason)

    def test_only_new_samples_consumed_by_sequence(self):
        sm = make_sm()
        io = FakeIO()
        io.stop_result = (10.05, True)
        sm.request_stop(10.0, "x")
        io.set_pose(10.1, [0.3, 0.0, 0.4])
        io.set_speed(10.1, [0.0] * 6)
        for _ in range(10):
            sm.observe(io)                 # 同一缓存反复 observe
        self.assertEqual(len(sm.pos_samples), 1)
        self.assertEqual(len(sm.spd_samples), 1)
        # 只更新速度通道：仅速度侧增长
        for i in range(10):
            io.set_speed(10.15 + i * 0.02, [0.0] * 6)
            sm.observe(io)
        self.assertEqual(len(sm.pos_samples), 1)
        self.assertEqual(len(sm.spd_samples), 11)

    def test_position_and_speed_channels_checked_independently(self):
        sm = make_sm()
        io = FakeIO()
        io.stop_result = (10.05, True)
        sm.request_stop(10.0, "x")
        io.set_pose(10.1, [0.3, 0.0, 0.4])
        io.set_speed(10.1, [0.0] * 6)
        sm.observe(io)
        for i in range(10):
            io.set_speed(10.15 + i * 0.02, [0.0] * 6)  # 只有速度在更新
            sm.observe(io)
            sm.evaluate(10.2 + i * 0.02)
        self.assertEqual(sm.state, sm.STOP_REQUESTED)  # 位置窗口不足：不确认
        sm.evaluate(12.2)
        self.assertEqual(sm.state, sm.LOCKED)
        self.assertIn("位置", sm.last_reason)

    def test_sample_keeps_own_receive_time_not_loop_time(self):
        # 样本必须携带自己的接收时间：主循环时间前进不得被当作采样时间
        sm = make_sm()
        io = FakeIO()
        io.stop_result = (10.05, True)
        sm.request_stop(10.0, "x")
        io.set_pose(10.1, [0.3, 0.0, 0.4])
        io.set_speed(10.1, [0.0] * 6)
        for i in range(30):
            sm.observe(io)                 # 缓存时间停在 10.1
            sm.evaluate(10.2 + i * 0.05)   # 主循环时间不断前进
        # 只有一个样本，跨度 0 < 0.2s：不得因主循环时间前进而确认
        self.assertEqual(sm.state, sm.STOP_REQUESTED)

    def test_speed_requires_exactly_six_axes(self):
        sm = make_sm()
        sm.request_stop(10.0, "x")
        sm.add_stop_result(10.05, True)
        t = 10.1
        while t <= 10.6:
            sm.add_position_sample(t, [0.3, 0.0, 0.4])
            sm.add_speed_sample(t, [0.0] * 5)   # 5 轴 → 缺失/非法
            t += 0.05
        sm.evaluate(10.65)
        self.assertEqual(sm.state, sm.STOP_REQUESTED)
        sm.evaluate(12.2)
        self.assertEqual(sm.state, sm.LOCKED)
        self.assertIn("6 轴", sm.last_reason)

    def test_still_tail_after_deceleration_confirms(self):
        # 减速结束后的连续静止区间应可确认（早期运动样本不阻碍）
        sm = make_sm()
        sm.request_stop(10.0, "x")
        sm.add_stop_result(10.05, True)
        t = 10.1
        while t <= 10.25:  # 减速段：速度 0.5 RPM、位置漂移
            sm.add_position_sample(t, [0.3 + (t - 10.1) * 0.01, 0.0, 0.4])
            sm.add_speed_sample(t, [0.5] * 6)
            t += 0.05
        while t <= 10.5:   # 静止段：速度 0、位置恒定
            sm.add_position_sample(t, [0.3, 0.0, 0.4])
            sm.add_speed_sample(t, [0.0] * 6)
            t += 0.05
        sm.evaluate(10.55)
        self.assertEqual(sm.state, sm.STOPPED)

    def test_samples_bounded_not_accumulated_forever(self):
        sm = make_sm()
        sm.request_stop(10.0, "x")
        t = 10.1
        while t <= 20.0:  # 远超确认超时窗口
            sm.add_position_sample(t, [0.3, 0.0, 0.4])
            sm.add_speed_sample(t, [0.0] * 6)
            t += 0.05
        self.assertLess(len(sm.pos_samples), 100)  # 只保留超时窗口内历史
        self.assertLess(len(sm.spd_samples), 100)


class PlanTickLockedRegressionTests(unittest.TestCase):
    """plan_tick 旧错误回归：LOCKED 被松开重捏合解除并产生目标。"""

    def _setup(self):
        sm = make_sm(dry_run=False)
        io = FakeIO(enable_motion=False)
        state = make_state()
        cfg = make_cfg()
        planner = make_planner()
        tracker = rc.InFlightTracker(result_timeout_s=2.0)
        return sm, io, state, cfg, planner, tracker

    def test_locked_not_unlocked_by_release_pinch(self):
        # 复现报告：move_stop_result=false 后，松开重捏合仍进入 ACTIVE 并产生目标
        sm, io, state, cfg, planner, tracker = self._setup()
        sm.state = sm.ACTIVE
        r = rc.plan_tick(sm, io, make_snap(active=False), 10.0,
                         cfg, planner, tracker)
        self.assertTrue(r["stop_publish"])
        self.assertEqual(sm.state, sm.STOP_REQUESTED)
        sm.add_stop_result(10.1, False)      # 停止接口返回失败 → LOCKED
        self.assertEqual(sm.state, sm.LOCKED)
        # 松开
        r = rc.plan_tick(sm, io, make_snap(active=False, release_count=1),
                         10.2, cfg, planner, tracker)
        self.assertEqual(sm.state, sm.LOCKED)
        # 再捏合：不得进入 ACTIVE、不得产生目标
        r = rc.plan_tick(sm, io, make_snap(release_count=1), 10.3,
                         cfg, planner, tracker)
        self.assertIsNone(r["msg"])
        self.assertEqual(sm.state, sm.LOCKED)
        self.assertIn("锁定", sm.last_reason)

    def test_locked_by_confirm_timeout_not_unlocked_by_gestures(self):
        sm, io, state, cfg, planner, tracker = self._setup()
        sm.state = sm.ACTIVE
        r = rc.plan_tick(sm, io, make_snap(active=False), 10.0,
                         cfg, planner, tracker)
        self.assertTrue(r["stop_publish"])
        sm.evaluate(12.2)                    # 确认超时 → LOCKED
        self.assertEqual(sm.state, sm.LOCKED)
        r = rc.plan_tick(sm, io, make_snap(active=False, release_count=1),
                         12.3, cfg, planner, tracker)
        r = rc.plan_tick(sm, io, make_snap(release_count=1), 12.4,
                         cfg, planner, tracker)
        self.assertIsNone(r["msg"])
        self.assertEqual(sm.state, sm.LOCKED)

    def test_stopped_normal_recovery_still_allowed(self):
        # 正常确认的停止（STOPPED）仍可经松开→捏合恢复（dry-run 路径）
        sm = make_sm(dry_run=True)
        io = FakeIO(enable_motion=False)
        state = make_state()
        cfg = make_cfg()
        planner = make_planner()
        tracker = rc.InFlightTracker(result_timeout_s=2.0)
        set_robot(state, [0.3, 0.0, 0.4], 10.0)
        send(state, make_packet(0, control=False), 10.0)
        send(state, make_packet(1, control=True, pos=(0.5, 0.0, 0.0)), 10.1)
        rc.plan_tick(sm, io, state.snapshot(10.15), 10.15, cfg, planner, tracker)
        send(state, make_packet(2, control=False), 10.2)
        rc.plan_tick(sm, io, state.snapshot(10.25), 10.25, cfg, planner, tracker)
        self.assertEqual(sm.state, sm.STOPPED)
        set_robot(state, [0.3, 0.0, 0.4], 10.3)
        send(state, make_packet(3, control=False), 10.3)
        r = rc.plan_tick(sm, io, state.snapshot(10.35), 10.35, cfg, planner, tracker)
        self.assertEqual(sm.state, sm.IDLE)
        send(state, make_packet(4, control=True, pos=(0.5, 0.0, 0.0)), 10.4)
        r = rc.plan_tick(sm, io, state.snapshot(10.45), 10.45, cfg, planner, tracker)
        self.assertEqual(sm.state, sm.ACTIVE)
        self.assertEqual(r["msg"]["target"], [0.3, 0.0, 0.4])


if __name__ == "__main__":
    unittest.main(verbosity=2)
