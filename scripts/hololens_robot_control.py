#!/usr/bin/env python3
"""HoloLens control review v3: real motion remains hard-locked.

Normal motion result timeout: 2s. After a stop request, wait up to 15s
for the outstanding motion result, independently of the 2s physical-stop
confirmation deadline. These bounds are diagnostic policy, not certified
physical stopping times. Repeated stop requests cannot extend deadlines.

Only one outstanding motion is allowed. No next command while awaiting its
result, even if stop acknowledgement and stationary feedback have arrived.
A false Bool motion result cannot distinguish SDK -6 from a fault; it remains
process-locked. Late results cannot unlock a timed-out process.

Release/tracking-loss edges survive UDP batching. Recovery after a successful
stop requires a new release after both stop confirmation and result drainage.
Only known-success results allow this recovery; false/unknown never do.

Stop confirmation requires fresh continuous position and six-joint speed
feedback plus move_stop_result=true. Stop results have no command identifier:
无命令编号的延迟结果无法与上一周期的结果区分; external publishers and
controller-side queued commands remain unresolved. This revision does not
validate coordinate mapping or enable continuous robot control.

SIGINT/SIGTERM request orderly exit with ROS alive for bounded stop/result
waiting. SIGKILL cannot be handled. Dry-run creates no command publishers.
"""

from __future__ import annotations

import argparse
import math
import signal
import socket
import sys
import time
from collections import deque
from pathlib import Path

from hololens_live_preview import Preview, vector_text

MOTION_TOPIC = "/rm_driver/movel_cmd"
STOP_TOPIC = "/rm_driver/move_stop_cmd"
POSE_TOPIC = "/rm_driver/udp_arm_position"
SPEED_TOPIC = "/rm_driver/udp_joint_speed"
STOP_RESULT_TOPIC = "/rm_driver/move_stop_result"
MOVEL_RESULT_TOPIC = "/rm_driver/movel_result"

MOTION_GATE_LOCKED_REASON = (
    "真实运动入口锁定：--enable-motion 已被禁用。movel_cmd 订阅队列"
    "（ParametersQoS 深度 10，MutuallyExclusive 回调组）在 block=1 阻塞回调"
    "期间可能堆积，停止后已排队命令是否恢复执行未在实机核实；在完成核实前"
    "不允许启用真实运动。")


def motion_entry_allowed(enable_motion: bool):
    """队列风险核实前，传入 --enable-motion 一律拒绝（报错退出路径）。"""
    if enable_motion:
        return False, MOTION_GATE_LOCKED_REASON
    return True, ""


# ---- 纯几何限制（无 ROS 依赖，离线可测） ----

def is_finite_number(x) -> bool:
    return (isinstance(x, (int, float)) and not isinstance(x, bool)
            and math.isfinite(x))


def is_finite_vector(value, size: int) -> bool:
    return (isinstance(value, (list, tuple)) and len(value) == size
            and all(is_finite_number(e) for e in value))


def clamp_radius(target, anchor, radius):
    """相对锚点的最大总位移限制。任何输入非有限或 radius 非法 → ValueError。"""
    if not is_finite_vector(target, 3):
        raise ValueError("clamp_radius: target 必须为 3 维有限数值")
    if not is_finite_vector(anchor, 3):
        raise ValueError("clamp_radius: anchor 必须为 3 维有限数值")
    if not (is_finite_number(radius) and radius > 0):
        raise ValueError("clamp_radius: radius 必须为有限正数")
    delta = [target[i] - anchor[i] for i in range(3)]
    length = math.sqrt(sum(v * v for v in delta))
    if length <= radius or length == 0:
        return list(target)
    factor = radius / length
    return [anchor[i] + delta[i] * factor for i in range(3)]


def limit_step(target, previous, max_step):
    """相邻命令步长限制。previous 为空基准时拒绝（ValueError），绝不绕过。"""
    if previous is None:
        raise ValueError("limit_step: previous 为 None（空基准），拒绝绕过限制")
    if not is_finite_vector(target, 3):
        raise ValueError("limit_step: target 必须为 3 维有限数值")
    if not is_finite_vector(previous, 3):
        raise ValueError("limit_step: previous 必须为 3 维有限数值")
    if not (is_finite_number(max_step) and max_step > 0):
        raise ValueError("limit_step: max_step 必须为有限正数")
    delta = [target[i] - previous[i] for i in range(3)]
    length = math.sqrt(sum(v * v for v in delta))
    if length <= max_step or length == 0:
        return list(target)
    factor = max_step / length
    return [previous[i] + delta[i] * factor for i in range(3)]


def check_target_deviation(target, feedback, max_deviation) -> bool:
    """目标与实际反馈偏差检查：超过阈值返回 False（不能只限制相邻命令目标）。"""
    if not is_finite_vector(target, 3) or not is_finite_vector(feedback, 3):
        raise ValueError("check_target_deviation: 目标/反馈必须为 3 维有限数值")
    if not (is_finite_number(max_deviation) and max_deviation > 0):
        raise ValueError("check_target_deviation: max_deviation 必须为有限正数")
    delta = [target[i] - feedback[i] for i in range(3)]
    return math.sqrt(sum(v * v for v in delta)) <= max_deviation


# ---- 停止状态机（纯逻辑，离线可测） ----

class StopStateMachine:
    IDLE = "idle"
    ACTIVE = "active"
    STOP_REQUESTED = "stop_requested"
    STOPPED = "stopped_confirmed"
    LOCKED = "locked"

    def __init__(self, still_window_s=0.2, still_pos_epsilon_m=5e-4,
                 still_speed_rpm=0.1, stop_confirm_timeout_s=2.0,
                 feedback_max_age_s=0.5, max_sample_gap_s=0.05,
                 dry_run=False):
        for value in (still_window_s, still_pos_epsilon_m, still_speed_rpm,
                      stop_confirm_timeout_s, feedback_max_age_s,
                      max_sample_gap_s):
            if not (is_finite_number(value) and value > 0):
                raise ValueError("StopStateMachine 参数必须为有限正数")
        self.state = self.IDLE
        self.dry_run = bool(dry_run)
        self.still_window_s = float(still_window_s)
        self.still_pos_epsilon_m = float(still_pos_epsilon_m)
        self.still_speed_rpm = float(still_speed_rpm)
        self.stop_confirm_timeout_s = float(stop_confirm_timeout_s)
        self.feedback_max_age_s = float(feedback_max_age_s)
        self.max_sample_gap_s = float(max_sample_gap_s)
        self.stop_request_time = None
        self.stop_result = None      # (t, ok)，仅接受 t > stop_request_time
        self.pos_samples = deque()   # (t, position)，仅停止请求之后的新样本
        self.spd_samples = deque()   # (t, speed|None)，仅停止请求之后的新样本
        self._pose_seq_consumed = None   # 已消费的位置回调序号（只消费新样本）
        self._speed_seq_consumed = None  # 已消费的速度回调序号（只消费新样本）
        self.release_seen = False
        self.ever_moved = False
        self.last_reason = "未开始"

    def note_motion_published(self):
        self.ever_moved = True

    def note_release_seen(self):
        self.release_seen = True

    def request_stop(self, now, reason):
        """进入停止流程。返回 True 表示本次需要发布一次停止命令。"""
        if self.state in (self.STOP_REQUESTED, self.STOPPED, self.LOCKED):
            return False
        self.state = self.STOP_REQUESTED
        self.stop_request_time = float(now)
        self.stop_result = None
        self.pos_samples.clear()
        self.spd_samples.clear()
        self._pose_seq_consumed = None
        self._speed_seq_consumed = None
        self.release_seen = False
        self.last_reason = reason
        if self.dry_run:
            self.state = self.STOPPED
            self.last_reason = f"只读模式：未发布任何命令（{reason}）"
        return True

    def add_stop_result(self, t, ok):
        if t is None or ok is None:
            return
        if self.state != self.STOP_REQUESTED:
            return
        if t <= self.stop_request_time:
            return  # 时间门控只能排除请求之前的结果；无命令编号的延迟结果
                    # 无法与上一周期的结果区分，故该结果仅作辅助，确认以反馈为准。
        self.stop_result = (float(t), bool(ok))
        if not ok:
            self.state = self.LOCKED
            self.last_reason = "move_stop_result=false：驱动停止接口返回失败"

    def add_position_sample(self, t, position):
        """记录停止请求之后的新位置样本（请求前样本排除；按序号只消费新样本）。"""
        if self.state != self.STOP_REQUESTED or t < self.stop_request_time:
            return
        if not is_finite_vector(position, 3):
            return
        self._append(self.pos_samples, (float(t), list(position)))

    def add_speed_sample(self, t, speed):
        """记录停止请求之后的新速度样本；严格校验 6 轴，缺失/非法记为 None。"""
        if self.state != self.STOP_REQUESTED or t < self.stop_request_time:
            return
        if (isinstance(speed, (list, tuple)) and len(speed) == 6
                and all(is_finite_number(v) for v in speed)):
            self._append(self.spd_samples, (float(t), list(speed)))
        else:
            self._append(self.spd_samples, (float(t), None))

    def _append(self, samples, item):
        """追加并裁剪：只保留确认超时窗口内的历史（减速后的静止尾部仍可评估）。"""
        samples.append(item)
        horizon = self.stop_confirm_timeout_s + self.still_window_s
        while samples and samples[-1][0] - samples[0][0] > horizon:
            samples.popleft()

    def evaluate(self, now):
        if self.state != self.STOP_REQUESTED:
            return
        if self.stop_result is not None and self.stop_result[1] is False:
            return  # add_stop_result 已进入 LOCKED
        if now - self.stop_request_time >= self.stop_confirm_timeout_s:
            self.state = self.LOCKED
            self.last_reason = self._timeout_reason()
            return
        if self.stop_result is None:
            return  # 没有本次请求对应的 move_stop_result
        # 减速结束后的连续静止尾部：早期减速/漂移样本不阻碍确认，
        # 但要求位置与速度两路各自存在跨度≥窗口、无断层、且静止的尾部区间，
        # 并分别检查两路尾部末样本的新鲜度。
        pos_tail = self._still_tail(self.pos_samples, "pos")
        if pos_tail is None:
            return
        spd_tail = self._still_tail(self.spd_samples, "spd")
        if spd_tail is None:
            return
        t0 = max(pos_tail[0], spd_tail[0])
        t1 = min(pos_tail[1], spd_tail[1])
        if t1 - t0 < self.still_window_s:
            return  # 两路交叠的静止区间不足窗口时长
        if now - t1 > self.feedback_max_age_s:
            return  # 两路尾部末样本均已过期：不确认
        self.state = self.STOPPED
        self.last_reason = ("停止已确认：move_stop_result=true 且停止请求后"
                            "位置与关节速度均存在连续静止窗口")

    def _still_tail(self, samples, kind):
        """最长尾部静止区间的 (first_t, last_t)；不满足则 None。

        kind="pos"：相对尾部首样本位移 < epsilon；
        kind="spd"：全部样本速度非空且 |v| < 阈值。
        尾部内部不允许断层；窗口按持续时长判定。
        """
        samples = list(samples)  # deque 不支持切片
        limit = self.max_sample_gap_s + 1e-9
        for start in range(len(samples)):
            if kind == "pos":
                ref = samples[start][1]
                still_ok = all(
                    all(abs(v - ref[i]) < self.still_pos_epsilon_m
                        for i, v in enumerate(p))
                    for _, p in samples[start:])
            else:
                still_ok = all(
                    s is not None
                    and all(abs(v) < self.still_speed_rpm for v in s)
                    for _, s in samples[start:])
            if not still_ok:
                continue
            if any(samples[i + 1][0] - samples[i][0] > limit
                   for i in range(start, len(samples) - 1)):
                continue
            first_t, last_t = samples[start][0], samples[-1][0]
            if last_t - first_t < self.still_window_s:
                continue
            return (first_t, last_t)
        return None

    def _span_of(self, samples):
        """整通道的时间跨度与连续性（用于超时原因诊断）。"""
        if len(samples) < 2:
            return None
        limit = self.max_sample_gap_s + 1e-9
        if any(samples[i + 1][0] - samples[i][0] > limit
               for i in range(len(samples) - 1)):
            return None
        return (samples[0][0], samples[-1][0])

    def _timeout_reason(self):
        has_pos, has_spd = bool(self.pos_samples), bool(self.spd_samples)
        if not has_pos and not has_spd:
            return "停止确认超时：停止请求后无新鲜反馈，无法确认机械臂静止"
        if has_pos and not has_spd:
            return ("停止确认超时：关节速度反馈缺失（检查 joint_speed_enable / "
                    "udp_joint_speed_state），无法确认机械臂静止")
        if not has_pos and has_spd:
            return "停止确认超时：位置反馈缺失，无法确认机械臂静止"
        if any(s is None for _, s in self.spd_samples):
            return ("停止确认超时：关节速度样本缺失/非法（严格校验 6 轴），"
                    "无法确认机械臂静止")
        pos_span = self._span_of(self.pos_samples)
        spd_span = self._span_of(self.spd_samples)
        if pos_span is None and spd_span is not None:
            return ("停止确认超时：位置反馈更新不足/不连续（静止窗口未被持续"
                    "观测），无法确认机械臂静止")
        if spd_span is None and pos_span is not None:
            return ("停止确认超时：关节速度反馈更新不足/不连续（静止窗口未被"
                    "持续观测），无法确认机械臂静止")
        if pos_span is None and spd_span is None:
            return ("停止确认超时：位置与关节速度反馈更新不足/不连续，"
                    "无法确认机械臂静止")
        return ("停止确认超时：反馈窗口内未满足位置/关节速度静止条件，"
                "无法确认机械臂静止")

    def observe(self, io):
        """从 IO 消费新样本：位置与速度按各自回调序号去重，只消费新样本；
        样本携带各自的接收时间（不得用主循环时间替代）；请求之前的样本
        在 add_* 中被排除。停止结果单独处理（时间门控仅作辅助）。"""
        if getattr(io, "stop_result", None) is not None:
            self.add_stop_result(io.stop_result[0], io.stop_result[1])
        pose = getattr(io, "pose", None)
        pose_seq = getattr(io, "pose_seq", None)
        if pose is not None and pose_seq != self._pose_seq_consumed:
            self._pose_seq_consumed = pose_seq
            self.add_position_sample(pose[0], pose[1])
        spd = getattr(io, "speed", None)
        spd_seq = getattr(io, "speed_seq", None)
        if spd is not None and spd_seq != self._speed_seq_consumed:
            self._speed_seq_consumed = spd_seq
            self.add_speed_sample(spd[0], spd[1])


# ---- 在途纪律：一次最多一个在途运动命令 ----

class InFlightTracker:
    """一次最多一个在途运动命令（在途纪律；真实运动解锁前仅离线验证）。

    规则:
      - 发布新命令前必须无在途且未锁定；发布后置在途；
      - 上一条的 movel_result 到达前不产生下一条（结果按"排除法"绑定：
        无命令编号，仅当有且只有一条在途时才能归属）；
      - 结果 false → 上一条被停止指令中断（rm_movel 返回 -6）或失败 → 锁定；
      - 结果超时未到 → 状态未知 → 锁定；
      - 锁定为进程级：没有自动恢复路径，松开→再捏合不能解除（必须重启本
        控制程序；重启不能保证控制器侧已清除旧命令，须人工确认）；在途已
        解除后到达的延迟结果一律忽略，不得解锁或触发下一条。

    能解决: 本方不再向 driver 的订阅队列堆积多条命令，停止后不存在由本方
    排队的命令可恢复执行；停止时至多中断一条在途命令。
    不能解决: 其他发布者仍可向 /rm_driver/movel_cmd 排队；driver 订阅侧
    QoS 队列深度（ParametersQoS 深度 10）不变；把本发布者 QoS 深度设为 1
    只影响本方出站队列，不是完整解决方案；movel_result 无编号，延迟/丢失
    的结果只能按未知处理（锁定），且不证明机械臂物理静止。
    """

    def __init__(self, result_timeout_s=2.0, stop_result_timeout_s=15.0):
        if not (is_finite_number(result_timeout_s) and result_timeout_s > 0):
            raise ValueError("result_timeout_s 必须为有限正数")
        self.result_timeout_s = float(result_timeout_s)
        if not (is_finite_number(stop_result_timeout_s) and stop_result_timeout_s > 0):
            raise ValueError("stop_result_timeout_s 必须为有限正数")
        self.stop_result_timeout_s = float(stop_result_timeout_s)
        self.stop_requested_at = None
        self.in_flight = False
        self.in_flight_since = None
        self.last_outcome = None    # "reached" | "interrupted" | "unknown"
        self.blocked = False

    def on_publish(self, now):
        if self.in_flight:
            raise RuntimeError("已有在途运动命令，禁止再发布")
        if self.blocked:
            raise RuntimeError("上一命令结果未知/被中断：进程级锁定，禁止自动恢复")
        self.in_flight = True
        self.in_flight_since = float(now)
        self.last_outcome = None
        self.stop_requested_at = None

    def note_stop_requested(self, now):
        """独立计时，不放宽正常运动期限；重复请求不能刷新期限。"""
        self.check(now)
        if self.in_flight and self.stop_requested_at is None:
            self.stop_requested_at = float(now)

    def on_result(self, now, ok):
        # 必须先裁定截止时间；晚到的成功回执也不能解除超时锁定。
        self.check(now)
        if not self.in_flight or self.in_flight_since is None:
            return  # 在途已解除：延迟/旧结果一律忽略，不得解锁或触发下一条
        if now <= self.in_flight_since:
            return  # 早于本次发布的旧结果，不得绑定
        self.in_flight = False
        if ok:
            self.last_outcome = "reached"
        else:
            self.last_outcome = "failed_or_interrupted"
            self.blocked = True

    def check(self, now):
        since = (self.stop_requested_at if self.stop_requested_at is not None
                 else self.in_flight_since)
        timeout = (self.stop_result_timeout_s if self.stop_requested_at is not None
                   else self.result_timeout_s)
        if self.in_flight and now - since >= timeout:
            self.in_flight = False
            self.last_outcome = "unknown"
            self.blocked = True


# ---- 单节拍规划（纯逻辑，离线可测） ----

def plan_tick(sm, io, snap, now, cfg, planner, tracker):
    """一次控制节拍。返回 dict(msg|None, stop_publish:bool, force_pause|None)。

    msg = {"target": [...], "quat": [...]}，由调用方构造 ROS 消息；
    本函数不做任何发布（发布门控在 gated_publish_*，只读模式绝不发布）。
    planner 就地更新: last_sent / last_send / reference / release_count_seen。
    tracker（在途纪律）仅在 io.enable_motion 时生效：上一命令结果未明确前
    不产出新命令；结果未知/被中断则请求停止并锁定。
    """
    out = {"msg": None, "stop_publish": False, "force_pause": None}

    # 松开边沿检测：只有"新出现"的有效松开才算恢复条件（防旧状态误恢复）。
    release_count = snap.get("release_count", -1)
    release_edge = release_count > planner.get("release_count_seen", 0)
    if release_count != planner.get("release_count_seen", -1):
        sm.note_release_seen()
        planner["release_count_seen"] = release_count

    tracking = bool(getattr(io, "enable_motion", False))
    interruption = snap.get("interruption_count", 0)
    interrupted = interruption != planner.get("interruption_count_seen", 0)
    planner["interruption_count_seen"] = interruption
    # 不让同批次的松开/追踪丢失被随后捏合覆盖。
    if sm.state == sm.ACTIVE and (interrupted or release_edge or not snap.get("active")):
        out["stop_publish"] = sm.request_stop(now, "松开或追踪中断；等待完整停止流程")
        out["force_pause"] = "停止处理中；完成后重新松开再捏合"
    if tracking and sm.stop_request_time is not None:
        tracker.note_stop_requested(sm.stop_request_time)

    sm.observe(io)
    sm.evaluate(now)

    if tracking:
        result = getattr(io, "movel_result", None)
        if result is not None:
            tracker.on_result(result[0], result[1])
        tracker.check(now)

    active = bool(snap.get("active") and snap.get("robot_target_preview_base_m"))

    if sm.state == sm.STOPPED:
        if tracking and tracker.in_flight:
            sm.release_seen = False
            sm.last_reason = "反馈已确认静止；仍等待旧运动回执，禁止新命令"
            return out
        if tracking and tracker.blocked:
            sm.state = sm.LOCKED
            sm.last_reason = "旧运动结果失败/中断或超时；Bool不能区分原因，保持锁定"
            return out
        if tracking:
            pose, speed = getattr(io, "pose", None), getattr(io, "speed", None)
            if (pose is None or speed is None or
                    now - pose[0] > sm.feedback_max_age_s or
                    now - speed[0] > sm.feedback_max_age_s or
                    not is_finite_vector(speed[1], 6) or
                    any(abs(v) >= sm.still_speed_rpm for v in speed[1])):
                sm.state = sm.LOCKED
                sm.last_reason = "等待旧运动回执后反馈过期或速度不满足静止要求，禁止恢复"
                return out
        # 停止期间收到的松开不用于恢复；必须在完整流程结束后再松开。
        if planner.get("completed_stop_time") != sm.stop_request_time:
            planner["completed_stop_time"] = sm.stop_request_time
            sm.release_seen = False
            out["force_pause"] = "停止流程完成，请重新松开再捏合"
            return out
        if sm.release_seen:
            sm.state = sm.IDLE
            sm.stop_request_time = None
            sm.last_reason = "已看到有效松开，控制重新就绪"
    if sm.state == sm.LOCKED:
        # 故障锁定（停止失败/确认超时/结果未知）为进程级：手势不得解除，
        # 与 InFlightTracker 是否 blocked 无关。重启本控制程序也不能保证
        # 控制器侧已清除旧命令，须人工确认。
        if sm.release_seen:
            sm.last_reason = ("故障锁定：停止失败/超时/结果未知，松开再捏合"
                              "不能解除；需重启本控制程序（重启不能保证控制器"
                              "侧已清除旧命令，须人工确认）")
        return out
    if sm.state == sm.IDLE:
        if active and sm.release_seen:
            if tracking and tracker.blocked:
                # 进程级锁定：松开→再捏合不能解除（必须重启程序）
                sm.last_reason = ("在途运动命令失败/超时/结果未知：运动入口已锁定，"
                                  "松开再捏合不能解除，需重启程序")
                return out
            sm.state = sm.ACTIVE
            sm.last_reason = "捏合开始（本捏合从零增量起算）"
            planner["last_sent"] = None
            # 落到下方 ACTIVE 分支：首帧立即产出目标（=反馈锚点）
        else:
            return out
    if sm.state == sm.ACTIVE:
        if not active:
            out["stop_publish"] = sm.request_stop(
                now, snap.get("reason") or "松开/失效")
            return out
        if tracking:
            if tracker.blocked:
                out["stop_publish"] = sm.request_stop(
                    now, "上一运动命令结果未知/被中断：禁止自动恢复")
                out["force_pause"] = "上一命令状态不明确，请先松开"
                return out
            if tracker.in_flight:
                return out  # 上一条执行状态未明确前，不发布下一条
        target_raw = snap.get("robot_target_preview_base_m")
        ref = snap.get("robot_reference_at_pinch") or {}
        fb = snap.get("robot_feedback")
        quat = snap.get("robot_target_preview_rotation_xyzw")
        ref_pos = ref.get("position_m")
        if (not is_finite_vector(target_raw, 3)
                or not is_finite_vector(ref_pos, 3)
                or fb is None or not is_finite_vector(fb.get("position_m"), 3)
                or not is_finite_vector(quat, 4)):
            out["stop_publish"] = sm.request_stop(
                now, "目标/锚点/反馈/四元数非有限或缺失")
            out["force_pause"] = "数据异常，请先松开"
            return out
        # 新捏合检测（含同批次"松开→重新捏合"）：锚点变化 → 步长基准重置。
        if planner["reference"] is None or planner["reference"] != ref_pos:
            planner["last_sent"] = None
            planner["reference"] = list(ref_pos)
        try:
            target = clamp_radius(target_raw, ref_pos, cfg.max_offset)
            previous = (planner["last_sent"] if planner["last_sent"] is not None
                        else fb["position_m"])
            target = limit_step(target, previous, cfg.max_step)
        except ValueError as exc:
            out["stop_publish"] = sm.request_stop(now, f"限制器拒绝: {exc}")
            out["force_pause"] = "限制器拒绝，请先松开"
            return out
        if not check_target_deviation(target, fb["position_m"],
                                      cfg.max_feedback_deviation):
            out["stop_publish"] = sm.request_stop(now, "目标与实际反馈偏差超限")
            out["force_pause"] = "偏差超限，请先松开"
            return out
        if now - planner["last_send"] < cfg.period:
            return out
        out["msg"] = {"target": target, "quat": list(quat)}
        planner["last_sent"] = list(target)
        planner["last_send"] = now
        return out
    # STOP_REQUESTED（或其它）：一律禁止新运动命令
    return out


# ---- 发布门控（只读模式绝不发布） ----

def gated_publish_motion(io, msg):
    """只读模式不发布运动命令；真实模式调用 io.publish_motion。"""
    if not io.enable_motion:
        return "skipped_readonly"
    io.publish_motion(msg)
    return "published"


def gated_publish_stop(io):
    """只读模式不发布停止命令；真实模式调用 io.publish_stop。"""
    if not io.enable_motion:
        return "skipped_readonly"
    io.publish_stop()
    return "published"


# ---- 退出处理 ----

class ExitController:
    """SIGINT/SIGTERM 只设置退出请求标志；停止与确认随后有序执行。"""

    def __init__(self):
        self.requested = False
        self.signum = None

    def request(self, signum=None, frame=None):
        self.requested = True
        self.signum = signum


def install_exit_handlers(exit_ctrl):
    """接管 SIGINT/SIGTERM：先退出主循环，再由 finalize_stop 在 ROS 上下文
    仍有效时执行有界停止与确认。必须在 rclpy.init 之后调用（覆盖其默认
    处理器），且只能从主线程调用。"""
    signal.signal(signal.SIGINT, exit_ctrl.request)
    signal.signal(signal.SIGTERM, exit_ctrl.request)


def finalize_stop(sm, io, now_fn, spin_once, stop_confirm_timeout_s,
                  sleep_s=0.01, tracker=None):
    """程序退出前的有界停止与确认。返回报告行列表。

    上下文已失效时明确报告"无法确认停止"；只读模式不发送任何消息。
    """
    lines = []
    if sm.dry_run:
        lines.append("[退出] 只读模式：本次未发布任何命令，无需停止确认。")
        return lines
    if not sm.ever_moved and sm.state == sm.IDLE:
        lines.append("[退出] 本次未发布过运动命令，不发送停止。")
        return lines
    pending = tracker is not None and tracker.in_flight
    if sm.state == sm.STOPPED and not pending:
        lines.append(f"[退出] {sm.last_reason}")
        return lines
    if sm.state == sm.LOCKED and not pending:
        lines.append(f"[退出] 无法确认停止：{sm.last_reason}")
        return lines
    # ACTIVE / IDLE：需要在上下文仍有效时发送停止并做有界确认
    if not io.context_ok():
        lines.append("[退出] ROS 上下文已失效：无法发送停止命令，无法确认停止。")
        return lines
    if sm.request_stop(now_fn(), "程序退出"):
        if tracker is not None:
            tracker.note_stop_requested(sm.stop_request_time)
        io.publish_stop()
        lines.append("[退出] 已发送停止命令，等待确认。")
    if tracker is not None and sm.stop_request_time is not None:
        tracker.note_stop_requested(sm.stop_request_time)
    deadline = now_fn() + max(stop_confirm_timeout_s,
                              tracker.stop_result_timeout_s if tracker else 0)
    while now_fn() < deadline and (sm.state == sm.STOP_REQUESTED or
                                   (tracker is not None and tracker.in_flight)):
        try:
            spin_once()
        except Exception:
            lines.append("[退出] ROS 上下文在确认过程中失效：无法确认停止。")
            return lines
        sm.observe(io)
        sm.evaluate(now_fn())
        if tracker is not None:
            result = getattr(io, "movel_result", None)
            if result is not None:
                tracker.on_result(*result)
            tracker.check(now_fn())
        time.sleep(sleep_s)
    # 退出流程自洽收口：无论状态机内部超时参数如何，超时未确认必须锁定。
    if sm.state == sm.STOP_REQUESTED:
        sm.state = sm.LOCKED
        sm.last_reason = "退出流程停止确认超时：无法确认机械臂静止"
    if sm.state == sm.STOPPED:
        lines.append(f"[退出] {sm.last_reason}")
    elif sm.state == sm.LOCKED:
        lines.append(f"[退出] 无法确认停止：{sm.last_reason}")
    else:
        lines.append("[退出] 停止确认超时：无法确认停止。")
    if tracker is not None:
        if tracker.in_flight:
            tracker.blocked = True
            tracker.last_outcome = "unknown"
        lines.append(f"[退出] 旧运动结果: {tracker.last_outcome}；锁定: {tracker.blocked}")
    return lines


# ---- ROS IO 封装（仅在 ROS 可用时实例化） ----

class RobotIO:
    """ROS 话题收发封装。enable_motion=False 时 publish_* 直接抛错（双保险门控）。

    停止确认数据源:
      - /rm_driver/move_stop_result: std_msgs/Bool，无命令编号；
      - /rm_driver/udp_arm_position: geometry_msgs/Pose（约 200 Hz）；
      - /rm_driver/udp_joint_speed: Jointspeed，float32[] 单位 RPM、
        精度 0.02 RPM，要求控制器 UDP 上报 joint_speed_enable=true。
    """

    def __init__(self, node, enable_motion, robot_timeout):
        from rclpy.qos import (QoSProfile, ReliabilityPolicy, HistoryPolicy,
                               DurabilityPolicy)
        from geometry_msgs.msg import Pose
        from std_msgs.msg import Bool, Empty
        from rm_ros_interfaces.msg import Jointspeed, Movel
        import rclpy as _rclpy

        self._rclpy = _rclpy
        self.node = node
        self.enable_motion = bool(enable_motion)
        self.robot_timeout = float(robot_timeout)
        self.motion_pub = node.create_publisher(Movel, MOTION_TOPIC, 1) if enable_motion else None
        self.stop_pub = node.create_publisher(Empty, STOP_TOPIC, 1) if enable_motion else None
        self._Empty = Empty   # publish_stop 依赖此绑定（曾因局部导入产生 NameError）
        qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)
        self.pose = None           # (t, pos, quat)
        self.speed = None          # (t, [v...])，单位 RPM，严格 6 轴
        self.stop_result = None    # (t, bool)，无命令编号
        self.movel_result = None   # (t, bool)，无命令编号；在途纪律用
        self.pose_seq = 0          # 位置回调序号（observe 只消费新样本）
        self.speed_seq = 0         # 速度回调序号（observe 只消费新样本）
        node.create_subscription(Pose, POSE_TOPIC, self._on_pose, qos)
        node.create_subscription(Jointspeed, SPEED_TOPIC, self._on_speed, qos)
        result_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                               durability=DurabilityPolicy.VOLATILE)
        node.create_subscription(Bool, STOP_RESULT_TOPIC,
                                 self._on_stop_result, result_qos)
        node.create_subscription(Bool, MOVEL_RESULT_TOPIC,
                                 self._on_movel_result, result_qos)

    def _on_pose(self, msg):
        now = time.monotonic()
        pos = [msg.position.x, msg.position.y, msg.position.z]
        quat = [msg.orientation.x, msg.orientation.y,
                msg.orientation.z, msg.orientation.w]
        self.pose_seq += 1
        if is_finite_vector(pos, 3) and is_finite_vector(quat, 4):
            self.pose = (now, pos, quat)
        else:
            self.pose = None

    def _on_speed(self, msg):
        now = time.monotonic()
        self.speed_seq += 1
        values = list(msg.joint_speed)
        # 严格校验为本机械臂 6 轴：任意非空长度一律拒绝
        if len(values) == 6 and all(is_finite_number(v) for v in values):
            self.speed = (now, values)
        else:
            self.speed = None

    def _on_stop_result(self, msg):
        self.stop_result = (time.monotonic(), bool(msg.data))

    def _on_movel_result(self, msg):
        self.movel_result = (time.monotonic(), bool(msg.data))

    def context_ok(self):
        return self._rclpy.ok() and self.node.context.ok()

    def publish_motion(self, msg):
        if not self.enable_motion:
            raise RuntimeError("只读模式禁止发布运动命令")
        self.motion_pub.publish(msg)

    def publish_stop(self):
        if not self.enable_motion:
            raise RuntimeError("只读模式禁止发布停止命令")
        self.stop_pub.publish(self._Empty())


def build_movel_msg(movel_cls, target, quat, speed, block):
    msg = movel_cls()
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = target
    msg.pose.orientation.x, msg.pose.orientation.y = quat[0], quat[1]
    msg.pose.orientation.z, msg.pose.orientation.w = quat[2], quat[3]
    msg.speed = speed
    msg.trajectory_connect = 0
    # block 保持 True（未改）。block=1 阻塞期间 driver 订阅队列可能堆积、
    # 停止后是否恢复执行未核实，见模块 docstring。
    msg.block = block
    return msg


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="HoloLens→RML63 受限控制；真实运动入口已禁用"
                    "（--enable-motion 报错退出），当前仅只读预览")
    parser.add_argument("--bind", default="172.20.10.2")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--peer", default="172.20.10.3")
    parser.add_argument("--scale", type=float, default=0.2)
    parser.add_argument("--axis-signs", default="1,1,1")
    parser.add_argument("--axis-order", choices=("x,y,z", "x,z,y"), default="x,y,z",
                        help="Base XYZ分别采用的Unity轴；x,z,y交换Unity Y/Z，仅位置映射")
    parser.add_argument("--hand-timeout", type=float, default=0.2)
    parser.add_argument("--robot-timeout", type=float, default=0.5)
    parser.add_argument("--max-offset", type=float, default=0.05,
                        help="相对每次捏合锚点的最大总位移（米）")
    parser.add_argument("--max-step", type=float, default=0.003,
                        help="连续两次命令间最大步长（米）")
    parser.add_argument("--max-feedback-deviation", type=float, default=0.06,
                        help="目标与实际反馈最大允许偏差（米），超限即停止锁定")
    parser.add_argument("--publish-hz", type=float, default=2.0)
    parser.add_argument("--speed", type=int, default=5,
                        help="Movel 速度百分比 1..100")
    parser.add_argument("--still-window", type=float, default=0.2,
                        help="停止确认所需连续静止窗口（秒）")
    parser.add_argument("--still-pos-epsilon", type=float, default=5e-4,
                        help="静止窗口内位置允许漂移（米）")
    parser.add_argument("--still-speed-threshold", type=float, default=0.1,
                        help="关节速度静止阈值（RPM；Jointspeed 精度 0.02 RPM，"
                             "实机需按噪声基线标定）")
    parser.add_argument("--stop-confirm-timeout", type=float, default=2.0,
                        help="停止确认超时（秒），超时未确认即锁定")
    parser.add_argument("--max-sample-gap", type=float, default=0.05,
                        help="静止窗口内相邻反馈样本最大间隔（秒），"
                             "超限视为窗口不连续，不确认静止")
    parser.add_argument("--movel-result-timeout", type=float, default=2.0,
                        help="在途命令结果超时（秒）；超时视为未知并锁定，"
                             "禁止自动恢复")
    parser.add_argument("--stop-movel-result-timeout", type=float, default=15.0,
                        help="停止请求后等待旧运动回执的独立上限（秒）；不等于物理停止期限")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--enable-motion", action="store_true",
                        help="允许真实发布 Movel/move_stop。当前被禁用："
                             "队列风险未核实，传入即报错退出")
    args = parser.parse_args(argv)

    def require_positive_finite(name, value):
        if not (is_finite_number(value) and value > 0):
            parser.error(f"--{name} 必须为有限正数")

    for name in ("scale", "hand-timeout", "robot-timeout", "max-offset",
                 "max-step", "max-feedback-deviation", "publish-hz",
                 "still-window", "still-pos-epsilon",
                 "still-speed-threshold", "stop-confirm-timeout",
                 "max-sample-gap", "movel-result-timeout", "stop-movel-result-timeout"):
        require_positive_finite(name, getattr(args, name.replace("-", "_")))
    if not 0 <= args.port <= 65535:
        parser.error("--port 必须在 0-65535 之间")
    if not (is_finite_number(args.duration) and args.duration >= 0):
        parser.error("--duration 必须为有限非负数（0 表示持续运行）")
    if not (isinstance(args.speed, int) and 1 <= args.speed <= 100):
        parser.error("--speed 必须是 1..100 的整数")
    try:
        axis_signs = tuple(float(v.strip()) for v in args.axis_signs.split(","))
        if len(axis_signs) != 3 or any(v not in (-1.0, 1.0) for v in axis_signs):
            raise ValueError
    except ValueError:
        parser.error("--axis-signs 必须是三个 +1/-1，例如 1,1,1")

    # 真实运动入口硬性门控：队列风险核实前，--enable-motion 一律报错退出。
    # 位于 ROS 导入/节点/发布器创建之前：该路径不会创建任何发布器或发送消息。
    allowed, gate_reason = motion_entry_allowed(args.enable_motion)
    if not allowed:
        print(f"[错误] {gate_reason}", file=sys.stderr)
        return 3

    try:
        import rclpy
        from rclpy.node import Node
        from rm_ros_interfaces.msg import Movel
    except ImportError:
        print("[错误] 请先 source ROS2 Humble 和 DYS_ws", file=sys.stderr)
        return 1

    state = Preview(args.peer, args.scale, args.hand_timeout,
                    args.robot_timeout, axis_signs,
                    tuple("xyz".index(s) for s in args.axis_order.split(",")))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((args.bind, args.port))
        sock.setblocking(False)
    except OSError as exc:
        print(f"[错误] 无法监听 {args.bind}:{args.port}: {exc}", file=sys.stderr)
        sock.close()
        return 1

    from rclpy.signals import SignalHandlerOptions
    rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
    node = Node("hololens_robot_control", enable_rosout=False,
                start_parameter_services=False)
    io = RobotIO(node, args.enable_motion, args.robot_timeout)
    sm = StopStateMachine(
        still_window_s=args.still_window, still_pos_epsilon_m=args.still_pos_epsilon,
        still_speed_rpm=args.still_speed_threshold,
        stop_confirm_timeout_s=args.stop_confirm_timeout,
        feedback_max_age_s=args.robot_timeout,
        max_sample_gap_s=args.max_sample_gap,
        dry_run=not args.enable_motion)
    exit_ctrl = ExitController()
    try:
        install_exit_handlers(exit_ctrl)
    except (ValueError, OSError) as exc:
        print(f"[警告] 无法安装信号处理: {exc}", file=sys.stderr)

    args.period = 1.0 / args.publish_hz
    planner = {"last_sent": None, "last_send": 0.0, "reference": None,
               "release_count_seen": 0}
    tracker = InFlightTracker(result_timeout_s=args.movel_result_timeout,
                              stop_result_timeout_s=args.stop_movel_result_timeout)

    mode = ("只读预览（不发布任何消息）；真实运动入口锁定："
            "--enable-motion 被拒绝")
    print(f"[启动] {mode}；最大总偏移 {args.max_offset:.3f}m；"
          f"单步上限 {args.max_step:.3f}m；频率 {args.publish_hz:g}Hz；"
          f"反馈偏差上限 {args.max_feedback_deviation:.3f}m；"
          f"在途结果超时 {args.movel_result_timeout:.1f}s", flush=True)
    print("[说明] 先松开，再捏合；松手/追踪丢失/超时/偏差超限即停止并锁定，"
          "确认静止（move_stop_result + 位置/关节速度窗口）后才能恢复。",
          flush=True)

    end_reason = "duration_reached"
    started = time.monotonic()
    last_display = 0.0
    last_pose_seq = -1
    try:
        while (rclpy.ok() and not exit_ctrl.requested
               and (args.duration == 0
                    or time.monotonic() - started < args.duration)):
            now = time.monotonic()
            rclpy.spin_once(node, timeout_sec=0.005)
            if io.pose_seq != last_pose_seq:
                last_pose_seq = io.pose_seq
                if io.pose is not None:
                    state.robot_pose(io.pose[1], io.pose[2], io.pose[0])
            batch_start = time.monotonic()
            for _ in range(64):
                if time.monotonic() - batch_start > 0.005:
                    break
                try:
                    raw, peer = sock.recvfrom(65535)
                except BlockingIOError:
                    break
                state.receive(raw, peer, time.monotonic())
            now = time.monotonic()
            snap = state.snapshot(now)
            result = plan_tick(sm, io, snap, now, args, planner, tracker)
            if result["force_pause"]:
                state.pause(result["force_pause"])
            if result["stop_publish"]:
                tracker.note_stop_requested(sm.stop_request_time)
                action = gated_publish_stop(io)
                print(f"[停止] {sm.last_reason} | 发布: {action}", flush=True)
            if result["msg"] is not None:
                msg = build_movel_msg(Movel, result["msg"]["target"],
                                      result["msg"]["quat"], args.speed, True)
                if io.enable_motion:
                    tracker.on_publish(time.monotonic())
                    sm.note_motion_published()  # 发布异常也按可能已发送处理，退出时请求停止。
                action = gated_publish_motion(io, msg)
                if action == "published":
                    sm.note_motion_published()
                print(f"[目标] {vector_text(result['msg']['target'])}"
                      f" | 状态 {sm.state} | 发布: {action}", flush=True)
            if now - last_display >= 0.5:
                fb = snap.get("robot_feedback")
                print(f"[{time.strftime('%H:%M:%S')}] 状态 {sm.state} | "
                      f"{sm.last_reason} | {snap.get('reason')}\n"
                      f"  目标预览: {vector_text(snap.get('robot_target_preview_base_m'))}\n"
                      f"  机械臂实际: {vector_text(fb['position_m'] if fb else None)}",
                      flush=True)
                last_display = now
            time.sleep(0.005)
        if exit_ctrl.requested:
            end_reason = ("sigint" if exit_ctrl.signum == signal.SIGINT
                          else "sigterm")
        elif not rclpy.ok():
            end_reason = "ros_shutdown"
    except KeyboardInterrupt:
        end_reason = "keyboard_interrupt"
    except Exception as exc:
        end_reason = "error"
        print(f"[错误] {type(exc).__name__}: {exc}", file=sys.stderr)
    try:
        reports = finalize_stop(
            sm, io, now_fn=time.monotonic,
            spin_once=lambda: rclpy.spin_once(node, timeout_sec=0.01),
            stop_confirm_timeout_s=args.stop_confirm_timeout, tracker=tracker)
        for line in reports:
            print(line, flush=True)
    except Exception as exc:
        print(f"[退出] 停止收尾异常: {exc!r}；无法确认停止", file=sys.stderr)
    finally:
        # 仅资源清理；停止与确认已在上方显式完成，不依赖这里的 rclpy.ok()。
        try:
            sock.close()
        except OSError:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
        print(f"[结束] {end_reason} | 状态 {sm.state} | {sm.last_reason}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
