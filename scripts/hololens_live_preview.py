#!/usr/bin/env python3
"""Receive hand UDP + subscribe to robot Pose; terminal preview only, no motion output.

Requires the existing hololens_receiver.py and hololens_target_converter.py
in the same scripts directory. ROS imports are deferred so the core can be
tested without ROS. Unity preview and robot feedback deliberately stay separate.
"""
from __future__ import annotations

import argparse
import json
import math
import signal
import socket
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from hololens_receiver import validate_message
from hololens_target_converter import TargetConverter


def finite_vector(value, size):
    return (isinstance(value, (list, tuple)) and len(value) == size
            and all(type(x) in (int, float) and math.isfinite(x) for x in value))


def reject_constant(value):
    raise ValueError("Non-finite JSON: " + value)


class LiveConverter(TargetConverter):
    def pause(self):
        # Narrow adapter to the supplied converter, retaining its last target.
        self._break_anchor()

    def begin_pinch(self):
        # 实时路径：新捏合从预览原点起算，不携带上一次捏合的累计虚拟位移。
        # 必须在捏合首帧交给 convert_record 之前调用（调用顺序由测试验证）。
        self.begin_new_pinch()


class Preview:
    def __init__(self, peer="172.20.10.3", scale=0.2,
                 hand_timeout=0.2, robot_timeout=0.5,
                 axis_signs=(1.0, 1.0, 1.0), axis_order=(0, 1, 2)):
        for value in (scale, hand_timeout, robot_timeout):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("scale/timeouts must be finite and positive")
        self.peer, self.scale = peer, scale
        if (len(axis_signs) != 3 or
                any(value not in (-1.0, 1.0) for value in axis_signs)):
            raise ValueError("axis_signs must contain exactly three values, each +1 or -1")
        self.axis_signs = tuple(float(value) for value in axis_signs)
        if (len(axis_order) != 3 or any(type(i) is not int for i in axis_order)
                or sorted(axis_order) != [0, 1, 2]):
            raise ValueError("axis_order must be a permutation of 0,1,2")
        self.axis_order = tuple(axis_order)
        self.hand_timeout, self.robot_timeout = hand_timeout, robot_timeout
        self.converter = LiveConverter(scale, hand_timeout)
        self.session = None
        self.retired_sessions = set()
        self.last_seq = None
        self.hand_time = self.robot_time = None
        self.robot = self.hand = self.target = None
        self.robot_reference = self.hand_anchor = None
        self.require_release = True
        self.release_count = 0      # 有效松开（control_enabled=false）计数，供控制层边沿检测
        self.active = False
        self.interruption_count = 0
        self.reason = "等待头显和机械臂状态；先松开捏合"
        self.counts = Counter()

    def pause(self, reason, require_release=True):
        if self.active:
            self.interruption_count += 1
        self.converter.pause()
        self.active = False
        self.target = None
        self.hand_anchor = None
        self.robot_reference = None
        self.reason = reason
        if require_release:
            self.require_release = True

    def robot_pose(self, position, quaternion, now):
        if (not finite_vector(position, 3) or not finite_vector(quaternion, 4)
                or abs(math.hypot(*quaternion) - 1) > 1e-3):
            self.robot = None
            self.robot_time = None
            self.counts["bad_robot_pose"] += 1
            self.pause("机械臂状态异常；恢复后先松开")
            return
        # Detect a gap even when the callback runs before the main-loop tick.
        if self.robot_time is not None and now - self.robot_time >= self.robot_timeout:
            self.pause("机械臂状态曾超时；先松开再捏合")
        self.robot = {"position_m": list(position), "quaternion_xyzw": list(quaternion)}
        self.robot_time = now

    def tick(self, now):
        if self.robot_time is None or now - self.robot_time >= self.robot_timeout:
            self.pause("等待/超时：机械臂状态；恢复后先松开")
        elif self.hand_time is None or now - self.hand_time >= self.hand_timeout:
            self.hand = None
            self.pause("等待/超时：头显数据；恢复后先松开")

    def receive(self, raw, peer, now):
        self.counts["received"] += 1
        if peer[0] != self.peer:
            self.counts["wrong_source"] += 1
            return {"accepted": False, "reason": "wrong_source"}
        self.tick(now)
        try:
            msg = json.loads(raw.decode("utf-8"), parse_constant=reject_constant)
            error = validate_message(msg)
            if error:
                raise ValueError(error)
            expected = {"source": "hololens2", "hand": "right",
                        "pose_type": "openxr_grip", "coordinate_frame": "unity_tracking_space",
                        "position_unit": "m", "quaternion_order": "xyzw"}
            for key, value in expected.items():
                if msg.get(key) != value:
                    raise ValueError("unexpected " + key)
            if type(msg.get("control_enabled")) is not bool:
                raise ValueError("control_enabled must be boolean")
            if msg["valid"] and not msg["tracked"]:
                raise ValueError("valid=true with tracked=false")
        except (ValueError, UnicodeError, OverflowError, RecursionError) as exc:
            self.counts["malformed"] += 1
            self.hand = None
            self.pause("头显数据异常；恢复后先松开")
            return {"accepted": False, "reason": str(exc)}

        sid, seq = msg["session_id"], msg["sequence"]
        if sid in self.retired_sessions:
            self.counts["retired_session"] += 1
            self.pause("旧会话数据被拒绝；先松开")
            return {"accepted": False, "reason": "retired_session"}
        if sid != self.session:
            if self.session is not None:
                self.retired_sessions.add(self.session)
            self.session, self.last_seq = sid, None
            self.converter = LiveConverter(self.scale, self.hand_timeout)
            self.pause("新会话，虚拟目标归零；先松开")
        if self.last_seq is not None and seq <= self.last_seq:
            self.counts["non_increasing_sequence"] += 1
            self.pause("重复/倒退序号；先松开再捏合")
            return {"accepted": False, "reason": "non_increasing_sequence"}
        self.last_seq, self.hand_time = seq, now
        self.counts["validated"] += 1
        self.hand = list(msg["position"]) if msg["valid"] else None
        # Original sender metadata is kept only in raw_record in the log.
        # The converter always receives a receiver-owned monotonic timestamp.
        record = {key: msg[key] for key in
                  ("session_id", "sequence", "valid", "control_enabled")}
        record.update(position=msg.get("position"), _receiver={"monotonic_s": now})
        if not msg["valid"] or not msg["tracked"]:
            self.pause("右手追踪不可用；恢复后先松开")
        elif self.robot_time is None or now - self.robot_time >= self.robot_timeout:
            self.pause("机械臂状态未就绪；恢复后先松开")
        elif msg["control_enabled"] is False:
            self.pause("已松开，等待捏合", require_release=False)
            self.require_release = False
            self.release_count += 1
        elif self.require_release:
            self.pause("请先松开捏合，再捏合开始预览")
        else:
            if not self.active:
                # 新捏合首帧：本捏合从零增量起算（不携带上次累计虚拟位移），
                # 并捕获当前有效机械臂反馈作为机器人锚点（反馈新鲜度已在上方
                # 检查）。begin_pinch 必须先于本帧 convert_record 执行。
                self.converter.begin_pinch()
                self.hand_anchor = list(msg["position"])
                self.robot_reference = {
                    "position_m": list(self.robot["position_m"]),
                    "quaternion_xyzw": list(self.robot["quaternion_xyzw"]),
                }
            result = self.converter.convert_record(record)["preview_target"]
            if result["target_valid"] and finite_vector(result["target_position"], 3):
                self.target = result["target_position"]
                self.active = True
                self.reason = "预览更新中（未标定，不发送运动命令）"
                self.counts["preview_updates"] += 1
            else:
                self.pause("目标计算异常；先松开")
        return {"accepted": True, "raw_record": msg,
                "receiver": {"monotonic_s": now, "from_ip": peer[0], "from_port": peer[1]}}

    def snapshot(self, now):
        self.tick(now)
        robot_fresh = self.robot_time is not None and now - self.robot_time < self.robot_timeout
        hand_fresh = self.hand_time is not None and now - self.hand_time < self.hand_timeout
        robot_target = None
        if self.active and self.robot_reference is not None and self.target is not None:
            robot_target = [
                self.robot_reference["position_m"][i]
                + self.axis_signs[i] * self.target[self.axis_order[i]]
                for i in range(3)
            ]
        return {"monotonic_s": now, "active": self.active,
                "reason": self.reason, "release_required": self.require_release,
                "release_count": self.release_count,
                "interruption_count": self.interruption_count,
                "hand_age_s": None if self.hand_time is None else now - self.hand_time,
                "robot_age_s": None if self.robot_time is None else now - self.robot_time,
                "hand_unity_m": self.hand if hand_fresh else None,
                "hand_delta_unity_m": [self.hand[i] - self.hand_anchor[i] for i in range(3)]
                if self.active else None,
                "preview_unity_m": self.target if self.active else None,
                "robot_feedback": self.robot if robot_fresh else None,
                "robot_reference_at_pinch": self.robot_reference if self.active else None,
                "robot_target_preview_base_m": robot_target,
                "robot_target_preview_rotation_xyzw":
                    self.robot_reference["quaternion_xyzw"] if self.active and self.robot_reference else None,
                "axis_signs": list(self.axis_signs),
                "axis_order": list(self.axis_order),
                "target_base": robot_target, "target_rotation": None,
                "calibrated": False, "motion_output_enabled": False,
                "counts": dict(self.counts)}


def vector_text(value):
    return "--" if value is None else "[" + ", ".join(f"{v:.4f}" for v in value) + "]"


def main(argv=None):
    parser = argparse.ArgumentParser(description="实时只读预览：Unity虚拟目标和机械臂反馈分开显示，无运动输出")
    parser.add_argument("--bind", default="172.20.10.2")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--peer", default="172.20.10.3")
    parser.add_argument("--scale", type=float, default=0.2)
    parser.add_argument("--hand-timeout", type=float, default=0.2)
    parser.add_argument("--robot-timeout", type=float, default=0.5)
    parser.add_argument("--axis-signs", default="1,1,1",
                        help="Unity位移对应Base轴的符号，例如 1,-1,1；只允许±1")
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--axis-order", choices=("x,y,z", "x,z,y"), default="x,y,z",
                        help="Base XYZ分别采用的Unity轴；x,z,y为手右/前/上对应Base X/Y/Z")
    parser.add_argument("--log-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "datasets/hololens_checks")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535 or not math.isfinite(args.duration) or args.duration < 0:
        parser.error("端口需在0..65535；duration需为有限非负数，0表示持续运行")
    try:
        try:
            axis_signs = tuple(float(value.strip()) for value in args.axis_signs.split(","))
        except ValueError:
            parser.error("--axis-signs 必须是三个逗号分隔的 +1/-1，例如 1,1,1")
        order = tuple("xyz".index(s) for s in args.axis_order.split(","))
        state = Preview(args.peer, args.scale, args.hand_timeout, args.robot_timeout, axis_signs, order)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
        from geometry_msgs.msg import Pose
    except ImportError:
        print("[错误] 请使用系统 Python3，并先 source /opt/ros/humble/setup.bash", file=sys.stderr)
        return 1

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((args.bind, args.port))
        sock.setblocking(False)
    except OSError as exc:
        sock.close()
        print(f"[错误] 无法监听：{exc}。请检查IP，或在旧接收器终端按Ctrl+C释放端口。", file=sys.stderr)
        return 1
    node = log = None
    run_dir = None
    started = time.monotonic()
    end_reason = "duration_reached"
    rc = 0
    stop_requested = False

    def stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, stop)
    try:
        rclpy.init(args=[])
        node = Node("hololens_readonly_preview", enable_rosout=False,
                    start_parameter_services=False)
        qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)

        def on_pose(msg):
            p, q = msg.position, msg.orientation
            state.robot_pose([p.x, p.y, p.z], [q.x, q.y, q.z, q.w], time.monotonic())

        subscription = node.create_subscription(Pose, "/rm_driver/udp_arm_position", on_pose, qos)
        args.log_root.mkdir(parents=True, exist_ok=True)
        stem = "live_preview_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = args.log_root / stem
        run_dir.mkdir()
        log = (run_dir / "events.jsonl").open("x", encoding="utf-8")
        print(f"[启动] UDP {args.bind}:{sock.getsockname()[1]}；只订阅机械臂状态；日志 {run_dir}", flush=True)
        print(f"[说明] 仅位置增量预览；轴顺序 {args.axis_order}；符号 {args.axis_signs}。"
              "以捏合时机器人位置为锚点；方向近似，未精确标定，不发送运动。", flush=True)
        last_display = last_snapshot = 0.0
        while rclpy.ok() and not stop_requested:
            now = time.monotonic()
            if args.duration and now - started >= args.duration:
                break
            state.tick(now)
            rclpy.spin_once(node, timeout_sec=0.005)
            # Bound work to avoid starvation of the freshness checks.
            batch_start = time.monotonic()
            for _ in range(64):
                if time.monotonic() - batch_start > 0.005:
                    break
                try:
                    raw, peer = sock.recvfrom(65535)
                except BlockingIOError:
                    break
                event = state.receive(raw, peer, time.monotonic())
                if peer[0] == args.peer:
                    log.write(json.dumps({"kind": "hand_packet", "event": event,
                                          "state": state.snapshot(time.monotonic())},
                                         ensure_ascii=False, allow_nan=False) + "\n")
            now = time.monotonic()
            snap = state.snapshot(now)
            if now - last_snapshot >= 0.1:
                log.write(json.dumps({"kind": "tick", "state": snap},
                                     ensure_ascii=False, allow_nan=False) + "\n")
                log.flush()
                last_snapshot = now
            if now - last_display >= 0.5:
                robot = snap["robot_feedback"]
                print(f"[{time.strftime('%H:%M:%S')}] {snap['reason']}\n"
                      f"  Unity虚拟目标(m): {vector_text(snap['preview_unity_m'])}\n"
                      f"  机械臂相对目标预览Base(m): {vector_text(snap['robot_target_preview_base_m'])}\n"
                      f"  机械臂实际位置(m): {vector_text(robot['position_m'] if robot else None)}",
                      flush=True)
                last_display = now
        if stop_requested:
            end_reason = "sigterm"
        elif not rclpy.ok():
            end_reason = "ros_shutdown"
    except KeyboardInterrupt:
        end_reason = "keyboard_interrupt"
    except Exception as exc:
        end_reason = "error"
        rc = 1
        print(f"[错误] {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        state.pause("程序结束")
        sock.close()
        if log:
            log.close()
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if run_dir:
            summary = {"end_reason": end_reason, "counts": dict(state.counts),
                       "duration_s": time.monotonic() - started, "calibrated": False,
                       "motion_output_enabled": False,
                       "note": "收包时间不证明源端新鲜度；未标定；非机械臂安全控制器"}
            (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[结束] {end_reason}；日志 {run_dir}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
