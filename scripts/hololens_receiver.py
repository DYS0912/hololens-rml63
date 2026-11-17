#!/usr/bin/env python3
"""HoloLens 2 手部数据接收工具（仅接收：不发送任何消息，不建立控制通道）。

数据协议: rml63.hand_pose.v1（HoloLens 2 -> Ubuntu 直连 UDP）
  schema / source / hand / pose_type / coordinate_frame / position_unit /
  quaternion_order / session_id / sequence / timestamp_unix_ms /
  monotonic_seconds / tracked / valid / invalid_reason / position / rotation

用法（项目根目录，仅 Python 标准库）:
    python3 scripts/hololens_receiver.py [--bind 172.20.10.2] [--port 5005]
        [--peer 172.20.10.3] [--duration 60] [--no-data-timeout 5.0]
        [--log-root datasets/hololens_checks]

行为:
  - 仅接收 UDP，不发送任何消息，不创建任何控制通道；
  - 来源 IP 过滤：非 --peer 的包只计数（wrong_source），不进入本次记录；
  - 逐包校验 JSON / schema / 字段类型 / 维度 / 有限性 / 四元数范数，
    畸形数据单独计数并记录原因，单个错误数据包不会导致程序退出；
  - valid=false 是正常的“姿态不可用”记录：保留原始空数组/null 与
    invalid_reason，不做零填充，不伪造坐标；
  - 不修改发送端任何原始字段。接收端附加信息写入 "_receiver" 子对象
    （若发送端恰好已有该键，则把键名前置下划线再试，绝不覆盖）；
  - 每次运行写入 datasets/hololens_checks/run_<时间戳>/：
      packets.jsonl  每个通过校验的包一行（含 valid=false）
      summary.json   运行参数与统计（正常结束时生成；kill -9 不保证生成）
  - Ctrl+C / 到达 --duration / SIGTERM 时关闭 socket、刷新文件并保存统计。

统计口径:
  - --duration 从监听（bind）成功开始计时，首包前的等待不计入频率；
  - 平均接收频率 = (N-1)/(末包接收单调时间 - 首包接收单调时间)，
    N 为通过校验的包数；仅当 N>=2 且时间差>0 时输出，否则为 null 并注明原因；
  - 序号按 session_id 分别跟踪：新会话序号从头开始不视为丢包；区分重复、
    乱序、暂时缺口与迟到补齐；“missing_sequences”只表示某些序号未被本次
    接收记录覆盖（未收到或未通过校验被剔除），不宣称网络丢包；
  - 两端时钟未同步：不用发送/接收墙钟时间之差估算单向延迟；
  - 接收间隔与抖动不是端到端控制延迟；
  - 有效比例 = 有效姿态数 / 全部通过校验的包数，与每秒屏幕显示采样无关。
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import socket
import sys
import threading
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_ROOT = PROJECT_ROOT / "datasets" / "hololens_checks"

SCHEMA_EXPECTED = "rml63.hand_pose.v1"
QUATERNION_NORM_TOLERANCE = 1e-3  # |norm(q) - 1| 的容差（见文档与 summary）
DEFAULT_NO_DATA_TIMEOUT_S = 5.0   # 无“通过校验”数据超过该秒数则显示等待
DISPLAY_INTERVAL_S = 1.0
SOCKET_TIMEOUT_S = 0.2
FLUSH_EVERY_PACKETS = 10
MAX_MISSING_LISTED = 100          # summary 中每个会话最多列出的缺失序号数
RECV_BUFSIZE = 65535

_NOTES = [
    "missing_sequences 表示这些序号未被本次接收记录覆盖（未收到，或收到但未通过校验被剔除），不代表确证网络丢包。",
    "发送端与接收端时钟未同步：不要用发送时间戳与接收墙钟时间之差估算单向延迟。",
    "接收间隔均值/P95/最大值反映接收节奏，不是端到端控制延迟。",
    "valid_ratio 基于全部通过校验的数据包计算，与每秒屏幕显示的采样无关。",
    "平均接收频率按 (N-1)/(末包接收单调时间-首包接收单调时间) 计算，N 为通过校验的包数；从监听成功起计时，排除首包前的等待时间。",
    "强制结束进程（如 kill -9）不保证生成最终 summary.json；请尽量用 Ctrl+C 停止。",
]


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


def validate_message(msg, expected_schema: str = SCHEMA_EXPECTED,
                     quat_tol: float = QUATERNION_NORM_TOLERANCE):
    """校验一个已解析的消息。返回 None 表示通过，否则返回失败原因字符串。

    不修改 msg 的任何内容。
    """
    if not isinstance(msg, dict):
        return "not_json_object"
    if msg.get("schema") != expected_schema:
        return "schema_mismatch"
    if not isinstance(msg.get("valid"), bool):
        return "valid_not_bool"
    if not isinstance(msg.get("tracked"), bool):
        return "tracked_not_bool"
    seq = msg.get("sequence")
    if not (isinstance(seq, int) and not isinstance(seq, bool) and seq >= 0):
        return "sequence_invalid"
    sid = msg.get("session_id")
    if not (isinstance(sid, str) and sid != ""):
        return "session_id_invalid"
    if not _is_finite_number(msg.get("timestamp_unix_ms")):
        return "timestamp_unix_ms_invalid"
    if not _is_finite_number(msg.get("monotonic_seconds")):
        return "monotonic_seconds_invalid"
    if msg["valid"]:
        # 有效姿态：position 3 维、rotation 4 维，全部为有限数值；
        # 四元数范数须在容差内接近 1。
        if not _is_finite_vector(msg.get("position"), 3):
            return "position_invalid"
        if not _is_finite_vector(msg.get("rotation"), 4):
            return "rotation_invalid"
        q = msg["rotation"]
        norm = math.sqrt(sum(e * e for e in q))
        if abs(norm - 1.0) > quat_tol:
            return "quaternion_norm_out_of_tolerance"
    else:
        # 无效姿态是正常记录：position/rotation 允许 null / [] / 缺省，
        # 也允许（虽然不用）完整的向量；其他类型算畸形。
        pos = msg.get("position")
        rot = msg.get("rotation")
        if not (pos is None or pos == [] or _is_finite_vector(pos, 3)):
            return "invalid_pose_position_invalid"
        if not (rot is None or rot == [] or _is_finite_vector(rot, 4)):
            return "invalid_pose_rotation_invalid"
    return None


class SessionTracker:
    """单个 session_id 的序号跟踪：重复 / 乱序 / 缺口 / 迟到补齐。"""

    def __init__(self):
        self.packet_count = 0
        self.valid_count = 0
        self.invalid_count = 0
        self.first_sequence = None
        self.max_sequence = None
        self.duplicate_count = 0
        self.out_of_order_count = 0
        self.late_fill_count = 0
        self._seen = set()
        self._missing = set()

    def add(self, seq: int, valid: bool):
        self.packet_count += 1
        if valid:
            self.valid_count += 1
        else:
            self.invalid_count += 1
        if self.first_sequence is None:
            # 会话首包：不追溯更早的序号（会话可能从任意序号开始）。
            self.first_sequence = seq
            self.max_sequence = seq
            self._seen.add(seq)
            return
        if seq in self._seen:
            self.duplicate_count += 1
            return
        if seq < self.max_sequence:
            self.out_of_order_count += 1
            if seq > self.first_sequence and seq in self._missing:
                self.late_fill_count += 1
                self._missing.discard(seq)
        else:
            for x in range(self.max_sequence + 1, seq):
                self._missing.add(x)
            self.max_sequence = seq
        self._seen.add(seq)

    @property
    def missing_sequences(self):
        return sorted(self._missing)

    def summary(self) -> dict:
        miss = self.missing_sequences
        return {
            "packet_count": self.packet_count,
            "valid_pose": self.valid_count,
            "invalid_pose": self.invalid_count,
            "first_sequence": self.first_sequence,
            "max_sequence": self.max_sequence,
            "duplicate_packets": self.duplicate_count,
            "out_of_order_packets": self.out_of_order_count,
            "gaps_filled_by_late_packets": self.late_fill_count,
            "missing_sequence_count": len(miss),
            "missing_sequences": miss[:MAX_MISSING_LISTED],
            "missing_sequences_truncated": len(miss) > MAX_MISSING_LISTED,
        }


def make_run_dir(root: Path) -> Path:
    """在 root 下创建 run_<时间戳> 目录；同名时追加 _2、_3…，绝不覆盖。"""
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    for i in range(1000):
        suffix = "" if i == 0 else f"_{i + 1}"
        run_dir = root / f"{stamp}{suffix}"
        try:
            run_dir.mkdir(exist_ok=False)
            return run_dir
        except FileExistsError:
            continue
    raise RuntimeError(f"无法在 {root} 创建独立运行目录（1 秒内重名过多）")


def _ms_to_iso(ms):
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000.0).astimezone().isoformat(
        timespec="milliseconds")


class HoloLensReceiver:
    """UDP 接收 + 校验 + 落盘 + 统计。start() 后调用 run()；测试可拆分调用。"""

    def __init__(self, bind_ip: str = "172.20.10.2", port: int = 5005,
                 peer_ip: str = "172.20.10.3", duration: float = 0.0,
                 log_root=None, no_data_timeout: float = DEFAULT_NO_DATA_TIMEOUT_S):
        self.bind_ip = bind_ip
        self.port = port
        self.actual_port = None
        self.peer_ip = peer_ip
        self.duration = float(duration)
        self.no_data_timeout = float(no_data_timeout)
        self.log_root = (Path(log_root) if log_root is not None
                         else DEFAULT_LOG_ROOT)

        self.sock = None
        self.run_dir = None
        self.packets_path = None
        self.summary_path = None
        self.packets_file = None
        self.listen_start_wall = None
        self.listen_start_mono = None
        self.last_display_mono = None
        self.stop_reason = None       # "sigterm" 等
        self.fatal_error = None

        # 计数
        self.total_received = 0
        self.wrong_source = 0
        self.malformed = 0
        self.malformed_reasons = Counter()
        self.validated = 0
        self.valid_count = 0
        self.invalid_count = 0
        self.packets_written = 0

        # 时序
        self.first_validated_mono = None
        self.first_validated_wall_ms = None
        self.last_validated_mono = None
        self.last_validated_wall_ms = None
        self.intervals = []           # 相邻通过校验包的接收单调时间差
        self.recent_times = deque()   # 最近 1 秒显示用

        # 会话 / 序号
        self.sessions = {}

        # 显示状态
        self.last_status = None       # "valid" | "invalid" | None
        self.last_position = None

    # ---- 启动 ----

    def open_run_files(self):
        """创建本次运行目录并打开 packets.jsonl（不涉及 socket）。"""
        try:
            self.run_dir = make_run_dir(self.log_root)
            self.packets_path = self.run_dir / "packets.jsonl"
            self.summary_path = self.run_dir / "summary.json"
            self.packets_file = open(self.packets_path, "w", encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(
                f"无法在 {self.log_root} 创建运行目录或输出文件: {exc}") from exc

    def start(self):
        """创建运行目录、绑定 UDP socket；监听成功后开始计时。"""
        self.open_run_files()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(SOCKET_TIMEOUT_S)
        try:
            self.sock.bind((self.bind_ip, self.port))
        except OSError as exc:
            self.sock.close()
            self.sock = None
            if self.packets_file is not None:
                self.packets_file.close()
                self.packets_file = None
            hint = (
                f"\n提示：端口 {self.port} 可能被占用。如果旧的接收脚本"
                "（如 ~/rml63_hand_receiver.py）仍在运行，请回到旧终端"
                "按 Ctrl+C 停止它，不要自行结束不明进程，然后重新运行本程序。")
            raise RuntimeError(
                f"无法绑定 {self.bind_ip}:{self.port}: {exc}{hint}") from exc
        self.actual_port = self.sock.getsockname()[1]
        # --duration 从监听成功这一刻开始计时
        self.listen_start_wall = datetime.now().astimezone()
        self.listen_start_mono = time.monotonic()
        self.last_display_mono = self.listen_start_mono
        self._install_signal_handler()
        duration_text = (f"{self.duration:.1f}s"
                         if self.duration > 0 else "持续运行（Ctrl+C 停止）")
        print(f"[启动] 监听 {self.bind_ip}:{self.actual_port}"
              f"，期望来源 {self.peer_ip}，时长 {duration_text}"
              f"，无数据超时阈值 {self.no_data_timeout:.1f}s", flush=True)
        print(f"[启动] 运行目录: {self.run_dir}"
              f"（packets.jsonl；结束时生成 summary.json）", flush=True)

    def _install_signal_handler(self):
        if threading.current_thread() is not threading.main_thread():
            return  # 非主线程（测试场景）不注册信号处理

        def _on_sigterm(signum, frame):
            self.stop_reason = "sigterm"

        try:
            signal.signal(signal.SIGTERM, _on_sigterm)
        except (ValueError, OSError):
            pass

    # ---- 数据处理 ----

    def process_datagram(self, raw: bytes, addr) -> str:
        """处理一个 UDP 数据报。

        返回分类: "wrong_source" | "malformed" | "saved_valid" |
        "saved_invalid" | "write_error"。单个错误数据包不会抛异常。
        """
        self.total_received += 1
        if addr[0] != self.peer_ip:
            self.wrong_source += 1
            return "wrong_source"
        wall_ms = time.time() * 1000.0
        mono = time.monotonic()
        try:
            msg = json.loads(raw.decode("utf-8"),
                             parse_constant=_reject_constant)
        except _NonFiniteJsonConstant:
            self._count_malformed("non_finite_json_constant")
            return "malformed"
        except UnicodeDecodeError:
            self._count_malformed("not_utf8")
            return "malformed"
        except ValueError:
            self._count_malformed("not_json")
            return "malformed"
        reason = validate_message(msg)
        if reason is not None:
            self._count_malformed(reason)
            return "malformed"

        record = dict(msg)  # 复制，不修改发送端原始内容
        self._attach_rx_fields(record, addr, wall_ms, mono)
        line = json.dumps(record, ensure_ascii=False) + "\n"
        if self.packets_file is None:
            self.fatal_error = "输出文件未打开（请先调用 start()）"
            return "write_error"
        try:
            self.packets_file.write(line)
            self.packets_written += 1
            if self.packets_written % FLUSH_EVERY_PACKETS == 0:
                self.packets_file.flush()
        except OSError as exc:
            self.fatal_error = (f"写入 {self.packets_path} 失败: {exc}。"
                                "本次及之后的数据未保存，程序即将停止。")
            return "write_error"

        self.validated += 1
        valid = msg["valid"]
        if valid:
            self.valid_count += 1
            self.last_status = "valid"
            self.last_position = list(msg["position"])
        else:
            self.invalid_count += 1
            self.last_status = "invalid"
            self.last_position = None
        if self.first_validated_mono is None:
            self.first_validated_mono = mono
            self.first_validated_wall_ms = wall_ms
        else:
            self.intervals.append(mono - self.last_validated_mono)
        self.last_validated_mono = mono
        self.last_validated_wall_ms = wall_ms
        self.recent_times.append(mono)
        tracker = self.sessions.setdefault(msg["session_id"],
                                           SessionTracker())
        tracker.add(msg["sequence"], valid)
        return "saved_valid" if valid else "saved_invalid"

    def _count_malformed(self, reason: str):
        self.malformed += 1
        self.malformed_reasons[reason] += 1

    def _attach_rx_fields(self, record: dict, addr, wall_ms: float,
                          mono: float):
        """附加接收端信息；若键名与发送端字段冲突则前置下划线，绝不覆盖。"""
        key = "_receiver"
        while key in record:
            key = "_" + key
        record[key] = {
            "wall_time_ms": wall_ms,     # 接收端墙钟（unix 毫秒）
            "monotonic_s": mono,         # 接收端单调时钟（秒）
            "from_ip": addr[0],          # 来源 IP
            "from_port": addr[1],        # 来源端口
        }

    # ---- 运行循环 ----

    def run(self) -> int:
        """主循环。返回进程退出码：0 正常结束，1 出错。"""
        if self.sock is None:
            raise RuntimeError("必须先 start() 再 run()")
        end_reason = "duration_reached"
        try:
            while True:
                if self.stop_reason is not None:
                    end_reason = self.stop_reason
                    break
                if self.fatal_error:
                    end_reason = "disk_write_error"
                    break
                if (self.duration > 0 and self.listen_start_mono is not None
                        and time.monotonic() - self.listen_start_mono
                        >= self.duration):
                    end_reason = "duration_reached"
                    break
                try:
                    raw, addr = self.sock.recvfrom(RECV_BUFSIZE)
                except socket.timeout:
                    self._maybe_display(time.monotonic())
                    continue
                except OSError as exc:
                    print(f"[错误] socket 读取异常: {exc}", file=sys.stderr)
                    end_reason = "socket_error"
                    break
                self.process_datagram(raw, addr)
                self._maybe_display(time.monotonic())
        except KeyboardInterrupt:
            end_reason = "keyboard_interrupt"
        return self._finish(end_reason)

    def _finish(self, end_reason: str) -> int:
        """关闭 socket、刷新并关闭数据文件、写 summary。返回退出码。"""
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None
        rc = 0
        if self.fatal_error:
            print(f"[错误] {self.fatal_error}", file=sys.stderr)
            rc = 1
        if self.packets_file is not None:
            try:
                self.packets_file.flush()
            except OSError as exc:
                print(f"[错误] 刷新 {self.packets_path} 失败: {exc}",
                      file=sys.stderr)
                rc = 1
            finally:
                try:
                    self.packets_file.close()
                except OSError:
                    pass
                self.packets_file = None

        end_wall = datetime.now().astimezone()
        end_mono = time.monotonic()
        summary = self.build_summary(end_reason, end_wall, end_mono)
        try:
            with open(self.summary_path, "w", encoding="utf-8") as fh:
                json.dump(summary, fh, ensure_ascii=False, indent=2)
                fh.write("\n")
        except OSError as exc:
            print(f"[错误] 无法保存 {self.summary_path}: {exc}",
                  file=sys.stderr)
            rc = 1
        else:
            print(f"[结束] 已保存统计: {self.summary_path}", flush=True)
        reason_text = {
            "duration_reached": "到达设定时长",
            "keyboard_interrupt": "收到 Ctrl+C",
            "sigterm": "收到 SIGTERM",
            "disk_write_error": "磁盘写入失败",
            "socket_error": "socket 异常",
            "unexpected_error": "未预期异常",
        }.get(end_reason, end_reason)
        print(f"[结束] 原因: {reason_text} | 通过校验: {self.validated} 包"
              f"（有效 {self.valid_count} / 无效 {self.invalid_count}）"
              f" | 格式错误 {self.malformed} | 来源不符 {self.wrong_source}"
              f" | 数据目录: {self.run_dir}", flush=True)
        return rc

    # ---- 显示 ----

    def status_line(self, now: float) -> str:
        """生成一秒汇总行（纯函数式，便于测试；会更新最近1秒窗口）。"""
        while self.recent_times and self.recent_times[0] < now - 1.0:
            self.recent_times.popleft()
        recent = len(self.recent_times)
        tstr = time.strftime("%H:%M:%S")
        counts = (f"最近1秒: {recent}包 | 累计: 有效 {self.valid_count}"
                  f" / 无效 {self.invalid_count} / 格式错误 {self.malformed}"
                  f" / 来源不符 {self.wrong_source}")
        stale = (self.last_validated_mono is None
                 or (now - self.last_validated_mono) > self.no_data_timeout)
        if stale:
            # 超时：显示等待，且绝不沿用旧的有效坐标
            return (f"[{tstr}] 等待头显数据……"
                    f"（{self.no_data_timeout:.1f}s 无有效数据） | {counts}")
        if self.last_status == "valid" and self.last_position is not None:
            pos = ", ".join(f"{v:.3f}" for v in self.last_position)
            return f"[{tstr}] 右手追踪正常 | 位置：[{pos}] | {counts}"
        return f"[{tstr}] 右手姿态不可用 | 无坐标 | {counts}"

    def _maybe_display(self, now: float):
        if self.last_display_mono is not None and \
                now - self.last_display_mono < DISPLAY_INTERVAL_S:
            return
        self.last_display_mono = now
        if self.packets_file is not None:
            try:
                self.packets_file.flush()
            except OSError as exc:
                self.fatal_error = f"刷新 {self.packets_path} 失败: {exc}"
                return
        print(self.status_line(now), flush=True)

    # ---- 统计 ----

    def build_summary(self, end_reason: str, end_wall: datetime,
                      end_mono: float) -> dict:
        n = self.validated
        if n == 0:
            valid_ratio = None
            valid_ratio_reason = "没有通过校验的数据包，无法计算比例"
        else:
            valid_ratio = self.valid_count / n
            valid_ratio_reason = None

        wait_s = None
        wait_reason = None
        if self.first_validated_mono is None:
            wait_reason = "未收到任何通过校验的数据包"
        elif self.listen_start_mono is None:
            wait_reason = "缺少监听起始单调时间（仅测试合成场景会出现）"
        else:
            wait_s = max(0.0, self.first_validated_mono
                         - self.listen_start_mono)

        rate = None
        rate_reason = None
        if n < 2:
            rate_reason = "通过校验的包少于 2 个，无法计算平均接收频率"
        elif (self.first_validated_mono is None
              or self.last_validated_mono is None):
            rate_reason = "缺少首/末包接收单调时间（仅测试合成场景会出现）"
        elif self.last_validated_mono <= self.first_validated_mono:
            rate_reason = "首末包接收单调时间差为 0，无法计算平均接收频率"
        else:
            rate = (n - 1) / (self.last_validated_mono
                              - self.first_validated_mono)

        intervals = self.intervals
        mean_s = p95_s = max_s = None
        interval_reason = None
        if len(intervals) == 0:
            interval_reason = "接收间隔不足 1 个（需要至少 2 个通过校验的包）"
        else:
            mean_s = sum(intervals) / len(intervals)
            ordered = sorted(intervals)
            max_s = ordered[-1]
            # 最近秩法 P95
            idx = min(len(ordered) - 1,
                      max(0, math.ceil(0.95 * len(ordered)) - 1))
            p95_s = ordered[idx]

        return {
            "tool": "rml63_hololens_receiver",
            "schema_expected": SCHEMA_EXPECTED,
            "quaternion_norm_tolerance": QUATERNION_NORM_TOLERANCE,
            "run": {
                "run_id": self.run_dir.name if self.run_dir else None,
                "log_root": str(self.log_root),
                "run_dir": str(self.run_dir) if self.run_dir else None,
                "packets_file": str(self.packets_path)
                if self.packets_path else None,
                "summary_file": str(self.summary_path)
                if self.summary_path else None,
                "bind_ip": self.bind_ip,
                "port": self.port,
                "actual_port": self.actual_port,
                "peer_ip": self.peer_ip,
                "duration_s": self.duration,
                "no_data_timeout_s": self.no_data_timeout,
                "start_wall_time": (self.listen_start_wall.isoformat(
                    timespec="milliseconds")
                    if self.listen_start_wall else None),
                "start_monotonic_s": self.listen_start_mono,
                "end_wall_time": end_wall.isoformat(timespec="milliseconds"),
                "end_monotonic_s": end_mono,
                "end_reason": end_reason,
            },
            "packets": {
                "total_received": self.total_received,
                "wrong_source": self.wrong_source,
                "malformed": self.malformed,
                "malformed_reasons": dict(self.malformed_reasons),
                "validated": self.validated,
                "packets_written": self.packets_written,
                "valid_pose": self.valid_count,
                "invalid_pose": self.invalid_count,
                "valid_ratio": valid_ratio,
                "valid_ratio_reason": valid_ratio_reason,
            },
            "timing": {
                "wait_before_first_packet_s": wait_s,
                "wait_before_first_packet_reason": wait_reason,
                "first_validated_wall_time":
                    _ms_to_iso(self.first_validated_wall_ms),
                "first_validated_monotonic_s": self.first_validated_mono,
                "last_validated_wall_time":
                    _ms_to_iso(self.last_validated_wall_ms),
                "last_validated_monotonic_s": self.last_validated_mono,
                "avg_receive_rate_hz": rate,
                "avg_receive_rate_reason": rate_reason,
                "interval_mean_s": mean_s,
                "interval_p95_s": p95_s,
                "interval_max_s": max_s,
                "interval_stats_reason": interval_reason,
                "p95_method": "nearest-rank",
            },
            "sessions": {
                sid: tracker.summary()
                for sid, tracker in sorted(self.sessions.items())
            },
            "notes": _NOTES,
        }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="HoloLens 2 手部数据接收工具（仅接收 UDP，不发送、不控制）")
    parser.add_argument("--bind", default="172.20.10.2",
                        help="本机监听 IP（默认 172.20.10.2）")
    parser.add_argument("--port", type=int, default=5005,
                        help="UDP 接收端口（默认 5005）")
    parser.add_argument("--peer", default="172.20.10.3",
                        help="期望的头显来源 IP，其他来源只计数不记录"
                             "（默认 172.20.10.3）")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="运行秒数，从监听成功起计时；0 表示持续运行"
                             "直到 Ctrl+C（默认 0）")
    parser.add_argument("--no-data-timeout", type=float,
                        default=DEFAULT_NO_DATA_TIMEOUT_S,
                        help="超过多少秒没有通过校验的数据就显示"
                             "“等待头显数据……”（默认 5.0）")
    parser.add_argument("--log-root", default=None,
                        help="运行日志根目录（默认项目内 "
                             "datasets/hololens_checks）")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("--port 必须在 0-65535 之间")
    if args.duration < 0:
        parser.error("--duration 不能为负数")
    if args.no_data_timeout <= 0:
        parser.error("--no-data-timeout 必须大于 0")

    receiver = HoloLensReceiver(
        bind_ip=args.bind, port=args.port, peer_ip=args.peer,
        duration=args.duration, log_root=args.log_root,
        no_data_timeout=args.no_data_timeout)
    try:
        receiver.start()
    except (OSError, RuntimeError) as exc:
        print(f"[错误] 启动失败: {exc}", file=sys.stderr)
        return 1
    try:
        return receiver.run()
    except Exception as exc:  # 兜底：尽量保留已收数据并生成 summary
        print(f"[错误] 运行中未预期异常: {exc!r}", file=sys.stderr)
        receiver.fatal_error = repr(exc)
        return receiver._finish("unexpected_error")


if __name__ == "__main__":
    sys.exit(main())
