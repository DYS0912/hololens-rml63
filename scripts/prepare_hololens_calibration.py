#!/usr/bin/env python3
"""离线坐标标定准备工具：从 target_packets.jsonl 选出标定样本。

为以后建立 HoloLens 坐标系 → RML63 基坐标系的映射做准备。
本次只处理已有数据：不启动 ROS2、不连接机械臂、不发送任何控制指令、
不修改任何原始文件。

用法（项目根目录，仅 Python 标准库）:
    python3 scripts/prepare_hololens_calibration.py \
        --input datasets/hololens_checks/run_<id>/target_packets.jsonl \
        --output datasets/hololens_checks/run_<id>/calibration_samples.jsonl \
        [--min-distance 0.02] [--max-samples 0]

选取规则:
  - 仅选 target_valid 严格为 true、control_enabled 严格为 true、
    且 target_position 为 3 个有限数值的帧（有效目标帧）；
  - 相邻样本去重：与上一个选中样本的 hand_position 欧氏距离小于
    --min-distance 的帧跳过（防止保存大量重复点）。按 hand_position 计算
    而非 target_position：target_position 是 0.2 倍缩放的平移预览，捏合
    开始帧 target 不动而手会大幅移动，按 target 去重会误删有效配对点；
  - --max-samples 限制最多保存的样本数（0 = 不限制）；
  - 按输入顺序输出。

重要边界（务必明确）:
  - hand_position / target_position 均为未标定的 Unity tracking space 坐标，
    不是 RML63 基坐标，不是 LeRobot action；每条样本标记
    coordinate_status = "uncalibrated_unity_tracking_space"；
  - 当前没有 robot_tcp_position（未连接机械臂、未采集机器人 TCP），
    因此不能计算 HoloLens 到 RML63 的真实变换；
  - estimate_rigid_transform（Umeyama 刚体变换估计，纯标准库实现）只用于
    人工点的虚拟测试（见 tests/test_prepare_hololens_calibration.py），
    人工点仅验证算法，不写成真实实验数据。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

DEFAULT_MIN_DISTANCE_M = 0.02   # 相邻样本手位置最小距离（米）
DEFAULT_MAX_SAMPLES = 0         # 0 = 不限制
COORDINATE_STATUS = "uncalibrated_unity_tracking_space"

NOTES = [
    "hand_position/target_position 均为未标定的 Unity tracking space 坐标，"
    "不是 RML63 基坐标，也不是 LeRobot action。",
    "robot_calibrated=false：当前没有 robot_tcp_position（未连接机械臂、"
    "未采集机器人 TCP），无法计算 HoloLens 到 RML63 的真实刚体变换。",
    "estimate_rigid_transform 仅用于人工点虚拟测试，未对真实数据计算变换。",
    "相邻样本去重按 hand_position 的欧氏距离计算：target_position 是缩放"
    "平移预览，捏合开始帧 target 不动而手会移动，按 target 去重会误删"
    "有效配对点。",
]


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_finite_number(x) -> bool:
    return _is_number(x) and math.isfinite(x)


def _is_finite_vector(x, dim: int) -> bool:
    return (isinstance(x, list) and len(x) == dim
            and all(_is_finite_number(e) for e in x))


def _euclid(a, b) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def select_calibration_samples(records, min_distance: float = DEFAULT_MIN_DISTANCE_M,
                               max_samples: int = DEFAULT_MAX_SAMPLES):
    """从 target_packets 记录流中选出标定样本。

    records: 可迭代的 dict（target_packets.jsonl 的行）。
    返回 (samples: list[dict], stats: dict)。
    """
    stats = {
        "input_total_frames": 0,
        "valid_target_frames": 0,
        "selected_calibration_samples": 0,
        "skipped_by_distance": 0,
        "skipped_by_max_samples": 0,
        "excluded_invalid_hand_position": 0,
        "coordinate_frame": "unity_tracking_space",
        "robot_calibrated": False,
        "min_distance_m": min_distance,
        "max_samples": max_samples,
        "notes": NOTES,
    }
    samples = []
    last_hand = None
    for rec in records:
        stats["input_total_frames"] += 1
        pt = rec.get("preview_target") or rec.get("_preview_target") or {}
        if not (rec.get("control_enabled") is True
                and pt.get("target_valid") is True
                and _is_finite_vector(pt.get("target_position"), 3)):
            continue
        stats["valid_target_frames"] += 1
        hand = rec.get("position")
        if not _is_finite_vector(hand, 3):
            stats["excluded_invalid_hand_position"] += 1
            continue
        if (last_hand is not None and min_distance > 0
                and _euclid(hand, last_hand) < min_distance):
            stats["skipped_by_distance"] += 1
            continue
        if max_samples > 0 and len(samples) >= max_samples:
            stats["skipped_by_max_samples"] += 1
            continue
        samples.append({
            "session_id": rec.get("session_id"),
            "sequence": rec.get("sequence"),
            "timestamp_unix_ms": rec.get("timestamp_unix_ms"),
            "hand_position": [float(v) for v in hand],
            "target_position": [float(v) for v in pt["target_position"]],
            "target_rotation": pt.get("target_rotation"),  # 恒 null（旋转映射未实现）
            "coordinate_frame": rec.get("coordinate_frame",
                                        "unity_tracking_space"),
            "coordinate_status": COORDINATE_STATUS,
            "robot_tcp_position": None,  # 当前没有机器人 TCP 数据
        })
        last_hand = [float(v) for v in hand]
    stats["selected_calibration_samples"] = len(samples)
    return samples, stats


# ---- 刚体变换估计（Umeyama，纯标准库；仅用于人工点虚拟测试） ----

def _jacobi_eigen_sym3(a):
    """3x3 实对称矩阵特征分解（Jacobi 旋转法）。

    返回 (values, vectors)：values 为按降序排列的特征值；
    vectors 为 3x3 行主序矩阵，其第 j 列是 values[j] 对应的特征向量。
    """
    mat = [[float(a[i][j]) for j in range(3)] for i in range(3)]
    vec = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    for _ in range(64):
        off = abs(mat[0][1]) + abs(mat[0][2]) + abs(mat[1][2])
        if off <= 1e-14:
            break
        for p, q in ((0, 1), (0, 2), (1, 2)):
            apq = mat[p][q]
            if abs(apq) <= 1e-16:
                continue
            tau = (mat[q][q] - mat[p][p]) / (2.0 * apq)
            t = ((1.0 if tau >= 0 else -1.0)
                 / (abs(tau) + math.sqrt(1.0 + tau * tau)))
            c = 1.0 / math.sqrt(1.0 + t * t)
            s = t * c
            for k in range(3):
                akp = mat[k][p]
                akq = mat[k][q]
                mat[k][p] = c * akp - s * akq
                mat[k][q] = s * akp + c * akq
            for k in range(3):
                apk = mat[p][k]
                aqk = mat[q][k]
                mat[p][k] = c * apk - s * aqk
                mat[q][k] = s * apk + c * aqk
            for k in range(3):
                vkp = vec[k][p]
                vkq = vec[k][q]
                vec[k][p] = c * vkp - s * vkq
                vec[k][q] = s * vkp + c * vkq
    order = sorted(range(3), key=lambda i: mat[i][i], reverse=True)
    values = [mat[i][i] for i in order]
    vectors = [[vec[i][order[j]] for j in range(3)] for i in range(3)]
    return values, vectors


def _matmul(a, b):
    n = len(a)
    m = len(b[0])
    p = len(b)
    return [[sum(a[i][k] * b[k][j] for k in range(p)) for j in range(m)]
            for i in range(n)]


def _det3(a) -> float:
    return (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))


def estimate_rigid_transform(source, target):
    """刚体变换估计（Umeyama，纯标准库实现，仅供人工点虚拟测试）。

    source / target: 等长的 3 维点序列（列表或元组均可，source -> target
    的映射）。
    返回 (R, t, rms)：R 为 3x3 行主序旋转矩阵（det=+1），t 为平移向量，
    rms 为拟合均方根误差。
    点数少于 3 对、长度不一致、含非有限值、点退化（共线/重合）时
    返回 None。

    注意: 本函数目前只用于虚拟测试验证算法本身；真实数据的标定需要
    成对的 (手位置, robot_tcp_position)，当前没有机器人 TCP 数据。
    """
    if len(source) != len(target) or len(source) < 3:
        return None

    def _to_points(points):
        out = []
        for p in points:
            if not (isinstance(p, (list, tuple)) and len(p) == 3
                    and all(_is_finite_number(x) for x in p)):
                return None
            out.append([float(x) for x in p])
        return out

    src = _to_points(source)
    dst = _to_points(target)
    if src is None or dst is None:
        return None
    n = len(src)
    mu_s = [sum(p[i] for p in src) / n for i in range(3)]
    mu_t = [sum(q[i] for q in dst) / n for i in range(3)]
    # H = Σ (t_i - μ_t)(s_i - μ_s)ᵀ（注意：目标点在左，源点在右）
    h = [[0.0] * 3 for _ in range(3)]
    for p, q in zip(src, dst):
        ds = [p[i] - mu_s[i] for i in range(3)]
        dt = [q[i] - mu_t[i] for i in range(3)]
        for i in range(3):
            for j in range(3):
                h[i][j] += dt[i] * ds[j]
    # H = U Σ Vᵀ：对 HᵀH（对称半正定）做 Jacobi 特征分解得 V 与 Σ²
    hth = _matmul([[h[j][i] for j in range(3)] for i in range(3)], h)
    vals, v = _jacobi_eigen_sym3(hth)
    svals = [math.sqrt(max(x, 0.0)) for x in vals]
    if svals[2] <= 1e-9 * max(svals[0], 1e-12):
        return None  # 退化：共线或重合点，旋转不可确定
    # U = H V Σ⁻¹
    u = [[sum(h[i][k] * v[k][j] for k in range(3)) / svals[j]
          for j in range(3)] for i in range(3)]
    d = 1.0 if _det3(u) * _det3(v) >= 0 else -1.0
    vt = [[v[j][i] for j in range(3)] for i in range(3)]
    us = [[u[i][0], u[i][1], u[i][2] * d] for i in range(3)]
    r = _matmul(us, vt)
    t = [mu_t[i] - sum(r[i][k] * mu_s[k] for k in range(3))
         for i in range(3)]
    rms = math.sqrt(sum(
        sum((sum(r[i][k] * p[k] for k in range(3)) + t[i] - q[i]) ** 2
            for i in range(3))
        for p, q in zip(src, dst)) / n)
    return r, t, rms


def _print_stats(stats: dict):
    print(f"[统计] 输入总帧数: {stats['input_total_frames']}")
    print(f"[统计] 有效目标帧数: {stats['valid_target_frames']}")
    print(f"[统计] 选中标定样本数: {stats['selected_calibration_samples']}"
          f"（按距离跳过 {stats['skipped_by_distance']}"
          f" / 超上限跳过 {stats['skipped_by_max_samples']}"
          f" / 手位置异常排除 {stats['excluded_invalid_hand_position']}）")
    print(f"[统计] 坐标系: {stats['coordinate_frame']}")
    print(f"[统计] 已完成真实机器人标定: {stats['robot_calibrated']}")
    print("[统计JSON]", flush=True)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="离线坐标标定准备：从 target_packets.jsonl 选取标定样本"
                    "（不启动 ROS2、不连接机械臂、不发送控制指令）")
    parser.add_argument("--input", required=True,
                        help="输入 target_packets.jsonl 路径")
    parser.add_argument("--output", required=True,
                        help="输出 calibration_samples.jsonl 路径"
                             "（默认拒绝覆盖已存在文件）")
    parser.add_argument("--min-distance", type=float,
                        default=DEFAULT_MIN_DISTANCE_M,
                        help="相邻样本 hand_position 最小欧氏距离（米，"
                             f"默认 {DEFAULT_MIN_DISTANCE_M}；0 关闭去重）")
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES,
                        help="最多保存的样本数（默认 0 = 不限制）")
    parser.add_argument("--overwrite", action="store_true",
                        help="允许覆盖已存在的输出文件")
    args = parser.parse_args(argv)
    if not (math.isfinite(args.min_distance) and args.min_distance >= 0):
        parser.error("--min-distance 必须为有限非负数")
    if args.max_samples < 0:
        parser.error("--max-samples 必须为非负整数")

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

    records = []
    with open(in_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    samples, stats = select_calibration_samples(
        records, min_distance=args.min_distance, max_samples=args.max_samples)

    tmp_path = out_path.with_name(out_path.name + f".tmp{os.getpid()}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            for sample in samples:
                fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
        os.replace(tmp_path, out_path)
    except OSError as exc:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        print(f"[错误] 写入失败: {exc}", file=sys.stderr)
        return 1

    stats["input"] = str(in_path)
    stats["output"] = str(out_path)
    _print_stats(stats)
    print(f"[结果] 输出: {out_path}")
    print("[结果] 当前仍未完成真实坐标标定：没有 robot_tcp_position，"
          "无法计算 HoloLens 到 RML63 的真实变换。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
