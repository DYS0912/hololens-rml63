"""Offline mock observation replay over ZMQ PUB.

Sends ONLY observation JSON. There is deliberately no receive path in this
module: no SUB socket, no action, no command — nothing can flow back from
the subscriber. No ROS2 node is created, no robot or camera is contacted.

Usage (offline only):
    python3 -m rml63_bridge.replay_observation \
        --count 100 --fps 30 --drop-invalid --address tcp://127.0.0.1:5557
"""

from __future__ import annotations

import argparse
import time
from typing import Iterator

from .mock_data import DEFAULT_FPS, DEFAULT_INVALID_EVERY, make_mock_observation
from .observation import Observation
from .transport import DEFAULT_ZMQ_ADDRESS, ZmqObservationTransport


def frame_sequence(
    count: int,
    fps: float = DEFAULT_FPS,
    invalid_every: int = DEFAULT_INVALID_EVERY,
    drop_invalid: bool = False,
    start_timestamp: float = 0.0,
) -> Iterator[Observation]:
    """Pure generator of the frames to send (unit-testable without ZMQ).

    With drop_invalid=True, invalid frames are filtered out before sending.
    """
    for index in range(count):
        obs = make_mock_observation(
            index, fps=fps, invalid_every=invalid_every, start_timestamp=start_timestamp
        )
        if drop_invalid and not obs.valid:
            continue
        yield obs


def main(args=None) -> None:
    parser = argparse.ArgumentParser(
        description="Replay mock RML63 observations over ZMQ PUB (offline, read-only)"
    )
    parser.add_argument("--count", type=int, default=10, help="总帧数")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="帧率（时间戳按 1/fps 递增）")
    parser.add_argument("--drop-invalid", action="store_true", help="过滤无效帧，只发送 valid=true")
    parser.add_argument("--address", default=DEFAULT_ZMQ_ADDRESS, help="ZMQ PUB 地址")
    parser.add_argument(
        "--invalid-every",
        type=int,
        default=DEFAULT_INVALID_EVERY,
        help="每 N 帧产生 1 帧 invalid；0 表示全部 valid",
    )
    args = parser.parse_args(args)
    if args.fps <= 0:
        parser.error("--fps 必须 > 0")
    if args.count < 0:
        parser.error("--count 必须 >= 0")

    transport = ZmqObservationTransport(args.address).start()
    print(
        f"replay: address={args.address} count={args.count} fps={args.fps} "
        f"invalid_every={args.invalid_every} drop_invalid={args.drop_invalid}"
    )
    print("replay: ZMQ PUB 已启动（只发送 observation JSON，无接收通道）")

    interval = 1.0 / args.fps
    next_t = time.monotonic()
    sent = 0
    try:
        for obs in frame_sequence(
            args.count,
            fps=args.fps,
            invalid_every=args.invalid_every,
            drop_invalid=args.drop_invalid,
        ):
            transport.send(obs)  # 只发送 observation JSON
            sent += 1
            next_t += interval
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        pass
    finally:
        transport.close()
    print(f"replay: sent {sent} frames")


if __name__ == "__main__":
    main()
