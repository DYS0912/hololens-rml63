"""离线 dataflow 模拟器：不安装 dora 即可按 dataflow 语义跑通全链路。

    rml63_state (mock) ─┐
    d435i_camera (mock)─┼─→ rml63_dataset_recorder ─→ LeRobot 0.6.2 Parquet
    mock_action ────────┘

每个 tick 依次执行：state/camera/action 节点产出 -> recorder 输入通道 ->
控制命令 -> recorder 心跳。与 dataflows/rml63_d435i_record.yml 的通道
连接一一对应。仅离线：无 ROS2、无硬件、无 Dora、不发送控制指令。

用法:
    python3 -m rml63_bridge.dora_nodes.simulate_dataflow \
        --count 12 --dataset-root /tmp/rml63_dora_stage \
        --output-root /tmp/rml63_dora_lerobot

模拟 action 仅用于数据格式验证，不代表真实示范动作，不用于研究结论。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from rml63_bridge.convert_to_lerobot import MOCK_ACTION_NOTICE
from rml63_bridge.camera_packet import DEFAULT_FPS
from rml63_bridge.mock_data import DEFAULT_INVALID_EVERY
from rml63_bridge.dora_nodes.d435i_camera_dora_node import CameraNodeCore
from rml63_bridge.dora_nodes.mock_action_dora_node import MockActionCore
from rml63_bridge.dora_nodes.rml63_dataset_recorder_dora_node import RecorderNodeCore
from rml63_bridge.dora_nodes.rml63_state_dora_node import StateNodeCore


@dataclass
class SimulationResult:
    dataset_root: str
    output_root: str
    frames_in: int = 0
    frames_written: int = 0
    action_frames: int = 0
    episodes_committed: int = 0
    skipped: Dict[str, int] = field(default_factory=dict)
    final: dict = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"staging(root): {self.dataset_root}",
            f"LeRobot 输出:  {self.output_root}",
            f"输入 tick:     {self.frames_in}",
            f"写入帧:        {self.frames_written}",
            f"含 action 帧:  {self.action_frames}",
            f"提交 episodes: {self.episodes_committed}",
            f"跳过统计:      {json.dumps(self.skipped, ensure_ascii=False)}",
            f"声明: {MOCK_ACTION_NOTICE}",
        ]
        if self.errors:
            lines.append("错误/拒绝:")
            lines += [f"  - {e}" for e in self.errors]
        return "\n".join(lines)


def run_dataflow_simulation(
    count: int = 12,
    fps: float = DEFAULT_FPS,
    invalid_every: int = DEFAULT_INVALID_EVERY,
    action_mode: str = "mock",
    dataset_root: str = "/tmp/rml63_dora_stage",
    output_root: str = "/tmp/rml63_dora_lerobot",
    task: str = "rebar_tying",
    robot_type: str = "rml63",
    max_time_delta: float = 0.05,
    require_action: bool = True,
    control_commands: Optional[Dict[int, str]] = None,
    control_file: Optional[str] = None,
    finalize: bool = True,
) -> SimulationResult:
    """按 dataflow 通道语义跑一轮离线模拟。

    control_commands: {tick_index: "save"|"next"|"discard"|"stop"}，
    在对应 tick 的数据喂入后、recorder 心跳前执行。
    finalize=False 模拟录制被中断（用于测试元数据恢复）。
    """
    state = StateNodeCore(input_mode="mock", mock_fps=fps, mock_invalid_every=invalid_every)
    camera = CameraNodeCore(input_mode="mock", mock_fps=fps, mock_invalid_every=invalid_every)
    action = MockActionCore(mode=action_mode)
    recorder = RecorderNodeCore(
        dataset_root=dataset_root,
        output_root=output_root,
        task=task,
        fps=fps,
        robot_type=robot_type,
        max_time_delta=max_time_delta,
        require_action=require_action,
        control_file=control_file,
    )
    commands = dict(control_commands or {})

    result = SimulationResult(dataset_root=dataset_root, output_root=output_root)
    for i in range(count):
        now = i / fps
        state_out = state.on_tick()
        camera_out = camera.on_tick()
        action_out = action.on_tick()

        if action_out is not None:
            recorder.on_action(action_out, now=now)
        if state_out:
            recorder.on_state(**state_out, now=now)
        if camera_out:
            recorder.on_camera(**camera_out, now=now)

        cmd = commands.get(i)
        if cmd:
            recorder.apply_command(cmd)
            if not recorder.running:  # stop 命令提前结束
                break
        recorder.on_tick(now=now)

        result.frames_in += 1
        if not recorder.running:
            break

    result.frames_written = recorder.frames_written
    result.action_frames = recorder.action_frames
    result.skipped = dict(recorder.skipped)

    if finalize:
        final = recorder.finalize()
        result.episodes_committed = final.get("episodes_committed", recorder.episodes_committed)
        result.errors = list(final.get("errors", []))
        result.final = final
    else:
        # 模拟被中断：只把已开启的 episode 落盘（不写 meta、不转换）
        if recorder._episode_open and recorder._episode_frames > 0:
            recorder.writer.finish_episode()
        result.episodes_committed = recorder.episodes_committed
        result.errors = ["模拟录制中断（未 finalize）——可用 recover_offline_dataset 恢复"]
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="离线 dataflow 模拟器（无 Dora/无 ROS2/无硬件）"
    )
    parser.add_argument("--count", type=int, default=12, help="模拟 tick 数")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--invalid-every", type=int, default=DEFAULT_INVALID_EVERY,
                        help="每 N 帧注入 1 帧 invalid（<=0 关闭）")
    parser.add_argument("--action-mode", choices=["mock", "none"], default="mock",
                        help="none = 全部帧无 action（验证拒绝训练数据）")
    parser.add_argument("--dataset-root", default="/tmp/rml63_dora_stage",
                        help="离线 staging 数据集目录")
    parser.add_argument("--output-root", default="/tmp/rml63_dora_lerobot",
                        help="LeRobot 0.6.2 输出目录")
    parser.add_argument("--task", default="rebar_tying")
    parser.add_argument("--robot-type", default="rml63")
    parser.add_argument("--max-time-delta", type=float, default=0.05)
    parser.add_argument("--no-require-action", action="store_true",
                        help="关闭 action 必需校验（默认开启）")
    args = parser.parse_args(argv)

    result = run_dataflow_simulation(
        count=args.count,
        fps=args.fps,
        invalid_every=args.invalid_every,
        action_mode=args.action_mode,
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        task=args.task,
        robot_type=args.robot_type,
        max_time_delta=args.max_time_delta,
        require_action=not args.no_require_action,
    )
    print(result.summary())
    return 1 if result.errors else 0


if __name__ == "__main__":
    sys.exit(main())
