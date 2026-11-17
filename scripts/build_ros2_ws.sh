#!/usr/bin/env bash
# 构建 ROS2 工作空间（入口 1）
# 用法: bash scripts/build_ros2_ws.sh [额外 colcon 参数]
# 环境: 系统 Python 3.10（ROS2 Humble）
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="$PROJECT_ROOT/ros2_ws"

if [ ! -f /opt/ros/humble/setup.bash ]; then
    echo "错误: 未找到 /opt/ros/humble/setup.bash" >&2
    exit 1
fi
# 注意: 不能在 set -u 下 source ROS2 setup.bash（内部引用未导出变量）
source /opt/ros/humble/setup.bash
set -u

cd "$WS"
# 只构建本工程包与官方 SDK；不使用任何旧工作空间的 overlay
colcon build --symlink-install "$@"

echo
echo "构建完成。之后在本终端使用:"
echo "  source $WS/install/setup.bash"
echo "验证包唯一性: colcon list | grep rml63_bridge（应只有一行）"
