"""ROS2 single-command diagnostic. Default: read-only. No HoloLens input."""
import argparse
import copy
import json
import math
import signal
import time
from pathlib import Path


def finite(values):
    return all(math.isfinite(v) for v in values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-single-test', action='store_true',
                        help='Send exactly one Base +Z 5mm target, then stop')
    args = parser.parse_args()
    import rclpy
    from rclpy.signals import SignalHandlerOptions
    from geometry_msgs.msg import Pose
    from std_msgs.msg import Bool, Empty
    from rm_ros_interfaces.msg import Movel
    from rosidl_runtime_py.utilities import get_message

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node('robot_stop_check_v2')
    origin = time.monotonic()
    events = []
    latest = {}
    results = {}
    cancel = [False]
    motion_attempted = False
    stop_attempted = False
    stop_at = None
    subscriptions = []
    folder = Path.home() / 'rml63_robot_project/datasets/hololens_checks'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ('stop_check_v2_' + str(time.time_ns()) + '.jsonl')
    log = path.open('x', encoding='utf-8', buffering=1)

    def record(kind, value):
        event = dict(t=time.monotonic()-origin, kind=kind, value=value)
        events.append(event)
        log.write(json.dumps(event, ensure_ascii=False, allow_nan=False)+'\n')

    def pose_cb(msg):
        p, q = msg.position, msg.orientation
        values = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
        if not finite(values) or abs(math.sqrt(sum(v*v for v in values[3:]))-1) > .01:
            latest.pop('pose', None)
            return
        latest['pose'] = (time.monotonic(), copy.deepcopy(msg))
        record('pose', values)

    def speed_cb(msg):
        values = list(msg.joint_speed)
        if len(values) != 6 or not finite(values):
            latest.pop('speed', None)
            return
        latest['speed'] = (time.monotonic(), values)
        record('joint_speed', values)

    def result_cb(kind, msg):
        results[kind] = (time.monotonic(), bool(msg.data))
        record(kind, bool(msg.data))
        print('[回执]', kind, bool(msg.data), flush=True)

    for topic, cls, callback in [
        ('udp_arm_position', Pose, pose_cb),
        ('movel_result', Bool, lambda m: result_cb('movel_result', m)),
        ('move_stop_result', Bool, lambda m: result_cb('stop_result', m)),
    ]:
        subscriptions.append(node.create_subscription(cls, '/rm_driver/'+topic, callback, 200))
    move_pub = stop_pub = None
    if args.run_single_test:
        move_pub = node.create_publisher(Movel, '/rm_driver/movel_cmd', 1)
        stop_pub = node.create_publisher(Empty, '/rm_driver/move_stop_cmd', 1)

    def on_signal(signum, frame):
        cancel[0] = True

    previous = {s: signal.signal(s, on_signal) for s in (signal.SIGINT, signal.SIGTERM)}

    def spin():
        rclpy.spin_once(node, timeout_sec=.005)

    def fresh():
        now = time.monotonic()
        return all(k in latest and now-latest[k][0] < .2 for k in ('pose', 'speed'))

    def stop(reason):
        nonlocal stop_at, stop_attempted
        if stop_attempted:
            return
        stop_attempted = True
        stop_at = time.monotonic()
        stop_pub.publish(Empty())
        record('stop_requested', reason)
        print('[停止请求]', reason, flush=True)

    try:
        print('[模式]', '单次运动验证' if args.run_single_test else '只读，不创建运动/停止发布器', flush=True)
        print('[日志]', path, flush=True)
        # Discover the locally installed speed message type instead of guessing it.
        deadline = time.monotonic()+5
        speed_subscribed = False
        while time.monotonic() < deadline and not cancel[0]:
            if not speed_subscribed:
                types = dict(node.get_topic_names_and_types()).get('/rm_driver/udp_joint_speed', [])
                if len(types) == 1:
                    cls = get_message(types[0])
                    if not hasattr(cls(), 'joint_speed'):
                        raise RuntimeError('关节速度消息没有 joint_speed 字段')
                    subscriptions.append(node.create_subscription(cls, '/rm_driver/udp_joint_speed', speed_cb, 200))
                    speed_subscribed = True
            spin()
        if cancel[0]:
            raise RuntimeError('用户取消，未发送运动')
        if not fresh():
            raise RuntimeError('位置或六轴速度反馈缺失/过期，未发送运动')
        print('[当前位置]', latest['pose'][1].position, flush=True)
        print('[六轴速度]', latest['speed'][1], flush=True)
        if not args.run_single_test:
            print('[只读检查完成] 未发送运动；这不是停止能力验收。', flush=True)
            return
        recent = [e for e in events if e['t'] >= time.monotonic()-origin-.5 and e['kind']=='joint_speed']
        if len(recent)<10 or any(max(abs(v) for v in e['value'])>.1 for e in recent):
            raise RuntimeError('未获得连续低速反馈，未发送运动')
        if any(b['t']-a['t']>.05 for a,b in zip(recent,recent[1:])) or recent[-1]['t']-recent[0]['t'] < .4:
            raise RuntimeError('速度反馈窗口不足或有间断，未发送运动')
        if move_pub.get_subscription_count()!=1 or stop_pub.get_subscription_count()!=1:
            raise RuntimeError('运动/停止订阅者数量异常，未发送运动')
        if node.count_publishers('/rm_driver/movel_cmd')!=1:
            raise RuntimeError('发现其他 Movel 发布者，未发送运动')
        baseline = copy.deepcopy(latest['pose'][1])
        cmd = Movel()
        cmd.pose = copy.deepcopy(baseline)
        cmd.pose.position.z += .005
        cmd.speed, cmd.trajectory_connect, cmd.block = 1, 0, True
        results.clear()
        motion_attempted = True
        sent_at = time.monotonic()
        move_pub.publish(cmd)
        record('motion_sent', {'start': [baseline.position.x, baseline.position.y, baseline.position.z],
                               'delta_base_m': [0, 0, .005], 'speed_percent': 1})
        print('[运动] 仅一次 Base +Z 5mm 目标，不自动重试或返回起点', flush=True)
        while stop_at is None:
            spin()
            p = latest['pose'][1].position
            dist = math.sqrt(sum((getattr(p,k)-getattr(baseline.position,k))**2 for k in ('x','y','z')))
            if cancel[0] or not fresh() or dist>=.0005 or time.monotonic()-sent_at>=.15:
                stop('取消/反馈过期/位移达到0.5mm/发送后150ms阈值')
        # Keep ROS alive for late motion completion. Never send a second target.
        while time.monotonic()-stop_at < 15:
            spin()
            if time.monotonic()-stop_at>=3 and all(k in results for k in ('movel_result','stop_result')):
                break
        missing = [k for k in ('movel_result','stop_result') if k not in results]
        record('diagnostic_end', {'missing_results': missing, 'feedback_fresh': fresh()})
        print('[结束] 缺少回执:', missing, '；反馈新鲜:', fresh(), flush=True)
        print('不自动判定通过。movel_result=False须结合驱动原始错误码解释。', flush=True)
    except Exception as exc:
        record('error', str(exc))
        print('[错误]', exc, flush=True)
    finally:
        try:
            if motion_attempted and not stop_attempted and rclpy.ok():
                stop('异常退出')
                deadline = time.monotonic()+3
                while time.monotonic()<deadline:
                    spin()
        finally:
            log.close()
            for s, handler in previous.items():
                signal.signal(s, handler)
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
            print('[日志已保存]', path, flush=True)


if __name__ == '__main__':
    main()
