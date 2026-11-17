"""HoloLens pinch triggers ONE fixed Base +Z 5mm stop test, never hand following.
Default is read-only. --run-single-test explicitly enables the one-shot test.
Arrival time is not proof of sender freshness; use only on the verified local link.
"""
import argparse
import copy
import json
import math
import signal
import socket
import time
from pathlib import Path


def finite(values):
    return all(math.isfinite(v) for v in values)


class PinchGate:
    def __init__(self):
        self.session = None
        self.sequence = -1
        self.last = None
        self.open_since = self.pinch_since = None
        self.armed = self.fired = self.held = False
        self.interrupted = False

    def invalidate(self):
        self.open_since = self.pinch_since = None
        self.armed = self.held = False
        if self.fired:
            self.interrupted = True

    def tick(self, now):
        if self.last is None or now-self.last >= .2:
            self.invalidate()

    def receive(self, raw, now):
        self.tick(now)
        try:
            m = json.loads(raw)
            expected = dict(schema='rml63.hand_pose.v1', source='hololens2',
                            hand='right', pose_type='openxr_grip',
                            coordinate_frame='unity_tracking_space',
                            position_unit='m', quaternion_order='xyzw')
            if not isinstance(m, dict) or any(m.get(k)!=v for k,v in expected.items()):
                raise ValueError('协议不匹配')
            if any(type(m.get(k)) is not bool for k in ('valid','tracked','control_enabled')):
                raise ValueError('布尔字段异常')
            sid, seq = m.get('session_id'), m.get('sequence')
            if not isinstance(sid,str) or not sid or type(seq) is not int or seq<0:
                raise ValueError('会话/序号异常')
            if sid != self.session:
                if self.session is not None:
                    self.invalidate()
                    # Session change ends this run; avoids alternating stale sessions.
                    raise RuntimeError('头显会话改变，请结束后重新启动测试')
                self.session, self.sequence = sid, -1
            if seq <= self.sequence:
                raise ValueError('重复/倒退包')
            self.sequence, self.last = seq, now
            if not m['valid'] or not m['tracked']:
                raise ValueError('追踪不可用')
            for key, size in (('position',3),('rotation',4)):
                v = m.get(key)
                if (not isinstance(v,list) or len(v)!=size or
                        any(type(x) not in (int,float) for x in v) or not finite(v)):
                    raise ValueError('姿态异常')
            if abs(math.sqrt(sum(x*x for x in m['rotation']))-1)>.01:
                raise ValueError('四元数异常')
            self.held = m['control_enabled']
            if not self.held:
                if self.fired:
                    self.interrupted = True
                self.pinch_since = None
                if self.open_since is None:
                    self.open_since = now
                if now-self.open_since >= .3:
                    self.armed = True
                return False
            self.open_since = None
            if not self.armed or self.fired:
                return False
            if self.pinch_since is None:
                self.pinch_since = now
            if now-self.pinch_since >= .2:
                self.fired = True
                return True
        except (ValueError, TypeError, OverflowError):
            self.invalidate()
        return False


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
    node = rclpy.create_node('hololens_single_trigger')
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
    path = folder / ('single_trigger_' + str(time.time_ns()) + '.jsonl')
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

    hand_socket = None
    gate = PinchGate()

    def read_hand():
        triggered = False
        for _ in range(64):
            try:
                raw, peer = hand_socket.recvfrom(65535)
            except BlockingIOError:
                break
            if peer[0] != '172.20.10.3':
                continue
            triggered = gate.receive(raw, time.monotonic()) or triggered
        else:
            raise RuntimeError('UDP积压/流量过大，结束本次测试')
        gate.tick(time.monotonic())
        return triggered and gate.held and not gate.interrupted

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
        hand_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        hand_socket.bind(('172.20.10.2', 5005))
        hand_socket.setblocking(False)
        print('[手势等待] 45秒内：右手保持可见，先张开1秒，再捏合保持。', flush=True)
        print('[范围] 固定Base +Z目标5mm；不会跟随手的位置，触发一次后结束。', flush=True)
        deadline = time.monotonic()+45
        while time.monotonic()<deadline:
            spin()
            if cancel[0] or not fresh():
                raise RuntimeError('取消或机器人反馈过期，未发送运动')
            if read_hand():
                record('gesture_triggered', {'session':gate.session, 'sequence':gate.sequence})
                print('[手势触发] 已识别张开后持续捏合', flush=True)
                break
            if gate.fired:
                raise RuntimeError('触发同批次发生松开/失效，未发送运动')
        else:
            raise RuntimeError('45秒内没有合格手势，未发送运动')
        if not args.run_single_test:
            print('[只读完成] 手势触发成功，未创建运动发布器，未发送运动。', flush=True)
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
            read_hand()
            p = latest['pose'][1].position
            dist = math.sqrt(sum((getattr(p,k)-getattr(baseline.position,k))**2 for k in ('x','y','z')))
            if cancel[0]:
                stop('用户取消')
            elif not fresh():
                stop('机器人反馈过期')
            elif not gate.held or gate.interrupted:
                stop('松手、追踪失效或头显数据超时')
            elif dist>=.0005:
                stop('观测位移达到0.5mm')
            elif time.monotonic()-sent_at>=.15:
                stop('发送运动后150ms')
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
                deadline = time.monotonic()+15
                while time.monotonic()<deadline:
                    spin()
        finally:
            if hand_socket is not None:
                hand_socket.close()
            log.close()
            for s, handler in previous.items():
                signal.signal(s, handler)
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
            print('[日志已保存]', path, flush=True)


if __name__ == '__main__':
    main()
