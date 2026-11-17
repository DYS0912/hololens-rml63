"""Offline reference gate. NOT connected to rm_driver or a robot.

Must run on the final command consumer side, serialised with SDK sends and stop.
Publisher-side use cannot invalidate the existing rm_driver subscriber queue.
All timestamps below are receiver-owned monotonic times. An epoch is issued by
the receiver after stop confirmation, never chosen by a stale sender.
"""
import math
import secrets

class CommandGate:
    def __init__(self, max_age=.15, max_tick_gap=.05):
        if not all(math.isfinite(x) and x>0 for x in (max_age,max_tick_gap)):
            raise ValueError('Invalid timeout')
        self.max_age,self.max_tick_gap=max_age,max_tick_gap
        self.state='STOPPED'
        self.epoch=None
        self.sequence=-1
        self.latest=None
        self.last_tick=None
        self.reason='Not armed'

    def arm(self, now):
        if self.state!='STOPPED' or not math.isfinite(now):
            raise ValueError('Stop confirmation required')
        self.epoch=secrets.token_hex(16)
        self.sequence=-1
        self.latest=None
        self.last_tick=now
        self.state='ARMED'
        return self.epoch

    def stop(self, reason):
        self.epoch=None
        self.latest=None
        self.state='STOP_PENDING'
        self.reason=reason

    def confirm_stop(self, *, sdk_stop_ok, fresh_stationary, send_worker_idle):
        if self.state!='STOP_PENDING': raise ValueError('No pending stop')
        if not (sdk_stop_ok is True and fresh_stationary is True and send_worker_idle is True):
            return False
        self.state='STOPPED'
        return True

    def receive(self, epoch, sequence, position, received_at):
        if self.state!='ARMED' or epoch!=self.epoch:
            return False
        if (type(sequence) is not int or sequence<=self.sequence
                or not math.isfinite(received_at)
                or received_at<self.last_tick
                or len(position)!=3
                or any(type(v) not in (int,float) or not math.isfinite(v) for v in position)):
            self.stop('Invalid or reordered command')
            return False
        self.sequence=sequence
        self.latest=(received_at,tuple(position))  # replace, never append
        return True

    def take_latest(self, now):
        if self.state!='ARMED': return None
        if not math.isfinite(now) or not 0<=now-self.last_tick<=self.max_tick_gap:
            self.stop('Command worker timing gap')
            return None
        self.last_tick=now
        if self.latest is None:
            # No startup jump or implicit hold target.
            return None
        received_at,position=self.latest
        if not 0<=now-received_at<self.max_age:
            self.stop('Target expired')
            return None
        return position
