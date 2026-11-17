"""Offline translation planner. No ROS, sockets, SDK or hardware output.
Workspace bounds must be surveyed in robot Base coordinates, not inferred here.
"""
import math

def vec(x):
    if len(x)!=3 or any(type(v) not in (int,float) or not math.isfinite(v) for v in x):
        raise ValueError('Expected finite 3-vector')
    return list(x)

class FollowTarget:
    def __init__(self, lower, upper, speed=.02, acceleration=.05):
        self.lower,self.upper=vec(lower),vec(upper)
        if any(a>=b for a,b in zip(self.lower,self.upper)):
            raise ValueError('Invalid workspace')
        if not all(type(x) in (int,float) and math.isfinite(x) and x>0 for x in (speed,acceleration)):
            raise ValueError('Invalid limits')
        self.speed,self.acceleration=speed,acceleration
        self.active=False
        self.stop_pending=False
        self.release_seen=False
        self.velocity=[0.,0.,0.]

    def inside(self,p):
        return all(a<=v<=b for a,v,b in zip(self.lower,p,self.upper))

    def release(self):
        if self.active:
            self.active=False
            self.stop_pending=True
            self.release_seen=False
        elif not self.stop_pending:
            self.release_seen=True

    def fault(self):
        self.active=False
        self.stop_pending=True
        self.release_seen=False

    def confirm_stopped(self):
        # Transport layer must confirm fresh stationary feedback and command drain.
        # Does not automatically rearm. A later release is required.
        self.stop_pending=False
        self.release_seen=False
        self.velocity=[0.,0.,0.]

    def begin(self,hand,robot):
        hand,robot=vec(hand),vec(robot)
        if self.active or self.stop_pending or not self.release_seen:
            raise ValueError('Not armed')
        if not self.inside(robot): raise ValueError('Robot outside workspace')
        self.hand_anchor=hand
        self.robot_anchor=robot
        self.position=list(robot)
        self.velocity=[0.,0.,0.]
        self.release_seen=False
        self.active=True

    def step(self,hand,dt):
        hand=vec(hand)
        if not self.active: return None
        if type(dt) not in (int,float) or not math.isfinite(dt) or not 0<dt<=.05:
            self.fault()
            raise ValueError('Scheduler gap; stop required')
        delta=[hand[i]-self.hand_anchor[i] for i in (0,2,1)]
        desired=[a+b for a,b in zip(self.robot_anchor,delta)]
        if not self.inside(desired):
            self.fault()
            raise ValueError('Workspace exceeded; stop required')
        requested=[(b-a)/dt for a,b in zip(self.position,desired)]
        norm=math.hypot(*requested)
        requested=[v*min(1,self.speed/max(norm,1e-12)) for v in requested]
        dv=[a-b for a,b in zip(requested,self.velocity)]
        factor=min(1,self.acceleration*dt/max(math.hypot(*dv),1e-12))
        velocity=[v+factor*d for v,d in zip(self.velocity,dv)]
        candidate=[p+v*dt for p,v in zip(self.position,velocity)]
        if not self.inside(candidate):
            self.fault()
            raise ValueError('Limited trajectory exceeds workspace')
        self.position,self.velocity=candidate,velocity
        return list(candidate)
