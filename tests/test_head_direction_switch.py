"""Regression checks for alternating gestures and 6DRepNet side-view thresholds."""
from action_calibration import AutomaticActions

def pose(a, pitch, time, yaw=0):
    return a.update_measurements(pitch,yaw,0,.30,down=10,up=10,now=time)

a=AutomaticActions("positive",side_factor_scale=.35)
pose(a,180,0)
pose(a,192,.1)
pose(a,180,.2)
pose(a,180,.3)
pose(a,180,.42)
assert pose(a,168,.7)[0]=="HEAD_UP"
assert (a.head_down_count,a.head_up_count)==(1,1)
pose(a,180,.8)
pose(a,180,.95)
pose(a,180,1.05)
assert pose(a,192,1.3)[0]=="HEAD_DOWN"
assert (a.head_down_count,a.head_up_count)==(2,1)

# A rebound without a settled neutral pause must not become an opposite count.
b=AutomaticActions("positive",side_factor_scale=.35)
pose(b,180,0);pose(b,192,.1);pose(b,180,.2)
pose(b,168,.3);pose(b,168,.7)
assert (b.head_down_count,b.head_up_count)==(1,0)

# At 55 degrees yaw the experimental threshold is 13.675 degrees, not 20.5.
c=AutomaticActions("positive",side_factor_scale=.35)
pose(c,180,0,55)
assert pose(c,194,.1,55)[0]=="HEAD_DOWN"
assert c.head_down_count==1
print("PASS: alternating directions, rebound lock, 6D side threshold")
