"""Deterministic calibration/state tests using explicit time and angles."""
from action_calibration import AutomaticActions
a=AutomaticActions()
def step(pitch=180,yaw=0,ear=.3,t=0):
    return a.update_measurements(pitch,yaw,0,ear,now=t)
for i in range(30): step(215,t=i*.05)
assert a.baseline is None, "held non-neutral tilt must not initialize"
for i in range(30): step(t=2+i*.05)
assert a.calibration_progress==1
baseline=a.baseline
for i in range(80): step(195,t=4+i*.05)
assert a.head_down_count==1 and a.baseline==baseline
step(t=9)
step(165,t=9.1)
assert a.head_up_count==1
step(t=9.2)
# Crossing the 180-degree representation boundary is a small angle.
step(-179,t=9.3)
assert a.head_down_count==1
# Missing face during closed eyes must not create a blink.
step(ear=.1,t=9.4);a.missing(9.5);step(t=9.6)
assert a.blink_count==0
# Large yaw suspends counts, then requires return to normal to rearm.
step(yaw=55,t=9.7);step(195,t=9.8)
assert a.head_down_count==1
step(t=9.9);step(195,t=10)
assert a.head_down_count==2
# Slowly drifting neutral can update a little, bounded by initial anchor.
step(t=10.1)
for i in range(200):step(182,t=10.2+i*.05)
assert 0<a.baseline-baseline<3
print("PASS: automatic neutral initialization, held actions, wraparound, loss, yaw and bounded adaptation")
