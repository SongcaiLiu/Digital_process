"""Regression tests for time-based blink detection."""
from action_calibration import AutomaticActions

def ready():
    a=AutomaticActions(); a.baseline=a.anchor=180.; a.yaw_anchor=0.; a.ear_ref=.30; a.calibration_progress=1.
    return a

def step(a,t,ear=.30,pitch=180.):
    return a.update_measurements(pitch,0.,0.,ear,now=t)

a=ready()
step(a,0); step(a,.10,.10); step(a,.20,.10); step(a,.30,.30); step(a,.40,.30)
assert a.blink_count==1, a.blink_count

a=ready()
step(a,0); step(a,.10,.10); step(a,.20,.10,195); step(a,.30,.24,165)
step(a,.40,.10,180); step(a,.50,.30,180); step(a,.60,.30,180)
assert a.blink_count==0, a.blink_count

a=ready()
step(a,0); step(a,.10,.10); step(a,.80,.10); step(a,.90,.30); step(a,1.0,.30)
assert a.blink_count==0, a.blink_count

a=ready()
step(a,0); step(a,.10,.10); a.missing(.20); step(a,.30,.30); step(a,.40,.30)
assert a.blink_count==0, a.blink_count
# A short, less tightly closed blink at about 15 FPS must still register.
a=ready()
step(a,0); step(a,.067,.195,182); step(a,.134,.30,181); step(a,.201,.30,180)
assert a.blink_count==1, a.blink_count

print('PASS: normal blink, pitch-motion rejection, long closure and missing face')
