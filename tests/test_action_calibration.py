"""Head action regression tests with explicit timestamps."""
from action_calibration import AutomaticActions

def step(a,pitch=180.,yaw=0.,t=0.,translation=False):
    return a.update_measurements(pitch,yaw,0.,.30,now=t,translation=translation)

# A single-frame nod counts promptly; holding it cannot count again.
a=AutomaticActions()
step(a,180,t=0)
assert step(a,185,t=.1)[0]=='HEAD_DOWN'
for i in range(1,31):step(a,185,t=.1+i*.05)
assert a.head_down_count==1

# A brief return toward neutral or liveness dropout must keep the gesture latched.
step(a,180,t=1.7)
step(a,185,t=1.8)
a.missing(1.85)
step(a,185,t=1.9)
assert a.head_down_count==1

# Only a sustained neutral return rearms the next gesture.
for i in range(12):step(a,180,t=2.0+i*.05)
assert a.head_state=='NORMAL'
step(a,185,t=2.7)
assert a.head_down_count==2

# Returning from a nod displays NORMAL immediately, while the same cycle
# stays locked even if the rebound briefly crosses the opposite threshold.
r=AutomaticActions()
step(r,180,t=0)
step(r,185,t=.1)
assert step(r,180,t=.2)[0]=='NORMAL'
step(r,175,t=.3)
assert r.head_down_count==1 and r.head_up_count==0
for i in range(12):step(r,180,t=.4+i*.05)
assert r.head_state=='NORMAL'

# A short non-LIVE interval can track a pending nod, but never count on the
# non-LIVE frame. It counts once only if LIVE resumes while the nod persists.
g=AutomaticActions()
step(g,180,t=0)
g.update_measurements(185,0,0,.30,now=.1,count_enabled=False)
assert g.head_down_count==0
g.update_measurements(185,0,0,.30,now=.2,count_enabled=True)
assert g.head_down_count==1

# Upward gesture uses the same latch and cooldown.
b=AutomaticActions()
step(b,180,t=0)
step(b,175,t=.1)
for i in range(1,31):step(b,175,t=.1+i*.05)
assert b.head_up_count==1

# Whole-face translation does not count, then stable local pose can count.
c=AutomaticActions()
step(c,180,t=0)
step(c,188,t=.1,translation=True)
assert c.head_down_count==0
for t in (.2,.35,.55):step(c,188,t=t)
assert not c.reanchor_required
step(c,193,t=.7)
assert c.head_down_count==1

# Side pose supports gestures, while extreme profile is unknown.
d=AutomaticActions()
step(d,188,45,t=0)
step(d,188,25,t=.1);step(d,190,5,t=.2)
for t in (.3,.45,.65):step(d,190,5,t=t)
assert not d.reanchor_required
step(d,190,25,t=.8);step(d,193,55,t=.9)
for t in (1.0,1.2,1.4):step(d,193,55,t=t)
assert not d.reanchor_required
step(d,201,55,t=1.55)
assert d.head_down_count==1
assert step(d,193,75,t=1.65)[0]=='UNKNOWN'
print('PASS: prompt gestures, held-pose lock, dropout, neutral rearm, translation and side pose')
