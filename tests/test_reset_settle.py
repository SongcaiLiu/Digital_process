from action_calibration import AutomaticActions

a=AutomaticActions()
a.begin_reference_reset()
for i in range(20):
    t=i*.05
    pitch=180.+(.8 if i%2 else -.6)
    head,_,_,_=a.update_measurements(pitch,0,0,.30,now=t)
    assert a.head_down_count==a.head_up_count==0
assert not a.reset_settle_active
assert abs(a.baseline-180.)<1.
for i in range(20,40):
    a.update_measurements(180.+(.8 if i%2 else -.6),0,0,.30,now=i*.05)
assert a.head_down_count==a.head_up_count==0

a=AutomaticActions()
a.begin_reference_reset()
a.update_measurements(180,0,0,.30,now=0)
a.update_measurements(188,0,0,.30,now=.1)
assert a.reset_settle_active and a.head_down_count==0
for i in range(2,18):a.update_measurements(188,0,0,.30,now=i*.05)
assert not a.reset_settle_active and abs(a.baseline-188)<.1
assert a.head_down_count==0
print('PASS: C-reset ignores first-frame noise and motion, then arms from stable median')
