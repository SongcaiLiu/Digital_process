"""Brief liveness dropouts during a known live gesture may still count."""
import numpy as np
from unittest.mock import patch
from live_rgbd_system import ActionState

face=np.zeros((478,3),dtype=np.float32)
face[:,0]=np.linspace(280,360,478)
face[:,1]=np.linspace(150,250,478)
for a,b in ((33,133),(362,263)):
    face[a,:2]=[300,200];face[b,:2]=[330,200]
for i in (160,158,385,387):face[i,:2]=[315,196]
for i in (153,144,373,380):face[i,:2]=[315,204]
clock=[0.]
pitch=[180.]

with patch('live_rgbd_system.monotonic',side_effect=lambda:clock[0]), \
     patch('action_calibration.monotonic',side_effect=lambda:clock[0]), \
     patch('live_rgbd_system.head_pose_angles',side_effect=lambda *_:(pitch[0],0.,0.)):
    a=ActionState()
    assert a.update_if_live(face,{},3.5,3.5,'PHOTO')[0]=='UNKNOWN'
    assert a.head_down_count==0
    a.update_if_live(face,{},3.5,3.5,'LIVE')
    assert a.last_confirmed_live==0.
    clock[0]=.1;pitch[0]=185.
    assert a.update_if_live(face,{},3.5,3.5,'PHOTO')[0]=='HEAD_DOWN'
    assert a.head_down_count==1
    assert a.last_confirmed_live==0.  # Red frames never extend the window.
    clock[0]=.25;pitch[0]=180.
    a.update_if_live(face,{},3.5,3.5,'UNKNOWN')
    clock[0]=.7;pitch[0]=185.
    assert a.update_if_live(face,{},3.5,3.5,'PHOTO')[0]=='UNKNOWN'
    assert a.head_down_count==1

    # A blink that finishes during the same short grace window can count too.
    b=ActionState()
    clock[0]=2.;pitch[0]=180.
    b.update_if_live(face,{},3.5,3.5,'LIVE')
    for i in (160,158,385,387):face[i,1]=199.5
    for i in (153,144,373,380):face[i,1]=200.5
    clock[0]=2.1;b.update_if_live(face,{},3.5,3.5,'PHOTO')
    for i in (160,158,385,387):face[i,1]=196
    for i in (153,144,373,380):face[i,1]=204
    clock[0]=2.18;b.update_if_live(face,{},3.5,3.5,'PHOTO')
    clock[0]=2.24;b.update_if_live(face,{},3.5,3.5,'PHOTO')
    assert b.blink_count==1
    assert b.last_confirmed_live==2.
print('PASS: short red-frame gestures count; prolonged red frames do not')
