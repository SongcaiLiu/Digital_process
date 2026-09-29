import numpy as np
from unittest.mock import patch
from live_rgbd_system import ActionState

def face(y=200., scale=1.):
    p=np.zeros((478,3),dtype=np.float32)
    xs=np.linspace(-40,40,478,dtype=np.float32)*scale+320
    ys=np.linspace(-50,50,478,dtype=np.float32)*scale+y
    p[:,0]=xs;p[:,1]=ys
    for a,b in [(33,133),(362,263)]:
        p[a,:2]=[300,y];p[b,:2]=[330,y]
    for i in (160,158,385,387):p[i,:2]=[315,y-4]
    for i in (153,144,373,380):p[i,:2]=[315,y+4]
    return p

a=ActionState()
with patch('live_rgbd_system.head_pose_angles',side_effect=[(180,0,0),(184.5,4,0),(180,0,0)]):
    a.update(face(200),{},3.5,3.5)
    a.update(face(230),{},3.5,3.5)
    assert a.head_down_count==1,(a.head_down_count,a.reanchor_required)
    a.update(face(200),{},3.5,3.5)

b=ActionState()
with patch('live_rgbd_system.head_pose_angles',side_effect=[(180,0,0),(188,0,0),(188,0,0),(188,0,0),(188,0,0)]):
    b.update(face(200),{},3.5,3.5)
    b.update(face(200,1.25),{},3.5,3.5)
    assert b.head_down_count==0
print('PASS: vertical face motion does not suppress nod; strong scale change suppresses translation artifact')
