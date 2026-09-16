"""Synthetic geometry checks, not real-device accuracy measurements."""
import numpy as np
from geometric_liveness import analyze_geometry, DEFAULTS
h,w=160,200
y,x=np.indices((h,w)); intr=dict(fx=220.,fy=220.,ppx=100.,ppy=80.)
mask=np.zeros((h,w),np.uint8);mask[15:145,20:180]=255
lm=np.zeros((478,3))
for ids,xy in [([1,4,5],(100,80)),([50,101,205],(55,80)),([280,330,425],(145,80))]:
    lm[ids,:2]=xy
def run(z):
    return analyze_geometry(np.round(z*1000).astype(np.uint16),.001,intr,mask,lm)
for ax,ay in [(0,0),(.4,0),(0,.4),(.4,.4),(-.4,.3)]:
    z=.8/(1+ax*(x-100)/220+ay*(y-80)/220)
    r=run(z)
    assert r["state"]=="PHOTO",r
z=.8-.035*np.exp(-((x-100)**2/400+(y-80)**2/700))
r=run(z)
assert r["state"]=="LIVE",r
assert run(np.zeros((h,w)))["state"]=="UNKNOWN"
rng=np.random.default_rng(7)
r=run(.8+rng.normal(0,.012,(h,w)))
assert r["state"]=="UNKNOWN",r
print("PASS: front/tilted/diagonal planes, curved face, missing and noisy depth")
