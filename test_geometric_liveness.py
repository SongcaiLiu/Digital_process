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

# One cheek occluded: sufficient global coverage and remaining cheek still work.
occluded=z.copy(); occluded[65:96,130:160]=0
assert run(occluded)["state"]=="LIVE", run(occluded)
# Large missing areas and near-range objects must not be labelled photo.
holes=np.full((h,w),.8); holes[:,20:130]=0
assert run(holes)["state"]=="UNKNOWN"
assert run(np.full((h,w),.25))["state"]=="UNKNOWN"
# Missing nose on a perfect plane cannot turn it into a live face.
plane=np.full((h,w),.8);plane[70:91,90:111]=0
assert run(plane)["state"]=="PHOTO",run(plane)
print("PASS: single-cheek visibility, coverage/range gates and flat surface with nose holes")

# Distributed holes leave only 35% depth but enough independent plane evidence.
partial=np.full((h,w),.8)
partial[rng.random((h,w))>.35]=0
r=run(partial)
assert r["state"]=="PHOTO",r
# The same sparse sampling of a curved surface cannot bypass live coverage rules.
partial_face=z.copy();partial_face[partial==0]=0
r=run(partial_face)
assert r["state"]=="UNKNOWN",r
print("PASS: sparse distributed plane accepted, sparse curved face rejected")
