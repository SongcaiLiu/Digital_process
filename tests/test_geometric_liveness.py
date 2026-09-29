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
assert run(np.full((h,w),.15))["state"]=="UNKNOWN"
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

# Previously excluded nonzero planes now reach geometry; live range stays narrower.
for distance in (.25,.35,1.7,2.5):
    r=run(np.full((h,w),distance))
    assert r["state"]=="PHOTO",r
    assert abs(r["face_distance_m"]-distance)<.002,r
for distance in (.15,3.5):
    r=run(np.full((h,w),distance))
    assert r["state"]=="UNKNOWN",r
    assert r["too_near_ratio"]+r["too_far_ratio"]>.99,r
for distance in (.3,2.):
    curve=distance-.035*np.exp(-((x-100)**2/400+(y-80)**2/700))
    r=run(curve)
    assert r["state"]!="LIVE",r
print("PASS: extended plane range, separate near/far diagnostics, unchanged live range")

# A slightly turned face can yield a numeric but unreliable nose/cheek value.
saved=lm.copy()
lm[[1,4,5],:2]=(55,80)
coherent=.8+.055*((x-100)/80)**2+.035*((y-80)/65)**2
r=run(coherent)
assert r["state"]=="LIVE",r
assert "整体连续曲面" in r["reason"],r
# Plane classification remains first, even with misplaced nose landmarks.
for ax,ay in [(.4,0),(0,.4),(.4,.4),(-.4,.3)]:
    plane=.8/(1+ax*(x-100)/220+ay*(y-80)/220)
    r=run(plane)
    assert r["state"]=="PHOTO",r
# Random depth noise must not pass the strong-curve route.
noisy=.8+rng.normal(0,.012,(h,w))
assert run(noisy)["state"]=="UNKNOWN",run(noisy)
lm[:]=saved
print("PASS: strong coherent curve tolerates unstable nose while planes/noise remain rejected")
