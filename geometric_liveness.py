"""Training-free metric face geometry; input must be depth aligned to RGB."""
import json
from pathlib import Path
import cv2
import numpy as np

DEFAULTS = dict(min_valid_ratio=.60, min_points=350, min_face_width=70,
                near_m=.40, far_m=1.50, plane_tolerance_mm=3.,
                flat_spread_mm=6., flat_support=.85,
                live_spread_mm=10., live_spatial_mm=3.,
                nose_min_mm=5., nose_max_mm=55., max_noise_mm=4.)
OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,
        379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,
        127,162,21,54,103,67,109]

def load_config(path=None):
    config = DEFAULTS.copy()
    if path:
        config.update(json.loads(Path(path).read_text(encoding="utf-8")))
    if set(config) != set(DEFAULTS):
        raise ValueError("几何配置存在未知字段")
    if any(not np.isfinite(v) or v <= 0 for v in config.values()):
        raise ValueError("几何阈值必须是有限正数")
    return config

def fit_plane(points):
    """Trimmed TLS in metric XYZ, returning perpendicular residual in mm."""
    sample = points[::max(1, len(points)//2500)]
    keep = np.ones(len(sample), dtype=bool)
    for _ in range(5):
        center = np.mean(sample[keep], axis=0)
        _, _, vh = np.linalg.svd(sample[keep]-center, full_matrices=False)
        normal = vh[-1]
        if normal[2] < 0: normal = -normal
        r = (sample-center) @ normal
        keep = np.abs(r-np.median(r)) <= max(.003, np.quantile(np.abs(r-np.median(r)), .85))
    return normal, center, (points-center) @ normal * 1000

def analyze_geometry(depth, scale, intr, mask, landmarks, config=None):
    c = DEFAULTS if config is None else config
    result = dict(state="UNKNOWN", reason="人脸有效深度不足", success=False)
    yy, xx = np.nonzero(mask)
    if len(xx) < c["min_points"]: return result
    if np.ptp(xx) < c["min_face_width"]:
        result["reason"]="人脸像素过少，请靠近"; return result
    z = depth[yy,xx].astype(float)*scale
    valid = (z >= c["near_m"]) & (z <= c["far_m"])
    result["valid_ratio"]=float(valid.mean())
    if valid.mean() < c["min_valid_ratio"] or valid.sum() < c["min_points"]: return result
    x,y,z = xx[valid],yy[valid],z[valid]
    # Reject disconnected/background depths without filling missing pixels.
    center_depth=np.median(z)
    good=np.abs(z-center_depth)<.30
    if good.mean()<.90:
        result["reason"]="人脸深度混入背景或异常点"; return result
    x,y,z=x[good],y[good],z[good]
    points=np.column_stack(((x-intr["ppx"])*z/intr["fx"],
                            (y-intr["ppy"])*z/intr["fy"],z))
    normal,center,residual=fit_plane(points)
    spread=float(np.percentile(residual,95)-np.percentile(residual,5))
    support=float(np.mean(np.abs(residual-np.median(residual)) <= c["plane_tolerance_mm"]))
    # Cell medians retain coherent surface shape; within-cell MAD measures noise.
    gx=np.minimum(7,((x-x.min())/max(np.ptp(x)+1,1)*8).astype(int))
    gy=np.minimum(7,((y-y.min())/max(np.ptp(y)+1,1)*8).astype(int))
    medians=[]; noises=[]
    for cell in range(64):
        v=residual[gy*8+gx==cell]
        if len(v)>=8:
            m=np.median(v); medians.append(m); noises.append(np.median(np.abs(v-m))*1.4826)
    spatial=float(np.std(medians)) if medians else 0.
    noise=float(np.median(noises)) if noises else float("inf")
    # Row/column residual variance is diagnostic only, after removing tilt.
    axis_std=[]
    for axis in (x,y):
        values=[np.std(residual[axis==a]) for a in np.unique(axis) if np.sum(axis==a)>=15]
        axis_std.append(float(np.median(values)) if values else 0.)
    def landmark_residual(ids):
        values=[]
        for idx in ids:
            px,py=landmarks[idx,:2]
            d=(x-px)**2+(y-py)**2
            v=residual[d <= max(3., np.ptp(x)*.035)**2]
            if len(v)>=5: values.append(np.median(v))
        return float(np.median(values)) if len(values)>=2 else None
    nose=landmark_residual([1,4,5])
    left=landmark_residual([50,101,205])
    right=landmark_residual([280,330,425])
    protrusion=None if any(v is None for v in (nose,left,right)) else (left+right)/2-nose
    result.update(success=True, spread_mm=spread, support=support,
                  spatial_mm=spatial, noise_mm=noise, nose_mm=protrusion,
                  row_std_mm=axis_std[0], column_std_mm=axis_std[1])
    if noise > c["max_noise_mm"]:
        result["reason"]="深度噪声过大"; return result
    if spread <= c["flat_spread_mm"] and support >= c["flat_support"]:
        result.update(state="PHOTO",reason="去除倾斜后，人脸区域接近平面")
    elif (spread >= c["live_spread_mm"] and spatial >= c["live_spatial_mm"]
          and protrusion is not None and c["nose_min_mm"] <= protrusion <= c["nose_max_mm"]):
        result.update(state="LIVE",reason="检测到连续曲面与鼻部凸出")
    else:
        result["reason"]="曲面证据不足或鼻部深度不可用"
    return result

def classify_face(depth, box, scale, intr, landmarks, config=None):
    if landmarks is None or len(landmarks)<=454:
        return dict(state="UNKNOWN",reason="人脸关键点不可用",success=False)
    mask=np.zeros(depth.shape,np.uint8)
    hull=cv2.convexHull(np.round(landmarks[OVAL,:2]).astype(np.int32))
    cv2.fillConvexPoly(mask,hull,255)
    margin=max(3,int((box[2]-box[0])*.04))
    mask=cv2.erode(mask,np.ones((margin*2+1,margin*2+1),np.uint8))
    return analyze_geometry(depth,scale,intr,mask,landmarks,config)
