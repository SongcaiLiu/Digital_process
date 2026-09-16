"""Training-free metric face geometry; input must be depth aligned to RGB."""
import json
from pathlib import Path
import cv2
import numpy as np

DEFAULTS = dict(min_valid_ratio=.60, min_points=350, min_face_width=70,
                near_m=.40, far_m=1.50, plane_tolerance_mm=3.,
                flat_spread_mm=6., flat_support=.85,
                live_spread_mm=10., live_spatial_mm=3.,
                nose_min_mm=5., nose_max_mm=55., max_noise_mm=4.,
                min_cells=24, curve_gain_min=.35, curve_spread_max_mm=70.)
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
    if len(xx) < c["min_points"]:
        result["reason"]="人脸内部区域过小"; return result
    if np.ptp(xx) < c["min_face_width"]:
        result["reason"]="人脸像素过少，请靠近"; return result
    z = depth[yy,xx].astype(float)*scale
    valid = (z >= c["near_m"]) & (z <= c["far_m"])
    result["valid_ratio"]=float(valid.mean())
    if valid.mean() < c["min_valid_ratio"]:
        missing=float(np.mean(z==0))
        outside=float(np.mean((z>0)&~valid))
        result["reason"]=("有效深度%.0f%%<%.0f%%：空洞%.0f%%，超距%.0f%%" %
                          (valid.mean()*100,c["min_valid_ratio"]*100,missing*100,outside*100))
        return result
    if valid.sum() < c["min_points"]:
        result["reason"]="有效深度点数不足"; return result
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
    medians=[]; noises=[]; cell_xy=[]
    for cell in range(64):
        selected=gy*8+gx==cell
        v=residual[selected]
        if len(v)>=12:
            medians.append(float(np.median(v)))
            cell_xy.append((cell%8,cell//8))
            # Remove local surface slope before estimating random depth noise.
            design=np.column_stack((x[selected]-np.mean(x[selected]),
                                    y[selected]-np.mean(y[selected]),np.ones(len(v))))
            keep=np.ones(len(v),bool)
            for _ in range(3):
                beta=np.linalg.lstsq(design[keep],v[keep],rcond=None)[0]
                error=v-design@beta
                cutoff=max(1.,np.quantile(np.abs(error-np.median(error)),.90))
                keep=np.abs(error-np.median(error))<=cutoff
            noises.append(float(np.median(np.abs(error-np.median(error)))*1.4826))
    spatial=float(np.std(medians)) if medians else 0.
    noise=float(np.median(noises)) if noises else float("inf")
    # A quadratic must explain spatial cell medians much better than a plane.
    curve_gain=0.
    if len(medians)>=c["min_cells"]:
        uv=np.asarray(cell_xy,dtype=float)/7
        u,v=uv.T; target=np.asarray(medians)
        linear=np.column_stack((u,v,np.ones(len(u))))
        quadratic=np.column_stack((linear,u*u,u*v,v*v))
        e1=target-linear@np.linalg.lstsq(linear,target,rcond=None)[0]
        e2=target-quadratic@np.linalg.lstsq(quadratic,target,rcond=None)[0]
        curve_gain=float(1-np.mean(e2*e2)/max(np.mean(e1*e1),1e-6))
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
    cheeks=[v for v in (left,right) if v is not None]
    protrusion=None if nose is None or not cheeks else float(np.mean(cheeks)-nose)
    result.update(success=True, spread_mm=spread, support=support,
                  spatial_mm=spatial, noise_mm=noise, nose_mm=protrusion,
                  row_std_mm=axis_std[0], column_std_mm=axis_std[1],
                  curve_gain=curve_gain, cells=len(medians), cheeks_available=len(cheeks),
                  nose_available=nose is not None)
    result["quality_ok"]=False
    if noise > c["max_noise_mm"]:
        result["reason"]="局部深度噪声%.1f毫米超限" % noise; return result
    if len(medians)<c["min_cells"]:
        result["reason"]="有效空间网格不足：%d/%d" % (len(medians),c["min_cells"]); return result
    result["quality_ok"]=True
    if spread <= c["flat_spread_mm"] and support >= c["flat_support"]:
        result.update(state="PHOTO",reason="去除倾斜后，人脸区域接近平面")
    elif spread < c["live_spread_mm"]:
        result["reason"]="非平面起伏不足，尚未满足照片平面条件"
    elif spatial < c["live_spatial_mm"]:
        result["reason"]="连续空间起伏不足"
    elif protrusion is not None and c["nose_min_mm"] <= protrusion <= c["nose_max_mm"]:
        result.update(state="LIVE",reason="连续曲面及鼻部凸出（%s）" %
                      ("双颊可用" if len(cheeks)==2 else "单侧脸颊可用"))
    elif (protrusion is None and curve_gain>=c["curve_gain_min"]
          and spread<=c["curve_spread_max_mm"]):
        result.update(state="LIVE",reason="局部关键深度缺失，连续曲面拟合提供辅助证据")
    elif nose is None:
        result["reason"]="鼻部深度缺失，曲面辅助证据不足"
    elif not cheeks:
        result["reason"]="双颊深度缺失，曲面辅助证据不足"
    else:
        result["reason"]="鼻颊凸出%.1f毫米不在%.0f～%.0f范围" % (
            protrusion,c["nose_min_mm"],c["nose_max_mm"])
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
