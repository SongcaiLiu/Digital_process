"""Live D455 demo: person -> face -> 3-D liveness -> head-down/blink."""

from __future__ import annotations

# Prevent user-level packages under AppData/Roaming from overriding this conda environment.
import site
import sys
from pathlib import Path
_user_sites=site.getusersitepackages()
if isinstance(_user_sites,str):_user_sites=[_user_sites]
_user_sites={str(Path(value).resolve()).lower() for value in _user_sites}
sys.path[:]=[value for value in sys.path
             if str(Path(value or ".").resolve()).lower() not in _user_sites]
site.ENABLE_USER_SITE=False

import argparse
from collections import deque
from datetime import datetime
from time import monotonic

import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image, ImageDraw, ImageFont

from rgbd_project_core import (
    FaceLocator, PersonLocator, depth_preview, measure_person_3d,
)


from geometric_liveness import classify_face, load_config

LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)


def eye_aspect_ratio(points: np.ndarray, ids) -> float:
    p = points[list(ids), :2]
    vertical = np.linalg.norm(p[1]-p[5]) + np.linalg.norm(p[2]-p[4])
    return float(vertical / max(2*np.linalg.norm(p[0]-p[3]), 1e-6))


def head_pose_angles(points: np.ndarray, intr: dict):
    if len(points) <= 291:
        return None
    image_points = points[[1, 152, 33, 263, 61, 291], :2].astype(np.float64)
    model_points = np.array([
        (0, 0, 0), (0, -330, -65), (-225, 170, -135),
        (225, 170, -135), (-150, -150, -125), (150, -150, -125),
    ], dtype=np.float64)
    camera = np.array([[intr["fx"], 0, intr["ppx"]], [0, intr["fy"], intr["ppy"]], [0, 0, 1]], dtype=np.float64)
    ok, rotation, _ = cv2.solvePnP(model_points, image_points, camera, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    matrix, _ = cv2.Rodrigues(rotation)
    angles = cv2.RQDecomp3x3(matrix)[0]
    return float(angles[0]), float(angles[1]), float(angles[2])



from action_calibration import AutomaticActions, relative_angle
from diagnostic_recording import DiagnosticRecorder, camera_angle

class ActionState(AutomaticActions):
    def __init__(self, direction="positive", pose_model=None, pose_interval=2):
        super().__init__(direction,side_factor_scale=.35 if pose_model is not None else 1.0)
        self.face_motion_anchor=None
        self.face_motion_previous=None
        self.pose_model=pose_model
        self.pose_interval=pose_interval
        self.pose_tick=0
        self.cached_pose=None

    def suspend(self):
        """Discard incomplete gestures when the target is not confirmed live."""
        self.face_motion_anchor=None
        self.face_motion_previous=None
        self.last_angles=None
        self.cached_pose=None
        return self.missing()

    def update_if_live(self, landmarks, intr, down_threshold, up_threshold, liveness,
                       color=None, face_box=None):
        if landmarks is None:
            return self.suspend()
        if liveness == "LIVE":
            return self.update(landmarks, intr, down_threshold, up_threshold,
                               color=color, face_box=face_box)
        # A brief red-frame dropout during a gesture may be caused by the
        # changing pose. Count within the grace window, but do not refresh
        # last_confirmed_live until a genuinely LIVE frame arrives.
        if (self.last_confirmed_live is not None and
                0<=monotonic()-self.last_confirmed_live<=.6):
            return self.update(
                landmarks, intr, down_threshold, up_threshold,
                count_enabled=True, confirmed_live=False,
                color=color, face_box=face_box)
        return self.suspend()

    def update(self, landmarks, intr, down_threshold, up_threshold, count_enabled=True,
               confirmed_live=True, color=None, face_box=None):
        if landmarks is None:
            self.face_motion_previous=None
            return self.missing()
        if self.pose_model is None:
            angles=head_pose_angles(landmarks,intr)
        else:
            self.pose_tick+=1
            if self.cached_pose is None or self.pose_tick % self.pose_interval==0:
                crop=pose_face_crop(color,face_box)
                if crop is None:
                    self.cached_pose=None
                else:
                    pitch,yaw,roll=self.pose_model.predict(crop)
                    self.cached_pose=(float(np.asarray(pitch).item()),
                                      float(np.asarray(yaw).item()),
                                      float(np.asarray(roll).item()))
            angles=self.cached_pose
        self.last_angles=angles
        if angles is None:
            self.face_motion_previous=None
            return self.missing()
        ear=(eye_aspect_ratio(landmarks,LEFT_EYE)+eye_aspect_ratio(landmarks,RIGHT_EYE))/2
        xy=landmarks[:,:2]
        center=np.median(xy,axis=0)
        width=max(float(np.ptp(xy[:,0])),1.)
        height=max(float(np.ptp(xy[:,1])),1.)
        size=max((width*height)**.5,1.)
        translation=False
        if self.face_motion_anchor is None:
            self.face_motion_anchor=(center.copy(),size)
        else:
            anchor_center,anchor_size=self.face_motion_anchor
            anchor_shift=abs(float(center[0]-anchor_center[0]))/max(anchor_size,1.)
            anchor_scale=abs(float(np.log(size/max(anchor_size,1.))))
            translation=anchor_shift>.28 or anchor_scale>.14
        if self.face_motion_previous is not None:
            previous_center,previous_size=self.face_motion_previous
            frame_shift=abs(float(center[0]-previous_center[0]))/max(previous_size,1.)
            frame_scale=abs(float(np.log(size/max(previous_size,1.))))
            translation=translation or frame_shift>.18 or frame_scale>.09
        self.face_motion_previous=(center.copy(),size)
        if translation:
            self.face_motion_anchor=(center.copy(),size)
        elif not self.reanchor_required:
            anchor_center,anchor_size=self.face_motion_anchor
            self.face_motion_anchor=(.995*anchor_center+.005*center,
                                     .995*anchor_size+.005*size)
        return self.update_measurements(*angles,ear,down_threshold,up_threshold,
                                        translation=translation,
                                        count_enabled=count_enabled,
                                        confirmed_live=confirmed_live)


def pose_face_crop(color, face_box):
    """Return a padded square face crop for the pretrained pose model."""
    if color is None or face_box is None:
        return None
    x1,y1,x2,y2=face_box
    h,w=color.shape[:2]
    side=max(x2-x1,y2-y1)*1.3
    cx,cy=(x1+x2)/2,(y1+y2)/2
    left=max(0,int(cx-side/2)); top=max(0,int(cy-side/2))
    right=min(w,int(cx+side/2)); bottom=min(h,int(cy+side/2))
    if right-left<32 or bottom-top<32:
        return None
    return color[top:bottom,left:right].copy()


def intrinsics_dict(value):
    return {"fx": value.fx, "fy": value.fy, "ppx": value.ppx, "ppy": value.ppy,
            "width": value.width, "height": value.height}


FONT_REGULAR = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT_BOLD = Path(r"C:\Windows\Fonts\msyhbd.ttc")
_FONT_CACHE = {}


def draw_texts(image, items):
    """Draw Chinese text once per surface to keep the live UI responsive."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)
    for text, xy, color, size, bold in items:
        font_path = FONT_BOLD if bold and FONT_BOLD.is_file() else FONT_REGULAR
        key=(str(font_path),size)
        if key not in _FONT_CACHE:
            _FONT_CACHE[key]=ImageFont.truetype(str(font_path),size) if font_path.is_file() else ImageFont.load_default()
        draw.text(xy, text, font=_FONT_CACHE[key], fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def build_dashboard(color, depth_color, values):
    """Compose one presentation window from RGB, aligned depth and status values."""
    h, w = color.shape[:2]
    depth_color = cv2.resize(depth_color, (w, h))
    color = draw_texts(color, [("RGB彩色图与检测框", (12, h-34), (255,255,255), 20, False)])
    depth_color = draw_texts(depth_color, [("对齐到RGB的深度图", (12, h-34), (255,255,255), 20, False)])
    video = np.hstack((color, depth_color))
    panel_h = 275
    panel = np.full((panel_h, video.shape[1], 3), 24, dtype=np.uint8)
    live_color = (30, 220, 30) if values["live"] == "LIVE" else ((0, 70, 255) if values["live"] == "PHOTO" else (0, 210, 255))
    mid = video.shape[1]//2
    panel = draw_texts(panel, [
        (f"RGB-D 人体检测与动作识别    FPS：{values['processing_fps']}    头姿：{values['pose_backend']}", (20, 8), (255,210,80), 28, True),
        (f"YOLO人体检测：{values['yolo_status']}", (20, 48), (220,220,220), 19, False),
        (f"MediaPipe人脸检测：{values['mp_status']}", (20, 80), (220,220,220), 19, False),
        (f"RGB-D深度特征：{values['feature_status']}", (20, 112), (220,220,220), 19, False),
        (f"分类结果：{values['live_cn']}", (20, 148), live_color, 31, True),
        (f"人体三维：{values['person3d_cn']}    人脸距离：{values['face_distance']}", (20, 196), (220,220,220), 19, False),
        (values["geometry_summary"], (20, 229), (210,210,210), 18, False),
        (f"眨眼次数：{values['blinks']}", (mid+20, 45), (80,255,170), 27, True),
        (f"低头次数：{values['head_down_count']}    抬头次数：{values['head_up_count']}", (mid+20, 84), (80,255,170), 25, True),
        (f"当前动作：眼睛={values['eyes_cn']}  头部={values['head_cn']}", (mid+20, 126), (230,230,230), 19, False),
        (f"诊断原因：{values['reason']}", (mid+20, 166), (80,190,255), 18, False),
        (values["hint"], (mid+20, 237), (170,170,170), 16, False),
    ])
    return np.vstack((video, panel))


def main():
    p = argparse.ArgumentParser(description="D455 RGB-D初版实时系统")
    p.add_argument("--geometry-config", type=Path, default=Path("geometric_config.json"))
    p.add_argument("--person-model", default="yolo11n.pt")
    p.add_argument("--width", type=int, default=640); p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--yolo-interval", type=int, default=3, help="YOLO每几帧更新一次")
    p.add_argument("--geometry-interval", type=int, default=3, help="RGB-D几何每几帧更新一次")
    p.add_argument("--head-down-threshold", type=float, default=None)
    p.add_argument("--head-up-threshold", type=float, default=None)
    p.add_argument("--head-down-direction", choices=["positive", "negative"], default=None)
    p.add_argument("--head-pose-backend", choices=["pnp", "6drepnet"], default="pnp",
                   help="头姿算法；默认原有PnP，6drepnet为预训练模型实验模式")
    p.add_argument("--head-pose-interval", type=int, default=2,
                   help="6DRepNet每几帧推理一次；1最灵敏但更慢")
    p.add_argument("--allow-fallback", action="store_true", help="缺少YOLO/MediaPipe时仍以降级模式运行")
    p.add_argument("--snapshots", type=Path, default=Path("demo_results"))
    p.add_argument("--record-test", action="store_true", help="后台记录姿态测试数据")
    p.add_argument("--test-output", type=Path, default=Path("demo_results/pose_tests"))
    p.add_argument("--test-sample-fps", type=float, default=2., help="RGB-D样本保存频率")
    args = p.parse_args()
    if args.test_sample_fps<=0: p.error("--test-sample-fps 必须大于0")
    if args.yolo_interval<1 or args.geometry_interval<1 or args.head_pose_interval<1:
        p.error("检测间隔必须大于0")
    geometry_config = load_config(args.geometry_config)
    person = PersonLocator(args.person_model)
    face = FaceLocator(allow_fallback=args.allow_fallback)
    if not args.allow_fallback and (person.backend == "full-frame-fallback" or not face.backend.startswith("mediapipe")):
        missing = []
        if person.backend == "full-frame-fallback": missing.append("Ultralytics/YOLO权重")
        if not face.backend.startswith("mediapipe"): missing.append("MediaPipe")
        details=[]
        if person.backend == "full-frame-fallback" and getattr(person,"error",None):
            details.append("YOLO错误："+person.error)
        raise SystemExit("完整演示所需组件不可用：" + "、".join(missing) +
                         ("\n" + "\n".join(details) if details else "") +
                         "\n请先运行 python check_project.py；仅调试降级模式可加 --allow-fallback")

    pose_model=None
    if args.head_pose_backend=="6drepnet":
        try:
            from sixdrepnet import SixDRepNet
            pose_model=SixDRepNet()
        except Exception as exc:
            raise SystemExit("6DRepNet加载失败；请检查sixdrepnet、scipy及模型权重："+str(exc)) from exc
    direction=args.head_down_direction or ("negative" if pose_model is not None else "positive")
    down_threshold=(10.0 if pose_model is not None else 3.5) if args.head_down_threshold is None else args.head_down_threshold
    up_threshold=(10.0 if pose_model is not None else 3.5) if args.head_up_threshold is None else args.head_up_threshold
    action = ActionState(direction,pose_model,args.head_pose_interval)
    pipeline = rs.pipeline(); config = rs.config()
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = pipeline.start(config); align = rs.align(rs.stream.color)
    scale = profile.get_device().first_depth_sensor().get_depth_scale()
    intr = intrinsics_dict(profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics())
    print(f"Person backend: {person.backend}; face backend: {face.backend}")
    cv2.namedWindow("RGB-D Course Project Demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RGB-D Course Project Demo", args.width*2, args.height+275)
    print("姿态参考会在后台自动适应；Q/Esc退出，C重置参考，R计数清零，S保存截图")
    recorder=None
    started=monotonic()
    frame_index=0
    cached_person=(None,0.0,person.backend)
    cached_geometry={}
    cached_person3d={"state":"UNKNOWN"}
    fps_times=deque(maxlen=45)
    try:
        if args.record_test:
            recorder=DiagnosticRecorder(args.test_output,
                dict(depth_scale=scale,intrinsics=intr,geometry_config=geometry_config,
                     person_backend=person.backend,face_backend=face.backend),
                args.test_sample_fps)
            print("按1正视、2左右转头、3低头抬头、4组合姿态、5照片；Q结束并生成报告")
        while True:
            frames = align.process(pipeline.wait_for_frames(5000))
            cf, df = frames.get_color_frame(), frames.get_depth_frame()
            if not cf or not df: continue
            frame_index+=1
            color = np.asanyarray(cf.get_data()).copy(); depth = np.asanyarray(df.get_data()).copy()
            if frame_index==1 or frame_index % args.yolo_interval==0:
                cached_person=person.locate(color,depth)
            person_box,_,_=cached_person
            person_3d=cached_person3d
            live_text=cached_geometry.get("state","UNKNOWN"); geometry=cached_geometry
            debug_reason = "未检测到人体"
            feature_status = "未运行"
            face_box = None; landmarks = None
            if person_box is not None:
                debug_reason="未检测到人脸"
                face_box,landmarks,_=face.locate(color,person_box)
                if face_box is not None:
                    geometry_due=(not cached_geometry or frame_index % args.geometry_interval==0)
                    if geometry_due:
                        cached_geometry=classify_face(depth,face_box,scale,intr,landmarks,geometry_config)
                        cached_person3d=measure_person_3d(depth,person_box,scale,intr)
                    geometry=cached_geometry;person_3d=cached_person3d
                    feature_status=("有效" if geometry.get("quality_ok") else
                                    ("已计算，质量不足" if geometry.get("success") else "不足"))
                    live_text=geometry.get("state","UNKNOWN");debug_reason=geometry.get("reason","深度证据不足")
                else:
                    cached_geometry={};live_text="UNKNOWN";geometry={}
            else:
                cached_geometry={};cached_person3d={"state":"UNKNOWN"}
                live_text="UNKNOWN";geometry={}
            head, eyes, pitch, ear = action.update_if_live(
                landmarks, intr, down_threshold,
                up_threshold, live_text, color=color, face_box=face_box)
            if recorder is not None:
                angles=getattr(action,"last_angles",None) if landmarks is not None else None
                pose={}
                if angles is not None:
                    pose=dict(zip(("pitch_raw","yaw_raw","roll_raw"),angles))
                    pose.update(zip(("pitch_camera","yaw_camera","roll_camera"),
                                    (camera_angle(v) for v in angles)))
                    if action.baseline is not None:
                        pose["relative_pitch"]=action.direction*relative_angle(angles[0],action.baseline)
                recorder.record(color,depth,landmarks,person_box,face_box,geometry,
                    dict(pose,elapsed_s=monotonic()-started,camera_ms=cf.get_timestamp(),
                         classification=live_text,reason=debug_reason,
                         person_detected=person_box is not None,face_detected=landmarks is not None,
                         feature_status=feature_status,head_state=head,eye_state=eyes,
                         calibrated=action.calibration_progress>=1))
            if person_box is not None:
                x1,y1,x2,y2=person_box; cv2.rectangle(color,(x1,y1),(x2,y2),(255,180,0),2)
            if face_box is not None:
                x1,y1,x2,y2=face_box; cv2.rectangle(color,(x1,y1),(x2,y2),(0,255,0) if live_text=="LIVE" else (0,0,255),2)
            hint = ("姿态参考稳定中，请保持约0.6秒" if action.reset_settle_active else
                    "姿态参考自动适应｜C重置参考｜R计数清零｜S截图｜Q退出")
            if (live_text!="LIVE" and landmarks is not None and
                    action.last_confirmed_live is not None and
                    0<=monotonic()-action.last_confirmed_live<=.6):
                hint="真人短时丢失：动作计数宽限中（0.6秒）｜Q退出"
            if recorder is not None:
                hint="记录中："+recorder.segment+"｜1～5切换动作段｜Q结束"
                if recorder.error: hint="记录失败："+recorder.error
            live_cn = {"LIVE":"真人", "PHOTO":"照片", "UNKNOWN":"未知"}[live_text]
            yolo_status = "已检测" if person_box is not None else "未检测"
            mp_status = "已检测" if landmarks is not None else "未检测"
            person3d_cn = {"3D_OK":"三维结构正常", "FLAT":"接近平面", "UNKNOWN":"无法判断"}.get(person_3d["state"], person_3d["state"])
            eyes_cn = {"OPEN":"睁眼", "CLOSED":"闭眼", "BLINK":"眨眼", "CALIBRATING":"校准中", "LOCKED":"未启用", "UNKNOWN":"无法判断"}.get(eyes, eyes)
            head_cn = {"NORMAL":"正常", "HEAD_DOWN":"低头", "HEAD_UP":"抬头", "CALIBRATING":"校准中", "UNKNOWN":"未知"}.get(head, head)
            fps_times.append(monotonic())
            processing_fps=((len(fps_times)-1)/(fps_times[-1]-fps_times[0])
                            if len(fps_times)>1 and fps_times[-1]>fps_times[0] else 0.)
            dashboard = build_dashboard(color, depth_preview(depth, scale, .3, 2.5), {
                "processing_fps": f"{processing_fps:.1f}",
                "pose_backend": "6DRepNet" if pose_model is not None else "PnP",
                "yolo_status": yolo_status,
                "mp_status": mp_status, "feature_status": feature_status,
                "person3d_cn": person3d_cn, "live": live_text, "live_cn": live_cn,
                "face_distance": ("%.2f m" % geometry["face_distance_m"])
                    if geometry.get("face_distance_m") is not None else "--",
                "blinks": action.blink_count,
                "head_down_count": action.head_down_count, "head_up_count": action.head_up_count,
                "eyes_cn": eyes_cn, "head_cn": head_cn,
                "geometry_summary": ("去倾斜起伏：%.1f mm｜平面比例：%.0f%%" %
                    (geometry["spread_mm"], geometry["support"]*100)) if geometry.get("success") else "去倾斜起伏：--",

                "reason": debug_reason, "hint": hint,
            })
            cv2.imshow("RGB-D Course Project Demo", dashboard)
            key=cv2.waitKey(1)&0xff
            if recorder is not None and ord('1')<=key<=ord('8'):
                recorder.set_segment(chr(key))
            if key in (ord('q'),ord('Q'),27): break
            if key in (ord('c'),ord('C')):
                action=ActionState(direction,pose_model,args.head_pose_interval)
                action.begin_reference_reset()
            if key in (ord('r'),ord('R')):
                action.reset_counts()
            if key in (ord('s'),ord('S')):
                args.snapshots.mkdir(parents=True, exist_ok=True)
                target = args.snapshots/(datetime.now().strftime("demo_%Y%m%d_%H%M%S")+".png")
                cv2.imwrite(str(target), dashboard); print(f"截图已保存：{target.resolve()}")
    finally:
        pipeline.stop(); face.close(); cv2.destroyAllWindows()
        if recorder is not None: recorder.close()


if __name__ == "__main__":
    main()
