"""Live D455 demo: person -> face -> 3-D liveness -> head-down/blink."""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image, ImageDraw, ImageFont

from rgbd_project_core import (
    FaceLocator, OneClassModel, PersonLocator, depth_preview,
    extract_depth_features, measure_person_3d,
)


LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)


def eye_aspect_ratio(points: np.ndarray, ids) -> float:
    p = points[list(ids), :2]
    vertical = np.linalg.norm(p[1]-p[5]) + np.linalg.norm(p[2]-p[4])
    return float(vertical / max(2*np.linalg.norm(p[0]-p[3]), 1e-6))


def head_pitch(points: np.ndarray, intr: dict) -> float | None:
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
    return float(angles[0])


class ActionState:
    def __init__(self, direction: str):
        self.direction = 1 if direction == "positive" else -1
        self.ear_baseline = deque(maxlen=40)
        self.pitch_baseline = deque(maxlen=40)
        self.closed_frames = 0
        self.blink_flash = 0
        self.blink_count = 0
        self.head_down_count = 0
        self.is_head_down = False

    @property
    def calibration_progress(self):
        return min(1.0, len(self.pitch_baseline)/self.pitch_baseline.maxlen)

    def reset_counts(self):
        self.blink_count = 0
        self.head_down_count = 0
        self.blink_flash = 0

    def update(self, landmarks, intr, head_threshold):
        if landmarks is None:
            return "UNKNOWN", "UNKNOWN", None, None
        ear = (eye_aspect_ratio(landmarks, LEFT_EYE)+eye_aspect_ratio(landmarks, RIGHT_EYE))/2
        pitch = head_pitch(landmarks, intr)
        if len(self.ear_baseline) < self.ear_baseline.maxlen:
            self.ear_baseline.append(ear)
        if pitch is not None and len(self.pitch_baseline) < self.pitch_baseline.maxlen:
            self.pitch_baseline.append(pitch)
        if len(self.ear_baseline) < self.ear_baseline.maxlen or len(self.pitch_baseline) < self.pitch_baseline.maxlen:
            self.closed_frames = 0
            return "CALIBRATING", "CALIBRATING", pitch, ear
        ear_ref = float(np.median(self.ear_baseline)) if self.ear_baseline else ear
        closed = ear < .68*ear_ref
        if closed:
            self.closed_frames += 1
        else:
            if 1 <= self.closed_frames <= 8:
                self.blink_flash = 6
                self.blink_count += 1
            self.closed_frames = 0
        eye_state = "BLINK" if self.blink_flash > 0 else ("CLOSED" if closed else "OPEN")
        self.blink_flash = max(0, self.blink_flash-1)
        head_state = "CALIBRATING"
        if pitch is not None:
            baseline = float(np.median(self.pitch_baseline))
            delta = self.direction*(pitch-baseline)
            if len(self.pitch_baseline) == self.pitch_baseline.maxlen:
                if not self.is_head_down and delta > head_threshold:
                    self.is_head_down = True
                    self.head_down_count += 1
                elif self.is_head_down and delta < head_threshold*.65:
                    self.is_head_down = False
                head_state = "HEAD_DOWN" if self.is_head_down else "NORMAL"
        return head_state, eye_state, pitch, ear


def intrinsics_dict(value):
    return {"fx": value.fx, "fy": value.fy, "ppx": value.ppx, "ppy": value.ppy,
            "width": value.width, "height": value.height}


FONT_REGULAR = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT_BOLD = Path(r"C:\Windows\Fonts\msyhbd.ttc")


def draw_texts(image, items):
    """Draw Chinese text once per surface to keep the live UI responsive."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)
    for text, xy, color, size, bold in items:
        font_path = FONT_BOLD if bold and FONT_BOLD.is_file() else FONT_REGULAR
        font = ImageFont.truetype(str(font_path), size) if font_path.is_file() else ImageFont.load_default()
        draw.text(xy, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def build_dashboard(color, depth_color, values):
    """Compose one presentation window from RGB, aligned depth and status values."""
    h, w = color.shape[:2]
    depth_color = cv2.resize(depth_color, (w, h))
    color = draw_texts(color, [("RGB彩色图与检测框", (12, h-34), (255,255,255), 20, False)])
    depth_color = draw_texts(depth_color, [("对齐到RGB的深度图", (12, h-34), (255,255,255), 20, False)])
    video = np.hstack((color, depth_color))
    panel_h = 250
    panel = np.full((panel_h, video.shape[1], 3), 24, dtype=np.uint8)
    live_color = (30, 220, 30) if values["live"] == "LIVE" else ((0, 70, 255) if values["live"] == "NON_LIVE" else (0, 210, 255))
    mid = video.shape[1]//2
    panel = draw_texts(panel, [
        ("RGB-D 人体检测与动作识别", (20, 10), (255,210,80), 28, True),
        (f"画面人数：{values['people']}    人体三维：{values['person3d_cn']}", (20, 55), (220,220,220), 22, False),
        (f"真人判别：{values['live_cn']}", (20, 92), live_color, 31, True),
        (f"目标距离：{values['distance']}", (20, 140), (220,220,220), 22, False),
        (f"当前帧模型距离：{values['raw_score']}    阈值：{values['threshold']}", (20, 177), (210,210,210), 19, False),
        (f"人脸深度有效率：{values['depth_valid']}", (20, 211), (210,210,210), 19, False),
        (f"眨眼次数：{values['blinks']}", (mid+20, 48), (80,255,170), 29, True),
        (f"低头次数：{values['head_count']}", (mid+20, 91), (80,255,170), 29, True),
        (f"当前动作：眼睛={values['eyes_cn']}  头部={values['head_cn']}", (mid+20, 137), (230,230,230), 20, False),
        (f"诊断原因：{values['reason']}", (mid+20, 174), (80,190,255), 19, False),
        (values["hint"], (mid+20, 213), (170,170,170), 16, False),
    ])
    return np.vstack((video, panel))


def main():
    p = argparse.ArgumentParser(description="D455 RGB-D初版实时系统")
    p.add_argument("--model", type=Path, default=Path("models/liveness_oneclass.json"))
    p.add_argument("--person-model", default="yolo11n.pt")
    p.add_argument("--width", type=int, default=640); p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--head-down-threshold", type=float, default=15.0)
    p.add_argument("--head-down-direction", choices=["positive", "negative"], default="positive")
    p.add_argument("--allow-fallback", action="store_true", help="缺少YOLO/MediaPipe时仍以降级模式运行")
    p.add_argument("--snapshots", type=Path, default=Path("demo_results"))
    args = p.parse_args()
    model = OneClassModel.load(args.model)
    person = PersonLocator(args.person_model)
    face = FaceLocator(allow_fallback=args.allow_fallback)
    if not args.allow_fallback and (person.backend == "full-frame-fallback" or not face.backend.startswith("mediapipe")):
        missing = []
        if person.backend == "full-frame-fallback": missing.append("Ultralytics/YOLO权重")
        if not face.backend.startswith("mediapipe"): missing.append("MediaPipe")
        raise SystemExit("完整演示所需组件不可用：" + "、".join(missing) +
                         "\n请执行 python -m pip install -r project_requirements.txt；仅调试降级模式可加 --allow-fallback")
    action = ActionState(args.head_down_direction)
    decisions = deque(maxlen=7)
    pipeline = rs.pipeline(); config = rs.config()
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = pipeline.start(config); align = rs.align(rs.stream.color)
    scale = profile.get_device().first_depth_sensor().get_depth_scale()
    intr = intrinsics_dict(profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics())
    print(f"Person backend: {person.backend}; face backend: {face.backend}")
    cv2.namedWindow("RGB-D Course Project Demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RGB-D Course Project Demo", args.width*2, args.height+250)
    print("保持正视并睁眼约2秒完成校准；Q/Esc退出，C重新校准，R计数清零，S保存截图")
    try:
        while True:
            frames = align.process(pipeline.wait_for_frames(5000))
            cf, df = frames.get_color_frame(), frames.get_depth_frame()
            if not cf or not df: continue
            color = np.asanyarray(cf.get_data()).copy(); depth = np.asanyarray(df.get_data()).copy()
            person_box, confidence, _ = person.locate(color, depth)
            person_3d = {"state": "UNKNOWN"}
            live_text = "UNKNOWN"; ratio = None; raw_distance = None
            face_valid_ratio = None; debug_reason = "未检测到人体"
            face_box = None; landmarks = None; face_source = "not-found"
            if person_box is not None:
                person_3d = measure_person_3d(depth, person_box, scale, intr)
                debug_reason = "未检测到人脸"
                face_box, landmarks, face_source = face.locate(color, person_box)
                if face_box is not None:
                    features, quality = extract_depth_features(depth, face_box, scale, intr)
                    face_valid_ratio = quality.get("valid_ratio")
                    reason_map = {
                        "empty_roi": "人脸裁剪区域为空",
                        "too_few_depth_points": "人脸有效深度点不足",
                        "too_few_foreground_points": "人脸前景深度点不足",
                        "invalid_features": "人脸三维特征计算失败",
                    }
                    debug_reason = reason_map.get(quality.get("reason"), "人脸深度证据不足")
                    if features is not None and quality["valid_ratio"] >= .20:
                        raw_distance = float(model.distances(features)[0]); ratio = raw_distance/model.threshold
                        frame_live = raw_distance <= model.threshold
                        decisions.append(frame_live)
                        live_text = "LIVE" if sum(decisions) >= max(1, len(decisions)*.6) else "NON_LIVE"
                        debug_reason = "真人模型接受当前人脸" if frame_live else "真人模型距离超过阈值"
                        if person.backend != "full-frame-fallback" and person_3d["state"] == "FLAT":
                            live_text = "NON_LIVE"
                            debug_reason = "人体区域点云接近平面"
                    else:
                        decisions.clear()
                else:
                    decisions.clear()
            else:
                decisions.clear()
            if live_text == "LIVE":
                head, eyes, pitch, ear = action.update(landmarks, intr, args.head_down_threshold)
            else:
                head, eyes, pitch, ear = "LOCKED", "LOCKED", None, None
            if person_box is not None:
                x1,y1,x2,y2=person_box; cv2.rectangle(color,(x1,y1),(x2,y2),(255,180,0),2)
            if face_box is not None:
                x1,y1,x2,y2=face_box; cv2.rectangle(color,(x1,y1),(x2,y2),(0,255,0) if live_text=="LIVE" else (0,0,255),2)
            distance_text = f"{person_3d['distance_m']:.2f} m" if "distance_m" in person_3d else "--"
            if action.calibration_progress < 1 and landmarks is not None:
                hint = f"正在校准：请正视并睁眼 {action.calibration_progress:.0%}"
            else:
                hint = "按键：C重新校准｜R计数清零｜S截图｜Q退出"
            live_cn = {"LIVE":"真人", "NON_LIVE":"非真人", "UNKNOWN":"无法判断"}[live_text]
            person3d_cn = {"3D_OK":"三维结构正常", "FLAT":"接近平面", "UNKNOWN":"无法判断"}.get(person_3d["state"], person_3d["state"])
            eyes_cn = {"OPEN":"睁眼", "CLOSED":"闭眼", "BLINK":"眨眼", "CALIBRATING":"校准中", "LOCKED":"未启用", "UNKNOWN":"无法判断"}.get(eyes, eyes)
            head_cn = {"NORMAL":"正常", "HEAD_DOWN":"低头", "CALIBRATING":"校准中", "LOCKED":"未启用", "UNKNOWN":"无法判断"}.get(head, head)
            dashboard = build_dashboard(color, depth_preview(depth, scale, .3, 2.5), {
                "people": person.last_count, "person3d_cn": person3d_cn, "live": live_text, "live_cn": live_cn,
                "distance": distance_text, "blinks": action.blink_count,
                "head_count": action.head_down_count, "eyes_cn": eyes_cn, "head_cn": head_cn,
                "raw_score": "--" if raw_distance is None else f"{raw_distance:.2f}",
                "threshold": f"{model.threshold:.2f}",
                "depth_valid": "--" if face_valid_ratio is None else f"{face_valid_ratio:.1%}",
                "reason": debug_reason, "hint": hint,
            })
            cv2.imshow("RGB-D Course Project Demo", dashboard)
            key=cv2.waitKey(1)&0xff
            if key in (ord('q'),ord('Q'),27): break
            if key in (ord('c'),ord('C')):
                action=ActionState(args.head_down_direction); decisions.clear()
            if key in (ord('r'),ord('R')):
                action.reset_counts()
            if key in (ord('s'),ord('S')):
                args.snapshots.mkdir(parents=True, exist_ok=True)
                target = args.snapshots/(datetime.now().strftime("demo_%Y%m%d_%H%M%S")+".png")
                cv2.imwrite(str(target), dashboard); print(f"截图已保存：{target.resolve()}")
    finally:
        pipeline.stop(); face.close(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
