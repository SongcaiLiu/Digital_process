"""Shared pretrained detectors and RGB-D display utilities."""
from __future__ import annotations

import math
import os
from pathlib import Path

import cv2
import numpy as np


def clamp_box(box: tuple[int, int, int, int], shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    x1, y1, x2, y2 = box
    x1, x2 = sorted((max(0, min(w - 1, int(x1))), max(1, min(w, int(x2)))))
    y1, y2 = sorted((max(0, min(h - 1, int(y1))), max(1, min(h, int(y2)))))
    return x1, y1, x2, y2


def central_face_box(shape: tuple[int, ...], person_box: tuple[int, int, int, int] | None = None) -> tuple[int, int, int, int]:
    """Conservative fallback for the existing close, centered acquisition data."""
    h, w = shape[:2]
    if person_box is None:
        # Existing acquisition protocol places a close face near image center.
        # Keep shoulders/background mostly outside this emergency-only crop.
        return clamp_box((0.25 * w, 0.06 * h, 0.75 * w, 0.68 * h), shape)
    x1, y1, x2, y2 = person_box
    pw, ph = x2 - x1, y2 - y1
    return clamp_box((x1 + .22 * pw, y1 + .03 * ph, x1 + .78 * pw, y1 + .48 * ph), shape)


class FaceLocator:
    """MediaPipe face landmarks when installed, with an explicit center fallback."""

    def __init__(self, allow_fallback: bool = True, static_image_mode: bool = False,
                 model_path: Path | str = Path("models/face_landmarker.task")):
        self.allow_fallback = allow_fallback
        self.mesh = None
        self.mp = None
        self.tasks_api = False
        self.backend = "center-fallback"
        try:
            import mediapipe as mp  # type: ignore
            self.mp = mp
            solutions = getattr(mp, "solutions", None)
            if solutions is not None:
                self.mesh = solutions.face_mesh.FaceMesh(
                    static_image_mode=static_image_mode,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=.5,
                    min_tracking_confidence=.5,
                )
                self.backend = "mediapipe-solutions"
            elif hasattr(mp, "tasks") and Path(model_path).is_file():
                options = mp.tasks.vision.FaceLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=str(Path(model_path).resolve())),
                    running_mode=mp.tasks.vision.RunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=.5,
                    min_face_presence_confidence=.5,
                    min_tracking_confidence=.5,
                )
                self.mesh = mp.tasks.vision.FaceLandmarker.create_from_options(options)
                self.tasks_api = True
                self.backend = "mediapipe-tasks"
        except Exception:
            self.mesh = None

    def locate(self, bgr: np.ndarray, person_box: tuple[int, int, int, int] | None = None):
        h, w = bgr.shape[:2]
        landmarks = None
        if self.mesh is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if self.tasks_api:
                mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                result = self.mesh.detect(mp_image)
                lm = result.face_landmarks[0] if result.face_landmarks else None
            else:
                result = self.mesh.process(rgb)
                lm = result.multi_face_landmarks[0].landmark if result.multi_face_landmarks else None
            if lm:
                xs = np.array([p.x * w for p in lm])
                ys = np.array([p.y * h for p in lm])
                margin_x = .12 * (xs.max() - xs.min())
                margin_y = .10 * (ys.max() - ys.min())
                box = clamp_box((xs.min()-margin_x, ys.min()-margin_y,
                                 xs.max()+margin_x, ys.max()+margin_y), bgr.shape)
                landmarks = np.array([(p.x*w, p.y*h, p.z*w) for p in lm], dtype=np.float32)
                return box, landmarks, self.backend
        if self.allow_fallback:
            return central_face_box(bgr.shape, person_box), landmarks, "center-fallback"
        return None, None, "not-found"

    def close(self):
        if self.mesh is not None:
            self.mesh.close()


class PersonLocator:
    """COCO person detector. Full-frame fallback keeps the prototype runnable."""

    def __init__(self, model_path: str = "yolo11n.pt", confidence: float = .35):
        self.model = None
        self.last_count = 0
        self.confidence = confidence
        self.backend = "full-frame-fallback"
        try:
            config_dir = Path("models/ultralytics_config").resolve()
            config_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
            from ultralytics import YOLO  # type: ignore
            self.model = YOLO(model_path)
            self.backend = f"ultralytics:{model_path}"
        except Exception as exc:
            self.error = str(exc)

    def locate(self, bgr: np.ndarray, depth_raw: np.ndarray | None = None):
        h, w = bgr.shape[:2]
        if self.model is None:
            self.last_count = 1
            return (0, 0, w, h), .0, self.backend
        result = self.model.predict(bgr, classes=[0], conf=self.confidence, verbose=False)[0]
        candidates = []
        if result.boxes is not None:
            for xyxy, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                box = clamp_box(tuple(xyxy), bgr.shape)
                x1, y1, x2, y2 = box
                area = (x2-x1)*(y2-y1)/(w*h)
                cx, cy = (x1+x2)/2/w, (y1+y2)/2/h
                center_score = 1.0 - min(1.0, math.hypot(cx-.5, cy-.5))
                near_score = 0.0
                if depth_raw is not None:
                    values = depth_raw[y1:y2, x1:x2]
                    values = values[values > 0]
                    if values.size:
                        near_score = 1.0 / max(float(np.median(values)), 1.0)
                score = 2.0*area + .5*center_score + 500.0*near_score
                candidates.append((score, box, float(conf)))
        if not candidates:
            self.last_count = 0
            return None, .0, self.backend
        self.last_count = len(candidates)
        _, box, conf = max(candidates, key=lambda item: item[0])
        return box, conf, self.backend


def measure_person_3d(depth_raw: np.ndarray, box: tuple[int, int, int, int], depth_scale: float, intrinsics: dict) -> dict:
    """Measure a detected person's robust 3-D extent in camera coordinates."""
    x1, y1, x2, y2 = clamp_box(box, depth_raw.shape)
    raw = depth_raw[y1:y2, x1:x2]
    z = raw.astype(np.float32)*float(depth_scale)
    valid0 = (z > .25) & (z < 4.0)
    valid_ratio = float(valid0.mean()) if valid0.size else 0.0
    if np.count_nonzero(valid0) < 300:
        return {"state": "UNKNOWN", "valid_ratio": valid_ratio}
    median = float(np.median(z[valid0]))
    valid = valid0 & (np.abs(z-median) < .50)
    yy, xx = np.indices(z.shape, dtype=np.float32)
    zz = z[valid]; u = xx[valid]+x1; v = yy[valid]+y1
    X = (u-float(intrinsics["ppx"]))/float(intrinsics["fx"])*zz
    Y = (v-float(intrinsics["ppy"]))/float(intrinsics["fy"])*zz
    width = float(np.percentile(X, 95)-np.percentile(X, 5))
    height = float(np.percentile(Y, 95)-np.percentile(Y, 5))
    thickness = float(np.percentile(zz, 95)-np.percentile(zz, 5))
    geometry_ok = .12 <= width <= 2.0 and .18 <= height <= 2.5
    state = "3D_OK" if geometry_ok and thickness >= .035 else ("FLAT" if geometry_ok else "UNKNOWN")
    return {"state": state, "valid_ratio": valid_ratio, "distance_m": median,
            "width_m": width, "height_m": height, "thickness_m": thickness}


def depth_preview(depth_raw: np.ndarray, scale: float, near=.3, far=2.0):
    z = depth_raw.astype(np.float32)*scale
    valid = depth_raw > 0
    gray = np.zeros(z.shape, np.uint8)
    clipped = np.clip(z, near, far)
    gray[valid] = ((far-clipped[valid])/(far-near)*255).astype(np.uint8)
    out = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    out[~valid] = 0
    return out
