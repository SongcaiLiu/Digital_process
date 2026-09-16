"""Quick, camera-free readiness check for the initial project."""

import importlib
import os
from pathlib import Path


def import_status(module, capability=None):
    try:
        value = importlib.import_module(module)
        if capability and not hasattr(value, capability):
            return "BROKEN"
        return "OK"
    except Exception:
        return "BROKEN"


def mediapipe_status():
    try:
        mp = importlib.import_module("mediapipe")
        if hasattr(mp, "solutions"):
            return "OK"
        if (hasattr(mp, "tasks") and hasattr(mp.tasks.vision, "FaceLandmarker")
                and Path("models/face_landmarker.task").is_file()):
            return "OK"
        return "BROKEN"
    except Exception:
        return "BROKEN"


def main():
    yolo_config = Path("models/ultralytics_config").resolve()
    yolo_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_config))
    required = {"cv2": ("OpenCV", None), "numpy": ("NumPy", None), "pyrealsense2": ("RealSense", None)}
    optional = {"ultralytics": ("YOLO人体检测", "YOLO")}
    print("基础组件：")
    for module, (name, capability) in required.items():
        print(f"  {import_status(module, capability):7} {name}")
    print("完整功能组件：")
    print(f"  {mediapipe_status():7} 人脸关键点/动作")
    for module, (name, capability) in optional.items():
        print(f"  {import_status(module, capability):7} {name}")
    for path in (Path("processed/real_features.csv"), Path("models/liveness_oneclass.json")):
        print(f"  {'OK' if path.is_file() else 'MISSING':7} {path}")


if __name__ == "__main__":
    main()
