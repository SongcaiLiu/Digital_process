"""Download the external model files required by the live demo."""

from pathlib import Path
from urllib.request import urlretrieve


FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


def main():
    models = Path("models")
    models.mkdir(exist_ok=True)
    face_model = models / "face_landmarker.task"
    if not face_model.exists():
        print("正在下载 MediaPipe Face Landmarker...")
        urlretrieve(FACE_MODEL_URL, face_model)
    else:
        print("MediaPipe模型已存在")

    print("正在检查 YOLO 权重...")
    from ultralytics import YOLO
    YOLO("yolo11n.pt")
    print("模型准备完成")


if __name__ == "__main__":
    main()
