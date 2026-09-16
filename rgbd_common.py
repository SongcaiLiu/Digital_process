"""Shared helpers for D455 RGB-D collection tools."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


def safe_name(value: str, field: str) -> str:
    value = value.strip()
    if not value or any(char in value for char in '<>:"/\\|?*'):
        raise ValueError(f"{field} 不能为空，也不能包含 Windows 路径非法字符")
    return value


def intrinsics_to_dict(value: rs.intrinsics) -> dict:
    return {
        "width": value.width,
        "height": value.height,
        "fx": value.fx,
        "fy": value.fy,
        "ppx": value.ppx,
        "ppy": value.ppy,
        "distortion_model": str(value.model),
        "coefficients": list(value.coeffs),
    }


def device_to_dict(device: rs.device) -> dict:
    fields = {
        "name": rs.camera_info.name,
        "serial_number": rs.camera_info.serial_number,
        "firmware_version": rs.camera_info.firmware_version,
        "usb_type": rs.camera_info.usb_type_descriptor,
    }
    return {
        name: device.get_info(info) if device.supports(info) else "unknown"
        for name, info in fields.items()
    }


def colorize_depth(
    depth_raw: np.ndarray,
    depth_scale: float,
    near_m: float = 0.3,
    far_m: float = 3.0,
) -> np.ndarray:
    """Create an 8-bit preview; invalid depth is black."""
    depth_m = depth_raw.astype(np.float32) * depth_scale
    valid = depth_raw > 0
    display = np.zeros(depth_raw.shape, dtype=np.uint8)
    clipped = np.clip(depth_m, near_m, far_m)
    display[valid] = np.round(
        (far_m - clipped[valid]) / (far_m - near_m) * 255
    ).astype(np.uint8)
    result = cv2.applyColorMap(display, cv2.COLORMAP_TURBO)
    result[~valid] = 0
    return result


def center_median_depth(depth_raw: np.ndarray, depth_scale: float) -> float:
    height, width = depth_raw.shape
    region = depth_raw[
        height // 2 - 10 : height // 2 + 11,
        width // 2 - 10 : width // 2 + 11,
    ]
    valid = region[region > 0]
    return float(np.median(valid) * depth_scale) if valid.size else 0.0


@dataclass
class DatasetWriter:
    root: Path
    label: str
    subject: str
    session: str
    metadata: dict

    def __post_init__(self) -> None:
        self.label = safe_name(self.label, "label")
        self.subject = safe_name(self.subject, "subject")
        self.session = safe_name(self.session, "session")
        self.session_dir = self.root / self.label / self.subject / self.session
        if self.session_dir.exists():
            raise FileExistsError(f"采集场次已存在，请更换 --session：{self.session_dir}")
        for name in ("rgb", "depth", "mask", "depth_preview"):
            (self.session_dir / name).mkdir(parents=True, exist_ok=False)
        (self.session_dir / "metadata.json").write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._csv_file = (self.session_dir / "frames.csv").open(
            "w", newline="", encoding="utf-8-sig"
        )
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(
            [
                "frame_id",
                "system_time",
                "color_timestamp_ms",
                "depth_timestamp_ms",
                "center_depth_m",
                "valid_depth_ratio",
                "label",
                "subject",
            ]
        )
        self.count = 0

    def save(
        self,
        color: np.ndarray,
        depth_raw: np.ndarray,
        depth_preview: np.ndarray,
        color_timestamp_ms: float,
        depth_timestamp_ms: float,
        depth_scale: float,
    ) -> None:
        if depth_raw.dtype != np.uint16:
            raise TypeError(f"深度图必须是 uint16，实际为 {depth_raw.dtype}")
        stem = f"{self.count:06d}"
        mask = np.where(depth_raw > 0, 255, 0).astype(np.uint8)
        outputs = {
            self.session_dir / "rgb" / f"{stem}.png": color,
            self.session_dir / "depth" / f"{stem}.png": depth_raw,
            self.session_dir / "mask" / f"{stem}.png": mask,
            self.session_dir / "depth_preview" / f"{stem}.jpg": depth_preview,
        }
        for path, image in outputs.items():
            if not cv2.imwrite(str(path), image):
                raise OSError(f"保存失败：{path}")
        self._csv.writerow(
            [
                self.count,
                datetime.now().isoformat(timespec="milliseconds"),
                f"{color_timestamp_ms:.3f}",
                f"{depth_timestamp_ms:.3f}",
                f"{center_median_depth(depth_raw, depth_scale):.6f}",
                f"{np.count_nonzero(depth_raw) / depth_raw.size:.6f}",
                self.label,
                self.subject,
            ]
        )
        self._csv_file.flush()
        self.count += 1

    def close(self) -> None:
        self._csv_file.close()


def build_metadata(
    profile: rs.pipeline_profile,
    depth_scale: float,
    label: str,
    subject: str,
    sample_fps: float,
    source: str,
) -> dict:
    depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    extrinsics = depth_profile.get_extrinsics_to(color_profile)
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "label": label,
        "subject": subject,
        "device": device_to_dict(profile.get_device()),
        "sample_fps": sample_fps,
        "depth_scale_m_per_unit": depth_scale,
        "depth_formula": "distance_m = uint16_png_value * depth_scale_m_per_unit",
        "saved_depth": "Z16 depth aligned to RGB pixel coordinates",
        "color_intrinsics": intrinsics_to_dict(color_profile.get_intrinsics()),
        "raw_depth_intrinsics": intrinsics_to_dict(depth_profile.get_intrinsics()),
        "depth_to_color_extrinsics": {
            "rotation": list(extrinsics.rotation),
            "translation_m": list(extrinsics.translation),
        },
    }
