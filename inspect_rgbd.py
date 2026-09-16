"""Validate and browse a captured RGB-D session."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from rgbd_common import colorize_depth


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 RGB-D 数据完整性和距离")
    parser.add_argument("session", type=Path)
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args()
    metadata_path = args.session / "metadata.json"
    if not metadata_path.is_file():
        parser.error(f"不是有效场次目录：{args.session}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    scale = float(metadata["depth_scale_m_per_unit"])
    rgb = {path.stem: path for path in (args.session / "rgb").glob("*.png")}
    depth = {path.stem: path for path in (args.session / "depth").glob("*.png")}
    mask = {path.stem: path for path in (args.session / "mask").glob("*.png")}
    common = sorted(rgb.keys() & depth.keys() & mask.keys())
    missing = sorted((rgb.keys() | depth.keys() | mask.keys()) - set(common))
    if not common:
        raise RuntimeError("没有找到完整的 RGB/depth/mask 图片对")

    rows = []
    for stem in common:
        color = cv2.imread(str(rgb[stem]), cv2.IMREAD_COLOR)
        raw = cv2.imread(str(depth[stem]), cv2.IMREAD_UNCHANGED)
        valid = raw > 0 if raw is not None else np.zeros(1, dtype=bool)
        if color is None or raw is None or raw.dtype != np.uint16 or color.shape[:2] != raw.shape:
            raise RuntimeError(f"格式或尺寸错误：{stem}")
        values_m = raw[valid].astype(np.float32) * scale
        rows.append(
            {
                "frame": stem,
                "valid_ratio": float(valid.mean()),
                "median_m": float(np.median(values_m)) if values_m.size else 0.0,
                "min_m": float(np.percentile(values_m, 1)) if values_m.size else 0.0,
                "max_m": float(np.percentile(values_m, 99)) if values_m.size else 0.0,
            }
        )
    report_path = args.session / "quality_report.csv"
    with report_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"完整图片对：{len(common)}；缺失配对文件：{len(missing)}")
    print(f"深度类型：uint16；深度单位：{scale} 米/数值")
    print(f"平均有效深度比例：{np.mean([row['valid_ratio'] for row in rows]):.1%}")
    print(f"质量报告：{report_path.resolve()}")
    if args.no_window:
        return

    index = 0
    while True:
        stem = common[index]
        color = cv2.imread(str(rgb[stem]), cv2.IMREAD_COLOR)
        raw = cv2.imread(str(depth[stem]), cv2.IMREAD_UNCHANGED)
        preview = colorize_depth(raw, scale)
        row = rows[index]
        cv2.putText(
            color,
            f"{stem} median {row['median_m']:.3f}m valid {row['valid_ratio']:.1%}",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
        )
        cv2.imshow("RGB-D Inspector | A/D: previous/next, Q: quit", np.hstack((color, preview)))
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (ord("a"), ord("A"), 81):
            index = (index - 1) % len(common)
        elif key in (ord("d"), ord("D"), 83, 32):
            index = (index + 1) % len(common)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
