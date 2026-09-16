"""Detect/crop faces and extract aligned depth/point-cloud features."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from rgbd_project_core import FEATURE_NAMES, FaceLocator, depth_preview, extract_depth_features, iter_sessions


def main():
    p = argparse.ArgumentParser(description="生成初版 RGB-D 人脸裁剪和三维特征")
    p.add_argument("--input", type=Path, default=Path("data"))
    p.add_argument("--output", type=Path, default=Path("processed"))
    p.add_argument("--label", default="real")
    p.add_argument("--no-center-fallback", action="store_true", help="检测不到人脸时丢弃，不使用中心裁剪")
    p.add_argument("--preview-every", type=int, default=5, help="每隔多少帧保存一组检查图，0表示不保存")
    args = p.parse_args()
    locator = FaceLocator(allow_fallback=not args.no_center_fallback, static_image_mode=True)
    total = kept = fallback = 0
    all_rows = []
    sessions = list(iter_sessions(args.input, args.label))
    if not sessions:
        raise SystemExit(f"没有找到场次：{args.input / args.label}")
    for session in sessions:
        metadata = json.loads((session/"metadata.json").read_text(encoding="utf-8"))
        scale = float(metadata["depth_scale_m_per_unit"])
        intr = metadata["color_intrinsics"]
        subject, session_name = session.parent.name, session.name
        out = args.output/args.label/subject/session_name
        for folder in ("face_rgb", "face_depth", "face_mask", "face_depth_preview"):
            (out/folder).mkdir(parents=True, exist_ok=True)
        rows = []
        for rgb_path in sorted((session/"rgb").glob("*.png")):
            total += 1
            depth_path = session/"depth"/rgb_path.name
            color = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if color is None or depth is None or depth.dtype != np.uint16:
                continue
            box, _, source = locator.locate(color)
            if box is None:
                continue
            fallback += int(source == "center-fallback")
            features, quality = extract_depth_features(depth, box, scale, intr)
            if features is None:
                continue
            kept += 1
            x1, y1, x2, y2 = box
            row = {
                "label": args.label, "subject": subject, "session": session_name,
                "frame": rgb_path.stem, "face_source": source,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "median_depth_m": quality["median_depth_m"],
                "point_count": quality["point_count"],
            }
            row.update(dict(zip(FEATURE_NAMES, features)))
            rows.append(row); all_rows.append(row)
            if args.preview_every > 0 and int(rgb_path.stem) % args.preview_every == 0:
                crop_rgb = color[y1:y2, x1:x2]
                crop_depth = depth[y1:y2, x1:x2]
                cv2.imwrite(str(out/"face_rgb"/rgb_path.name), crop_rgb)
                cv2.imwrite(str(out/"face_depth"/rgb_path.name), crop_depth)
                cv2.imwrite(str(out/"face_mask"/rgb_path.name), np.where(crop_depth>0,255,0).astype(np.uint8))
                cv2.imwrite(str(out/"face_depth_preview"/(rgb_path.stem+".jpg")), depth_preview(crop_depth, scale))
        if rows:
            with (out/"features.csv").open("w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
            (out/"processing.json").write_text(json.dumps({
                "source": str(session.resolve()), "face_backend": locator.backend,
                "frames_kept": len(rows), "feature_names": FEATURE_NAMES,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"{subject}/{session_name}: 保留 {len(rows)} 帧")
    locator.close()
    if not all_rows:
        raise SystemExit("没有提取到可用特征")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/f"{args.label}_features.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys()); writer.writeheader(); writer.writerows(all_rows)
    print(f"完成：输入 {total} 帧，保留 {kept} 帧；中心降级裁剪 {fallback} 帧")
    print(f"汇总特征：{(args.output/f'{args.label}_features.csv').resolve()}")


if __name__ == "__main__":
    main()
