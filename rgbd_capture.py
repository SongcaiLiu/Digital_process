"""Interactive or timed RGB-D dataset collection from a live D455."""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from rgbd_common import DatasetWriter, build_metadata, center_median_depth, colorize_depth


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="D455 RGB-D 数据采集")
    parser.add_argument("--label", required=True, choices=["real", "print", "screen", "mask", "test"])
    parser.add_argument("--subject", required=True, help="匿名编号，例如 person_001")
    parser.add_argument("--session", help="场次名；默认使用当前时间")
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--sample-fps", type=float, default=5.0, help="连续采集每秒保存几对")
    parser.add_argument("--seconds", type=float, default=0.0, help="自动录制秒数；0 为键盘控制")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--near", type=float, default=0.3, help="预览最近距离（米）")
    parser.add_argument("--far", type=float, default=3.0, help="预览最远距离（米）")
    args = parser.parse_args()
    if not 0 < args.sample_fps <= args.fps:
        parser.error("--sample-fps 必须大于 0 且不超过相机 FPS")
    if args.headless and args.seconds <= 0:
        parser.error("--headless 必须同时指定 --seconds")
    if args.near >= args.far:
        parser.error("--near 必须小于 --far")
    return args


def main() -> None:
    args = arguments()
    session = args.session or datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    writer = DatasetWriter(
        args.output,
        args.label,
        args.subject,
        session,
        build_metadata(profile, depth_scale, args.label, args.subject, args.sample_fps, "live D455"),
    )
    recording = args.seconds > 0
    start_time = time.monotonic()
    last_save = -1e9
    warmup = 0

    print(f"保存目录：{writer.session_dir.resolve()}")
    print("按键：R 开始/暂停连续采集，S 保存单帧，Q 或 Esc 退出")
    try:
        while True:
            frames = pipeline.wait_for_frames(5000)
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            warmup += 1
            color = np.asanyarray(color_frame.get_data()).copy()
            depth_raw = np.asanyarray(depth_frame.get_data()).copy()
            preview = colorize_depth(depth_raw, depth_scale, args.near, args.far)
            valid_ratio = np.count_nonzero(depth_raw) / depth_raw.size
            center_depth = center_median_depth(depth_raw, depth_scale)
            now = time.monotonic()
            save_this_frame = (
                recording
                and warmup > 15
                and now - last_save >= 1.0 / args.sample_fps
            )

            display = color.copy()
            status = "REC" if recording else "READY"
            status_color = (0, 0, 255) if recording else (0, 255, 0)
            cv2.putText(display, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(
                display,
                f"center {center_depth:.3f} m | valid {valid_ratio:.1%} | saved {writer.count}",
                (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
            )
            key = -1
            if not args.headless:
                cv2.imshow("D455 RGB-D Capture | RGB + aligned depth", np.hstack((display, preview)))
                key = cv2.waitKey(1) & 0xFF
            if key in (ord("r"), ord("R")):
                recording = not recording
                print("连续采集：开始" if recording else "连续采集：暂停")
            if key in (ord("s"), ord("S")):
                save_this_frame = True
            if key in (ord("q"), ord("Q"), 27):
                break

            if save_this_frame:
                writer.save(
                    color,
                    depth_raw,
                    preview,
                    color_frame.get_timestamp(),
                    depth_frame.get_timestamp(),
                    depth_scale,
                )
                last_save = now
            if args.seconds > 0 and now - start_time >= args.seconds:
                break
    finally:
        pipeline.stop()
        writer.close()
        cv2.destroyAllWindows()
    print(f"完成：保存 {writer.count} 对 RGB-D 数据")


if __name__ == "__main__":
    main()
