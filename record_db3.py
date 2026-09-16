"""Record synchronized raw RealSense streams to the current .db3 format."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from rgbd_common import colorize_depth


def main() -> None:
    parser = argparse.ArgumentParser(description="将 D455 原始流录制为 .db3")
    parser.add_argument("output", type=Path)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".db3":
        parser.error("新版 RealSense 录制文件必须以 .db3 结尾")
    if args.output.exists():
        parser.error("输出文件已经存在；为避免覆盖，请换一个文件名")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_record_to_file(str(args.output.resolve()))
    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    start = time.monotonic()
    frame_count = 0
    print(f"正在录制：{args.output.resolve()}")
    try:
        while time.monotonic() - start < args.seconds:
            frames = pipeline.wait_for_frames(5000)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            frame_count += 1
            if not args.headless:
                color = np.asanyarray(color_frame.get_data()).copy()
                depth = np.asanyarray(depth_frame.get_data())
                preview = colorize_depth(depth, depth_scale)
                cv2.putText(
                    color,
                    f"RAW DB3 REC {time.monotonic() - start:.1f}/{args.seconds:.1f}s",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2,
                )
                cv2.imshow("D455 DB3 Recorder", np.hstack((color, preview)))
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    print(f"录制完成：{frame_count} 个 frameset，{args.output.resolve()}")


if __name__ == "__main__":
    main()
