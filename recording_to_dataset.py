"""Export aligned RGB-D pairs from current .db3 or legacy .bag recordings."""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pyrealsense2 as rs

from rgbd_common import DatasetWriter, build_metadata, colorize_depth


def main() -> None:
    parser = argparse.ArgumentParser(description="从 .db3/.bag 导出对齐 RGB-D 数据集")
    parser.add_argument("recording", type=Path)
    parser.add_argument("--label", required=True, choices=["real", "print", "screen", "mask", "test"])
    parser.add_argument("--subject", required=True)
    parser.add_argument("--session")
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--near", type=float, default=0.3)
    parser.add_argument("--far", type=float, default=3.0)
    args = parser.parse_args()
    if not args.recording.is_file():
        parser.error(f"文件不存在：{args.recording}")
    if args.recording.suffix.lower() not in (".db3", ".bag"):
        parser.error("只支持新版 .db3 或旧版 .bag")
    if args.sample_fps <= 0:
        parser.error("--sample-fps 必须大于 0")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(str(args.recording.resolve()), repeat_playback=False)
    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    session = args.session or datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = DatasetWriter(
        args.output,
        args.label,
        args.subject,
        session,
        build_metadata(
            profile,
            depth_scale,
            args.label,
            args.subject,
            args.sample_fps,
            str(args.recording.resolve()),
        ),
    )
    interval_ms = 1000.0 / args.sample_fps
    last_timestamp = -1e12
    idle_since = time.monotonic()
    try:
        while True:
            frames = pipeline.poll_for_frames()
            if not frames:
                if playback.current_status() == rs.playback_status.stopped:
                    break
                if time.monotonic() - idle_since > 10:
                    raise RuntimeError("连续 10 秒没有读到帧，请检查录制文件是否包含 RGB 和 Depth")
                time.sleep(0.001)
                continue
            idle_since = time.monotonic()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            timestamp = color_frame.get_timestamp()
            if timestamp - last_timestamp < interval_ms:
                continue
            color = np.asanyarray(color_frame.get_data()).copy()
            depth = np.asanyarray(depth_frame.get_data()).copy()
            preview = colorize_depth(depth, depth_scale, args.near, args.far)
            writer.save(
                color,
                depth,
                preview,
                timestamp,
                depth_frame.get_timestamp(),
                depth_scale,
            )
            last_timestamp = timestamp
            if writer.count % 25 == 0:
                print(f"已导出 {writer.count} 对")
    finally:
        pipeline.stop()
        writer.close()
    print(f"完成：{writer.count} 对，目录：{writer.session_dir.resolve()}")


if __name__ == "__main__":
    main()
