"""Check Python packages and the D455 connection before collection."""

from __future__ import annotations

import sys


def main() -> None:
    print("Python:", sys.executable)
    print("Version:", sys.version.split()[0])
    try:
        import numpy as np
        import cv2
        import pyrealsense2 as rs
    except Exception as error:
        print("环境检查失败：", repr(error))
        print("请在当前环境重新运行：python -m pip install --force-reinstall -r requirements.txt")
        raise SystemExit(1) from error

    required_numpy_api = hasattr(np, "asanyarray") and hasattr(np, "count_nonzero")
    if not required_numpy_api:
        print("NumPy 安装不完整，模块路径：", getattr(np, "__file__", None))
        print("请运行：python -m pip install --force-reinstall numpy==2.2.6")
        raise SystemExit(1)
    print("NumPy:", np.__version__, np.__file__)
    print("OpenCV:", cv2.__version__, cv2.__file__)
    print("pyrealsense2:", getattr(rs, "__version__", "已导入"))

    devices = list(rs.context().query_devices())
    if not devices:
        print("D455：未连接。请重新插入 USB 3 接口并关闭 Viewer 后重试。")
        raise SystemExit(2)
    for device in devices:
        name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)
        firmware = device.get_info(rs.camera_info.firmware_version)
        usb = (
            device.get_info(rs.camera_info.usb_type_descriptor)
            if device.supports(rs.camera_info.usb_type_descriptor)
            else "unknown"
        )
        print(f"设备：{name} | S/N {serial} | FW {firmware} | USB {usb}")


if __name__ == "__main__":
    main()
