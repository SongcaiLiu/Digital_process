# RGB-D 人体检测与动作识别

基于 Intel RealSense D455 的数字图像处理课设。系统使用 YOLO 检测人体、MediaPipe 定位人脸与眼部关键点、RGB-D 几何特征区分真人/照片/未知，并用 PnP 或 6DRepNet 估计头姿、识别低头和抬头。

## 常用入口

| 文件 | 用途 |
|---|---|
| `live_rgbd_system.py` | 实时演示主程序 |
| `start_demo.bat` | Windows 一键启动默认演示 |
| `check_project.py` | 检查完整演示依赖和模型 |
| `check_environment.py` | 检查基础环境及 D455 连接 |
| `main.py` | RGB-D 图片对采集入口 |
| `record_db3.py` | 录制 RealSense `.db3` |
| `recording_to_dataset.py` | 从 `.db3`/`.bag` 抽取数据集 |
| `inspect_rgbd.py` | 检查采集数据质量 |
| `analyze_pose_test.py` | 重建姿态测试报告 |

## 核心模块

| 文件 | 职责 |
|---|---|
| `rgbd_project_core.py` | YOLO、MediaPipe、人体三维测量 |
| `geometric_liveness.py` | RGB-D 真人/照片几何判别 |
| `action_calibration.py` | 自动姿态参考、眨眼及头部动作状态机 |
| `diagnostic_recording.py` | 后台记录和汇总姿态测试 |
| `rgbd_common.py` | RGB-D 采集、保存的公共工具 |
| `geometric_config.json` | 几何判别阈值 |

## 运行

```powershell
conda activate digital_process
python check_project.py
python live_rgbd_system.py
```

使用 6DRepNet 头姿模式：

```powershell
python live_rgbd_system.py --head-pose-backend 6drepnet
```

运行全部回归测试：

```powershell
python -m tests.run_all
```

## 文档

- [RGB-D 数据采集](docs/RGBD_采集说明.md)
- [几何真人判别](docs/RGBD_几何判别说明.md)
- [姿态诊断记录](docs/姿态测试说明.md)
- [6DRepNet 对照实验](docs/6DRepNet头姿对照实验.md)

`data/`、`recordings/`、`processed/`、`demo_results/` 存放本地采集或生成数据，均不提交 Git。`models/` 和 `yolo11n.pt` 存放下载的模型权重。
