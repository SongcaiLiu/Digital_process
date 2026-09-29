# 6DRepNet 头姿对照实验

原有 PnP 头姿算法仍是默认模式。本实验模式使用公开的 6DRepNet 预训练模型从 RGB 人脸裁剪图预测俯仰、左右转头、侧倾角；后面的自动参考、动作状态、真人门槛和眨眼计算仍沿用本项目原逻辑。它不需要用你采集的两人数据训练。公开数据集上的精度不是本机 D455 场景的保证。

## 安装与运行

在 PowerShell 中激活 `digital_process` 后：

```powershell
python -m pip install --no-deps sixdrepnet==0.1.6 scipy==1.15.3
python live_rgbd_system.py --head-pose-backend 6drepnet
```

第一次运行时，官方包会自动下载约 150 MB 的预训练权重到用户的 PyTorch 缓存。要比较原算法：

```powershell
python live_rgbd_system.py --head-pose-backend pnp
```

画面标题会显示当前头姿算法。6DRepNet 默认每 2 帧推理一次，低头、抬头基础阈值均为 10°；PnP 仍为原来的 3.5°。若动作较快而漏计，可试 `--head-pose-interval 1`，但帧率可能降低。若方向颠倒，可试 `--head-down-direction positive`；实验模式默认方向为 negative。阈值可用 `--head-down-threshold 8 --head-up-threshold 8` 单独调整。真人状态下可增加动作计数；若动作造成分类短暂变红，最近一次确认真人后的 0.6 秒宽限期内也可以完成计数。

## 建议的对照方法

先固定相机、座位和距离，分别用 PnP 与 6DRepNet 做正视低头/抬头、左右侧脸低头/抬头、仅前后移动身体三组实验。每组 10 次，记录漏计、误计和画面 FPS。先看当前状态方向是否正确，再比较计数。较大侧脸或 MediaPipe 人脸丢失时，后续动作链仍可能暂停；6DRepNet 只能改善角度估计本身，无法修复人脸检测或 RGB-D 真人判断失败。

参考：[6DRepNet 官方代码](https://github.com/thohemp/6DRepNet)；[论文](https://arxiv.org/abs/2202.12555)。
