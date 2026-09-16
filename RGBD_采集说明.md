# D455 RGB-D 数据采集与录制说明

## 1. 四个程序分别做什么

| 程序 | 用途 |
|---|---|
| `main.py` / `rgbd_capture.py` | 直接从 D455 采集对齐后的训练图片对 |
| `record_db3.py` | 用 Python 录制完整的原始 `.db3` 文件 |
| `recording_to_dataset.py` | 从 Viewer/Python 的 `.db3` 或旧 `.bag` 抽取训练图片对 |
| `inspect_rgbd.py` | 检查 RGB、深度、掩码是否配对，生成质量报告 |

开始前先进入环境并安装依赖：

```powershell
conda activate digital_process
python -m pip install -r requirements.txt
```

安装后先运行：

```powershell
python check_environment.py
```

同一时间只能由一个程序使用 D455。运行 Python 前，应关闭 RealSense Viewer；使用 Viewer 前，应结束 Python 程序。

## 2. 推荐的数据采集方法

训练数据优先采用以下两种方法之一：

### 方法 A：直接保存训练图片对

适合先快速检查数据，也适合正式采集。程序会实时把深度对齐到 RGB 坐标，再按相同编号保存。

```powershell
python main.py --label test --subject person_001
```

窗口左边是 RGB，右边是对齐深度。按键：

- `S`：保存当前一对 RGB-D 图片。
- `R`：开始或暂停连续采集，默认每秒保存 5 对。
- `Q` 或 `Esc`：安全结束，关闭文件。

自动采集 10 秒：

```powershell
python main.py --label test --subject person_001 --seconds 10
```

正式采集真人：

```powershell
python main.py --label real --subject person_001 --session real_front_01
```

采集打印照片攻击：

```powershell
python main.py --label print --subject person_001 --session print_front_01
```

采集屏幕照片攻击：

```powershell
python main.py --label screen --subject person_001 --session phone_front_01
```

### 方法 B：先录原始 `.db3`，以后反复抽帧

正式采集更推荐保留 `.db3` 原始文件。它保存 RGB、原始深度、时间戳、内外参和传感器状态，之后可以用不同抽帧率重新生成数据集。

录制 20 秒：

```powershell
python record_db3.py recordings\person_001_real_01.db3 --seconds 20
```

无窗口录制：

```powershell
python record_db3.py recordings\person_001_real_01.db3 --seconds 20 --headless
```

从 `.db3` 每秒抽取 5 对训练数据：

```powershell
python recording_to_dataset.py recordings\person_001_real_01.db3 --label real --subject person_001 --session real_01 --sample-fps 5
```

旧版 `.bag` 使用同一条命令：

```powershell
python recording_to_dataset.py recordings\old_recording.bag --label real --subject person_001 --session legacy_01 --sample-fps 5
```

## 3. Viewer 为什么生成 `.db3` 而不是 `.bag`

这是新版 RealSense SDK 的正常变化，并不是录制错误。从 SDK 2.58.1 开始，RealSense 使用 ROS 2 `rosbag2` 的 SQLite `.db3` 作为当前录制格式；旧版 `.bag` 属于 ROS 1 格式。

新版 Viewer 中录制：

1. 关闭所有正在读取 D455 的 Python 程序。
2. 打开 Viewer，展开 D455。
3. 设置并开启 `Stereo Module` 的 `Depth / Z16 / 640×480 / 30 FPS`。
4. 设置并开启 `RGB Camera / 640×480 / 30 FPS`。
5. 点击设备名称旁边的菜单（条形图或 More 图标）。
6. 选择 `Record to File...`，文件名以 `.db3` 结尾。
7. 出现红色录制圆点后，完成真人或照片动作。
8. 停止所有流或点击停止录制，等待文件写入完成后再关闭 Viewer。

Viewer 回放：点击 `Add Source` → `Load Recorded Sequence` → 选择 `.db3`。

当前版本不能直接要求 Viewer 生成旧 `.bag`。如果没有必须兼容旧软件的要求，应继续使用 `.db3`。没有必要为了得到 `.bag` 降级 SDK。

Viewer 设置中的压缩会影响其他 ROS 2 工具的兼容性。如果还要用 Foxglove 或 ROS 2 检查文件，在 `Settings > Playback & Record` 中选择 `Never Compress`；只用本项目和 RealSense SDK 回放时可以使用压缩节省空间。

## 4. 保存后的数据长什么样

```text
data/
└─ real/
   └─ person_001/
      └─ real_front_01/
         ├─ rgb/             # 000000.png：RGB 彩色图
         ├─ depth/           # 000000.png：对齐后的 uint16 原始深度
         ├─ mask/            # 000000.png：255 有效，0 无效
         ├─ depth_preview/   # 000000.jpg：只用于观看
         ├─ metadata.json    # 相机、深度单位、内参、外参
         └─ frames.csv       # 时间戳、有效比例、中心距离
```

同名 RGB 和 depth 是一对。深度 PNG 必须用 `cv2.IMREAD_UNCHANGED` 读取：

```python
depth_raw = cv2.imread("depth/000000.png", cv2.IMREAD_UNCHANGED)
distance_m = depth_raw[y, x] * depth_scale_m_per_unit
```

不要使用 `depth_preview/*.jpg` 训练距离网络。它是 8 位伪彩色图，已经丢失原始测距精度。

## 5. 怎样检查数据是否准确

```powershell
python inspect_rgbd.py data\test\person_001\你的场次目录
```

程序会检查：

- RGB、depth、mask 文件名是否一一对应。
- 深度是否保持 `uint16`。
- RGB 与深度尺寸是否相同。
- 每帧有效深度比例。
- 深度的中位数及 1%～99% 范围。

并生成 `quality_report.csv`。窗口中按 `A/D` 或左右方向键查看前后帧，按 `Q` 退出。

只生成报告、不弹窗口：

```powershell
python inspect_rgbd.py data\test\person_001\你的场次目录 --no-window
```

测量准确性建议使用卷尺验证：让一块平整纸板分别位于约 0.6、0.8、1.0、1.5 米，在图像中央保存 20 帧，比较 `frames.csv` 的 `center_depth_m` 与卷尺距离。不要用单个像素判断，程序使用中心 21×21 区域的有效值中位数。

## 6. 正式采集真人与照片

相机固定，尽量保持人脸在 0.6～1.5 米范围。每位参与者使用匿名编号，每种条件录制独立场次：

- 真人：正面、左右转头、低头抬头、眨眼、前后小幅移动。
- 打印照片：正放、倾斜、前后移动、轻微弯曲、不同纸张。
- 屏幕照片：手机、平板或显示器，改变角度和屏幕亮度。
- 环境：至少两种光照、两个背景、两个不同时间段。

每个场次录 10～20 秒即可；图片抽帧率建议 3～5 FPS。连续视频的 30 帧非常相似，全部作为训练图片只会制造大量重复数据。

训练集、验证集、测试集必须按参与者编号划分。同一个人及其照片不能同时出现在训练集和测试集，也不能把同一段录制随机拆帧到不同集合。

## 7. 普通视频该怎么使用

普通 RGB MP4 可以观看，也可以辅助标注动作，但不能保存真实深度。彩色深度预览 MP4 同样已经失去 16 位深度，不应用作精确 RGB-D 训练输入。

本项目推荐：

```text
.db3/.bag = 可重新处理的原始录像
rgb + uint16 depth PNG = 实际训练数据
depth_preview JPG/MP4 = 人工检查画面
```

先保存原始 `.db3`，再通过 `recording_to_dataset.py` 导出图片对，是最容易修正抽帧率、对齐方法和数据标签的方案。
