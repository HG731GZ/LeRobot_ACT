# UR5e 数据转换为 LeRobot ACT 格式说明

本文档说明如何使用 `scripts/convert_ur5e_to_lerobot.py`，把 UR5e 采集的 CSV 数值数据和相机图像转换成 LeRobot ACT 可读取的数据集。

## 环境安装

建议在 `LeRobot` conda 环境中运行：

```bash
conda activate LeRobot
python -m pip install -r requirements.txt
conda install -n LeRobot -c conda-forge ffmpeg -y
```

其中 `requirements.txt` 安装 Python 包，`ffmpeg` 负责给 LeRobot 默认的视频解码后端提供动态库。

如果要使用 NVIDIA GPU 编码，确认系统能看到显卡和 NVENC：

```bash
nvidia-smi
conda run -n LeRobot ffmpeg -hide_banner -encoders | grep h264_nvenc
```

当前脚本默认 `--vcodec auto`。在本机 `LeRobot` 环境中，LeRobot 会把它解析为 `h264_nvenc`，即 NVIDIA GPU 编码。

## 原始数据目录

单条轨迹的目录结构应类似：

```text
data/test1/
  episode_000/
    numeric/
      TCP_POSE.csv
      GRIPPER.csv
    images/
      CAMERA_1/
        step_000000_rgb.png
        step_000001_rgb.png
      CAMERA_2/
        step_000000_rgb.png
        step_000001_rgb.png
```

`TCP_POSE.csv` 需要包含：

```text
timestamp,TCP_POSE_1,TCP_POSE_2,TCP_POSE_3,TCP_POSE_4,TCP_POSE_5,TCP_POSE_6
```

`GRIPPER.csv` 需要包含：

```text
timestamp,GRIPPER_1,GRIPPER_2
```

其中 `TCP_POSE_1..3` 是 TCP 位置，`TCP_POSE_4..6` 是 UR 风格的旋转向量 `rx, ry, rz`；`GRIPPER_1` 是夹钳开度，`GRIPPER_2` 是夹钳电流。

## 批量转换

可以批量转换。是的，只要在 `test1` 文件夹里并列放置多个 `episode_xxx` 文件夹即可，例如：

```text
data/test1/
  episode_000/
  episode_001/
  episode_002/
```

然后运行：

```bash
conda activate LeRobot
python scripts/convert_ur5e_to_lerobot.py \
  --raw-root data/test1 \
  --output-root data/lerobot/ur5e_cylinder_to_box \
  --repo-id local/ur5e_cylinder_to_box \
  --overwrite
```

脚本会自动发现 `data/test1` 下所有包含 `numeric/TCP_POSE.csv` 和 `numeric/GRIPPER.csv` 的 episode 目录。也可以把 `--raw-root` 直接指向单个 episode 目录，只转换这一条。

注意：批量转换时，不同 episode 的相机集合和图像尺寸需要一致。例如第一条 episode 有 `CAMERA_1`、`CAMERA_2`，后续 episode 也应有同样的相机目录和分辨率。

## GPU 视频编码

脚本默认使用：

```bash
--vcodec auto
```

LeRobot 会自动探测硬件编码器。在当前机器上，探测结果是：

```text
h264_nvenc
```

因此默认会走 NVIDIA NVENC，而不是 CPU 编码。如果想明确指定，也可以写：

```bash
--vcodec h264_nvenc
```

如果想强制回到 CPU 编码，可以写：

```bash
--vcodec h264
```

默认模式会先把每帧图像写成临时 PNG，再调用编码器生成 MP4。这样最稳，不容易丢帧，但仍然有图像读写开销。

如果想进一步加速，可以启用 streaming 编码：

```bash
python scripts/convert_ur5e_to_lerobot.py \
  --raw-root data/test1 \
  --output-root data/lerobot/ur5e_cylinder_to_box \
  --repo-id local/ur5e_cylinder_to_box \
  --overwrite \
  --streaming-encoding
```

streaming 模式会在 `add_frame` 时直接把图像送进视频编码器，减少“临时 PNG 写入再读取”的往返。对 GPU 编码更友好。脚本默认把 `--encoder-queue-maxsize` 设为 `1024`，用于降低离线批量转换时队列满导致丢帧的风险。

如果看到类似 `dropped frame(s)` 的警告，说明编码速度跟不上图像送入速度。可以调大队列：

```bash
--encoder-queue-maxsize 4096
```

或者关闭 streaming，回到更稳的两阶段编码：

```bash
--no-streaming-encoding
```

本机样例 `episode_000` 测试结果：`--vcodec auto` 解析为 `h264_nvenc`；再加 `--streaming-encoding` 后，可以正常生成 272 帧视频，没有丢帧警告。

## 长度对齐和截断

同一条 episode 内，脚本按行号和图像序号对齐各模态：

- `TCP_POSE.csv` 第 i 行
- `GRIPPER.csv` 第 i 行
- `CAMERA_1/step_00000i_rgb.png`
- `CAMERA_2/step_00000i_rgb.png`

如果图像数量少于数值数据，脚本会取所有模态的公共最短长度，直接丢弃末尾多出来的数据。dry-run 会打印每条 episode 实际保留的长度和丢弃数量：

```bash
python scripts/convert_ur5e_to_lerobot.py --raw-root data/test1 --dry-run
```

## 输出字段

转换后的 LeRobot 数据集包含：

```text
observation.state
action
observation.images.camera_1
observation.images.camera_2
```

`observation.state` 是 8 维：

```text
[tcp_x, tcp_y, tcp_z, tcp_rx, tcp_ry, tcp_rz, gripper_opening, gripper_current]
```

这里的 `tcp_rx, tcp_ry, tcp_rz` 保留原始 UR 旋转向量。这样做的原因是 state 表示当前观测，原始数据最直接，也便于回放和排查。

`action` 默认是 7 维增量：

```text
[delta_tcp_x, delta_tcp_y, delta_tcp_z,
 delta_tcp_rx, delta_tcp_ry, delta_tcp_rz,
 delta_gripper_opening]
```

第 0 帧 action 固定为全 0。第 i 帧 action 表示从第 i-1 帧到第 i 帧的变化量。

## 姿态表达和旋转增量

UR 的 TCP 后三位 `rx, ry, rz` 是旋转向量，也可以理解为轴角表达：向量方向是旋转轴，向量长度是旋转角度。

对单帧姿态来说，旋转向量很紧凑；但对增量 action 来说，不能总是直接做：

```text
[rx_i - rx_{i-1}, ry_i - ry_{i-1}, rz_i - rz_{i-1}]
```

原因是同一个物理姿态可能有多个等价旋转向量表示，尤其在旋转角接近 `pi` 或 `-pi` 附近时，数值可能发生跳变。此时直接相减会把一个很小的真实旋转误判成接近 `2*pi` 的巨大增量。

本脚本默认使用更稳定的相对旋转向量：

```text
R_delta = R_prev.T @ R_curr
delta_rotvec = log(R_delta)
```

也就是说：

1. 先把第 i-1 帧和第 i 帧的 UR 旋转向量转换为旋转矩阵。
2. 计算从上一帧到当前帧的相对旋转。
3. 再把这个相对旋转转换回旋转向量，作为 action 的旋转增量。

这对增量控制更友好，因为 action 表示的是实际的小步相对运动，而不是两个可能跳变的绝对旋转向量之间的坐标差。

在当前 `data/test1/episode_000` 样例里，直接相减的旋转增量最大值约为 `6.28 rad`，而相对旋转向量的最大值约为 `0.02 rad`。这说明样例轨迹已经出现了接近 `-pi/pi` 等价表示导致的跳变，默认的 `relative-rotvec` 是必要的。

如需使用原始旋转向量直接相减，可以显式指定：

```bash
python scripts/convert_ur5e_to_lerobot.py \
  --raw-root data/test1 \
  --rotation-delta-mode raw-rotvec \
  --overwrite
```

一般不建议这样做，除非你后续控制器或训练代码明确要求直接相减的 UR rotvec 增量。

## 常用参数

指定输出目录：

```bash
--output-root data/lerobot/ur5e_cylinder_to_box
```

指定数据集 ID：

```bash
--repo-id local/ur5e_cylinder_to_box
```

指定任务文字：

```bash
--task "Pick up the cylinder and place it into the box"
```

手动指定 FPS：

```bash
--fps 20
```

只转换部分相机：

```bash
--cameras CAMERA_1 CAMERA_2
```

使用图像文件而不是视频存储：

```bash
--image-storage image
```

使用 GPU 自动编码：

```bash
--vcodec auto
```

启用 streaming 加速：

```bash
--streaming-encoding
```

默认会自动从 TCP 时间戳估计 FPS，并四舍五入成整数。当前样例轨迹估计为 `20 FPS`。

## 验证输出

转换完成后，可以用 LeRobot 读回检查：

```bash
conda activate LeRobot
python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset(
    "local/ur5e_cylinder_to_box",
    root="data/lerobot/ur5e_cylinder_to_box",
)
print(ds)
sample = ds[0]
print(sample["action"])
print(sample["observation.state"].shape)
print(sample["observation.images.camera_1"].shape)
PY
```

正常情况下，第 0 帧 action 应为全 0，图像张量形状为 `(3, height, width)`。
