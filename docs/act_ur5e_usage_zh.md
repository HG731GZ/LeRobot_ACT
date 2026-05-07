# UR5e 圆柱入盒任务 ACT 训练与测试说明

本文档对应：

- `scripts/train_act_ur5e.py`
- `scripts/test_act_ur5e.py`
- 数据集：`data/lerobot/ur5e_cylinder_to_box`

当前数据集是 LeRobot v3.0 格式，共 50 条轨迹、11515 帧、19 FPS、两路 720p 相机。`observation.state` 为 8 维：

```text
[tcp_x, tcp_y, tcp_z, tcp_rx, tcp_ry, tcp_rz, gripper_opening, gripper_current]
```

`action` 为 7 维相邻帧增量：

```text
[delta_tcp_x, delta_tcp_y, delta_tcp_z,
 delta_tcp_rx, delta_tcp_ry, delta_tcp_rz,
 delta_gripper_opening]
```

旋转增量默认是 `relative-rotvec`，测试脚本也按这个方式把网络动作还原成下一步 UR TCP 位姿。

## 环境

```bash
conda activate LeRobot
python -m pip install -r requirements.txt
conda install -n LeRobot -c conda-forge ffmpeg -y
```

确认数据集可读：

```bash
python scripts/train_act_ur5e.py --dry-run
```

本仓库训练脚本默认使用 `--video-backend pyav`，因为当前环境里 `torchcodec` 可能受 FFmpeg/C++ 动态库版本影响。如果你在 A6000 工作站上确认 torchcodec 可用，可以加 `--video-backend torchcodec`。

## 在 1660Ti 上训练

1660Ti 显存通常较小，脚本会使用较小 ACT 模型、`batch_size=1` 和 AMP：

```bash
conda activate LeRobot
python scripts/train_act_ur5e.py \
  --preset 1660ti \
  --output-dir outputs/train/ur5e_act_1660ti
```

如果 torchvision 第一次下载 ImageNet ResNet18 权重失败，可以先不用预训练 backbone：

```bash
python scripts/train_act_ur5e.py \
  --preset 1660ti \
  --no-pretrained-backbone \
  --output-dir outputs/train/ur5e_act_1660ti_no_pretrain
```

## 在 A6000 工作站上训练

A6000 可以使用更大的模型和 batch：

```bash
conda activate LeRobot
python scripts/train_act_ur5e.py \
  --preset a6000 \
  --output-dir outputs/train/ur5e_act_a6000
```

常用可调参数：

```bash
python scripts/train_act_ur5e.py \
  --preset a6000 \
  --batch-size 12 \
  --steps 80000 \
  --chunk-size 80 \
  --n-action-steps 40 \
  --save-freq 5000 \
  --output-dir outputs/train/ur5e_act_a6000_chunk80
```

训练输出的最新 checkpoint 通常在：

```text
outputs/train/<run_name>/checkpoints/last/pretrained_model
```

继续训练：

```bash
python scripts/train_act_ur5e.py \
  --resume-config outputs/train/ur5e_act_a6000/checkpoints/last/pretrained_model/train_config.json
```

## 无机械臂测试推理链路

本机没有接真实 UR5e，测试脚本默认从数据集中读观测帧，并打印策略输出动作和目标 TCP：

```bash
python scripts/test_act_ur5e.py \
  --policy-path outputs/train/ur5e_act_a6000 \
  --backend mock_dataset \
  --max-steps 20
```

`--policy-path` 可以传训练目录、`checkpoints/last`，或直接传 `pretrained_model` 目录。

## 实时推理接口

如果已经有自己的 UR5e 控制程序、相机采集程序和夹爪接口，可以直接使用 `scripts/ur5e_act_realtime_inference.py`。这个脚本不包含 mock，也不处理机器人连接，只提供一个推理类和一个公开推理方法。

初始化：

```python
from scripts.ur5e_act_realtime_inference import UR5eACTRealtimeInference

predictor = UR5eACTRealtimeInference(
    policy_path="outputs/train/ur5e_act_a6000",
    device="cuda",
    inference_interval_s=0.05,
)
```

`inference_interval_s` 是两次推理的最小时间间隔，单位是秒。比如 `0.05` 约等于 20 Hz；如果外部控制循环自己限频，可以设为 `0`。

唯一需要调用的推理接口：

```python
next_tcp_pose = predictor.predict_next_tcp_pose(
    camera_1_rgb,
    camera_2_rgb,
    current_tcp_pose,
    gripper_opening_current,
)
```

输入要求：

```text
camera_1_rgb: np.ndarray, HWC RGB 图像，uint8 或 float
camera_2_rgb: np.ndarray, HWC RGB 图像，uint8 或 float
current_tcp_pose: np.ndarray, shape=(6,), [x, y, z, rx, ry, rz]
gripper_opening_current: np.ndarray, shape=(2,), [gripper_opening, gripper_current]
```

输出：

```text
next_tcp_pose: np.ndarray, shape=(6,), [x, y, z, rx, ry, rz]
```

最小接入示例：

```python
import numpy as np
from scripts.ur5e_act_realtime_inference import UR5eACTRealtimeInference

predictor = UR5eACTRealtimeInference(
    policy_path="outputs/train/ur5e_act_a6000",
    inference_interval_s=0.05,
    max_pos_delta=0.005,
    max_rot_delta=0.03,
)

while True:
    camera_1_rgb = read_camera_1_rgb()
    camera_2_rgb = read_camera_2_rgb()
    current_tcp_pose = read_ur_tcp_pose()
    gripper_state = np.array([read_gripper_opening(), read_gripper_current()])

    next_tcp_pose = predictor.predict_next_tcp_pose(
        camera_1_rgb,
        camera_2_rgb,
        current_tcp_pose,
        gripper_state,
    )
    send_ur_target_tcp_pose(next_tcp_pose)
```

脚本内部会把 ACT 输出的 6 维 TCP 增量通过 `scripts/ur_action_to_pose.py` 转成下一步 UR TCP Pose。默认旋转模式是 `relative-rotvec`，和数据转换脚本保持一致。

## 接真实 UR5e

真实机器人接口在 `scripts/test_act_ur5e.py` 的 `UR5eRuntimeInterface` 里，已经用中文注释标出需要实现的位置：

- `connect()`：连接 UR、夹爪、相机；
- `read_tcp_pose()`：读取 `[x, y, z, rx, ry, rz]`；
- `read_gripper()`：读取夹爪开度和电流；
- `read_camera()`：读取 RGB 图像；
- `send_target()`：发送目标 TCP 和夹爪开度；
- `stop()` / `close()`：保护性停止与资源释放。

接口实现后，先小步 dry-run，再执行：

```bash
python scripts/test_act_ur5e.py \
  --policy-path outputs/train/ur5e_act_a6000 \
  --backend ur5e \
  --max-steps 200 \
  --max-pos-delta 0.005 \
  --max-rot-delta 0.03
```

确认坐标系、相机顺序、夹爪单位和安全限幅都正确后，再加：

```bash
--execute
```

真实机械臂首次闭环时建议把 `--max-pos-delta` 和 `--max-rot-delta` 设得更保守，并准备物理急停。
