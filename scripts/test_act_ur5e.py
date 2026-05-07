#!/usr/bin/env python3
"""Run or dry-run a trained LeRobot ACT policy for the UR5e task.

This machine is not connected to the real UR5e, so the real robot interface is
left as a clearly marked adapter. The default backend replays observations from
the local LeRobot dataset and prints the target commands that would be sent.

Expected ACT action layout:
[delta_tcp_x, delta_tcp_y, delta_tcp_z, delta_tcp_rx, delta_tcp_ry, delta_tcp_rz,
gripper_target_opening]. The final gripper value is an absolute target opening.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

try:
    from scripts.ur_action_to_pose import apply_pose_delta
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from ur_action_to_pose import apply_pose_delta


STATE_NAMES = [
    "tcp_x",
    "tcp_y",
    "tcp_z",
    "tcp_rx",
    "tcp_ry",
    "tcp_rz",
    "gripper_opening",
    "gripper_current",
]
ACTION_NAMES = [
    "delta_tcp_x",
    "delta_tcp_y",
    "delta_tcp_z",
    "delta_tcp_rx",
    "delta_tcp_ry",
    "delta_tcp_rz",
    "gripper_target_opening",
]


@dataclass
class UR5eObservation:
    state: np.ndarray
    images: dict[str, Any]

    @property
    def tcp_pose(self) -> np.ndarray:
        return self.state[:6]

    @property
    def gripper_opening(self) -> float:
        return float(self.state[6])


class UR5eRuntimeInterface:
    """真实 UR5e/夹爪/相机的接口骨架。

    当前电脑没有连接实际机械臂，所以这里不直接写死任何硬件 SDK。接入现场设备时，
    建议在这个类里封装：
    - UR 机械臂：RTDE receive/control 或者 URScript servoL/moveL；
    - 夹爪：Robotiq/OnRobot/自研夹爪的开合接口；
    - 相机：RealSense、工业相机 SDK 或 OpenCV VideoCapture。
    """

    def __init__(self, camera_keys: list[str]):
        self.camera_keys = camera_keys

    def connect(self) -> None:
        # TODO: 在这里建立机械臂、夹爪、相机连接，例如创建 rtde_control/rtde_receive 对象。
        raise NotImplementedError("请在 UR5eRuntimeInterface.connect 中接入真实硬件。")

    def close(self) -> None:
        # TODO: 在这里关闭相机句柄，并停止/断开机械臂与夹爪连接。
        raise NotImplementedError("请在 UR5eRuntimeInterface.close 中接入真实硬件。")

    def read_tcp_pose(self) -> np.ndarray:
        # TODO: 返回 UR 当前 TCP 位姿 [x, y, z, rx, ry, rz]。
        # 注意 rx/ry/rz 必须是 UR 风格旋转向量，和采集数据保持一致。
        raise NotImplementedError

    def read_gripper(self) -> tuple[float, float]:
        # TODO: 返回 (gripper_opening, gripper_current)。
        # gripper_opening 需要和示教数据里的 GRIPPER_1 单位一致。
        raise NotImplementedError

    def read_camera(self, camera_key: str) -> np.ndarray:
        # TODO: 返回 RGB 图像，形状建议为 HWC uint8，例如 (720, 1280, 3)。
        # camera_key 会是 observation.images.camera_1 / camera_2 这样的 LeRobot 字段名。
        raise NotImplementedError

    def read_observation(self) -> UR5eObservation:
        tcp_pose = self.read_tcp_pose()
        gripper_opening, gripper_current = self.read_gripper()
        state = np.asarray([*tcp_pose, gripper_opening, gripper_current], dtype=np.float32)
        images = {key: self.read_camera(key) for key in self.camera_keys}
        return UR5eObservation(state=state, images=images)

    def send_target(self, target_tcp_pose: np.ndarray, target_gripper_opening: float) -> None:
        # TODO: 把目标 TCP 位姿和夹爪开度发给硬件。
        # 对 UR 来说，可以先用较小速度的 servoL/servoj/速度控制闭环验证；
        # 不建议一开始直接 moveL 大步执行网络输出。
        raise NotImplementedError

    def stop(self) -> None:
        # TODO: 触发保护性停止，例如 rtde_control.servoStop()/stopScript()。
        raise NotImplementedError


class MockDatasetRuntime:
    """从本地 LeRobot 数据集中读取观测，用来在无机械臂电脑上检查推理链路。"""

    def __init__(
        self,
        dataset_root: Path,
        repo_id: str,
        episode_index: int,
        camera_keys: list[str],
        execute: bool,
    ):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.dataset = LeRobotDataset(
            repo_id,
            root=str(dataset_root),
            episodes=[episode_index],
            video_backend="pyav",
        )
        self.camera_keys = camera_keys
        self.execute = execute
        self.cursor = 0

    def connect(self) -> None:
        print("Mock backend: 使用数据集帧作为观测，不会连接真实机械臂。")

    def close(self) -> None:
        print("Mock backend: 结束。")

    def read_observation(self) -> UR5eObservation:
        sample = self.dataset[self.cursor % len(self.dataset)]
        self.cursor += 1
        images = {key: sample[key] for key in self.camera_keys}
        return UR5eObservation(state=sample["observation.state"].cpu().numpy(), images=images)

    def send_target(self, target_tcp_pose: np.ndarray, target_gripper_opening: float) -> None:
        prefix = "EXECUTE" if self.execute else "DRY-RUN"
        pose = np.array2string(target_tcp_pose, precision=5, suppress_small=False)
        print(f"{prefix} target_tcp_pose={pose}, gripper={target_gripper_opening:.5f}")

    def stop(self) -> None:
        print("Mock backend: stop()")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ACT policy test runner for the UR5e cylinder-to-box task.")
    parser.add_argument(
        "--policy-path",
        type=Path,
        required=True,
        help="Run dir, checkpoint dir, or pretrained_model dir.",
    )
    parser.add_argument("--backend", choices=["mock_dataset", "ur5e"], default="mock_dataset")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/lerobot/ur5e_cylinder_to_box"))
    parser.add_argument("--repo-id", default="local/ur5e_cylinder_to_box")
    parser.add_argument("--mock-episode", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--execute", action="store_true", help="Actually send commands for non-mock backends.")
    parser.add_argument(
        "--rotation-delta-mode",
        choices=["relative-rotvec", "raw-rotvec"],
        default="relative-rotvec",
        help="Must match scripts/convert_ur5e_to_lerobot.py.",
    )
    parser.add_argument("--max-pos-delta", type=float, default=0.015, help="Safety clamp per xyz axis, meters.")
    parser.add_argument("--max-rot-delta", type=float, default=0.08, help="Safety clamp for rotvec norm, radians.")
    parser.add_argument(
        "--max-gripper-step",
        type=float,
        default=None,
        help=(
            "Optional safety clamp for the absolute gripper target. "
            "When set, target is limited to current_opening +/- this value."
        ),
    )
    parser.add_argument(
        "--max-gripper-delta",
        dest="max_gripper_step",
        type=float,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--gripper-min", type=float, default=0.0)
    parser.add_argument("--gripper-max", type=float, default=1.0)
    return parser.parse_args()


def resolve_pretrained_model_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    candidates = [
        path,
        path / "pretrained_model",
        path / "checkpoints" / "last" / "pretrained_model",
        path / "last" / "pretrained_model",
    ]
    for candidate in candidates:
        if (candidate / "config.json").is_file() and (candidate / "model.safetensors").is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"无法在 {path} 下找到 LeRobot pretrained_model。请传入训练目录或 checkpoints/last/pretrained_model。"
    )


def load_policy_and_processors(pretrained_dir: Path, device: str):
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy = ACTPolicy.from_pretrained(
        pretrained_dir,
        local_files_only=True,
        cli_overrides=[f"--device={device}", "--push_to_hub=false"],
    )
    policy.eval()
    policy.reset()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(pretrained_dir),
        preprocessor_overrides={"device_processor": {"device": policy.config.device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    return policy, preprocessor, postprocessor


def prepare_image(image: Any, expected_shape: tuple[int, int, int]) -> torch.Tensor:
    """Convert HWC uint8/PIL/tensor images into CHW float tensors in [0, 1]."""

    expected_c, expected_h, expected_w = expected_shape

    if isinstance(image, torch.Tensor):
        tensor = image.detach().cpu()
        if tensor.ndim != 3:
            raise ValueError(f"图像 tensor 必须是 3 维，得到 {tuple(tensor.shape)}")
        if tensor.shape[0] in (1, 3):
            chw = tensor.float()
        else:
            chw = tensor.permute(2, 0, 1).float()
        if chw.max() > 1.5:
            chw = chw / 255.0
    else:
        if isinstance(image, Image.Image):
            pil = image.convert("RGB")
        else:
            array = np.asarray(image)
            if array.ndim != 3 or array.shape[2] != 3:
                raise ValueError(f"图像数组必须是 HWC RGB，得到 {array.shape}")
            pil = Image.fromarray(array.astype(np.uint8), mode="RGB")

        if pil.size != (expected_w, expected_h):
            pil = pil.resize((expected_w, expected_h), Image.BILINEAR)
        array = np.asarray(pil, dtype=np.float32) / 255.0
        chw = torch.from_numpy(array).permute(2, 0, 1).contiguous()

    if tuple(chw.shape) != (expected_c, expected_h, expected_w):
        chw = torch.nn.functional.interpolate(
            chw.unsqueeze(0),
            size=(expected_h, expected_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return chw.to(dtype=torch.float32)


def build_policy_batch(observation: UR5eObservation, policy: Any) -> dict[str, torch.Tensor]:
    batch: dict[str, torch.Tensor] = {
        "observation.state": torch.as_tensor(observation.state, dtype=torch.float32)
    }
    for key, feature in policy.config.image_features.items():
        if key not in observation.images:
            raise KeyError(f"观测缺少相机 {key}，当前只有 {list(observation.images)}")
        batch[key] = prepare_image(observation.images[key], feature.shape)
    return batch


def clamp_action(
    action: np.ndarray,
    max_pos_delta: float,
    max_rot_delta: float,
    gripper_min: float,
    gripper_max: float,
    current_gripper_opening: float,
    max_gripper_step: float | None,
) -> np.ndarray:
    safe = np.asarray(action, dtype=np.float64).copy()
    if safe.shape[0] < len(ACTION_NAMES):
        raise ValueError(f"ACT action must have at least {len(ACTION_NAMES)} values, got {safe.shape[0]}")

    safe[:3] = np.clip(safe[:3], -max_pos_delta, max_pos_delta)

    rot_norm = float(np.linalg.norm(safe[3:6]))
    if rot_norm > max_rot_delta and rot_norm > 1e-12:
        safe[3:6] *= max_rot_delta / rot_norm

    safe[6] = float(np.clip(safe[6], gripper_min, gripper_max))
    if max_gripper_step is not None:
        safe[6] = float(
            np.clip(
                safe[6],
                current_gripper_opening - max_gripper_step,
                current_gripper_opening + max_gripper_step,
            )
        )
        safe[6] = float(np.clip(safe[6], gripper_min, gripper_max))
    return safe


def make_runtime(args: argparse.Namespace, camera_keys: list[str]):
    if args.backend == "mock_dataset":
        return MockDatasetRuntime(
            dataset_root=args.dataset_root,
            repo_id=args.repo_id,
            episode_index=args.mock_episode,
            camera_keys=camera_keys,
            execute=args.execute,
        )

    runtime = UR5eRuntimeInterface(camera_keys=camera_keys)
    if not args.execute:
        print("提示：当前未加 --execute。真实硬件 backend 仍需你实现接口后再谨慎执行。")
    return runtime


def main() -> int:
    args = parse_args()
    if args.gripper_min >= args.gripper_max:
        raise ValueError("--gripper-min must be smaller than --gripper-max")
    if args.max_gripper_step is not None and args.max_gripper_step < 0:
        raise ValueError("--max-gripper-step must be >= 0")

    pretrained_dir = resolve_pretrained_model_dir(args.policy_path)
    policy, preprocessor, postprocessor = load_policy_and_processors(pretrained_dir, args.device)
    camera_keys = list(policy.config.image_features.keys())

    runtime = make_runtime(args, camera_keys)
    runtime.connect()

    dt = 1.0 / args.fps if args.fps > 0 else 0.0
    try:
        for step in range(args.max_steps):
            start = time.perf_counter()
            observation = runtime.read_observation()
            batch = preprocessor(build_policy_batch(observation, policy))

            with torch.inference_mode():
                normalized_action = policy.select_action(batch)
                action = postprocessor(normalized_action)

            action_np = np.asarray(action.squeeze(0), dtype=np.float64)
            action_np = clamp_action(
                action_np,
                max_pos_delta=args.max_pos_delta,
                max_rot_delta=args.max_rot_delta,
                gripper_min=args.gripper_min,
                gripper_max=args.gripper_max,
                current_gripper_opening=observation.gripper_opening,
                max_gripper_step=args.max_gripper_step,
            )

            target_tcp_pose = np.asarray(
                apply_pose_delta(
                    observation.tcp_pose,
                    action_np[:6],
                    rotation_delta_mode=args.rotation_delta_mode,
                ),
                dtype=np.float64,
            )
            target_gripper = float(action_np[6])

            print(
                f"step={step:04d} action={np.array2string(action_np, precision=5, suppress_small=False)}"
            )
            runtime.send_target(target_tcp_pose, target_gripper)

            elapsed = time.perf_counter() - start
            if dt > elapsed:
                time.sleep(dt - elapsed)
    except KeyboardInterrupt:
        print("收到 Ctrl+C，准备停止。")
    finally:
        runtime.stop()
        runtime.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
