#!/usr/bin/env python3
"""Realtime ACT inference interface for a real UR5e controller.

This module intentionally does not contain a mock backend or robot SDK glue.
Create one ``UR5eACTRealtimeInference`` instance in your UR5e control program,
then call ``predict_next_tcp_pose`` with the latest two RGB camera frames and
robot state. The method returns a 7D command:
``[x, y, z, rx, ry, rz, gripper_opening]``.

The ACT action layout expected by this wrapper is:
``[delta_tcp_x, delta_tcp_y, delta_tcp_z, delta_tcp_rx, delta_tcp_ry,
delta_tcp_rz, gripper_target_opening]``. The gripper value is an absolute
target opening, not a delta.
"""

from __future__ import annotations

import sys
import time
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


class UR5eACTRealtimeInference:
    """ACT policy wrapper with one public inference method.

    Example:
        predictor = UR5eACTRealtimeInference(
            policy_path="outputs/train/ur5e_act_a6000",
            inference_interval_s=0.05,
        )
        next_command = predictor.predict_next_tcp_pose(
            camera_1_rgb,
            camera_2_rgb,
            current_tcp_pose,
            np.array([gripper_opening, gripper_current], dtype=np.float32),
        )
    """

    def __init__(
        self,
        policy_path: str | Path,
        *,
        device: str = "cuda",
        inference_interval_s: float = 0.05,
        rotation_delta_mode: str = "relative-rotvec",
        max_pos_delta: float | None = 0.015,
        max_rot_delta: float | None = 0.08,
        max_gripper_step: float | None = None,
        max_gripper_delta: float | None = None,
        gripper_min: float = 0.0,
        gripper_max: float = 1.0,
        camera_keys: tuple[str, str] = (
            "observation.images.camera_1",
            "observation.images.camera_2",
        ),
    ) -> None:
        """Load a trained ACT policy.

        Args:
            policy_path: Training run directory, checkpoint directory, or
                ``pretrained_model`` directory.
            device: Torch device, usually ``cuda`` in the LeRobot conda env.
            inference_interval_s: Minimum time between two policy calls. The
                method sleeps when called too quickly. Use 0 to disable.
            rotation_delta_mode: Must match the data conversion script. The
                default matches ``scripts/convert_ur5e_to_lerobot.py``.
            max_pos_delta: Optional safety clamp for dx/dy/dz in meters.
            max_rot_delta: Optional safety clamp for rotvec delta norm in radians.
            max_gripper_step: Optional safety clamp limiting the absolute
                gripper target to current_opening +/- this value.
            max_gripper_delta: Deprecated alias for max_gripper_step.
            gripper_min: Lower bound for the absolute gripper target.
            gripper_max: Upper bound for the absolute gripper target.
            camera_keys: LeRobot camera feature keys corresponding to the two
                ndarray image inputs.
        """

        if inference_interval_s < 0:
            raise ValueError("inference_interval_s must be >= 0")
        if rotation_delta_mode not in {"relative-rotvec", "raw-rotvec"}:
            raise ValueError("rotation_delta_mode must be 'relative-rotvec' or 'raw-rotvec'")
        if gripper_min >= gripper_max:
            raise ValueError("gripper_min must be smaller than gripper_max")
        if max_gripper_step is not None and max_gripper_step < 0:
            raise ValueError("max_gripper_step must be >= 0")
        if max_gripper_delta is not None:
            if max_gripper_delta < 0:
                raise ValueError("max_gripper_delta must be >= 0")
            if max_gripper_step is not None:
                raise ValueError("Use only one of max_gripper_step or max_gripper_delta")
            max_gripper_step = max_gripper_delta

        self.inference_interval_s = float(inference_interval_s)
        self.rotation_delta_mode = rotation_delta_mode
        self.max_pos_delta = max_pos_delta
        self.max_rot_delta = max_rot_delta
        self.max_gripper_step = max_gripper_step
        self.gripper_min = gripper_min
        self.gripper_max = gripper_max
        self.camera_keys = camera_keys
        self._last_inference_time: float | None = None

        pretrained_dir = _resolve_pretrained_model_dir(Path(policy_path))
        self.policy, self.preprocessor, self.postprocessor = _load_policy_and_processors(
            pretrained_dir, device
        )
        self._validate_camera_keys()

    def predict_next_tcp_pose(
        self,
        camera_1_image: np.ndarray,
        camera_2_image: np.ndarray,
        current_tcp_pose: np.ndarray,
        gripper_opening_current: np.ndarray,
    ) -> np.ndarray:
        """Predict the next UR TCP pose and gripper opening.

        Args:
            camera_1_image: RGB image ndarray from camera 1, HWC uint8 or float.
            camera_2_image: RGB image ndarray from camera 2, HWC uint8 or float.
            current_tcp_pose: Current UR TCP pose ``[x, y, z, rx, ry, rz]``.
            gripper_opening_current: ``[gripper_opening, gripper_current]``.

        Returns:
            Next command as ``np.ndarray`` with shape ``(7,)``:
            ``[x, y, z, rx, ry, rz, gripper_opening]``.
            The gripper opening is the policy's absolute target opening,
            clipped to ``[gripper_min, gripper_max]``.
        """

        self._wait_for_interval()

        current_tcp_pose = _as_vector(current_tcp_pose, 6, "current_tcp_pose")
        gripper_opening_current = _as_vector(
            gripper_opening_current, 2, "gripper_opening_current"
        )

        state = np.asarray(
            [*current_tcp_pose.tolist(), *gripper_opening_current.tolist()],
            dtype=np.float32,
        )
        batch = self._build_policy_batch(camera_1_image, camera_2_image, state)
        batch = self.preprocessor(batch)

        with torch.inference_mode():
            normalized_action = self.policy.select_action(batch)
            action = self.postprocessor(normalized_action)

        action_np = np.asarray(action.squeeze(0), dtype=np.float64)
        action_np = self._clamp_action(action_np, gripper_opening_current[0])
        next_tcp_pose = apply_pose_delta(
            current_tcp_pose,
            action_np[:6],
            rotation_delta_mode=self.rotation_delta_mode,
        )
        next_gripper_opening = action_np[6]
        self._last_inference_time = time.perf_counter()
        return np.asarray([*next_tcp_pose, next_gripper_opening], dtype=np.float64)

    def _wait_for_interval(self) -> None:
        if self.inference_interval_s <= 0 or self._last_inference_time is None:
            return
        elapsed = time.perf_counter() - self._last_inference_time
        remaining = self.inference_interval_s - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _validate_camera_keys(self) -> None:
        policy_camera_keys = set(self.policy.config.image_features.keys())
        missing = [key for key in self.camera_keys if key not in policy_camera_keys]
        if missing:
            raise KeyError(
                f"Policy is missing requested camera keys {missing}. "
                f"Available keys: {sorted(policy_camera_keys)}"
            )

    def _build_policy_batch(
        self,
        camera_1_image: np.ndarray,
        camera_2_image: np.ndarray,
        state: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        images = {
            self.camera_keys[0]: camera_1_image,
            self.camera_keys[1]: camera_2_image,
        }
        batch: dict[str, torch.Tensor] = {
            "observation.state": torch.as_tensor(state, dtype=torch.float32)
        }
        for key in self.camera_keys:
            feature = self.policy.config.image_features[key]
            batch[key] = _prepare_image(images[key], feature.shape)
        return batch

    def _clamp_action(self, action: np.ndarray, current_gripper_opening: float) -> np.ndarray:
        safe = np.asarray(action, dtype=np.float64).copy()

        if safe.shape[0] < 7:
            raise ValueError(f"ACT action must have at least 7 values, got {safe.shape[0]}")

        if self.max_pos_delta is not None:
            safe[:3] = np.clip(safe[:3], -self.max_pos_delta, self.max_pos_delta)

        if self.max_rot_delta is not None:
            rot_norm = float(np.linalg.norm(safe[3:6]))
            if rot_norm > self.max_rot_delta and rot_norm > 1e-12:
                safe[3:6] *= self.max_rot_delta / rot_norm

        safe[6] = np.clip(safe[6], self.gripper_min, self.gripper_max)
        if self.max_gripper_step is not None:
            safe[6] = np.clip(
                safe[6],
                current_gripper_opening - self.max_gripper_step,
                current_gripper_opening + self.max_gripper_step,
            )
            safe[6] = np.clip(safe[6], self.gripper_min, self.gripper_max)

        return safe


def _resolve_pretrained_model_dir(path: Path) -> Path:
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
        f"Cannot find a LeRobot pretrained_model under {path}. "
        "Pass the run dir, checkpoints/last/pretrained_model, or pretrained_model dir."
    )


def _load_policy_and_processors(pretrained_dir: Path, device: str) -> tuple[Any, Any, Any]:
    try:
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import LeRobot. Activate the environment first: conda activate LeRobot"
        ) from exc

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


def _as_vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape[0] != size:
        raise ValueError(f"{name} must have shape ({size},), got {tuple(vector.shape)}")
    return vector


def _prepare_image(image: np.ndarray, expected_shape: tuple[int, int, int]) -> torch.Tensor:
    expected_c, expected_h, expected_w = expected_shape
    array = np.asarray(image)

    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"camera image must be HWC RGB with 3 channels, got {array.shape}")

    if np.issubdtype(array.dtype, np.floating):
        if array.max(initial=0.0) <= 1.5:
            array = np.clip(array, 0.0, 1.0) * 255.0
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    image_pil = Image.fromarray(array, mode="RGB")
    if image_pil.size != (expected_w, expected_h):
        image_pil = image_pil.resize((expected_w, expected_h), Image.BILINEAR)

    image_array = np.asarray(image_pil, dtype=np.float32) / 255.0
    image_chw = torch.from_numpy(image_array).permute(2, 0, 1).contiguous()

    if tuple(image_chw.shape) != (expected_c, expected_h, expected_w):
        image_chw = torch.nn.functional.interpolate(
            image_chw.unsqueeze(0),
            size=(expected_h, expected_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    return image_chw.to(dtype=torch.float32)
