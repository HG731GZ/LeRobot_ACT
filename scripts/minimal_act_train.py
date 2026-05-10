#!/usr/bin/env python3
"""教学版 ACT 训练脚本：把 LeRobot 官方训练器封装起来的关键步骤摊开。

这个脚本的目标不是替代 `lerobot-train`，而是帮助你直接看清楚 ACT 在
LeRobot 里的完整训练链路：

    LeRobotDataset
        -> DataLoader
        -> 图像缩放/类型检查
        -> LeRobot preprocessor（设备搬运 + 归一化）
        -> ACTPolicy.forward(batch)
        -> L1 action reconstruction loss + kl_weight * VAE KL loss
        -> backward
        -> gradient clipping
        -> AdamW optimizer.step
        -> checkpoint

使用方式：
    直接在 VSCode / PyCharm 里选择 conda 环境 LeRobot，然后运行本文件即可。
    不需要命令行参数；需要改训练步数、batch size、数据路径时，改下面的
    ScriptConfig 默认值即可。
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Any

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader
except ImportError as exc:
    raise SystemExit(
        "无法导入 PyTorch / numpy。请在 VSCode 或 PyCharm 中选择 conda 环境：LeRobot。"
    ) from exc


# 项目根目录。无论你从项目根目录运行，还是从 scripts 目录运行，都能稳定定位数据。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ScriptConfig:
    """脚本配置。

    这里故意不用 argparse，因为你希望能在 IDE 里直接点运行按钮。
    如果想改设置，直接修改这个 dataclass 的默认值即可。
    """

    # LeRobotDataset 需要一个 repo_id。对于本地数据集，这个值主要作为名字使用。
    repo_id: str = "local/ur5e_cylinder_to_box"

    # 兼容你描述的 data/data 路径和当前仓库实际存在的 data 路径；脚本会选择第一个存在的。
    dataset_root_candidates: list[Path] = field(
        default_factory=lambda: [
            PROJECT_ROOT / "data/data/lerobot/ur5e_cylinder_to_box",
            PROJECT_ROOT / "data/lerobot/ur5e_cylinder_to_box",
        ]
    )

    # 输出根目录。默认每次运行会创建一个带时间戳的子目录，避免覆盖上一次实验。
    output_base_dir: Path = PROJECT_ROOT / "outputs/minimal_act_train/ur5e_act_teaching"
    create_timestamped_run_dir: bool = True
    run_name: str = "run"

    # 如果要从本脚本保存的 checkpoint 继续训练，把路径填在这里。
    resume_checkpoint: Path | None = None

    # 训练步数。先用较小值跑通和学习流程；真正训练可改成 30000 或更高。
    train_steps: int = 2000
    batch_size: int = 1
    num_workers: int = 2
    seed: int = 42

    # 设备设置。auto 会优先使用 cuda，其次 mps，最后 cpu。
    device: str = "auto"
    # 为了教学脚本默认稳定，先关闭 AMP。GTX 1660 Ti 上 ACT + VAE 用 FP16 可能出现 NaN。
    # 如果你确认当前 GPU 上混合精度稳定，并且想省显存，可以改成 True。
    use_amp: bool = False

    # 训练日志和 checkpoint 频率。
    log_freq: int = 20
    save_freq: int = 500
    grad_clip_norm: float = 10.0

    # 视频后端。当前环境中 pyav 对本地 avi 数据更稳。
    video_backend: str = "pyav"

    # 为了让普通 GPU 和 IDE 运行更轻松，默认把 720p 图像缩小后再送入 ACT。
    # 如果你想完全使用原始分辨率，把它改成 None。
    image_resize_hw: tuple[int, int] | None = (240, 320)

    # 是否使用 ImageNet mean/std 归一化图像；这和 LeRobot 官方训练入口的默认行为一致。
    use_imagenet_stats: bool = True

    # ACT 的 action chunk 设置。
    # chunk_size 表示一次 forward 预测未来多少帧动作，训练标签形状为 [B, chunk_size, action_dim]。
    # n_action_steps 主要用于推理时一次执行多少个动作，训练时仍然学习完整 chunk。
    chunk_size: int = 32
    n_action_steps: int = 16

    # ACT Transformer 容量。这个配置偏轻量，适合先学习和本地试跑。
    dim_model: int = 256
    n_heads: int = 4
    dim_feedforward: int = 1024
    n_encoder_layers: int = 2
    n_decoder_layers: int = 1

    # ACT 的 VAE encoder 设置。use_vae=True 时，训练时会把真实 action chunk 编成 latent z。
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 2
    kl_weight: float = 10.0
    dropout: float = 0.1

    # 视觉 backbone。当前机器已有 ResNet18 权重缓存；如果你的机器不能下载/没有缓存，可改成 None。
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"

    # 优化器参数。ACTPolicy.get_optim_params 会把 backbone 单独放到一个 param group。
    lr: float = 1e-5
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4

    # 启动后先跑一个不反传的 forward，并打印关键模块输入输出形状。
    print_debug_forward: bool = True

    # 最后额外保存一份 LeRobot/HuggingFace 风格的 policy 目录，方便后续推理脚本加载。
    save_pretrained_at_end: bool = True


def import_lerobot_symbols() -> dict[str, Any]:
    """集中导入 LeRobot 符号，便于给出中文报错和兼容路径变化。"""
    try:
        from lerobot import __version__ as lerobot_version
    except Exception:
        lerobot_version = "unknown"

    try:
        from lerobot.datasets.factory import IMAGENET_STATS
        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
    except ImportError as exc:
        raise SystemExit(
            "无法导入 LeRobot ACT 相关模块。请确认 IDE 使用的是 conda 环境 LeRobot。"
        ) from exc

    return {
        "lerobot_version": lerobot_version,
        "IMAGENET_STATS": IMAGENET_STATS,
        "LeRobotDataset": LeRobotDataset,
        "LeRobotDatasetMetadata": LeRobotDatasetMetadata,
        "ACTConfig": ACTConfig,
        "ACTPolicy": ACTPolicy,
        "make_pre_post_processors": make_pre_post_processors,
        "FeatureType": FeatureType,
        "PolicyFeature": PolicyFeature,
        "ACTION": ACTION,
        "OBS_IMAGES": OBS_IMAGES,
        "OBS_STATE": OBS_STATE,
    }


def resolve_dataset_root(cfg: ScriptConfig) -> Path:
    """从候选路径中找到实际存在的 LeRobot 数据集目录。"""
    for root in cfg.dataset_root_candidates:
        root = root.expanduser().resolve()
        if (root / "meta/info.json").is_file():
            return root

    candidates = "\n".join(f"  - {p.expanduser().resolve()}" for p in cfg.dataset_root_candidates)
    raise FileNotFoundError(
        "没有找到 LeRobot 数据集 meta/info.json。请在 ScriptConfig.dataset_root_candidates 中修改路径。\n"
        f"已尝试：\n{candidates}"
    )


def make_output_dir(cfg: ScriptConfig) -> Path:
    """创建本次运行的输出目录。"""
    if cfg.resume_checkpoint is not None:
        ckpt = cfg.resume_checkpoint.expanduser().resolve()
        if ckpt.parent.name == "checkpoints":
            return ckpt.parent.parent

    base = cfg.output_base_dir.expanduser().resolve()
    if not cfg.create_timestamped_run_dir:
        base.mkdir(parents=True, exist_ok=True)
        return base

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base / f"{cfg.run_name}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def set_seed(seed: int) -> None:
    """固定随机种子，让同一配置下的学习曲线更容易复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_name: str) -> torch.device:
    """根据配置选择训练设备。"""
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_action_delta_timestamps(chunk_size: int, fps: int, action_key: str) -> dict[str, list[float]]:
    """构造 ACT 需要的未来动作时间戳。

    ACTConfig.action_delta_indices 等价于 list(range(chunk_size))。
    LeRobotDataset 看到这些时间戳后，会把单帧 action 扩展为未来 action chunk：
        action: [action_dim] -> [chunk_size, action_dim]
        action_is_pad: [chunk_size]
    """
    return {action_key: [index / fps for index in range(chunk_size)]}


def load_dataset(cfg: ScriptConfig, symbols: dict[str, Any], dataset_root: Path):
    """加载 LeRobotDataset，并让它返回 ACT 所需的未来动作 chunk。"""
    LeRobotDataset = symbols["LeRobotDataset"]
    LeRobotDatasetMetadata = symbols["LeRobotDatasetMetadata"]
    ACTION = symbols["ACTION"]

    meta = LeRobotDatasetMetadata(cfg.repo_id, root=dataset_root)
    delta_timestamps = make_action_delta_timestamps(cfg.chunk_size, meta.fps, ACTION)

    dataset = LeRobotDataset(
        cfg.repo_id,
        root=dataset_root,
        delta_timestamps=delta_timestamps,
        video_backend=cfg.video_backend,
    )

    if cfg.use_imagenet_stats:
        apply_imagenet_stats(dataset, symbols["IMAGENET_STATS"])

    return dataset, delta_timestamps


def apply_imagenet_stats(dataset: Any, imagenet_stats: dict[str, list[float]]) -> None:
    """把图像归一化统计量改成 ImageNet mean/std，与 LeRobot 官方训练入口保持一致。"""
    for camera_key in dataset.meta.camera_keys:
        for stats_name, stats in imagenet_stats.items():
            dataset.meta.stats[camera_key][stats_name] = torch.tensor(stats, dtype=torch.float32)


def feature_shape_to_channel_first(shape: Iterable[int], names: Any) -> tuple[int, int, int]:
    """把 LeRobot 元数据里的图像 shape 转成 PyTorch 的 C,H,W。"""
    shape = tuple(int(x) for x in shape)
    if len(shape) != 3:
        raise ValueError(f"图像特征必须是 3 维，实际 shape={shape}")

    # 当前数据集的图像 shape 是 [H, W, C]，names 是 ["height", "width", "channel"]。
    if names and len(names) == 3 and str(names[2]).lower() in {"channel", "channels"}:
        return shape[2], shape[0], shape[1]
    return shape  # 已经是 C,H,W 时直接返回。


def make_policy_features(dataset: Any, cfg: ScriptConfig, symbols: dict[str, Any]) -> tuple[dict, dict]:
    """显式地把 dataset.meta.features 转成 ACTConfig 需要的 input/output features。

    这里没有使用 LeRobot 的 make_policy 自动推断，是为了让“模型到底看哪些输入、
    预测哪些输出”在脚本中一眼可见。
    """
    FeatureType = symbols["FeatureType"]
    PolicyFeature = symbols["PolicyFeature"]
    ACTION = symbols["ACTION"]
    OBS_STATE = symbols["OBS_STATE"]

    input_features = {}
    output_features = {}

    for key, feature in dataset.meta.features.items():
        dtype = feature.get("dtype")
        shape = feature.get("shape")
        names = feature.get("names")

        if dtype in {"image", "video"}:
            c, h, w = feature_shape_to_channel_first(shape, names)
            if cfg.image_resize_hw is not None:
                h, w = cfg.image_resize_hw
            input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(c, h, w))
        elif key == OBS_STATE:
            input_features[key] = PolicyFeature(type=FeatureType.STATE, shape=tuple(shape))
        elif key == ACTION:
            output_features[key] = PolicyFeature(type=FeatureType.ACTION, shape=tuple(shape))

    if OBS_STATE not in input_features:
        raise KeyError(f"数据集中没有找到 {OBS_STATE}")
    if ACTION not in output_features:
        raise KeyError(f"数据集中没有找到 {ACTION}")

    return input_features, output_features


def build_policy(cfg: ScriptConfig, dataset: Any, symbols: dict[str, Any], device: torch.device):
    """构造 ACTPolicy，并显式传入网络结构相关配置。"""
    ACTConfig = symbols["ACTConfig"]
    ACTPolicy = symbols["ACTPolicy"]

    input_features, output_features = make_policy_features(dataset, cfg, symbols)
    policy_cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        device=str(device),
        use_amp=cfg.use_amp and device.type == "cuda",
        push_to_hub=False,
        chunk_size=cfg.chunk_size,
        n_action_steps=cfg.n_action_steps,
        vision_backbone=cfg.vision_backbone,
        pretrained_backbone_weights=cfg.pretrained_backbone_weights,
        dim_model=cfg.dim_model,
        n_heads=cfg.n_heads,
        dim_feedforward=cfg.dim_feedforward,
        n_encoder_layers=cfg.n_encoder_layers,
        n_decoder_layers=cfg.n_decoder_layers,
        use_vae=cfg.use_vae,
        latent_dim=cfg.latent_dim,
        n_vae_encoder_layers=cfg.n_vae_encoder_layers,
        dropout=cfg.dropout,
        kl_weight=cfg.kl_weight,
        optimizer_lr=cfg.lr,
        optimizer_lr_backbone=cfg.lr_backbone,
        optimizer_weight_decay=cfg.weight_decay,
    )

    policy = ACTPolicy(policy_cfg)
    policy.to(device)
    return policy


def build_preprocessor(policy: nn.Module, dataset: Any, symbols: dict[str, Any]):
    """构造 LeRobot 的 ACT pre/post processor。

    preprocessor 主要做三件事：
      1. 单样本时补 batch 维度；DataLoader 已有 batch 维度时不会重复补。
      2. 把 tensor 搬到 policy.config.device。
      3. 根据 dataset.meta.stats 对 state、action、image 做归一化。
    """
    make_pre_post_processors = symbols["make_pre_post_processors"]
    return make_pre_post_processors(policy_cfg=policy.config, dataset_stats=dataset.meta.stats)


def build_optimizer(cfg: ScriptConfig, policy: nn.Module) -> torch.optim.Optimizer:
    """构造 AdamW 优化器。

    ACTPolicy.get_optim_params 会返回两个参数组：
      - 非视觉 backbone 参数
      - 视觉 backbone 参数，使用 optimizer_lr_backbone
    这里显式调用它，方便你看到 backbone 可以单独设学习率。
    """
    params = policy.get_optim_params() if hasattr(policy, "get_optim_params") else policy.parameters()
    return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)


def cycle(loader: DataLoader):
    """无限循环 DataLoader；训练步数由 train_steps 控制，不由 epoch 控制。"""
    while True:
        for batch in loader:
            yield batch


def convert_uint8_images_to_float(batch: dict[str, Any], camera_keys: list[str]) -> dict[str, Any]:
    """如果图像是 uint8，则转成 float32 并缩放到 [0, 1]。

    当前数据集的视频解码已经返回 float32，但这个函数保留在训练链路里，
    方便你看到官方训练里常见的图像预处理步骤。
    """
    batch = dict(batch)
    for key in camera_keys:
        value = batch.get(key)
        if torch.is_tensor(value) and value.dtype == torch.uint8:
            batch[key] = value.float() / 255.0
    return batch


def resize_images(batch: dict[str, Any], camera_keys: list[str], image_resize_hw: tuple[int, int] | None):
    """在送入 preprocessor 前缩放图像，降低 ResNet 和 Transformer 的显存占用。"""
    if image_resize_hw is None:
        return batch

    batch = dict(batch)
    target_h, target_w = image_resize_hw
    for key in camera_keys:
        image = batch.get(key)
        if not torch.is_tensor(image):
            continue
        if image.dim() != 4:
            raise ValueError(f"{key} 期望形状 [B,C,H,W]，实际 shape={tuple(image.shape)}")
        if image.shape[-2:] == (target_h, target_w):
            continue
        batch[key] = F.interpolate(image, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return batch


def prepare_batch_for_policy(batch: dict[str, Any], dataset: Any, cfg: ScriptConfig, preprocessor: Any):
    """把 DataLoader 的原始 batch 处理成 ACTPolicy.forward 可以直接接收的 batch。"""
    batch = convert_uint8_images_to_float(batch, dataset.meta.camera_keys)
    batch = resize_images(batch, dataset.meta.camera_keys, cfg.image_resize_hw)
    batch = preprocessor(batch)
    return batch


def tensor_summary(tensor: torch.Tensor) -> str:
    """把 tensor 的形状、类型和设备压成一行字符串。"""
    return f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}"


def object_shape_summary(obj: Any) -> str:
    """用于 forward hook：兼容 tensor、dict、list、tuple 的形状摘要。"""
    if torch.is_tensor(obj):
        return tensor_summary(obj)
    if isinstance(obj, dict):
        parts = [f"{key}: {object_shape_summary(value)}" for key, value in obj.items()]
        return "{" + ", ".join(parts) + "}"
    if isinstance(obj, (list, tuple)):
        parts = [object_shape_summary(value) for value in obj[:3]]
        suffix = "" if len(obj) <= 3 else f", ... len={len(obj)}"
        return "[" + ", ".join(parts) + suffix + "]"
    return type(obj).__name__


class TextRecorder:
    """同时打印到终端并记录到文本文件的轻量工具。"""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def write(self, line: str = "") -> None:
        print(line)
        self.lines.append(line)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def print_nested_shapes(title: str, batch: dict[str, Any], recorder: TextRecorder) -> None:
    """打印 batch 内所有 key 的形状，便于理解 LeRobot 的数据结构。"""
    recorder.write(f"\n========== {title} ==========")

    def visit(prefix: str, value: Any) -> None:
        if torch.is_tensor(value):
            recorder.write(f"{prefix}: {tensor_summary(value)}")
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(f"{prefix}.{key}" if prefix else str(key), child)
        elif isinstance(value, (list, tuple)):
            recorder.write(f"{prefix}: {type(value).__name__}(len={len(value)})")
            for index, child in enumerate(value[:3]):
                visit(f"{prefix}[{index}]", child)
        else:
            recorder.write(f"{prefix}: {type(value).__name__} = {value!r}")

    for key, value in batch.items():
        visit(str(key), value)


def count_parameters(module: nn.Module, only_trainable: bool = False) -> int:
    """统计参数量。"""
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not only_trainable))


def count_parameters_by_prefix(policy: nn.Module) -> dict[str, int]:
    """按 ACT 的关键模块粗略统计可训练参数量。"""
    groups = {
        "vision_backbone": "model.backbone",
        "vae_encoder": "model.vae_encoder",
        "main_encoder": "model.encoder",
        "main_decoder": "model.decoder",
        "action_head": "model.action_head",
    }
    result = {name: 0 for name in groups}
    result["other"] = 0

    for param_name, param in policy.named_parameters():
        if not param.requires_grad:
            continue
        matched = False
        for group_name, prefix in groups.items():
            if param_name.startswith(prefix):
                result[group_name] += param.numel()
                matched = True
                break
        if not matched:
            result["other"] += param.numel()
    return result


def print_environment_summary(
    cfg: ScriptConfig,
    symbols: dict[str, Any],
    dataset: Any,
    device: torch.device,
    delta_timestamps: dict[str, list[float]],
    recorder: TextRecorder,
) -> None:
    """打印运行环境、数据集和 action chunk 信息。"""
    recorder.write("========== 运行环境 ==========")
    recorder.write(f"Python: {sys.version.split()[0]}")
    recorder.write(f"PyTorch: {torch.__version__}")
    recorder.write(f"LeRobot: {symbols['lerobot_version']}")
    recorder.write(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        recorder.write(f"CUDA device: {torch.cuda.get_device_name(0)}")
    recorder.write(f"Selected device: {device}")

    recorder.write("\n========== 数据集 ==========")
    recorder.write(f"repo_id: {cfg.repo_id}")
    recorder.write(f"root: {dataset.root}")
    recorder.write(f"num_frames: {dataset.num_frames}")
    recorder.write(f"num_episodes: {dataset.num_episodes}")
    recorder.write(f"fps: {dataset.fps}")
    recorder.write(f"camera_keys: {dataset.meta.camera_keys}")
    recorder.write(f"feature_keys: {list(dataset.meta.features.keys())}")
    recorder.write(f"stats_keys: {list(dataset.meta.stats.keys())}")

    action_key = symbols["ACTION"]
    recorder.write("\n========== ACT action chunk ==========")
    recorder.write(f"chunk_size: {cfg.chunk_size}")
    recorder.write(f"n_action_steps: {cfg.n_action_steps}")
    recorder.write(f"action_delta_timestamps[{action_key}]: {delta_timestamps[action_key]}")
    recorder.write("含义：dataset[i]['action'] 不再是单帧动作，而是从当前帧开始的未来动作序列。")


def print_policy_summary(policy: nn.Module, dataset: Any, cfg: ScriptConfig, symbols: dict[str, Any]) -> str:
    """生成一份 ACT 网络结构说明，并返回完整字符串用于保存。"""
    ACTION = symbols["ACTION"]
    OBS_STATE = symbols["OBS_STATE"]

    state_dim = policy.config.input_features[OBS_STATE].shape[0]
    action_dim = policy.config.output_features[ACTION].shape[0]
    image_features = policy.config.image_features
    trainable = count_parameters(policy, only_trainable=True)
    total = count_parameters(policy, only_trainable=False)
    by_prefix = count_parameters_by_prefix(policy)

    lines = [
        "========== ACT 网络结构速览 ==========",
        f"状态输入: {OBS_STATE}, dim={state_dim}",
        f"图像输入: {list(image_features.keys())}",
        f"动作输出: {ACTION}, dim={action_dim}",
        f"chunk_size={cfg.chunk_size}, 所以 action_head 输出 [B, {cfg.chunk_size}, {action_dim}]",
        f"VAE: use_vae={cfg.use_vae}, latent_dim={cfg.latent_dim}, kl_weight={cfg.kl_weight}",
        f"Transformer: dim_model={cfg.dim_model}, n_heads={cfg.n_heads}, "
        f"encoder_layers={cfg.n_encoder_layers}, decoder_layers={cfg.n_decoder_layers}, "
        f"vae_encoder_layers={cfg.n_vae_encoder_layers}",
        "",
        "训练时的数据流：",
        "1. VAE encoder 读取 [CLS token, 当前机器人状态 token, 真实 action chunk tokens]，输出 μ 和 log(σ²)。",
        "2. 用重参数化 z = μ + σ * ε 采样 latent；推理时没有真实 action，所以 latent 置零。",
        "3. 主 encoder 读取 [latent token, 状态 token, 所有相机的视觉 tokens] 做多模态融合。",
        "4. decoder 用 chunk_size 个可学习位置查询并行生成动作 token，不是自回归逐步生成。",
        "5. action_head 把每个动作 token 回归为 7 维 UR5e 动作。",
        "",
        "可训练参数量：",
        f"total={total:,}, trainable={trainable:,}",
        pformat(by_prefix, sort_dicts=False),
        "",
        "========== PyTorch module tree ==========",
        str(policy),
    ]
    return "\n".join(lines)


def print_optimizer_summary(optimizer: torch.optim.Optimizer, recorder: TextRecorder) -> None:
    """打印优化器参数组，观察 backbone 是否单独设了学习率。"""
    recorder.write("\n========== Optimizer param groups ==========")
    for index, group in enumerate(optimizer.param_groups):
        param_count = sum(p.numel() for p in group["params"] if p.requires_grad)
        recorder.write(
            f"group {index}: params={param_count:,}, lr={group.get('lr'):.2e}, "
            f"weight_decay={group.get('weight_decay')}"
        )


def register_debug_hooks(policy: nn.Module) -> tuple[list[Any], list[str]]:
    """给 ACT 的关键模块挂 forward hook，用真实 batch 打印网络内部形状。"""
    records: list[str] = []
    handles: list[Any] = []

    module_names = [
        "model.vae_encoder_action_input_proj",
        "model.vae_encoder",
        "model.backbone",
        "model.encoder_img_feat_input_proj",
        "model.encoder",
        "model.decoder",
        "model.action_head",
    ]

    named_modules = dict(policy.named_modules())

    def make_hook(name: str):
        def hook(_module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            in_summary = object_shape_summary(inputs)
            out_summary = object_shape_summary(output)
            records.append(f"{name}: input={in_summary} -> output={out_summary}")

        return hook

    for name in module_names:
        module = named_modules.get(name)
        if module is not None:
            handles.append(module.register_forward_hook(make_hook(name)))

    return handles, records


def run_debug_forward(policy: nn.Module, batch: dict[str, Any], recorder: TextRecorder) -> None:
    """训练前跑一次 forward，打印 loss 和关键模块的真实张量形状。"""
    recorder.write("\n========== Debug forward hooks ==========")
    handles, records = register_debug_hooks(policy)
    was_training = policy.training
    policy.train()  # VAE 训练路径需要 training=True 才会读取真实 action chunk。

    with torch.no_grad():
        loss, loss_dict = policy.forward(batch)

    for handle in handles:
        handle.remove()
    policy.train(was_training)

    recorder.write(f"debug loss: {float(loss.detach().cpu()):.6f}")
    recorder.write(f"debug loss_dict: {loss_dict}")
    for line in records:
        recorder.write(line)


def make_jsonable(value: Any) -> Any:
    """把 Path、PolicyFeature、enum、tuple 等对象转成 JSON 可写格式。"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "type") and hasattr(value, "shape"):
        return {"type": make_jsonable(value.type), "shape": list(value.shape)}
    if isinstance(value, dict):
        return {str(key): make_jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_jsonable(child) for child in value]
    return value


def save_config(
    path: Path,
    cfg: ScriptConfig,
    dataset_root: Path,
    output_dir: Path,
    device: torch.device,
    policy: nn.Module,
    delta_timestamps: dict[str, list[float]],
) -> None:
    """保存本次运行使用的配置，避免训练后忘记当时改过哪些参数。"""
    payload = {
        "script_config": make_jsonable(asdict(cfg)),
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "device": str(device),
        "delta_timestamps": delta_timestamps,
        "policy_input_features": make_jsonable(policy.config.input_features),
        "policy_output_features": make_jsonable(policy.config.output_features),
        "policy_normalization_mapping": make_jsonable(policy.config.normalization_mapping),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_checkpoint(
    output_dir: Path,
    step: int,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    cfg: ScriptConfig,
    loss: float | None,
) -> None:
    """保存普通 PyTorch checkpoint，包含模型、优化器和 AMP scaler。"""
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "script_config": make_jsonable(asdict(cfg)),
        "loss": loss,
    }

    step_path = ckpt_dir / f"step_{step:06d}.pt"
    last_path = ckpt_dir / "last.pt"
    torch.save(payload, step_path)
    torch.save(payload, last_path)
    print(f"[checkpoint] saved: {step_path}")


def load_checkpoint(
    checkpoint_path: Path,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> int:
    """从本脚本保存的 checkpoint 恢复训练状态，返回已经完成的 step。"""
    checkpoint_path = checkpoint_path.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    policy.load_state_dict(checkpoint["policy_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    step = int(checkpoint.get("step", 0))
    print(f"[resume] loaded {checkpoint_path}, start_step={step}")
    return step


def append_train_log(log_path: Path, row: dict[str, Any]) -> None:
    """把训练指标追加写入 CSV。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.is_file()
    with log_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def grad_norm_to_float(grad_norm: torch.Tensor | float) -> float:
    """把 clip_grad_norm_ 的返回值转成 Python float。"""
    if torch.is_tensor(grad_norm):
        return float(grad_norm.detach().cpu())
    return float(grad_norm)


def train_loop(
    cfg: ScriptConfig,
    dataset: Any,
    policy: nn.Module,
    preprocessor: Any,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    device: torch.device,
    start_step: int,
    scaler: torch.amp.GradScaler,
) -> None:
    """最小但完整的 ACT 训练循环。"""
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    dl_iter = cycle(dataloader)
    log_path = output_dir / "train_log.csv"

    policy.train()
    print("\n========== Start training ==========")
    print(f"steps: {start_step + 1} -> {cfg.train_steps}")

    for step in range(start_step + 1, cfg.train_steps + 1):
        data_t0 = time.perf_counter()
        raw_batch = next(dl_iter)
        batch = prepare_batch_for_policy(raw_batch, dataset, cfg, preprocessor)
        data_s = time.perf_counter() - data_t0

        update_t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        amp_enabled = scaler.is_enabled()
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            loss, loss_dict = policy.forward(batch)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "loss 出现 NaN/Inf。常见原因是 AMP/FP16 在当前 GPU 上不稳定；"
                "请先把 ScriptConfig.use_amp 设为 False，或降低学习率/模型尺寸。"
            )

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip_norm)
            optimizer.step()

        update_s = time.perf_counter() - update_t0

        loss_value = float(loss.detach().cpu())
        l1_loss = float(loss_dict.get("l1_loss", math.nan))
        kld_loss = float(loss_dict.get("kld_loss", 0.0))
        grad_norm_value = grad_norm_to_float(grad_norm)
        lr = float(optimizer.param_groups[0]["lr"])

        if step % cfg.log_freq == 0 or step == 1:
            print(
                f"[step {step:06d}] "
                f"loss={loss_value:.6f} "
                f"l1={l1_loss:.6f} "
                f"kld={kld_loss:.6f} "
                f"grad={grad_norm_value:.3f} "
                f"lr={lr:.2e} "
                f"data={data_s:.3f}s "
                f"update={update_s:.3f}s"
            )
            append_train_log(
                log_path,
                {
                    "step": step,
                    "loss": loss_value,
                    "l1_loss": l1_loss,
                    "kld_loss": kld_loss,
                    "grad_norm": grad_norm_value,
                    "lr": lr,
                    "data_s": data_s,
                    "update_s": update_s,
                },
            )

        if step % cfg.save_freq == 0 or step == cfg.train_steps:
            save_checkpoint(output_dir, step, policy, optimizer, scaler, cfg, loss_value)


def save_pretrained_policy(output_dir: Path, policy: nn.Module, preprocessor: Any, postprocessor: Any) -> None:
    """保存一份 LeRobot 风格目录，包含 config、model.safetensors 和 processor 配置。"""
    pretrained_dir = output_dir / "policy_pretrained"
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(pretrained_dir, push_to_hub=False)
    preprocessor.save_pretrained(pretrained_dir)
    postprocessor.save_pretrained(pretrained_dir)
    print(f"[pretrained] saved: {pretrained_dir}")


def main(cfg: ScriptConfig | None = None) -> None:
    """脚本入口。"""
    cfg = cfg or ScriptConfig()
    symbols = import_lerobot_symbols()
    set_seed(cfg.seed)

    device = select_device(cfg.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    dataset_root = resolve_dataset_root(cfg)
    output_dir = make_output_dir(cfg)

    dataset, delta_timestamps = load_dataset(cfg, symbols, dataset_root)
    policy = build_policy(cfg, dataset, symbols, device)
    preprocessor, postprocessor = build_preprocessor(policy, dataset, symbols)
    optimizer = build_optimizer(cfg, policy)
    scaler_device = "cuda" if device.type == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(scaler_device, enabled=cfg.use_amp and device.type == "cuda")

    recorder = TextRecorder()
    print_environment_summary(cfg, symbols, dataset, device, delta_timestamps, recorder)
    print_optimizer_summary(optimizer, recorder)

    sample = dataset[0]
    print_nested_shapes("raw dataset[0]", sample, recorder)

    debug_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    raw_batch = next(iter(debug_loader))
    print_nested_shapes("raw DataLoader batch", raw_batch, recorder)

    resized_batch = resize_images(
        convert_uint8_images_to_float(raw_batch, dataset.meta.camera_keys),
        dataset.meta.camera_keys,
        cfg.image_resize_hw,
    )
    print_nested_shapes("after image float conversion / resize", resized_batch, recorder)

    policy_batch = preprocessor(resized_batch)
    print_nested_shapes("after LeRobot ACT preprocessor", policy_batch, recorder)

    if cfg.print_debug_forward:
        run_debug_forward(policy, policy_batch, recorder)

    recorder.save(output_dir / "debug_batch_shapes.txt")

    network_summary = print_policy_summary(policy, dataset, cfg, symbols)
    (output_dir / "network_summary.txt").write_text(network_summary, encoding="utf-8")
    print("\n" + "\n".join(network_summary.splitlines()[:30]))
    print(f"\n完整网络结构已保存到：{output_dir / 'network_summary.txt'}")

    save_config(
        output_dir / "config_used.json",
        cfg,
        dataset_root,
        output_dir,
        device,
        policy,
        delta_timestamps,
    )

    start_step = 0
    if cfg.resume_checkpoint is not None:
        start_step = load_checkpoint(cfg.resume_checkpoint, policy, optimizer, scaler)

    train_loop(cfg, dataset, policy, preprocessor, optimizer, output_dir, device, start_step, scaler)

    if cfg.save_pretrained_at_end:
        save_pretrained_policy(output_dir, policy, preprocessor, postprocessor)

    print("\n========== Done ==========")
    print(f"输出目录：{output_dir}")
    print(f"训练日志：{output_dir / 'train_log.csv'}")
    print(f"最后 checkpoint：{output_dir / 'checkpoints/last.pt'}")


if __name__ == "__main__":
    main()
