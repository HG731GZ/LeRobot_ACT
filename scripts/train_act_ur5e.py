#!/usr/bin/env python3
"""在 UR5e cylinder-to-box 数据集上训练 LeRobot ACT 策略。

# ========== ACT（Action Chunking with Transformers）简介 ==========
#
# ACT 是一种面向机器人操作的模仿学习策略。它的核心思想是：给定当前观测
# （通常是多路 RGB 图像 + 当前机器人本体状态），一次性预测未来多个时间步的
# 动作序列，这段动作序列称为 action chunk。
#
# 在 LeRobot 当前实现中，ACT 默认 n_obs_steps=1，也就是输入当前时刻的观测，
# 而不是显式输入一段历史观测序列。时间上的平滑性主要来自 action chunk、
# action queue，以及可选的 temporal ensemble。
#
# 为什么一次性预测多个动作？因为机器人执行需要短时间内连续、协调的动作。
# 若每一步都独立预测，容易出现抖动和误差累积；预测一段动作可以让模型学到
# 更平滑的局部轨迹。推理时通常只执行 chunk 中的前 n_action_steps 步，然后
# 重新读取观测并预测下一段动作。
#
# ========== LeRobot ACT 的数据流 ==========
#
#   当前观测 batch
#     ├─ observation.images.* ─► ResNet18 backbone ─► feature map
#     │                                             └► 1x1 Conv 投影
#     │                                             └► 展平成视觉 token
#     ├─ observation.state ─► Linear ─► 状态 token
#     └─ 训练时 action chunk ─► VAE encoder ─► latent z；推理时 z = 0
#
#   [latent token, 状态 token, 视觉 tokens]
#       └─► Transformer Encoder 融合观测信息
#             └─► Transformer Decoder 使用 chunk_size 个可学习位置查询并行生成动作 token
#                   └─► action_head 线性回归
#                         └─► [B, chunk_size, action_dim]
#
# ========== ACT 的 Transformer / VAE 结构 ==========
#
# LeRobot 代码里容易混淆的三个模块：vae_encoder、encoder、decoder。
#
# 1. **视觉 backbone**：
#    - 使用去掉分类头的 ResNet18。它不是做图像分类，也不输出类别概率。
#    - ResNet 的 layer4 输出是一张低分辨率 feature map，例如 [B, C, H', W']。
#    - 之后通过 1x1 Conv 映射到 dim_model，再把 H'×W' 个空间位置展平成视觉 token。
#    - 因此每张图像不是变成一个固定长度分类向量，而是变成一串带空间位置的视觉 token。
#
# 2. **VAE encoder（仅训练时使用，且 use_vae=True 时启用）**：
#    - 它不是把主 Transformer encoder 的输出再压缩。
#    - 它单独接收 [CLS token, 当前机器人状态 token, 真实 action chunk tokens]，
#      输出 latent 分布参数 μ 和 log(σ²)，并通过重参数化采样得到 z。
#    - 这个 z 表示演示动作序列中的隐含变化模式。推理时没有真实 action chunk，
#      因此 LeRobot 会直接令 z 为全 0。
#
# 3. **主 Transformer encoder / decoder**：
#    - encoder 接收 latent token、状态 token、可选环境状态 token，以及所有视觉 token，
#      通过 self-attention 融合这些观测信息。
#    - decoder 的输入是长度为 chunk_size 的查询序列。它不是像语言模型那样
#      自回归地一步步预测，也没有使用 causal mask 来禁止“看未来动作”。
#      在 LeRobot ACT 中，decoder 用 self-attention 和 cross-attention 并行生成
#      chunk_size 个动作 token。
#    - 最后的 action_head 把每个动作 token 回归为一个 action 向量。
#
# ========== 关键超参数 ==========
#
# • dim_model：Transformer 中 token 的隐藏维度。视觉 token、状态 token、latent token
#   最终都会被投影到这个维度。
# • n_heads：多头注意力的头数。dim_model 必须能被 n_heads 整除。
# • dim_feedforward：Transformer 层中 FFN 的中间层维度，通常大于 dim_model。
# • n_encoder_layers：主 Transformer encoder 层数，用于融合视觉、状态和 latent token。
# • n_decoder_layers：Transformer decoder 层数。LeRobot 默认保持为 1，以匹配原 ACT 实现行为。
# • n_vae_encoder_layers：VAE encoder 的层数；它是独立的动作序列编码器，不是主 encoder 的附加层。
# • chunk_size：模型一次预测的动作序列长度，即输出 [B, chunk_size, action_dim]。
# • n_action_steps：不使用 temporal ensemble 时，一次 forward 后实际放入 action queue 并执行的动作数。
#   它必须 ≤ chunk_size。例如 chunk_size=64、n_action_steps=32 表示预测 64 帧，先执行前 32 帧，
#   然后重新观测并预测下一段。
# • kl_weight：VAE KL 散度损失权重。总损失为动作重构损失 + kl_weight × KL 损失。
#
# ========== 训练流程 ==========
#
# 本脚本不直接实现 ACT 模型，而是：
# 1. 根据预设（preset）和命令行参数生成 LeRobot 训练配置文件
# 2. 调用官方的 lerobot.scripts.lerobot_train 启动训练
# 3. 这样保留了 LeRobot 官方的 checkpoint、processor 保存和断点续训机制
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_DATASET_ROOT = Path("data/lerobot/ur5e_cylinder_to_box")
DEFAULT_REPO_ID = "local/ur5e_cylinder_to_box"
CONFIG_DIR = Path("outputs/train_configs")
# UR5e 机器人观测状态名称：TCP 的 x/y/z 坐标、rx/ry/rz 旋转角、夹爪开度和电流
EXPECTED_STATE_NAMES = [
    "tcp_x",
    "tcp_y",
    "tcp_z",
    "tcp_rx",
    "tcp_ry",
    "tcp_rz",
    "gripper_opening",
    "gripper_current",
]
# UR5e 动作名称：TCP 坐标的增量变化（delta）和夹爪目标开度
# 使用增量（delta）而非绝对位置，是因为增量控制更稳定，模型学的是"往哪移多少"
EXPECTED_ACTION_NAMES = [
    "delta_tcp_x",
    "delta_tcp_y",
    "delta_tcp_z",
    "delta_tcp_rx",
    "delta_tcp_ry",
    "delta_tcp_rz",
    "gripper_target_opening",
]


@dataclass(frozen=True)
class TrainPreset:
    """训练超参数预设，针对不同 GPU 显存容量做了调整。

    Transformer 相关参数说明：
    - dim_model: Transformer 隐藏层维度（所有位置的向量都用这个维度）
    - n_heads: 多头注意力的头数（dim_model 必须能被 n_heads 整除）
    - dim_feedforward: 前馈网络的中间层维度
    - n_encoder_layers: 编码器 Transformer 层数
    - n_decoder_layers: 解码器 Transformer 层数
    - n_vae_encoder_layers: VAE encoder 自身的 Transformer 层数
    - chunk_size: 一次 forward 预测的动作块大小（帧数）
    - n_action_steps: 不使用 temporal ensemble 时每次预测后放入执行队列的动作步数
    """
    batch_size: int
    num_workers: int
    steps: int
    save_freq: int
    log_freq: int
    use_amp: bool
    dim_model: int
    n_heads: int
    dim_feedforward: int
    n_encoder_layers: int
    n_decoder_layers: int
    n_vae_encoder_layers: int
    chunk_size: int
    n_action_steps: int
    pretrained_backbone: bool


PRESETS: dict[str, TrainPreset] = {
    # 1660Ti 通常是 6GB 显存；两路 720p 图像会比较吃紧，所以模型和 batch 都收小。
    # Transformer 使用较小容量：dim_model=256（隐藏维度256），n_heads=4（4个注意力头，每个头64维），
    # encoder 2层，decoder 1层，属于轻量级配置。
    "1660ti": TrainPreset(
        batch_size=1,
        num_workers=2,
        steps=30_000,
        save_freq=3_000,
        log_freq=50,
        use_amp=True,
        dim_model=256,
        n_heads=4,
        dim_feedforward=1_024,
        n_encoder_layers=2,
        n_decoder_layers=1,
        n_vae_encoder_layers=2,
        chunk_size=32,
        n_action_steps=16,
        pretrained_backbone=True,
    ),
    # A6000 显存余量大（48GB），可以更接近 LeRobot ACT 默认容量。
    # Transformer 使用更大容量：dim_model=512（隐藏维度512），n_heads=8（8个注意力头，每个头64维），
    # 主 encoder 4层，VAE encoder 4层，能学习更丰富的观测-动作表示。
    "a6000": TrainPreset(
        batch_size=8,
        num_workers=8,
        steps=60_000,
        save_freq=5_000,
        log_freq=100,
        use_amp=True,
        dim_model=512,
        n_heads=8,
        dim_feedforward=3_200,
        n_encoder_layers=4,
        n_decoder_layers=1,
        n_vae_encoder_layers=4,
        chunk_size=64,
        n_action_steps=32,
        pretrained_backbone=True,
    ),
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """解析命令行参数。

    返回 (已解析的参数, LeRobot 透传参数列表)。
    透传参数会被原样传给 lerobot.scripts.lerobot_train。
    """
    parser = argparse.ArgumentParser(
        description=(
            "为 UR5e cylinder-to-box 数据集生成 LeRobot ACT 训练配置，"
            "并启动 LeRobot 官方训练器。"
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="训练输出目录。默认为 outputs/train/ur5e_act_<preset>。",
    )
    parser.add_argument(
        "--preset",
        choices=["auto", "1660ti", "a6000", "custom"],
        default="auto",
        help="硬件预设。选 custom 时所有值均从 CLI 参数传入。",
    )
    parser.add_argument("--device", default="cuda", help="传给策略的 PyTorch 设备。")
    parser.add_argument("--resume-config", type=Path, default=None, help="断点续训：指向已有的 train_config.json。")
    parser.add_argument("--overwrite-output", action="store_true", help="训练前删除 output-dir。")
    parser.add_argument("--dry-run", action="store_true", help="仅打印生成的配置和命令，不执行训练。")
    parser.add_argument("--config-output", type=Path, default=None, help="生成的配置文件写入位置。")
    parser.add_argument(
        "--skip-dataset-schema-check",
        action="store_true",
        help=(
            "跳过 observation.state/action 名称与当前 UR5e schema 的校验。"
            "正常情况下保留此校验，以检测旧版 gripper-delta 数据集。"
        ),
    )

    # —— 训练超参数（会覆盖预设值） ——
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--save-freq", type=int, default=None)
    parser.add_argument("--log-freq", type=int, default=None)
    parser.add_argument("--eval-freq", type=int, default=0, help="离线真实数据设为 0。")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--video-backend",
        default="pyav",
        choices=["pyav", "torchcodec", "video_reader"],
        help=(
            "LeRobotDataset 的视频解码后端。pyav 是当前环境最安全的选择；"
            "当 torchcodec 的 FFmpeg 库与 PyTorch 安装匹配时可更快。"
        ),
    )

    # —— ACT / Transformer 超参数 ——
    # 以下参数会直接写入 ACTConfig，覆盖预设值。对这些参数的含义详见文件顶部注释。
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="一次预测的动作块大小（帧数）")
    parser.add_argument("--n-action-steps", type=int, default=None,
                        help="不使用 temporal ensemble 时，每次预测后实际执行/入队的动作步数（≤ chunk_size）")
    parser.add_argument("--dim-model", type=int, default=None,
                        help="Transformer 隐藏层维度")
    parser.add_argument("--n-heads", type=int, default=None,
                        help="多头注意力头数")
    parser.add_argument("--dim-feedforward", type=int, default=None,
                        help="前馈网络隐藏层维度")
    parser.add_argument("--n-encoder-layers", type=int, default=None,
                        help="编码器 Transformer 层数")
    parser.add_argument("--n-decoder-layers", type=int, default=None,
                        help="解码器 Transformer 层数")
    parser.add_argument("--n-vae-encoder-layers", type=int, default=None,
                        help="VAE 编码器的 Transformer 层数")
    parser.add_argument("--latent-dim", type=int, default=32,
                        help="VAE 潜在空间维度")
    parser.add_argument("--kl-weight", type=float, default=10.0,
                        help="VAE KL 散度损失权重。越大则潜在空间越接近标准正态分布。")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Transformer 中的 Dropout 比例，用于防止过拟合")
    parser.add_argument("--optimizer-lr", type=float, default=1e-5,
                        help="优化器学习率（除 backbone 外的参数）")
    parser.add_argument("--optimizer-lr-backbone", type=float, default=1e-5,
                        help="视觉 backbone 的学习率（通常设得比整体学习率小）")
    parser.add_argument("--optimizer-weight-decay", type=float, default=1e-4,
                        help="优化器权重衰减（L2 正则化系数）")
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=None,
                        help="是否使用自动混合精度训练（节省显存）")
    parser.add_argument("--use-vae", action=argparse.BooleanOptionalAction, default=True,
                        help="是否使用 VAE 变分目标；训练时额外编码真实 action chunk 得到 latent z")
    parser.add_argument(
        "--pretrained-backbone",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="使用 torchvision ImageNet 预训练的 ResNet18 权重。若机器无法下载权重请关闭。",
    )
    parser.add_argument(
        "--use-imagenet-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="使用 ImageNet 的归一化统计量（mean/std）处理摄像头图像，与 LeRobot 默认行为一致。",
    )
    parser.add_argument("--wandb", action="store_true", help="启用 Weights & Biases 日志记录。")
    parser.add_argument("--wandb-project", default="lerobot")
    parser.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"])

    # parse_known_args 允许透传未知参数给 LeRobot 训练器
    args, unknown = parser.parse_known_args()
    return args, unknown


def select_preset(name: str) -> tuple[str, TrainPreset]:
    """根据用户选择或自动检测 GPU 类型选择预设。

    自动检测逻辑：
    - 检测到 A6000 或显存 ≥24GB → 使用 a6000 预设（大模型）
    - 其他情况 → 使用 1660ti 预设（小模型，保守配置）
    """
    if name == "custom":
        return "custom", PRESETS["1660ti"]
    if name != "auto":
        return name, PRESETS[name]

    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name.lower()
            total_gb = props.total_memory / 1024**3
            if "a6000" in gpu_name or total_gb >= 24:
                return "a6000", PRESETS["a6000"]
    except Exception:
        pass

    return "1660ti", PRESETS["1660ti"]


def choose(value: Any, fallback: Any) -> Any:
    """如果显式值为 None，则使用回退值（来自预设）。"""
    return fallback if value is None else value


def ensure_lerobot_imports():
    """验证 LeRobot 依赖可用，返回所需的类和函数。"""
    try:
        import draccus
        from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig
        from lerobot.configs.train import TrainPipelineConfig
        from lerobot.policies.act.configuration_act import ACTConfig
    except ImportError as exc:
        raise SystemExit(
            "无法导入 LeRobot 训练依赖。请先运行：conda activate LeRobot，然后再执行本脚本。"
        ) from exc

    return draccus, DatasetConfig, EvalConfig, WandBConfig, TrainPipelineConfig, ACTConfig


def build_config(args: argparse.Namespace, preset_name: str, preset: TrainPreset):
    """根据命令行参数和预设构建完整的 LeRobot 训练配置对象。

    配置的核心是 ACTConfig，它定义了 ACT Transformer 的所有结构参数。
    """
    draccus, DatasetConfig, EvalConfig, WandBConfig, TrainPipelineConfig, ACTConfig = ensure_lerobot_imports()

    output_dir = args.output_dir or Path("outputs/train") / f"ur5e_act_{preset_name}"
    pretrained_backbone = choose(args.pretrained_backbone, preset.pretrained_backbone)
    # 使用 ImageNet 预训练的 ResNet18 作为视觉 backbone
    pretrained_backbone_weights = "ResNet18_Weights.IMAGENET1K_V1" if pretrained_backbone else None

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=args.repo_id,
            root=str(args.dataset_root),
            use_imagenet_stats=args.use_imagenet_stats,
            video_backend=args.video_backend,
        ),
        # ACT 策略配置 —— 这里是所有 Transformer 结构参数的实际落点
        policy=ACTConfig(
            device=args.device,
            use_amp=choose(args.use_amp, preset.use_amp),
            push_to_hub=False,
            # == 动作分块参数 ==
            # chunk_size: 模型一次 forward 输出的动作序列长度，输出形状为 [B, chunk_size, action_dim]
            # n_action_steps: 不使用 temporal ensemble 时，每次预测后实际放入 action queue 执行的步数
            chunk_size=choose(args.chunk_size, preset.chunk_size),
            n_action_steps=choose(args.n_action_steps, preset.n_action_steps),
            # == 视觉 backbone ==
            # ResNet18 去掉分类头后作为特征提取器，输出 layer4 feature map；
            # LeRobot 随后用 1x1 Conv 投影并展平成视觉 tokens 输入 Transformer 编码器
            vision_backbone="resnet18",
            pretrained_backbone_weights=pretrained_backbone_weights,
            # == Transformer 结构参数 ==
            # dim_model: 所有 Transformer 层的隐藏向量维度（模型的"宽度"）
            dim_model=choose(args.dim_model, preset.dim_model),
            # n_heads: 多头注意力头数。每个头在 dim_model/n_heads 的子空间中计算注意力
            n_heads=choose(args.n_heads, preset.n_heads),
            # dim_feedforward: 前馈网络中间层维度，通常是 dim_model 的 4 倍
            dim_feedforward=choose(args.dim_feedforward, preset.dim_feedforward),
            # n_encoder_layers: 编码器堆叠的 Transformer 层数（模型的"深度"）
            n_encoder_layers=choose(args.n_encoder_layers, preset.n_encoder_layers),
            # n_decoder_layers: 解码器堆叠的 Transformer 层数
            n_decoder_layers=choose(args.n_decoder_layers, preset.n_decoder_layers),
            # == VAE（变分自编码器）参数 ==
            # use_vae: 是否使用“变分目标”。启用后会额外建立一个 VAE encoder，
            #   训练时把 [CLS, 当前机器人状态, 真实 action chunk] 编码成 latent 分布，
            #   得到 μ 和 log(σ²)，再采样 latent z 供主 Transformer 使用。
            #   推理时没有真实 action chunk，因此 z 直接置为 0。
            use_vae=args.use_vae,
            # latent_dim: latent z 的维度，不是主 encoder 输出维度，而是 VAE encoder 输出的隐变量维度
            latent_dim=args.latent_dim,
            # n_vae_encoder_layers: VAE encoder 自身的 Transformer 层数。
            #   它是独立的动作序列编码器，不是在主 Transformer encoder 后面继续堆叠。
            #   输出的 μ 和 log(σ²) 用于构造潜在分布 N(μ, σ²)
            n_vae_encoder_layers=choose(args.n_vae_encoder_layers, preset.n_vae_encoder_layers),
            # == 正则化与优化参数 ==
            # dropout: 训练时随机"丢弃"一部分神经元，防止过拟合
            dropout=args.dropout,
            # kl_weight: VAE 的 KL 散度损失权重。
            #   Loss_total = Loss_reconstruction + kl_weight × KL(N(μ,σ²) || N(0,1))
            #   其中 KL 散度衡量学到的分布与标准正态分布的差异
            kl_weight=args.kl_weight,
            optimizer_lr=args.optimizer_lr,
            optimizer_lr_backbone=args.optimizer_lr_backbone,
            optimizer_weight_decay=args.optimizer_weight_decay,
        ),
        output_dir=output_dir,
        seed=args.seed,
        num_workers=choose(args.num_workers, preset.num_workers),
        batch_size=choose(args.batch_size, preset.batch_size),
        steps=choose(args.steps, preset.steps),
        eval_freq=args.eval_freq,
        log_freq=choose(args.log_freq, preset.log_freq),
        save_checkpoint=True,
        save_freq=choose(args.save_freq, preset.save_freq),
        eval=EvalConfig(n_episodes=1, batch_size=1),
        wandb=WandBConfig(enable=args.wandb, project=args.wandb_project, mode=args.wandb_mode),
    )

    return cfg, draccus


def write_config(cfg: Any, draccus: Any, path: Path) -> None:
    """将配置对象序列化为 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        with draccus.config_type("json"):
            draccus.dump(cfg, stream, indent=2)


def print_json(path: Path) -> None:
    """打印 JSON 文件内容到终端。"""
    with path.open("r") as stream:
        payload = json.load(stream)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def validate_dataset_schema(dataset_root: Path) -> dict[str, Any]:
    """验证数据集的 observation.state 和 action 名称是否与预期一致。

    这很重要：如果数据集的列名和模型期望的不匹配（例如旧数据集用
    delta_gripper_opening 而模型期望 gripper_target_opening），
    训练会静默产生错误结果。
    """
    info_path = dataset_root / "meta" / "info.json"
    with info_path.open("r") as stream:
        info = json.load(stream)

    features = info.get("features", {})
    state_names_payload = features.get("observation.state", {}).get("names") or {}
    action_names_payload = features.get("action", {}).get("names") or {}
    state_names = state_names_payload.get("state") if isinstance(state_names_payload, dict) else None
    action_names = action_names_payload.get("action") if isinstance(action_names_payload, dict) else None

    errors: list[str] = []
    if state_names != EXPECTED_STATE_NAMES:
        errors.append(
            "observation.state names mismatch:\n"
            f"  expected: {EXPECTED_STATE_NAMES}\n"
            f"  got:      {state_names}"
        )
    if action_names != EXPECTED_ACTION_NAMES:
        errors.append(
            "action names mismatch:\n"
            f"  expected: {EXPECTED_ACTION_NAMES}\n"
            f"  got:      {action_names}\n"
            "如果这里看到 delta_gripper_opening，说明数据集还是旧转换结果；"
            "请先用新版 convert_ur5e_to_lerobot.py 重新转换。"
        )

    if errors:
        raise ValueError("\n".join(errors))
    return info


def resume_dataset_root(config_path: Path) -> Path | None:
    """从已有训练配置中读取数据集根路径（用于断点续训时验证数据集）。"""
    with config_path.open("r") as stream:
        payload = json.load(stream)
    root = payload.get("dataset", {}).get("root")
    if root is None:
        return None
    return Path(root).expanduser().resolve()


def print_dataset_summary(info: dict[str, Any]) -> None:
    """打印数据集概要信息。"""
    print(
        "Dataset schema OK: "
        f"episodes={info.get('total_episodes')}, "
        f"frames={info.get('total_frames')}, "
        f"fps={info.get('fps')}, "
        "action[-1]=gripper_target_opening"
    )


def main() -> int:
    """主入口函数。

    两种模式：
    1. 断点续训模式（--resume-config）：直接恢复已有训练
    2. 新建训练模式：生成配置 → 启动 LeRobot 训练器
    """
    args, unknown_lerobot_args = parse_args()

    # —— 断点续训模式 ——
    if args.resume_config is not None:
        resume_config = args.resume_config.expanduser().resolve()
        if not resume_config.is_file():
            raise FileNotFoundError(resume_config)
        if not args.skip_dataset_schema_check:
            resumed_dataset_root = resume_dataset_root(resume_config)
            if resumed_dataset_root is not None:
                print_dataset_summary(validate_dataset_schema(resumed_dataset_root))
        command = [
            sys.executable,
            "-m",
            "lerobot.scripts.lerobot_train",
            f"--config_path={resume_config}",
            "--resume=true",
            *unknown_lerobot_args,
        ]
        print("Resume command:")
        print(" ".join(str(part) for part in command))
        if args.dry_run:
            return 0
        return subprocess.run(command, check=False).returncode

    # —— 新建训练模式 ——
    dataset_root = args.dataset_root.expanduser().resolve()
    if not (dataset_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"找不到 LeRobot 数据集元信息：{dataset_root / 'meta' / 'info.json'}")
    if not args.skip_dataset_schema_check:
        print_dataset_summary(validate_dataset_schema(dataset_root))

    preset_name, preset = select_preset(args.preset)
    cfg, draccus = build_config(args, preset_name, preset)
    output_dir = Path(cfg.output_dir)
    if output_dir.exists():
        if args.overwrite_output:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"{output_dir} 已存在。继续训练请用 --resume-config；重新训练请加 --overwrite-output。"
            )

    config_path = args.config_output
    if config_path is None:
        config_path = CONFIG_DIR / f"{output_dir.name}.json"
    config_path = config_path.expanduser().resolve()
    write_config(cfg, draccus, config_path)

    command = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--config_path={config_path}",
        *unknown_lerobot_args,
    ]

    print(f"Selected preset: {preset_name}")
    print(f"Generated config: {config_path}")
    print(f"Output dir: {output_dir}")
    print("Training command:")
    print(" ".join(str(part) for part in command))

    if args.dry_run:
        print("\nGenerated LeRobot config:")
        print_json(config_path)
        return 0

    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
