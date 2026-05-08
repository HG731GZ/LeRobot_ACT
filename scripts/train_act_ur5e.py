#!/usr/bin/env python3
"""在 UR5e cylinder-to-box 数据集上训练 LeRobot ACT 策略。

# ========== ACT（Action Chunking Transformer）简介 ==========
#
# ACT 是一种基于 Transformer 的模仿学习策略，专门用于机器人操作任务。
# 它的核心思想是：给定一段观测序列（图像 + 关节状态），一次性预测未来
# 多个时间步的动作（称为 action chunk），而不是逐步预测。
#
# 为什么一次性预测多个动作？因为机器人执行动作时需要保持时间上的一致性，
# 逐步预测容易导致抖动和误差累积。一次性预测一整段动作序列，能让模型
# 学到更平滑、更协调的运动轨迹。
#
# ========== ACT 的 Transformer 架构 ==========
#
# ACT 使用了一个 **编码器-解码器（Encoder-Decoder）Transformer**，类似于
# 机器翻译中的 seq2seq 模型：
#
#   ┌──────────────────────────────────────────────────────────────────┐
#   │                        ACT 架构总览                               │
#   │                                                                  │
#   │   观测（图像+状态）──► 编码器(Encoder) ──► 潜在向量 ──► 解码器(Decoder) ──► 动作序列
#   │                         │                    │                      │
#   │                    多层 self-attn      VAE 压缩/采样          cross-attn
#   │                                                                  │
#   └──────────────────────────────────────────────────────────────────┘
#
# 1. **编码器 (Encoder)**：
#    - 视觉主干 (vision backbone)：用预训练的 ResNet18 将摄像头图像转换为特征向量。
#      每张图像经过 ResNet 卷积层后，变成一个固定长度的向量表示。
#    - 状态嵌入：将机器人关节状态（如 TCP 坐标、夹爪开度）映射为向量。
#    - Transformer 编码器层：用 **自注意力（Self-Attention）** 机制处理特征序列。
#      自注意力的意思是：序列中的每个位置都会"关注"序列中所有其他位置，
#      从而捕捉全局上下文信息。比如模型可以同时"看到"当前图像和关节状态，
#      理解它们之间的关系。
#    - VAE 编码器（可选）：将编码器输出压缩为一个低维潜在向量。
#      VAE（变分自编码器）的作用是学习一个紧凑的、有意义的动作表示空间。
#
# 2. **解码器 (Decoder)**：
#    - 接收编码器产生的"记忆"（memory）和当前的动作查询（action query），
#      通过**交叉注意力（Cross-Attention）** 机制，让解码器在生成每个动作时
#      都能"回顾"编码器的观测记忆。
#      Cross-Attention 与 Self-Attention 的区别：
#        · Self-Attention：查询(Query)和键(Key)来自同一个序列（自己看自己）
#        · Cross-Attention：查询来自解码器，键来自编码器（解码器看编码器输出）
#    - 使用**因果注意力掩码（Causal Attention Mask）**，确保预测第 t 步动作时
#      只能看到第 t 步之前的动作，不能"偷看"未来的动作。这保证了动作序列的
#      自回归（autoregressive）生成顺序。
#    - 输出一个长度为 chunk_size 的动作序列，每个动作包含 n_action_steps 步实际指令。
#
# ========== 关键超参数 ==========
#
# • dim_model：Transformer 中所有隐藏向量的维度。越大容量越大，但也越吃显存。
# • n_heads：多头注意力（Multi-Head Attention）的头数。把 dim_model 分成 n_heads 份，
#   每组独立计算注意力，最后拼接。多头让模型从不同角度理解输入关系。
#   例如 dim_model=512, n_heads=8，则每个头处理 512/8=64 维的子空间。
# • dim_feedforward：前馈网络（Feed-Forward Network, FFN）的隐藏层维度。
#   每个 Transformer 层中，在注意力之后接一个两层 MLP，先升维再降维。
#   通常设为 dim_model 的 4 倍左右。
# • n_encoder_layers：编码器中 Transformer 层的数量。更深 → 更强的表示能力。
# • n_decoder_layers：解码器中 Transformer 层的数量。
# • n_vae_encoder_layers：VAE 编码器中 Transformer 层的数量，用于学习动作潜在空间。
# • chunk_size：一次预测的动作序列长度（帧数）。例如 chunk_size=32 表示
#   模型一次预测 32 帧的动作。
# • n_action_steps：实际执行的动作步数（≤ chunk_size）。预测了 chunk_size 帧，
#   但只执行前 n_action_steps 帧，然后重新预测。这就是"滚动预测"机制：
#   执行一部分 → 重新观测 → 再预测 → 再执行。
# • kl_weight：VAE 的 KL 散度损失权重。KL 散度衡量潜在分布与标准正态分布的差异。
#   权重越大，潜在空间越接近正态分布（更规整但可能丢失信息）；越小则重构越精确
#   （但潜在空间可能不连续）。
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
    - n_vae_encoder_layers: VAE 编码器的 Transformer 层数
    - chunk_size: 一次预测的动作块大小（帧数）
    - n_action_steps: 实际执行的动作步数
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
    # encoder 4层，VAE encoder 4层，能学到更丰富的特征表示。
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
                        help="实际执行的动作步数（≤ chunk_size）")
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
                        help="是否使用 VAE 变分自编码器学习动作潜在空间")
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
            # chunk_size: Transformer 解码器一次性输出的动作序列长度
            # n_action_steps: 实际执行前多少步，然后重新预测（滚动预测）
            chunk_size=choose(args.chunk_size, preset.chunk_size),
            n_action_steps=choose(args.n_action_steps, preset.n_action_steps),
            # == 视觉 backbone ==
            # ResNet18 将图像编码为特征向量，作为 Transformer 编码器的输入
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
            # use_vae: 是否在编码器后使用 VAE。VAE 将编码器输出映射到一个概率分布
            #   （均值 μ 和方差 σ²），从中采样潜在向量 z。
            #   这样做的好处：(1) 潜在空间更平滑连续 (2) 有助于生成多样化的动作
            #   (3) KL 散度约束防止过拟合
            use_vae=args.use_vae,
            # latent_dim: 潜在向量 z 的维度。编码器输出被压缩到这个维度
            latent_dim=args.latent_dim,
            # n_vae_encoder_layers: VAE 编码器中额外的 Transformer 层数
            #   VAE 编码器在主干编码器之上再堆叠几层 Transformer，
            #   然后分别输出 μ 和 log(σ²)，用于构造潜在分布 N(μ, σ²)
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
