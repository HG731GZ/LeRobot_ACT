#!/usr/bin/env python3
"""尽量少依赖 LeRobot 的 ACT 教学训练脚本。

本文件的定位：
1. 数据仍然用 LeRobotDataset 读取，因为你的数据就是 LeRobot 格式。
2. 网络结构按当前环境中的 LeRobot ACTPolicy/ACT 源码复刻，但不导入 LeRobot 的策略类。
3. 归一化、forward、loss、反传、优化器、checkpoint 都在这里显式写出。
4. 不做命令行参数，不做 json 配置，不做 wandb/hub/eval/resume，只保留学习 ACT 必需的东西。

直接在 VSCode / PyCharm 里选择 conda 环境 LeRobot，然后运行本文件即可。
"""

from __future__ import annotations

import math
import random
import time
from collections import deque
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

# 只在数据加载处使用 LeRobot。模型、loss、归一化和训练过程都在本文件中实现。
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


# ============================== 你通常只需要改这里 ==============================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 可选值："1660ti" 或 "a6000"。
# 1660Ti 预设偏保守，适合小电脑看懂并跑起来；A6000 预设更接近 LeRobot ACT 默认容量。
HARDWARE_PRESET = "1660ti"

# 兼容你描述的 data/data 路径和当前仓库实际存在的 data 路径。
DATASET_ROOT_CANDIDATES = [
    PROJECT_ROOT / "data/data/lerobot/ur5e_cylinder_to_box",
    PROJECT_ROOT / "data/lerobot/ur5e_cylinder_to_box",
]

# LeRobotDataset 需要 repo_id。对本地数据集来说，它主要是一个名字。
DATASET_REPO_ID = "local/ur5e_cylinder_to_box"

# 输出目录只存 checkpoint 和少量文本日志，不另存 json 配置。
OUTPUT_DIR = PROJECT_ROOT / "outputs/minimal_act_pytorch"

# 训练步数。先跑小一点便于学习；真正训练时可以改成 30000 或更多。
TRAIN_STEPS = 2000
LOG_FREQ = 20
SAVE_FREQ = 500
SEED = 42

# 当前数据的视频后端用 pyav 更稳。
VIDEO_BACKEND = "pyav"

# 是否把图像统计量替换为 ImageNet mean/std，等价于 LeRobot 官方训练入口的常见设置。
USE_IMAGENET_STATS = True

# 是否使用 torchvision 的 ImageNet 预训练 ResNet18。
# 如果你的机器没有缓存并且无法联网下载，可改成 None。
PRETRAINED_BACKBONE_WEIGHTS = "ResNet18_Weights.IMAGENET1K_V1"


@dataclass(frozen=True)
class HardwarePreset:
    """只保留两套硬件相关参数，避免 LeRobot 官方配置系统里大量泛化参数。"""

    batch_size: int
    num_workers: int
    image_resize_hw: tuple[int, int] | None
    chunk_size: int
    n_action_steps: int
    dim_model: int
    n_heads: int
    dim_feedforward: int
    n_encoder_layers: int
    n_decoder_layers: int
    n_vae_encoder_layers: int
    latent_dim: int
    lr: float
    lr_backbone: float
    weight_decay: float
    grad_clip_norm: float
    use_amp: bool


PRESETS = {
    "1660ti": HardwarePreset(
        batch_size=1,
        num_workers=2,
        image_resize_hw=(240, 320),
        chunk_size=32,
        n_action_steps=16,
        dim_model=256,
        n_heads=4,
        dim_feedforward=1024,
        n_encoder_layers=2,
        n_decoder_layers=1,
        n_vae_encoder_layers=2,
        latent_dim=32,
        lr=1e-5,
        lr_backbone=1e-5,
        weight_decay=1e-4,
        grad_clip_norm=10.0,
        # 1660Ti 上 ACT+VAE 用 FP16 容易出现 NaN，所以默认关闭。
        use_amp=False,
    ),
    "a6000": HardwarePreset(
        batch_size=8,
        num_workers=8,
        # A6000 可改成 None 使用原始 720p；这里保留适度缩放，训练速度和显存更友好。
        image_resize_hw=(360, 640),
        chunk_size=64,
        n_action_steps=32,
        dim_model=512,
        n_heads=8,
        dim_feedforward=3200,
        n_encoder_layers=4,
        n_decoder_layers=1,
        n_vae_encoder_layers=4,
        latent_dim=32,
        lr=1e-5,
        lr_backbone=1e-5,
        weight_decay=1e-4,
        grad_clip_norm=10.0,
        use_amp=True,
    ),
}


# ============================== LeRobot 字段名常量 ==============================

ACTION = "action"
OBS_STATE = "observation.state"
OBS_IMAGES = "observation.images"

IMAGE_NET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
IMAGE_NET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


# ============================== ACT 配置与特征描述 ==============================


@dataclass(frozen=True)
class FeatureSpec:
    """比 LeRobot 的 PolicyFeature 更小的本地版本，只记录类型和形状。"""

    type: str
    shape: tuple[int, ...]


@dataclass
class ACTConfig:
    """只保留本任务需要的 ACT 参数，命名尽量贴近 LeRobot ACTConfig。"""

    input_features: dict[str, FeatureSpec]
    output_features: dict[str, FeatureSpec]

    chunk_size: int
    n_action_steps: int

    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = PRETRAINED_BACKBONE_WEIGHTS
    replace_final_stride_with_dilation: bool = False

    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1

    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    dropout: float = 0.1
    kl_weight: float = 10.0

    @property
    def image_features(self) -> dict[str, FeatureSpec]:
        return {key: spec for key, spec in self.input_features.items() if spec.type == "VISUAL"}

    @property
    def robot_state_feature(self) -> FeatureSpec | None:
        spec = self.input_features.get(OBS_STATE)
        if spec is not None and spec.type == "STATE":
            return spec
        return None

    @property
    def action_feature(self) -> FeatureSpec:
        spec = self.output_features[ACTION]
        if spec.type != "ACTION":
            raise ValueError("action feature 类型必须是 ACTION")
        return spec


# ============================== 与 LeRobot 一致的 ACT 模型 ==============================


def get_activation_fn(name: str):
    """返回 Transformer FFN 的激活函数。LeRobot ACT 默认使用 relu。"""
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    if name == "glu":
        return F.glu
    raise ValueError(f"不支持的激活函数：{name}")


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """一维正弦位置编码，和 LeRobot ACT 的实现一致。"""

    def get_position_angle_vec(position: int) -> list[float]:
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    table[:, 0::2] = np.sin(table[:, 0::2])
    table[:, 1::2] = np.cos(table[:, 1::2])
    return torch.from_numpy(table).float()


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """二维正弦位置编码，和 LeRobot ACT 的图像 feature map 位置编码一致。"""

    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self.two_pi = 2 * math.pi
        self.eps = 1e-6
        self.temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, H, W]，返回 [1, C_pos, H, W]。
        not_mask = torch.ones_like(x[0, :1])
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        y_range = y_range / (y_range[:, -1:, :] + self.eps) * self.two_pi
        x_range = x_range / (x_range[:, :, -1:] + self.eps) * self.two_pi

        inverse_frequency = self.temperature ** (
            2
            * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2)
            / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency
        y_range = y_range.unsqueeze(-1) / inverse_frequency

        pos_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class ACTEncoderLayer(nn.Module):
    """和 LeRobot ACTEncoderLayer 等价的 Transformer encoder 层。"""

    def __init__(self, cfg: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(cfg.dim_model, cfg.n_heads, dropout=cfg.dropout)

        self.linear1 = nn.Linear(cfg.dim_model, cfg.dim_feedforward)
        self.dropout = nn.Dropout(cfg.dropout)
        self.linear2 = nn.Linear(cfg.dim_feedforward, cfg.dim_model)

        self.norm1 = nn.LayerNorm(cfg.dim_model)
        self.norm2 = nn.LayerNorm(cfg.dim_model)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.dropout2 = nn.Dropout(cfg.dropout)

        self.activation = get_activation_fn(cfg.feedforward_activation)
        self.pre_norm = cfg.pre_norm

    def forward(self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)[0]
        x = skip + self.dropout1(x)

        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x

        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x


class ACTEncoder(nn.Module):
    """多层 ACT encoder；VAE encoder 和主 encoder 共用这份结构。"""

    def __init__(self, cfg: ACTConfig, is_vae_encoder: bool = False):
        super().__init__()
        layer_count = cfg.n_vae_encoder_layers if is_vae_encoder else cfg.n_encoder_layers
        self.layers = nn.ModuleList([ACTEncoderLayer(cfg) for _ in range(layer_count)])
        self.norm = nn.LayerNorm(cfg.dim_model) if cfg.pre_norm else nn.Identity()

    def forward(self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        return self.norm(x)


class ACTDecoderLayer(nn.Module):
    """和 LeRobot ACTDecoderLayer 等价的 Transformer decoder 层。"""

    def __init__(self, cfg: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(cfg.dim_model, cfg.n_heads, dropout=cfg.dropout)
        self.multihead_attn = nn.MultiheadAttention(cfg.dim_model, cfg.n_heads, dropout=cfg.dropout)

        self.linear1 = nn.Linear(cfg.dim_model, cfg.dim_feedforward)
        self.dropout = nn.Dropout(cfg.dropout)
        self.linear2 = nn.Linear(cfg.dim_feedforward, cfg.dim_model)

        self.norm1 = nn.LayerNorm(cfg.dim_model)
        self.norm2 = nn.LayerNorm(cfg.dim_model)
        self.norm3 = nn.LayerNorm(cfg.dim_model)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.dropout2 = nn.Dropout(cfg.dropout)
        self.dropout3 = nn.Dropout(cfg.dropout)

        self.activation = get_activation_fn(cfg.feedforward_activation)
        self.pre_norm = cfg.pre_norm

    @staticmethod
    def maybe_add_pos_embed(x: Tensor, pos_embed: Tensor | None) -> Tensor:
        return x if pos_embed is None else x + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]
        x = skip + self.dropout1(x)

        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x

        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]
        x = skip + self.dropout2(x)

        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x

        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x


class ACTDecoder(nn.Module):
    """多层 ACT decoder，最后固定接一个 LayerNorm。"""

    def __init__(self, cfg: ACTConfig):
        super().__init__()
        self.layers = nn.ModuleList([ACTDecoderLayer(cfg) for _ in range(cfg.n_decoder_layers)])
        self.norm = nn.LayerNorm(cfg.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed)
        return self.norm(x)


def torchvision_resnet_weights(name: str | None):
    """把字符串形式的权重名转成 torchvision 权重对象。"""
    if name is None:
        return None
    if name == "ResNet18_Weights.IMAGENET1K_V1":
        return torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    raise ValueError(f"本脚本只显式支持 ResNet18 权重，收到：{name}")


class ACT(nn.Module):
    """按 LeRobot ACT 源码复刻的网络本体。

    训练时：
        真实 action chunk -> VAE encoder -> latent z
        [z token, state token, image tokens] -> 主 encoder -> decoder -> action chunk

    推理时：
        没有真实 action chunk，因此 z 直接置零。
    """

    def __init__(self, cfg: ACTConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.use_vae:
            self.vae_encoder = ACTEncoder(cfg, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, cfg.dim_model)
            if cfg.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    cfg.robot_state_feature.shape[0], cfg.dim_model
                )
            self.vae_encoder_action_input_proj = nn.Linear(cfg.action_feature.shape[0], cfg.dim_model)
            self.vae_encoder_latent_output_proj = nn.Linear(cfg.dim_model, cfg.latent_dim * 2)

            num_vae_tokens = 1 + cfg.chunk_size
            if cfg.robot_state_feature:
                num_vae_tokens += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_vae_tokens, cfg.dim_model).unsqueeze(0),
            )

        if cfg.image_features:
            backbone_model = getattr(torchvision.models, cfg.vision_backbone)(
                replace_stride_with_dilation=[False, False, cfg.replace_final_stride_with_dilation],
                weights=torchvision_resnet_weights(cfg.pretrained_backbone_weights),
                norm_layer=FrozenBatchNorm2d,
            )
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        self.encoder = ACTEncoder(cfg)
        self.decoder = ACTDecoder(cfg)

        if cfg.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(cfg.robot_state_feature.shape[0], cfg.dim_model)

        self.encoder_latent_input_proj = nn.Linear(cfg.latent_dim, cfg.dim_model)

        if cfg.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(backbone_model.fc.in_features, cfg.dim_model, kernel_size=1)

        n_1d_tokens = 1
        if cfg.robot_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, cfg.dim_model)

        if cfg.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(cfg.dim_model // 2)

        self.decoder_pos_embed = nn.Embedding(cfg.chunk_size, cfg.dim_model)
        self.action_head = nn.Linear(cfg.dim_model, cfg.action_feature.shape[0])

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # LeRobot 只对主 encoder/decoder 做 Xavier uniform；其他 Linear/Embedding 保留 PyTorch 默认初始化。
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: dict[str, Tensor | list[Tensor]]) -> tuple[Tensor, tuple[Tensor | None, Tensor | None]]:
        if self.cfg.use_vae and self.training:
            assert ACTION in batch, "训练 VAE 目标时必须提供 action chunk"

        images = batch[OBS_IMAGES]
        assert isinstance(images, list)
        batch_size = images[0].shape[0]
        state = batch[OBS_STATE]
        assert torch.is_tensor(state)

        if self.cfg.use_vae and ACTION in batch and self.training:
            action = batch[ACTION]
            assert torch.is_tensor(action)

            cls_embed = self.vae_encoder_cls_embed.weight.repeat(batch_size, 1, 1)
            pieces = [cls_embed]

            if self.cfg.robot_state_feature:
                state_embed = self.vae_encoder_robot_state_input_proj(state).unsqueeze(1)
                pieces.append(state_embed)

            action_embed = self.vae_encoder_action_input_proj(action)
            pieces.append(action_embed)
            vae_input = torch.cat(pieces, dim=1)

            pos_embed = self.vae_encoder_pos_enc.clone().detach()
            prefix_len = 2 if self.cfg.robot_state_feature else 1
            prefix_pad = torch.full((batch_size, prefix_len), False, device=state.device)
            key_padding_mask = torch.cat([prefix_pad, batch["action_is_pad"]], dim=1)

            cls_token_out = self.vae_encoder(
                vae_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.cfg.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, self.cfg.latent_dim :]
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros([batch_size, self.cfg.latent_dim], dtype=torch.float32, device=state.device)

        encoder_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_pos = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))

        if self.cfg.robot_state_feature:
            encoder_tokens.append(self.encoder_robot_state_input_proj(state))

        for image in images:
            cam_features = self.backbone(image)["feature_map"]
            cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
            cam_features = self.encoder_img_feat_input_proj(cam_features)

            cam_features = cam_features.permute(2, 3, 0, 1).flatten(0, 1)
            cam_pos_embed = cam_pos_embed.permute(2, 3, 0, 1).flatten(0, 1)

            encoder_tokens.extend(list(cam_features))
            encoder_pos.extend(list(cam_pos_embed))

        encoder_tokens = torch.stack(encoder_tokens, dim=0)
        encoder_pos = torch.stack(encoder_pos, dim=0)

        encoder_out = self.encoder(encoder_tokens, pos_embed=encoder_pos)
        decoder_in = torch.zeros(
            (self.cfg.chunk_size, batch_size, self.cfg.dim_model),
            dtype=encoder_pos.dtype,
            device=encoder_pos.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_pos,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )
        decoder_out = decoder_out.transpose(0, 1)
        actions = self.action_head(decoder_out)
        return actions, (mu, log_sigma_x2)


class PureACTPolicy(nn.Module):
    """一个很薄的 policy 外壳，复刻 LeRobot ACTPolicy 的 forward/loss/select_action 行为。"""

    def __init__(self, cfg: ACTConfig):
        super().__init__()
        self.cfg = cfg
        self.model = ACT(cfg)
        self.action_queue = deque([], maxlen=cfg.n_action_steps)

    def get_optim_params(self) -> list[dict[str, Any]]:
        """和 LeRobot 一样，把 backbone 参数单独放进一个 param group。"""
        non_backbone = []
        backbone = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("model.backbone"):
                backbone.append(param)
            else:
                non_backbone.append(param)
        return [{"params": non_backbone}, {"params": backbone, "lr": PRESETS[HARDWARE_PRESET].lr_backbone}]

    def make_model_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor | list[Tensor]]:
        model_batch: dict[str, Tensor | list[Tensor]] = dict(batch)
        model_batch[OBS_IMAGES] = [batch[key] for key in self.cfg.image_features]
        return model_batch

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        model_batch = self.make_model_batch(batch)
        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(model_batch)

        # 这里故意完全照 LeRobot 官方 ACTPolicy.forward：
        # padding 的位置乘 0 后再对整个张量 mean，而不是只除以有效动作数量。
        l1_loss = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": float(l1_loss.detach().cpu())}
        if self.cfg.use_vae:
            assert mu_hat is not None and log_sigma_x2_hat is not None
            kld_loss = (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - log_sigma_x2_hat.exp())).sum(-1).mean()
            loss_dict["kld_loss"] = float(kld_loss.detach().cpu())
            loss = l1_loss + kld_loss * self.cfg.kl_weight
        else:
            loss = l1_loss
        return loss, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        model_batch = self.make_model_batch(batch)
        return self.model(model_batch)[0]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """最简单的 ACT action queue：队列空了就预测一段，只执行前 n_action_steps。"""
        self.eval()
        if len(self.action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.cfg.n_action_steps]
            self.action_queue.extend(actions.transpose(0, 1))
        return self.action_queue.popleft()

    def reset_action_queue(self) -> None:
        self.action_queue.clear()


# ============================== 数据读取与手写预处理 ==============================


def resolve_dataset_root() -> Path:
    for root in DATASET_ROOT_CANDIDATES:
        root = root.expanduser().resolve()
        if (root / "meta/info.json").is_file():
            return root
    candidates = "\n".join(f"  - {p.expanduser().resolve()}" for p in DATASET_ROOT_CANDIDATES)
    raise FileNotFoundError(f"找不到 LeRobot 数据集 meta/info.json，已尝试：\n{candidates}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_delta_timestamps(chunk_size: int, fps: int) -> dict[str, list[float]]:
    """让 LeRobotDataset 返回未来 action chunk，而不是单帧 action。"""
    return {ACTION: [index / fps for index in range(chunk_size)]}


def load_lerobot_dataset(preset: HardwarePreset) -> LeRobotDataset:
    dataset_root = resolve_dataset_root()
    meta = LeRobotDatasetMetadata(DATASET_REPO_ID, root=dataset_root)
    delta_timestamps = make_delta_timestamps(preset.chunk_size, meta.fps)
    dataset = LeRobotDataset(
        DATASET_REPO_ID,
        root=dataset_root,
        delta_timestamps=delta_timestamps,
        video_backend=VIDEO_BACKEND,
    )
    print(f"数据集: {dataset_root}")
    print(f"帧数: {dataset.num_frames}, episode 数: {dataset.num_episodes}, fps: {dataset.fps}")
    print(f"相机: {dataset.meta.camera_keys}")
    print(f"action chunk 时间戳: {delta_timestamps[ACTION]}")
    return dataset


def image_shape_to_chw(feature: dict[str, Any], image_resize_hw: tuple[int, int] | None) -> tuple[int, int, int]:
    shape = tuple(int(x) for x in feature["shape"])
    names = feature.get("names")
    if names and len(names) == 3 and str(names[2]).lower() in {"channel", "channels"}:
        chw = (shape[2], shape[0], shape[1])
    else:
        chw = shape
    if image_resize_hw is None:
        return chw
    return chw[0], image_resize_hw[0], image_resize_hw[1]


def build_act_config(dataset: LeRobotDataset, preset: HardwarePreset) -> ACTConfig:
    input_features: dict[str, FeatureSpec] = {}
    output_features: dict[str, FeatureSpec] = {}

    state_shape = tuple(dataset.meta.features[OBS_STATE]["shape"])
    action_shape = tuple(dataset.meta.features[ACTION]["shape"])
    input_features[OBS_STATE] = FeatureSpec("STATE", state_shape)
    output_features[ACTION] = FeatureSpec("ACTION", action_shape)

    for camera_key in dataset.meta.camera_keys:
        input_features[camera_key] = FeatureSpec(
            "VISUAL",
            image_shape_to_chw(dataset.meta.features[camera_key], preset.image_resize_hw),
        )

    return ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=preset.chunk_size,
        n_action_steps=preset.n_action_steps,
        pretrained_backbone_weights=PRETRAINED_BACKBONE_WEIGHTS,
        dim_model=preset.dim_model,
        n_heads=preset.n_heads,
        dim_feedforward=preset.dim_feedforward,
        n_encoder_layers=preset.n_encoder_layers,
        n_decoder_layers=preset.n_decoder_layers,
        latent_dim=preset.latent_dim,
        n_vae_encoder_layers=preset.n_vae_encoder_layers,
        kl_weight=10.0,
    )


def stats_tensor(dataset: LeRobotDataset, key: str, stat_name: str, device: torch.device, dtype: torch.dtype) -> Tensor:
    value = dataset.meta.stats[key][stat_name]
    if not torch.is_tensor(value):
        value = torch.tensor(value)
    return value.to(device=device, dtype=dtype)


def normalize_mean_std(x: Tensor, mean: Tensor, std: Tensor, kind: str) -> Tensor:
    """复刻 ACT 默认 NormalizationMode.MEAN_STD。"""
    if kind == "image":
        mean = mean.flatten().view(1, -1, 1, 1)
        std = std.flatten().view(1, -1, 1, 1)
    elif kind == "state":
        mean = mean.flatten().view(1, -1)
        std = std.flatten().view(1, -1)
    elif kind == "action":
        mean = mean.flatten().view(1, 1, -1)
        std = std.flatten().view(1, 1, -1)
    else:
        raise ValueError(kind)
    return (x - mean) / (std + 1e-8)


def preprocess_batch(
    raw_batch: dict[str, Any],
    dataset: LeRobotDataset,
    preset: HardwarePreset,
    device: torch.device,
) -> dict[str, Tensor]:
    """手写 LeRobot ACT preprocessor 中本任务真正需要的部分。

    处理内容：
    1. 图像 uint8 -> float32 [0,1]。
    2. 选定硬件预设需要时 resize 图像。
    3. 搬到训练设备。
    4. 对图像、state、action 做 mean/std 归一化。
    """
    batch: dict[str, Tensor] = {}

    state = raw_batch[OBS_STATE].to(device=device, dtype=torch.float32, non_blocking=True)
    action = raw_batch[ACTION].to(device=device, dtype=torch.float32, non_blocking=True)
    action_is_pad = raw_batch["action_is_pad"].to(device=device, dtype=torch.bool, non_blocking=True)

    state_mean = stats_tensor(dataset, OBS_STATE, "mean", device, state.dtype)
    state_std = stats_tensor(dataset, OBS_STATE, "std", device, state.dtype)
    action_mean = stats_tensor(dataset, ACTION, "mean", device, action.dtype)
    action_std = stats_tensor(dataset, ACTION, "std", device, action.dtype)

    batch[OBS_STATE] = normalize_mean_std(state, state_mean, state_std, "state")
    batch[ACTION] = normalize_mean_std(action, action_mean, action_std, "action")
    batch["action_is_pad"] = action_is_pad

    for camera_key in dataset.meta.camera_keys:
        image = raw_batch[camera_key]
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        else:
            image = image.float()
        if preset.image_resize_hw is not None and image.shape[-2:] != preset.image_resize_hw:
            image = F.interpolate(image, size=preset.image_resize_hw, mode="bilinear", align_corners=False)
        image = image.to(device=device, dtype=torch.float32, non_blocking=True)

        if USE_IMAGENET_STATS:
            mean = IMAGE_NET_MEAN.to(device=device, dtype=image.dtype)
            std = IMAGE_NET_STD.to(device=device, dtype=image.dtype)
        else:
            mean = stats_tensor(dataset, camera_key, "mean", device, image.dtype)
            std = stats_tensor(dataset, camera_key, "std", device, image.dtype)
        batch[camera_key] = normalize_mean_std(image, mean, std, "image")

    return batch


# ============================== 训练、日志与 checkpoint ==============================


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def print_batch_shapes(title: str, batch: dict[str, Any]) -> None:
    print(f"\n========== {title} ==========")
    for key, value in batch.items():
        if torch.is_tensor(value):
            print(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}")
        elif isinstance(value, list):
            print(f"{key}: list(len={len(value)})")
        else:
            print(f"{key}: {type(value).__name__}")


def print_model_summary(policy: PureACTPolicy, cfg: ACTConfig) -> None:
    print("\n========== ACT 结构 ==========")
    print(f"图像输入: {list(cfg.image_features.keys())}")
    print(f"状态维度: {cfg.robot_state_feature.shape[0]}")
    print(f"动作维度: {cfg.action_feature.shape[0]}")
    print(f"chunk_size: {cfg.chunk_size}, n_action_steps: {cfg.n_action_steps}")
    print(
        "Transformer: "
        f"dim_model={cfg.dim_model}, heads={cfg.n_heads}, "
        f"encoder_layers={cfg.n_encoder_layers}, decoder_layers={cfg.n_decoder_layers}, "
        f"vae_encoder_layers={cfg.n_vae_encoder_layers}"
    )
    print(f"VAE: use_vae={cfg.use_vae}, latent_dim={cfg.latent_dim}, kl_weight={cfg.kl_weight}")
    print(f"参数量: total={count_parameters(policy):,}, trainable={count_parameters(policy, True):,}")
    print("\n关键路径：")
    print("1. VAE encoder: [CLS, state, action chunk] -> mu/log_sigma_x2 -> z")
    print("2. ResNet18 layer4: 每个相机图像 -> feature map -> 1x1 conv -> 视觉 tokens")
    print("3. 主 encoder: [latent token, state token, 所有视觉 tokens] 融合")
    print("4. decoder: chunk_size 个可学习位置查询并行生成动作 token")
    print("5. action_head: 每个动作 token -> action_dim")


def make_optimizer(policy: PureACTPolicy, preset: HardwarePreset) -> torch.optim.Optimizer:
    params = policy.get_optim_params()
    return torch.optim.AdamW(params, lr=preset.lr, weight_decay=preset.weight_decay)


def cycle(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def save_checkpoint(output_dir: Path, step: int, policy: PureACTPolicy, optimizer: torch.optim.Optimizer, loss: float) -> None:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss": loss,
        "note": "minimal_act_pytorch.py 保存的纯 PyTorch ACT checkpoint；本脚本不实现 resume。",
    }
    step_path = checkpoint_dir / f"step_{step:06d}.pt"
    last_path = checkpoint_dir / "last.pt"
    torch.save(payload, step_path)
    torch.save(payload, last_path)
    print(f"[checkpoint] {step_path}")


def train() -> None:
    if HARDWARE_PRESET not in PRESETS:
        raise ValueError(f"HARDWARE_PRESET 必须是 {list(PRESETS)}，当前是 {HARDWARE_PRESET!r}")

    preset = PRESETS[HARDWARE_PRESET]
    set_seed(SEED)
    device = select_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    print(f"硬件预设: {HARDWARE_PRESET}")
    print(f"训练设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    dataset = load_lerobot_dataset(preset)
    cfg = build_act_config(dataset, preset)
    policy = PureACTPolicy(cfg).to(device)
    optimizer = make_optimizer(policy, preset)
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu", enabled=preset.use_amp and device.type == "cuda")

    print_model_summary(policy, cfg)

    debug_loader = DataLoader(dataset, batch_size=preset.batch_size, shuffle=False, num_workers=0)
    raw_debug_batch = next(iter(debug_loader))
    print_batch_shapes("原始 DataLoader batch", raw_debug_batch)
    debug_batch = preprocess_batch(raw_debug_batch, dataset, preset, device)
    print_batch_shapes("手写预处理后的 policy batch", debug_batch)
    with torch.no_grad():
        policy.train()
        debug_loss, debug_loss_dict = policy(debug_batch)
    print(f"\n第一次 debug forward: loss={float(debug_loss):.6f}, {debug_loss_dict}")

    dataloader = DataLoader(
        dataset,
        batch_size=preset.batch_size,
        shuffle=True,
        num_workers=preset.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if preset.num_workers > 0 else None,
    )
    loader_iter = cycle(dataloader)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    policy.train()
    print("\n========== 开始训练 ==========")

    for step in range(1, TRAIN_STEPS + 1):
        data_t0 = time.perf_counter()
        raw_batch = next(loader_iter)
        batch = preprocess_batch(raw_batch, dataset, preset, device)
        data_s = time.perf_counter() - data_t0

        update_t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        amp_enabled = scaler.is_enabled()
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            loss, loss_dict = policy(batch)

        if not torch.isfinite(loss):
            raise FloatingPointError("loss 出现 NaN/Inf；请关闭 AMP、降低学习率或缩小模型。")

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), preset.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), preset.grad_clip_norm)
            optimizer.step()

        update_s = time.perf_counter() - update_t0
        loss_value = float(loss.detach().cpu())

        if step == 1 or step % LOG_FREQ == 0:
            print(
                f"[step {step:06d}] "
                f"loss={loss_value:.6f} "
                f"l1={loss_dict.get('l1_loss', float('nan')):.6f} "
                f"kld={loss_dict.get('kld_loss', 0.0):.6f} "
                f"grad={float(grad_norm.detach().cpu()):.3f} "
                f"data={data_s:.3f}s "
                f"update={update_s:.3f}s"
            )

        if step % SAVE_FREQ == 0 or step == TRAIN_STEPS:
            save_checkpoint(OUTPUT_DIR, step, policy, optimizer, loss_value)

    print("\n训练结束")
    print(f"最后 checkpoint: {OUTPUT_DIR / 'checkpoints/last.pt'}")


if __name__ == "__main__":
    train()
