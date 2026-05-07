#!/usr/bin/env python3
"""Train a LeRobot ACT policy on the UR5e cylinder-to-box dataset.

The script builds a LeRobot train_config JSON from a small set of practical
arguments, then launches the official LeRobot trainer. Keeping the last step in
LeRobot's trainer preserves the normal checkpoint, processor, and resume layout.
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
    # A6000 显存余量大，可以更接近 LeRobot ACT 默认容量。
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
    parser = argparse.ArgumentParser(
        description=(
            "Generate a LeRobot ACT training config for the UR5e cylinder-to-box "
            "dataset and launch the official LeRobot trainer."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Training output directory. Defaults to outputs/train/ur5e_act_<preset>.",
    )
    parser.add_argument(
        "--preset",
        choices=["auto", "1660ti", "a6000", "custom"],
        default="auto",
        help="Hardware preset. Use custom when you want every value to come from CLI arguments.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device passed to the policy.")
    parser.add_argument("--resume-config", type=Path, default=None, help="Path to a saved train_config.json.")
    parser.add_argument("--overwrite-output", action="store_true", help="Delete output-dir before training.")
    parser.add_argument("--dry-run", action="store_true", help="Only print the generated config and command.")
    parser.add_argument("--config-output", type=Path, default=None, help="Where to write the generated config.")
    parser.add_argument(
        "--skip-dataset-schema-check",
        action="store_true",
        help=(
            "Skip validation that observation.state/action names match the current UR5e schema. "
            "Normally keep this enabled to catch legacy gripper-delta datasets."
        ),
    )

    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--save-freq", type=int, default=None)
    parser.add_argument("--log-freq", type=int, default=None)
    parser.add_argument("--eval-freq", type=int, default=0, help="Keep 0 for real-world offline data.")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--video-backend",
        default="pyav",
        choices=["pyav", "torchcodec", "video_reader"],
        help=(
            "Video decoder backend for LeRobotDataset. pyav is the safest default in this environment; "
            "torchcodec can be faster when its FFmpeg libraries match your PyTorch install."
        ),
    )

    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument("--dim-model", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--dim-feedforward", type=int, default=None)
    parser.add_argument("--n-encoder-layers", type=int, default=None)
    parser.add_argument("--n-decoder-layers", type=int, default=None)
    parser.add_argument("--n-vae-encoder-layers", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--kl-weight", type=float, default=10.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--optimizer-lr", type=float, default=1e-5)
    parser.add_argument("--optimizer-lr-backbone", type=float, default=1e-5)
    parser.add_argument("--optimizer-weight-decay", type=float, default=1e-4)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-vae", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--pretrained-backbone",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use torchvision ImageNet ResNet18 weights. Disable if the machine cannot download weights.",
    )
    parser.add_argument(
        "--use-imagenet-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ImageNet normalization stats for cameras, matching LeRobot defaults.",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default="lerobot")
    parser.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"])

    args, unknown = parser.parse_known_args()
    return args, unknown


def select_preset(name: str) -> tuple[str, TrainPreset]:
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
    return fallback if value is None else value


def ensure_lerobot_imports():
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
    draccus, DatasetConfig, EvalConfig, WandBConfig, TrainPipelineConfig, ACTConfig = ensure_lerobot_imports()

    output_dir = args.output_dir or Path("outputs/train") / f"ur5e_act_{preset_name}"
    pretrained_backbone = choose(args.pretrained_backbone, preset.pretrained_backbone)
    pretrained_backbone_weights = "ResNet18_Weights.IMAGENET1K_V1" if pretrained_backbone else None

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=args.repo_id,
            root=str(args.dataset_root),
            use_imagenet_stats=args.use_imagenet_stats,
            video_backend=args.video_backend,
        ),
        policy=ACTConfig(
            device=args.device,
            use_amp=choose(args.use_amp, preset.use_amp),
            push_to_hub=False,
            chunk_size=choose(args.chunk_size, preset.chunk_size),
            n_action_steps=choose(args.n_action_steps, preset.n_action_steps),
            vision_backbone="resnet18",
            pretrained_backbone_weights=pretrained_backbone_weights,
            dim_model=choose(args.dim_model, preset.dim_model),
            n_heads=choose(args.n_heads, preset.n_heads),
            dim_feedforward=choose(args.dim_feedforward, preset.dim_feedforward),
            n_encoder_layers=choose(args.n_encoder_layers, preset.n_encoder_layers),
            n_decoder_layers=choose(args.n_decoder_layers, preset.n_decoder_layers),
            use_vae=args.use_vae,
            latent_dim=args.latent_dim,
            n_vae_encoder_layers=choose(args.n_vae_encoder_layers, preset.n_vae_encoder_layers),
            dropout=args.dropout,
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        with draccus.config_type("json"):
            draccus.dump(cfg, stream, indent=2)


def print_json(path: Path) -> None:
    with path.open("r") as stream:
        payload = json.load(stream)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def validate_dataset_schema(dataset_root: Path) -> dict[str, Any]:
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
    with config_path.open("r") as stream:
        payload = json.load(stream)
    root = payload.get("dataset", {}).get("root")
    if root is None:
        return None
    return Path(root).expanduser().resolve()


def print_dataset_summary(info: dict[str, Any]) -> None:
    print(
        "Dataset schema OK: "
        f"episodes={info.get('total_episodes')}, "
        f"frames={info.get('total_frames')}, "
        f"fps={info.get('fps')}, "
        "action[-1]=gripper_target_opening"
    )


def main() -> int:
    args, unknown_lerobot_args = parse_args()

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
