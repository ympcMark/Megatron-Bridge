# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run original-size MAGI-2 pretraining or a reduced integration smoke test."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from megatron.core.models.magi2 import (
    Magi2Model,
    Magi2MultiHeadTopKRouter,
    load_magi2_official_safetensors,
)
from megatron.core.optimizer_param_scheduler import get_canonical_lr_for_logging

from megatron.bridge.diffusion.models.magi2.magi2_step import Magi2ForwardStep
from megatron.bridge.recipes.magi2.h100.magi2 import (
    magi2_114b_pretrain_64gpu_h100_bf16_config,
)
from megatron.bridge.training.callbacks import CallbackContext, CallbackManager
from megatron.bridge.training.pretrain import pretrain


def parse_args() -> argparse.Namespace:
    """Parse reproducible MAGI-2 training arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-iters", type=int, default=50)
    parser.add_argument(
        "--lr-decay-iters",
        type=int,
        help="Schedule horizon; defaults to train-iters and may extend past a checkpoint smoke run.",
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--loss-output", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path)
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument(
        "--exit-interval",
        type=int,
        help="Save and exit after this many iterations while preserving the configured training horizon.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Load and train without writing a final checkpoint (useful for resume verification).",
    )
    parser.add_argument("--load", type=Path)
    parser.add_argument(
        "--official-checkpoint",
        type=Path,
        help="Initialize the native MCore model from the public MAGI-2 safetensors directory.",
    )
    parser.add_argument("--router-bias-ema", type=float, default=0.999)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _unwrap_modules(model: torch.nn.Module):
    yield from model.modules()


def _load_official_checkpoint(
    checkpoint_dir: Path,
    model_chunks: list[Magi2Model],
) -> list[Magi2Model]:
    """Load public MAGI-2 tensors before mixed-precision and DDP wrapping."""
    if len(model_chunks) != 1:
        raise ValueError("official MAGI-2 loading does not support virtual pipeline model chunks")
    report = load_magi2_official_safetensors(model_chunks[0], checkpoint_dir, strict=True)
    if torch.distributed.get_rank() == 0:
        print(
            "Loaded official MAGI-2 checkpoint: "
            f"source_tensors={len(report.consumed_source_keys)}, "
            f"target_tensors={len(report.populated_target_keys)}"
        )
    return model_chunks


def _step_callback(
    output: Path,
    router_bias_ema: float,
    global_batch_size: int,
    tokens_per_sample: int,
    context: CallbackContext,
) -> None:
    for model_chunk in context.model:
        for module in _unwrap_modules(model_chunk):
            if isinstance(module, Magi2MultiHeadTopKRouter):
                module.update_expert_bias_ema(router_bias_ema)
    now = time.perf_counter()
    step_time = now - context.user_state["magi2_last_step_time"]
    context.user_state["magi2_last_step_time"] = now
    rank_stats = torch.tensor(
        [
            torch.cuda.max_memory_allocated(),
            torch.cuda.max_memory_reserved(),
            step_time,
        ],
        dtype=torch.float64,
        device=torch.cuda.current_device(),
    )
    stats_sum = rank_stats.clone()
    stats_min = rank_stats.clone()
    stats_max = rank_stats.clone()
    torch.distributed.all_reduce(stats_sum, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(stats_min, op=torch.distributed.ReduceOp.MIN)
    torch.distributed.all_reduce(stats_max, op=torch.distributed.ReduceOp.MAX)
    torch.cuda.reset_peak_memory_stats()
    if torch.distributed.get_rank() != 0:
        return
    world_size = torch.distributed.get_world_size()
    learning_rate = get_canonical_lr_for_logging(context.optimizer.param_groups)
    losses = {name: value.detach().float().cpu().tolist() for name, value in sorted((context.loss_dict or {}).items())}
    row = {
        "step": context.state.train_state.step + 1,
        "grad_norm": context.grad_norm,
        "grad_norm_finite": context.grad_norm is None or math.isfinite(float(context.grad_norm)),
        "skipped_iteration": bool(context.skipped_iter),
        "learning_rate": None if learning_rate is None else float(learning_rate),
        "losses": losses,
        "losses_finite": all(
            torch.isfinite(value.detach()).all().item() for value in (context.loss_dict or {}).values()
        ),
        "step_time_seconds": {
            "min": stats_min[2].item(),
            "mean": stats_sum[2].item() / world_size,
            "max": stats_max[2].item(),
        },
        "global_samples_per_second": global_batch_size / stats_max[2].item(),
        "global_tokens_per_second": global_batch_size * tokens_per_sample / stats_max[2].item(),
        "memory_allocated_gib": {
            "min": stats_min[0].item() / 1024**3,
            "mean": stats_sum[0].item() / world_size / 1024**3,
            "max": stats_max[0].item() / 1024**3,
        },
        "memory_reserved_gib": {
            "min": stats_min[1].item() / 1024**3,
            "mean": stats_sum[1].item() / world_size / 1024**3,
            "max": stats_max[1].item() / 1024**3,
        },
    }
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _apply_smoke_overrides(cfg, world_size: int) -> None:
    cfg.model.require_original_size = False
    cfg.model.num_layers = 2
    cfg.model.hidden_size = 64
    cfg.model.num_attention_heads = 4
    cfg.model.num_query_groups = 4
    cfg.model.kv_channels = 16
    cfg.model.seq_length = 6
    cfg.model.ffn_hidden_size = 128
    cfg.model.magi2_video_in_channels = 8
    cfg.model.magi2_audio_in_channels = 8
    cfg.model.magi2_text_in_channels = 16
    cfg.model.magi2_intermediate_factor = 2
    cfg.model.magi2_mm_layers = (0,)
    cfg.model.magi2_moe_layers = (1,)
    cfg.model.magi2_moe_num_heads = 4
    cfg.model.magi2_moe_num_experts_per_head = 4
    cfg.model.magi2_moe_top_k = 2
    cfg.model.moe_router_topk = 2
    cfg.model.magi2_moe_expert_intermediate_size = 32
    cfg.model.moe_ffn_hidden_size = 32
    cfg.model.magi2_shared_expert_intermediate_size = 32
    cfg.model.magi2_modality_expert_intermediate_size = 32
    cfg.model.magi2_mhc_num_streams = 2
    cfg.model.num_moe_experts = 16
    cfg.model.expert_model_parallel_size = world_size
    cfg.dataset.video_in_channels = 8
    cfg.dataset.audio_in_channels = 8
    cfg.dataset.text_in_channels = 16
    cfg.dataset.video_tokens = 2
    cfg.dataset.audio_tokens = 2
    cfg.dataset.text_tokens = 1
    cfg.dataset.time_tokens = 1
    cfg.train.global_batch_size = world_size


def main() -> None:
    """Configure callbacks, checkpointing, and execute Bridge pretraining."""
    args = parse_args()
    if args.train_iters <= 0 or args.save_interval <= 0:
        raise ValueError("train-iters and save-interval must be positive")
    if args.exit_interval is not None and args.exit_interval <= 0:
        raise ValueError("exit-interval must be positive")
    if args.load is not None and args.official_checkpoint is not None:
        raise ValueError("--load and --official-checkpoint are mutually exclusive")
    if args.official_checkpoint is not None and not args.official_checkpoint.is_dir():
        raise ValueError(f"official checkpoint directory does not exist: {args.official_checkpoint}")
    if args.lr_decay_iters is not None and args.lr_decay_iters < args.train_iters:
        raise ValueError("lr-decay-iters must be greater than or equal to train-iters")
    if not 0.0 <= args.router_bias_ema < 1.0:
        raise ValueError("router-bias-ema must be in [0, 1)")
    world_size = int(os.environ["WORLD_SIZE"])
    if not args.smoke and world_size != 64:
        raise ValueError("original-size recipe requires WORLD_SIZE=64")
    if not args.no_save:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.loss_output.parent.mkdir(parents=True, exist_ok=True)
    if args.loss_output.exists():
        raise ValueError(f"loss output already exists: {args.loss_output}")

    cfg = magi2_114b_pretrain_64gpu_h100_bf16_config()
    if args.smoke:
        _apply_smoke_overrides(cfg, world_size)
    cfg.train.train_iters = args.train_iters
    cfg.train.exit_interval = args.exit_interval
    cfg.scheduler.lr_decay_iters = args.lr_decay_iters or args.train_iters
    cfg.checkpoint.save = None if args.no_save else str(args.checkpoint_dir.resolve())
    cfg.checkpoint.save_interval = args.save_interval
    if args.load is not None:
        cfg.checkpoint.load = str(args.load.resolve())
        cfg.checkpoint.load_optim = True
        cfg.checkpoint.load_rng = True
    else:
        cfg.checkpoint.load = None
    if args.official_checkpoint is not None:
        official_checkpoint = args.official_checkpoint.resolve()
        cfg.model.register_pre_wrap_hook(
            lambda model_chunks: _load_official_checkpoint(official_checkpoint, model_chunks)
        )
    cfg.logger.tensorboard_dir = str(args.tensorboard_dir.resolve()) if args.tensorboard_dir is not None else None
    if int(os.environ["RANK"]) == 0:
        architecture = {
            "num_layers": cfg.model.num_layers,
            "hidden_size": cfg.model.hidden_size,
            "num_attention_heads": cfg.model.num_attention_heads,
            "video_in_channels": cfg.model.magi2_video_in_channels,
            "audio_in_channels": cfg.model.magi2_audio_in_channels,
            "text_in_channels": cfg.model.magi2_text_in_channels,
            "mm_layers": cfg.model.magi2_mm_layers,
            "moe_layers": cfg.model.magi2_moe_layers,
            "moe_num_heads": cfg.model.magi2_moe_num_heads,
            "moe_num_experts_per_head": cfg.model.magi2_moe_num_experts_per_head,
            "moe_top_k": cfg.model.magi2_moe_top_k,
            "mhc_num_streams": cfg.model.magi2_mhc_num_streams,
        }
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model": "MAGI-2-preview",
            "architecture": architecture,
            "backends": {
                "model": "native megatron.core.models.magi2.Magi2Model",
                "transformer": "MCore TransformerBlock with MAGI-2 dense and MoE layer specs",
                "attention": "native differentiable MAGI-2 variable-length correctness attention",
                "experts": "MCore all-to-all MoELayer with Transformer Engine grouped GEMM",
                "router": "native MAGI-2 head-constrained sigmoid top-k in fp32",
            },
            "parameter_count": cfg.model.magi2_parameter_count,
            "parameter_count_breakdown": cfg.model.magi2_parameter_count_breakdown(),
            "parallelism": {
                "world_size": world_size,
                "tensor": cfg.model.tensor_model_parallel_size,
                "pipeline": cfg.model.pipeline_model_parallel_size,
                "context": cfg.model.context_parallel_size,
                "expert": cfg.model.expert_model_parallel_size,
            },
            "training": {
                "train_iters": args.train_iters,
                "lr_decay_iters": cfg.scheduler.lr_decay_iters,
                "micro_batch_size": cfg.train.micro_batch_size,
                "global_batch_size": cfg.train.global_batch_size,
                "precision": "bf16_mixed",
                "checkpoint_format": cfg.checkpoint.ckpt_format,
                "save_interval": cfg.checkpoint.save_interval,
                "save": cfg.checkpoint.save,
                "load": None if args.load is None else str(args.load.resolve()),
                "official_checkpoint": (
                    None if args.official_checkpoint is None else str(args.official_checkpoint.resolve())
                ),
            },
            "data": {
                "kind": "deterministic synthetic latent integration data",
                "seed": cfg.dataset.seed,
                "tokens_per_sample": cfg.dataset.video_tokens
                + cfg.dataset.audio_tokens
                + cfg.dataset.text_tokens
                + cfg.dataset.time_tokens,
                "video_channels": cfg.dataset.video_in_channels,
                "audio_channels": cfg.dataset.audio_in_channels,
                "text_channels": cfg.dataset.text_in_channels,
            },
            "software": {
                "torch": torch.__version__,
                "container_image": os.environ.get("CONTAINER_IMAGE"),
            },
        }
        manifest_path = args.loss_output.with_suffix(args.loss_output.suffix + ".manifest.json")
        if manifest_path.exists():
            raise ValueError(f"run manifest already exists: {manifest_path}")
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    callbacks = CallbackManager()
    callbacks.register(
        "on_train_start",
        lambda context: context.user_state.update({"magi2_last_step_time": time.perf_counter()}),
    )
    callbacks.register(
        "on_train_step_end",
        lambda context: _step_callback(
            args.loss_output.resolve(),
            args.router_bias_ema,
            cfg.train.global_batch_size,
            cfg.dataset.video_tokens + cfg.dataset.audio_tokens + cfg.dataset.text_tokens + cfg.dataset.time_tokens,
            context,
        ),
    )
    pretrain(config=cfg, forward_step_func=Magi2ForwardStep(), callbacks=callbacks)


if __name__ == "__main__":
    main()
