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

"""Original-size MAGI-2 pretraining recipe for 64 H100 GPUs."""

from __future__ import annotations

from megatron.bridge.diffusion.models.magi2.data import Magi2LatentDatasetConfig
from megatron.bridge.diffusion.models.magi2.provider import Magi2ModelProvider
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.optimizer_utils import (
    distributed_fused_adam_with_cosine_annealing,
)
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import get_mixed_precision_config


def magi2_114b_pretrain_64gpu_h100_bf16_config() -> ConfigContainer:
    """Return the first original-size MAGI-2 expert-parallel recipe."""
    cfg = _pretrain_common()
    cfg.model = Magi2ModelProvider()
    cfg.model.expert_model_parallel_size = 64
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.dataset = Magi2LatentDatasetConfig()
    cfg.train.train_iters = 50
    cfg.train.global_batch_size = 64
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 10
    cfg.validation.eval_interval = None
    cfg.validation.eval_iters = 0
    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=5,
        lr_decay_iters=50,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        weight_decay=0.01,
        max_lr=1e-4,
        min_lr=1e-5,
        start_weight_decay=0.01,
        end_weight_decay=0.01,
        lr_decay_style="cosine",
    )
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.average_in_collective = True
    cfg.logger.log_interval = 1
    cfg.checkpoint.save_interval = 25
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True
    cfg.checkpoint.async_save = False
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = 1
    cfg.rng.seed = 42
    cfg.mixed_precision = get_mixed_precision_config("bf16_mixed")
    return cfg


__all__ = ["magi2_114b_pretrain_64gpu_h100_bf16_config"]
