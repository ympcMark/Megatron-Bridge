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

"""Megatron Bridge model provider for MAGI-2 pretraining."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from megatron.bridge.models.model_provider import ModelProviderMixin
from megatron.bridge.models.transformer_config import TransformerConfig
from megatron.core.process_groups_config import ProcessGroupCollection

from megatron.bridge.diffusion.models.magi2.modeling_magi2.model import (
    Magi2ArchitectureConfig,
    Magi2Model,
)


@dataclass
class Magi2ModelProvider(TransformerConfig, ModelProviderMixin[Magi2Model]):
    """Build the public 114B MAGI-2 architecture with MCore expert parallelism."""

    num_layers: int = 40
    hidden_size: int = 3072
    num_attention_heads: int = 24
    num_query_groups: int = 24
    kv_channels: int = 128
    seq_length: int = 12
    vocab_size: int = 1
    make_vocab_size_divisible_by: int = 1
    share_embeddings_and_output_weights: bool = False
    ffn_hidden_size: int = 8192
    num_moe_experts: int = 3072
    moe_ffn_hidden_size: int = 1280
    params_dtype: torch.dtype = torch.bfloat16
    pipeline_dtype: torch.dtype = torch.bfloat16
    bf16: bool = True
    add_bias_linear: bool = False
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    normalization: str = "RMSNorm"
    layernorm_epsilon: float = 1e-6
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 8
    expert_tensor_parallel_size: int = 1
    sequence_parallel: bool = False
    moe_grouped_gemm: bool = True
    moe_token_dispatcher_type: str = "alltoall"
    moe_router_load_balancing_type: str = "none"
    moe_router_topk: int = 6
    moe_router_score_function: str = "sigmoid"
    moe_router_dtype: str = "fp32"
    moe_router_enable_expert_bias: bool = True
    moe_router_bias_update_rate: float = 1e-3
    moe_aux_loss_coeff: float = 0.0
    # MAGI-2 uses the public modality-normalized mHC implementation in model.py,
    # not MCore's transformer-layer wrapper.
    enable_mhc_connections: bool = False
    mhc_num_residual_streams: int = 4
    mhc_init_gating_factor: float = 0.01
    mhc_sinkhorn_iterations: int = 20
    use_fused_mhc: bool = False
    use_cpu_initialization: bool = False

    video_in_channels: int = 48
    audio_in_channels: int = 64
    text_in_channels: int = 5120
    intermediate_factor: int = 4
    mm_layers: tuple[int, ...] = (0, 1, 38, 39)
    moe_layers: tuple[int, ...] = tuple(range(2, 38))
    moe_num_heads: int = 12
    moe_num_experts_per_head: int = 256
    moe_top_k: int = 6
    moe_expert_intermediate_size: int = 1280
    shared_expert_intermediate_size: int = 1280
    modality_expert_intermediate_size: int = 1280
    route_scale: float = 4.9
    route_norm: bool = True
    route_norm_eps: float = 1e-12
    sink_token_num: int = 1
    require_original_size: bool = True

    def architecture_config(self) -> Magi2ArchitectureConfig:
        """Construct the model-only architecture configuration."""
        return Magi2ArchitectureConfig(
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            head_dim=self.kv_channels,
            num_query_groups=self.num_query_groups,
            video_in_channels=self.video_in_channels,
            audio_in_channels=self.audio_in_channels,
            text_in_channels=self.text_in_channels,
            intermediate_factor=self.intermediate_factor,
            mm_layers=self.mm_layers,
            moe_layers=self.moe_layers,
            moe_num_heads=self.moe_num_heads,
            moe_num_experts_per_head=self.moe_num_experts_per_head,
            moe_top_k=self.moe_top_k,
            moe_expert_intermediate_size=self.moe_expert_intermediate_size,
            shared_expert_intermediate_size=self.shared_expert_intermediate_size,
            modality_expert_intermediate_size=self.modality_expert_intermediate_size,
            route_scale=self.route_scale,
            route_norm=self.route_norm,
            route_norm_eps=self.route_norm_eps,
            sink_token_num=self.sink_token_num,
            mhc_num_streams=self.mhc_num_residual_streams,
        )

    def finalize(self) -> None:
        """Validate supported parallelism and the published architecture."""
        architecture = self.architecture_config()
        if self.tensor_model_parallel_size != 1:
            raise ValueError("MAGI-2 correctness attention currently requires TP=1")
        if self.pipeline_model_parallel_size != 1:
            raise ValueError("MAGI-2 public mHC currently requires PP=1")
        if self.context_parallel_size != 1:
            raise ValueError("MAGI-2 correctness attention currently requires CP=1")
        if architecture.flattened_num_experts % self.expert_model_parallel_size:
            raise ValueError("flattened experts must be divisible by expert parallel size")
        if self.num_moe_experts != architecture.flattened_num_experts:
            raise ValueError("num_moe_experts must equal the flattened MAGI-2 expert count")
        if self.moe_top_k != self.moe_router_topk:
            raise ValueError("MAGI-2 and MCore router top-k values differ")
        if self.require_original_size and architecture != Magi2ArchitectureConfig():
            raise ValueError("require_original_size=True rejects reduced architecture overrides")
        self.sequence_parallel = False
        super().finalize()

    def provide(
        self,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> Magi2Model:
        """Instantiate a non-pipelined MAGI-2 model on the active construction device."""
        if pre_process is False or post_process is False or vp_stage is not None:
            raise ValueError("MAGI-2 currently supports a single non-virtual pipeline stage")
        pg_collection = getattr(self, "_pg_collection", None)
        if not isinstance(pg_collection, ProcessGroupCollection):
            raise RuntimeError("MAGI-2 provider requires initialized process groups")
        return Magi2Model(self, self.architecture_config(), pg_collection)


__all__ = ["Magi2ModelProvider"]
