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

"""Megatron-Core training model for the public MAGI-2 preview architecture.

The module follows the public SandAI architecture and tensor dimensions while
using Megatron-Core hyper-connections and distributed grouped experts.  The
attention implementation is a differentiable correctness path intended for
initial pretraining bring-up; a fused variable-length attention backend can be
selected after numerical parity is established.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.diffusion.models.magi2.modeling_magi2.distributed_multi_head_moe import (
    Magi2DistributedMultiHeadMoE,
)


class Magi2Modality(IntEnum):
    """Token modality identifiers used by the public model."""

    VIDEO = 0
    AUDIO = 1
    TEXT = 2
    TIME = 3


@dataclass(frozen=True)
class Magi2ArchitectureConfig:
    """Architecture fields from ``configs/magi2_preview.json``."""

    num_layers: int = 40
    hidden_size: int = 3072
    head_dim: int = 128
    num_query_groups: int = 24
    video_in_channels: int = 48
    audio_in_channels: int = 64
    text_in_channels: int = 5120
    intermediate_factor: int = 4
    mm_layers: tuple[int, ...] = (0, 1, 38, 39)
    moe_layers: tuple[int, ...] = field(default_factory=lambda: tuple(range(2, 38)))
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
    mhc_num_streams: int = 4

    def __post_init__(self) -> None:
        """Validate topology-defining dimensions."""
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.hidden_size <= 0 or self.head_dim <= 0:
            raise ValueError("hidden_size and head_dim must be positive")
        if self.hidden_size % self.head_dim != 0:
            raise ValueError("hidden_size must be divisible by head_dim")
        if self.num_query_groups != self.hidden_size // self.head_dim:
            raise ValueError("MAGI-2 preview requires one KV head per query head")
        if self.hidden_size % self.moe_num_heads != 0:
            raise ValueError("hidden_size must be divisible by moe_num_heads")
        if not 0 < self.moe_top_k <= self.moe_num_experts_per_head:
            raise ValueError("moe_top_k must be in [1, moe_num_experts_per_head]")
        layer_ids = set(self.mm_layers) | set(self.moe_layers)
        if any(layer < 0 or layer >= self.num_layers for layer in layer_ids):
            raise ValueError("layer indices must be within [0, num_layers)")
        if set(self.mm_layers) & set(self.moe_layers):
            raise ValueError("mm_layers and moe_layers must be disjoint")
        if self.mhc_num_streams <= 0:
            raise ValueError("mhc_num_streams must be positive")

    @property
    def num_attention_heads(self) -> int:
        """Number of query heads."""
        return self.hidden_size // self.head_dim

    @property
    def flattened_num_experts(self) -> int:
        """Number of globally addressable ``(head, expert)`` pairs."""
        return self.moe_num_heads * self.moe_num_experts_per_head

    @property
    def adapter_width(self) -> int:
        """Width of the flattened residual streams."""
        return self.hidden_size * self.mhc_num_streams

    def parameter_count_breakdown(self) -> dict[str, int]:
        """Return the global parameter count implied by this implementation."""
        hidden = self.hidden_size
        adapter = self.adapter_width
        expert_hidden = hidden // self.moe_num_heads
        dense_intermediate = (
            max(128, int(hidden * self.intermediate_factor * 2 / 3) // 128 * 128)
        )
        mapping_width = 2 * self.mhc_num_streams + self.mhc_num_streams**2
        mhc_branch = (
            adapter * mapping_width
            + 3
            + 2 * self.mhc_num_streams
            + self.mhc_num_streams**2
        )
        routed_experts = (
            len(self.moe_layers)
            * self.flattened_num_experts
            * 3
            * expert_hidden
            * self.moe_expert_intermediate_size
        )
        adapters = adapter * (
            self.video_in_channels
            + self.audio_in_channels
            + self.text_in_channels
            + 3
        ) + 2 * adapter + adapter * (self.video_in_channels + self.audio_in_channels)
        attention_and_mhc = 0
        mlps_and_routers = 0
        for layer in range(self.num_layers):
            modalities = 3 if layer in self.mm_layers else 1
            attention_and_mhc += (
                modalities * hidden
                + 2 * modalities * self.head_dim
                + modalities * self.num_attention_heads * hidden
                + modalities * 4 * hidden * hidden
                + self.sink_token_num * self.num_attention_heads
                + adapter * modalities
                + 2 * mhc_branch
            )
            if layer in self.moe_layers:
                mlps_and_routers += (
                    3 * hidden
                    + 2 * hidden * hidden
                    + self.flattened_num_experts * expert_hidden
                    + 3 * hidden * self.shared_expert_intermediate_size
                    + 9 * hidden * self.modality_expert_intermediate_size
                )
            else:
                mlps_and_routers += modalities * (
                    hidden + 3 * hidden * dense_intermediate
                )
        return {
            "adapters": adapters,
            "attention_and_mhc": attention_and_mhc,
            "routed_experts": routed_experts,
            "other_mlp_and_router": mlps_and_routers,
        }

    @property
    def parameter_count(self) -> int:
        """Global parameter count, counting each expert exactly once."""
        return sum(self.parameter_count_breakdown().values())


class Magi2ModalityDispatcher:
    """Stable token permutation and group metadata for modality-specific weights."""

    def __init__(self, modality_mapping: Tensor, num_modalities: int = 3) -> None:
        if modality_mapping.ndim != 1:
            raise ValueError("modality_mapping must be one-dimensional")
        if modality_mapping.numel() == 0:
            raise ValueError("modality_mapping must contain at least one token")
        if modality_mapping.min() < 0 or modality_mapping.max() >= num_modalities:
            raise ValueError("modality_mapping contains an unsupported modality")
        self.num_modalities = num_modalities
        self.permute_mapping = torch.argsort(modality_mapping, stable=True)
        self.inverse_mapping = torch.argsort(self.permute_mapping)
        permuted_mapping = modality_mapping.index_select(0, self.permute_mapping)
        self.group_sizes = torch.bincount(permuted_mapping, minlength=num_modalities)

    def permute(self, value: Tensor) -> Tensor:
        """Move equal-modality tokens into contiguous groups."""
        return value.index_select(0, self.permute_mapping)

    def inverse_permute(self, value: Tensor) -> Tensor:
        """Restore original packed-sequence token order."""
        return value.index_select(0, self.inverse_mapping)

    def split(self, value: Tensor) -> tuple[Tensor, ...]:
        """Split a permuted tensor into modality groups."""
        sizes = tuple(int(size) for size in self.group_sizes.cpu().tolist())
        return torch.split(value, sizes, dim=0)


class Magi2ModalityLinear(nn.Module):
    """Dense linear projection with one weight set per modality."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        num_modalities: int = 1,
        bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_modalities = num_modalities
        self.weight = nn.Parameter(
            torch.empty(num_modalities, out_features, in_features, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(num_modalities, out_features, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize every modality projection independently."""
        for weight in self.weight.unbind(0):
            nn.init.xavier_uniform_(weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        value: Tensor,
        dispatcher: Optional[Magi2ModalityDispatcher] = None,
    ) -> Tensor:
        """Apply the matching modality projection to each contiguous group."""
        if self.num_modalities == 1:
            return F.linear(value, self.weight[0], None if self.bias is None else self.bias[0])
        if dispatcher is None:
            raise ValueError("dispatcher is required for modality-specific linear layers")
        groups = dispatcher.split(value)
        outputs = [
            F.linear(group, self.weight[index], None if self.bias is None else self.bias[index])
            for index, group in enumerate(groups)
        ]
        return torch.cat(outputs, dim=0)


class Magi2RMSNorm(nn.Module):
    """Zero-centered RMSNorm with optional modality-specific weights."""

    def __init__(
        self,
        hidden_size: int,
        *,
        num_modalities: int = 1,
        eps: float = 1e-6,
        out_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_modalities = num_modalities
        self.eps = eps
        self.out_dtype = out_dtype
        self.weight = nn.Parameter(torch.zeros(num_modalities, hidden_size, dtype=torch.float32))
        self._keep_in_float32_parameter_names = ("weight",)

    @staticmethod
    def _normalize(
        value: Tensor,
        weight: Tensor,
        eps: float,
        out_dtype: Optional[torch.dtype],
    ) -> Tensor:
        output = value.float() * torch.rsqrt(value.float().square().mean(dim=-1, keepdim=True) + eps)
        return (output * (weight + 1.0)).to(out_dtype or value.dtype)

    def forward(
        self,
        value: Tensor,
        dispatcher: Optional[Magi2ModalityDispatcher] = None,
    ) -> Tensor:
        """Normalize over the last dimension."""
        if value.shape[-1] != self.hidden_size:
            raise ValueError(f"expected last dimension {self.hidden_size}")
        if self.num_modalities == 1:
            return self._normalize(value, self.weight[0], self.eps, self.out_dtype)
        if dispatcher is None:
            raise ValueError("dispatcher is required for modality-specific RMSNorm")
        groups = dispatcher.split(value)
        outputs = [
            self._normalize(group, self.weight[index], self.eps, self.out_dtype)
            for index, group in enumerate(groups)
        ]
        return torch.cat(outputs, dim=0)


class Magi2HyperConnectionBranch(nn.Module):
    """One public MAGI-2 manifold-constrained hyper-connection branch."""

    def __init__(self, hidden_size: int, num_streams: int, alpha_init: float = 0.01) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_streams = num_streams
        self.matmul_scale = 1.0 / math.sqrt(num_streams * hidden_size)
        mapping_width = 2 * num_streams + num_streams * num_streams
        self.alpha_pre = nn.Parameter(torch.full((1,), alpha_init, dtype=torch.float32))
        self.alpha_post = nn.Parameter(torch.full((1,), alpha_init, dtype=torch.float32))
        self.alpha_res = nn.Parameter(torch.full((1,), alpha_init, dtype=torch.float32))
        self.bias_pre = nn.Parameter(torch.zeros(num_streams, dtype=torch.float32))
        self.bias_post = nn.Parameter(torch.zeros(num_streams, dtype=torch.float32))
        self.bias_res = nn.Parameter(
            torch.zeros(num_streams, num_streams, dtype=torch.float32)
        )
        self.phi_fused = nn.Parameter(
            torch.empty(num_streams * hidden_size, mapping_width, dtype=torch.float32)
        )
        self._keep_in_float32_parameter_names = (
            "alpha_pre",
            "alpha_post",
            "alpha_res",
            "bias_pre",
            "bias_post",
            "bias_res",
            "phi_fused",
        )
        nn.init.xavier_uniform_(self.phi_fused)

    @staticmethod
    def _sinkhorn(logits: Tensor, iterations: int = 20, eps: float = 1e-12) -> Tensor:
        matrix = torch.exp(logits - logits.amax(dim=(-2, -1), keepdim=True))
        for _ in range(iterations):
            matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + eps)
            matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + eps)
        return matrix

    def pre(self, hidden_states: Tensor, normalized_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute mappings and aggregate ``n`` residual streams to one."""
        tokens = hidden_states.shape[0]
        streams = hidden_states.reshape(tokens, self.num_streams, self.hidden_size)
        projected = normalized_states.float() @ self.phi_fused
        raw_pre, raw_post, raw_res = torch.split(
            projected,
            [self.num_streams, self.num_streams, self.num_streams**2],
            dim=-1,
        )
        h_pre = torch.sigmoid(
            self.alpha_pre * self.matmul_scale * raw_pre + self.bias_pre
        ).to(hidden_states.dtype)
        h_post = (
            2.0
            * torch.sigmoid(
                self.alpha_post * self.matmul_scale * raw_post + self.bias_post
            )
        ).to(hidden_states.dtype)
        h_res_logits = raw_res.reshape(tokens, self.num_streams, self.num_streams)
        h_res = self._sinkhorn(
            self.alpha_res * self.matmul_scale * h_res_logits + self.bias_res
        ).to(hidden_states.dtype)
        aggregated = torch.einsum("tn,tnc->tc", h_pre, streams)
        return aggregated, h_post, h_res

    def merge(
        self,
        residual: Tensor,
        output: Tensor,
        h_post: Tensor,
        h_res: Tensor,
    ) -> Tensor:
        """Mix residual streams and inject the single-stream branch output."""
        residual_streams = residual.reshape(
            residual.shape[0], self.num_streams, self.hidden_size
        )
        mixed_residual = torch.einsum("tij,tjc->tic", h_res, residual_streams)
        expanded_output = torch.einsum("tn,tc->tnc", h_post, output)
        return (mixed_residual + expanded_output).reshape(residual.shape)


def magi2_quick_geglu(value: Tensor) -> Tensor:
    """Apply the public interleaved QuickGEGLU7 activation."""
    output_dtype = value.dtype
    value = value.float()
    gate = value[..., ::2].clamp(max=7.0)
    linear = value[..., 1::2].clamp(min=-7.0, max=7.0)
    return (gate * torch.sigmoid(1.702 * gate) * (linear + 1.0)).to(output_dtype)


class Magi2FourierRoPE(nn.Module):
    """Three-axis element-wise Fourier embedding used by MAGI-2."""

    def __init__(self, head_dim: int, temperature: float = 10_000.0) -> None:
        super().__init__()
        num_bands = head_dim // 8
        bands = temperature ** (
            -torch.arange(num_bands, dtype=torch.float32) / max(num_bands, 1)
        )
        self.register_buffer("bands", bands)

    def forward(self, coordinates: Tensor) -> Tensor:
        """Return concatenated sine/cosine values for ``[T, 9]`` coordinates."""
        if coordinates.ndim != 2 or coordinates.shape[1] != 9:
            raise ValueError("coordinates must have shape [tokens, 9]")
        xyz = coordinates[:, :3].float()
        sizes = coordinates[:, 3:6].float()
        references = coordinates[:, 6:9].float()
        scales = (references - 1.0) / (sizes - 1.0)
        singleton = (references == 1.0) & (sizes == 1.0)
        scales = torch.where(singleton, torch.ones_like(scales), scales)
        if not torch.isfinite(scales).all():
            raise ValueError("coordinate scaling produced a non-finite value")
        centers = (sizes - 1.0) / 2.0
        centers[:, 0] = 0.0
        projection = (xyz - centers).unsqueeze(-1) * scales.unsqueeze(-1) * self.bands
        return torch.cat((projection.sin(), projection.cos()), dim=1).flatten(1)


def _apply_magi2_rope(value: Tensor, rope: Tensor) -> Tensor:
    """Apply non-interleaved partial rotary embedding to ``[T, H, D]``."""
    sin, cos = rope.tensor_split(2, dim=-1)
    sin = torch.cat((sin, sin), dim=-1).unsqueeze(1)
    cos = torch.cat((cos, cos), dim=-1).unsqueeze(1)
    rotary_dim = cos.shape[-1]
    rotary_value = value[..., :rotary_dim]
    first, second = rotary_value.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    embedded = rotary_value * cos + rotated * sin
    return torch.cat((embedded, value[..., rotary_dim:]), dim=-1)


class Magi2Attention(nn.Module):
    """Differentiable variable-length attention with gating and sink logits."""

    def __init__(
        self,
        architecture: Magi2ArchitectureConfig,
        *,
        num_modalities: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        hidden_size = architecture.hidden_size
        num_heads = architecture.num_attention_heads
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = architecture.head_dim
        self.pre_norm = Magi2RMSNorm(hidden_size, num_modalities=num_modalities)
        self.q_norm = Magi2RMSNorm(
            self.head_dim, num_modalities=num_modalities, out_dtype=torch.float32
        )
        self.k_norm = Magi2RMSNorm(
            self.head_dim, num_modalities=num_modalities, out_dtype=torch.float32
        )
        self.linear_g = Magi2ModalityLinear(
            hidden_size, num_heads, num_modalities=num_modalities, dtype=dtype
        )
        self.linear_qkv = Magi2ModalityLinear(
            hidden_size,
            3 * hidden_size,
            num_modalities=num_modalities,
            dtype=dtype,
        )
        self.linear_proj = Magi2ModalityLinear(
            hidden_size,
            hidden_size,
            num_modalities=num_modalities,
            dtype=dtype,
        )
        self.sinks = nn.Parameter(
            torch.zeros(architecture.sink_token_num, num_heads, dtype=torch.float32)
        )
        self._keep_in_float32_parameter_names = ("sinks",)

    def _attention(self, q: Tensor, k: Tensor, v: Tensor, cu_seqlens: Tensor) -> Tensor:
        output = torch.empty_like(q)
        scale = self.head_dim**-0.5
        boundaries = tuple(int(item) for item in cu_seqlens.cpu().tolist())
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            q_segment = q[start:end].transpose(0, 1).float()
            k_segment = k[start:end].transpose(0, 1).float()
            v_segment = v[start:end].transpose(0, 1).float()
            logits = torch.matmul(q_segment, k_segment.transpose(-1, -2)) * scale
            sink_logits = self.sinks.T.unsqueeze(1).expand(-1, end - start, -1)
            weights = torch.softmax(torch.cat((logits, sink_logits), dim=-1), dim=-1)
            attended = torch.matmul(weights[..., : end - start], v_segment)
            output[start:end] = attended.transpose(0, 1).to(output.dtype)
        return output

    def forward(
        self,
        hidden_states: Tensor,
        rope: Tensor,
        cu_seqlens: Tensor,
        dispatcher: Magi2ModalityDispatcher,
    ) -> Tensor:
        """Run attention in packed order while projections stay modality-grouped."""
        normed = self.pre_norm(hidden_states, dispatcher)
        gates = self.linear_g(normed, dispatcher).reshape(-1, self.num_heads, 1)
        qkv = self.linear_qkv(normed, dispatcher).reshape(-1, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=1)
        q = self.q_norm(q, dispatcher)
        k = self.k_norm(k, dispatcher)
        q = _apply_magi2_rope(dispatcher.inverse_permute(q), rope)
        k = _apply_magi2_rope(dispatcher.inverse_permute(k), rope)
        v = dispatcher.inverse_permute(v)
        output = dispatcher.permute(self._attention(q, k, v, cu_seqlens))
        output = output * torch.sigmoid(gates)
        output = output.reshape(-1, self.hidden_size).to(self.linear_proj.weight.dtype)
        return self.linear_proj(output, dispatcher)


class Magi2DenseMLP(nn.Module):
    """Dense QuickGEGLU block used in the four multimodal layers."""

    def __init__(
        self,
        architecture: Magi2ArchitectureConfig,
        *,
        num_modalities: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        intermediate_size = (
            max(
                128,
                int(architecture.hidden_size * architecture.intermediate_factor * 2 / 3)
                // 128
                * 128,
            )
        )
        self.pre_norm = Magi2RMSNorm(
            architecture.hidden_size, num_modalities=num_modalities
        )
        self.up_gate = Magi2ModalityLinear(
            architecture.hidden_size,
            2 * intermediate_size,
            num_modalities=num_modalities,
            dtype=dtype,
        )
        self.down = Magi2ModalityLinear(
            intermediate_size,
            architecture.hidden_size,
            num_modalities=num_modalities,
            dtype=dtype,
        )

    def forward(self, hidden_states: Tensor, dispatcher: Magi2ModalityDispatcher) -> Tensor:
        """Apply the dense multimodal feed-forward block."""
        hidden_states = self.pre_norm(hidden_states, dispatcher)
        hidden_states = magi2_quick_geglu(self.up_gate(hidden_states, dispatcher))
        return self.down(hidden_states, dispatcher)


class Magi2MoEMLP(nn.Module):
    """Routed multi-head experts plus global and modality-shared experts."""

    def __init__(
        self,
        config: TransformerConfig,
        architecture: Magi2ArchitectureConfig,
        pg_collection: ProcessGroupCollection,
        layer_number: int,
    ) -> None:
        super().__init__()
        hidden_size = architecture.hidden_size
        dtype = config.params_dtype
        self.pre_norm = Magi2RMSNorm(hidden_size, num_modalities=3)
        self.split_linear = Magi2ModalityLinear(hidden_size, hidden_size, dtype=dtype)
        self.merge_linear = Magi2ModalityLinear(hidden_size, hidden_size, dtype=dtype)
        self.routed = Magi2DistributedMultiHeadMoE(
            config,
            pg_collection=pg_collection,
            num_heads=architecture.moe_num_heads,
            num_experts_per_head=architecture.moe_num_experts_per_head,
            top_k=architecture.moe_top_k,
            expert_intermediate_size=architecture.moe_expert_intermediate_size,
            route_norm=architecture.route_norm,
            route_scale=architecture.route_scale,
            route_norm_eps=architecture.route_norm_eps,
        )
        self.routed.set_layer_number(layer_number)
        self.shared_fc1 = Magi2ModalityLinear(
            hidden_size, 2 * architecture.shared_expert_intermediate_size, dtype=dtype
        )
        self.shared_fc2 = Magi2ModalityLinear(
            architecture.shared_expert_intermediate_size, hidden_size, dtype=dtype
        )
        self.modality_fc1 = Magi2ModalityLinear(
            hidden_size,
            2 * architecture.modality_expert_intermediate_size,
            num_modalities=3,
            dtype=dtype,
        )
        self.modality_fc2 = Magi2ModalityLinear(
            architecture.modality_expert_intermediate_size,
            hidden_size,
            num_modalities=3,
            dtype=dtype,
        )

    def forward(self, hidden_states: Tensor, dispatcher: Magi2ModalityDispatcher) -> Tensor:
        """Sum routed, global-shared, and modality-shared expert outputs."""
        normed = self.pre_norm(hidden_states, dispatcher)
        routed, routed_bias = self.routed(self.split_linear(normed))
        if routed_bias is not None:
            raise RuntimeError("MAGI-2 routed experts must be bias-free")
        routed = self.merge_linear(routed)
        shared = self.shared_fc2(magi2_quick_geglu(self.shared_fc1(normed)))
        modality = self.modality_fc2(
            magi2_quick_geglu(self.modality_fc1(normed, dispatcher)), dispatcher
        )
        return routed + shared + modality


class Magi2TransformerLayer(nn.Module):
    """One attention/MLP pair, each wrapped by an independent mHC mapping."""

    def __init__(
        self,
        config: TransformerConfig,
        architecture: Magi2ArchitectureConfig,
        pg_collection: ProcessGroupCollection,
        layer_index: int,
    ) -> None:
        super().__init__()
        num_modalities = 3 if layer_index in architecture.mm_layers else 1
        self.mhc_norm = Magi2RMSNorm(
            architecture.adapter_width,
            num_modalities=num_modalities,
            out_dtype=torch.float32,
        )
        self.attention_mhc = Magi2HyperConnectionBranch(
            architecture.hidden_size, architecture.mhc_num_streams
        )
        self.mlp_mhc = Magi2HyperConnectionBranch(
            architecture.hidden_size, architecture.mhc_num_streams
        )
        self.attention = Magi2Attention(
            architecture, num_modalities=num_modalities, dtype=config.params_dtype
        )
        if layer_index in architecture.moe_layers:
            self.mlp: nn.Module = Magi2MoEMLP(
                config, architecture, pg_collection, layer_index + 1
            )
        else:
            self.mlp = Magi2DenseMLP(
                architecture, num_modalities=num_modalities, dtype=config.params_dtype
            )

    def forward(
        self,
        hidden_states: Tensor,
        rope: Tensor,
        cu_seqlens: Tensor,
        dispatcher: Magi2ModalityDispatcher,
    ) -> Tensor:
        """Run attention and feed-forward mHC branches."""
        attention_input, h_post, h_res = self.attention_mhc.pre(
            hidden_states, self.mhc_norm(hidden_states, dispatcher)
        )
        attention_output = self.attention(
            attention_input, rope, cu_seqlens, dispatcher
        )
        hidden_states = self.attention_mhc.merge(
            hidden_states, attention_output, h_post, h_res
        )

        mlp_input, h_post, h_res = self.mlp_mhc.pre(
            hidden_states, self.mhc_norm(hidden_states, dispatcher)
        )
        mlp_output = self.mlp(mlp_input, dispatcher)
        return self.mlp_mhc.merge(hidden_states, mlp_output, h_post, h_res)


class Magi2PreAdapter(nn.Module):
    """Modality-aware input projection into four residual streams."""

    def __init__(self, architecture: Magi2ArchitectureConfig) -> None:
        super().__init__()
        width = architecture.adapter_width
        self.architecture = architecture
        self.video = nn.Linear(architecture.video_in_channels, width, dtype=torch.float32)
        self.audio = nn.Linear(architecture.audio_in_channels, width, dtype=torch.float32)
        self.text = nn.Linear(architecture.text_in_channels, width, dtype=torch.float32)
        self.rope = Magi2FourierRoPE(architecture.head_dim)
        for linear in (self.video, self.audio, self.text):
            linear._keep_in_float32_parameter_names = ("weight", "bias")

    def forward(
        self,
        value: Tensor,
        coordinates: Tensor,
        modality_mapping: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Project each input modality and construct its 3D rotary embedding."""
        output = torch.zeros(
            value.shape[0],
            self.architecture.adapter_width,
            device=value.device,
            dtype=torch.float32,
        )
        for modality, channels, projection in (
            (Magi2Modality.VIDEO, self.architecture.video_in_channels, self.video),
            (Magi2Modality.AUDIO, self.architecture.audio_in_channels, self.audio),
            (Magi2Modality.TEXT, self.architecture.text_in_channels, self.text),
            (Magi2Modality.TIME, self.architecture.text_in_channels, self.text),
        ):
            indices = torch.nonzero(modality_mapping == modality, as_tuple=False).flatten()
            if indices.numel() > 0:
                projected = projection(value.index_select(0, indices)[:, :channels].float())
                output.index_copy_(0, indices, projected)
        return output, self.rope(coordinates)


class Magi2PostAdapter(nn.Module):
    """Project residual streams back to video/audio velocity targets."""

    def __init__(self, architecture: Magi2ArchitectureConfig) -> None:
        super().__init__()
        width = architecture.adapter_width
        self.architecture = architecture
        self.video_norm = Magi2RMSNorm(width)
        self.audio_norm = Magi2RMSNorm(width)
        self.video = nn.Linear(width, architecture.video_in_channels, bias=False, dtype=torch.float32)
        self.audio = nn.Linear(width, architecture.audio_in_channels, bias=False, dtype=torch.float32)
        self.video._keep_in_float32_parameter_names = ("weight",)
        self.audio._keep_in_float32_parameter_names = ("weight",)

    def forward(self, value: Tensor, modality_mapping: Tensor) -> Tensor:
        """Return a packed output with the public maximum channel width."""
        output_width = max(
            self.architecture.video_in_channels, self.architecture.audio_in_channels
        )
        output = torch.zeros(
            value.shape[0], output_width, device=value.device, dtype=torch.float32
        )
        for modality, norm, projection, channels in (
            (
                Magi2Modality.VIDEO,
                self.video_norm,
                self.video,
                self.architecture.video_in_channels,
            ),
            (
                Magi2Modality.AUDIO,
                self.audio_norm,
                self.audio,
                self.architecture.audio_in_channels,
            ),
        ):
            indices = torch.nonzero(modality_mapping == modality, as_tuple=False).flatten()
            if indices.numel() > 0:
                selected = value.index_select(0, indices)
                projected = projection(norm(selected).float())
                output[:, :channels].index_copy_(0, indices, projected)
        return output


class Magi2Model(MegatronModule):
    """Trainable packed-sequence MAGI-2 preview model."""

    def __init__(
        self,
        config: TransformerConfig,
        architecture: Optional[Magi2ArchitectureConfig] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        super().__init__(config)
        self.architecture = architecture or Magi2ArchitectureConfig()
        if config.hidden_size != self.architecture.hidden_size:
            raise ValueError("TransformerConfig and MAGI-2 hidden sizes differ")
        if config.num_layers != self.architecture.num_layers:
            raise ValueError("TransformerConfig and MAGI-2 layer counts differ")
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pre_adapter = Magi2PreAdapter(self.architecture)
        self.layers = nn.ModuleList(
            [
                Magi2TransformerLayer(config, self.architecture, pg_collection, layer_index)
                for layer_index in range(self.architecture.num_layers)
            ]
        )
        self.post_adapter = Magi2PostAdapter(self.architecture)
        self.input_tensor: Optional[Tensor] = None

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Store pipeline input for compatibility; PP is currently restricted to one."""
        self.input_tensor = input_tensor

    def forward(
        self,
        inputs: Tensor,
        coordinates: Tensor,
        modality_mapping: Tensor,
        cu_seqlens: Tensor,
    ) -> Tensor:
        """Run packed video/audio/text tokens through the full model."""
        if inputs.ndim != 2 or inputs.shape[0] != modality_mapping.shape[0]:
            raise ValueError("inputs must have shape [tokens, input_channels]")
        if cu_seqlens.ndim != 1 or cu_seqlens[0] != 0 or cu_seqlens[-1] != inputs.shape[0]:
            raise ValueError("cu_seqlens must start at zero and end at the token count")
        original_mapping = modality_mapping
        model_mapping = modality_mapping.clone()
        model_mapping[model_mapping == Magi2Modality.TIME] = Magi2Modality.TEXT
        hidden_states, rope = self.pre_adapter(inputs, coordinates, original_mapping)
        dispatcher = Magi2ModalityDispatcher(model_mapping)
        hidden_states = dispatcher.permute(hidden_states).to(self.config.params_dtype)
        for layer in self.layers:
            hidden_states = layer(hidden_states, rope, cu_seqlens, dispatcher)
        hidden_states = dispatcher.inverse_permute(hidden_states)
        return self.post_adapter(hidden_states, original_mapping)


__all__ = [
    "Magi2ArchitectureConfig",
    "Magi2HyperConnectionBranch",
    "Magi2Model",
    "Magi2Modality",
    "Magi2ModalityDispatcher",
    "Magi2ModalityLinear",
    "Magi2RMSNorm",
    "magi2_quick_geglu",
]
