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

"""Runtime-aware model FLOP accounting for MAGI-2 training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from megatron.core.models.magi2 import Magi2Config


MAGI2_FLOPS_RUNTIME_STAT_NAMES = (
    "magi2_video_tokens",
    "magi2_audio_tokens",
    "magi2_text_tokens",
    "magi2_padding_tokens",
)


@dataclass(frozen=True, slots=True)
class Magi2FlopEstimate:
    """MAGI-2 matrix-multiplication FLOPs for one global training step.

    The accounting follows Megatron's model-FLOPs convention: one multiply-add
    is two FLOPs, and backward is approximated as twice the forward work.
    Elementwise normalization, activation, routing selection, loss, and
    communication are intentionally excluded.

    Attributes:
        adapters_forward: Input and output adapter forward FLOPs.
        attention_forward: QKV, gate, core attention, and projection FLOPs.
        mhc_forward: Dynamic mHC projection and residual-mixing FLOPs.
        dense_mlp_forward: Dense multimodal MLP FLOPs.
        moe_router_forward: Head-local router-logit FLOPs.
        moe_routed_forward: Top-k routed expert FLOPs.
        moe_shared_forward: Shared and modality-shared expert FLOPs.
        moe_split_merge_forward: MoE split and merge projection FLOPs.
        recompute_forward: Extra forward FLOPs executed by full activation
            recomputation. This is ``None`` for selective recomputation because
            the selected submodule boundaries require kernel-level accounting.
    """

    adapters_forward: int
    attention_forward: int
    mhc_forward: int
    dense_mlp_forward: int
    moe_router_forward: int
    moe_routed_forward: int
    moe_shared_forward: int
    moe_split_merge_forward: int
    recompute_forward: int | None

    @property
    def transformer_forward(self) -> int:
        """Return forward FLOPs executed inside the TransformerBlock."""
        return (
            self.attention_forward
            + self.mhc_forward
            + self.dense_mlp_forward
            + self.moe_router_forward
            + self.moe_routed_forward
            + self.moe_shared_forward
            + self.moe_split_merge_forward
        )

    @property
    def forward(self) -> int:
        """Return total model forward FLOPs."""
        return self.adapters_forward + self.transformer_forward

    @property
    def algorithmic_training(self) -> int:
        """Return conventional model FLOPs for forward plus backward."""
        return 3 * self.forward

    @property
    def executed_training(self) -> int | None:
        """Return training FLOPs including full activation recomputation."""
        if self.recompute_forward is None:
            return None
        return self.algorithmic_training + self.recompute_forward


def _full_recompute_forward_flops(
    config: Magi2Config,
    *,
    attention_per_layer: int,
    mhc_per_layer: int,
    dense_mlp_per_layer: int,
    moe_per_layer: int,
) -> int | None:
    """Return extra forward work caused by the configured recomputation policy."""
    if config.recompute_granularity is None:
        return 0
    if config.recompute_granularity != "full":
        return None

    if config.recompute_method in (None, "uniform"):
        recomputed_layers = set(range(config.num_layers))
    elif config.recompute_method == "block":
        num_recomputed = config.recompute_num_layers or 0
        recomputed_layers = set(range(min(num_recomputed, config.num_layers)))
    else:
        raise ValueError(f"unsupported MAGI-2 recompute method: {config.recompute_method}")

    moe_layers = set(config.magi2_moe_layers)
    return sum(
        attention_per_layer + mhc_per_layer + (moe_per_layer if layer_index in moe_layers else dense_mlp_per_layer)
        for layer_index in recomputed_layers
    )


def estimate_magi2_training_flops(
    config: Magi2Config,
    *,
    seqlen_sum: int,
    seqlen_squared_sum: int,
    modality_token_counts: Mapping[str, int],
) -> Magi2FlopEstimate:
    """Estimate exact MAGI-2 model matmul FLOPs for one global batch.

    Args:
        config: Finalized MAGI-2 model configuration.
        seqlen_sum: Number of physical tokens processed by linear layers.
        seqlen_squared_sum: Sum of squared logical packed-sequence lengths.
        modality_token_counts: Logical video, audio, and text token counts plus
            the physical CP padding count, keyed by
            :data:`MAGI2_FLOPS_RUNTIME_STAT_NAMES`.

    Returns:
        Component-level forward, algorithmic-training, and recompute FLOPs.

    Raises:
        ValueError: If runtime counts are incomplete or inconsistent.
    """
    if seqlen_sum <= 0 or seqlen_squared_sum <= 0:
        raise ValueError("MAGI-2 FLOP accounting requires positive runtime sequence statistics")
    missing = set(MAGI2_FLOPS_RUNTIME_STAT_NAMES) - modality_token_counts.keys()
    if missing:
        raise ValueError(f"missing MAGI-2 FLOP runtime statistics: {sorted(missing)}")

    video_tokens = int(modality_token_counts["magi2_video_tokens"])
    audio_tokens = int(modality_token_counts["magi2_audio_tokens"])
    text_tokens = int(modality_token_counts["magi2_text_tokens"])
    padding_tokens = int(modality_token_counts["magi2_padding_tokens"])
    if min(video_tokens, audio_tokens, text_tokens, padding_tokens) < 0:
        raise ValueError("MAGI-2 modality token counts must be non-negative")
    if video_tokens + audio_tokens + text_tokens + padding_tokens != seqlen_sum:
        raise ValueError("MAGI-2 modality and padding token counts must sum to seqlen_sum")

    tokens = seqlen_sum
    hidden = config.hidden_size
    streams = config.magi2_mhc_num_streams
    adapter_width = config.magi2_adapter_width

    pre_adapter = (
        2
        * adapter_width
        * (
            video_tokens * config.magi2_video_in_channels
            + audio_tokens * config.magi2_audio_in_channels
            + (text_tokens + padding_tokens) * config.magi2_text_in_channels
        )
    )
    post_adapter = (
        2
        * adapter_width
        * (video_tokens * config.magi2_video_in_channels + audio_tokens * config.magi2_audio_in_channels)
    )
    adapters_forward = pre_adapter + post_adapter

    attention_per_layer = (
        8 * tokens * hidden**2 + 2 * tokens * hidden * config.num_attention_heads + 4 * seqlen_squared_sum * hidden
    )

    mapping_width = 2 * streams + streams**2
    mhc_branch = (
        2 * tokens * adapter_width * mapping_width + 2 * tokens * streams * hidden + 2 * tokens * streams**2 * hidden
    )
    mhc_per_layer = 2 * mhc_branch

    dense_mlp_per_layer = 6 * tokens * hidden * config.magi2_dense_intermediate_size

    moe_router_per_layer = 2 * tokens * hidden * config.magi2_moe_num_experts_per_head
    moe_routed_per_layer = 6 * tokens * config.magi2_moe_top_k * hidden * config.magi2_moe_expert_intermediate_size
    moe_shared_per_layer = (
        6
        * tokens
        * hidden
        * (config.magi2_shared_expert_intermediate_size + config.magi2_modality_expert_intermediate_size)
    )
    moe_split_merge_per_layer = 4 * tokens * hidden**2
    moe_per_layer = moe_router_per_layer + moe_routed_per_layer + moe_shared_per_layer + moe_split_merge_per_layer

    num_dense_layers = len(config.magi2_mm_layers)
    num_moe_layers = len(config.magi2_moe_layers)
    recompute_forward = _full_recompute_forward_flops(
        config,
        attention_per_layer=attention_per_layer,
        mhc_per_layer=mhc_per_layer,
        dense_mlp_per_layer=dense_mlp_per_layer,
        moe_per_layer=moe_per_layer,
    )
    return Magi2FlopEstimate(
        adapters_forward=adapters_forward,
        attention_forward=config.num_layers * attention_per_layer,
        mhc_forward=config.num_layers * mhc_per_layer,
        dense_mlp_forward=num_dense_layers * dense_mlp_per_layer,
        moe_router_forward=num_moe_layers * moe_router_per_layer,
        moe_routed_forward=num_moe_layers * moe_routed_per_layer,
        moe_shared_forward=num_moe_layers * moe_shared_per_layer,
        moe_split_merge_forward=num_moe_layers * moe_split_merge_per_layer,
        recompute_forward=recompute_forward,
    )


__all__ = [
    "MAGI2_FLOPS_RUNTIME_STAT_NAMES",
    "Magi2FlopEstimate",
    "estimate_magi2_training_flops",
]
