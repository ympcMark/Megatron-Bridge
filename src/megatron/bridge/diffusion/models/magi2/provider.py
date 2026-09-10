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

"""Megatron Bridge provider for the native MCore MAGI-2 model."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from megatron.core.models.magi2 import (
    Magi2Config,
    Magi2DotProductAttention,
    Magi2Model,
    Magi2TransformerLayerSpecs,
    get_magi2_layer_specs,
)
from megatron.core.process_groups_config import ProcessGroupCollection

from megatron.bridge.models.model_provider import ModelProviderMixin


_ORIGINAL_ARCHITECTURE_FIELDS = (
    "num_layers",
    "hidden_size",
    "num_attention_heads",
    "num_query_groups",
    "kv_channels",
    "ffn_hidden_size",
    "num_moe_experts",
    "moe_ffn_hidden_size",
    "magi2_video_in_channels",
    "magi2_audio_in_channels",
    "magi2_text_in_channels",
    "magi2_intermediate_factor",
    "magi2_mm_layers",
    "magi2_moe_layers",
    "magi2_moe_num_heads",
    "magi2_moe_num_experts_per_head",
    "magi2_moe_top_k",
    "magi2_moe_expert_intermediate_size",
    "magi2_shared_expert_intermediate_size",
    "magi2_modality_expert_intermediate_size",
    "magi2_route_scale",
    "magi2_route_norm",
    "magi2_route_norm_eps",
    "magi2_sink_token_num",
    "magi2_mhc_num_streams",
)


def default_magi2_layer_specs(_: Magi2Config) -> Magi2TransformerLayerSpecs:
    """Select the correctness-first native MCore MAGI-2 layer specs."""
    return get_magi2_layer_specs(Magi2DotProductAttention)


def _architecture_mismatches(config: Magi2Config) -> list[str]:
    original = Magi2Config()
    return [name for name in _ORIGINAL_ARCHITECTURE_FIELDS if getattr(config, name) != getattr(original, name)]


@dataclass
class Magi2ModelProvider(Magi2Config, ModelProviderMixin[Magi2Model]):
    """Configure and instantiate MCore's native MAGI-2 model.

    Bridge owns only training-facing defaults and model construction policy.
    The architecture, TransformerBlock assembly, attention, mHC, dense MLP,
    and distributed multi-head MoE all live in ``megatron.core.models.magi2``.
    """

    seq_length: int = 12
    vocab_size: int = 1
    make_vocab_size_divisible_by: int = 1
    share_embeddings_and_output_weights: bool = False
    bf16: bool = True
    fp16: bool = False
    expert_model_parallel_size: int = 8
    expert_tensor_parallel_size: int | None = 1
    require_original_size: bool = True
    magi2_layer_specs: Magi2TransformerLayerSpecs | Callable[[Magi2Config], Magi2TransformerLayerSpecs] = (
        default_magi2_layer_specs
    )
    _pg_collection: ProcessGroupCollection | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Defer MCore validation until Bridge has applied recipe overrides."""

    def finalize(self) -> None:
        """Validate the finalized native MCore configuration."""
        if self.require_original_size:
            mismatches = _architecture_mismatches(self)
            if mismatches:
                raise ValueError(
                    "require_original_size=True rejects reduced architecture fields: " + ", ".join(mismatches)
                )
        Magi2Config.__post_init__(self)

    def provide(
        self,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> Magi2Model:
        """Instantiate the native MCore MAGI-2 model on the active stage."""
        if vp_stage is not None:
            raise ValueError("MAGI-2 does not support virtual pipeline stages yet")
        if self._pg_collection is None:
            raise RuntimeError("MAGI-2 provider requires initialized process groups")

        layer_specs = self.magi2_layer_specs
        if not isinstance(layer_specs, Magi2TransformerLayerSpecs):
            layer_specs = layer_specs(self)

        return Magi2Model(
            config=self,
            layer_specs=layer_specs,
            pg_collection=self._pg_collection,
            pre_process=True if pre_process is None else pre_process,
            post_process=True if post_process is None else post_process,
            vp_stage=vp_stage,
        )


__all__ = ["Magi2ModelProvider", "default_magi2_layer_specs"]
