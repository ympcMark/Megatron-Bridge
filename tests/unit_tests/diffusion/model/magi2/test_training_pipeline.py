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

import pytest
import torch

from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.diffusion.models.magi2.data import Magi2LatentDataset
from megatron.bridge.diffusion.models.magi2.magi2_step import (
    magi2_flow_loss,
    magi2_timestep_embedding,
)
from megatron.bridge.diffusion.models.magi2.modeling_magi2.distributed_multi_head_moe import (
    build_magi2_expert_config,
)
from megatron.bridge.diffusion.models.magi2.provider import Magi2ModelProvider


pytestmark = [pytest.mark.unit]


def test_original_provider_matches_public_architecture():
    provider = Magi2ModelProvider()
    architecture = provider.architecture_config()

    assert provider.require_original_size
    assert architecture.num_layers == 40
    assert architecture.hidden_size == 3072
    assert architecture.flattened_num_experts == 3072
    assert architecture.moe_top_k == 6


def test_expert_config_from_deferred_bridge_provider_is_concrete_mcore_config():
    provider = Magi2ModelProvider()
    provider.finalize()

    expert_config = build_magi2_expert_config(
        provider,
        num_heads=12,
        num_experts_per_head=256,
        top_k=6,
        expert_intermediate_size=1280,
        score_func="sigmoid",
        route_scale=4.9,
    )

    assert type(expert_config) is TransformerConfig
    assert expert_config.hidden_size == 256
    assert expert_config.num_moe_experts == 3072
    assert expert_config.moe_ffn_hidden_size == 1280
    assert expert_config.moe_router_enable_expert_bias


def test_latent_dataset_is_reproducible_and_uses_public_channel_widths():
    dataset = Magi2LatentDataset(
        2,
        seed=123,
        video_tokens=2,
        audio_tokens=2,
        text_tokens=1,
        time_tokens=1,
        video_in_channels=48,
        audio_in_channels=64,
        text_in_channels=5120,
    )

    first = dataset[0]
    repeated = dataset[0]

    assert first["clean_inputs"].shape == (6, 5120)
    assert first["coordinates"].shape == (6, 9)
    assert torch.equal(first["clean_inputs"], repeated["clean_inputs"])
    assert torch.equal(first["modality_mapping"], repeated["modality_mapping"])


def test_timestep_embedding_and_flow_loss_are_finite():
    embedding = magi2_timestep_embedding(torch.tensor([0.25, 0.75]), 9)
    squared_error = torch.tensor([[1.0, 4.0], [9.0, 16.0]])
    mask = torch.tensor([[1.0, 1.0], [0.0, 0.0]])

    loss, valid_tokens, metrics = magi2_flow_loss(mask, squared_error)

    assert embedding.shape == (2, 9)
    assert torch.isfinite(embedding).all()
    torch.testing.assert_close(loss, torch.tensor(2.5))
    assert valid_tokens.item() == 1
    torch.testing.assert_close(metrics["flow_mse"], torch.tensor([5.0, 2.0]))
