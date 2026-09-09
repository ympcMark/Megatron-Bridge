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

from unittest.mock import Mock

import examples.models.magi2.pretrain_magi2 as pretrain_module
import pytest
import torch
from megatron.core.models.magi2 import Magi2Config, Magi2Model, Magi2TransformerLayerSpecs
from megatron.core.process_groups_config import ProcessGroupCollection

import megatron.bridge.diffusion.models.magi2.provider as provider_module
from megatron.bridge.diffusion.models.magi2.data import Magi2LatentDataset
from megatron.bridge.diffusion.models.magi2.magi2_step import magi2_flow_loss, magi2_timestep_embedding
from megatron.bridge.diffusion.models.magi2.provider import Magi2ModelProvider


pytestmark = [pytest.mark.unit]


def test_original_provider_is_native_mcore_config():
    provider = Magi2ModelProvider()

    assert isinstance(provider, Magi2Config)
    assert Magi2Model.__module__.startswith("megatron.core.models.magi2")
    assert provider.require_original_size
    assert provider.num_layers == 40
    assert provider.hidden_size == 3072
    assert provider.magi2_flattened_num_experts == 3072
    assert provider.magi2_moe_top_k == 6
    assert provider.magi2_parameter_count == 113_934_732_336


def test_provider_only_selects_specs_and_instantiates_native_model(monkeypatch):
    provider = Magi2ModelProvider()
    pg_collection = Mock(spec=ProcessGroupCollection)
    native_model = Mock(spec=Magi2Model)
    constructor = Mock(return_value=native_model)
    provider._pg_collection = pg_collection
    monkeypatch.setattr(provider_module, "Magi2Model", constructor)

    result = provider.provide()

    assert result is native_model
    constructor.assert_called_once()
    kwargs = constructor.call_args.kwargs
    assert kwargs["config"] is provider
    assert kwargs["pg_collection"] is pg_collection
    assert isinstance(kwargs["layer_specs"], Magi2TransformerLayerSpecs)
    assert kwargs["pre_process"]
    assert kwargs["post_process"]


def test_original_size_guard_reports_changed_native_field():
    provider = Magi2ModelProvider()
    provider.magi2_moe_num_experts_per_head = 4

    with pytest.raises(ValueError, match="magi2_moe_num_experts_per_head"):
        provider.finalize()


def test_reduced_provider_finalizes_with_native_mcore_validation():
    provider = Magi2ModelProvider()
    provider.require_original_size = False
    provider.num_layers = 2
    provider.hidden_size = 64
    provider.num_attention_heads = 4
    provider.num_query_groups = 4
    provider.kv_channels = 16
    provider.ffn_hidden_size = 128
    provider.num_moe_experts = 16
    provider.moe_ffn_hidden_size = 32
    provider.moe_router_topk = 2
    provider.expert_model_parallel_size = 1
    provider.magi2_video_in_channels = 8
    provider.magi2_audio_in_channels = 8
    provider.magi2_text_in_channels = 16
    provider.magi2_intermediate_factor = 2
    provider.magi2_mm_layers = (0,)
    provider.magi2_moe_layers = (1,)
    provider.magi2_moe_num_heads = 4
    provider.magi2_moe_num_experts_per_head = 4
    provider.magi2_moe_top_k = 2
    provider.magi2_moe_expert_intermediate_size = 32
    provider.magi2_shared_expert_intermediate_size = 32
    provider.magi2_modality_expert_intermediate_size = 32
    provider.magi2_mhc_num_streams = 2

    provider.finalize()

    assert provider.magi2_adapter_width == 128
    assert provider.magi2_moe_head_dim == 16


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


def test_official_checkpoint_hook_delegates_to_native_mcore_loader(monkeypatch, tmp_path):
    model = Mock(spec=Magi2Model)
    report = Mock(consumed_source_keys=("source",), populated_target_keys=("target",))
    loader = Mock(return_value=report)
    monkeypatch.setattr(pretrain_module, "load_magi2_official_safetensors", loader)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    result = pretrain_module._load_official_checkpoint(tmp_path, [model])

    assert result == [model]
    loader.assert_called_once_with(model, tmp_path, strict=True)
