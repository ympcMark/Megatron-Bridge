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
import torch.nn.functional as F

from megatron.bridge.diffusion.models.magi2.modeling_magi2.model import (
    Magi2ArchitectureConfig,
    Magi2DenseMLP,
    Magi2FourierRoPE,
    Magi2HyperConnectionBranch,
    Magi2ModalityDispatcher,
    Magi2ModalityLinear,
    Magi2RMSNorm,
    magi2_quick_geglu,
)


pytestmark = [pytest.mark.unit]


def test_original_architecture_dimensions_match_public_preview():
    config = Magi2ArchitectureConfig()

    assert config.num_layers == 40
    assert config.num_attention_heads == 24
    assert config.moe_layers == tuple(range(2, 38))
    assert config.flattened_num_experts == 3072
    assert config.adapter_width == 12288

    expert_parameters = (
        2 * (config.hidden_size // config.moe_num_heads) * config.moe_expert_intermediate_size
        + config.moe_expert_intermediate_size * (config.hidden_size // config.moe_num_heads)
    )
    routed_parameters = len(config.moe_layers) * config.flattened_num_experts * expert_parameters
    assert routed_parameters == 108_716_359_680
    assert config.parameter_count_breakdown()["routed_experts"] == routed_parameters
    assert config.parameter_count == 113_934_732_336


def test_modality_linear_matches_independent_grouped_projection():
    dispatcher = Magi2ModalityDispatcher(torch.tensor([2, 0, 1, 0, 2, 1]))
    value = dispatcher.permute(torch.arange(24, dtype=torch.float32).reshape(6, 4))
    linear = Magi2ModalityLinear(4, 3, num_modalities=3, bias=True, dtype=torch.float32)

    actual = linear(value, dispatcher)
    expected = torch.cat(
        [
            F.linear(group, linear.weight[index], linear.bias[index])
            for index, group in enumerate(dispatcher.split(value))
        ]
    )

    torch.testing.assert_close(actual, expected)


def test_modality_rms_norm_uses_zero_centered_weights():
    dispatcher = Magi2ModalityDispatcher(torch.tensor([0, 1, 2, 0, 1, 2]))
    value = dispatcher.permute(torch.arange(1, 25, dtype=torch.float32).reshape(6, 4))
    norm = Magi2RMSNorm(4, num_modalities=3)
    norm.weight.data.copy_(
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5, 0.5],
                [-0.25, -0.25, -0.25, -0.25],
            ]
        )
    )

    actual = norm(value, dispatcher)
    expected_groups = []
    for index, group in enumerate(dispatcher.split(value)):
        normalized = group * torch.rsqrt(group.square().mean(dim=-1, keepdim=True) + 1e-6)
        expected_groups.append(normalized * (norm.weight[index] + 1.0))

    torch.testing.assert_close(actual, torch.cat(expected_groups))


def test_quick_geglu_matches_public_interleaved_formula_and_preserves_dtype():
    value = torch.tensor(
        [[-8.0, -9.0, 2.0, 3.0, 9.0, 8.0]], dtype=torch.bfloat16
    )

    actual = magi2_quick_geglu(value)
    gate = value.float()[..., ::2].clamp(max=7.0)
    linear = value.float()[..., 1::2].clamp(min=-7.0, max=7.0)
    expected = (gate * torch.sigmoid(1.702 * gate) * (linear + 1.0)).to(torch.bfloat16)

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected)


def test_fourier_rope_has_public_partial_rotary_width_and_finite_values():
    coordinates = torch.tensor(
        [
            [0, 0, 0, 1, 2, 2, 1, 2, 2],
            [0, 0, 1, 1, 2, 2, 1, 2, 2],
            [0, 1, 0, 1, 2, 2, 1, 2, 2],
            [0, 1, 1, 1, 2, 2, 1, 2, 2],
        ],
        dtype=torch.float32,
    )

    rope = Magi2FourierRoPE(128)(coordinates)

    assert rope.shape == (4, 96)
    assert torch.isfinite(rope).all()


def test_public_mhc_mapping_is_doubly_stochastic_and_differentiable():
    branch = Magi2HyperConnectionBranch(hidden_size=8, num_streams=4)
    hidden = torch.randn(3, 32, requires_grad=True)
    normalized = torch.randn(3, 32, requires_grad=True)

    branch_input, post_mapping, residual_mapping = branch.pre(hidden, normalized)
    merged = branch.merge(hidden, branch_input, post_mapping, residual_mapping)

    assert branch_input.shape == (3, 8)
    assert post_mapping.shape == (3, 4)
    assert residual_mapping.shape == (3, 4, 4)
    torch.testing.assert_close(
        residual_mapping.sum(dim=-1), torch.ones(3, 4), atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        residual_mapping.sum(dim=-2), torch.ones(3, 4), atol=1e-5, rtol=1e-5
    )
    merged.square().mean().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert branch.phi_fused.grad is not None and torch.isfinite(branch.phi_fused.grad).all()


def test_reduced_dense_mlp_keeps_nonzero_aligned_width():
    architecture = Magi2ArchitectureConfig(
        num_layers=1,
        hidden_size=64,
        head_dim=16,
        num_query_groups=4,
        intermediate_factor=2,
        mm_layers=(0,),
        moe_layers=(),
        moe_num_heads=4,
        moe_num_experts_per_head=4,
        moe_top_k=2,
        mhc_num_streams=2,
    )
    mlp = Magi2DenseMLP(
        architecture, num_modalities=3, dtype=torch.bfloat16
    )

    assert mlp.up_gate.weight.shape == (3, 256, 64)
    assert mlp.down.weight.shape == (3, 64, 128)
