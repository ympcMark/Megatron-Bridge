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

from megatron.bridge.diffusion.models.magi2 import multi_head_topk_routing


pytestmark = [pytest.mark.unit]


def _oracle_route(
    hidden_states,
    gate,
    expert_bias,
    *,
    num_heads,
    num_experts_per_head,
    top_k,
    score_func,
    route_norm,
    route_scale,
    route_norm_eps,
):
    flattened = hidden_states.reshape(-1, hidden_states.shape[-1])
    output_probs = torch.zeros(
        flattened.shape[0], num_heads * num_experts_per_head, dtype=torch.float32
    )
    output_map = torch.zeros_like(output_probs, dtype=torch.bool)
    for token_head in range(flattened.shape[0]):
        head = token_head % num_heads
        first_expert = head * num_experts_per_head
        last_expert = first_expert + num_experts_per_head
        logits = flattened[token_head].float() @ gate[first_expert:last_expert].float().T
        scores = torch.sigmoid(logits) if score_func == "sigmoid" else torch.softmax(logits, dim=-1)
        indices = torch.topk(scores + expert_bias[first_expert:last_expert], top_k).indices
        probabilities = scores[indices]
        if route_norm:
            probabilities = F.normalize(probabilities, p=1, dim=-1, eps=route_norm_eps)
        indices = indices + first_expert
        output_probs[token_head, indices] = probabilities * route_scale
        output_map[token_head, indices] = True
    return output_probs, output_map


@pytest.mark.parametrize("score_func", ["sigmoid", "softmax"])
@pytest.mark.parametrize("route_norm", [False, True])
def test_multi_head_topk_routing_matches_independent_oracle(score_func, route_norm):
    torch.manual_seed(1234)
    hidden_states = torch.randn(12, 1, 4)
    gate = torch.randn(15, 4, requires_grad=True)
    expert_bias = torch.linspace(-0.25, 0.25, 15)
    kwargs = {
        "num_heads": 3,
        "num_experts_per_head": 5,
        "top_k": 2,
        "score_func": score_func,
        "route_norm": route_norm,
        "route_scale": 4.9,
        "route_norm_eps": 1e-12,
    }

    actual_probs, actual_map = multi_head_topk_routing(
        hidden_states, gate, expert_bias, **kwargs
    )
    expected_probs, expected_map = _oracle_route(
        hidden_states, gate, expert_bias, **kwargs
    )

    torch.testing.assert_close(actual_probs, expected_probs)
    torch.testing.assert_close(actual_map, expected_map)


def test_multi_head_topk_routing_only_selects_matching_head_experts():
    hidden_states = torch.ones(8, 1, 2)
    gate = torch.zeros(12, 2)
    expert_bias = torch.arange(12, dtype=torch.float32)

    _, routing_map = multi_head_topk_routing(
        hidden_states,
        gate,
        expert_bias,
        num_heads=4,
        num_experts_per_head=3,
        top_k=1,
        score_func="sigmoid",
        route_norm=False,
        route_scale=1.0,
        route_norm_eps=1e-12,
    )

    selected_experts = routing_map.int().argmax(dim=-1)
    expected_heads = torch.arange(8) % 4
    assert torch.equal(selected_experts // 3, expected_heads)
    assert torch.equal(selected_experts % 3, torch.full((8,), 2))


def test_multi_head_topk_routing_backward_reaches_input_and_gate():
    torch.manual_seed(5678)
    hidden_states = torch.randn(8, 1, 3, requires_grad=True)
    gate = torch.randn(8, 3, requires_grad=True)

    probs, _ = multi_head_topk_routing(
        hidden_states,
        gate,
        None,
        num_heads=2,
        num_experts_per_head=4,
        top_k=2,
        score_func="sigmoid",
        route_norm=False,
        route_scale=0.7,
        route_norm_eps=1e-12,
    )
    probs.square().sum().backward()

    assert hidden_states.grad is not None
    assert gate.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
    assert torch.isfinite(gate.grad).all()
    assert hidden_states.grad.abs().sum() > 0
    assert gate.grad.abs().sum() > 0


@pytest.mark.parametrize(
    ("hidden_states", "gate", "expert_bias", "kwargs", "message"),
    [
        (torch.ones(5, 2), torch.ones(6, 2), None, {}, "divisible"),
        (torch.ones(6, 2), torch.ones(5, 2), None, {}, "gate shape"),
        (torch.ones(6, 2), torch.ones(6, 2), torch.ones(5), {}, "expert_bias shape"),
        (torch.ones(6, 2), torch.ones(6, 2), None, {"top_k": 4}, "top_k"),
    ],
)
def test_multi_head_topk_routing_rejects_invalid_layout(
    hidden_states, gate, expert_bias, kwargs, message
):
    values = {
        "num_heads": 2,
        "num_experts_per_head": 3,
        "top_k": 2,
        "score_func": "sigmoid",
        "route_norm": True,
        "route_scale": 1.0,
        "route_norm_eps": 1e-12,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        multi_head_topk_routing(hidden_states, gate, expert_bias, **values)
