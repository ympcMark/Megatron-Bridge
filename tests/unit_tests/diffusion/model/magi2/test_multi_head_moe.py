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

from megatron.bridge.diffusion.models.magi2 import (
    Magi2MultiHeadMoEConfig,
    Magi2ReferenceMultiHeadMoE,
)


pytestmark = [pytest.mark.unit]


def _config(**overrides) -> Magi2MultiHeadMoEConfig:
    values = {
        "hidden_size": 8,
        "num_heads": 2,
        "num_experts": 3,
        "top_k": 2,
        "expert_intermediate_size": 5,
        "params_dtype": torch.float32,
    }
    unknown = (
        set(overrides)
        - set(values)
        - {
            "score_func",
            "route_norm",
            "route_scale",
            "route_norm_eps",
        }
    )
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")
    values.update(overrides)
    return Magi2MultiHeadMoEConfig(**values)


def _oracle_forward(module: Magi2ReferenceMultiHeadMoE, x: torch.Tensor) -> torch.Tensor:
    config = module.config
    d_head = config.hidden_size // config.num_heads
    x_heads = x.reshape(-1, config.num_heads, d_head)
    gate = module.gate.view(config.num_heads, config.num_experts, d_head)
    logits = torch.einsum("shd,hed->hse", x_heads.float(), gate)
    scores = torch.sigmoid(logits) if config.score_func == "sigmoid" else torch.softmax(logits, dim=-1)
    selection_scores = scores + module.router.expert_bias.view(config.num_heads, 1, config.num_experts)
    indices = selection_scores.topk(config.top_k, dim=-1).indices
    probabilities = scores.gather(-1, indices)
    if config.route_norm:
        probabilities = F.normalize(probabilities, p=1, dim=-1, eps=config.route_norm_eps)
    probabilities = probabilities * config.route_scale

    output = torch.zeros_like(x_heads, dtype=torch.float32)
    for head in range(config.num_heads):
        for token in range(x_heads.shape[0]):
            for route in range(config.top_k):
                expert = indices[head, token, route].item()
                flat_expert = head * config.num_experts + expert
                expert_input = x_heads[token, head].float()
                projected_gate = expert_input @ module.W_gate[flat_expert].float()
                projected_up = expert_input @ module.W_up[flat_expert].float()
                projected_gate = projected_gate.clamp(max=7.0)
                projected_up = projected_up.clamp(min=-7.0, max=7.0)
                hidden = projected_gate * torch.sigmoid(1.702 * projected_gate) * (projected_up + 1.0)
                expert_output = hidden @ module.W_down[flat_expert].float()
                output[token, head] += probabilities[head, token, route] * expert_output
    return output.reshape(x.shape).to(x.dtype)


class TestMagi2MultiHeadMoEConfig:
    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"hidden_size": 7}, "divisible"),
            ({"num_heads": 0}, "positive"),
            ({"num_experts": 0}, "positive"),
            ({"top_k": 4}, "top_k"),
            ({"expert_intermediate_size": 0}, "positive"),
            ({"params_dtype": torch.int64}, "floating-point"),
            ({"score_func": "invalid"}, "score_func"),
            ({"route_scale": 0.0}, "route_scale"),
            ({"route_norm_eps": 0.0}, "route_norm_eps"),
        ],
    )
    def test_invalid_config_is_rejected(self, overrides, message):
        with pytest.raises(ValueError, match=message):
            _config(**overrides)


class TestMagi2ReferenceMultiHeadMoE:
    def test_parameter_layout_matches_preview_checkpoint(self):
        module = Magi2ReferenceMultiHeadMoE(_config())

        assert module.gate.shape == (6, 4)
        assert module.W_gate.shape == (6, 4, 5)
        assert module.W_up.shape == (6, 4, 5)
        assert module.W_down.shape == (6, 5, 4)
        assert set(module.state_dict()) == {
            "gate",
            "W_gate",
            "W_up",
            "W_down",
            "router.expert_bias",
            "router.expert_bias_ema",
        }

    @pytest.mark.parametrize("score_func", ["sigmoid", "softmax"])
    @pytest.mark.parametrize("route_norm", [False, True])
    def test_forward_matches_independent_oracle(self, score_func, route_norm):
        torch.manual_seed(1234)
        module = Magi2ReferenceMultiHeadMoE(_config(score_func=score_func, route_norm=route_norm, route_scale=0.7))
        module.router.expert_bias.copy_(torch.linspace(-0.2, 0.2, module.flatten_num_experts))
        x = torch.randn(2, 3, module.config.hidden_size)

        actual = module(x)
        expected = _oracle_forward(module, x)

        assert actual.shape == x.shape
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_expert_bias_changes_selection_but_not_probability(self):
        module = Magi2ReferenceMultiHeadMoE(
            _config(
                hidden_size=4,
                num_heads=1,
                num_experts=3,
                top_k=1,
                expert_intermediate_size=2,
                route_norm=False,
            )
        )
        module.gate.detach().zero_()
        module.router.expert_bias.copy_(torch.tensor([-1.0, 2.0, 0.0]))
        x_heads = torch.ones(1, 1, 4)

        probabilities, indices = module._route(x_heads)

        torch.testing.assert_close(indices, torch.tensor([[[1]]]))
        torch.testing.assert_close(probabilities, torch.tensor([[[0.5]]]))

    def test_backward_populates_input_and_parameter_gradients(self):
        torch.manual_seed(4321)
        module = Magi2ReferenceMultiHeadMoE(_config(route_norm=False))
        x = torch.randn(4, module.config.hidden_size, requires_grad=True)

        module(x).square().mean().backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert x.grad.abs().sum() > 0
        for parameter in module.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0

    def test_optimizer_step_updates_trainable_parameters(self):
        torch.manual_seed(9876)
        module = Magi2ReferenceMultiHeadMoE(_config(route_norm=False))
        optimizer = torch.optim.SGD(module.parameters(), lr=0.05)
        x = torch.randn(4, module.config.hidden_size)
        target = torch.randn_like(x)
        before = {name: parameter.detach().clone() for name, parameter in module.named_parameters()}

        loss = F.mse_loss(module(x), target)
        loss.backward()
        optimizer.step()

        for name, parameter in module.named_parameters():
            assert not torch.equal(before[name], parameter)

    @pytest.mark.parametrize("shape", [(8,), (2, 7), (2, 3, 8)])
    def test_invalid_input_is_rejected(self, shape):
        module = Magi2ReferenceMultiHeadMoE(_config())
        x = torch.ones(shape, dtype=torch.int64 if shape == (2, 3, 8) else torch.float32)

        with pytest.raises(ValueError):
            module(x)
