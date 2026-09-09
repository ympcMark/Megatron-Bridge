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

"""Differentiable PyTorch reference for MAGI-2 multi-head MoE.

The public MAGI-2 preview implements this block with a forward-only fused
Triton kernel.  This module preserves the public checkpoint tensor layout and
routing/activation math while using ordinary PyTorch operations so autograd
can establish the pretraining path.  It is intentionally a correctness
reference: expert parallelism and fused production kernels are separate work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor, nn


Magi2RouterScoreFunction: TypeAlias = Literal["sigmoid", "softmax"]


@dataclass(frozen=True)
class Magi2MultiHeadMoEConfig:
    """Configuration for the MAGI-2 multi-head MoE reference module."""

    hidden_size: int
    num_heads: int
    num_experts: int
    top_k: int
    expert_intermediate_size: int
    params_dtype: torch.dtype
    score_func: Magi2RouterScoreFunction = "sigmoid"
    route_norm: bool = True
    route_scale: float = 1.0
    route_norm_eps: float = 1e-12

    def __post_init__(self) -> None:
        """Reject configurations that cannot represent the published block."""
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if not 0 < self.top_k <= self.num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        if self.expert_intermediate_size <= 0:
            raise ValueError("expert_intermediate_size must be positive")
        if not self.params_dtype.is_floating_point:
            raise ValueError("params_dtype must be a floating-point dtype")
        if self.score_func not in ("sigmoid", "softmax"):
            raise ValueError(f"Unknown score_func: {self.score_func}")
        if self.route_scale <= 0.0:
            raise ValueError("route_scale must be positive")
        if self.route_norm_eps <= 0.0:
            raise ValueError("route_norm_eps must be positive")


class Magi2ReferenceMultiHeadMoE(nn.Module):
    """Autograd-capable reference for the MAGI-2 multi-head expert block.

    Parameter names and shapes match ``CoreMultiHeadMoE`` in the public
    MAGI-2 preview.  Inputs may have arbitrary leading dimensions and must end
    in ``hidden_size``; the output has the same shape and dtype as the input.
    """

    _SWIGLU_ALPHA = 1.702
    _SWIGLU_LIMIT = 7.0

    def __init__(self, config: Magi2MultiHeadMoEConfig) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.num_experts = config.num_experts
        self.topk = config.top_k
        self.d_head = config.hidden_size // config.num_heads
        self.d_expert = config.expert_intermediate_size
        self.flatten_num_experts = self.num_heads * self.num_experts

        self.gate = nn.Parameter(torch.empty(self.flatten_num_experts, self.d_head, dtype=torch.float32))
        self.W_gate = nn.Parameter(
            torch.empty(
                self.flatten_num_experts,
                self.d_head,
                self.d_expert,
                dtype=config.params_dtype,
            )
        )
        self.W_up = nn.Parameter(
            torch.empty(
                self.flatten_num_experts,
                self.d_head,
                self.d_expert,
                dtype=config.params_dtype,
            )
        )
        self.W_down = nn.Parameter(
            torch.empty(
                self.flatten_num_experts,
                self.d_expert,
                self.d_head,
                dtype=config.params_dtype,
            )
        )

        # Keep the same nested state-dict keys as the preview implementation.
        self.router = nn.Module()
        self.router.register_buffer("expert_bias", torch.zeros(self.flatten_num_experts, dtype=torch.float32))
        self.router.register_buffer(
            "expert_bias_ema",
            torch.zeros(self.flatten_num_experts, dtype=torch.float32),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize every head/expert independently for scratch training."""
        nn.init.normal_(self.gate, mean=0.0, std=self.d_head**-0.5)
        for weight in (self.W_gate, self.W_up, self.W_down):
            for expert_weight in weight.unbind(0):
                nn.init.xavier_uniform_(expert_weight)

    def _route(self, x_heads: Tensor) -> tuple[Tensor, Tensor]:
        """Return unbiased routing probabilities and bias-selected expert IDs."""
        gate = self.gate.view(self.num_heads, self.num_experts, self.d_head)
        router_logits = torch.einsum("shd,hed->hse", x_heads.float(), gate)
        if self.config.score_func == "sigmoid":
            router_scores = torch.sigmoid(router_logits)
        else:
            router_scores = F.softmax(router_logits, dim=-1)

        expert_bias = self.router.expert_bias.view(self.num_heads, 1, self.num_experts)
        topk_indices = torch.topk(router_scores + expert_bias, self.topk, dim=-1).indices
        topk_probs = router_scores.gather(-1, topk_indices)
        if self.config.route_norm:
            topk_probs = F.normalize(topk_probs, p=1, dim=-1, eps=self.config.route_norm_eps)
        return topk_probs * self.config.route_scale, topk_indices

    def _swiglu7(self, gate: Tensor, up: Tensor) -> Tensor:
        """Apply the clamped SwiGLU7 function used by the preview kernel."""
        gate = gate.clamp(max=self._SWIGLU_LIMIT)
        up = up.clamp(min=-self._SWIGLU_LIMIT, max=self._SWIGLU_LIMIT)
        return gate * torch.sigmoid(self._SWIGLU_ALPHA * gate) * (up + 1.0)

    def forward(self, x: Tensor) -> Tensor:
        """Route each hidden-state head through its selected experts."""
        if x.ndim < 2:
            raise ValueError("x must have at least two dimensions")
        if x.shape[-1] != self.config.hidden_size:
            raise ValueError(f"x last dimension must equal hidden_size ({self.config.hidden_size})")
        if not x.dtype.is_floating_point:
            raise ValueError("x must have a floating-point dtype")

        original_shape = x.shape
        x_heads = x.reshape(-1, self.num_heads, self.d_head)
        topk_probs, topk_indices = self._route(x_heads)
        x_by_head = x_heads.permute(1, 0, 2)

        W_gate = self.W_gate.view(self.num_heads, self.num_experts, self.d_head, self.d_expert)
        W_up = self.W_up.view(self.num_heads, self.num_experts, self.d_head, self.d_expert)
        W_down = self.W_down.view(self.num_heads, self.num_experts, self.d_expert, self.d_head)
        head_indices = torch.arange(self.num_heads, device=x.device).view(self.num_heads, 1, 1)
        selected_W_gate = W_gate[head_indices, topk_indices]
        selected_W_up = W_up[head_indices, topk_indices]
        selected_W_down = W_down[head_indices, topk_indices]

        compute_dtype = torch.promote_types(x.dtype, torch.float32)
        x_compute = x_by_head.to(compute_dtype)
        gate_projection = torch.einsum("hsd,hskdf->hskf", x_compute, selected_W_gate.to(compute_dtype))
        up_projection = torch.einsum("hsd,hskdf->hskf", x_compute, selected_W_up.to(compute_dtype))
        hidden = self._swiglu7(gate_projection, up_projection)
        expert_output = torch.einsum("hskf,hskfd->hskd", hidden, selected_W_down.to(compute_dtype))
        output = torch.sum(expert_output * topk_probs.unsqueeze(-1), dim=2)
        return output.permute(1, 0, 2).reshape(original_shape).to(x.dtype)


__all__ = [
    "Magi2MultiHeadMoEConfig",
    "Magi2ReferenceMultiHeadMoE",
    "Magi2RouterScoreFunction",
]
