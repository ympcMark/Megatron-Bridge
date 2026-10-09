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

"""Flow-matching data, forward, and loss steps for MAGI-2 pretraining."""

from __future__ import annotations

import math
from collections.abc import Iterable
from functools import partial

import torch
from megatron.core.models.magi2 import Magi2Modality, Magi2Model
from torch import Tensor

from megatron.bridge.training.state import GlobalState


def magi2_timestep_embedding(timestep: Tensor, width: int) -> Tensor:
    """Create MAGI-2's per-token time-channel embedding."""
    if width < 0:
        raise ValueError("time-channel width must be non-negative")
    if width == 0:
        return torch.empty(timestep.numel(), 0, dtype=torch.float32, device=timestep.device)
    if width == 1:
        return timestep.float().reshape(-1, 1)
    half = width // 2
    frequencies = torch.exp(
        -math.log(10_000.0) * torch.arange(half, dtype=torch.float32, device=timestep.device) / half
    )
    angles = timestep.float().reshape(-1, 1) * 1000.0 * frequencies.reshape(1, -1)
    embedding = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
    if width % 2:
        embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
    return embedding


def magi2_prepare_flow_inputs(
    clean_inputs: Tensor,
    modality_mapping: Tensor,
    token_sigmas: Tensor,
    *,
    video_in_channels: int,
    audio_in_channels: int,
    time_channel_dim: int,
    noise: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build official MAGI-2 flow inputs, targets, masks, and time channels."""
    if clean_inputs.ndim != 2:
        raise ValueError("clean_inputs must have shape [tokens, channels]")
    token_count = clean_inputs.shape[0]
    if modality_mapping.shape != (token_count,):
        raise ValueError("modality_mapping must have shape [tokens]")
    if token_sigmas.shape != (token_count,):
        raise ValueError("token_sigmas must have shape [tokens]")
    if noise is None:
        noise = torch.randn_like(clean_inputs)
    elif noise.shape != clean_inputs.shape:
        raise ValueError("noise must have the same shape as clean_inputs")

    model_inputs = clean_inputs.clone()
    output_width = max(video_in_channels, audio_in_channels)
    target = clean_inputs.new_zeros(token_count, output_width)
    loss_mask = clean_inputs.new_zeros(token_count, output_width)
    time_values = token_sigmas.new_zeros(token_count)

    for modality, channels in (
        (Magi2Modality.VIDEO, video_in_channels),
        (Magi2Modality.AUDIO, audio_in_channels),
    ):
        selected = modality_mapping == modality
        selected_sigma = token_sigmas[selected].unsqueeze(-1)
        clean = clean_inputs[selected, :channels]
        selected_noise = noise[selected, :channels]
        model_inputs[selected, :channels] = (1.0 - selected_sigma) * clean + selected_sigma * selected_noise
        target[selected, :channels] = selected_noise - clean
        loss_mask[selected, :channels] = 1.0
        time_values[selected] = token_sigmas[selected]

    time_token_sequence = magi2_timestep_embedding(time_values, time_channel_dim)
    return model_inputs, target, loss_mask, time_token_sequence


def magi2_flow_loss(
    loss_mask: Tensor,
    output_tensor: Tensor,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Normalize flow-matching MSE over active modality channels."""
    loss_sum = (output_tensor.float() * loss_mask).sum()
    denominator = loss_mask.sum().float().clamp_min(1.0)
    loss = loss_sum / denominator
    valid_tokens = loss_mask.any(dim=-1).sum().to(torch.int)
    return loss, valid_tokens, {"flow_mse": torch.stack((loss_sum.detach(), denominator.detach()))}


class Magi2ForwardStep:
    """Noise clean modality latents and run one rectified-flow training step."""

    def __call__(
        self,
        state: GlobalState,
        data_iterator: Iterable,
        model: Magi2Model,
    ) -> tuple[Tensor, partial]:
        """Prepare a packed batch and bind its flow-matching loss."""
        state.timers("batch-generator", log_level=2).start()
        with state.straggler_timer(bdata=True):
            batch = next(data_iterator)
            batch = {
                name: value.cuda(non_blocking=True) if torch.is_tensor(value) else value
                for name, value in batch.items()
            }
        state.timers("batch-generator").stop()

        clean_inputs = batch["clean_inputs"]
        coordinates = batch["coordinates"]
        modality_mapping = batch["modality_mapping"]
        batch_size, sequence_length, input_width = clean_inputs.shape
        flattened_clean = clean_inputs.reshape(-1, input_width)
        flattened_coordinates = coordinates.reshape(-1, coordinates.shape[-1])
        flattened_mapping = modality_mapping.reshape(-1)
        cu_seqlens = torch.arange(
            0,
            (batch_size + 1) * sequence_length,
            sequence_length,
            dtype=torch.int32,
            device=clean_inputs.device,
        )

        sigmas = torch.rand(batch_size, device=clean_inputs.device)
        token_sigmas = sigmas.repeat_interleave(sequence_length)
        model_inputs, target, loss_mask, time_token_sequence = magi2_prepare_flow_inputs(
            flattened_clean,
            flattened_mapping,
            token_sigmas,
            video_in_channels=state.cfg.model.magi2_video_in_channels,
            audio_in_channels=state.cfg.model.magi2_audio_in_channels,
            time_channel_dim=state.cfg.dataset.time_channel_dim,
        )

        output = model(
            model_inputs,
            flattened_coordinates,
            flattened_mapping,
            cu_seqlens,
            time_token_sequence=time_token_sequence,
        )
        squared_error = (output.float() - target).square()
        return squared_error, partial(magi2_flow_loss, loss_mask)


__all__ = [
    "Magi2ForwardStep",
    "magi2_flow_loss",
    "magi2_prepare_flow_inputs",
    "magi2_timestep_embedding",
]
