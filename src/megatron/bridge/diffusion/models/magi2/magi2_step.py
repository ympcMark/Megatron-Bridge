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
from torch import Tensor

from megatron.bridge.training.state import GlobalState

from megatron.bridge.diffusion.models.magi2.modeling_magi2.model import (
    Magi2Model,
    Magi2Modality,
)


def magi2_timestep_embedding(timestep: Tensor, width: int) -> Tensor:
    """Create the sinusoidal time embedding consumed as a text-width token."""
    half = width // 2
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, dtype=torch.float32, device=timestep.device)
        / max(half, 1)
    )
    angles = timestep.float().reshape(-1, 1) * 1000.0 * frequencies.reshape(1, -1)
    embedding = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
    if width % 2:
        embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
    return embedding


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

        timesteps = torch.rand(batch_size, device=clean_inputs.device)
        token_timesteps = timesteps.repeat_interleave(sequence_length)
        noise = torch.randn_like(flattened_clean)
        model_inputs = flattened_clean.clone()
        output_width = max(state.cfg.model.video_in_channels, state.cfg.model.audio_in_channels)
        target = torch.zeros(
            flattened_clean.shape[0], output_width, device=clean_inputs.device
        )
        loss_mask = torch.zeros_like(target)

        for modality, channels in (
            (Magi2Modality.VIDEO, state.cfg.model.video_in_channels),
            (Magi2Modality.AUDIO, state.cfg.model.audio_in_channels),
        ):
            selected = flattened_mapping == modality
            selected_timestep = token_timesteps[selected].unsqueeze(-1)
            clean = flattened_clean[selected, :channels]
            selected_noise = noise[selected, :channels]
            model_inputs[selected, :channels] = (
                selected_timestep * clean + (1.0 - selected_timestep) * selected_noise
            )
            target[selected, :channels] = clean - selected_noise
            loss_mask[selected, :channels] = 1.0

        time_selected = flattened_mapping == Magi2Modality.TIME
        if time_selected.any():
            time_values = token_timesteps[time_selected]
            model_inputs[time_selected, : state.cfg.model.text_in_channels] = (
                magi2_timestep_embedding(time_values, state.cfg.model.text_in_channels)
            )

        output = model(
            model_inputs,
            flattened_coordinates,
            flattened_mapping,
            cu_seqlens,
        )
        squared_error = (output.float() - target).square()
        return squared_error, partial(magi2_flow_loss, loss_mask)


__all__ = ["Magi2ForwardStep", "magi2_flow_loss", "magi2_timestep_embedding"]
