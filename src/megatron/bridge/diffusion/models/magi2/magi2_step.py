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
from megatron.bridge.training.utils.packed_seq_utils import get_thd_cp_partition_indices
from megatron.bridge.training.utils.pg_utils import get_pg_collection


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
    else:
        noise = noise.to(device=clean_inputs.device, dtype=clean_inputs.dtype)

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
        selected_sigma = token_sigmas[selected].to(dtype=clean_inputs.dtype).unsqueeze(-1)
        clean = clean_inputs[selected, :channels]
        selected_noise = noise[selected, :channels]
        model_inputs[selected, :channels] = (1.0 - selected_sigma) * clean + selected_sigma * selected_noise
        target[selected, :channels] = selected_noise - clean
        loss_mask[selected, :channels] = 1.0
        time_values[selected] = token_sigmas[selected]

    time_token_sequence = magi2_timestep_embedding(time_values, time_channel_dim)
    return model_inputs, target, loss_mask, time_token_sequence


def magi2_deterministic_flow_random(
    clean_inputs: Tensor,
    sample_ids: Tensor,
    *,
    step: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Generate topology-independent synthetic sigmas and noise per sample."""
    if clean_inputs.ndim != 3:
        raise ValueError("clean_inputs must have shape [batch, tokens, channels]")
    if sample_ids.numel() != clean_inputs.shape[0]:
        raise ValueError("sample_ids must contain one identifier per sample")
    sequence_length = clean_inputs.shape[1]
    cu_seqlens = torch.arange(
        0,
        (clean_inputs.shape[0] + 1) * sequence_length,
        sequence_length,
        dtype=torch.int32,
        device=clean_inputs.device,
    )
    token_sigmas, noise = magi2_deterministic_packed_flow_random(
        clean_inputs.reshape(-1, clean_inputs.shape[-1]),
        sample_ids,
        cu_seqlens,
        step=step,
        seed=seed,
    )
    sigmas = token_sigmas.index_select(0, cu_seqlens[:-1].long())
    return sigmas, noise.view_as(clean_inputs)


def magi2_deterministic_packed_flow_random(
    clean_inputs: Tensor,
    sample_ids: Tensor,
    cu_seqlens: Tensor,
    *,
    step: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Generate one deterministic sigma and noise stream per packed sample."""
    if clean_inputs.ndim != 2:
        raise ValueError("packed clean_inputs must have shape [tokens, channels]")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() != sample_ids.numel() + 1:
        raise ValueError("cu_seqlens must contain one boundary per sample plus the terminal boundary")
    boundaries = tuple(int(item) for item in cu_seqlens.cpu().tolist())
    if boundaries[0] != 0 or boundaries[-1] != clean_inputs.shape[0]:
        raise ValueError("cu_seqlens must cover the full packed clean-input stream")
    if any(start >= end for start, end in zip(boundaries[:-1], boundaries[1:])):
        raise ValueError("every packed MAGI-2 sample must contain at least one token")

    token_sigmas = torch.empty(clean_inputs.shape[0], device=clean_inputs.device)
    noise = torch.empty_like(clean_inputs)
    for sample_id, start, end in zip(sample_ids.reshape(-1).cpu().tolist(), boundaries[:-1], boundaries[1:]):
        generator = torch.Generator(device=clean_inputs.device)
        generator.manual_seed(seed + int(sample_id) * 1_000_003 + step * 10_000_019)
        sigma = torch.rand((), generator=generator, device=clean_inputs.device)
        token_sigmas[start:end] = sigma
        noise[start:end] = torch.randn(
            (end - start, clean_inputs.shape[1]),
            generator=generator,
            device=clean_inputs.device,
            dtype=clean_inputs.dtype,
        )
    return token_sigmas, noise


def magi2_flow_loss(
    loss_mask: Tensor,
    output_tensor: Tensor,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Return the local flow-MSE sum and active-element normalization count."""
    loss_sum = (output_tensor.float() * loss_mask).sum()
    active_elements = loss_mask.sum().to(torch.int)
    return loss_sum, active_elements, {"flow_mse": torch.stack((loss_sum.detach(), active_elements.detach().float()))}


def _pad_packed_sequence_tensor(
    tensor: Tensor,
    logical_boundaries: tuple[int, ...],
    padded_lengths: tuple[int, ...],
    fill_value: float = 0,
) -> Tensor:
    """Pad every logical THD sequence independently and preserve its boundaries."""
    segments = []
    for start, end, padded_length in zip(logical_boundaries[:-1], logical_boundaries[1:], padded_lengths):
        segment = tensor[start:end]
        padded = tensor.new_full((padded_length, *tensor.shape[1:]), fill_value)
        padded[: end - start] = segment
        segments.append(padded)
    return torch.cat(segments, dim=0)


def partition_magi2_batch_for_context_parallel(
    tensors: dict[str, Tensor],
    *,
    cu_seqlens: Tensor,
    cp_group: torch.distributed.ProcessGroup,
) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor, Tensor]:
    """Pad and partition token-aligned MAGI-2 tensors with MCore's THD CP layout."""
    cp_size = cp_group.size()
    if cp_size <= 1:
        raise ValueError("context-parallel partitioning requires CP size greater than one")
    alignment = 2 * cp_size
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must contain at least one packed sequence")
    logical_boundaries = tuple(int(item) for item in cu_seqlens.cpu().tolist())
    total_tokens = logical_boundaries[-1]
    if logical_boundaries[0] != 0 or any(
        start >= end for start, end in zip(logical_boundaries[:-1], logical_boundaries[1:])
    ):
        raise ValueError("cu_seqlens must describe non-empty monotonically increasing sequences")
    if any(tensor.shape[0] != total_tokens for tensor in tensors.values()):
        raise ValueError("all token-aligned tensors must match the logical packed token count")
    logical_lengths = tuple(end - start for start, end in zip(logical_boundaries[:-1], logical_boundaries[1:]))
    padded_lengths = tuple(math.ceil(length / alignment) * alignment for length in logical_lengths)
    padded_tensors = {
        name: _pad_packed_sequence_tensor(
            tensor,
            logical_boundaries,
            padded_lengths,
            fill_value=(Magi2Modality.TEXT if name == "modality_mapping" else 0),
        )
        for name, tensor in tensors.items()
    }
    cu_seqlens_padded = torch.tensor(
        (0, *torch.tensor(padded_lengths, dtype=torch.int64).cumsum(0).tolist()),
        dtype=torch.int32,
        device=next(iter(tensors.values())).device,
    )
    physical_token_count = int(cu_seqlens_padded[-1].item())
    indices = get_thd_cp_partition_indices(
        cu_seqlens_padded,
        total_tokens=physical_token_count,
        cp_group=cp_group,
        device=cu_seqlens.device,
    )
    local_tensors = {name: tensor.index_select(0, indices) for name, tensor in padded_tensors.items()}
    cp_split_sizes = torch.full(
        (cp_size,),
        indices.numel(),
        dtype=torch.int32,
        device=cu_seqlens.device,
    )
    return local_tensors, cu_seqlens.to(torch.int32), cu_seqlens_padded, cp_split_sizes, indices


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
        if clean_inputs.ndim == 3:
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
        elif clean_inputs.ndim == 2:
            flattened_clean = clean_inputs
            flattened_coordinates = coordinates
            flattened_mapping = modality_mapping
            cu_seqlens = batch.get("cu_seqlens")
            if cu_seqlens is None:
                raise ValueError("packed MAGI-2 batches must provide cu_seqlens")
        else:
            raise ValueError("clean_inputs must have shape [tokens, channels] or [batch, tokens, channels]")

        cp_group = get_pg_collection(model).cp
        token_sigmas, noise = magi2_deterministic_packed_flow_random(
            flattened_clean,
            batch["sample_id"],
            cu_seqlens,
            step=state.train_state.step,
            seed=state.cfg.rng.seed,
        )
        model_inputs, target, loss_mask, time_token_sequence = magi2_prepare_flow_inputs(
            flattened_clean,
            flattened_mapping,
            token_sigmas,
            video_in_channels=state.cfg.model.magi2_video_in_channels,
            audio_in_channels=state.cfg.model.magi2_audio_in_channels,
            time_channel_dim=state.cfg.dataset.time_channel_dim,
            noise=noise,
        )

        cu_seqlens_padded = None
        cp_split_sizes = None
        cp_global_indices = None
        if cp_group.size() > 1:
            token_tensors, cu_seqlens, cu_seqlens_padded, cp_split_sizes, cp_global_indices = (
                partition_magi2_batch_for_context_parallel(
                    {
                        "model_inputs": model_inputs,
                        "target": target,
                        "loss_mask": loss_mask,
                        "coordinates": flattened_coordinates,
                        "modality_mapping": flattened_mapping,
                        "time_token_sequence": time_token_sequence,
                    },
                    cu_seqlens=cu_seqlens,
                    cp_group=cp_group,
                )
            )
            model_inputs = token_tensors["model_inputs"]
            target = token_tensors["target"]
            loss_mask = token_tensors["loss_mask"]
            flattened_coordinates = token_tensors["coordinates"]
            flattened_mapping = token_tensors["modality_mapping"]
            time_token_sequence = token_tensors["time_token_sequence"]

        output = model(
            model_inputs,
            flattened_coordinates,
            flattened_mapping,
            cu_seqlens,
            cu_seqlens_padded=cu_seqlens_padded,
            cp_split_sizes=cp_split_sizes,
            cp_global_indices=cp_global_indices,
            time_token_sequence=time_token_sequence,
        )
        squared_error = (output.float() - target).square()
        return squared_error, partial(magi2_flow_loss, loss_mask)


__all__ = [
    "Magi2ForwardStep",
    "magi2_flow_loss",
    "magi2_deterministic_flow_random",
    "magi2_deterministic_packed_flow_random",
    "magi2_prepare_flow_inputs",
    "magi2_timestep_embedding",
    "partition_magi2_batch_for_context_parallel",
]
