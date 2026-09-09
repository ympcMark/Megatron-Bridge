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

"""Deterministic packed latent data for MAGI-2 training bring-up."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from megatron.core.models.magi2 import Magi2Modality
from torch.utils.data import Dataset

from megatron.bridge.data.base import DatasetBuildContext, DatasetProvider


class Magi2LatentDataset(Dataset):
    """Generate deterministic clean latents and conditioning embeddings."""

    def __init__(
        self,
        size: int,
        *,
        seed: int,
        video_tokens: int,
        audio_tokens: int,
        text_tokens: int,
        time_tokens: int,
        video_in_channels: int,
        audio_in_channels: int,
        text_in_channels: int,
    ) -> None:
        self.size = size
        self.seed = seed
        self.video_tokens = video_tokens
        self.audio_tokens = audio_tokens
        self.text_tokens = text_tokens
        self.time_tokens = time_tokens
        self.video_in_channels = video_in_channels
        self.audio_in_channels = audio_in_channels
        self.text_in_channels = text_in_channels

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(self.seed + index)
        modality_mapping = torch.tensor(
            [Magi2Modality.VIDEO] * self.video_tokens
            + [Magi2Modality.AUDIO] * self.audio_tokens
            + [Magi2Modality.TEXT] * self.text_tokens
            + [Magi2Modality.TIME] * self.time_tokens,
            dtype=torch.long,
        )
        token_count = modality_mapping.numel()
        input_width = max(self.video_in_channels, self.audio_in_channels, self.text_in_channels)
        clean_inputs = torch.zeros(token_count, input_width, dtype=torch.float32)
        video = modality_mapping == Magi2Modality.VIDEO
        audio = modality_mapping == Magi2Modality.AUDIO
        text = modality_mapping == Magi2Modality.TEXT
        clean_inputs[video, : self.video_in_channels] = torch.randn(
            int(video.sum()), self.video_in_channels, generator=generator
        )
        clean_inputs[audio, : self.audio_in_channels] = torch.randn(
            int(audio.sum()), self.audio_in_channels, generator=generator
        )
        clean_inputs[text, : self.text_in_channels] = 0.02 * torch.randn(
            int(text.sum()), self.text_in_channels, generator=generator
        )

        coordinates = torch.zeros(token_count, 9, dtype=torch.float32)
        coordinates[:, 0] = torch.arange(token_count, dtype=torch.float32)
        coordinates[:, 3] = token_count
        coordinates[:, 4:6] = 1
        coordinates[:, 6] = token_count
        coordinates[:, 7:9] = 1
        return {
            "clean_inputs": clean_inputs,
            "coordinates": coordinates,
            "modality_mapping": modality_mapping,
            "sample_id": torch.tensor(index, dtype=torch.long),
        }


@dataclass(kw_only=True)
class Magi2LatentDatasetConfig(DatasetProvider):
    """Dataset provider for deterministic MAGI-2 flow-matching batches."""

    seed: int = 1234
    video_tokens: int = 4
    audio_tokens: int = 4
    text_tokens: int = 3
    time_tokens: int = 1
    video_in_channels: int = 48
    audio_in_channels: int = 64
    text_in_channels: int = 5120
    dataloader_type: Optional[str] = "cyclic"
    num_workers: int = 0
    persistent_workers: bool = False
    pin_memory: bool = True

    def build_datasets(
        self, context: DatasetBuildContext
    ) -> tuple[Optional[Magi2LatentDataset], Optional[Magi2LatentDataset], None]:
        """Build deterministic train/validation splits."""

        def build(size: int, seed_offset: int) -> Optional[Magi2LatentDataset]:
            if size <= 0:
                return None
            return Magi2LatentDataset(
                size,
                seed=self.seed + seed_offset,
                video_tokens=self.video_tokens,
                audio_tokens=self.audio_tokens,
                text_tokens=self.text_tokens,
                time_tokens=self.time_tokens,
                video_in_channels=self.video_in_channels,
                audio_in_channels=self.audio_in_channels,
                text_in_channels=self.text_in_channels,
            )

        return build(context.train_samples, 0), build(context.valid_samples, 1_000_000), None


__all__ = ["Magi2LatentDataset", "Magi2LatentDatasetConfig"]
