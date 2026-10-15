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

"""Bridge training integration for MCore's native MAGI-2 model."""

from megatron.bridge.diffusion.models.magi2.data import (
    Magi2DiskLatentDataset,
    Magi2LatentDataset,
    Magi2LatentDatasetConfig,
    generate_magi2_latent_sample,
    magi2_latent_collate,
    write_magi2_latent_dataset,
    write_magi2_precomputed_latent_dataset,
)
from megatron.bridge.diffusion.models.magi2.flops import Magi2FlopEstimate, estimate_magi2_training_flops
from megatron.bridge.diffusion.models.magi2.magi2_step import Magi2ForwardStep
from megatron.bridge.diffusion.models.magi2.provider import Magi2ModelProvider


__all__ = [
    "Magi2ForwardStep",
    "Magi2DiskLatentDataset",
    "Magi2FlopEstimate",
    "Magi2LatentDataset",
    "Magi2LatentDatasetConfig",
    "Magi2ModelProvider",
    "generate_magi2_latent_sample",
    "estimate_magi2_training_flops",
    "magi2_latent_collate",
    "write_magi2_latent_dataset",
    "write_magi2_precomputed_latent_dataset",
]
