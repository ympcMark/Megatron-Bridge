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

"""Disk-backed and deterministic latent data for MAGI-2 pretraining."""

from __future__ import annotations

import json
from bisect import bisect_right
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from megatron.core.models.magi2 import Magi2Modality
from torch import Tensor
from torch.utils.data import Dataset

from megatron.bridge.data.base import DatasetBuildContext, DatasetProvider


MAGI2_LATENT_DATASET_FORMAT = "magi2-latent-v1"


def _grid_coordinates(token_count: int) -> Tensor:
    """Build official ``(t, h, w, T, H, W, ref_T, ref_H, ref_W)`` coordinates."""
    coordinates = torch.zeros(token_count, 9, dtype=torch.float32)
    coordinates[:, 0] = torch.arange(token_count, dtype=torch.float32)
    coordinates[:, 3] = token_count
    coordinates[:, 4:6] = 1
    coordinates[:, 6] = token_count
    coordinates[:, 7:9] = 1
    return coordinates


def generate_magi2_latent_sample(
    sample_id: int,
    *,
    seed: int,
    video_tokens: int,
    audio_tokens: int,
    text_tokens: int,
    video_in_channels: int,
    audio_in_channels: int,
    text_in_channels: int,
) -> dict[str, Tensor]:
    """Generate one deterministic, structurally realistic MAGI-2 latent sample."""
    token_counts = (video_tokens, audio_tokens, text_tokens)
    if any(count <= 0 for count in token_counts):
        raise ValueError("every MAGI-2 modality must contain at least one token")
    channel_counts = (video_in_channels, audio_in_channels, text_in_channels)
    if any(count <= 0 for count in channel_counts):
        raise ValueError("every MAGI-2 modality must contain at least one channel")

    generator = torch.Generator().manual_seed(seed + sample_id)
    modality_mapping = torch.tensor(
        [Magi2Modality.VIDEO] * video_tokens
        + [Magi2Modality.AUDIO] * audio_tokens
        + [Magi2Modality.TEXT] * text_tokens,
        dtype=torch.long,
    )
    input_width = max(channel_counts)
    clean_inputs = torch.zeros(modality_mapping.numel(), input_width, dtype=torch.float32)
    modality_specs = (
        (Magi2Modality.VIDEO, video_in_channels, 1.0),
        (Magi2Modality.AUDIO, audio_in_channels, 1.0),
        (Magi2Modality.TEXT, text_in_channels, 0.02),
    )
    for modality, channels, scale in modality_specs:
        selected = modality_mapping == modality
        clean_inputs[selected, :channels] = scale * torch.randn(int(selected.sum()), channels, generator=generator)

    coordinates = torch.cat(tuple(_grid_coordinates(count) for count in token_counts), dim=0)
    return {
        "clean_inputs": clean_inputs,
        "coordinates": coordinates,
        "modality_mapping": modality_mapping,
        "sample_id": torch.tensor(sample_id, dtype=torch.long),
    }


def _validate_sample(sample: dict[str, Tensor]) -> None:
    required = {"clean_inputs", "coordinates", "modality_mapping", "sample_id"}
    if set(sample) != required:
        raise ValueError(f"latent sample keys must be {sorted(required)}, got {sorted(sample)}")
    clean_inputs = sample["clean_inputs"]
    token_count = clean_inputs.shape[0] if clean_inputs.ndim == 2 else -1
    if token_count <= 0:
        raise ValueError("clean_inputs must have shape [tokens, channels] with at least one token")
    if sample["coordinates"].shape != (token_count, 9):
        raise ValueError("coordinates must have shape [tokens, 9]")
    if sample["modality_mapping"].shape != (token_count,):
        raise ValueError("modality_mapping must have shape [tokens]")
    if sample["sample_id"].numel() != 1:
        raise ValueError("sample_id must contain exactly one value")


def _sample_modality_counts(sample: dict[str, Tensor]) -> dict[str, int]:
    mapping = sample["modality_mapping"]
    known_modalities = {
        "video": int(Magi2Modality.VIDEO),
        "audio": int(Magi2Modality.AUDIO),
        "text": int(Magi2Modality.TEXT),
    }
    counts = {name: int((mapping == value).sum().item()) for name, value in known_modalities.items()}
    if any(count <= 0 for count in counts.values()):
        raise ValueError(f"every MAGI-2 latent sample must contain video, audio, and text tokens: {counts}")
    known = torch.zeros_like(mapping, dtype=torch.bool)
    for value in known_modalities.values():
        known |= mapping == value
    if not bool(known.all()):
        raise ValueError("latent samples may only contain video, audio, and text modalities")
    return counts


def write_magi2_precomputed_latent_dataset(
    output_dir: str | Path,
    samples_by_split: Mapping[str, Iterable[dict[str, Tensor]]],
    *,
    channels: Mapping[str, int],
    shard_size: int = 64,
    seed: int = 1234,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Stream precomputed MAGI-2 samples into sharded ``magi2-latent-v1`` files."""
    allowed_splits = ("train", "valid", "test")
    unexpected_splits = set(samples_by_split) - set(allowed_splits)
    if unexpected_splits:
        raise ValueError(f"unsupported MAGI-2 latent splits: {sorted(unexpected_splits)}")
    normalized_channels = {name: int(channels.get(name, 0)) for name in ("video", "audio", "text")}
    if any(width <= 0 for width in normalized_channels.values()):
        raise ValueError(f"latent channel widths must be positive: {normalized_channels}")
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")

    root = Path(output_dir).resolve()
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset manifest: {manifest_path}")
    root.mkdir(parents=True, exist_ok=True)

    input_width = max(normalized_channels.values())
    token_minima = {name: None for name in normalized_channels}
    token_maxima = {name: 0 for name in normalized_channels}
    split_manifests: dict[str, dict[str, Any]] = {}
    seen_sample_ids: set[int] = set()

    for split in allowed_splits:
        if split not in samples_by_split:
            continue
        shard_entries: list[dict[str, Any]] = []
        shard_samples: list[dict[str, Tensor]] = []
        sample_count = 0

        def flush_shard() -> None:
            if not shard_samples:
                return
            shard_index = len(shard_entries)
            shard_name = f"{split}-{shard_index:05d}.pt"
            shard_path = root / shard_name
            if shard_path.exists():
                raise FileExistsError(f"refusing to overwrite existing latent shard: {shard_path}")
            temporary_path = shard_path.with_suffix(".pt.tmp")
            torch.save({"samples": shard_samples}, temporary_path)
            temporary_path.replace(shard_path)
            shard_entries.append({"file": shard_name, "samples": len(shard_samples)})
            shard_samples.clear()

        for sample in samples_by_split[split]:
            _validate_sample(sample)
            if sample["clean_inputs"].shape[1] != input_width:
                raise ValueError(
                    f"sample clean_inputs width {sample['clean_inputs'].shape[1]} "
                    f"does not match channels {normalized_channels}"
                )
            sample_id = int(sample["sample_id"].item())
            if sample_id in seen_sample_ids:
                raise ValueError(f"duplicate MAGI-2 latent sample_id: {sample_id}")
            seen_sample_ids.add(sample_id)
            counts = _sample_modality_counts(sample)
            for modality, count in counts.items():
                current_minimum = token_minima[modality]
                token_minima[modality] = count if current_minimum is None else min(current_minimum, count)
                token_maxima[modality] = max(token_maxima[modality], count)
            shard_samples.append({name: tensor.detach().cpu() for name, tensor in sample.items()})
            sample_count += 1
            if len(shard_samples) == shard_size:
                flush_shard()
        flush_shard()
        if sample_count:
            split_manifests[split] = {"samples": sample_count, "shards": shard_entries}

    if not split_manifests.get("train"):
        raise ValueError("a precomputed MAGI-2 dataset must contain at least one training sample")
    token_ranges = {name: [int(token_minima[name]), token_maxima[name]] for name in normalized_channels}
    manifest: dict[str, Any] = {
        "format": MAGI2_LATENT_DATASET_FORMAT,
        "seed": seed,
        "channels": normalized_channels,
        "token_ranges": token_ranges,
        "max_tokens_per_sample": sum(bounds[1] for bounds in token_ranges.values()),
        "splits": split_manifests,
    }
    if metadata is not None:
        manifest["metadata"] = dict(metadata)
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    return manifest_path


def magi2_latent_collate(samples: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Concatenate variable-length latent samples into one MCore THD batch."""
    if not samples:
        raise ValueError("cannot collate an empty MAGI-2 batch")
    for sample in samples:
        _validate_sample(sample)
    channel_widths = {sample["clean_inputs"].shape[1] for sample in samples}
    if len(channel_widths) != 1:
        raise ValueError("all MAGI-2 samples in a batch must use the same channel width")

    lengths = torch.tensor([sample["clean_inputs"].shape[0] for sample in samples], dtype=torch.int32)
    cu_seqlens = torch.cat((torch.zeros(1, dtype=torch.int32), lengths.cumsum(dim=0)))
    return {
        "clean_inputs": torch.cat([sample["clean_inputs"] for sample in samples], dim=0),
        "coordinates": torch.cat([sample["coordinates"] for sample in samples], dim=0),
        "modality_mapping": torch.cat([sample["modality_mapping"] for sample in samples], dim=0),
        "sample_id": torch.stack([sample["sample_id"].reshape(()) for sample in samples]),
        "cu_seqlens": cu_seqlens,
    }


class Magi2LatentDataset(Dataset):
    """Generate deterministic clean latents and conditioning embeddings in memory."""

    collate_fn = staticmethod(magi2_latent_collate)

    def __init__(
        self,
        size: int,
        *,
        seed: int,
        video_tokens: int,
        audio_tokens: int,
        text_tokens: int,
        video_in_channels: int,
        audio_in_channels: int,
        text_in_channels: int,
    ) -> None:
        self.size = size
        self.seed = seed
        self.video_tokens = video_tokens
        self.audio_tokens = audio_tokens
        self.text_tokens = text_tokens
        self.video_in_channels = video_in_channels
        self.audio_in_channels = audio_in_channels
        self.text_in_channels = text_in_channels

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if not 0 <= index < self.size:
            raise IndexError(index)
        return generate_magi2_latent_sample(
            index,
            seed=self.seed,
            video_tokens=self.video_tokens,
            audio_tokens=self.audio_tokens,
            text_tokens=self.text_tokens,
            video_in_channels=self.video_in_channels,
            audio_in_channels=self.audio_in_channels,
            text_in_channels=self.text_in_channels,
        )


class Magi2DiskLatentDataset(Dataset):
    """Read one split from a sharded ``magi2-latent-v1`` dataset."""

    collate_fn = staticmethod(magi2_latent_collate)

    def __init__(self, root: str | Path, split: str) -> None:
        self.root = Path(root).resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"MAGI-2 latent manifest does not exist: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != MAGI2_LATENT_DATASET_FORMAT:
            raise ValueError(
                f"unsupported MAGI-2 latent format {manifest.get('format')!r}; "
                f"expected {MAGI2_LATENT_DATASET_FORMAT!r}"
            )
        split_info = manifest.get("splits", {}).get(split)
        if not isinstance(split_info, dict):
            raise ValueError(f"MAGI-2 latent manifest has no {split!r} split")
        shards = split_info.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ValueError(f"MAGI-2 latent split {split!r} contains no shards")

        self.split = split
        self.manifest = manifest
        self._shards: list[tuple[Path, int]] = []
        self._shard_ends: list[int] = []
        total = 0
        for shard in shards:
            if not isinstance(shard, dict) or not isinstance(shard.get("file"), str):
                raise ValueError(f"invalid shard entry in split {split!r}: {shard!r}")
            sample_count = int(shard.get("samples", 0))
            if sample_count <= 0:
                raise ValueError(f"shard {shard['file']!r} must contain at least one sample")
            shard_path = (self.root / shard["file"]).resolve()
            try:
                shard_path.relative_to(self.root)
            except ValueError as error:
                raise ValueError(f"shard path escapes dataset root: {shard['file']!r}") from error
            if not shard_path.is_file():
                raise ValueError(f"MAGI-2 latent shard does not exist: {shard_path}")
            total += sample_count
            self._shards.append((shard_path, sample_count))
            self._shard_ends.append(total)
        if int(split_info.get("samples", -1)) != total:
            raise ValueError(f"split {split!r} sample count does not match its shards")
        self._size = total
        self._cached_shard_index: int | None = None
        self._cached_samples: list[dict[str, Tensor]] | None = None

    def __len__(self) -> int:
        return self._size

    def _load_shard(self, shard_index: int) -> list[dict[str, Tensor]]:
        if self._cached_shard_index == shard_index and self._cached_samples is not None:
            return self._cached_samples
        shard_path, expected_samples = self._shards[shard_index]
        payload = torch.load(shard_path, map_location="cpu", weights_only=True)
        samples = payload.get("samples") if isinstance(payload, dict) else None
        if not isinstance(samples, list) or len(samples) != expected_samples:
            raise ValueError(f"latent shard has an invalid sample payload: {shard_path}")
        for sample in samples:
            if not isinstance(sample, dict):
                raise ValueError(f"latent shard contains a non-mapping sample: {shard_path}")
            _validate_sample(sample)
        self._cached_shard_index = shard_index
        self._cached_samples = samples
        return samples

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if index < 0:
            index += self._size
        if not 0 <= index < self._size:
            raise IndexError(index)
        shard_index = bisect_right(self._shard_ends, index)
        shard_start = 0 if shard_index == 0 else self._shard_ends[shard_index - 1]
        return self._load_shard(shard_index)[index - shard_start]


def _sample_token_count(generator: torch.Generator, bounds: tuple[int, int]) -> int:
    lower, upper = bounds
    if lower <= 0 or upper < lower:
        raise ValueError(f"invalid token-count range {bounds}")
    return int(torch.randint(lower, upper + 1, (), generator=generator).item())


def write_magi2_latent_dataset(
    output_dir: str | Path,
    *,
    train_samples: int,
    valid_samples: int,
    test_samples: int = 0,
    shard_size: int = 64,
    seed: int = 1234,
    video_token_range: tuple[int, int] = (2, 4),
    audio_token_range: tuple[int, int] = (2, 4),
    text_token_range: tuple[int, int] = (1, 3),
    video_in_channels: int = 48,
    audio_in_channels: int = 64,
    text_in_channels: int = 5120,
) -> Path:
    """Materialize a deterministic, variable-length MAGI-2 latent dataset."""
    if train_samples <= 0 or valid_samples < 0 or test_samples < 0:
        raise ValueError("train_samples must be positive and other split sizes must be non-negative")
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    root = Path(output_dir).resolve()
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset manifest: {manifest_path}")
    root.mkdir(parents=True, exist_ok=True)

    split_sizes = {"train": train_samples, "valid": valid_samples, "test": test_samples}
    split_offsets = {"train": 0, "valid": 1_000_000, "test": 2_000_000}
    split_manifests: dict[str, dict[str, Any]] = {}
    token_ranges = (video_token_range, audio_token_range, text_token_range)
    for split, size in split_sizes.items():
        if size == 0:
            continue
        shard_entries = []
        for shard_start in range(0, size, shard_size):
            shard_end = min(shard_start + shard_size, size)
            samples = []
            for split_index in range(shard_start, shard_end):
                sample_id = split_offsets[split] + split_index
                count_generator = torch.Generator().manual_seed(seed + sample_id * 97)
                video_tokens, audio_tokens, text_tokens = (
                    _sample_token_count(count_generator, bounds) for bounds in token_ranges
                )
                samples.append(
                    generate_magi2_latent_sample(
                        sample_id,
                        seed=seed,
                        video_tokens=video_tokens,
                        audio_tokens=audio_tokens,
                        text_tokens=text_tokens,
                        video_in_channels=video_in_channels,
                        audio_in_channels=audio_in_channels,
                        text_in_channels=text_in_channels,
                    )
                )
            shard_index = shard_start // shard_size
            shard_name = f"{split}-{shard_index:05d}.pt"
            shard_path = root / shard_name
            if shard_path.exists():
                raise FileExistsError(f"refusing to overwrite existing latent shard: {shard_path}")
            temporary_path = shard_path.with_suffix(".pt.tmp")
            torch.save({"samples": samples}, temporary_path)
            temporary_path.replace(shard_path)
            shard_entries.append({"file": shard_name, "samples": len(samples)})
        split_manifests[split] = {"samples": size, "shards": shard_entries}

    manifest = {
        "format": MAGI2_LATENT_DATASET_FORMAT,
        "seed": seed,
        "channels": {
            "video": video_in_channels,
            "audio": audio_in_channels,
            "text": text_in_channels,
        },
        "token_ranges": {
            "video": list(video_token_range),
            "audio": list(audio_token_range),
            "text": list(text_token_range),
        },
        "max_tokens_per_sample": sum(bounds[1] for bounds in token_ranges),
        "splits": split_manifests,
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    return manifest_path


@dataclass(kw_only=True)
class Magi2LatentDatasetConfig(DatasetProvider):
    """Dataset provider for deterministic MAGI-2 flow-matching batches."""

    data_path: Path | None = None
    seed: int = 1234
    video_tokens: int = 4
    audio_tokens: int = 4
    text_tokens: int = 3
    time_channel_dim: int = 64
    video_in_channels: int = 48
    audio_in_channels: int = 64
    text_in_channels: int = 5120
    dataloader_type: Optional[str] = "cyclic"
    num_workers: int = 0
    persistent_workers: bool = False
    pin_memory: bool = True

    def build_datasets(
        self, context: DatasetBuildContext
    ) -> tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
        """Build disk-backed datasets when configured, otherwise deterministic fallback splits."""
        if self.data_path is not None:

            def build_disk(split: str, requested_samples: int) -> Optional[Magi2DiskLatentDataset]:
                if requested_samples <= 0:
                    return None
                return Magi2DiskLatentDataset(self.data_path, split)

            return (
                build_disk("train", context.train_samples),
                build_disk("valid", context.valid_samples),
                build_disk("test", context.test_samples),
            )

        def build(size: int, seed_offset: int) -> Optional[Magi2LatentDataset]:
            if size <= 0:
                return None
            return Magi2LatentDataset(
                size,
                seed=self.seed + seed_offset,
                video_tokens=self.video_tokens,
                audio_tokens=self.audio_tokens,
                text_tokens=self.text_tokens,
                video_in_channels=self.video_in_channels,
                audio_in_channels=self.audio_in_channels,
                text_in_channels=self.text_in_channels,
            )

        return build(context.train_samples, 0), build(context.valid_samples, 1_000_000), None


__all__ = [
    "MAGI2_LATENT_DATASET_FORMAT",
    "Magi2DiskLatentDataset",
    "Magi2LatentDataset",
    "Magi2LatentDatasetConfig",
    "generate_magi2_latent_sample",
    "magi2_latent_collate",
    "write_magi2_precomputed_latent_dataset",
    "write_magi2_latent_dataset",
]
