#!/usr/bin/env python3
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

"""Encode MP4/WAV/text records with the official MAGI-2 encoders."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterator
from pathlib import Path

import torch

from megatron.bridge.diffusion.models.magi2.data import write_magi2_precomputed_latent_dataset
from megatron.bridge.diffusion.models.magi2.media_preprocessing import (
    MAGI2_AUDIO_CHANNELS,
    MAGI2_TEXT_CHANNELS,
    MAGI2_VIDEO_CHANNELS,
    Magi2MediaRecord,
    OfficialMagi2Encoders,
    encode_magi2_media_record,
    read_magi2_media_manifest,
)


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse raw-media preprocessing arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True, help="JSONL with video, audio, text, and split")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output magi2-latent-v1 directory")
    parser.add_argument("--official-repo", type=Path, required=True, help="Checkout of SandAI-org/MAGI-2-preview")
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        required=True,
        help="Root containing vae/, text_encoder/, and stable-audio-open-1.0/",
    )
    parser.add_argument("--device", default="cuda", help="CUDA device used by the official encoders")
    parser.add_argument("--num-frames", type=int, default=17, help="Frames per clip; must be 8k+1")
    parser.add_argument("--fps", type=float, default=25.0, help="Video/audio sampling rate used by the data proxy")
    parser.add_argument("--height", type=int, default=320, help="Center-crop height, divisible by 32")
    parser.add_argument("--width", type=int, default=320, help="Center-crop width, divisible by 32")
    parser.add_argument("--text-max-length", type=int, default=7000)
    parser.add_argument("--shard-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234, help="Training noise seed recorded in the latent manifest")
    parser.add_argument("--storage-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    return parser.parse_args()


def main() -> None:
    """Encode the input manifest and write a sharded latent dataset."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    records = read_magi2_media_manifest(args.input_manifest)
    split_records = {
        split: [record for record in records if record.split == split] for split in ("train", "valid", "test")
    }
    split_records = {split: items for split, items in split_records.items() if items}
    logger.info("Loading official MAGI-2 video, audio, and text encoders")
    encoders = OfficialMagi2Encoders(
        args.official_repo,
        args.checkpoint_root,
        device=args.device,
        text_max_length=args.text_max_length,
    )
    storage_dtype = torch.bfloat16 if args.storage_dtype == "bfloat16" else torch.float32

    def encode_records(items: list[Magi2MediaRecord]) -> Iterator[dict[str, torch.Tensor]]:
        for index, record in enumerate(items, start=1):
            logger.info("Encoding sample_id=%d split=%s (%d/%d)", record.sample_id, record.split, index, len(items))
            yield encode_magi2_media_record(
                record,
                encoders,
                num_frames=args.num_frames,
                fps=args.fps,
                height=args.height,
                width=args.width,
                storage_dtype=storage_dtype,
            )

    samples_by_split = {split: encode_records(items) for split, items in split_records.items()}
    manifest = write_magi2_precomputed_latent_dataset(
        args.output_dir,
        samples_by_split,
        channels={
            "video": MAGI2_VIDEO_CHANNELS,
            "audio": MAGI2_AUDIO_CHANNELS,
            "text": MAGI2_TEXT_CHANNELS,
        },
        shard_size=args.shard_size,
        seed=args.seed,
        metadata={
            "source": "official-magi2-media-encoders",
            "input_manifest": str(args.input_manifest.resolve()),
            "official_repo": str(args.official_repo.resolve()),
            "checkpoint_root": str(args.checkpoint_root.resolve()),
            "num_frames": args.num_frames,
            "fps": args.fps,
            "height": args.height,
            "width": args.width,
            "storage_dtype": args.storage_dtype,
        },
    )
    logger.info("Wrote official MAGI-2 latent manifest to %s", manifest)


if __name__ == "__main__":
    main()
