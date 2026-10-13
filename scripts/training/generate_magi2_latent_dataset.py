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

"""Generate a deterministic disk-backed latent dataset for MAGI-2 bring-up."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from megatron.bridge.diffusion.models.magi2.data import write_magi2_latent_dataset


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse dataset generation arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--valid-samples", type=int, default=16)
    parser.add_argument("--test-samples", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--video-tokens", type=int, nargs=2, metavar=("MIN", "MAX"), default=(2, 4))
    parser.add_argument("--audio-tokens", type=int, nargs=2, metavar=("MIN", "MAX"), default=(2, 4))
    parser.add_argument("--text-tokens", type=int, nargs=2, metavar=("MIN", "MAX"), default=(1, 3))
    parser.add_argument("--video-in-channels", type=int, default=48)
    parser.add_argument("--audio-in-channels", type=int, default=64)
    parser.add_argument("--text-in-channels", type=int, default=5120)
    return parser.parse_args()


def main() -> None:
    """Write latent shards and their manifest."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    manifest = write_magi2_latent_dataset(
        args.output_dir,
        train_samples=args.train_samples,
        valid_samples=args.valid_samples,
        test_samples=args.test_samples,
        shard_size=args.shard_size,
        seed=args.seed,
        video_token_range=tuple(args.video_tokens),
        audio_token_range=tuple(args.audio_tokens),
        text_token_range=tuple(args.text_tokens),
        video_in_channels=args.video_in_channels,
        audio_in_channels=args.audio_in_channels,
        text_in_channels=args.text_in_channels,
    )
    logger.info("Wrote MAGI-2 latent manifest to %s", manifest)


if __name__ == "__main__":
    main()
