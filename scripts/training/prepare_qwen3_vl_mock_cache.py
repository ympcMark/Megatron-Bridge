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

"""Prepare one reusable Qwen3-VL mock microbatch."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoProcessor

from megatron.bridge.data.vlm_datasets.mock_provider import MockVLMConversationProvider
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.bridge.models.qwen_vl.data.collate_fn import qwen2_5_collate_fn


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", required=True, help="Local path or Hugging Face identifier for the processor")
    parser.add_argument("--output", required=True, type=Path, help="Output .pt file")
    parser.add_argument("--seq-length", required=True, type=int)
    parser.add_argument("--micro-batch-size", required=True, type=int)
    return parser.parse_args()


def prepare_cache(args: argparse.Namespace) -> None:
    """Generate one cached mock microbatch unless it already exists.

    Args:
        args: Parsed cache-generation arguments.
    """
    if args.output.is_file():
        logger.info("Reusing mock cache: %s", args.output)
        return

    provider = MockVLMConversationProvider(
        seq_length=args.seq_length,
        hf_processor_path=args.hf_path,
    )
    processor = AutoProcessor.from_pretrained(
        args.hf_path,
        trust_remote_code=is_safe_repo(
            trust_remote_code=provider.trust_remote_code,
            hf_path=args.hf_path,
        ),
    )
    batch = qwen2_5_collate_fn(provider._make_base_examples(args.micro_batch_size), processor)
    batch["visual_inputs"] = batch["visual_inputs"].as_model_kwargs()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(batch, args.output)
    logger.info("Mock cache ready: %s", args.output)


def main() -> None:
    """Prepare the requested mock cache."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    prepare_cache(parse_args())


if __name__ == "__main__":
    main()
