# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Prepare one reusable, fully collated Qwen3-VL mock microbatch."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoProcessor

from megatron.bridge.data.builders.mock_vlm_sft import MockVLMSFTDatasetConfig, make_mock_vlm_examples
from megatron.bridge.data.datasets.direct_sft import DirectSFTDataset
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse cache-generation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seq-length", required=True, type=int)
    parser.add_argument("--micro-batch-size", required=True, type=int)
    parser.add_argument("--ratio", required=True, type=float)
    parser.add_argument("--image-size", type=int, nargs=2, default=(768, 768), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def prepare_cache(args: argparse.Namespace) -> None:
    """Generate and save one processor/collator-complete CPU microbatch."""
    metadata = {
        "seq_length": args.seq_length,
        "micro_batch_size": args.micro_batch_size,
        "ratio": args.ratio,
        "image_size": list(args.image_size),
        "num_images": args.num_images,
    }
    if args.output.is_file() and not args.force:
        payload = torch.load(args.output, map_location="cpu", weights_only=True)
        cached_metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        if cached_metadata == metadata:
            logger.info("Reusing matching mock cache: %s", args.output)
            return
        raise ValueError(
            f"Existing cache metadata differs: expected {metadata}, got {cached_metadata}. "
            "Use --force or choose another output path."
        )

    config = MockVLMSFTDatasetConfig(
        seq_length=args.seq_length,
        hf_processor_path=args.hf_path,
        ratio=args.ratio,
        image_size=tuple(args.image_size),
        num_images=args.num_images,
        num_base_examples=args.micro_batch_size,
        pad_to_max_length=True,
    )
    config.validate()
    processor = AutoProcessor.from_pretrained(
        args.hf_path,
        trust_remote_code=is_safe_repo(
            trust_remote_code=config.trust_remote_code,
            hf_path=args.hf_path,
        ),
    )
    examples = make_mock_vlm_examples(config)
    dataset = DirectSFTDataset(
        base_examples=examples,
        target_length=args.micro_batch_size,
        processor=processor,
        sequence_length=args.seq_length,
        pad_to_max_length=True,
    )
    batch = dataset.collate_fn(examples[: args.micro_batch_size])
    visual_inputs = batch.get("visual_inputs")
    if visual_inputs is not None and hasattr(visual_inputs, "as_model_kwargs"):
        batch["visual_inputs"] = visual_inputs.as_model_kwargs()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "batch": batch}, args.output)
    image_count = 0
    visual_inputs_dict = batch.get("visual_inputs")
    if isinstance(visual_inputs_dict, dict):
        grid = visual_inputs_dict.get("image_grid_thw")
        image_count = int(grid.shape[0]) if isinstance(grid, torch.Tensor) else 0
    logger.info(
        "Mock cache ready: %s; input_ids=%s; images=%d",
        args.output,
        tuple(batch["input_ids"].shape),
        image_count,
    )


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    prepare_cache(parse_args())


if __name__ == "__main__":
    main()
