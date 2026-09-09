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

"""Qwen VL collator implementations."""

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from megatron.bridge.data.collators.sequence import prepare_sequence_batch
from megatron.bridge.data.collators.sequence_padding import use_processor_right_padding
from megatron.bridge.data.collators.visual import THW_GRID_VISUAL_KEYS
from megatron.bridge.data.conversation_processing import (
    assistant_mask_boundary_config_from_markers,
    build_assistant_loss_mask,
    chat_template_kwargs_from_example,
)
from megatron.bridge.data.datasets.utils import IGNORE_INDEX
from megatron.bridge.data.packing.in_batch import build_mcore_thd_sequence_batch_from_rows
from megatron.bridge.data.token_utils import extract_skipped_token_ids
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.rope import get_rope_index
from megatron.bridge.training.utils.visual_inputs import GenericVisualInputs


MISSING_QWEN_VL_UTILS_MSG = (
    "qwen_vl_utils is required for Qwen2.5 VL processing. Please `pip install qwen-vl-utils` or"
    " provide compatible vision preprocessing."
)
QWEN_VL_MIN_PIXELS = 200704
QWEN_VL_MAX_PIXELS = 1003520
CHATML_ASSISTANT_START = "<|im_start|>assistant\n"
CHATML_ASSISTANT_END = "<|im_end|>\n"
CHATML_OTHER_ROLE_STARTS = {role: f"<|im_start|>{role}\n" for role in ("system", "developer", "user", "tool")}
QWEN_VISUAL_KEYS = (*THW_GRID_VISUAL_KEYS, "second_per_grid_ts")
_QWEN_SPECIAL_TOKEN_DEFAULTS = {
    "image_token_id": ("<|image_pad|>", 151655),
    "video_token_id": ("<|video_pad|>", 151656),
    "vision_start_token_id": ("<|vision_start|>", 151652),
}

try:
    from qwen_vl_utils import process_vision_info

    HAVE_QWEN_VL_UTILS = True
except ImportError:
    HAVE_QWEN_VL_UTILS = False


def _resolve_qwen_special_token_id(processor: Any, attribute: str) -> int:
    """Resolve a Qwen visual token ID without tying the collator to one model size."""
    tokenizer = getattr(processor, "tokenizer", processor)
    for owner in (processor, tokenizer):
        value = getattr(owner, attribute, None)
        if isinstance(value, int):
            return value

    token, fallback = _QWEN_SPECIAL_TOKEN_DEFAULTS[attribute]
    convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert_tokens_to_ids):
        value = convert_tokens_to_ids(token)
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        if isinstance(value, int) and value >= 0 and value != unk_token_id:
            return value
    return fallback


def _qwen_spatial_merge_size(processor: Any) -> int:
    image_processor = getattr(processor, "image_processor", None)
    for owner in (image_processor, processor):
        for attribute in ("merge_size", "spatial_merge_size"):
            value = getattr(owner, attribute, None)
            if isinstance(value, int) and value > 0:
                return value
    return 2


def _flatten_grid_thw(grid_thw: torch.Tensor | None) -> torch.Tensor | None:
    if not isinstance(grid_thw, torch.Tensor) or grid_thw.numel() == 0:
        return None
    return grid_thw.reshape(-1, grid_thw.size(-1)).contiguous()


def _build_cpu_mrope_position_ids(
    processor: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    image_grid_thw: torch.Tensor | None,
    video_grid_thw: torch.Tensor | None,
) -> torch.Tensor:
    """Build explicit Qwen M-RoPE IDs while token/grid metadata is still on CPU."""
    if input_ids.device.type != "cpu":
        raise ValueError("Collate-time M-RoPE construction expects CPU input_ids.")
    position_ids, _ = get_rope_index(
        spatial_merge_size=_qwen_spatial_merge_size(processor),
        image_token_id=_resolve_qwen_special_token_id(processor, "image_token_id"),
        video_token_id=_resolve_qwen_special_token_id(processor, "video_token_id"),
        vision_start_token_id=_resolve_qwen_special_token_id(processor, "vision_start_token_id"),
        input_ids=input_ids.unsqueeze(0),
        image_grid_thw=_flatten_grid_thw(image_grid_thw),
        video_grid_thw=_flatten_grid_thw(video_grid_thw),
        attention_mask=attention_mask.unsqueeze(0) if attention_mask is not None else None,
    )
    return position_ids[:, 0].contiguous()


def _build_cpu_mrope_batch(
    processor: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    image_grid_thw: torch.Tensor | None,
    video_grid_thw: torch.Tensor | None,
    image_counts: list[int],
    video_counts: list[int],
) -> torch.Tensor:
    """Build ``[3, B, S]`` position IDs and preserve per-row media ownership."""
    image_grid_thw = _flatten_grid_thw(image_grid_thw)
    video_grid_thw = _flatten_grid_thw(video_grid_thw)
    image_offset = 0
    video_offset = 0
    rows = []
    for row_idx, (image_count, video_count) in enumerate(zip(image_counts, video_counts, strict=True)):
        row_image_grid = (
            image_grid_thw[image_offset : image_offset + image_count]
            if image_count and image_grid_thw is not None
            else None
        )
        row_video_grid = (
            video_grid_thw[video_offset : video_offset + video_count]
            if video_count and video_grid_thw is not None
            else None
        )
        row_attention_mask = attention_mask[row_idx] if attention_mask is not None else None
        rows.append(
            _build_cpu_mrope_position_ids(
                processor,
                input_ids[row_idx],
                row_attention_mask,
                row_image_grid,
                row_video_grid,
            )
        )
        image_offset += image_count
        video_offset += video_count

    if image_grid_thw is not None and image_offset != image_grid_thw.size(0):
        raise ValueError("image_grid_thw rows do not match the images assigned to collated examples.")
    if video_grid_thw is not None and video_offset != video_grid_thw.size(0):
        raise ValueError("video_grid_thw rows do not match the videos assigned to collated examples.")
    return torch.stack(rows, dim=1).contiguous()


def _normalize_qwen_video_paths(example: dict[str, Any]) -> dict[str, Any]:
    """Map path-based video parts to the inline schema expected by Qwen processors."""
    conversation = example.get("conversation")
    if not isinstance(conversation, list):
        return example

    normalized_conversation = []
    example_changed = False
    for turn in conversation:
        if not isinstance(turn, Mapping) or not isinstance(turn.get("content"), list):
            normalized_conversation.append(turn)
            continue

        normalized_content = []
        turn_changed = False
        for part in turn["content"]:
            if isinstance(part, Mapping) and part.get("type") == "video" and "video" not in part and "path" in part:
                normalized_part = dict(part)
                normalized_part["video"] = normalized_part.pop("path")
                normalized_content.append(normalized_part)
                turn_changed = True
            else:
                normalized_content.append(part)

        if turn_changed:
            normalized_turn = dict(turn)
            normalized_turn["content"] = normalized_content
            normalized_conversation.append(normalized_turn)
            example_changed = True
        else:
            normalized_conversation.append(turn)

    if not example_changed:
        return example
    normalized_example = dict(example)
    normalized_example["conversation"] = normalized_conversation
    return normalized_example


def qwen2_5_collate_fn(
    examples: list,
    processor,
    min_pixels: int | None = QWEN_VL_MIN_PIXELS,
    max_pixels: int | None = QWEN_VL_MAX_PIXELS,
    visual_keys: object = None,
    require_assistant_matches: bool = False,
    sequence_length: int | None = None,
    pad_to_max_length: bool = False,
    pad_to_multiple_of: int = 128,
    enable_in_batch_packing: bool = False,
    in_batch_packing_pad_to_multiple_of: int = 1,
) -> dict[str, torch.Tensor]:
    """Collate function for Qwen2.5 VL model."""
    del visual_keys

    if not HAVE_QWEN_VL_UTILS:
        raise ImportError(MISSING_QWEN_VL_UTILS_MSG)

    examples = [_normalize_qwen_video_paths(example) for example in examples]
    skipped_tokens = extract_skipped_token_ids(processor)
    boundary_config = assistant_mask_boundary_config_from_markers(
        processor,
        assistant_start=CHATML_ASSISTANT_START,
        assistant_end=CHATML_ASSISTANT_END,
        assistant_end_fallbacks=("<|im_end|>",),
        role_start_markers=CHATML_OTHER_ROLE_STARTS,
    )

    texts = [
        processor.apply_chat_template(
            example["conversation"],
            tokenize=False,
            **chat_template_kwargs_from_example(example),
        )
        for example in examples
    ]
    # Build per-example media (list) and split by presence.  Qwen processors accept
    # nested per-example image/video lists; splitting avoids passing empty media
    # kwargs for text-only rows.
    per_example_images = []
    per_example_videos = []
    has_media = []

    for example in examples:
        imgs, videos = process_vision_info(example["conversation"])
        if imgs is None:
            imgs = []
        elif not isinstance(imgs, list):
            imgs = [imgs]
        if videos is None:
            videos = []
        elif not isinstance(videos, list):
            videos = [videos]
        per_example_images.append(imgs)
        per_example_videos.append(videos)
        has_media.append(len(imgs) > 0 or len(videos) > 0)

    if enable_in_batch_packing:
        sequence_rows = []
        visual_values: dict[str, list[torch.Tensor]] = {key: [] for key in QWEN_VISUAL_KEYS}
        with use_processor_right_padding(processor):
            for example, text, images, videos in zip(
                examples, texts, per_example_images, per_example_videos, strict=True
            ):
                processor_kwargs = {
                    "text": [text],
                    "padding": False,
                    "return_tensors": "pt",
                }
                if min_pixels is not None:
                    processor_kwargs["min_pixels"] = min_pixels
                if max_pixels is not None:
                    processor_kwargs["max_pixels"] = max_pixels
                if images:
                    processor_kwargs["images"] = images
                if videos:
                    processor_kwargs["videos"] = videos

                sample_batch = {
                    key: value.contiguous() if isinstance(value, torch.Tensor) else value
                    for key, value in processor(**processor_kwargs).items()
                }
                input_ids = sample_batch["input_ids"][0]
                attention_mask = sample_batch.get("attention_mask")
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids)
                else:
                    attention_mask = attention_mask[0]
                position_ids = _build_cpu_mrope_position_ids(
                    processor,
                    input_ids,
                    attention_mask,
                    sample_batch.get("image_grid_thw"),
                    sample_batch.get("video_grid_thw"),
                )

                loss_mask = build_assistant_loss_mask(
                    example,
                    input_ids,
                    processor,
                    skipped_tokens,
                    boundary_config=boundary_config,
                    warn_on_all_masked=not require_assistant_matches,
                ).to(device=input_ids.device, dtype=torch.float32)
                labels = torch.cat([input_ids[1:], input_ids.new_full((1,), IGNORE_INDEX)])
                if skipped_tokens.numel() > 0:
                    labels = labels.masked_fill(
                        torch.isin(labels, skipped_tokens.to(device=labels.device)), IGNORE_INDEX
                    )
                shifted_loss_mask = torch.cat([loss_mask[1:], loss_mask.new_zeros(1)])
                sequence_rows.append(
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "position_ids": position_ids,
                        "labels": labels.masked_fill(shifted_loss_mask == 0, IGNORE_INDEX),
                        "loss_mask": shifted_loss_mask,
                    }
                )

                for key in QWEN_VISUAL_KEYS:
                    value = sample_batch.get(key)
                    if not isinstance(value, torch.Tensor):
                        continue
                    if key in {"pixel_values", "pixel_values_videos"} and value.dim() == 5:
                        value = value.flatten(0, 1)
                    elif key in {"image_grid_thw", "video_grid_thw"} and value.dim() == 3:
                        value = value.flatten(0, 1)
                    visual_values[key].append(value)

        packed_batch = build_mcore_thd_sequence_batch_from_rows(
            sequence_rows,
            sequence_length=sequence_length,
            pad_token_id=0,
            ignore_index=IGNORE_INDEX,
            pad_to_multiple_of=in_batch_packing_pad_to_multiple_of,
        )
        packed_batch["visual_inputs"] = GenericVisualInputs(
            **{key: torch.cat(values, dim=0) for key, values in visual_values.items() if values}
        )
        return packed_batch

    idx_with = [i for i, h in enumerate(has_media) if h]
    idx_without = [i for i, h in enumerate(has_media) if not h]

    batch_with = None
    batch_without = None

    with use_processor_right_padding(processor):
        if idx_with:
            texts_with = [texts[i] for i in idx_with]
            images_with = [per_example_images[i] for i in idx_with]
            videos_with = [per_example_videos[i] for i in idx_with]
            processor_kwargs = {
                "text": texts_with,
                "padding": True,
                "return_tensors": "pt",
            }
            if min_pixels is not None:
                processor_kwargs["min_pixels"] = min_pixels
            if max_pixels is not None:
                processor_kwargs["max_pixels"] = max_pixels
            if any(images_with):
                processor_kwargs["images"] = images_with
            if any(videos_with):
                processor_kwargs["videos"] = videos_with
            batch_with = {
                key: value.contiguous() if isinstance(value, torch.Tensor) else value
                for key, value in processor(**processor_kwargs).items()
            }

        if idx_without:
            texts_without = [texts[i] for i in idx_without]
            batch_without = {
                key: value.contiguous() if isinstance(value, torch.Tensor) else value
                for key, value in processor(
                    text=texts_without,
                    padding=True,
                    return_tensors="pt",
                ).items()
            }
    # Merge batches back to original order
    if batch_with is not None and batch_without is None:
        batch = batch_with
    elif batch_with is None and batch_without is not None:
        batch = batch_without
    else:
        # Both exist: pad to common max length and interleave rows
        pad_id = getattr(processor.tokenizer, "pad_token_id", 0) or 0
        in_with = batch_with["input_ids"]
        in_without = batch_without["input_ids"]
        max_len = max(in_with.shape[1], in_without.shape[1])

        def pad_to(x, tgt_len, value):
            if x.shape[1] == tgt_len:
                return x
            pad_len = tgt_len - x.shape[1]
            return F.pad(x, (0, pad_len), value=value)

        in_with = pad_to(in_with, max_len, pad_id)
        in_without = pad_to(in_without, max_len, pad_id)

        input_ids = torch.full((len(examples), max_len), pad_id, dtype=in_with.dtype)
        # Place rows
        for row, i in enumerate(idx_with):
            input_ids[i] = in_with[row]
        for row, i in enumerate(idx_without):
            input_ids[i] = in_without[row]

        batch = {"input_ids": input_ids}
        if "attention_mask" in batch_with and "attention_mask" in batch_without:
            attn_with = pad_to(batch_with["attention_mask"], max_len, 0)
            attn_without = pad_to(batch_without["attention_mask"], max_len, 0)
            attention_mask = torch.zeros((len(examples), max_len), dtype=attn_with.dtype)
            for row, i in enumerate(idx_with):
                attention_mask[i] = attn_with[row]
            for row, i in enumerate(idx_without):
                attention_mask[i] = attn_without[row]
            batch["attention_mask"] = attention_mask
        # Carry over vision tensors if present
        for key in QWEN_VISUAL_KEYS:
            if key in batch_with:
                batch[key] = batch_with[key]

    batch["position_ids"] = _build_cpu_mrope_batch(
        processor,
        batch["input_ids"],
        batch.get("attention_mask"),
        batch.get("image_grid_thw"),
        batch.get("video_grid_thw"),
        [len(images) for images in per_example_images],
        [len(videos) for videos in per_example_videos],
    )

    loss_mask = torch.stack(
        [
            build_assistant_loss_mask(
                example,
                input_ids,
                processor,
                skipped_tokens,
                boundary_config=boundary_config,
                warn_on_all_masked=not require_assistant_matches,
            )
            for example, input_ids in zip(examples, batch["input_ids"])
        ]
    ).to(device=batch["input_ids"].device, dtype=torch.float32)
    labels = batch["input_ids"].clone()[:, 1:].contiguous()
    labels = torch.cat([labels, IGNORE_INDEX * torch.ones_like(labels[:, :1])], dim=1)
    if skipped_tokens.numel() > 0:
        labels = labels.masked_fill(torch.isin(labels, skipped_tokens.to(device=labels.device)), IGNORE_INDEX)
    loss_mask = torch.cat([loss_mask[:, 1:], torch.zeros_like(loss_mask[:, :1])], dim=1)
    batch["labels"] = labels.masked_fill(loss_mask == 0, IGNORE_INDEX)
    batch["loss_mask"] = loss_mask

    visual_inputs = GenericVisualInputs(
        pixel_values=batch.get("pixel_values"),
        pixel_values_videos=batch.get("pixel_values_videos"),
        image_grid_thw=batch.get("image_grid_thw"),
        video_grid_thw=batch.get("video_grid_thw"),
        second_per_grid_ts=batch.get("second_per_grid_ts"),
    )
    for key in QWEN_VISUAL_KEYS:
        batch.pop(key, None)
    batch["visual_inputs"] = visual_inputs
    prepare_sequence_batch(
        batch,
        sequence_length=sequence_length,
        pad_to_max_length=pad_to_max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        enable_in_batch_packing=enable_in_batch_packing,
        in_batch_packing_pad_to_multiple_of=in_batch_packing_pad_to_multiple_of,
        ignore_index=IGNORE_INDEX,
    )
    return batch
