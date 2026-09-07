# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from collections.abc import Callable
from typing import Optional

import torch
from megatron.core import InferenceParams
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.spec_utils import ModuleSpec
from torch import nn
from torch.nn import functional as F

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_block import Qwen3VLVisionTransformerBlock
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_config import Qwen3VLTransformerConfig
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import (
    AllGatherVisionEmbeddings,
    Qwen3VLVisionPatchEmbed,
    Qwen3VLVisionPatchMerger,
    Qwen3VLVisionRotaryEmbedding,
)


class _FrozenVisionCudaGraphEntry:
    """Static buffers and captured outputs for one exact packed signature."""

    def __init__(
        self,
        graph: torch.cuda.CUDAGraph,
        hidden_states: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        packed_seq_params: PackedSeqParams,
        output: torch.Tensor,
        deepstack_outputs: list[torch.Tensor],
    ) -> None:
        self.graph = graph
        self.hidden_states = hidden_states
        self.rotary_pos_emb = rotary_pos_emb
        self.packed_seq_params = packed_seq_params
        self.output = output
        self.deepstack_outputs = deepstack_outputs

    def copy_inputs(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        packed_seq_params: PackedSeqParams,
    ) -> None:
        self.hidden_states.copy_(hidden_states)
        self.rotary_pos_emb.copy_(rotary_pos_emb)
        for name in (
            "cu_seqlens_q",
            "cu_seqlens_kv",
            "cu_seqlens_q_padded",
            "cu_seqlens_kv_padded",
            "seq_idx",
        ):
            source = getattr(packed_seq_params, name)
            target = getattr(self.packed_seq_params, name)
            if source is not None and target is not None:
                target.copy_(source)


def _clone_packed_seq_params(packed_seq_params: PackedSeqParams) -> PackedSeqParams:
    """Clone tensor metadata while preserving the packed-attention contract."""

    def clone_tensor(name: str) -> Optional[torch.Tensor]:
        tensor = getattr(packed_seq_params, name)
        return tensor.clone() if tensor is not None else None

    static_params = PackedSeqParams(
        qkv_format=packed_seq_params.qkv_format,
        cu_seqlens_q=clone_tensor("cu_seqlens_q"),
        cu_seqlens_kv=clone_tensor("cu_seqlens_kv"),
        cu_seqlens_q_padded=clone_tensor("cu_seqlens_q_padded"),
        cu_seqlens_kv_padded=clone_tensor("cu_seqlens_kv_padded"),
        max_seqlen_q=packed_seq_params.max_seqlen_q,
        max_seqlen_kv=packed_seq_params.max_seqlen_kv,
        local_cp_size=packed_seq_params.local_cp_size,
        cp_group=packed_seq_params.cp_group,
        total_tokens=packed_seq_params.total_tokens,
    )
    if packed_seq_params.seq_idx is not None:
        static_params.seq_idx = packed_seq_params.seq_idx.clone()
    return static_params


def _packed_tensor_signature(tensor: Optional[torch.Tensor]) -> Optional[tuple[int, ...]]:
    if tensor is None:
        return None
    return tuple(tensor.detach().cpu().tolist())


def _maybe_pad_vision_sequence_for_cuda_graph(
    hidden_states: torch.Tensor,
    rotary_pos_emb: torch.Tensor,
    seq_len: int,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pad vision token tensors to ``max_seq_len`` for fixed-shape CUDA graphs.

    Args:
        hidden_states: ``[seq_len, hidden_size]``.
        rotary_pos_emb: ``[seq_len, 1, 1, dim]`` (same layout as after ``reshape``/``repeat`` in :meth:`Qwen3VLVisionModel.forward`).
        seq_len: Current sequence length (must match tensor leading size).
        max_seq_len: Target length for CUDA graph capture.

    Returns:
        Tuple of (padded hidden_states, padded rotary_pos_emb, new seq_len).

    Raises:
        ValueError: If ``seq_len`` exceeds ``max_seq_len``.
    """
    if seq_len > max_seq_len:
        raise ValueError(
            f"Vision input sequence length ({seq_len}) exceeds max_vision_cuda_graph_seq_length ({max_seq_len}). "
            f"Increase max_vision_cuda_graph_seq_length in config or disable vision CUDA graphs."
        )
    if seq_len < max_seq_len:
        pad_len = max_seq_len - seq_len
        hidden_states = F.pad(hidden_states, (0, 0, 0, pad_len), value=0.0)
        rotary_pos_emb = F.pad(rotary_pos_emb, (0, 0, 0, 0, 0, 0, 0, pad_len), value=0.0)
        seq_len = max_seq_len
    return hidden_states, rotary_pos_emb, seq_len


def _vision_forward_packed_attention_setup(
    use_cuda_graph_padding: bool,
    hidden_states: torch.Tensor,
    original_seq_len: int,
    seq_len: int,
    grid_thw: torch.Tensor,
    build_packed_seq_params: Callable[[torch.Tensor], PackedSeqParams],
) -> tuple[Optional[PackedSeqParams], Optional[torch.Tensor]]:
    """Return ``(packed_seq_params, attention_mask)`` for vision encoder forward.

    When using CUDA graphs, packed sequence metadata (non-tensors) cannot be passed; use full
    attention on a fixed-length padded sequence and optionally an additive mask to ignore padding.

    Args:
        use_cuda_graph_padding: Whether vision CUDA graph padding path is active.
        hidden_states: Vision hidden states after adding the batch dimension, shape ``[S, 1, H]``.
        original_seq_len: Sequence length before padding.
        seq_len: Sequence length after optional padding (equals ``hidden_states`` leading size).
        grid_thw: Grid sizes per image/frame (used only when not using CUDA graph padding).
        build_packed_seq_params: Callback to build :class:`PackedSeqParams` from ``grid_thw``.

    Returns:
        ``packed_seq_params`` (``None`` when using CUDA graph padding) and ``attention_mask``
        (additive mask for padded CUDA graph runs, else ``None``).
    """
    if use_cuda_graph_padding:
        packed_seq_params = None
        if original_seq_len < seq_len:
            attention_mask = torch.ones(
                (1, 1, seq_len, seq_len), dtype=hidden_states.dtype, device=hidden_states.device
            )
            attention_mask[:, :, :, original_seq_len:] = 0
            attention_mask[:, :, original_seq_len:, :] = 0
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
        else:
            attention_mask = None
        return packed_seq_params, attention_mask

    packed_seq_params = build_packed_seq_params(grid_thw)
    return packed_seq_params, None


class Qwen3VLVisionModel(VisionModule):
    """Qwen3 ViT vision model.

    Args:
        transformer_config (TransformerConfig): Transformer config.
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers.
        patch_merger_spec (ModuleSpec): Specifies module to use for transformer layers.
    """

    def __init__(
        self,
        transformer_config: Qwen3VLTransformerConfig,
        transformer_layer_spec: ModuleSpec,
        patch_merger_spec: ModuleSpec,
        pre_process: bool = True,
        post_process: bool = True,
        pg_collection: Optional[ProcessGroupCollection] = None,
        projector_config: Optional[Qwen3VLTransformerConfig] = None,
        projector_tp_group: Optional[torch.distributed.ProcessGroup] = None,
        vision_dp_over_tp_cp: bool = False,
    ) -> None:
        assert post_process and pre_process, "not support pp for deepstack_merger_list"
        super().__init__(config=transformer_config)
        self.spatial_merge_size = transformer_config.spatial_merge_size
        self.patch_size = transformer_config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection
        self.tp_group = self.pg_collection.tp
        self.projector_config = projector_config or transformer_config
        self.projector_tp_group = projector_tp_group or self.tp_group
        self.vision_dp_over_tp_cp = vision_dp_over_tp_cp
        self._vision_encoder_weights_synced = not vision_dp_over_tp_cp

        assert transformer_config.context_parallel_size == 1, (
            f"context_parallel_size should be 1 in vision model but got {transformer_config.context_parallel_size}"
        )

        self.patch_embed = Qwen3VLVisionPatchEmbed(transformer_config)
        self.pos_embed = nn.Embedding(transformer_config.num_position_embeddings, transformer_config.hidden_size)
        self.num_grid_per_side = int(transformer_config.num_position_embeddings**0.5)

        head_dim = transformer_config.hidden_size // transformer_config.num_attention_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)

        self.model_type = ModelType.encoder_or_decoder
        self.pre_process = pre_process
        self.post_process = post_process

        # Transformer layers.
        self.decoder = Qwen3VLVisionTransformerBlock(
            config=transformer_config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            post_layer_norm=False,
            patch_merger_spec=patch_merger_spec,
            pg_collection=self.pg_collection,
            patch_merger_config=self.projector_config,
            patch_merger_tp_group=self.projector_tp_group,
            gather_frozen_encoder_features=vision_dp_over_tp_cp,
        )

        self.merger = None
        if self.post_process:
            self.merger = Qwen3VLVisionPatchMerger(
                self.projector_config,
                patch_merger_spec,
                use_postshuffle_norm=False,
                tp_group=self.projector_tp_group,
            )

        self.input_tensor = None
        self._vision_encoder_cuda_graph_entries: dict[tuple, _FrozenVisionCudaGraphEntry] = {}
        self._vision_encoder_cuda_graph_fallback_signatures: set[tuple] = set()
        self._vision_encoder_cuda_graph_per_image_validated = False

    def _synchronize_replicated_encoder_weights(self) -> None:
        """Make TP=1 frozen encoder replicas identical within each LLM TP group."""
        if self._vision_encoder_weights_synced:
            return
        source_rank = torch.distributed.get_global_rank(self.projector_tp_group, 0)
        encoder_modules = [self.patch_embed, self.pos_embed, self.decoder.layers]
        if self.decoder.final_layernorm is not None:
            encoder_modules.append(self.decoder.final_layernorm)
        with torch.no_grad():
            for module in encoder_modules:
                for parameter in module.parameters():
                    if parameter.is_meta:
                        raise RuntimeError("ViT encoder parameters are still on meta device at first forward")
                    torch.distributed.broadcast(parameter, src=source_rank, group=self.projector_tp_group)
        self._vision_encoder_weights_synced = True

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """Sets input tensor to the model.

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        if self.pre_process:  # always True
            self.input_tensor = input_tensor
        else:
            raise NotImplementedError()

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size

        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim // 2)
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)  # block row indices
            block_cols = torch.arange(merged_w, device=device)  # block col indices
            intra_row = torch.arange(merge_size, device=device)  # intra-block row offsets
            intra_col = torch.arange(merge_size, device=device)  # intra-block col offsets

            # Compute full-resolution positions
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]  # lookup rotary embeddings
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw):
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=self.pos_embed.weight.device)
        weight_tensor = torch.tensor(
            weight_list,
            dtype=self.pos_embed.weight.dtype,
            device=self.pos_embed.weight.device,
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def _get_max_vision_seq_length(self) -> int:
        """Get the maximum sequence length for vision encoder CUDA graphs."""
        if hasattr(self.config, "max_vision_cuda_graph_seq_length") and self.config.max_vision_cuda_graph_seq_length:
            return self.config.max_vision_cuda_graph_seq_length
        # Default: calculate from num_position_embeddings
        return self.config.num_position_embeddings // (self.config.spatial_merge_size**2)

    def _uses_vision_cuda_graph(self) -> bool:
        """Check if vision encoder CUDA graphs are enabled."""
        return (
            hasattr(self.config, "cuda_graph_impl")
            and self.config.cuda_graph_impl == "transformer_engine"
            and self.training
        )

    def _uses_frozen_encoder_cuda_graph(self) -> bool:
        """Whether the packed-sequence-preserving frozen encoder graph is enabled."""
        if not (
            self.training
            and torch.cuda.is_available()
            and getattr(self.config, "vision_encoder_cuda_graph", False)
        ):
            return False
        if self._uses_vision_cuda_graph():
            raise RuntimeError(
                "vision_encoder_cuda_graph and vision_cuda_graph_impl cannot be enabled together"
            )
        if any(parameter.requires_grad for parameter in self.decoder.layers.parameters()):
            raise RuntimeError("vision_encoder_cuda_graph requires frozen ViT encoder layers")
        return True

    @staticmethod
    def _vision_encoder_cuda_graph_signature(
        hidden_states: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        packed_seq_params: PackedSeqParams,
    ) -> tuple:
        """Use exact packed offsets so a cached graph never changes image isolation."""
        return (
            tuple(hidden_states.shape),
            hidden_states.dtype,
            tuple(rotary_pos_emb.shape),
            rotary_pos_emb.dtype,
            packed_seq_params.qkv_format,
            packed_seq_params.max_seqlen_q,
            packed_seq_params.max_seqlen_kv,
            _packed_tensor_signature(packed_seq_params.cu_seqlens_q),
            _packed_tensor_signature(packed_seq_params.cu_seqlens_kv),
            _packed_tensor_signature(packed_seq_params.cu_seqlens_q_padded),
            _packed_tensor_signature(packed_seq_params.cu_seqlens_kv_padded),
        )

    def _capture_frozen_encoder_cuda_graph(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        packed_seq_params: PackedSeqParams,
    ) -> _FrozenVisionCudaGraphEntry:
        """Warm up and capture the frozen encoder for one exact packed shape."""
        static_hidden_states = hidden_states.clone()
        static_rotary_pos_emb = rotary_pos_emb.clone()
        static_packed_seq_params = _clone_packed_seq_params(packed_seq_params)

        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        reference_outputs = None
        with torch.cuda.stream(capture_stream), torch.no_grad():
            for _ in range(3):
                warmup_output, warmup_deepstack = self.decoder.forward_frozen_encoder(
                    static_hidden_states,
                    static_rotary_pos_emb,
                    static_packed_seq_params,
                )
            if getattr(self.config, "vision_encoder_cuda_graph_validate", False):
                reference_outputs = [warmup_output.clone()]
                reference_outputs.extend(output.clone() for output in warmup_deepstack)

        torch.cuda.current_stream().wait_stream(capture_stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(graph, stream=capture_stream):
            output, deepstack_outputs = self.decoder.forward_frozen_encoder(
                static_hidden_states,
                static_rotary_pos_emb,
                static_packed_seq_params,
            )
        graph.replay()

        entry = _FrozenVisionCudaGraphEntry(
            graph,
            static_hidden_states,
            static_rotary_pos_emb,
            static_packed_seq_params,
            output,
            deepstack_outputs,
        )

        if reference_outputs is not None:
            captured_outputs = [output, *deepstack_outputs]
            max_abs_error = max(
                float((captured - reference).abs().max().item())
                for captured, reference in zip(captured_outputs, reference_outputs)
            )
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            print(
                f"[Vision CUDA Graph] rank={rank} packed correctness max_abs_error="
                f"{max_abs_error:.8e}",
                flush=True,
            )
            if max_abs_error > 0.125:
                raise RuntimeError(
                    "Vision CUDA Graph failed eager parity: "
                    f"max_abs_error={max_abs_error:.8e}"
                )
        return entry

    def _run_frozen_encoder_cuda_graph(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        packed_seq_params: PackedSeqParams,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Replay a matching graph, capture a new signature, or safely run eager."""
        signature = self._vision_encoder_cuda_graph_signature(
            hidden_states, rotary_pos_emb, packed_seq_params
        )
        entry = self._vision_encoder_cuda_graph_entries.get(signature)
        if entry is None:
            max_entries = int(getattr(self.config, "vision_encoder_cuda_graph_max_entries", 2))
            if len(self._vision_encoder_cuda_graph_entries) >= max_entries:
                if signature not in self._vision_encoder_cuda_graph_fallback_signatures:
                    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                    print(
                        f"[Vision CUDA Graph] rank={rank} exact-shape cache full "
                        f"({max_entries}); using eager fallback",
                        flush=True,
                    )
                    self._vision_encoder_cuda_graph_fallback_signatures.add(signature)
                with torch.no_grad():
                    return self.decoder.forward_frozen_encoder(
                        hidden_states, rotary_pos_emb, packed_seq_params
                    )
            entry = self._capture_frozen_encoder_cuda_graph(
                hidden_states, rotary_pos_emb, packed_seq_params
            )
            self._vision_encoder_cuda_graph_entries[signature] = entry
            return entry.output, entry.deepstack_outputs

        entry.copy_inputs(hidden_states, rotary_pos_emb, packed_seq_params)
        entry.graph.replay()
        return entry.output, entry.deepstack_outputs

    def _run_frozen_encoder_cuda_graph_per_image(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        grid_thw: torch.Tensor,
        packed_seq_params: PackedSeqParams,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Replay the frozen encoder independently for every image/video item."""
        validate_stitching = (
            getattr(self.config, "vision_encoder_cuda_graph_validate", False)
            and not self._vision_encoder_cuda_graph_per_image_validated
        )
        reference_outputs = None
        if validate_stitching:
            with torch.no_grad():
                reference_output, reference_deepstack = self.decoder.forward_frozen_encoder(
                    hidden_states,
                    rotary_pos_emb,
                    packed_seq_params,
                )
            reference_outputs = [reference_output.clone()]
            reference_outputs.extend(output.clone() for output in reference_deepstack)

        token_counts = torch.prod(grid_thw, dim=1).detach().cpu().tolist()
        output_chunks = []
        deepstack_chunks: Optional[list[list[torch.Tensor]]] = None
        token_offset = 0
        for item_idx, token_count_value in enumerate(token_counts):
            token_count = int(token_count_value)
            token_end = token_offset + token_count
            item_packed_seq_params = self.build_packed_seq_params(grid_thw[item_idx : item_idx + 1])
            item_output, item_deepstack = self._run_frozen_encoder_cuda_graph(
                hidden_states[token_offset:token_end],
                rotary_pos_emb[token_offset:token_end],
                item_packed_seq_params,
            )
            output_chunks.append(item_output.clone())
            if deepstack_chunks is None:
                deepstack_chunks = [[] for _ in item_deepstack]
            for deepstack_idx, feature in enumerate(item_deepstack):
                deepstack_chunks[deepstack_idx].append(feature.clone())
            token_offset = token_end

        assert token_offset == hidden_states.shape[0]
        output = torch.cat(output_chunks, dim=0)
        deepstack_outputs = [
            torch.cat(feature_chunks, dim=0) for feature_chunks in (deepstack_chunks or [])
        ]

        if reference_outputs is not None:
            stitched_outputs = [output, *deepstack_outputs]
            differences = [
                (stitched - reference).abs()
                for stitched, reference in zip(stitched_outputs, reference_outputs)
            ]
            max_abs_error = max(float(difference.max().item()) for difference in differences)
            mean_abs_error = sum(float(difference.mean().item()) for difference in differences) / len(
                differences
            )
            reference_abs_max = max(float(reference.abs().max().item()) for reference in reference_outputs)
            relative_max_error = max_abs_error / max(reference_abs_max, 1.0e-12)
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            print(
                f"[Vision CUDA Graph per-image] rank={rank} packed-vs-stitched "
                f"max_abs_error={max_abs_error:.8e} mean_abs_error={mean_abs_error:.8e} "
                f"relative_max_error={relative_max_error:.8e}",
                flush=True,
            )
            if max_abs_error > 0.5 and relative_max_error > 0.05:
                raise RuntimeError(
                    "Per-image Vision CUDA Graph failed packed eager parity: "
                    f"max_abs_error={max_abs_error:.8e}, relative_max_error={relative_max_error:.8e}"
                )
            self._vision_encoder_cuda_graph_per_image_validated = True

        return output, deepstack_outputs

    def forward(
        self,
        hidden_states: Optional[torch.Tensor],
        grid_thw: torch.Tensor,
        inference_params: Optional[InferenceParams] = None,
        extra_block_kwargs: dict = None,
        vision_tp_gather_seqlens: Optional[list[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward function of the Qwen3 Vision Model. This function passes the input tensors
        through the embedding layer and then the transformer.

        Args:
            x (torch.Tensor): input image/video data of shape [n_tokens, n_dims]
            grid_thw (torch.Tensor): the size tensor indicates grid size of each image/frame
            packed_seq_params (PackedSeqParams): parameters to build attention mask in the backend

        Returns:
            x (torch.Tensor): output after final transformer block of shape [b, s, h].
        """
        assert grid_thw is not None
        assert self.input_tensor is None
        assert inference_params is None

        self._synchronize_replicated_encoder_weights()

        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        seq_len, _ = hidden_states.size()

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, 1, 1, -1).repeat(1, 1, 1, 2)

        # Check if we need to pad for CUDA graphs
        use_cuda_graph_padding = self._uses_vision_cuda_graph()
        original_seq_len = seq_len
        if use_cuda_graph_padding:
            max_seq_len = self._get_max_vision_seq_length()
            hidden_states, rotary_pos_emb, seq_len = _maybe_pad_vision_sequence_for_cuda_graph(
                hidden_states, rotary_pos_emb, seq_len, max_seq_len
            )
        hidden_states = hidden_states[:, None]
        packed_seq_params, attention_mask = _vision_forward_packed_attention_setup(
            use_cuda_graph_padding=use_cuda_graph_padding,
            hidden_states=hidden_states,
            original_seq_len=original_seq_len,
            seq_len=seq_len,
            grid_thw=grid_thw,
            build_packed_seq_params=self.build_packed_seq_params,
        )
        if self._uses_frozen_encoder_cuda_graph():
            if getattr(self.config, "vision_encoder_cuda_graph_per_image", False):
                hidden_states, deepstack_hidden_states = (
                    self._run_frozen_encoder_cuda_graph_per_image(
                        hidden_states,
                        rotary_pos_emb,
                        grid_thw,
                        packed_seq_params,
                    )
                )
            else:
                hidden_states, deepstack_hidden_states = self._run_frozen_encoder_cuda_graph(
                    hidden_states,
                    rotary_pos_emb,
                    packed_seq_params,
                )
            deepstack_feature_lists = [
                self.decoder._project_deepstack_feature(
                    feature,
                    deepstack_idx,
                    vision_tp_gather_seqlens,
                )
                for deepstack_idx, feature in enumerate(deepstack_hidden_states)
            ]
        else:
            hidden_states, deepstack_feature_lists = self.decoder(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                packed_seq_params=packed_seq_params,
                vision_tp_gather_seqlens=vision_tp_gather_seqlens,
                **(extra_block_kwargs or {}),
            )
        # Remove padding if we added it
        if use_cuda_graph_padding and original_seq_len < seq_len:
            hidden_states = hidden_states[:original_seq_len]
            # Unpad deepstack features - they go through a merger that reduces by spatial_merge_size^2
            # So their length is seq_len // (spatial_merge_size^2)
            original_merged_seq_len = original_seq_len // (self.spatial_merge_size**2)
            deepstack_feature_lists = [feat[:original_merged_seq_len] for feat in deepstack_feature_lists]
        if vision_tp_gather_seqlens is not None:
            hidden_states = AllGatherVisionEmbeddings.apply(
                hidden_states.detach(),
                vision_tp_gather_seqlens,
                self.projector_tp_group,
            )
        hidden_states = self.merger(hidden_states)

        # Encodes images into continuous embeddings that can be forwarded to the language model.
        if vision_tp_gather_seqlens is None:
            split_sizes = (grid_thw.prod(-1) // self.spatial_merge_size**2).tolist()
            hidden_states = torch.split(hidden_states, split_sizes)
            hidden_states = torch.cat(hidden_states, dim=0)
        return hidden_states, deepstack_feature_lists

    def build_packed_seq_params(
        self,
        grid_thw: Optional[torch.Tensor],
    ) -> PackedSeqParams:
        # NOTE: each frame is a sequence (rather than each grid)
        seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
        cu_seqlens = seqlens.cumsum(dim=0)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0).int()

        # Packed sequence metadata is shared by all vision layers. Materialize
        # this scalar once here instead of letting every attention wrapper
        # repeat a CUDA scalar comparison/conversion.
        max_seqlen_q = int(seqlens.max().item())
        return PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            qkv_format="thd",
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_q,
        )
