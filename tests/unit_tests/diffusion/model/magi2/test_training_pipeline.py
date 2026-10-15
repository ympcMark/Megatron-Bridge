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

import json
import math
from types import SimpleNamespace
from unittest.mock import Mock

import examples.models.magi2.pretrain_magi2 as pretrain_module
import pytest
import torch
from megatron.core.models.magi2 import (
    Magi2Config,
    Magi2ContextParallelDotProductAttention,
    Magi2Modality,
    Magi2Model,
    Magi2TransformerLayerSpecs,
)
from megatron.core.process_groups_config import ProcessGroupCollection

import megatron.bridge.diffusion.models.magi2.magi2_step as step_module
import megatron.bridge.diffusion.models.magi2.media_preprocessing as media_preprocessing_module
import megatron.bridge.diffusion.models.magi2.provider as provider_module
from megatron.bridge.diffusion.models.magi2.data import (
    Magi2DiskLatentDataset,
    Magi2LatentDataset,
    generate_magi2_latent_sample,
    magi2_latent_collate,
    write_magi2_latent_dataset,
    write_magi2_precomputed_latent_dataset,
)
from megatron.bridge.diffusion.models.magi2.flops import MAGI2_FLOPS_RUNTIME_STAT_NAMES
from megatron.bridge.diffusion.models.magi2.magi2_step import (
    magi2_deterministic_flow_random,
    magi2_deterministic_packed_flow_random,
    magi2_flow_loss,
    magi2_prepare_flow_inputs,
    magi2_timestep_embedding,
    partition_magi2_batch_for_context_parallel,
)
from megatron.bridge.diffusion.models.magi2.media_preprocessing import (
    magi2_latents_to_sample,
    read_magi2_media_manifest,
)
from megatron.bridge.diffusion.models.magi2.provider import Magi2ModelProvider
from megatron.bridge.recipes.magi2.h100.magi2 import (
    magi2_114b_pretrain_64gpu_h100_bf16_fsdp_config,
    magi2_114b_pretrain_64gpu_h100_bf16_fsdp_cp2_config,
)
from megatron.bridge.training.utils.flop_utils import (
    accumulate_flops_metadata,
    num_floating_point_operations,
    reset_flops_metadata,
    resolve_global_flops_runtime_stats,
)
from megatron.bridge.training.utils.theoretical_memory_utils import estimate_training_memory


pytestmark = [pytest.mark.unit]


def test_original_provider_is_native_mcore_config():
    provider = Magi2ModelProvider()

    assert isinstance(provider, Magi2Config)
    assert Magi2Model.__module__.startswith("megatron.core.models.magi2")
    assert provider.require_original_size
    assert provider.num_layers == 40
    assert provider.hidden_size == 3072
    assert provider.magi2_flattened_num_experts == 3072
    assert provider.magi2_moe_top_k == 6
    assert provider.magi2_parameter_count == 113_934_732_336


def test_original_fsdp_recipe_uses_fully_sharded_dtensor_contract():
    cfg = magi2_114b_pretrain_64gpu_h100_bf16_fsdp_config()

    assert cfg.dist.use_megatron_fsdp
    assert cfg.ddp.use_megatron_fsdp
    assert cfg.ddp.megatron_fsdp_version == 1
    assert cfg.ddp.data_parallel_sharding_strategy == "optim_grads_params"
    assert cfg.ddp.expert_data_parallel_sharding_strategy == "optim_grads_params"
    assert cfg.ddp.megatron_fsdp_main_params_dtype == torch.float32
    assert cfg.ddp.megatron_fsdp_main_grads_dtype == torch.float32
    assert cfg.ddp.megatron_fsdp_grad_comm_dtype == torch.float32
    assert not cfg.ddp.average_in_collective
    assert cfg.checkpoint.ckpt_format == "fsdp_dtensor"
    assert cfg.model.recompute_granularity == "full"
    assert cfg.model.recompute_method == "uniform"
    assert cfg.model.recompute_num_layers == 1
    assert cfg.model.recompute_modules is None

    estimate = estimate_training_memory(cfg, include_activation=False)
    dense_component, routed_component = estimate.model_state_components
    assert dense_component.parameter_count == 5_218_372_656
    assert routed_component.parameter_count == 108_716_359_680
    assert routed_component.parameter_count_per_gpu == 1_698_693_120
    assert estimate.total_parameters == 113_934_732_336


def test_original_recipe_reports_runtime_aware_algorithmic_and_recompute_flops():
    cfg = magi2_114b_pretrain_64gpu_h100_bf16_fsdp_config()
    modality_counts = {
        "magi2_video_tokens": 64 * 2000,
        "magi2_audio_tokens": 64 * 14,
        "magi2_text_tokens": 64 * 13,
        "magi2_padding_tokens": 0,
    }

    estimate = cfg.model.get_magi2_flop_estimate(
        seqlen_sum=64 * 2027,
        seqlen_squared_sum=64 * 2027**2,
        model_specific_stats=modality_counts,
    )

    assert estimate.adapters_forward == 409_498_288_128
    assert estimate.attention_forward == 521_780_047_380_480
    assert estimate.mhc_forward == 7_396_613_160_960
    assert estimate.dense_mlp_forward == 78_353_088_380_928
    assert estimate.moe_router_forward == 7_345_602_035_712
    assert estimate.moe_routed_forward == 661_104_183_214_080
    assert estimate.moe_shared_forward == 220_368_061_071_360
    assert estimate.moe_split_merge_forward == 176_294_448_857_088
    assert estimate.algorithmic_training == 5_019_154_627_166_208
    assert estimate.executed_training == 6_691_796_671_266_816
    assert (
        num_floating_point_operations(
            cfg,
            batch_size=64,
            seqlen_sum=64 * 2027,
            seqlen_squared_sum=64 * 2027**2,
            model_specific_stats=modality_counts,
        )
        == estimate.algorithmic_training
    )


def test_model_specific_flops_metadata_accumulates_and_scales_over_data_parallelism():
    state = SimpleNamespace()
    reset_flops_metadata(state, model_specific_stat_names=MAGI2_FLOPS_RUNTIME_STAT_NAMES)
    accumulate_flops_metadata(
        state,
        torch.zeros(1, 5),
        total_sequence_tokens=8,
        cu_seqlens=torch.tensor([0, 4, 8], dtype=torch.int32),
        cu_seqlens_unpadded=torch.tensor([0, 2, 5], dtype=torch.int32),
        model_specific_stats={
            "magi2_video_tokens": torch.tensor(2),
            "magi2_audio_tokens": torch.tensor(2),
            "magi2_text_tokens": torch.tensor(1),
            "magi2_padding_tokens": torch.tensor(3),
        },
    )

    stats = resolve_global_flops_runtime_stats(
        state,
        data_parallel_size=2,
        model_specific_stat_names=MAGI2_FLOPS_RUNTIME_STAT_NAMES,
    )

    assert stats.seqlen_sum == 16
    assert stats.seqlen_squared_sum == 2 * (2**2 + 3**2)
    assert stats.model_specific_stats == {
        "magi2_video_tokens": 4,
        "magi2_audio_tokens": 4,
        "magi2_text_tokens": 2,
        "magi2_padding_tokens": 6,
    }


def test_training_callbacks_report_synchronized_step_time_and_flops(tmp_path, monkeypatch):
    output = tmp_path / "metrics.jsonl"
    model_config = Magi2ModelProvider()
    runtime_stats = SimpleNamespace(
        seqlen_sum=5,
        seqlen_squared_sum=25,
        model_specific_stats={
            "magi2_video_tokens": 2,
            "magi2_audio_tokens": 2,
            "magi2_text_tokens": 1,
            "magi2_padding_tokens": 0,
        },
    )
    algorithmic_flops = model_config.get_magi2_flop_estimate(
        seqlen_sum=runtime_stats.seqlen_sum,
        seqlen_squared_sum=runtime_stats.seqlen_squared_sum,
        model_specific_stats=runtime_stats.model_specific_stats,
    ).algorithmic_training
    context = SimpleNamespace(
        state=SimpleNamespace(
            cfg=SimpleNamespace(model=model_config),
            train_state=SimpleNamespace(step=0),
            _flops_in_last_batch=algorithmic_flops,
            _flops_runtime_stats=runtime_stats,
        ),
        model=[SimpleNamespace(modules=lambda: ())],
        user_state={},
        optimizer=SimpleNamespace(param_groups=[]),
        loss_dict={"flow_mse": torch.tensor([1.0, 2.0])},
        grad_norm=3.0,
        skipped_iter=False,
    )
    synchronize = Mock()
    reset_peak_memory = Mock()
    clock = iter((10.0, 20.0))
    monkeypatch.setattr(pretrain_module.torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(pretrain_module.torch.cuda, "reset_peak_memory_stats", reset_peak_memory)
    monkeypatch.setattr(pretrain_module.torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(pretrain_module.torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(pretrain_module.torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(pretrain_module.torch.distributed, "all_reduce", lambda tensor, op: None)
    monkeypatch.setattr(pretrain_module.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(pretrain_module.torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(pretrain_module.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(pretrain_module, "get_canonical_lr_for_logging", lambda groups: 1e-4)

    pretrain_module._step_start_callback(context)
    pretrain_module._step_callback(output, 0.999, 1, 5, context)

    row = json.loads(output.read_text())
    assert row["step_time_seconds"] == {"min": 10.0, "mean": 10.0, "max": 10.0}
    assert row["algorithmic_flops"] == algorithmic_flops
    assert row["model_tflops_per_gpu"] == algorithmic_flops / 10.0 / 1e12
    assert row["executed_flops_including_recompute"] == algorithmic_flops
    assert row["global_tokens_per_second"] == 0.5
    assert synchronize.call_count == 2
    reset_peak_memory.assert_called_once_with()


def test_original_fsdp_cp2_recipe_uses_per_element_global_normalization():
    cfg = magi2_114b_pretrain_64gpu_h100_bf16_fsdp_cp2_config()

    assert cfg.model.context_parallel_size == 2
    assert cfg.model.cp_comm_type == "all_gather"
    assert cfg.model.seq_length == 12
    assert cfg.model.expert_model_parallel_size == 64
    assert cfg.model.calculate_per_token_loss
    assert not cfg.ddp.average_in_collective


def test_cp_provider_selects_mcore_context_parallel_attention():
    provider = Magi2ModelProvider()
    provider.context_parallel_size = 2
    provider.cp_comm_type = "all_gather"

    specs = provider_module.default_magi2_layer_specs(provider)

    assert specs.dense.submodules.self_attention.submodules.core_attention is Magi2ContextParallelDotProductAttention


def test_provider_only_selects_specs_and_instantiates_native_model(monkeypatch):
    provider = Magi2ModelProvider()
    pg_collection = Mock(spec=ProcessGroupCollection)
    native_model = Mock(spec=Magi2Model)
    constructor = Mock(return_value=native_model)
    provider._pg_collection = pg_collection
    monkeypatch.setattr(provider_module, "Magi2Model", constructor)

    result = provider.provide()

    assert result is native_model
    constructor.assert_called_once()
    kwargs = constructor.call_args.kwargs
    assert kwargs["config"] is provider
    assert kwargs["pg_collection"] is pg_collection
    assert isinstance(kwargs["layer_specs"], Magi2TransformerLayerSpecs)
    assert kwargs["pre_process"]
    assert kwargs["post_process"]


def test_original_size_guard_reports_changed_native_field():
    provider = Magi2ModelProvider()
    provider.magi2_moe_num_experts_per_head = 4

    with pytest.raises(ValueError, match="magi2_moe_num_experts_per_head"):
        provider.finalize()


def test_reduced_provider_finalizes_with_native_mcore_validation():
    provider = Magi2ModelProvider()
    provider.require_original_size = False
    provider.num_layers = 2
    provider.hidden_size = 64
    provider.num_attention_heads = 4
    provider.num_query_groups = 4
    provider.kv_channels = 16
    provider.ffn_hidden_size = 128
    provider.num_moe_experts = 16
    provider.moe_ffn_hidden_size = 32
    provider.moe_router_topk = 2
    provider.expert_model_parallel_size = 1
    provider.magi2_video_in_channels = 8
    provider.magi2_audio_in_channels = 8
    provider.magi2_text_in_channels = 16
    provider.magi2_intermediate_factor = 2
    provider.magi2_mm_layers = (0,)
    provider.magi2_moe_layers = (1,)
    provider.magi2_moe_num_heads = 4
    provider.magi2_moe_num_experts_per_head = 4
    provider.magi2_moe_top_k = 2
    provider.magi2_moe_expert_intermediate_size = 32
    provider.magi2_shared_expert_intermediate_size = 32
    provider.magi2_modality_expert_intermediate_size = 32
    provider.magi2_mhc_num_streams = 2

    provider.finalize()

    assert provider.magi2_adapter_width == 128
    assert provider.magi2_moe_head_dim == 16


def test_latent_dataset_is_reproducible_and_uses_public_channel_widths():
    dataset = Magi2LatentDataset(
        2,
        seed=123,
        video_tokens=2,
        audio_tokens=2,
        text_tokens=1,
        video_in_channels=48,
        audio_in_channels=64,
        text_in_channels=5120,
    )

    first = dataset[0]
    repeated = dataset[0]

    assert first["clean_inputs"].shape == (5, 5120)
    assert first["coordinates"].shape == (5, 9)
    assert torch.equal(first["clean_inputs"], repeated["clean_inputs"])
    assert torch.equal(first["modality_mapping"], repeated["modality_mapping"])
    assert not torch.any(first["modality_mapping"] == Magi2Modality.TIME)


def test_disk_latent_dataset_roundtrip_and_variable_length_collate(tmp_path):
    dataset_root = tmp_path / "latents"
    write_magi2_latent_dataset(
        dataset_root,
        train_samples=5,
        valid_samples=2,
        shard_size=2,
        seed=17,
        video_token_range=(1, 3),
        audio_token_range=(1, 2),
        text_token_range=(1, 2),
        video_in_channels=2,
        audio_in_channels=3,
        text_in_channels=4,
    )

    dataset = Magi2DiskLatentDataset(dataset_root, "train")
    assert len(dataset) == 5
    torch.testing.assert_close(dataset[0]["clean_inputs"], dataset[0]["clean_inputs"])
    assert dataset[-1]["sample_id"].item() == 4
    assert len(tuple(dataset_root.glob("train-*.pt"))) == 3

    short = generate_magi2_latent_sample(
        10,
        seed=1,
        video_tokens=1,
        audio_tokens=1,
        text_tokens=1,
        video_in_channels=2,
        audio_in_channels=3,
        text_in_channels=4,
    )
    long = generate_magi2_latent_sample(
        11,
        seed=1,
        video_tokens=2,
        audio_tokens=2,
        text_tokens=1,
        video_in_channels=2,
        audio_in_channels=3,
        text_in_channels=4,
    )
    batch = magi2_latent_collate([short, long])

    assert batch["clean_inputs"].shape == (8, 4)
    assert batch["coordinates"].shape == (8, 9)
    assert batch["cu_seqlens"].tolist() == [0, 3, 8]
    assert batch["sample_id"].tolist() == [10, 11]


def test_official_latents_pack_with_preview_token_order_and_coordinates():
    video = torch.arange(8, dtype=torch.float32).reshape(2, 2, 1, 2)
    audio = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    text = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    sample = magi2_latents_to_sample(
        video,
        audio,
        text,
        sample_id=7,
        storage_dtype=torch.float32,
        expected_channels={"video": 2, "audio": 3, "text": 4},
    )

    assert sample["clean_inputs"].shape == (9, 4)
    torch.testing.assert_close(sample["clean_inputs"][0], torch.tensor([0.0, 4.0, 0.0, 0.0]))
    assert sample["modality_mapping"].tolist() == [
        Magi2Modality.VIDEO,
        Magi2Modality.VIDEO,
        Magi2Modality.VIDEO,
        Magi2Modality.VIDEO,
        Magi2Modality.AUDIO,
        Magi2Modality.AUDIO,
        Magi2Modality.AUDIO,
        Magi2Modality.TEXT,
        Magi2Modality.TEXT,
    ]
    torch.testing.assert_close(sample["coordinates"][0], torch.tensor([0.0, 0.0, 0.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0]))
    torch.testing.assert_close(sample["coordinates"][4], torch.tensor([0.0, 0.0, 0.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    torch.testing.assert_close(sample["coordinates"][7], torch.tensor([-2.0, 0.0, 0.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0]))


def test_precomputed_latent_writer_streams_shards_and_records_observed_ranges(tmp_path):
    first = magi2_latents_to_sample(
        torch.zeros(2, 1, 1, 2),
        torch.zeros(3, 3),
        torch.zeros(2, 4),
        sample_id=10,
        storage_dtype=torch.float32,
    )
    second = magi2_latents_to_sample(
        torch.zeros(2, 2, 1, 2),
        torch.zeros(5, 3),
        torch.zeros(1, 4),
        sample_id=11,
        storage_dtype=torch.float32,
    )
    dataset_root = tmp_path / "precomputed"

    manifest_path = write_magi2_precomputed_latent_dataset(
        dataset_root,
        {"train": iter((first, second))},
        channels={"video": 2, "audio": 3, "text": 4},
        shard_size=1,
        metadata={"source": "unit-test"},
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["token_ranges"] == {"video": [2, 4], "audio": [3, 5], "text": [1, 2]}
    assert manifest["max_tokens_per_sample"] == 11
    assert manifest["metadata"] == {"source": "unit-test"}
    assert len(tuple(dataset_root.glob("train-*.pt"))) == 2
    dataset = Magi2DiskLatentDataset(dataset_root, "train")
    assert [dataset[index]["sample_id"].item() for index in range(2)] == [10, 11]


def test_media_manifest_resolves_relative_paths_and_rejects_duplicate_ids(tmp_path):
    (tmp_path / "clip.mp4").touch()
    (tmp_path / "clip.wav").touch()
    manifest_path = tmp_path / "media.jsonl"
    record = {"sample_id": 3, "split": "train", "video": "clip.mp4", "audio": "clip.wav", "text": "test"}
    manifest_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    records = read_magi2_media_manifest(manifest_path)

    assert records[0].sample_id == 3
    assert records[0].video_path == (tmp_path / "clip.mp4").resolve()
    assert records[0].audio_path == (tmp_path / "clip.wav").resolve()

    manifest_path.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate sample_id"):
        read_magi2_media_manifest(manifest_path)


def test_official_encoder_rejects_incompatible_transformers_before_loading_weights(monkeypatch, tmp_path):
    import transformers

    official_repo = tmp_path / "official"
    checkpoint_root = tmp_path / "checkpoint"
    required_files = (
        official_repo / "inference/model/vae2_2.py",
        official_repo / "inference/model/qwen35.py",
        official_repo / "inference/pipeline/audio_decoder.py",
        checkpoint_root / "vae/Wan2.2_VAE.pth",
        checkpoint_root / "stable-audio-open-1.0/model_config.json",
        checkpoint_root / "stable-audio-open-1.0/model.safetensors",
        checkpoint_root / "text_encoder/config.json",
        checkpoint_root / "text_encoder/model.safetensors",
    )
    for path in required_files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(transformers, "__version__", "5.3.0")

    with pytest.raises(RuntimeError, match=r"transformers>=5\.5\.0; found 5\.3\.0"):
        media_preprocessing_module.OfficialMagi2Encoders(official_repo, checkpoint_root)


def test_time_embedding_matches_official_magi2_definition():
    timesteps = torch.tensor([0.0, 0.25, 1.0])
    embedding = magi2_timestep_embedding(timesteps, 64)
    half = 32
    frequencies = torch.exp(-math.log(10_000.0) * torch.arange(half, dtype=torch.float32) / half)
    angles = timesteps[:, None] * 1000.0 * frequencies[None]
    official = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)

    torch.testing.assert_close(embedding, official, rtol=0.0, atol=0.0)


def test_flow_inputs_match_official_sigma_direction_and_time_channels():
    clean = torch.tensor(
        [
            [1.0, 2.0, 30.0],
            [4.0, 5.0, 60.0],
            [7.0, 8.0, 9.0],
        ]
    )
    noise = torch.tensor(
        [
            [11.0, 12.0, 130.0],
            [14.0, 15.0, 160.0],
            [17.0, 18.0, 19.0],
        ]
    )
    mapping = torch.tensor([Magi2Modality.VIDEO, Magi2Modality.AUDIO, Magi2Modality.TEXT])
    sigmas = torch.tensor([0.0, 1.0, 0.75])

    model_inputs, target, loss_mask, time_sequence = magi2_prepare_flow_inputs(
        clean,
        mapping,
        sigmas,
        video_in_channels=2,
        audio_in_channels=3,
        time_channel_dim=64,
        noise=noise,
    )

    torch.testing.assert_close(model_inputs[0], clean[0])
    torch.testing.assert_close(model_inputs[1], noise[1])
    torch.testing.assert_close(model_inputs[2], clean[2])
    torch.testing.assert_close(target[0], torch.tensor([10.0, 10.0, 0.0]))
    torch.testing.assert_close(target[1], torch.tensor([10.0, 10.0, 100.0]))
    torch.testing.assert_close(target[2], torch.zeros(3))
    torch.testing.assert_close(
        loss_mask,
        torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]),
    )
    torch.testing.assert_close(
        time_sequence,
        magi2_timestep_embedding(torch.tensor([0.0, 1.0, 0.0]), 64),
    )


def test_flow_inputs_preserve_bfloat16_latent_storage_dtype():
    clean = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    noise = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)
    mapping = torch.tensor([Magi2Modality.VIDEO, Magi2Modality.TEXT])
    sigmas = torch.tensor([0.25, 0.0], dtype=torch.float32)

    model_inputs, target, loss_mask, time_sequence = magi2_prepare_flow_inputs(
        clean,
        mapping,
        sigmas,
        video_in_channels=2,
        audio_in_channels=2,
        time_channel_dim=4,
        noise=noise,
    )

    assert model_inputs.dtype == torch.bfloat16
    assert target.dtype == torch.bfloat16
    assert loss_mask.dtype == torch.bfloat16
    assert time_sequence.dtype == torch.float32
    torch.testing.assert_close(model_inputs[0], torch.tensor([2.0, 3.0], dtype=torch.bfloat16))
    torch.testing.assert_close(target[0], torch.tensor([4.0, 4.0], dtype=torch.bfloat16))


def test_synthetic_flow_random_is_stable_per_sample_and_step():
    clean = torch.zeros(2, 3, 4)
    sample_ids = torch.tensor([9, 3])

    first = magi2_deterministic_flow_random(clean, sample_ids, step=2, seed=42)
    repeated = magi2_deterministic_flow_random(clean, sample_ids, step=2, seed=42)
    next_step = magi2_deterministic_flow_random(clean, sample_ids, step=3, seed=42)

    torch.testing.assert_close(first[0], repeated[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(first[1], repeated[1], rtol=0.0, atol=0.0)
    assert not torch.equal(first[0], next_step[0])
    assert not torch.equal(first[1], next_step[1])


def test_packed_flow_random_is_stable_for_variable_length_samples():
    clean = torch.zeros(8, 4)
    sample_ids = torch.tensor([10, 11])
    cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32)

    first = magi2_deterministic_packed_flow_random(clean, sample_ids, cu_seqlens, step=2, seed=42)
    repeated = magi2_deterministic_packed_flow_random(clean, sample_ids, cu_seqlens, step=2, seed=42)

    torch.testing.assert_close(first[0], repeated[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(first[1], repeated[1], rtol=0.0, atol=0.0)
    assert first[0][:3].unique().numel() == 1
    assert first[0][3:].unique().numel() == 1
    assert first[0][0] != first[0][3]


def test_timestep_embedding_and_flow_loss_are_finite():
    embedding = magi2_timestep_embedding(torch.tensor([0.25, 0.75]), 9)
    squared_error = torch.tensor([[1.0, 4.0], [9.0, 16.0]])
    mask = torch.tensor([[1.0, 1.0], [0.0, 0.0]])

    loss, valid_tokens, metrics = magi2_flow_loss(mask, squared_error)

    assert embedding.shape == (2, 9)
    assert torch.isfinite(embedding).all()
    torch.testing.assert_close(loss, torch.tensor(5.0))
    assert valid_tokens.item() == 2
    torch.testing.assert_close(metrics["flow_mse"], torch.tensor([5.0, 2.0]))


def test_context_parallel_partition_pads_each_sample_and_masks_padding(monkeypatch):
    cp_group = Mock()
    cp_group.size.return_value = 2
    indices = torch.tensor([0, 3, 4, 7, 8, 11])
    partition = Mock(return_value=indices)
    monkeypatch.setattr(step_module, "get_thd_cp_partition_indices", partition)
    mapping = torch.tensor(
        [
            Magi2Modality.VIDEO,
            Magi2Modality.AUDIO,
            Magi2Modality.TEXT,
            Magi2Modality.VIDEO,
            Magi2Modality.VIDEO,
            Magi2Modality.AUDIO,
            Magi2Modality.AUDIO,
            Magi2Modality.TEXT,
        ]
    )
    values = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32)

    local, logical, physical, split_sizes, global_indices = partition_magi2_batch_for_context_parallel(
        {"values": values, "modality_mapping": mapping},
        cu_seqlens=cu_seqlens,
        cp_group=cp_group,
    )

    assert logical.tolist() == [0, 3, 8]
    assert physical.tolist() == [0, 4, 12]
    assert split_sizes.tolist() == [6, 6]
    assert global_indices.tolist() == [0, 3, 4, 7, 8, 11]
    assert local["values"].shape == (6, 2)
    assert local["modality_mapping"].tolist() == [0, 2, 0, 1, 2, 2]
    partition.assert_called_once()


def test_official_checkpoint_hook_delegates_to_native_mcore_loader(monkeypatch, tmp_path):
    model = Mock(spec=Magi2Model)
    report = Mock(consumed_source_keys=("source",), populated_target_keys=("target",))
    loader = Mock(return_value=report)
    monkeypatch.setattr(pretrain_module, "load_magi2_official_safetensors", loader)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    result = pretrain_module._load_official_checkpoint(tmp_path, [model])

    assert result == [model]
    loader.assert_called_once_with(model, tmp_path, strict=True)
