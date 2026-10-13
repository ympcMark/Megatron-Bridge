# MAGI-2 pretraining in Megatron Bridge

This integration makes Megatron Bridge train the native
`megatron.core.models.magi2.Magi2Model`. Bridge contains the dataset, flow-loss
step, recipe, and a thin provider; model assembly and all MAGI-2 layers live in
Megatron Core. The default provider uses 40 layers, hidden size 3072, four mHC
residual streams, four multimodal dense layers, and 36 multi-head MoE layers
with 12 heads and 256 experts per head. The routed expert matrices alone contain
108,716,359,680 parameters, and the complete logical model contains
113,934,732,336 parameters.

The provider passes one native `Magi2Config`, MCore layer specs, and the active
`ProcessGroupCollection` to `Magi2Model`; it does not assemble Transformer
layers. The routed path uses MCore expert parallelism: head-local expert IDs are
flattened into 3072 experts, routed with the MAGI-2 sigmoid/top-6 rule,
dispatched by MCore all-to-all, and evaluated by Transformer Engine grouped
GEMM. The original-size recipes fix TP and PP to one, use EP=64, and support
either CP1 or CP2. The CP2 recipe also enables MCore FSDP V1.

## Integration run

`pretrain_magi2.py --smoke` runs a reduced model through the normal Bridge
pretraining loop. Without `--smoke`, the launcher requires exactly 64 ranks and
rejects architecture reductions.

During development against an MCore checkout that contains MAGI-2, set
`MCORE_REPO` to that checkout. If it is unset, the launcher uses Bridge's pinned
`3rdparty/Megatron-LM` submodule.

The checked-in Slurm launch performs 50 original-size optimizer steps, saves a
torch-dist model/optimizer/RNG checkpoint at step 50, then reloads it and
executes steps 51 and 52 without duplicating the multi-terabyte checkpoint:

```bash
sbatch examples/models/magi2/pretrain_magi2_114b_64gpu.slurm
```

Each run writes a JSON manifest plus one JSONL row per optimizer step. Rows record
finite-loss and finite-gradient checks, skipped iterations, cross-rank peak
memory, and throughput.

To run the full model with EP64, CP2, and fully sharded parameters, gradients,
and optimizer state, add:

```bash
--fsdp --context-parallel-size 2 --cp-comm-type all_gather
```

Bridge pads each logical 11-token sample to 12 physical THD positions, applies
MCore's per-document zigzag partition, and excludes the padding from the flow
loss. MCore gathers K/V with an autograd-aware collective and computes the
official learnable-sink attention for the local queries. `all_gather` is the
only supported CP communication mode because the current Transformer Engine
CP kernels do not support MAGI-2's learnable softmax offset.

## Checkpoints

Use `--official-checkpoint /path/to/MAGI-2-preview` to initialize the native
MCore model from the public safetensors checkpoint. Loading runs on every expert
rank after model construction and before mixed-precision/DDP wrapping; MCore
streams tensors and keeps only that rank's routed experts.

Use `--load /path/to/torch-dist-checkpoint` to resume a Bridge training run.
This restores the native model, distributed optimizer, scheduler, consumed
sample count, and RNG state. The two options are mutually exclusive: an
official checkpoint starts a new optimizer state, while a torch-dist checkpoint
continues an existing run.

## Data boundary

Without an explicit data path, `Magi2LatentDatasetConfig` supplies deterministic
in-memory latents and text conditioning as a minimal smoke fallback. To exercise
the real DataLoader boundary, first materialize variable-length latent shards:

```bash
uv run python scripts/training/generate_magi2_latent_dataset.py \
  --output-dir /path/to/magi2-latents \
  --train-samples 64 --valid-samples 16
```

Then pass the directory to training:

```bash
uv run python -m torch.distributed.run --nproc_per_node=4 \
  examples/models/magi2/pretrain_magi2.py \
  --smoke --latent-dataset /path/to/magi2-latents \
  --checkpoint-dir /path/to/checkpoints --loss-output /path/to/loss.jsonl
```

The `magi2-latent-v1` manifest describes channel widths, token ranges, splits,
and shards. The collator concatenates variable-length examples into one physical
THD stream and emits logical `cu_seqlens`, so attention cannot cross sample
boundaries. With CP enabled, every logical sample is padded independently to a
multiple of `2 * CP`, partitioned with MCore's per-document zigzag layout, and
excluded from the loss through an all-zero padding mask. Sigma and noise remain
keyed by sample ID and optimizer step, independent of batch position or CP
topology.

This generated dataset validates reading, sharding, collation, packing,
communication, and checkpoint integration, but its tensor values are synthetic.
For real media, create a JSONL manifest with one object per sample:

```json
{"sample_id": 0, "split": "train", "video": "clip.mp4", "audio": "clip.wav", "text": "A prompt"}
```

Relative media paths are resolved from the JSONL directory. Encode it with the
unmodified public video VAE, Stable Audio VAE, and Qwen3.5 text encoder:

```bash
uv run --extra diffusion python scripts/training/prepare_magi2_media_latents.py \
  --input-manifest /path/to/media.jsonl \
  --output-dir /path/to/magi2-official-latents \
  --official-repo /path/to/MAGI-2-preview \
  --checkpoint-root /path/to/MAGI-2-preview/ckpt
```

Only encoder weights are required: `vae/Wan2.2_VAE.pth`,
`stable-audio-open-1.0/{model_config.json,model.safetensors}`, and the complete
`text_encoder/` directory. They total roughly 64 GB in the public release; the
228 GB preview transformer is not used by preprocessing. The converter samples
8k+1 video frames at the official 25 fps, center-crops to a 32-aligned
resolution, aligns stereo audio to the same duration, calls the official
encoders directly, reproduces the preview token order and nine-field RoPE
coordinates, and streams shards in the same `magi2-latent-v1` contract.
Training therefore uses the same `--latent-dataset` argument for synthetic
bring-up and real encoded media.

For topology-independent integration comparisons, synthetic sigma and noise are
derived from the sample ID and optimizer step. Repartitioning the same global
batch between CP1 and CP2 therefore does not change the flow targets.

The flow step follows the released inference convention: it forms
`x_sigma = (1 - sigma) * x0 + sigma * noise`, predicts `noise - x0`, and passes
the official 64-channel sinusoidal timestep embedding for every packed token to
MCore. Video and audio tokens receive their sampled sigma; clean text tokens
receive sigma zero. The released configuration does not add a separate TIME
token.
