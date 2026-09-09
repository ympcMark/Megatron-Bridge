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
GEMM. The current correctness recipe fixes TP, PP, and CP to one and uses EP=64.

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

`Magi2LatentDatasetConfig` intentionally supplies deterministic synthetic packed
latents and text conditioning. It validates model, optimizer, communication, and
checkpoint integration but is not a scientific corpus-training dataset. A
production run should replace only this DatasetProvider with a provider that
emits VAE/audio latents, text-encoder states, 3-axis coordinates, and modality
IDs using the same batch keys. The model and pretraining step do not need to be
rewritten.
