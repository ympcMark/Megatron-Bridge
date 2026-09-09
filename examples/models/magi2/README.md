# MAGI-2 pretraining in Megatron Bridge

This integration instantiates the public MAGI-2-preview architecture inside
Megatron Bridge instead of defining an unrelated replacement model. The default
provider uses 40 layers, hidden size 3072, four mHC residual streams, four
multimodal dense layers, and 36 multi-head MoE layers with 12 heads and 256
experts per head. The routed expert matrices alone contain 108,716,359,680
parameters.

The routed path is backed by Megatron Core expert parallelism: head-local expert
IDs are flattened into 3072 experts, routed with the MAGI-2 sigmoid/top-6 rule,
dispatched by MCore all-to-all, and evaluated by Transformer Engine grouped GEMM.
The current correctness recipe fixes TP, PP, and CP to one and uses EP=64.

## Integration run

`pretrain_magi2.py --smoke` runs a reduced model through the normal Bridge
pretraining loop. Without `--smoke`, the launcher requires exactly 64 ranks and
rejects architecture reductions.

The checked-in Slurm launch performs 50 original-size optimizer steps, saves a
torch-dist model/optimizer/RNG checkpoint at step 50, then reloads it and
executes steps 51 and 52 without duplicating the multi-terabyte checkpoint:

```bash
sbatch examples/models/magi2/pretrain_magi2_114b_64gpu.slurm
```

Each run writes a JSON manifest plus one JSONL row per optimizer step. Rows record
finite-loss and finite-gradient checks, skipped iterations, cross-rank peak
memory, and throughput.

## Data boundary

`Magi2LatentDatasetConfig` intentionally supplies deterministic synthetic packed
latents and text conditioning. It validates model, optimizer, communication, and
checkpoint integration but is not a scientific corpus-training dataset. A
production run should replace only this DatasetProvider with a provider that
emits VAE/audio latents, text-encoder states, 3-axis coordinates, and modality
IDs using the same batch keys. The model and pretraining step do not need to be
rewritten.
