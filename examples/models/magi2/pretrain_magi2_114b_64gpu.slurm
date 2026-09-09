#!/bin/bash
#SBATCH -A coreai_devtech_all
#SBATCH -p batch
#SBATCH -J coreai_devtech_all-magi2:114b-formal
#SBATCH -N 8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH -t 04:00:00
#SBATCH --output=/lustre/fsw/coreai_devtech_all/pima/magi2_workspace/logs/magi2_114b_formal_%j.log

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN

WORKSPACE=/lustre/fsw/coreai_devtech_all/pima/magi2_workspace
REPO="$WORKSPACE/Megatron-Bridge"
export MCORE_REPO="${MCORE_REPO:-$WORKSPACE/Megatron-LM}"
CONTAINER_IMAGE=/lustre/fsw/coreai_devtech_all/pima/containers/nemo_26.04_qwen3vl_te-main-779f17d_fa4b26.sqsh
CONTAINER_MOUNTS=/lustre/fsw/coreai_devtech_all/pima:/lustre/fsw/coreai_devtech_all/pima
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
RUN_ROOT="$WORKSPACE/runs/magi2_114b_formal_${SLURM_JOB_ID}"
CHECKPOINT="$RUN_ROOT/checkpoint"
MAIN_LOSS="$RUN_ROOT/train_001_050.jsonl"
RESUME_LOSS="$RUN_ROOT/resume_051_052.jsonl"

test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT"
{
  printf 'Megatron-Bridge %s\n' "$(git -C "$REPO" rev-parse HEAD)"
  printf 'Megatron-LM %s\n' "$(git -C "$MCORE_REPO" rev-parse HEAD)"
} > "$RUN_ROOT/software_revisions.txt"

run_magi2() {
  local master_port=$1
  shift
  srun --mpi=pmix -N8 -n64 --ntasks-per-node=8 --gpu-bind=none \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="$CONTAINER_MOUNTS" \
    --export=ALL,MASTER_ADDR="$MASTER_ADDR",MASTER_PORT="$master_port",CONTAINER_IMAGE="$CONTAINER_IMAGE" \
    bash -c '
      set -euo pipefail
      export RANK=$SLURM_PROCID
      export WORLD_SIZE=$SLURM_NTASKS
      export LOCAL_RANK=$SLURM_LOCALID
      export PYTHONPATH="'"$MCORE_REPO"':'"$REPO"'/src${PYTHONPATH:+:$PYTHONPATH}"
      exec /opt/venv/bin/python \
        "'"$REPO"'/examples/models/magi2/launch_pretrain_magi2.py" "$@"
    ' bash "$@"
}

run_magi2 29651 \
  --train-iters 52 \
  --lr-decay-iters 52 \
  --save-interval 50 \
  --exit-interval 50 \
  --checkpoint-dir "$CHECKPOINT" \
  --loss-output "$MAIN_LOSS" \
  --tensorboard-dir "$RUN_ROOT/tensorboard"

test "$(wc -l < "$MAIN_LOSS")" -eq 50
test "$(<"$CHECKPOINT/latest_checkpointed_iteration.txt")" -eq 50

run_magi2 29652 \
  --train-iters 52 \
  --lr-decay-iters 52 \
  --save-interval 1000 \
  --no-save \
  --load "$CHECKPOINT" \
  --checkpoint-dir "$RUN_ROOT/resume_checkpoint" \
  --loss-output "$RESUME_LOSS"

test "$(wc -l < "$RESUME_LOSS")" -eq 2
python3 "$REPO/examples/models/magi2/analyze_magi2_run.py" \
  "$MAIN_LOSS" --expected-start 1 --expected-end 50 --warmup-steps 5 \
  --output "$RUN_ROOT/train_report.json"
python3 "$REPO/examples/models/magi2/analyze_magi2_run.py" \
  "$RESUME_LOSS" --expected-start 51 --expected-end 52 --warmup-steps 0 \
  --output "$RUN_ROOT/resume_report.json"
sha256sum \
  "$RUN_ROOT/software_revisions.txt" \
  "$MAIN_LOSS" "$MAIN_LOSS.manifest.json" \
  "$RESUME_LOSS" "$RESUME_LOSS.manifest.json" \
  "$RUN_ROOT/train_report.json" "$RUN_ROOT/resume_report.json"
du -sh "$CHECKPOINT"
echo "MAGI2_STAGE7_FORMAL status=passed world_size=64 ep=64 checkpoint_iteration=50 resume_iterations=51-52"
