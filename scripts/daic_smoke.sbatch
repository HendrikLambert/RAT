#!/usr/bin/env bash
# DAIC cluster smoke test (Apptainer) — verify CUDA + image + data loading before
# the full run. Smoke configs cap reads at limit_tokens=200000, so they use the
# SAME full bins under $SHARED/pg19 (no separate tiny slice needed).
# Submit any smoke variant via EXPERIMENT:
#   EXPERIMENT=rat_hier_smoke  sbatch scripts/daic_smoke.sbatch
#   EXPERIMENT=rat_smoke       sbatch scripts/daic_smoke.sbatch
#   EXPERIMENT=attention_smoke sbatch scripts/daic_smoke.sbatch

#SBATCH --account=ewi-st-dis
#SBATCH --partition=general
#SBATCH --qos=interactive
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=rat-smoke
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

SHARED="${SHARED:-/tudelft.net/staff-umbrella/CS4725/$USER}"
SIF="${SIF:-$SHARED/rat.sif}"

module use /opt/insy/modulefiles
module load cuda/12.1

cd "$SLURM_SUBMIT_DIR"

EXPERIMENT="${EXPERIMENT:-rat_hier_smoke}"
DATA_DIR="${DATA_DIR:-$SHARED/pg19}"

export RAT_DEVICE=cuda
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HOME="$SHARED/hf"

NPROC_PER_NODE=1
NNODES=1

echo "==> nodes=$NNODES gpus/node=$NPROC_PER_NODE experiment=pg19/$EXPERIMENT data_dir=$DATA_DIR sif=$SIF"

srun apptainer exec --nv \
  -B /tudelft.net/:/tudelft.net/ \
  -B "$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR" \
  --pwd "$SLURM_SUBMIT_DIR" \
  "$SIF" \
  torchrun \
    --nnodes="$NNODES" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1):29500" \
    src/benchmark_acc/lm.py \
    experiment=pg19/"$EXPERIMENT" \
    data.train.tokenized_file_path="$DATA_DIR/gpt2-train.bin" \
    data.val.tokenized_file_path="$DATA_DIR/gpt2-val.bin" \
    trainer.device=cuda \
    trainer.dtype=bfloat16 \
    trainer.max_epoch=1 \
    trainer.save_dir="$SHARED/ckpt" \
    trainer.torch_compile="${COMPILE:-false}" \
    base_dir="$SLURM_SUBMIT_DIR" \
    wandb_use=false
