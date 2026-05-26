#!/usr/bin/env bash
# DAIC (Delft AI Cluster) SLURM template for the 200M PG19 ablation run.
# Submit with: sbatch scripts/daic.sbatch
#
# Structure mirrors delftblue.sbatch; DAIC partition/account flags differ.

# EDIT: DAIC group / account
#SBATCH --account=insy
# EDIT: DAIC GPU partition (e.g. general / gpu-rtx / gpu-a100)
#SBATCH --partition=general
# EDIT: GPU type constraint (DAIC uses --gres for GPUs)
#SBATCH --gres=gpu:a100:4
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=rat-pg19
#SBATCH --output=slurm-%j.out

set -euo pipefail

# EDIT: load modules / activate env (DAIC typically uses conda)
# module load cuda/12.1
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate rat

cd "$SLURM_SUBMIT_DIR"

EXPERIMENT="${EXPERIMENT:-pg19/rat}"
DATA_DIR="${DATA_DIR:-/tudelft.net/staff-umbrella/$USER/pg19}"

export RAT_DEVICE=cuda
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Count requested GPUs from --gres=gpu:...:N
NPROC_PER_NODE="$(echo "$SLURM_JOB_GPUS" | tr ',' '\n' | wc -l)"
NNODES="$SLURM_JOB_NUM_NODES"

echo "==> nodes=$NNODES gpus/node=$NPROC_PER_NODE experiment=$EXPERIMENT data_dir=$DATA_DIR"

srun torchrun \
  --nnodes="$NNODES" \
  --nproc_per_node="$NPROC_PER_NODE" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1):29500" \
  src/benchmark_acc/lm.py \
  experiment="$EXPERIMENT" \
  data.train.tokenized_file_path="$DATA_DIR/gpt2-train.bin" \
  data.val.tokenized_file_path="$DATA_DIR/gpt2-val.bin" \
  trainer.max_epoch=1
