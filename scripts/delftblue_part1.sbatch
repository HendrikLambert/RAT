#!/usr/bin/env bash
# DelftBlue SLURM script for Part 1: Reproducibility Ablation (Baselines & Stretch Goals).
# Submit with:
#   # MVP Baselines
#   EXPERIMENT=pg19/attention sbatch scripts/delftblue_part1.sbatch
#   EXPERIMENT=pg19/rat_l128  sbatch scripts/delftblue_part1.sbatch
#   EXPERIMENT=pg19/rnn       sbatch scripts/delftblue_part1.sbatch
#
#   # Stretch Goals
#   EXPERIMENT=pg19/rat_l64   sbatch scripts/delftblue_part1.sbatch
#   EXPERIMENT=pg19/rat_l256  sbatch scripts/delftblue_part1.sbatch
#   EXPERIMENT=pg19/rat_l16   sbatch scripts/delftblue_part1.sbatch

#SBATCH --account=education-eemcs-msc-cs
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=23:59:00
#SBATCH --job-name=rat-part1
#SBATCH --output=slurm-part1-%j.out

set -euo pipefail

# Load required cluster modules
module load 2025
module load cuda/12.1
module load python/3.11.9

# Activate virtual environment. Update this path if your environment is installed elsewhere.
source "$HOME/envs/rat/bin/activate"

cd "$SLURM_SUBMIT_DIR"

EXPERIMENT="${EXPERIMENT:-pg19/rat_l128}"
DATA_DIR="${DATA_DIR:-/scratch/$USER/pg19}"

export RAT_DEVICE=cuda
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Count available GPUs assigned to this task
NPROC_PER_NODE=2
NNODES=1

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
  trainer.device=cuda \
  trainer.dtype=bfloat16 \
  trainer.max_epoch=1
