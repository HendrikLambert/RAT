#!/usr/bin/env bash
# DelftBlue SLURM script for cluster smoke testing.
# Submit with:
#   EXPERIMENT=attention_smoke sbatch scripts/delftblue_smoke.sbatch
#   EXPERIMENT=rat_smoke       sbatch scripts/delftblue_smoke.sbatch
#   EXPERIMENT=rnn_smoke       sbatch scripts/delftblue_smoke.sbatch
#   EXPERIMENT=rat_hier_smoke  sbatch scripts/delftblue_smoke.sbatch

#SBATCH --account=education-eemcs-msc-cs
#SBATCH --partition=gpu-a100-small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=rat-smoke
#SBATCH --output=slurm-smoke-%j.out

set -euo pipefail

# Load required cluster modules
module load 2025
module load cuda/12.1
module load python/3.11.9

# Activate virtual environment. Update this path if your environment is installed elsewhere.
source "$HOME/envs/rat/bin/activate"

cd "$SLURM_SUBMIT_DIR"

EXPERIMENT="${EXPERIMENT:-rat_smoke}"
DATA_DIR="${DATA_DIR:-./data/pg19}"

export RAT_DEVICE=cuda
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

NPROC_PER_NODE=1
NNODES=1

echo "==> nodes=$NNODES gpus/node=$NPROC_PER_NODE experiment=pg19/$EXPERIMENT data_dir=$DATA_DIR"

srun torchrun \
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
  trainer.max_epoch=1
