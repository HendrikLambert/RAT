#!/usr/bin/env bash
# DelftBlue SLURM script for Part 2: Improvement (Hierarchical RAT).
# Submit with:
#   sbatch scripts/delftblue_part2.sbatch

#SBATCH --account=education-eemcs-msc-cs
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=7500M
#SBATCH --time=10:00:00
#SBATCH --job-name=part2
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# Load required cluster modules
module load 2025
module load cuda/12.1
module load python/3.11.9

# Add virtual environment packages to PYTHONPATH (bypasses architecture-specific shebang issues)
export PYTHONPATH="$HOME/envs/rat/lib/python3.11/site-packages:${PYTHONPATH:-}"

cd "$SLURM_SUBMIT_DIR"

EXPERIMENT="pg19/rat_hier"
DATA_DIR="${DATA_DIR:-/scratch/$USER/pg19}"

export RAT_DEVICE=cuda
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Count available GPUs assigned to this task
NPROC_PER_NODE=2
NNODES=1

echo "==> nodes=$NNODES gpus/node=$NPROC_PER_NODE experiment=$EXPERIMENT data_dir=$DATA_DIR"

srun python3 -m torch.distributed.run \
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
  trainer.max_epoch=1 \
  trainer.save_dir="/scratch/$USER/ckpt" \
  base_dir="$SLURM_SUBMIT_DIR" \
  wandb_use=false
