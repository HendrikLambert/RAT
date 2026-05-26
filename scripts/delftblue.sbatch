#!/usr/bin/env bash
# DelftBlue SLURM template for the 200M PG19 ablation run.
# Submit with: sbatch scripts/delftblue.sbatch
#
# Reproduces the paper's preliminary studies at the 200M scale (12 layers,
# d_model=1024, seq_len=8192). Defaults to the RAT (chunk_size=128) variant;
# swap experiment=pg19/attention or pg19/rnn for the other baselines.
#
# Before submitting:
#   1. Fill in the lines marked EDIT: below.
#   2. Tokenize PG19 once and put the .bin files at $SCRATCH/pg19/, or
#      override data.train.tokenized_file_path / data.val.tokenized_file_path
#      via the hydra command-line.
#   3. Update --max_epoch to control the iteration count (paper uses 5; the
#      proposal targets 1 iteration ~ 3B tokens).

# EDIT: your TU Delft research account / allocation id
#SBATCH --account=research-eemcs-cs
# EDIT: pick the GPU partition you have access to on DelftBlue (gpu-a100 / gpu-h100)
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --job-name=rat-pg19
#SBATCH --output=slurm-%j.out

set -euo pipefail

# EDIT: load the modules your DelftBlue setup needs (CUDA, Python, etc).
module load 2023r1
module load cuda/12.1
module load python/3.11

# EDIT: path to your venv / conda env with torch>=2.5, transformers, datasets, hydra-core, ...
source "$HOME/envs/rat/bin/activate"

cd "$SLURM_SUBMIT_DIR"

EXPERIMENT="${EXPERIMENT:-pg19/rat}"
DATA_DIR="${DATA_DIR:-$HOME/scratch/pg19}"

export RAT_DEVICE=cuda
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

NPROC_PER_NODE="$SLURM_GPUS_ON_NODE"
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
