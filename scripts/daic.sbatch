#!/usr/bin/env bash
# DAIC (Delft AI Cluster) — Part 2: Hierarchical RAT (pg19/rat_hier) full run,
# executed inside the Apptainer image built by scripts/daic_build_container.sh.
# Mirrors scripts/delftblue_part2.sbatch. Submit (after build + tokenization):
#   sbatch scripts/daic.sbatch
#
# A40 (48 GB, bf16-capable, plentiful on `general`) is the default GPU; switch
# the --gres line to gpu:a100:2 for speed if you'll queue for the single 2xA100
# node. Avoid v100: no native bfloat16.

#SBATCH --account=ewi-st-dis
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --gres=gpu:a40:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=23:00:00
#SBATCH --job-name=rat-hier
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

SHARED="${SHARED:-/tudelft.net/staff-umbrella/CS4725/$USER}"
SIF="${SIF:-$SHARED/rat.sif}"

# Host CUDA driver libs for --nv (DAIC modules live under /opt/insy/modulefiles)
module use /opt/insy/modulefiles
module load cuda/12.1

cd "$SLURM_SUBMIT_DIR"

EXPERIMENT="${EXPERIMENT:-pg19/rat_hier}"
DATA_DIR="${DATA_DIR:-$SHARED/pg19}"

# These are inherited into the container (apptainer passes host env by default).
export RAT_DEVICE=cuda
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HOME="$SHARED/hf"
# torch.compile caches default to $HOME (only 8 GB on DAIC) — redirect to local
# /tmp so they don't fill the home quota or hit home-FS permission quirks.
export TORCHINDUCTOR_CACHE_DIR="/tmp/$USER/torchinductor"
export TRITON_CACHE_DIR="/tmp/$USER/triton"
export CUDA_CACHE_PATH="/tmp/$USER/nv"

# Count GPUs assigned to this job from --gres=gpu:<type>:N
NPROC_PER_NODE="$(echo "$SLURM_JOB_GPUS" | tr ',' '\n' | grep -c .)"
NNODES="$SLURM_JOB_NUM_NODES"

echo "==> nodes=$NNODES gpus/node=$NPROC_PER_NODE experiment=$EXPERIMENT data_dir=$DATA_DIR sif=$SIF"

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
    experiment="$EXPERIMENT" \
    data.train.tokenized_file_path="$DATA_DIR/gpt2-train.bin" \
    data.val.tokenized_file_path="$DATA_DIR/gpt2-val.bin" \
    trainer.device=cuda \
    trainer.dtype=bfloat16 \
    trainer.max_epoch=1 \
    trainer.save_dir="$SHARED/ckpt" \
    trainer.torch_compile=true \
    trainer.save_interval=0.05 \
    base_dir="$SLURM_SUBMIT_DIR" \
    wandb_use=false \
    "$@"
