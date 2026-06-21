#!/usr/bin/env bash
# DAIC — tokenize PG19 into shared storage as a CPU-only SLURM job.
# Login nodes SIGKILL heavy compute, so tokenization must run on a compute node.
# Submit with:
#   sbatch scripts/daic_tokenize.sbatch
#
# Requires the container (scripts/daic_build_container.sh) to already exist.
# If DAIC compute nodes have NO internet, the load_dataset download will fail —
# see the pre-cache note at the bottom.

#SBATCH --account=ewi-st-dis
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=rat-tok
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

SHARED="${SHARED:-/tudelft.net/staff-umbrella/CS4725/$USER}"
SIF="${SIF:-$SHARED/rat.sif}"

cd "$SLURM_SUBMIT_DIR"

export HF_HOME="$SHARED/hf"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# num_proc>1 forks workers; fine with a real CPU allocation. Drop to 1 if the
# fast-tokenizer fork ever "abruptly dies".
NUM_PROC="${NUM_PROC:-8}"

echo "==> tokenizing PG19 -> $SHARED/pg19 (num_proc=$NUM_PROC) sif=$SIF"

srun apptainer exec \
  -B /tudelft.net/:/tudelft.net/ \
  -B "$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR" \
  --pwd "$SLURM_SUBMIT_DIR" \
  "$SIF" \
  python tokenize/pg19.py --out_dir "$SHARED/pg19" --num_proc "$NUM_PROC"

# --- If the job fails with a network/connection error (no compute internet) ----
# Pre-cache on a node that DOES reach the internet, then re-run this job offline:
#   (login node, download only) HF_HOME=$SHARED/hf apptainer exec \
#     -B /tudelft.net/:/tudelft.net/ $SHARED/rat.sif python -c \
#     "from datasets import load_dataset; load_dataset('deepmind/pg19', trust_remote_code=True)"
#   then add  export HF_DATASETS_OFFLINE=1  above the srun and resubmit.
