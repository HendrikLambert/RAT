#!/usr/bin/env bash
# Single-process launcher for the smoke pipeline on a MacBook (no GPUs).
#
# Usage:
#   bash scripts/run_local.sh                # defaults to rat_smoke on CPU
#   bash scripts/run_local.sh rat_hier_smoke
#   RAT_DEVICE=mps bash scripts/run_local.sh rat_smoke
#
# We skip torchrun (its c10d rendezvous can hang on macOS DNS lookups for a
# single-process run) and instead set the dist env vars directly. The trainer
# still uses DDP with the gloo backend, which works fine with world_size=1.

set -euo pipefail

EXPERIMENT="${1:-rat_smoke}"
export RAT_DEVICE="${RAT_DEVICE:-cpu}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Point hydra's run dir at the repo so artifacts stay local (avoids the
# default /home/xwei/fake_path/... base_dir from configs/config.yaml).
BASE_DIR="${BASE_DIR:-$REPO_ROOT/exp}"
mkdir -p "$BASE_DIR"

# Direct dist env (equivalent to `torchrun --nnodes=1 --nproc-per-node=1`).
export RANK="${RANK:-0}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export LOCAL_RANK="${LOCAL_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

echo "==> experiment=pg19/${EXPERIMENT} device=${RAT_DEVICE} base_dir=${BASE_DIR}"

exec python -u src/benchmark_acc/lm.py \
  experiment=pg19/"${EXPERIMENT}" \
  base_dir="${BASE_DIR}" \
  wandb_use=false
