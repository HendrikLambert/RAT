#!/usr/bin/env bash
# =============================================================================
# Cloud Training Script for RAT (RunPod / any cloud GPU)
# =============================================================================
# Usage:
#   1. SSH into your RunPod instance
#   2. Run: bash cloud_train.sh <experiment>
#      where <experiment> is "hier" or "hybrid"
#
# Prerequisites:
#   - The PG19 tokenized data must already be transferred to /workspace/pg19/
#     (see instructions in the RunPod setup guide)
# =============================================================================

set -euo pipefail

EXPERIMENT="${1:-hier}"  # "hier" or "hybrid"

# Map experiment name to config
if [ "$EXPERIMENT" = "hier" ]; then
    CONFIG="pg19/rat_hier"
elif [ "$EXPERIMENT" = "hybrid" ]; then
    CONFIG="pg19/rat_hybrid"
else
    echo "Unknown experiment: $EXPERIMENT. Use 'hier' or 'hybrid'."
    exit 1
fi

echo "============================================"
echo "  RAT Cloud Training - $EXPERIMENT"
echo "============================================"

# --- 1. Setup ---
cd /workspace

# Clone repo if not already present
if [ ! -d "RAT" ]; then
    echo "[1/4] Cloning repository..."
    git clone https://github.com/HendrikLambert/RAT.git
    cd RAT
    git checkout rat-split-ready-pipe
else
    echo "[1/4] Repository already exists, pulling latest..."
    cd RAT
    git pull --rebase
fi

# --- 2. Install dependencies ---
echo "[2/4] Installing Python dependencies..."
pip install -q torch>=2.5 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -q transformers>=4.40 datasets>=2.18 hydra-core>=1.3 omegaconf>=2.3 \
    einops>=0.7 easydict>=1.10 torchmetrics>=1.3 tqdm>=4.66 wandb>=0.16 \
    numpy>=1.26 scikit-learn>=1.3 sentencepiece>=0.1.99

# --- 3. Verify GPU ---
echo "[3/4] Verifying GPU..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
n = torch.cuda.device_count()
print(f'GPU count: {n}')
for i in range(n):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
    print(f'  Memory: {torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB')
"

# --- 4. Verify data ---
DATA_DIR="/workspace/pg19"
if [ ! -f "$DATA_DIR/gpt2-train.bin" ] || [ ! -f "$DATA_DIR/gpt2-val.bin" ]; then
    echo ""
    echo "ERROR: PG19 tokenized data not found at $DATA_DIR/"
    echo "Please transfer from DelftBlue first:"
    echo "  scp mdchu@login.delftblue.tudelft.nl:/scratch/mdchu/pg19/gpt2-*.bin /workspace/pg19/"
    exit 1
fi
echo "[4/4] Data found at $DATA_DIR"

# --- 5. Detect GPU count and run training ---
NPROC=$(python3 -c "import torch; print(torch.cuda.device_count())")

echo ""
echo "============================================"
echo "  Starting training: $CONFIG"
echo "  GPUs: $NPROC"
echo "============================================"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

python3 -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node="$NPROC" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="localhost:29500" \
  src/benchmark_acc/lm.py \
  experiment="$CONFIG" \
  data.train.tokenized_file_path="$DATA_DIR/gpt2-train.bin" \
  data.val.tokenized_file_path="$DATA_DIR/gpt2-val.bin" \
  trainer.device=cuda \
  trainer.dtype=bfloat16 \
  trainer.max_epoch=1 \
  trainer.save_dir="/workspace/ckpt" \
  trainer.torch_compile=true \
  base_dir="/workspace/RAT" \
  wandb_use=false

echo ""
echo "============================================"
echo "  Training complete: $EXPERIMENT"
echo "============================================"
