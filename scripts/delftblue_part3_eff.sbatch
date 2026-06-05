#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=7500M
#SBATCH --time=04:00:00
#SBATCH --job-name=eff_bench
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

module load python/3.11.9
export PYTHONPATH="$HOME/envs/rat/lib/python3.11/site-packages:${PYTHONPATH:-}"

cd "$SLURM_SUBMIT_DIR"

echo "======================================"
echo "Running Hardware Efficiency Benchmarks"
echo "======================================"

echo "Running Attention Benchmarks..."
python3 src/benchmark_eff/attention_eff.py > eff_attention.log

echo "Running RAT Benchmarks..."
python3 src/benchmark_eff/rat_eff.py > eff_rat.log

echo "Running RNN Benchmarks..."
python3 src/benchmark_eff/rnn_eff.py > eff_rnn.log

echo "Done! Results saved to eff_attention.log, eff_rat.log, and eff_rnn.log"
