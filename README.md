# 🐭 RAT 

## Intro
**RAT: Bridging RNN Efficiency and Attention Accuracy in Language Modeling**

Xiuying Wei, Anunay Yadav, Razvan Pascanu, Caglar Gulcehre

![alt text](rat_all.png)
>  Transformers have become the cornerstone of modern large-scale language models; however, their dependence on softmax attention poses a major computational bottleneck, particularly in long-context settings. In this work, rather than following prevalent approaches such as linear attention (or SSMs) and local attention, we introduce an intermediate design called RAT between recurrence and attention mechanisms. It partitions the input into chunks, applies a simple linear recurrence within each chunk to capture local dependencies, and then performs softmax attention across chunks to model long-range interactions. By adjusting the size of the chunk, RAT enables flexible trade-offs, combining the strengths of RNN and attention. Empirically, with a chunk size of 16, the RAT layer achieves a $7\times$ improvement in training speed with 100K token sequences and $9\times$ in generation at 4K sequence length, while maintaining similar or sometimes even better accuracy compared to standard attention. We demonstrate this by training 1.3B parameter models from scratch and performing large-scale evaluations, including short- and long-context benchmarks, as well as supervised fine-tuning~(SFT). We further propose a hybrid architecture that interleaves RAT with local attention. By combining efficient long-range modeling with strong local interactions, this hybrid design not only improves inference speed and reduces cache memory usage compared to attention, but also consistently enhances performance, for example, achieving an average 1 point gain in commonsense reasoning tasks, up to 4 points on code tasks, and a 1 point Rouge-L increase in a summarization SFT task.

## File Organization
```
├── configs
│   ├── config.yaml
│   ├── data
│   ├── experiment:  entry to launch experiments
│   ├── model
│   ├── optim
│   ├── task
├── src
│   ├── benchmark_acc: entry to benchmark accuracy
│   ├── benchmark_eff: entry to benchmark efficiency
│   ├── data
│   ├── model
│   ├── model
│   │   ├── backbone: sequence model backbone, and layers, including attention, ffn, rat (ours), and rnn
│   │   ├── embedding: lm embedding and positional embedding
│   │   ├── head: lm head
│   ├── optim: lr scheduler and optimizer
│   ├── task: concatenate backbone, embedding, and head, and also metric and loss
│   ├── trainer: ddp trainer
│   ├── utils
```

## Env
We provide the environment in the Dockerfile

## Pretraining
* Prepare training data
    ```
    # downloading
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-100BT")
    # tokenize the data with LlaMA2 tokenizer
    cd tokenize && python fineweb_edu.py --tokenizer llama --num_proc 32
    ```

* pretraining the 1.3B model on 100B tokens
    ```
    torchrun --nnodes=4 --nproc_per_node=4 lm.py experiment=fineweb_edu/attention-xl
    torchrun --nnodes=4 --nproc-per-node=4 lm.py experiment=fineweb_edu/rat-xl
    torchrun --nnodes=4 --nproc-per-node=4 lm.py experiment=fineweb_edu/rnn-xl
    # interleave with local attention
    torchrun --nnodes=4 --nproc_per_node=4 lm.py experiment=fineweb_edu/attention_localattention_interleave-xl
    torchrun --nnodes=4 --nproc-per-node=4 lm.py experiment=fineweb_edu/rat_localattention_interleave-xl
    ```
    We obtain the perplexity shown in Figure 4(c).

* Pretrained checkpoints can be found in https://huggingface.co/barpitf/RAT/tree/main.
## Downstream results
After we obtain the pretrained models, we can conduct different downstream evaluations with them. Detailed configurations can be found in Appendix A.2.
* Table 2. We evaluate it using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) repo. We provide necessary files in eval/lm_harness

* Table 3. We evaluate it using [LongBenchV1](https://github.com/THUDM/LongBench) repo. We provide necessary files in eval/longbench.

* Table 4. As it is hard for pretrained-only models to deal with heavy prompt-style datasets in LongBench, we also compare different models on some SFT datasets.

    ```
    # prepare train/val/test data
    cd tokenize && python sft.py --task narrativeqa_summary --num_proc 32 --max_length 4096 --split train

    # fine-tuning the pretrained model on narrativeqa_summary by specifying data.\_name\_
    export config="optim.optimizer.lr=1.0e-5 data.global_batch_size=128 trainer.max_epoch=1 data._name_=narrativeqa_summary"
    torchrun --nnodes=1 --nproc-per-node=4 lm.py experiment=sft/rat-xl ${config}

    # generate answers
    export config="optim.optimizer.lr=1.0e-5 data.global_batch_size=128 trainer.max_epoch=1 data._name_=narrativeqa_summary"
    torchrun --nnodes=1 --nproc-per-node=4 generation.py experiment=sft/rat-xl  wandb_use=false ${config}"

    # use eval/sft_eval to score the answers
    ```

* Table 12. We evaluate using [RULER benchmark](https://github.com/NVIDIA/RULER) repo. We provide necessary files in eval/ruler.


## Local smoke run (MacBook M1, CPU)

The original repo is CUDA-only. This fork adds a minimal CPU/MPS fallback so the
full pipeline (tokenize → train → validate) can be exercised on a laptop before
submitting the real job to a cluster. Numbers from the smoke run are **not**
meaningful for the paper reproduction — they only prove the pipeline works.

```bash
# 1. Environment (Python 3.10+ recommended)
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. Tokenize a tiny PG19 slice (~50 books, streams in only what is needed)
python tokenize/pg19.py --out_dir ./data/pg19 --max_examples 50 --max_val_examples 5 --num_proc 4

# 3. Smoke train each baseline (each run takes a few minutes on M1)
bash scripts/run_local.sh attention_smoke
bash scripts/run_local.sh rat_smoke
bash scripts/run_local.sh rnn_smoke

# 4. Smoke train Hierarchical RAT (per-layer chunk size)
bash scripts/run_local.sh rat_hier_smoke
```

Each run prints `validation loss is X.XXXX and metric is X.XX` at the end.
Set `RAT_DEVICE=mps` before the command to try the Apple Silicon GPU backend
instead of CPU (slower start-up, faster steady state).

Verified on M1 Pro / Python 3.12 / torch 2.8 / 5 train books / seq_len=256:

| Variant | Wall time | Final val loss | Val PPL |
|---|---|---|---|
| `attention_smoke` | 30 s | 7.6540 | 2108.98 |
| `rat_smoke` (L=64)| 47 s | 7.6517 | 2104.24 |
| `rnn_smoke` | 37 s | 7.5908 | 1979.84 |
| `rat_hier_smoke` (L=[64, 128]) | 58 s | 7.6634 | 2128.94 |

These numbers are **not** a paper reproduction — the model is 12 M params
(50 257-vocab embedding dominates), trained on 200 K tokens at d_model=128.
They prove the pipeline executes end-to-end, nothing more. Real PPL trends
need the 200 M config on a GPU.

## Running on DelftBlue

Three specialized SLURM batch submission scripts are provided under `scripts/` for executing cluster experiments. They are pre-configured to use the education account (`education-eemcs-msc-cs`) and appropriate GPU allocations.

### 0. Prepare Dataset (Login Node)
Because DelftBlue compute nodes do not have internet access, you must download and tokenize the PG19 dataset on a **login node** (which has public internet access) before submitting training jobs.

To tokenize the full dataset to your `/scratch/mdchu/pg19` directory, run:
```bash
python tokenize/pg19.py --out_dir /scratch/mdchu/pg19 --num_proc 8
```
*(Optionally, for a quick cluster check, download a tiny slice of 50 books first using `python tokenize/pg19.py --out_dir ./data/pg19 --max_examples 50 --max_val_examples 5 --num_proc 4` on the login node).*

### 1. Cluster Smoke Test (Low Queue Time)
We use the dedicated `gpu-a100-small` partition for fast, lightweight pipeline verification (max 4h, 1 GPU, 2 CPUs). This allows you to verify that CUDA execution, conda/venv environments, and data loading work perfectly before running full experiments.

Submit any smoke variant by setting the `EXPERIMENT` environment variable:
```bash
# Verify standard Attention smoke test on cluster
EXPERIMENT=attention_smoke sbatch scripts/delftblue_smoke.sbatch

# Verify standard RAT smoke test on cluster
EXPERIMENT=rat_smoke sbatch scripts/delftblue_smoke.sbatch

# Verify standard RNN smoke test on cluster
EXPERIMENT=rnn_smoke sbatch scripts/delftblue_smoke.sbatch

# Verify Hierarchical RAT smoke test on cluster
EXPERIMENT=rat_hier_smoke sbatch scripts/delftblue_smoke.sbatch
```

### 2. Part 1: Reproducibility Ablation (200M Model, 3B Tokens)
Full reproducibility jobs request **2 GPUs** on the `gpu-a100` partition with a 24-hour wall time limit. Submit the different variants (MVP baselines and stretch goals) as follows:

```bash
# --- MVP Baselines (Sequence Length T = 8192) ---
# Attention (L=1)
EXPERIMENT=pg19/attention sbatch scripts/delftblue_part1.sbatch

# RNN (L=T)
EXPERIMENT=pg19/rnn sbatch scripts/delftblue_part1.sbatch

# RAT (L=128)
EXPERIMENT=pg19/rat_l128 sbatch scripts/delftblue_part1.sbatch

# --- Stretch Goals ---
# RAT (L=64)
EXPERIMENT=pg19/rat_l64 sbatch scripts/delftblue_part1.sbatch

# RAT (L=256)
EXPERIMENT=pg19/rat_l256 sbatch scripts/delftblue_part1.sbatch

# RAT L=16 (Tested at shorter context T=4096, global batch size 256 for ~1M tokens)
EXPERIMENT=pg19/rat_l16 sbatch scripts/delftblue_part1.sbatch
```

### 3. Part 2: Improvement (Hierarchical RAT)
To train the Split Hierarchical RAT configuration (layers 0-3 at $L=64$, layers 4-11 at $L=256$, matching the baseline's FLOP budget), submit the Part 2 job:

```bash
sbatch scripts/delftblue_part2.sbatch
```

## Efficiency results
We test latency on the GH200 GPU.
* single layer (including the input and output projections)

    ```
    cd src/benchmark_eff
    python attention_eff.py 
    python rnn_eff.py
    python rat_eff.py
    ```
* whole model's maximum throughput
    ```
    cd src/benchmark_eff
    # we run it twice. The first time is used to find the maximum batch size we can set. The second time is used to test the latency.
    torchrun --nnodes=1 --nproc-per-node=1 model_eff.py experiment=fineweb_edu/rat-xl
    ```
