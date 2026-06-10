#!/usr/bin/env python3
import os
import re
import json
import sys

def parse_slurm_files(dir_path="."):
    results = {}
    # Matches filenames like: slurm-attn-10028953.out, slurm-l16-10087274.out
    pattern = re.compile(r"slurm-(.+)-(\d+)\.out")
    
    for filename in os.listdir(dir_path):
        match = pattern.match(filename)
        if not match:
            continue
        job_name, job_id = match.groups()
        
        # Exclude temporary smoke test files and raw partition setup runs
        if "smoke" in job_name or "part" in job_name or "eff" in job_name:
            continue
            
        file_path = os.path.join(dir_path, filename)
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {filename}: {e}", file=sys.stderr)
            continue
            
        # Extract validation perplexity (true metric)
        val_loss = None
        val_ppl = None
        val_match = re.search(r"validation loss is\s+([\d.]+)\s+and metric is\s+([\d.]+)", content)
        if val_match:
            val_loss = float(val_match.group(1))
            val_ppl = float(val_match.group(2))
            
        # Extract training throughput & final training loss
        train_loss = None
        fwd_bwd = None
        # Finds lines containing: {'train_loss': ..., 'fwd+bwd': ...}
        dict_matches = re.findall(r"\{'train_loss':\s*([\d.]+),.*'fwd\+bwd':\s*([\d.]+)", content)
        if dict_matches:
            train_loss = float(dict_matches[-1][0])
            fwd_bwd = float(dict_matches[-1][1])
            
        # Determine job status
        status = "COMPLETED"
        if "ChildFailedError" in content or "BackendCompilerFailed" in content:
            status = "CRASHED (Compiler Error)"
        elif "OutOfMemoryError" in content or "CUDA out of memory" in content:
            status = "CRASHED (OOM)"
        elif "Traceback" in content:
            status = "CRASHED"
            
        results[job_name] = {
            "job_id": job_id,
            "filename": filename,
            "status": status,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "fwd_bwd_latency_s": fwd_bwd
        }
    return results

def parse_eff_logs(dir_path="."):
    eff_results = {}
    files = {
        "Attention": "eff_attention.log",
        "RAT": "eff_rat.log",
        "RNN": "eff_rnn.log"
    }
    
    for model_name, filename in files.items():
        file_path = os.path.join(dir_path, filename)
        if not os.path.exists(file_path):
            continue
            
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading {filename}: {e}", file=sys.stderr)
            continue
            
        runs = []
        gen_scans = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # Parse train/context/gen lines
            if line.startswith("seq:") and "bs:" in line:
                if "/" not in line: # Individual run info
                    seq_match = re.search(r"seq:\s*(\d+)", line)
                    bs_match = re.search(r"bs:\s*(\d+)", line)
                    if seq_match and bs_match:
                        seq = int(seq_match.group(1))
                        bs = int(bs_match.group(1))
                        
                        # Look at the next line for metric values
                        if i + 1 < len(lines) and lines[i+1].strip().startswith("train:"):
                            next_line = lines[i+1].strip()
                            train_m = re.search(r"train:\s*([\d.]+)", next_line)
                            context_m = re.search(r"context:\s*([\d.]+)", next_line)
                            gen_m = re.search(r"gen:\s*([\d.]+)", next_line)
                            
                            runs.append({
                                "seq": seq,
                                "bs": bs,
                                "train_ms": float(train_m.group(1)) if train_m else None,
                                "context_ms": float(context_m.group(1)) if context_m else None,
                                "gen_ms": float(gen_m.group(1)) if gen_m else None
                            })
                            i += 1
                else: # Generation scan across multiple batch sizes
                    seq_match = re.search(r"seq:\s*(\d+)", line)
                    bs_match = re.search(r"bs:\s*(.+)", line)
                    if seq_match and bs_match:
                        seq = int(seq_match.group(1))
                        bs_values = bs_match.group(1).split("/")
                        gen_scans.append({
                            "seq": seq,
                            "latencies": bs_values
                        })
            i += 1
            
        if runs or gen_scans:
            eff_results[model_name] = {
                "runs": runs,
                "gen_scans": gen_scans
            }
            
    return eff_results

def generate_markdown(slurm_data, eff_data):
    md = []
    md.append("# RAT Reproducibility & Improvement: Consolidated Results\n")
    md.append("This document was automatically generated by `scripts/extract_results.py` and contains the consolidated metrics for all cluster runs.\n")
    
    # 1. Main model runs table
    md.append("## 1. Model Performance & Validation Perplexity")
    md.append("| Model Name (Job Name) | Job ID | Status | Final Train Loss | Final Val Loss | Validation PPL (Metric) | Latency (s/step) |")
    md.append("| --- | --- | --- | --- | --- | --- | --- |")
    
    # Sort keys for readability
    sorted_keys = sorted(slurm_data.keys(), key=lambda x: (
        0 if x == "attn" else
        1 if x.startswith("l") and x[1:].isdigit() else
        2 if x == "rnn" else
        3 if x == "hier" else 4
    ))
    
    for name in sorted_keys:
        d = slurm_data[name]
        train_l = f"{d['train_loss']:.4f}" if d["train_loss"] is not None else "N/A"
        val_l = f"{d['val_loss']:.4f}" if d["val_loss"] is not None else "N/A"
        val_p = f"{d['val_ppl']:.4f}" if d["val_ppl"] is not None else "N/A"
        lat = f"{d['fwd_bwd_latency_s']:.2f}" if d["fwd_bwd_latency_s"] is not None else "N/A"
        
        md.append(f"| {name} | {d['job_id']} | {d['status']} | {train_l} | {val_l} | {val_p} | {lat} |")
    md.append("\n")
    
    # 2. Hardware efficiency benchmark results
    if eff_data:
        md.append("## 2. Hardware Efficiency Benchmarks (Triton / do_bench)")
        md.append("Latency is reported in milliseconds (ms).\n")
        
        for model_name, data in eff_data.items():
            if data["runs"]:
                md.append(f"### {model_name} Latency Profile")
                md.append("| Sequence Length | Batch Size | Train Step (ms) | Context Processing (ms) | Generation Step (ms) |")
                md.append("| --- | --- | --- | --- | --- |")
                for r in data["runs"]:
                    md.append(f"| {r['seq']} | {r['bs']} | {r['train_ms']:.2f} | {r['context_ms']:.2f} | {r['gen_ms']:.4f} |")
                md.append("\n")
                
            if data["gen_scans"]:
                md.append(f"### {model_name} Generation Scanning Latency")
                md.append("Generation latency across batch sizes `[64, 128, 256, 512, 1024, 4096]`:\n")
                md.append("| Sequence Length | BS=64 | BS=128 | BS=256 | BS=512 | BS=1024 | BS=4096 |")
                md.append("| --- | --- | --- | --- | --- | --- | --- |")
                for gs in data["gen_scans"]:
                    row = f"| {gs['seq']}"
                    for lat in gs["latencies"]:
                        row += f" | {lat}"
                    # Pad columns if OOM occurred early or list is short
                    row += " |" * (6 - len(gs["latencies"])) + " |"
                    md.append(row)
                md.append("\n")
    else:
        md.append("## 2. Hardware Efficiency Benchmarks")
        md.append("> [!NOTE]\n> Efficiency benchmark logs (`eff_*.log`) were not found in the current directory. Run `scripts/delftblue_part3_eff.sbatch` to generate them.\n")
        
    return "\n".join(md)

def main():
    dir_path = "."
    if len(sys.argv) > 1:
        dir_path = sys.argv[1]
        
    slurm_data = parse_slurm_files(dir_path)
    eff_data = parse_eff_logs(dir_path)
    
    if not slurm_data and not eff_data:
        print("No slurm log files (slurm-*.out) or efficiency logs (eff_*.log) found.", file=sys.stderr)
        return
        
    # Write structured JSON
    results_json = {
        "slurm_runs": slurm_data,
        "efficiency_benchmarks": eff_data
    }
    json_path = os.path.join(dir_path, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=4)
    print(f"Successfully saved structured results to: {json_path}")
    
    # Write Markdown summary
    markdown_content = generate_markdown(slurm_data, eff_data)
    md_path = os.path.join(dir_path, "results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    print(f"Successfully generated summary report at: {md_path}")

if __name__ == "__main__":
    main()
