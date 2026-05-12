"""Tokenize PG19 with the GPT2 tokenizer into uint16 .bin files for LMOrderedDataloader.

Paper's 200M preliminary studies use PG19 with the GPT2 tokenizer and reach 15B tokens
by iterating the train split 5x; that 5x is applied at training time via trainer.max_epoch=5,
so this script tokenizes each split exactly once.
"""
from argparse import ArgumentParser
import os

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import GPT2TokenizerFast


def tokenize_split(dataset, enc, eos_id, num_proc):
    def fn(example):
        ids = enc(example["text"], truncation=False, padding=False, add_special_tokens=False)["input_ids"]
        ids.append(eos_id)
        return {"ids": ids, "len": len(ids)}

    return dataset.map(
        fn,
        remove_columns=dataset.column_names,
        desc="tokenizing",
        num_proc=num_proc,
    )


def save_to_npmemmap(dset, filename, total_batches=1024):
    arr_len = int(np.sum(dset["len"], dtype=np.uint64))
    arr = np.memmap(filename, dtype=np.uint16, mode="w+", shape=(arr_len,))
    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format("numpy")
        batch_ids = np.concatenate(batch["ids"])
        arr[idx : idx + len(batch_ids)] = batch_ids
        idx += len(batch_ids)
    arr.flush()
    print(f"wrote {arr_len} tokens to {filename}")


def parse_args():
    p = ArgumentParser()
    p.add_argument("--out_dir", required=True, help="output directory for .bin files")
    p.add_argument("--dataset", default="deepmind/pg19", help="HuggingFace dataset path or local dir")
    p.add_argument("--num_proc", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    enc = GPT2TokenizerFast.from_pretrained("gpt2")
    eos_id = enc.eos_token_id  # 50256, fits in uint16

    ds = load_dataset(args.dataset)

    for split, out_name in [("train", "gpt2-train.bin"), ("validation", "gpt2-val.bin")]:
        if split not in ds:
            raise KeyError(f"split {split!r} not found in dataset; available: {list(ds.keys())}")
        tok = tokenize_split(ds[split], enc, eos_id, args.num_proc)
        save_to_npmemmap(tok, os.path.join(args.out_dir, out_name))


if __name__ == "__main__":
    main()
