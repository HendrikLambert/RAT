"""Portable fallback for the inter-chunk attention used by RAT.

torch.nn.attention.flex_attention currently has no MPS backend and only a stub
CPU path, which prevents the RAT layer from running on a MacBook. This module
provides a plain-PyTorch implementation that returns both the attention output
and the per-row log-sum-exp, matching the (out, lse) tuple that
flex_attention(..., return_lse=True) returns. It is intentionally allocation-
heavy and slow; the CUDA path keeps using flex_attention.
"""
import torch


def block_causal_attention_with_lse(q, k, v, chunk_size, scale):
    # q: (b, h, L, d), k/v: (b, h, C, d) with L = C * chunk_size
    # Mask: query at position i attends to chunk j iff i // chunk_size > j.
    b, h, L, d = q.shape
    C = k.shape[2]
    scores = torch.einsum("bhld,bhcd->bhlc", q, k) * scale
    q_idx = torch.arange(L, device=q.device).view(L, 1)
    kv_idx = torch.arange(C, device=q.device).view(1, C)
    mask = (q_idx // chunk_size) > kv_idx
    scores = scores.masked_fill(~mask, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    # First-chunk queries have no prior chunks; softmax over all -inf is NaN.
    # merge_last_token_naive sees lse=-inf and gives the inter branch weight 0,
    # so as long as `out` is finite the merged result is correct.
    attn = torch.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)
    out = torch.einsum("bhlc,bhcd->bhld", attn, v)
    return out, lse
