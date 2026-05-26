from itertools import product

import torch
import triton
from sgl_kernel.flash_attn import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    make_cu_seqlens_block_q,
)


FORCE_CHUNKPREFILL_NUM_SPLITS = -2


def flash_attn_baseline(
    q,
    k_cache,
    v_cache,
    causal,
    window_size,
    softmax_scale,
    sinks,
    cache_seqlens,
    page_table,
    cu_seqlens_q,
    cu_seqlens_block_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    num_splits,
):
    """Baseline Flash Attention implementation"""
    if page_table is not None:
        out, lse, *rest = flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            causal=causal,
            sinks=sinks,
            window_size=window_size,
            softmax_scale=softmax_scale,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_block_q=cu_seqlens_block_q,
            max_seqlen_q=max_seqlen_q,
            num_splits=num_splits,
            return_softmax_lse=True,
        )
        return out, lse
    else:
        out, lse, *rest = flash_attn_varlen_func(
            q,
            k_cache,
            v_cache,
            causal=causal,
            sinks=sinks,
            window_size=window_size,
            softmax_scale=softmax_scale,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            num_splits=num_splits,
            return_softmax_lse=True,
        )
        return out, lse


def make_mixed_q_lengths(batch_size, prefill_q_seq_length, device):
    decode_batch_size = batch_size // 2
    prefill_batch_size = batch_size - decode_batch_size
    q_lengths = torch.empty(batch_size, device=device, dtype=torch.int32)
    q_lengths[:decode_batch_size] = 1
    q_lengths[decode_batch_size:] = prefill_q_seq_length
    cu_seqlens_q = torch.cat(
        (
            torch.zeros(1, device=device, dtype=torch.int32),
            torch.cumsum(q_lengths, dim=0, dtype=torch.int32),
        )
    )
    return q_lengths, cu_seqlens_q, decode_batch_size, prefill_batch_size


def get_effective_attention_pairs(
    q_seq_length, kv_seq_length, causal, window_size=(-1, -1)
):
    diagonal_offset = kv_seq_length - q_seq_length
    window_size_left, window_size_right = window_size
    if causal:
        window_size_right = 0

    effective_pairs = 0
    for query_idx in range(q_seq_length):
        visible_kv_start = 0
        if window_size_left >= 0:
            visible_kv_start = max(0, query_idx + diagonal_offset - window_size_left)

        visible_kv_end = kv_seq_length - 1
        if window_size_right >= 0:
            visible_kv_end = min(
                kv_seq_length - 1, query_idx + diagonal_offset + window_size_right
            )

        visible_kv = max(0, visible_kv_end - visible_kv_start + 1)
        effective_pairs += max(0, visible_kv)
    return effective_pairs


# Benchmark configurations
causal = [False]
batch_size = [32]
q_seq_length_range = [4096]
head_dim = [128]
num_heads_q = [16]
num_heads_kv = [4]
kv_seq_length_range = [4096]
page_size_range = [128]
configs = list(
    filter(
        lambda cfg: (cfg[4] % cfg[5] == 0) and (cfg[6] >= cfg[7]),
        product(
            causal,
            batch_size,
            q_seq_length_range,
            head_dim,
            num_heads_q,
            num_heads_kv,
            kv_seq_length_range,
            page_size_range,
        ),
    )
)
all_results = []


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=[
            "causal",
            "batch_size",
            "q_seq_length",
            "head_dim",
            "num_heads_q",
            "num_heads_kv",
            "kv_seq_length",
            "page_size",
        ],
        x_vals=[list(c) for c in configs],
        line_arg="provider",
        line_vals=["prefill", "chunkprefill"],
        line_names=["Prefill", "ChunkPrefill"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="flash-attention-performance",
        args={},
    )
)
def benchmark(
    causal,
    batch_size,
    q_seq_length,
    head_dim,
    num_heads_q,
    num_heads_kv,
    kv_seq_length,
    page_size,
    provider,
):
    dtype = torch.bfloat16
    device = torch.device("xpu")
    q_lengths, cu_seqlens_q, decode_batch_size, prefill_batch_size = (
        make_mixed_q_lengths(batch_size, q_seq_length, device)
    )
    total_q = int(cu_seqlens_q[-1].item())

    # Create input tensors
    q = torch.randn(
        (total_q, num_heads_q, head_dim), device=device, dtype=dtype
    )
    if page_size > 0:
        num_pages = (batch_size * kv_seq_length + page_size - 1) // page_size
        k_cache = torch.randn(
            (num_pages, page_size, num_heads_kv, head_dim), device=device, dtype=dtype
        )
        v_cache = torch.randn(
            (num_pages, page_size, num_heads_kv, head_dim), device=device, dtype=dtype
        )
        page_table = (
            torch.randperm(num_pages, device=device, dtype=torch.int32)
            .reshape(batch_size, -1)
            .contiguous()
        )
    else:
        k_cache = torch.randn(
            (batch_size * kv_seq_length, num_heads_kv, head_dim),
            device=device,
            dtype=dtype,
        )
        v_cache = torch.randn(
            (batch_size * kv_seq_length, num_heads_kv, head_dim),
            device=device,
            dtype=dtype,
        )
        num_pages = 0
        page_table = None

    cache_seqlens = (
        torch.ones(batch_size, device=device, dtype=torch.int32) * kv_seq_length
    )
    cu_seqlens_block_q = make_cu_seqlens_block_q(cu_seqlens_q, head_dim)
    cu_seqlens_k = torch.arange(
        0,
        (batch_size + 1) * kv_seq_length,
        step=kv_seq_length,
        device=device,
        dtype=torch.int32,
    )
    max_seqlen_q = q_seq_length
    max_seqlen_k = kv_seq_length
    sinks = None

    softmax_scale = 1.0 / (head_dim**0.5)

    quantiles = [0.5, 0.2, 0.8]

    window_size = (-1, -1)
    if provider == "prefill":
        num_splits = 0
    elif provider == "chunkprefill":
        num_splits = FORCE_CHUNKPREFILL_NUM_SPLITS
    else:
        raise ValueError(f"Unknown provider: {provider}")

    if provider in ("prefill", "chunkprefill"):
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: flash_attn_baseline(
                q,
                k_cache,
                v_cache,
                causal=causal,
                window_size=window_size,
                softmax_scale=softmax_scale,
                sinks=sinks,
                cache_seqlens=cache_seqlens,
                page_table=page_table,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_block_q=cu_seqlens_block_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                num_splits=num_splits,
            ),
            quantiles=quantiles,
        )

    effective_attention_pairs = (
        decode_batch_size
        * get_effective_attention_pairs(
            q_seq_length=1,
            kv_seq_length=kv_seq_length,
            causal=causal,
            window_size=(-1, -1),
        )
        + prefill_batch_size
        * get_effective_attention_pairs(
            q_seq_length=q_seq_length,
            kv_seq_length=kv_seq_length,
            causal=causal,
            window_size=(-1, -1),
        )
    )
    total_attention_pairs = int(q_lengths.sum().item()) * kv_seq_length
    effective_attention_ratio = (
        effective_attention_pairs / total_attention_pairs
        if total_attention_pairs > 0
        else 0.0
    )

    flops_qk = num_heads_q * effective_attention_pairs * head_dim * 2
    flops_pv = num_heads_q * effective_attention_pairs * head_dim * 2
    tflops = (flops_qk + flops_pv) * 1e-12 / (ms * 1e-3)
    memory_qk = (
        q.element_size() * total_q * num_heads_q * head_dim
        + k_cache.element_size()
        * num_heads_kv
        * batch_size
        * kv_seq_length
        * head_dim
        * effective_attention_ratio
    )
    memory_pv = (
        v_cache.element_size()
        * num_heads_kv
        * batch_size
        * kv_seq_length
        * head_dim
        * effective_attention_ratio
        + q.element_size() * total_q * num_heads_q * head_dim
    )
    bandwidth = (memory_qk + memory_pv) * 1e-9 / (ms * 1e-3)
    all_results.append(
        {
            "batch": batch_size,
            "decode_batch": decode_batch_size,
            "prefill_batch": prefill_batch_size,
            "decode_q_seq_length": 1,
            "q_seq_length": q_seq_length,
            "total_q": total_q,
            "kv_seq_length": kv_seq_length,
            "num_heads_q": num_heads_q,
            "num_heads_kv": num_heads_kv,
            "head_dim": head_dim,
            "causal": causal,
            "local": False,
            "window_size_left": window_size[0],
            "window_size_right": window_size[1],
            "effective_attention_ratio": effective_attention_ratio,
            "use_sinks": False,
            "num_splits": num_splits,
            "page_size": page_size,
            "provider": provider,
            "tflops": tflops,
            "bandwidth": bandwidth,
            "ms": ms,
        }
    )
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    benchmark.run(print_data=False)
    print("Benchmark finished!")

    import pandas as pd

    df = pd.DataFrame(all_results)
    print(df.to_markdown())
