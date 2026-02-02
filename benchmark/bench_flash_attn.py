from itertools import product

import torch
import triton
from sgl_kernel.flash_attn import flash_attn_with_kvcache


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
    max_seqlen_q,
):
    """Baseline Flash Attention implementation"""
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
        max_seqlen_q=max_seqlen_q,
        return_softmax_lse=True,
    )
    return out, lse


# Benchmark configurations
causal = [True, False]
local = [True, False]
use_sinks = [True, False]
batch_size = [16, 32]
q_seq_length_range = [1, 128]
head_dim = [64, 128]
num_heads_q = [16]
num_heads_kv = [2, 4, 8]
kv_seq_length_range = [1024, 4096, 16384]
page_size_range = [128]
configs = list(
    filter(
        lambda cfg: not (cfg[0] and cfg[1])
        and (cfg[4] != 1 or (not cfg[0] and not cfg[1] and not cfg[2])) # 
        and (cfg[6] % cfg[7] == 0),
        product(
            causal,
            local,
            use_sinks,
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
            "local",
            "use_sinks",
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
        line_vals=["flash_attn"],
        line_names=["Flash Attention"],
        styles=[("blue", "-")],
        ylabel="us",
        plot_name="flash-attention-performance",
        args={},
    )
)
def benchmark(
    causal,
    local,
    use_sinks,
    batch_size,
    head_dim,
    num_heads_q,
    num_heads_kv,
    q_seq_length,
    kv_seq_length,
    page_size,
    provider,
):
    dtype = torch.bfloat16
    device = torch.device("xpu")
    # Create input tensors
    q = torch.randn(
        (batch_size * q_seq_length, num_heads_q, head_dim), device=device, dtype=dtype
    )
    num_pages = (batch_size * kv_seq_length + page_size - 1) // page_size
    k_cache = torch.randn(
        (num_pages, page_size, num_heads_kv, head_dim), device=device, dtype=dtype
    )
    v_cache = torch.randn(
        (num_pages, page_size, num_heads_kv, head_dim), device=device, dtype=dtype
    )
    cache_seqlens = (
        torch.ones(batch_size, device=device, dtype=torch.int32) * kv_seq_length
    )
    page_table = (
        torch.randperm(num_pages, device=device, dtype=torch.int32)
        .reshape(batch_size, -1)
        .contiguous()
    )
    cu_seqlens_q = torch.arange(
        0,
        (batch_size + 1) * q_seq_length,
        step=q_seq_length,
        device=device,
        dtype=torch.int32,
    )
    max_seqlen_q = q_seq_length
    window_size = (-1, -1) if not local else torch.randint(0, kv_seq_length, (2,))

    sinks = torch.randn(num_heads_q, device=device, dtype=dtype) if use_sinks else None

    softmax_scale = 1.0 / (head_dim**0.5)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "flash_attn":
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
                max_seqlen_q=max_seqlen_q,
            ),
            quantiles=quantiles,
        )

    flops_qk = batch_size * num_heads_q * q_seq_length * kv_seq_length * head_dim * 2
    flops_pv = batch_size * num_heads_q * q_seq_length * head_dim * kv_seq_length * 2
    tflops = (flops_qk + flops_pv) * 1e-12 / (ms * 1e-3)
    memory_qk = batch_size * (
        q.element_size() * num_heads_q * q_seq_length * head_dim
        + k_cache.element_size() * num_heads_kv * kv_seq_length * head_dim
    )
    memory_pv = (
        v_cache.element_size() * batch_size * num_heads_kv * kv_seq_length * head_dim
        + q.element_size() * batch_size * num_heads_q * q_seq_length * head_dim
    )
    bandwidth = (memory_qk + memory_pv) * 1e-9 / (ms * 1e-3)
    all_results.append(
        {
            "batch": batch_size,
            "q_seq_length": q_seq_length,
            "kv_seq_length": kv_seq_length,
            "num_heads_q": num_heads_q,
            "num_heads_kv": num_heads_kv,
            "head_dim": head_dim,
            "causal": causal,
            "local": local,
            "use_sinks": use_sinks,
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
    import pandas as pd

    df = pd.DataFrame(all_results)
    print(df.to_markdown())
    print("Benchmark finished!")
