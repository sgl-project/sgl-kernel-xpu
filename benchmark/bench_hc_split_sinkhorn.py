from itertools import product

import pandas as pd
import torch
import triton
import triton.language as tl
from sgl_kernel import hc_split_sinkhorn

batch_size_range = [1, 7, 32, 64, 384, 512]
seq_len_range = [1, 512, 4096, 131072]
sinkhorn_iters_range = [20]
hc = 4
col_size = (2 + hc) * hc  # 24 floats per token

MAX_TOKENS = 1_000_000

configs = [
    (b, s, it)
    for b, s, it in product(batch_size_range, seq_len_range, sinkhorn_iters_range)
    if b * s <= MAX_TOKENS
]

all_results = []


# Triton kernel implementation
@triton.jit
def _hc_split_sinkhorn_triton_kernel(
    mixes_ptr,  # [N, (2 + HC) * HC] float32
    hc_scale_ptr,  # [3]                float32
    hc_base_ptr,  # [(2 + HC) * HC]    float32
    pre_ptr,  # [N, HC]            float32
    post_ptr,  # [N, HC]            float32
    comb_ptr,  # [N, HC, HC]        float32
    N,
    HC: tl.constexpr,
    SINKHORN_ITERS: tl.constexpr,
    EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N:
        return

    HC2: tl.constexpr = HC * HC
    MIX_HC: tl.constexpr = (2 + HC) * HC

    s0 = tl.load(hc_scale_ptr + 0)
    s1 = tl.load(hc_scale_ptr + 1)
    s2 = tl.load(hc_scale_ptr + 2)

    h = tl.arange(0, HC)

    pre_mix = tl.load(mixes_ptr + pid * MIX_HC + h)
    pre_base = tl.load(hc_base_ptr + h)
    pre = tl.sigmoid(pre_mix * s0 + pre_base) + EPS
    tl.store(pre_ptr + pid * HC + h, pre)

    post_mix = tl.load(mixes_ptr + pid * MIX_HC + HC + h)
    post_base = tl.load(hc_base_ptr + HC + h)
    post = 2.0 * tl.sigmoid(post_mix * s1 + post_base)
    tl.store(post_ptr + pid * HC + h, post)

    j = tl.arange(0, HC)[:, None]
    k = tl.arange(0, HC)[None, :]
    idx = j * HC + k
    comb_mix = tl.load(mixes_ptr + pid * MIX_HC + 2 * HC + idx)
    comb_base = tl.load(hc_base_ptr + 2 * HC + idx)
    comb = comb_mix * s2 + comb_base

    row_max = tl.max(comb, axis=1)[:, None]
    comb = tl.exp(comb - row_max)
    row_sum = tl.sum(comb, axis=1)[:, None]
    comb = comb / row_sum + EPS

    col_sum = tl.sum(comb, axis=0)[None, :]
    comb = comb / (col_sum + EPS)

    for _ in tl.static_range(SINKHORN_ITERS - 1):
        row_sum = tl.sum(comb, axis=1)[:, None]
        comb = comb / (row_sum + EPS)
        col_sum = tl.sum(comb, axis=0)[None, :]
        comb = comb / (col_sum + EPS)

    tl.store(comb_ptr + pid * HC2 + idx, comb)


def _hc_split_sinkhorn_triton(
    mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6
):
    """Triton wrapper for HCSplitSinkhorn"""
    assert mixes.dtype == torch.float32, "mixes must be float32"
    assert hc_scale.dtype == torch.float32 and hc_base.dtype == torch.float32
    assert hc_mult & (hc_mult - 1) == 0, "hc_mult must be a power of two"
    assert sinkhorn_iters >= 1

    b, s, last = mixes.size()
    assert last == (2 + hc_mult) * hc_mult

    n = b * s
    pre = mixes.new_empty(b, s, hc_mult)
    post = mixes.new_empty(b, s, hc_mult)
    comb = mixes.new_empty(b, s, hc_mult, hc_mult)

    if n == 0:
        return pre, post, comb

    mixes_flat = mixes.reshape(n, (2 + hc_mult) * hc_mult).contiguous()
    pre_flat = pre.view(n, hc_mult)
    post_flat = post.view(n, hc_mult)
    comb_flat = comb.view(n, hc_mult, hc_mult)
    hc_base_c = hc_base.contiguous()
    hc_scale_c = hc_scale.contiguous()

    _hc_split_sinkhorn_triton_kernel[(n,)](
        mixes_flat,
        hc_scale_c,
        hc_base_c,
        pre_flat,
        post_flat,
        comb_flat,
        n,
        HC=hc_mult,
        SINKHORN_ITERS=sinkhorn_iters,
        EPS=eps,
        num_warps=1,
    )

    return pre, post, comb


def sglang_hc_split_sinkhorn(
    mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6
):
    """SGL Kernel wrapper for HCSplitSinkhorn"""
    return hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "seq_len", "sinkhorn_iters"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["triton", "sglang"],
        line_names=["Triton", "SGL Kernel"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="ms",
        plot_name="hc-split-sinkhorn-performance",
        args={},
    )
)
def benchmark(batch_size, seq_len, sinkhorn_iters, provider):
    device = torch.device("xpu")

    torch.manual_seed(42)
    torch.xpu.manual_seed_all(42)

    mixes = torch.randn(
        batch_size, seq_len, col_size, dtype=torch.float32, device=device
    )
    hc_scale = torch.rand(3, dtype=torch.float32, device=device) * 0.5 + 0.5
    hc_base = torch.randn(col_size, dtype=torch.float32, device=device) * 0.1

    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        fn = lambda: _hc_split_sinkhorn_triton(
            mixes, hc_scale, hc_base, hc, sinkhorn_iters
        )
    elif provider == "sglang":
        fn = lambda: sglang_hc_split_sinkhorn(
            mixes, hc_scale, hc_base, hc, sinkhorn_iters
        )

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    T = batch_size * seq_len
    read_numel = T * col_size
    write_numel = T * (hc + hc + hc * hc)
    total_bytes = (read_numel + write_numel) * 4
    bandwidth_gbs = total_bytes / (ms / 1e3) / 1e9

    all_results.append(
        {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "T": T,
            "sinkhorn_iters": sinkhorn_iters,
            "provider": provider,
            "time_ms": ms,
            "Mtok_per_sec": T / (ms / 1e3) / 1e6,
            "bandwidth_gbs": bandwidth_gbs,
        }
    )

    return ms, max_ms, min_ms


if __name__ == "__main__":

    benchmark.run(print_data=True)

    # Print results
    print("\n" + "=" * 80)
    print("Performance Results")
    print("=" * 80)

    df = pd.DataFrame(all_results)
    df["time_ms"] = df["time_ms"].round(7)
    df["Mtok_per_sec"] = df["Mtok_per_sec"].round(2)
    df["bandwidth_gbs"] = df["bandwidth_gbs"].round(4)

    print(df.to_markdown(index=False))

    # Print summary statistics per provider
    print("\n" + "=" * 80)
    print("Summary Statistics by Provider")
    print("=" * 80)
    summary = df.groupby("provider").agg(
        {
            "Mtok_per_sec": ["mean", "min", "max"],
            "bandwidth_gbs": ["mean", "min", "max"],
            "time_ms": ["mean", "min", "max"],
        }
    )
    print(summary.to_markdown())
