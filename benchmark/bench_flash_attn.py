import os
from itertools import product

import torch
import triton
from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache


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
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    k_descale=None,
    v_descale=None,
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
            max_seqlen_q=max_seqlen_q,
            k_descale=k_descale,
            v_descale=v_descale,
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
            return_softmax_lse=True,
        )
        return out, lse


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
causal = [True, False]
local = [True, False]
use_sinks = [True, False]
batch_size = [1, 8, 16]
q_seq_length_range = [1, 128]
head_dim_no_page = [72, 128, 192, 256, 512]
head_dim_paged = [64, 128, 256, 512]
num_heads_q = [16]
num_heads_kv = [4, 8]
kv_seq_length_range = [4096]
page_size_range = [0, 128]
# KV cache element type: "bf16" (default) or fp8. FP8 has two formats,
# e5m2 and e4m3; both are exercised ("fp8_e4m3" / "fp8_e5m2"), dequantized
# in-kernel via per-tensor k_descale / v_descale. fp8 only runs on the paged
# path.
kv_dtype_range = ["bf16", "fp8_e4m3", "fp8_e5m2"]
configs = list(
    filter(
        lambda cfg: (
            # Condition 1: causal and local cannot both be True
            not (cfg[0] and cfg[1])
            # Condition 2: when q_seq_length=1, causal must be False
            and (cfg[4] != 1 or not cfg[0])
            # Condition 3: num_heads_q must be a multiple of num_heads_kv (GQA requirement)
            and (cfg[6] % cfg[7] == 0)
            # Condition 4: kv_seq_length >= page_size
            and (cfg[8] >= cfg[9])
            # Condition 5: no_page mode (page_size=0) does not support sink logits
            and (cfg[9] != 0 or not cfg[2])
            # Condition 6: sink is only supported for head_size == 64
            and (not cfg[2] or cfg[5] == 64)
            # Condition 7: fp8 KV cache requires the paged path and is exercised
            # without sinks / local masking (matches the supported fp8 path)
            and (cfg[10] == "bf16" or (cfg[9] != 0 and not cfg[2] and not cfg[1]))
        ),
        [
            cfg
            for page_size in page_size_range
            for cfg in product(
                causal,
                local,
                use_sinks,
                batch_size,
                q_seq_length_range,
                head_dim_no_page if page_size == 0 else head_dim_paged,
                num_heads_q,
                num_heads_kv,
                kv_seq_length_range,
                [page_size],
                kv_dtype_range,
            )
        ],
    )
)

# Prefill-only subset (env FMHA_BENCH_PREFILL_ONLY): the persistent scheduler
# only affects prefill (q_seq_length > 1), so trim to prefill configs (drop
# decode q=1, local/sink/fp8, and rare head dims) for a fast A/B/C comparison.
if os.environ.get("FMHA_BENCH_PREFILL_ONLY", "0") not in ("0", "", "false", "False"):
    _prefill_head_dims = {128, 256}
    configs = [
        c
        for c in configs
        if c[4] > 1  # q_seq_length > 1
        and not c[1]  # local == False
        and not c[2]  # use_sinks == False
        and c[10] == "bf16"  # kv_dtype
        and c[5] in _prefill_head_dims  # head_dim
        and c[3] in (1, 8)  # batch_size (skip the heavy 16)
    ]

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
            "kv_dtype",
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
    kv_dtype,
    provider,
):
    dtype = torch.bfloat16
    device = torch.device("xpu")
    # fp8 KV cache: store K/V as e4m3 or e5m2 and dequantize in-kernel via
    # per-tensor k_descale / v_descale. Q/O stay bf16. Only valid on the paged
    # path.
    is_fp8 = kv_dtype.startswith("fp8")
    fp8_dtype = torch.float8_e5m2 if kv_dtype == "fp8_e5m2" else torch.float8_e4m3fn
    fp8_max = 57344.0 if kv_dtype == "fp8_e5m2" else 448.0
    k_descale = None
    v_descale = None
    # Create input tensors
    q = torch.randn(
        (batch_size * q_seq_length, num_heads_q, head_dim), device=device, dtype=dtype
    )
    if page_size > 0:
        num_pages = (batch_size * kv_seq_length + page_size - 1) // page_size
        k_cache = torch.randn(
            (num_pages, page_size, num_heads_kv, head_dim), device=device, dtype=dtype
        )
        v_cache = torch.randn(
            (num_pages, page_size, num_heads_kv, head_dim), device=device, dtype=dtype
        )
        if is_fp8:
            k_descale_val = k_cache.abs().max().item() / fp8_max
            v_descale_val = v_cache.abs().max().item() / fp8_max
            k_cache = (k_cache / k_descale_val).to(fp8_dtype)
            v_cache = (v_cache / v_descale_val).to(fp8_dtype)
            k_descale = torch.tensor(
                k_descale_val, dtype=torch.float32, device=device
            ).expand(batch_size, num_heads_kv)
            v_descale = torch.tensor(
                v_descale_val, dtype=torch.float32, device=device
            ).expand(batch_size, num_heads_kv)
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
    cu_seqlens_q = torch.arange(
        0,
        (batch_size + 1) * q_seq_length,
        step=q_seq_length,
        device=device,
        dtype=torch.int32,
    )
    cu_seqlens_k = torch.arange(
        0,
        (batch_size + 1) * kv_seq_length,
        step=kv_seq_length,
        device=device,
        dtype=torch.int32,
    )
    max_seqlen_q = q_seq_length
    max_seqlen_k = kv_seq_length
    if not local:
        window_size = (-1, -1)
    else:
        window_size = tuple(
            int(value) for value in torch.randint(0, kv_seq_length, (2,)).tolist()
        )

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
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                k_descale=k_descale,
                v_descale=v_descale,
            ),
            quantiles=quantiles,
        )

    total_attention_pairs = q_seq_length * kv_seq_length
    effective_attention_pairs = get_effective_attention_pairs(
        q_seq_length=q_seq_length,
        kv_seq_length=kv_seq_length,
        causal=causal,
        window_size=window_size,
    )
    effective_attention_ratio = (
        effective_attention_pairs / total_attention_pairs
        if total_attention_pairs > 0
        else 0.0
    )

    flops_qk = batch_size * num_heads_q * effective_attention_pairs * head_dim * 2
    flops_pv = batch_size * num_heads_q * effective_attention_pairs * head_dim * 2
    tflops = (flops_qk + flops_pv) * 1e-12 / (ms * 1e-3)
    memory_qk = batch_size * (
        q.element_size() * num_heads_q * q_seq_length * head_dim
        + k_cache.element_size()
        * num_heads_kv
        * kv_seq_length
        * head_dim
        * effective_attention_ratio
    )
    memory_pv = (
        v_cache.element_size()
        * batch_size
        * num_heads_kv
        * kv_seq_length
        * head_dim
        * effective_attention_ratio
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
            "window_size_left": window_size[0],
            "window_size_right": window_size[1],
            "effective_attention_ratio": effective_attention_ratio,
            "use_sinks": use_sinks,
            "page_size": page_size,
            "kv_dtype": kv_dtype,
            "provider": provider,
            "tflops": tflops,
            "bandwidth": bandwidth,
            "ms": ms,
        }
    )
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    import argparse
    import json
    import os
    import subprocess
    import sys
    import tempfile

    parser = argparse.ArgumentParser(description="Flash attention benchmark")
    parser.add_argument(
        "--out",
        default=None,
        help="Dump per-config results to this JSON path (in addition to printing).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run default / persistent / persistent+atomic prefill schedulers in "
        "separate subprocesses and print a merged A/B/C comparison table. Each "
        "mode needs its own process because the scheduler reads its env toggle "
        "once at first use.",
    )
    parser.add_argument(
        "--prefill-only",
        action="store_true",
        help="Restrict to prefill configs (q_seq_length > 1, bf16, no local/sink) "
        "for a fast comparison; the persistent scheduler only affects prefill.",
    )
    cli_args = parser.parse_args()

    # A/B/C driver: the persistent-scheduler env toggles are cached per process,
    # so each mode is benchmarked in a fresh subprocess and the ms results are
    # merged on the config key.
    if cli_args.compare:
        import numpy as np
        import pandas as pd

        modes = [
            ("default", {}),
            ("persistent", {"SGL_KERNEL_FMHA_PREFILL_PERSISTENT": "1"}),
            (
                "persistent_atomic",
                {
                    "SGL_KERNEL_FMHA_PREFILL_PERSISTENT": "1",
                    "SGL_KERNEL_FMHA_PREFILL_PERSISTENT_ATOMIC": "1",
                },
            ),
        ]
        tmpdir = tempfile.mkdtemp(prefix="fmha_bench_")
        mode_files = {}
        for name, extra_env in modes:
            out_path = os.path.join(tmpdir, f"{name}.json")
            env = dict(os.environ)
            env.pop("SGL_KERNEL_FMHA_PREFILL_PERSISTENT", None)
            env.pop("SGL_KERNEL_FMHA_PREFILL_PERSISTENT_ATOMIC", None)
            env.update(extra_env)
            if cli_args.prefill_only:
                env["FMHA_BENCH_PREFILL_ONLY"] = "1"
            print(f"[compare] running mode={name} ...", flush=True)
            subprocess.run(
                [sys.executable, __file__, "--out", out_path], env=env, check=True
            )
            mode_files[name] = out_path

        key_cols = [
            "batch",
            "q_seq_length",
            "kv_seq_length",
            "num_heads_q",
            "num_heads_kv",
            "head_dim",
            "causal",
            "local",
            "use_sinks",
            "page_size",
            "kv_dtype",
        ]
        merged = None
        for name, _ in modes:
            with open(mode_files[name]) as f:
                d = pd.DataFrame(json.load(f))
            d = d[key_cols + ["ms"]].rename(columns={"ms": f"ms_{name}"})
            merged = d if merged is None else merged.merge(d, on=key_cols, how="outer")

        for name, _ in modes[1:]:
            merged[f"speedup_{name}"] = merged["ms_default"] / merged[f"ms_{name}"]

        pd.set_option("display.width", 240)
        print(merged.to_markdown(index=False))

        for name, _ in modes[1:]:
            s = merged[f"speedup_{name}"].replace([np.inf, -np.inf], np.nan).dropna()
            if len(s):
                geo = float(np.exp(np.log(s).mean()))
                print(
                    f"[summary] {name}: geomean speedup vs default = {geo:.4f}, "
                    f"min = {s.min():.4f}, max = {s.max():.4f}"
                )
    else:
        benchmark.run(print_data=False)
        print("Benchmark finished!")

        import pandas as pd

        df = pd.DataFrame(all_results)
        if cli_args.out:
            df.to_json(cli_args.out, orient="records")
        print(df.to_markdown())
