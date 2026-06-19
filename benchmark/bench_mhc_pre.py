"""Benchmark for the full mhc_pre pipeline: hc_pre_gemm_sqr_sum -> hc_pre_big_fuse.

mhc_pre replaces the two non-prenorm TileLang paths. It runs two kernels
back-to-back: the CUTLASS hc_pre_gemm_sqr_sum (residual @ fnᵀ + row sqr-sum, written
as K-split partials) and hc_pre_big_fuse (reduce + RMS/Sinkhorn/mix, optional
RMSNorm). This times the whole thing end-to-end at the production shapes.
"""

import pandas as pd
import torch
import triton
from sgl_kernel import mhc_pre

# Production shapes (from the real mhc_pre call log): D=4096, hc_mult=4,
# hc_hidden = hc_mult * D = 16384, fn = [24, 16384]. Token counts span the
# ragged set; all <= 2048 so all take the split-k path (n_splits_pre=32).
configs = [
    # (b_s, seq_len, hidden_size)
    (16, 1, 4096),
    (48, 1, 4096),
    (128, 1, 4096),
    (512, 1, 4096),
    (896, 1, 4096),
    (1021, 1, 4096),
    (1024, 1, 4096),
    (1034, 1, 4096),
    (1038, 1, 4096),
    (1518, 1, 4096),
    (2048, 1, 4096),
]

sinkhorn_repeat = 20
hc = 4
hc_mult3 = (2 + hc) * hc  # 24

rms_eps = 1e-6
hc_pre_eps = 1e-6
hc_sinkhorn_eps = 1e-6
hc_post_mult_value = 2.0
norm_eps = 1e-6

all_results = []


def _n_splits_pre(T):
    return 32 if T <= 2048 else 1


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["b_s", "seq_len", "hidden_size"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["without_norm", "with_norm"],
        line_names=["Without Norm", "With Norm"],
        styles=[("green", "-"), ("blue", "--")],
        ylabel="Time (ms)",
        plot_name="mhc-pre-performance",
        args={},
    )
)
def benchmark(b_s, seq_len, hidden_size, provider):
    print(
        f"benchmark {provider} with b_s={b_s} seq_len={seq_len} "
        f"hidden_size={hidden_size}"
    )
    torch.set_default_device("xpu")
    torch.xpu.manual_seed_all(42)

    T = b_s * seq_len
    hc_hidden = hc * hidden_size
    use_norm = provider == "with_norm"

    # Inputs (match the production mhc_pre call: bf16 residual, fp32 fn).
    residual = torch.randn(T, hc, hidden_size, dtype=torch.bfloat16, device="xpu")
    fn = torch.randn(hc_mult3, hc_hidden, dtype=torch.float32, device="xpu")
    hc_scale = torch.rand(3, dtype=torch.float32, device="xpu") * 0.5 + 0.5
    hc_base = torch.randn(hc_mult3, dtype=torch.float32, device="xpu") * 0.1
    # RMSNorm weights cluster near 1 in practice.
    norm_weight = (
        torch.randn(hidden_size, dtype=torch.float32, device="xpu") * 0.5 + 1.0
    ).to(torch.bfloat16)

    nw = norm_weight if use_norm else None

    def run():
        return mhc_pre(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            hc_sinkhorn_eps=hc_sinkhorn_eps,
            hc_post_mult_value=hc_post_mult_value,
            sinkhorn_repeat=sinkhorn_repeat,
            norm_weight=nw,
            norm_eps=norm_eps,
        )

    # Warmup
    for _ in range(10):
        run()
    torch.xpu.synchronize()

    quantiles = [0.5, 0.25, 0.75]
    ms, _, _ = triton.testing.do_bench(run, quantiles=quantiles, return_mode="median")

    torch.xpu.empty_cache()

    n_splits_pre = _n_splits_pre(T)

    # Memory traffic (the dominant terms; small params elided).
    #   GEMM reads:  residual A (bf16) once widened, fn (fp32)
    #   GEMM writes: gemm_out_mul [n_splits,T,24] fp32, sqr_sum [n_splits,T] fp32
    #   Fuse reads:  those partials + residual (bf16) again (+ norm_weight)
    #   Fuse writes: post_mix, comb_mix (fp32), layer_input (bf16)
    gemm_read = T * hc_hidden * 2 + hc_mult3 * hc_hidden * 4  # residual bf16 + fn fp32
    gemm_write = n_splits_pre * T * hc_mult3 * 4 + n_splits_pre * T * 4
    fuse_read = (
        n_splits_pre * T * hc_mult3 * 4
        + n_splits_pre * T * 4
        + T * hc * hidden_size * 2  # residual bf16
    )
    if use_norm:
        fuse_read += hidden_size * 2
    fuse_write = T * hc * 4 + T * hc * hc * 4 + T * hidden_size * 2
    total_bytes = gemm_read + gemm_write + fuse_read + fuse_write
    bandwidth_gb_s = total_bytes / (ms / 1e3) / 1e9

    # GEMM compute: C = A @ Bᵀ is M*N*K MACs; the sqr-sum is a second M*N*K
    # GEMM. 2 flops/MAC, x2 for the two GEMMs.
    gemm_flops = 2 * 2 * T * hc_mult3 * hc_hidden
    tflops = gemm_flops / (ms / 1e3) / 1e12

    all_results.append(
        {
            "b_s": b_s,
            "seq_len": seq_len,
            "T": T,
            "hidden_size": hidden_size,
            "n_splits_pre": n_splits_pre,
            "with_norm": use_norm,
            "ms": ms,
            "Mtok_per_sec": T / (ms / 1e3) / 1e6,
            "gemm_TFLOPs": tflops,
            "bandwidth_gb_s": bandwidth_gb_s,
        }
    )
    return ms


if __name__ == "__main__":
    benchmark.run(print_data=False)
    print("Benchmark finished!")

    df = pd.DataFrame(all_results)
    print("\n" + "=" * 80)
    print("MHC_PRE BENCHMARK RESULTS")
    print("=" * 80)
    print(df.to_markdown(index=False))
    print("\n")

    print("Summary Statistics:")
    for with_norm_val in [False, True]:
        df_variant = df[df["with_norm"] == with_norm_val]
        variant_name = "With Norm" if with_norm_val else "Without Norm"
        print(f"\n{variant_name}:")
        print(f"  Mean throughput: {df_variant['Mtok_per_sec'].mean():.2f} Mtok/s")
        print(f"  Mean GEMM:       {df_variant['gemm_TFLOPs'].mean():.2f} TFLOP/s")
        print(f"  Mean bandwidth:  {df_variant['bandwidth_gb_s'].mean():.2f} GB/s")
        print(f"  Best throughput: {df_variant['Mtok_per_sec'].max():.2f} Mtok/s")
