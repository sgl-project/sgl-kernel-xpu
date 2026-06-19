import pandas as pd
import torch
import triton
from sgl_kernel import hc_pre_gemm_sqr_sum

# Production mhc_pre GEMM+sqr_sum stage shapes (Design B: K-split partials):
#   A      [M, K]            bf16   (residual.view(M, hc_hidden)); hc_hidden = hc_mult * hidden
#   B      [N, K]            fp32   (fn = [24, 16384] = [N, K])
#   C      [n_splits, M, N]  fp32   (gemm_out_mul partials), C = A @ B^T per K-slice
#   sqr_sum [n_splits, M]     fp32   (gemm_out_sqr_sum partials), sqr_sum[s,m] = partial sum A^2
# N (=hc_mult3) and K (=hc_mult*hidden) are fixed; only the token count M varies.
# n_splits follows the mhc_pre split-k rule (32 for M<=2048).
hc = 4
hc_mult3 = (2 + hc) * hc  # 24 -> N
hidden_size = 4096
K = hc * hidden_size  # 16384 -> K
N = hc_mult3  # 24


def _n_splits_pre(M):
    return 32 if M <= 2048 else 1


configs = [
    # (M, K, N)  -- same token sweep as bench_hc_pre_fuse.py (incl. ragged M)
    (128, K, N),
    (512, K, N),
    (896, K, N),
    (1021, K, N),
    (1024, K, N),
    (1034, K, N),
    (1038, K, N),
    (1518, K, N),
    (2048, K, N),
    (16, K, N),
    (48, K, N),
]

all_results = []


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "K", "N"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["sgl_kernel"],
        line_names=["sgl_kernel (tf32)"],
        styles=[("green", "-")],
        ylabel="Time (ms)",
        plot_name="gemm-sqr_sum-performance",
        args={},
    )
)
def benchmark(M, K, N, provider):
    print(f"benchmark {provider} with M={M} K={K} N={N}")
    torch.manual_seed(42)
    torch.set_default_device("xpu")

    n_splits = _n_splits_pre(M)

    A = torch.randn(M, K, dtype=torch.bfloat16, device="xpu")
    B = torch.randn(N, K, dtype=torch.float32, device="xpu")

    C = torch.empty(n_splits, M, N, dtype=torch.float32, device="xpu")
    sqr_sum = torch.empty(n_splits, M, dtype=torch.float32, device="xpu")

    run = lambda: hc_pre_gemm_sqr_sum(C, sqr_sum, A, B)

    # Warmup
    for _ in range(100):
        run()
    torch.xpu.synchronize()

    quantiles = [0.5, 0.25, 0.75]
    ms, _, _ = triton.testing.do_bench(run, quantiles=quantiles, return_mode="median")

    torch.xpu.empty_cache()

    # GEMM (2*M*N*K) + row sqr-sum (2*M*K: square + accumulate).
    flops = 2.0 * M * N * K + 2.0 * M * K
    tflops = flops / (ms / 1e3) / 1e12

    read_bytes = M * K * 2 + N * K * 4  # A (bf16) + B (fp32)
    write_bytes = n_splits * M * N * 4 + n_splits * M * 4  # C + sqr_sum partials (fp32)
    total_bytes = read_bytes + write_bytes
    bandwidth_gb_s = total_bytes / (ms / 1e3) / 1e9

    all_results.append(
        {
            "M": M,
            "K": K,
            "N": N,
            "n_splits": n_splits,
            "provider": provider,
            "ms": ms,
            "Mtok_per_sec": M / (ms / 1e3) / 1e6,
            "TFLOP_per_sec": tflops,
            "bandwidth_gb_s": bandwidth_gb_s,
        }
    )
    return ms


if __name__ == "__main__":
    benchmark.run(print_data=False)
    print("Benchmark finished!")

    df = pd.DataFrame(all_results)
    print("\n" + "=" * 80)
    print("HC_PRE_GEMM_SQUARE_SUM BENCHMARK RESULTS")
    print("=" * 80)
    print(df.to_markdown(index=False))
    print("\n")

    print("Summary Statistics:")
    print(f"  Mean throughput: {df['Mtok_per_sec'].mean():.2f} Mtok/s")
    print(f"  Mean compute:    {df['TFLOP_per_sec'].mean():.3f} TFLOP/s")
    print(f"  Mean bandwidth:  {df['bandwidth_gb_s'].mean():.2f} GB/s")
    print(f"  Best throughput: {df['Mtok_per_sec'].max():.2f} Mtok/s")
