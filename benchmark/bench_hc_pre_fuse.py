import pandas as pd
import torch
import triton
from sgl_kernel import hc_pre_big_fuse

configs = [
    # (batch_size, seq_len, hidden_size, n_splits)
    (16, 1, 4096, 1),  # 16 tokens
    (128, 1, 4096, 1),  # 128 tokens
    (512, 1, 4096, 1),  # 512 tokens
    (1024, 1, 4096, 1),  # 1024 tokens
    (2048, 1, 4096, 1),  # 2048 tokens
]

sinkhorn_iters = 20
hc = 4
hc_mult3 = (2 + hc) * hc  # 24 floats per token

all_results = []


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "seq_len", "hidden_size", "n_splits"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["without_norm", "with_norm"],
        line_names=["Without Norm", "With Norm"],
        styles=[("green", "-"), ("blue", "--")],
        ylabel="Time (ms)",
        plot_name="hc-pre-fuse-performance",
        args={},
    )
)
def benchmark(batch_size, seq_len, hidden_size, n_splits, provider):
    print(
        f"benchmark {provider} with batch_size={batch_size} seq_len={seq_len} "
        f"hidden_size={hidden_size} n_splits={n_splits}"
    )
    torch.set_default_device("xpu")
    torch.xpu.manual_seed_all(42)

    T = batch_size * seq_len

    # Create inputs
    gemm_out_mul = torch.randn(n_splits, T, hc_mult3, dtype=torch.float32, device="xpu")
    gemm_out_sqrsum = (
        torch.rand(n_splits, T, dtype=torch.float32, device="xpu") * 100 + 10
    )
    hc_scale = torch.rand(3, dtype=torch.float32, device="xpu") * 0.5 + 0.5
    hc_base = torch.randn(hc_mult3, dtype=torch.float32, device="xpu") * 0.1
    residual = torch.randn(T, hc, hidden_size, dtype=torch.bfloat16, device="xpu")
    norm_weight = torch.randn(hidden_size, dtype=torch.bfloat16, device="xpu")

    # Create output tensors
    post_mix = torch.empty(T, hc, dtype=torch.float32, device="xpu")
    comb_mix = torch.empty(T, hc * hc, dtype=torch.float32, device="xpu")
    layer_input = torch.empty(T, hidden_size, dtype=torch.bfloat16, device="xpu")

    # Determine which variant to benchmark
    use_norm = provider == "with_norm"

    # Warmup
    for _ in range(1000):
        if use_norm:
            hc_pre_big_fuse(
                gemm_out_mul,
                gemm_out_sqrsum,
                hc_scale,
                hc_base,
                residual,
                post_mix,
                comb_mix,
                layer_input,
                hc,
                sinkhorn_iters,
                n_splits,
                1e-5,  # rms_eps
                1e-6,  # hc_pre_eps
                1e-6,  # hc_sinkhorn_eps
                2.0,  # hc_post_mult_value
                norm_weight,  # norm_weight
                1e-6,  # norm_eps
            )
        else:
            hc_pre_big_fuse(
                gemm_out_mul,
                gemm_out_sqrsum,
                hc_scale,
                hc_base,
                residual,
                post_mix,
                comb_mix,
                layer_input,
                hc,
                sinkhorn_iters,
                n_splits,
                1e-5,  # rms_eps
                1e-6,  # hc_pre_eps
                1e-6,  # hc_sinkhorn_eps
                2.0,  # hc_post_mult_value
            )
    torch.xpu.synchronize()

    if use_norm:
        bench_lambda = lambda: hc_pre_big_fuse(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual,
            post_mix,
            comb_mix,
            layer_input,
            hc,
            sinkhorn_iters,
            n_splits,
            1e-5,
            1e-6,
            1e-6,
            2.0,
            norm_weight,
            1e-6,
        )
    else:
        bench_lambda = lambda: hc_pre_big_fuse(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual,
            post_mix,
            comb_mix,
            layer_input,
            hc,
            sinkhorn_iters,
            n_splits,
            1e-5,
            1e-6,
            1e-6,
            2.0,
        )

    quantiles = [0.5, 0.25, 0.75]
    ms, _, _ = triton.testing.do_bench(
        bench_lambda, quantiles=quantiles, return_mode="median"
    )

    torch.xpu.empty_cache()

    # Calculate memory bandwidth
    # Inputs: gemm_out_mul, gemm_out_sqrsum, hc_scale, hc_base, residual, (norm_weight if with_norm)
    # Outputs: post_mix, comb_mix, layer_input
    read_bytes = (
        n_splits * T * hc_mult3 * 4  # gemm_out_mul (fp32)
        + n_splits * T * 4  # gemm_out_sqrsum (fp32)
        + 3 * 4  # hc_scale (fp32)
        + hc_mult3 * 4  # hc_base (fp32)
        + T * hc * hidden_size * 2  # residual (bf16)
    )
    if use_norm:
        read_bytes += hidden_size * 2  # norm_weight (bf16)

    write_bytes = (
        T * hc * 4  # post_mix (fp32)
        + T * hc * hc * 4  # comb_mix (fp32)
        + T * hidden_size * 2  # layer_input (bf16)
    )
    total_bytes = read_bytes + write_bytes
    bandwidth_gb_s = total_bytes / (ms / 1e3) / 1e9

    all_results.append(
        {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "T": T,
            "hidden_size": hidden_size,
            "n_splits": n_splits,
            "with_norm": use_norm,
            "ms": ms,
            "Mtok_per_sec": T / (ms / 1e3) / 1e6,
            "bandwidth_gb_s": bandwidth_gb_s,
        }
    )
    return ms


if __name__ == "__main__":
    benchmark.run(print_data=False)
    print("Benchmark finished!")

    df = pd.DataFrame(all_results)
    print("\n" + "=" * 80)
    print("HC_PRE_FUSE BENCHMARK RESULTS")
    print("=" * 80)
    print(df.to_markdown(index=False))
    print("\n")

    # Summary statistics by variant
    print("Summary Statistics:")
    for with_norm_val in [False, True]:
        df_variant = df[df["with_norm"] == with_norm_val]
        variant_name = "With Norm" if with_norm_val else "Without Norm"
        print(f"\n{variant_name}:")
        print(f"  Mean throughput: {df_variant['Mtok_per_sec'].mean():.2f} Mtok/s")
        print(f"  Mean bandwidth: {df_variant['bandwidth_gb_s'].mean():.2f} GB/s")
        print(f"  Best throughput: {df_variant['Mtok_per_sec'].max():.2f} Mtok/s")
        print(f"  Best bandwidth: {df_variant['bandwidth_gb_s'].max():.2f} GB/s")
