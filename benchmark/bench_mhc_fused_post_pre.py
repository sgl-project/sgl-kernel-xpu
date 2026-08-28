import pandas as pd
import torch
import triton
from sgl_kernel import mhc_fused_post_pre

configs = [
    # (b_s, seq_len, hidden_size)
    (1, 1, 4096),
    (8, 1, 4096),
    (17, 1, 4096),
    (32, 1, 4096),
    (64, 1, 4096),
    (128, 1, 4096),
    (512, 1, 4096),
    (1024, 1, 4096),
    (2048, 1, 4096),
    (1, 1, 7168),
    (8, 1, 7168),
    (17, 1, 7168),
    (32, 1, 7168),
    (64, 1, 7168),
    (128, 1, 7168),
    (512, 1, 7168),
    (1024, 1, 7168),
    (2048, 1, 7168),
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


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["b_s", "seq_len", "hidden_size"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["without_norm", "with_norm"],
        line_names=["Without Norm", "With Norm"],
        styles=[("green", "-"), ("blue", "--")],
        ylabel="Time (ms)",
        plot_name="mhc-fused-post-pre-performance",
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
    D = hidden_size
    hc_hidden = hc * D
    use_norm = provider == "with_norm"

    x = torch.randn(T, D, dtype=torch.bfloat16, device="xpu")
    residual = torch.randn(T, hc, D, dtype=torch.bfloat16, device="xpu")
    post = torch.rand(T, hc, dtype=torch.float32, device="xpu") * 2.0
    comb = torch.rand(T, hc, hc, dtype=torch.float32, device="xpu")
    comb = comb / comb.sum(dim=-1, keepdim=True)
    fn = torch.randn(hc_mult3, hc_hidden, dtype=torch.float32, device="xpu")
    hc_scale = torch.rand(3, dtype=torch.float32, device="xpu") * 0.5 + 0.5
    hc_base = torch.randn(hc_mult3, dtype=torch.float32, device="xpu") * 0.1
    norm_weight = (torch.randn(D, dtype=torch.float32, device="xpu") * 0.5 + 1.0).to(
        torch.bfloat16
    )

    nw = norm_weight if use_norm else None

    def run():
        return mhc_fused_post_pre(
            x,
            residual,
            post,
            comb,
            fn,
            hc_scale,
            hc_base,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            hc_sinkhorn_eps=hc_sinkhorn_eps,
            hc_post_mult_value=hc_post_mult_value,
            sinkhorn_repeat=sinkhorn_repeat,
            norm_weight=nw,
            norm_eps=norm_eps if use_norm else None,
        )

    for _ in range(10):
        run()
    torch.xpu.synchronize()

    quantiles = [0.5, 0.25, 0.75]
    ms, _, _ = triton.testing.do_bench(run, quantiles=quantiles, return_mode="median")

    torch.xpu.empty_cache()

    all_results.append(
        {
            "b_s": b_s,
            "seq_len": seq_len,
            "T": T,
            "hidden_size": hidden_size,
            "with_norm": use_norm,
            "ms": ms,
            "Mtok_per_sec": T / (ms / 1e3) / 1e6,
        }
    )
    return ms


if __name__ == "__main__":
    benchmark.run(print_data=False)
    print("Benchmark finished!")

    df = pd.DataFrame(all_results)
    print("\n" + "=" * 80)
    print("MHC_FUSED_POST_PRE BENCHMARK RESULTS")
    print("=" * 80)
    print(df.to_markdown(index=False))
    print("\n")

    print("Summary Statistics:")
    for with_norm_val in [False, True]:
        df_variant = df[df["with_norm"] == with_norm_val]
        variant_name = "With Norm" if with_norm_val else "Without Norm"
        print(f"\n{variant_name}:")
        print(f"  Mean latency:    {df_variant['ms'].mean():.4f} ms")
        print(f"  Mean throughput: {df_variant['Mtok_per_sec'].mean():.2f} Mtok/s")
        print(f"  Best throughput: {df_variant['Mtok_per_sec'].max():.2f} Mtok/s")
