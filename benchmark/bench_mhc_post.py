import pandas as pd
import torch
import triton
from sgl_kernel import mhc_post

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

hc = 4
all_results = []


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["b_s", "seq_len", "hidden_size"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["sgl_kernel"],
        line_names=["SGL Kernel"],
        styles=[("green", "-")],
        ylabel="Time (ms)",
        plot_name="mhc-post-performance",
        args={},
    )
)
def benchmark(b_s, seq_len, hidden_size, provider):
    del provider
    torch.set_default_device("xpu")
    torch.xpu.manual_seed_all(42)

    T = b_s * seq_len

    x = torch.randn(T, hidden_size, dtype=torch.bfloat16, device="xpu")
    residual = torch.randn(T, hc, hidden_size, dtype=torch.bfloat16, device="xpu")
    post_layer_mix = torch.rand(T, hc, 1, dtype=torch.float32, device="xpu")
    comb_res_mix = torch.rand(T, hc, hc, dtype=torch.float32, device="xpu")

    def run():
        return mhc_post(x, residual, post_layer_mix, comb_res_mix)

    for _ in range(10):
        run()
    torch.xpu.synchronize()

    quantiles = [0.5, 0.25, 0.75]
    ms, _, _ = triton.testing.do_bench(run, quantiles=quantiles, return_mode="median")

    torch.xpu.empty_cache()

    read_bytes = (
        T * hidden_size * 2  # x bf16
        + T * hc * hidden_size * 2  # residual bf16
        + T * hc * 4  # post_layer_mix fp32
        + T * hc * hc * 4  # comb_res_mix fp32
    )
    write_bytes = T * hc * hidden_size * 2  # output bf16
    total_bytes = read_bytes + write_bytes
    bandwidth_gb_s = total_bytes / (ms / 1e3) / 1e9

    all_results.append(
        {
            "b_s": b_s,
            "seq_len": seq_len,
            "T": T,
            "hidden_size": hidden_size,
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
    print("MHC_POST BENCHMARK RESULTS")
    print("=" * 80)
    print(df.to_markdown(index=False))
    print("\n")

    print("Summary Statistics:")
    print(f"  Mean throughput: {df['Mtok_per_sec'].mean():.2f} Mtok/s")
    print(f"  Mean bandwidth:  {df['bandwidth_gb_s'].mean():.2f} GB/s")
    print(f"  Best throughput: {df['Mtok_per_sec'].max():.2f} Mtok/s")
    print(f"  Best bandwidth:  {df['bandwidth_gb_s'].max():.2f} GB/s")
