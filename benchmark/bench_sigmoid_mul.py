import argparse
import itertools

import sgl_kernel
import torch
import triton

# Attention output gate of Qwen3.5 full-attention and Qwen3.8 QSA layers: 24 heads x head_dim 256.
NUM_HEADS = 24
HEAD_DIM = 256
TOKENS = [1, 16, 64, 256, 1024, 4096, 8192]
GATES = ["flat", "strided_3d"]
PROVIDERS = ["sgl_kernel", "torch"]

times_us = {}


def make_gate(gate_layout, tokens, dtype, device):
    if gate_layout == "flat":
        return torch.randn(tokens, NUM_HEADS * HEAD_DIM, device=device, dtype=dtype)
    q_gate = torch.randn(tokens, NUM_HEADS, 2 * HEAD_DIM, device=device, dtype=dtype)
    return q_gate[..., HEAD_DIM:]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["gate_layout", "tokens"],
        x_vals=list(itertools.product(GATES, TOKENS)),
        line_arg="provider",
        line_vals=PROVIDERS,
        line_names=["sgl_kernel", "torch (eager)"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="sigmoid-mul-performance",
        args={},
    )
)
def benchmark(gate_layout, tokens, provider):
    dtype, device = torch.bfloat16, "xpu"
    torch.manual_seed(0)
    hidden = NUM_HEADS * HEAD_DIM
    x = torch.randn(tokens, hidden, device=device, dtype=dtype)
    gate = make_gate(gate_layout, tokens, dtype, device)

    def sgl_fn():
        sgl_kernel.fused_sigmoid_mul(x, gate, inplace=True)

    def torch_fn():
        x * torch.sigmoid(gate).reshape(tokens, hidden)

    fn = {"sgl_kernel": sgl_fn, "torch": torch_fn}[provider]

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    times_us[provider] = ms * 1000
    if provider == PROVIDERS[-1]:
        t = times_us
        gbps = 3 * tokens * hidden * x.element_size() / (t["sgl_kernel"] * 1e-6) / 1e9
        print(
            f"{gate_layout:<12}{tokens:>8}{t['sgl_kernel']:>14.2f}{t['torch']:>14.2f}"
            f"{gbps:>14.1f}{t['torch'] / t['sgl_kernel']:>14.2f}",
            flush=True,
        )
    return ms * 1000, min_ms * 1000, max_ms * 1000


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", nargs="+", type=int, default=TOKENS)
    parser.add_argument("--gates", nargs="+", choices=GATES, default=GATES)
    args = parser.parse_args()
    benchmark.benchmarks.x_vals = list(itertools.product(args.gates, args.tokens))

    print("fused_sigmoid_mul: median time (triton do_bench), us\n")
    print(
        f"{'gate':<12}{'tokens':>8}{'sgl_kernel':>14}{'torch':>14}{'sgl_GB/s':>14}{'vs_torch':>14}",
        flush=True,
    )
    benchmark.run(print_data=False)
