import argparse
import itertools

import sgl_kernel
import torch
import triton

# name -> (heads, head_dim, gate layout, element offset of the x/gate views). The first two are the attention
# output gate of Qwen3.5 full-attention and Qwen3.8 QSA layers (24 heads x head_dim 256, 8-element vector path).
# head_dim 12/20 and the 4-element offset exercise the 4-element vector path, head_dim 13 the scalar path.
CASES = {
    "qwen_flat": (24, 256, "flat", 0),
    "qwen_strided": (24, 256, "strided_3d", 0),
    "hd12_flat": (24, 12, "flat", 0),
    "hd12_strided": (24, 12, "strided_3d", 0),
    "hd20_flat": (24, 20, "flat", 0),
    "hd256_off4": (24, 256, "flat", 4),
    "hd13_flat": (24, 13, "flat", 0),
}
TOKENS = [1, 64, 1024, 8192, 65536]
PROVIDERS = ["sgl_kernel", "torch"]

times_us = {}


def make_inputs(case, tokens, dtype, device):
    heads, head_dim, layout, offset = CASES[case]
    hidden = heads * head_dim
    # A non-zero offset shifts both views off 16-byte alignment while keeping 8-byte alignment.
    x_buf = torch.randn(tokens * hidden + offset, device=device, dtype=dtype)
    x = x_buf[offset:].view(tokens, hidden)
    if layout == "flat":
        gate_buf = torch.randn(tokens * hidden + offset, device=device, dtype=dtype)
        return x, gate_buf[offset:].view(tokens, hidden)
    q_gate = torch.randn(tokens, heads, 2 * head_dim, device=device, dtype=dtype)
    return x, q_gate[..., head_dim:]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["case", "tokens"],
        x_vals=list(itertools.product(CASES, TOKENS)),
        line_arg="provider",
        line_vals=PROVIDERS,
        line_names=["sgl_kernel", "torch (eager)"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="sigmoid-mul-performance",
        args={},
    )
)
def benchmark(case, tokens, provider):
    dtype, device = torch.bfloat16, "xpu"
    torch.manual_seed(0)
    x, gate = make_inputs(case, tokens, dtype, device)
    hidden = x.size(1)

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
            f"{case:<14}{tokens:>8}{t['sgl_kernel']:>14.2f}{t['torch']:>14.2f}"
            f"{gbps:>14.1f}{t['torch'] / t['sgl_kernel']:>14.2f}",
            flush=True,
        )
    return ms * 1000, min_ms * 1000, max_ms * 1000


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", nargs="+", type=int, default=TOKENS)
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    args = parser.parse_args()
    benchmark.benchmarks.x_vals = list(itertools.product(args.cases, args.tokens))

    print("fused_sigmoid_mul: median time (triton do_bench), us\n")
    print(
        f"{'case':<14}{'tokens':>8}{'sgl_kernel':>14}{'torch':>14}{'sgl_GB/s':>14}{'vs_torch':>14}",
        flush=True,
    )
    benchmark.run(print_data=False)
