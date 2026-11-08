import argparse
import itertools
import math

import sgl_kernel
import torch
import triton

# Qwen3.8-Flash-Next PLE gate: 4 hyper-connection streams, hidden 2560. gate_value runs in decode
# (one token per request), gate_reduce in NEXTN verify (requests x draft width tokens).
HC_COUNT = 4
HIDDEN_SIZE = 2560
TOKENS = [1, 4, 16, 64, 128, 256, 1024]
OPS = ["gate_value", "gate_reduce"]
PROVIDERS = ["sgl_kernel", "torch"]

times_us = {}


def gate_activation_torch(gate):
    return torch.sigmoid(gate.abs().clamp_min(1e-6).sqrt() * gate.sign())


def gate_value_torch(gate, value):
    return gate_activation_torch(gate) * value.unsqueeze(-2)


def gate_reduce_torch(key, query, value):
    gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(key.shape[-1])
    return gate_activation_torch(gate) * value.unsqueeze(-2)


def bytes_moved(op, tokens, hidden):
    itemsize = 2
    if op == "gate_value":
        return (
            tokens * HC_COUNT + tokens * hidden + tokens * HC_COUNT * hidden
        ) * itemsize
    return (
        2 * tokens * HC_COUNT * hidden + tokens * hidden + tokens * HC_COUNT * hidden
    ) * itemsize


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["op", "tokens"],
        x_vals=list(itertools.product(OPS, TOKENS)),
        line_arg="provider",
        line_vals=PROVIDERS,
        line_names=["sgl_kernel", "torch (eager)"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="qwen4-ple-gate-performance",
        args={},
    )
)
def benchmark(op, tokens, provider):
    dtype, device = torch.bfloat16, "xpu"
    torch.manual_seed(0)
    value = torch.randn(tokens, HIDDEN_SIZE, device=device, dtype=dtype)
    out = torch.empty(tokens, HC_COUNT, HIDDEN_SIZE, device=device, dtype=dtype)
    if op == "gate_value":
        gate = torch.randn(tokens, HC_COUNT, 1, device=device, dtype=dtype)

        def sgl_fn():
            sgl_kernel.qwen4_gate_value(gate, value, out=out)

        def torch_fn():
            gate_value_torch(gate, value)

    else:
        key = torch.randn(tokens, HC_COUNT, HIDDEN_SIZE, device=device, dtype=dtype)
        query = torch.randn(tokens, HC_COUNT, HIDDEN_SIZE, device=device, dtype=dtype)

        def sgl_fn():
            sgl_kernel.qwen4_gate_reduce(key, query, value, out=out)

        def torch_fn():
            gate_reduce_torch(key, query, value)

    fn = {"sgl_kernel": sgl_fn, "torch": torch_fn}[provider]

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    times_us[provider] = ms * 1000
    if provider == PROVIDERS[-1]:
        t = times_us
        gbps = bytes_moved(op, tokens, HIDDEN_SIZE) / (t["sgl_kernel"] * 1e-6) / 1e9
        print(
            f"{op:<14}{tokens:>8}{t['sgl_kernel']:>14.2f}{t['torch']:>14.2f}"
            f"{gbps:>14.1f}{t['torch'] / t['sgl_kernel']:>14.2f}",
            flush=True,
        )
    return ms * 1000, min_ms * 1000, max_ms * 1000


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", nargs="+", type=int, default=TOKENS)
    parser.add_argument("--ops", nargs="+", choices=OPS, default=OPS)
    args = parser.parse_args()
    benchmark.benchmarks.x_vals = list(itertools.product(args.ops, args.tokens))

    print("qwen4 PLE gate ops: median time (triton do_bench), us\n")
    print(
        f"{'op':<14}{'tokens':>8}{'sgl_kernel':>14}{'torch':>14}{'sgl_GB/s':>14}{'vs_torch':>14}",
        flush=True,
    )
    benchmark.run(print_data=False)
