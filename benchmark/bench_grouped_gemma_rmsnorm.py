import argparse

import sgl_kernel
import torch
import triton

# Qwen3.8-Flash-Next PLE norms: 4 hyper-connection streams x 2560 hidden = 10240 per row.
GROUP_SIZE = 2560
HIDDEN_SIZE = 4 * GROUP_SIZE
DECODE_BATCHES = [1, 4, 20, 64, 128]
PREFILL_BATCHES = [1024, 2048, 8192]
EPS = 1e-6

# Set from the CLI in main().
config = {"hidden_size": HIDDEN_SIZE, "group_size": GROUP_SIZE, "dtype": torch.bfloat16}
times_us = {}
PROVIDERS = ["sgl_kernel", "per_group", "torch"]


def grouped_gemma_rmsnorm_torch(x, w, group_size, eps):
    # Eager baseline; allocates its intermediates and output on every call.
    xf = x.float().reshape(*x.shape[:-1], -1, group_size)
    xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf.reshape(x.shape) * (1.0 + w.float())).to(x.dtype)


def per_group_gemma_rmsnorm(x, w, group_size, eps, out):
    for g in range(x.shape[-1] // group_size):
        s = slice(g * group_size, (g + 1) * group_size)
        sgl_kernel.gemma_rmsnorm(x[:, s], w[s], eps, out=out[:, s])
    return out


def effective_bandwidth_gbps(batch_size, hidden_size, time_ms, itemsize):
    """Reads input+weight and writes output."""
    moved = (2 * batch_size * hidden_size + hidden_size) * itemsize
    return moved / (time_ms * 1e-3) / 1e9


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=DECODE_BATCHES + PREFILL_BATCHES,
        line_arg="provider",
        line_vals=PROVIDERS,
        line_names=["sgl_kernel", "4x gemma_rmsnorm", "torch (eager)"],
        styles=[("blue", "-"), ("orange", "-"), ("green", "-")],
        ylabel="us",
        plot_name="grouped-gemma-rmsnorm-performance",
        args={},
    )
)
def benchmark(batch_size, provider):
    hidden_size, group_size, dtype = (
        config["hidden_size"],
        config["group_size"],
        config["dtype"],
    )
    torch.manual_seed(0)
    x = torch.randn(batch_size, hidden_size, device="xpu", dtype=dtype)
    w = torch.randn(hidden_size, device="xpu", dtype=dtype)
    out = torch.empty_like(x)

    if provider == "sgl_kernel":

        def fn():
            sgl_kernel.grouped_gemma_rmsnorm(x, w, group_size, EPS, out=out)

    elif provider == "per_group":

        def fn():
            per_group_gemma_rmsnorm(x, w, group_size, EPS, out)

    elif provider == "torch":

        def fn():
            grouped_gemma_rmsnorm_torch(x, w, group_size, EPS)

    else:
        raise ValueError(f"unknown provider {provider}")

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    times_us[provider] = ms * 1000
    if provider == PROVIDERS[-1]:
        t = times_us
        gbps = effective_bandwidth_gbps(
            batch_size, hidden_size, t["sgl_kernel"] / 1000, x.element_size()
        )
        print(
            f"{batch_size:>6}{t['sgl_kernel']:>14.2f}{t['per_group']:>14.2f}{t['torch']:>14.2f}"
            f"{gbps:>14.1f}{t['per_group'] / t['sgl_kernel']:>16.2f}{t['torch'] / t['sgl_kernel']:>14.2f}",
            flush=True,
        )
    return ms * 1000, min_ms * 1000, max_ms * 1000


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--decode-batches", nargs="+", type=int, default=DECODE_BATCHES)
    parser.add_argument(
        "--prefill-batches", nargs="+", type=int, default=PREFILL_BATCHES
    )
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    args = parser.parse_args()
    assert (
        args.hidden_size % args.group_size == 0
    ), "hidden size must be divisible by group size"
    config.update(
        hidden_size=args.hidden_size,
        group_size=args.group_size,
        dtype=getattr(torch, args.dtype),
    )
    benchmark.benchmarks.x_vals = args.decode_batches + args.prefill_batches

    print(
        f"grouped_gemma_rmsnorm {args.dtype} hidden={args.hidden_size} group={args.group_size}: "
        "median time (triton do_bench), us\n"
    )
    print(
        f"{'rows':>6}{'sgl_kernel':>14}{'4x_gemma':>14}{'torch':>14}{'sgl_GB/s':>14}"
        f"{'vs_4x_gemma':>16}{'vs_torch':>14}",
        flush=True,
    )
    benchmark.run(print_data=False)
