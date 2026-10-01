import itertools

import pandas as pd
import sgl_kernel
import torch
import triton
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

# Shapes are the ones a real serving workload issues, taken from a Qwen3-30B-A3B TP=4 trace at
# ISL/OSL 1024/1024 (hidden=2048 throughout): a decode batch equal to the running-request count,
# a short decode tail as requests retire, and ragged chunked-prefill chunks of ~2000 rows. The
# 8192 column covers Llama-3.3-70B. Decode dominates call count by roughly 30:1 over prefill.
DECODE_BATCHES = [1, 4, 20, 64, 128]
PREFILL_BATCHES = [1024, 1961, 2048]
HIDDEN_SIZES = [2048, 8192]

# 3D and 4D shapes matter because the row-offset decomposition (compute_row_offset) costs four
# integer div/mod per call regardless of how many levels are actually live, and Xe has no hardware
# integer divide. A contiguous 3D/4D tensor folds to a single level, so these shapes should cost the
# same per row as the equivalent 2D shape -- if they do not, the fold is not being exploited.
SHAPES_ND = [(2, 10, 2048), (4, 8, 2048), (8, 8, 2048), (2, 3, 4, 2048), (4, 16, 8192)]

configs = list(itertools.product(DECODE_BATCHES + PREFILL_BATCHES, HIDDEN_SIZES))


def effective_bandwidth_gbps(batch_size, hidden_size, time_ms, itemsize=2):
    """fused_add_rmsnorm reads input+residual+weight and writes input+residual."""
    moved = (4 * batch_size * hidden_size + hidden_size) * itemsize
    return moved / (time_ms * 1e-3) / 1e9


def fused_add_rmsnorm_torch(x, residual, weight, eps):
    """Unfused baseline with the same in-place semantics: residual += x, then x = rmsnorm(residual)."""
    residual.add_(x)
    r = residual.float()
    x.copy_(r * torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + eps) * weight.float())


def device_us(fn, iters=100):
    """Device time per call, from profiler events filtered to DeviceType.XPU.

    Reported alongside the wall time because the two answer different questions. The profiler
    attributes the same self_device_time to both the CPU-side op and the XPU kernel it launched, so
    an unfiltered sum double-counts.
    """
    for _ in range(5):
        fn()
    torch.xpu.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
        for _ in range(iters):
            fn()
        torch.xpu.synchronize()
    total = 0.0
    for ev in prof.key_averages():
        if ev.device_type == DeviceType.XPU:
            total += max(ev.self_device_time_total or 0, 0)
    return total / iters


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "hidden_size"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["sgl_kernel", "torch"],
        line_names=["sgl_kernel", "torch (unfused)"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="fused-add-rmsnorm-performance",
        args={},
    )
)
def benchmark(batch_size, hidden_size, provider):
    dtype, device, eps = torch.bfloat16, "xpu", 1e-6
    torch.manual_seed(0)
    # Allocated once and reused: fused_add_rmsnorm is in-place, and reallocating per call would time
    # the allocator instead of the kernel. Repeated application is numerically stable here --
    # residual grows linearly from ~1e-3 and the normalised output stays bounded.
    x = (torch.randn(batch_size, hidden_size, dtype=torch.float32) * 1e-3).to(
        device, dtype
    )
    residual = (torch.randn(batch_size, hidden_size, dtype=torch.float32) * 1e-3).to(
        device, dtype
    )
    weight = (torch.randn(hidden_size, dtype=torch.float32) * 0.05 + 1.0).to(
        device, dtype
    )

    if provider == "sgl_kernel":

        def fn():
            sgl_kernel.fused_add_rmsnorm(x, residual, weight, eps)

    elif provider == "torch":

        def fn():
            fused_add_rmsnorm_torch(x, residual, weight, eps)

    else:
        raise ValueError(f"unknown provider {provider}")

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    dev_us = device_us(fn)
    wall_us = ms * 1000

    results.append(
        {
            "shape": f"{batch_size}x{hidden_size}",
            "rows": batch_size,
            "hidden_size": hidden_size,
            "provider": provider,
            "wall_us": round(wall_us, 3),
            "device_us": round(dev_us, 3),
            # How much of the wall time the device does NOT account for. At the decode shapes this
            # op is host-bound, so wall time is the metric that moves when host cost changes and
            # device time is the control that should not.
            "host_bound_us": round(wall_us - dev_us, 3),
            "device_share_pct": round(100.0 * dev_us / wall_us, 1),
            "gbps": round(effective_bandwidth_gbps(batch_size, hidden_size, ms), 1),
        }
    )
    return wall_us, min_ms * 1000, max_ms * 1000


def benchmark_nd():
    """3D/4D shapes, reported as a plain table rather than through perf_report.

    These matter because compute_row_offset costs four integer div/mod per call regardless of how
    many index levels are actually live, and Xe has no hardware integer divide. A contiguous 3D or 4D
    tensor folds to a single level, so each row should cost what the equivalent 2D shape costs. If a
    3D row is dearer than a 2D row at the same width, the fold is not being exploited.
    """
    dtype, device, eps = torch.bfloat16, "xpu", 1e-6
    print(
        f"\n  {'shape':<18}{'rows':>6}{'wall_us':>10}{'device_us':>11}{'wall/row_ns':>13}"
    )
    for shape in SHAPES_ND:
        torch.manual_seed(0)
        x = (torch.randn(*shape, dtype=torch.float32) * 1e-3).to(device, dtype)
        residual = (torch.randn(*shape, dtype=torch.float32) * 1e-3).to(device, dtype)
        weight = (torch.randn(shape[-1], dtype=torch.float32) * 0.05 + 1.0).to(
            device, dtype
        )

        def fn():
            sgl_kernel.fused_add_rmsnorm(x, residual, weight, eps)

        ms, _, _ = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
        rows = 1
        for d in shape[:-1]:
            rows *= d
        wall_us = ms * 1000
        print(
            f"  {'x'.join(map(str, shape)):<18}{rows:>6}{wall_us:>10.2f}"
            f"{device_us(fn):>11.2f}{wall_us / rows * 1000:>13.1f}"
        )


results = []

if __name__ == "__main__":
    print(
        "fused_add_rmsnorm: wall time (triton do_bench) and device time (torch profiler, XPU only)"
    )
    print(
        "At the decode shapes this op is host-bound, so wall_us responds to host-path changes"
    )
    print("while device_us is the control.\n")
    benchmark.run(print_data=True)
    benchmark_nd()
    df = pd.DataFrame(results)
    df.to_csv("bench_fused_add_rmsnorm.csv", index=False)
    print("\n" + df.to_string(index=False))
    wall = df.pivot_table(
        index=["rows", "hidden_size"], columns="provider", values="wall_us"
    )
    wall["speedup_vs_torch"] = (wall["torch"] / wall["sgl_kernel"]).round(2)
    print("\n" + wall.to_string())
