"""Production-shape Inkling TP4 routed-MoE pipeline component benchmark.

This reproduces the memory-bound stages surrounding the two MXFP4 grouped
GEMMs for one 4096-token routed layer: E=256, top-k=6, hidden=6144 and local
intermediate=768.  It calls the same registered sgl-kernel operators used by
``fused_experts`` and reports XPU-event p20/p50/p80 device time.
"""

import argparse
import statistics

import sgl_kernel  # noqa: F401
import torch

TOKENS = 4096
TOPK = 6
HIDDEN = 6144
INTERMEDIATE = 768
ROUTES = TOKENS * TOPK
MEMORY_ROOFLINE_GBPS = 350.0


def event_samples(fn, warmup: int, repetitions: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    samples = []
    for _ in range(repetitions):
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.xpu.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[round(fraction * (len(ordered) - 1))]


def report(name: str, samples: list[float], bytes_moved: int) -> None:
    p50 = statistics.median(samples)
    gbps = bytes_moved / (p50 * 1e-3) / 1e9
    roofline_ms = bytes_moved / (MEMORY_ROOFLINE_GBPS * 1e9) * 1e3
    attainment = roofline_ms / p50 * 100
    print(
        f"{name}: p20={percentile(samples, 0.2):.4f} "
        f"p50={p50:.4f} p80={percentile(samples, 0.8):.4f} ms "
        f"effective={gbps:.1f} GB/s roofline={roofline_ms:.4f} ms "
        f"attainment={attainment:.1f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("xpu")
    x = torch.randn((TOKENS, HIDDEN), device=device, dtype=torch.bfloat16)
    permutation = torch.randperm(ROUTES, device=device, dtype=torch.int64).to(
        torch.int32
    )
    scattered = torch.empty((ROUTES, HIDDEN), device=device, dtype=torch.bfloat16)
    gateup = torch.randn(
        (ROUTES, 2 * INTERMEDIATE), device=device, dtype=torch.bfloat16
    )
    activated = torch.empty((ROUTES, INTERMEDIATE), device=device, dtype=torch.bfloat16)
    down = torch.randn((ROUTES, HIDDEN), device=device, dtype=torch.bfloat16)
    combined = torch.empty_like(x)
    weights = torch.rand((TOKENS, TOPK), device=device, dtype=torch.bfloat16)

    scatter = lambda: torch.ops.sgl_kernel.scatter_tokens_to_experts.default(
        x, permutation, scattered
    )
    activation = lambda: torch.ops.sgl_kernel.silu_and_mul(activated, gateup)
    combine = lambda: torch.ops.sgl_kernel.apply_shuffle_mul_sum.default(
        down, combined, permutation, 1.0, weights
    )

    element_bytes = x.element_size()
    scatter_bytes = (TOKENS * HIDDEN + ROUTES * HIDDEN) * element_bytes
    activation_bytes = ROUTES * (3 * INTERMEDIATE) * element_bytes
    combine_bytes = (ROUTES * HIDDEN + TOKENS * HIDDEN) * element_bytes
    report(
        "scatter",
        event_samples(scatter, args.warmup, args.repetitions),
        scatter_bytes,
    )
    report(
        "activation",
        event_samples(activation, args.warmup, args.repetitions),
        activation_bytes,
    )
    report(
        "combine",
        event_samples(combine, args.warmup, args.repetitions),
        combine_bytes,
    )


if __name__ == "__main__":
    main()
