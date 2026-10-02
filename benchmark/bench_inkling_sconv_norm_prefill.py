"""Device-time roofline benchmark for Inkling reduced prefill SConv and RMSNorm."""

import argparse

import torch
import triton
from sgl_kernel import fused_add_rmsnorm, rmsnorm
from sgl_kernel.inkling_sconv import causal_conv1d

HBM_GBPS = 350.0
PEAK_TOPS = 50.0


def _dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[name]


def _bench(fn, warmup: int, rep: int) -> tuple[float, float, float]:
    return triton.testing.do_bench(
        fn, warmup=warmup, rep=rep, quantiles=[0.5, 0.2, 0.8]
    )


def _report(
    name: str,
    measured_ms: float,
    p20_ms: float,
    p80_ms: float,
    bytes_: int,
    flops: int,
    calls: int,
) -> float:
    total_bytes = bytes_ * calls
    total_flops = flops * calls
    total_ms = measured_ms * calls
    memory_ms = total_bytes / (HBM_GBPS * 1.0e9) * 1.0e3
    compute_ms = total_flops / (PEAK_TOPS * 1.0e12) * 1.0e3
    roofline_ms = max(memory_ms, compute_ms)
    efficiency = roofline_ms / total_ms * 100.0
    bandwidth = total_bytes / (total_ms * 1.0e-3) / 1.0e9
    intensity = total_flops / total_bytes
    print(
        f"{name:18s} calls={calls} median={total_ms:8.4f} ms "
        f"p20={p20_ms * calls:8.4f} ms p80={p80_ms * calls:8.4f} ms "
        f"bytes={total_bytes / 1.0e6:9.3f} MB roofline={roofline_ms:8.4f} ms "
        f"op/B={intensity:5.2f} bound={'memory' if memory_ms >= compute_ms else 'compute':7s} "
        f"effective={bandwidth:7.2f} GB/s efficiency={efficiency:6.2f}%"
    )
    return efficiency


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--M", type=int, default=4096)
    parser.add_argument("--H", type=int, default=6144)
    parser.add_argument("--W", type=int, default=4)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--min-efficiency", type=float, default=0.0)
    args = parser.parse_args()

    if not torch.xpu.is_available():
        raise RuntimeError("XPU device is required")
    dtype = _dtype(args.dtype)
    elem = torch.empty((), dtype=dtype).element_size()
    torch.manual_seed(123)

    x = torch.randn((args.M, args.H), device="xpu", dtype=dtype)
    weight = torch.randn((args.H, args.W), device="xpu", dtype=dtype) * 0.1
    cache = torch.randn((1, args.W - 1, args.H), device="xpu", dtype=dtype) * 0.1
    cache_mask = torch.ones((1, 1, 1), device="xpu", dtype=torch.bool)
    safe_idx = torch.zeros((1,), device="xpu", dtype=torch.int64)
    cu = torch.tensor([0, args.M], device="xpu", dtype=torch.int64)
    si = torch.zeros((args.M,), device="xpu", dtype=torch.int32)

    def run_sconv():
        causal_conv1d(
            x,
            weight,
            cache,
            cache_mask,
            safe_idx,
            cu,
            si,
            activation="silu",
            use_residual=True,
        )

    sconv = _bench(run_sconv, args.warmup, args.rep)
    # Compulsory HBM traffic: x read, y write, weights, and unique prefix state.
    sconv_bytes = (
        2 * args.M * args.H * elem
        + args.H * args.W * elem
        + (args.W - 1) * args.H * elem
        + args.M * torch.empty((), dtype=torch.int32).element_size()
    )
    sconv_flops = args.M * args.H * (2 * args.W + 5)
    efficiencies = {
        "attention_sconv": _report(
            "attention_sconv", *sconv, sconv_bytes, sconv_flops, args.layers
        ),
        "mlp_sconv": _report(
            "mlp_sconv", *sconv, sconv_bytes, sconv_flops, args.layers
        ),
    }

    norm_weight = torch.randn((args.H,), device="xpu", dtype=dtype)
    norm_out = torch.empty_like(x)

    def run_plain_norm():
        rmsnorm(x, norm_weight, out=norm_out)

    plain_norm = _bench(run_plain_norm, args.warmup, args.rep)
    plain_bytes = 2 * args.M * args.H * elem + args.H * elem
    plain_flops = 4 * args.M * args.H

    norm_x = torch.randn_like(x)
    residual = torch.randn_like(x)

    def run_fused_norm():
        fused_add_rmsnorm(norm_x, residual, norm_weight)

    # Restore values periodically by using a short timing window. The device
    # work and memory traffic are independent of the evolving in-place values.
    fused_norm = _bench(run_fused_norm, args.warmup, args.rep)
    fused_bytes = 4 * args.M * args.H * elem + args.H * elem
    fused_flops = 5 * args.M * args.H

    attention_ms = plain_norm[0] + (args.layers - 1) * fused_norm[0]
    attention_p20 = plain_norm[1] + (args.layers - 1) * fused_norm[1]
    attention_p80 = plain_norm[2] + (args.layers - 1) * fused_norm[2]
    attention_bytes = plain_bytes + (args.layers - 1) * fused_bytes
    attention_flops = plain_flops + (args.layers - 1) * fused_flops
    efficiencies["attention_norm"] = _report(
        "attention_norm",
        attention_ms,
        attention_p20,
        attention_p80,
        attention_bytes,
        attention_flops,
        1,
    )
    efficiencies["mlp_norm"] = _report(
        "mlp_norm", *fused_norm, fused_bytes, fused_flops, args.layers
    )
    failures = {
        name: efficiency
        for name, efficiency in efficiencies.items()
        if efficiency < args.min_efficiency
    }
    if failures:
        detail = ", ".join(f"{name}={value:.2f}%" for name, value in failures.items())
        raise RuntimeError(
            f"roofline efficiency below {args.min_efficiency:.2f}%: {detail}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
