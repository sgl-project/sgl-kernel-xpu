# Op-level benchmark for the tile-fused MXFP4 MoE grouped GEMM.
#
# Compares two providers:
#
#   bf16_dequant : dequantize packed MXFP4 → bf16, then run the standard
#                  moe_grouped_mm_nt_xe20 on the bf16 weights (the
#                  pre-fused path in fused_experts(use_mxfp4_w4a16=True)).
#   mxfp4_fused  : moe_grouped_mm_nt_xe20_mxfp4 directly on packed
#                  weights + fp32 direct-multiplier scales (tile-level
#                  dequant).
#
# Both paths compute the same mathematical result (MXFP4-rounded weights
# × bf16 activations). The fused path additionally avoids ever allocating
# the bf16 weight tensor — reported separately as peak transient GMEM.
#
# Run:
#   python benchmark/bench_moe_mxfp4_gemm.py

import sys
from pathlib import Path

import sgl_kernel  # noqa: F401 — registers the torch.ops.sgl_kernel namespace
import torch
import triton
from sgl_kernel.moe import dequantize_mxfp4_weights

# The CPU MXFP4 quantize/dequantize helpers live next to the tests.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from mxfp4_utils import MXFP4_BLOCK_SIZE  # noqa: E402
from mxfp4_utils import quantize_mxfp4_2d  # noqa: E402

NUM_EXPERTS = 8
BENCH_SHAPES = [
    # (avg_m_per_expert, gemm_n, gemm_k)   # total_m = num_experts * avg_m
    (16, 1024, 1024),
    (33, 1024, 1024),
    (64, 1024, 1024),
    (128, 1024, 1024),
    (33, 2048, 1024),
    (33, 1024, 2048),
    (128, 2048, 2048),
]


ALL_RESULTS = []


def _quantize_bf16_weights_mxfp4(w_bf16_cpu: torch.Tensor):
    """[E, N, K] bf16 → ([E, N, K/2] int8 packed, [E, N, K/32] fp32 direct mul)."""
    E, N, K = w_bf16_cpu.shape
    flat = w_bf16_cpu.reshape(E * N, K).float().cpu()
    packed_u8, scales_u8 = quantize_mxfp4_2d(flat, MXFP4_BLOCK_SIZE)
    packed_i8 = packed_u8.view(torch.int8).reshape(E, N, K // 2)
    scales_fp32 = torch.exp2((scales_u8.to(torch.int32) - 127).to(torch.float32))
    scales_fp32 = scales_fp32.reshape(E, N, K // MXFP4_BLOCK_SIZE)
    return packed_i8, scales_fp32


def _prepare_inputs(avg_m: int, gemm_n: int, gemm_k: int):
    """Build the XPU tensors shared across providers.

    We deliberately do NOT allocate a bf16-dequantized weight tensor
    up-front: the `bf16_dequant` provider pays for the XPU dequant
    inside its timed region (mirroring what fused_experts does today).
    """
    torch.manual_seed(0)
    torch.xpu.manual_seed_all(0)

    total_m = NUM_EXPERTS * avg_m
    total_rows = torch.full((NUM_EXPERTS,), avg_m, dtype=torch.int32, device="xpu")

    a = torch.empty((total_m, gemm_k), dtype=torch.bfloat16, device="xpu").normal_(
        0, 0.01
    )

    w_bf16_cpu = torch.empty(
        (NUM_EXPERTS, gemm_n, gemm_k), dtype=torch.bfloat16
    ).normal_(0, 0.01)
    packed_cpu, scales_cpu = _quantize_bf16_weights_mxfp4(w_bf16_cpu)

    packed_xpu = packed_cpu.to("xpu")
    scales_xpu = scales_cpu.to("xpu")

    output = torch.empty((total_m, gemm_n), dtype=torch.bfloat16, device="xpu")

    return {
        "activations": a,
        "total_rows": total_rows,
        "packed_xpu": packed_xpu,
        "scales_xpu": scales_xpu,
        "output": output,
        "total_m": total_m,
    }


def _run_bf16_dequant(inputs):
    # On-the-fly XPU dequant to bf16, then standard bf16 GEMM. This is
    # what fused_experts(use_mxfp4_w4a16=True) does today: dequant once per
    # GEMM call, immediately hand to moe_grouped_mm_nt_xe20.
    w_bf16 = dequantize_mxfp4_weights(
        inputs["packed_xpu"],
        inputs["scales_xpu"],
    )
    torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20(
        inputs["output"],
        inputs["activations"],
        w_bf16,
        None,
        inputs["total_rows"],
        NUM_EXPERTS,
        0,  # silu
        False,  # no fuse_act
        1.702,
        7.0,
    )


def _run_mxfp4_fused(inputs):
    torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20_mxfp4(
        inputs["output"],
        inputs["activations"],
        inputs["packed_xpu"],
        inputs["scales_xpu"],
        None,
        inputs["total_rows"],
        NUM_EXPERTS,
        0,  # silu
        False,  # no fuse_act
        1.702,
        7.0,
    )


def _peak_transient_bytes(run_fn, inputs) -> int:
    """Measure peak XPU allocator usage inside one call."""
    torch.xpu.synchronize()
    torch.xpu.reset_peak_memory_stats()
    run_fn(inputs)
    torch.xpu.synchronize()
    return torch.xpu.max_memory_allocated()


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["avg_m", "gemm_n", "gemm_k"],
        x_vals=BENCH_SHAPES,
        line_arg="provider",
        line_vals=["bf16_dequant", "mxfp4_fused"],
        line_names=[
            "bf16_dequant (pre-dequantized)",
            "mxfp4_fused (tile-level dequant)",
        ],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="Time (ms)",
        plot_name="moe-mxfp4-gemm-op-level",
        args={},
    )
)
def benchmark(avg_m, gemm_n, gemm_k, provider):
    inputs = _prepare_inputs(avg_m, gemm_n, gemm_k)
    run_fn = _run_bf16_dequant if provider == "bf16_dequant" else _run_mxfp4_fused

    # Warm up.
    for _ in range(5):
        run_fn(inputs)
    torch.xpu.synchronize()

    # Peak transient memory — note this captures only the transient,
    # not the resident weight tensors (which we allocated up-front in
    # _prepare_inputs). To contrast the real savings, see the
    # `weights_bytes` column: the fused path never needs to allocate
    # w_dq_xpu, whereas bf16_dequant requires it to exist.
    peak_transient = _peak_transient_bytes(run_fn, inputs)

    quantiles = [0.5, 0.2, 0.8]
    ms, ms_min, ms_max = triton.testing.do_bench(
        lambda: run_fn(inputs),
        warmup=50,
        rep=200,
        quantiles=quantiles,
    )

    # GEMM flop = 2 * M * N * K (MAC = 2 flops).
    total_m = inputs["total_m"]
    flop = 2 * total_m * gemm_n * gemm_k

    # Effective B-side bandwidth read from the resident weight tensor
    # that the GEMM consumes. For bf16_dequant the GEMM reads a bf16
    # weight tensor that was produced by the XPU dequant step (written
    # by dequantize_mxfp4_weights → read by moe_grouped_mm_nt_xe20, so
    # effectively bf16 bandwidth). For mxfp4_fused the GEMM reads the
    # packed MXFP4 directly (0.5 B/elem) plus fp32 scales (4 B per
    # MXFP4_BLOCK_SIZE elements).
    if provider == "bf16_dequant":
        b_bytes = NUM_EXPERTS * gemm_n * gemm_k * 2
    else:
        b_bytes = NUM_EXPERTS * gemm_n * gemm_k * (1 / 2 + 4 / MXFP4_BLOCK_SIZE)

    # Resident weight footprint. Both providers hold the same packed
    # tensor + scales as the "on-disk" truth; bf16_dequant additionally
    # materializes the bf16 tensor transiently per GEMM call (captured
    # by peak_transient_MB). Scales are fp32 (4 B per group).
    packed_bytes = (
        NUM_EXPERTS * gemm_n * (gemm_k // 2 + 4 * (gemm_k // MXFP4_BLOCK_SIZE))
    )
    bf16_transient = NUM_EXPERTS * gemm_n * gemm_k * 2
    weights_resident_bytes = packed_bytes
    transient_bf16_bytes = bf16_transient if provider == "bf16_dequant" else 0

    tflops = flop / (ms / 1e3) / 1e12
    b_gbps = b_bytes / (ms / 1e3) / 1e9

    ALL_RESULTS.append(
        {
            "provider": provider,
            "avg_m": avg_m,
            "total_m": total_m,
            "gemm_n": gemm_n,
            "gemm_k": gemm_k,
            "ms": round(ms, 4),
            "ms_min": round(ms_min, 4),
            "ms_max": round(ms_max, 4),
            "tflops": round(tflops, 2),
            "b_gbps": round(b_gbps, 1),
            "peak_transient_MB": round(peak_transient / 1024 / 1024, 2),
            "weights_resident_MB": round(weights_resident_bytes / 1024 / 1024, 2),
            "transient_bf16_MB": round(transient_bf16_bytes / 1024 / 1024, 2),
        }
    )
    return ms


if __name__ == "__main__":
    benchmark.run(print_data=False)
    print("\nBenchmark finished!\n")
    import pandas as pd

    df = pd.DataFrame(ALL_RESULTS)
    # Pivot on provider so the comparison is one row per shape.
    pivot_cols = [
        "ms",
        "tflops",
        "b_gbps",
        "peak_transient_MB",
        "weights_resident_MB",
        "transient_bf16_MB",
    ]
    pv = df.pivot_table(
        index=["avg_m", "total_m", "gemm_n", "gemm_k"],
        columns="provider",
        values=pivot_cols,
        aggfunc="first",
    )
    pv["speedup_fused_vs_bf16"] = (
        pv[("ms", "bf16_dequant")] / pv[("ms", "mxfp4_fused")]
    ).round(2)
    # bf16_dequant transiently materializes the bf16 weight tensor every
    # GEMM call; mxfp4_fused never does. This is the real GMEM saving.
    pv["transient_bf16_saved_MB"] = pv[("transient_bf16_MB", "bf16_dequant")].round(2)
    print(pv.to_markdown())
