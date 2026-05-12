# End-to-end benchmark: fused_experts(use_mxfp4_w4a16=True) — fused-kernel
# path vs dequant-then-bf16-GEMM path.
#
# Run:
#   python benchmark/bench_fused_experts_mxfp4.py
#
# Compares two providers at MoE-layer granularity (both GEMMs + scatter +
# activation + combine):
#
#   bf16_dequant   fused_experts(..., use_fused_mxfp4_kernel=False) — the
#                  current path: dequantize_mxfp4_weights on XPU produces
#                  a transient bf16 weight tensor, then moe_grouped_mm_nt_xe20.
#   mxfp4_fused    fused_experts(..., use_fused_mxfp4_kernel=True) — the new
#                  path: moe_grouped_mm_nt_xe20_mxfp4 dequants per-tile in
#                  registers, no intermediate bf16 weight.
#
# Both producers compute the same mathematical result (MXFP4-rounded weights
# × bf16 activations), so ms and peak transient GMEM are directly comparable.
#
# Shapes are scaled to fit BMG VRAM so the bf16_dequant path can actually run —
# the fused path tolerates much larger shapes, but that's exactly the point of
# the comparison.

import gc
import sys
from pathlib import Path

import sgl_kernel  # noqa: F401 — registers torch.ops.sgl_kernel
import torch
import triton
from sgl_kernel import fused_experts

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from mxfp4_utils import MXFP4_BLOCK_SIZE  # noqa: E402
from mxfp4_utils import quantize_mxfp4_2d  # noqa: E402

ALL_RESULTS = []


def _build_and_quantize_weights_per_expert(E: int, rows: int, cols: int):
    # Quantize one expert at a time. A full (E, rows, cols) bf16 tensor
    # followed by `.float()` inside quantize_mxfp4_2d can briefly need tens of
    # GB of host RAM at large shapes and trigger the Linux OOM killer.
    packed = torch.empty((E, rows, cols // 2), dtype=torch.uint8)
    scales = torch.empty((E, rows, cols // MXFP4_BLOCK_SIZE), dtype=torch.uint8)
    for e in range(E):
        w_e_bf16 = torch.empty((rows, cols), dtype=torch.bfloat16).normal_(0, 0.01)
        p_e, s_e = quantize_mxfp4_2d(w_e_bf16.float(), MXFP4_BLOCK_SIZE)
        packed[e].copy_(p_e)
        scales[e].copy_(s_e)
        del w_e_bf16, p_e, s_e
    return packed, scales


# (config_name, num_tokens, num_experts, topk, hidden, intermediate).
# Shapes chosen so the transient bf16 weight on the bf16_dequant path
# (E * 3 * I * H * 2B) stays under a few GiB on BMG.
SHAPES = [
    ("prefill_i512", 512, 64, 4, 7168, 512),
    ("prefill_i1024", 512, 64, 4, 7168, 1024),
    ("prefill_i2048", 512, 64, 4, 7168, 2048),
    ("decode_i512", 1, 64, 4, 7168, 512),
    ("decode_i1024", 1, 64, 4, 7168, 1024),
    ("decode_i2048", 1, 64, 4, 7168, 2048),
]


def _prepare_inputs(num_tokens, num_experts, topk, hidden, intermediate):
    torch.manual_seed(0)
    torch.xpu.manual_seed_all(0)

    a = torch.empty(num_tokens, hidden, dtype=torch.bfloat16, device="xpu").normal_(
        0, 0.01
    )

    # Build + quantize weights one expert at a time on CPU to avoid materializing
    # the full (E, rows, cols) bf16 tensor (plus its fp32 copy inside
    # quantize_mxfp4_2d). quantize_mxfp4_2d is CPU-only.
    w1_packed_cpu, w1_scale_cpu = _build_and_quantize_weights_per_expert(
        num_experts, 2 * intermediate, hidden
    )
    w2_packed_cpu, w2_scale_cpu = _build_and_quantize_weights_per_expert(
        num_experts, hidden, intermediate
    )

    # fused_experts expects int8 packed weights and fp32 K-outer scales
    # ([E, K/32, N]) for the kernel's coalesced scale load.
    w1_scale_fp32 = (
        torch.exp2((w1_scale_cpu.to(torch.int32) - 127).to(torch.float32))
        .transpose(1, 2)
        .contiguous()
    )
    w2_scale_fp32 = (
        torch.exp2((w2_scale_cpu.to(torch.int32) - 127).to(torch.float32))
        .transpose(1, 2)
        .contiguous()
    )

    score = torch.randn(num_tokens, num_experts, dtype=torch.bfloat16, device="xpu")
    score = torch.softmax(score.float(), dim=-1)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.to(torch.bfloat16)

    return {
        "a": a,
        "w1_packed": w1_packed_cpu.view(torch.int8).to("xpu"),
        "w2_packed": w2_packed_cpu.view(torch.int8).to("xpu"),
        "w1_scale": w1_scale_fp32.to("xpu"),
        "w2_scale": w2_scale_fp32.to("xpu"),
        "topk_weight": topk_weight,
        "topk_ids": topk_ids,
    }


def _run_unfused(inputs):
    return fused_experts(
        inputs["a"],
        inputs["w1_packed"],
        inputs["w2_packed"],
        inputs["topk_weight"],
        inputs["topk_ids"],
        None,
        None,
        activation="silu",
        use_mxfp4_w4a16=True,
        use_fused_mxfp4_kernel=False,
        w1_scale=inputs["w1_scale"],
        w2_scale=inputs["w2_scale"],
    )


def _run_fused(inputs):
    return fused_experts(
        inputs["a"],
        inputs["w1_packed"],
        inputs["w2_packed"],
        inputs["topk_weight"],
        inputs["topk_ids"],
        None,
        None,
        activation="silu",
        use_mxfp4_w4a16=True,
        use_fused_mxfp4_kernel=True,
        w1_scale=inputs["w1_scale"],
        w2_scale=inputs["w2_scale"],
    )


def _peak_transient_bytes(run_fn, inputs) -> int:
    torch.xpu.synchronize()
    torch.xpu.reset_peak_memory_stats()
    run_fn(inputs)
    torch.xpu.synchronize()
    return torch.xpu.max_memory_allocated()


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["name", "num_tokens", "num_experts", "topk", "hidden", "intermediate"],
        x_vals=SHAPES,
        line_arg="provider",
        line_vals=["bf16_dequant", "mxfp4_fused"],
        line_names=[
            "bf16_dequant (Python dequant + bf16 GEMM)",
            "mxfp4_fused (tile-fused MXFP4)",
        ],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="Time (ms)",
        plot_name="fused-experts-mxfp4-e2e",
        args={},
    )
)
def benchmark(name, num_tokens, num_experts, topk, hidden, intermediate, provider):
    # Free any lingering tensors from the previous (shape, provider) run before
    # building a new set — otherwise the unfused path's XPU bf16 weight copy
    # can stack on top of the previous run's allocations and OOM the device.
    gc.collect()
    torch.xpu.empty_cache()

    inputs = _prepare_inputs(num_tokens, num_experts, topk, hidden, intermediate)
    run_fn = _run_unfused if provider == "bf16_dequant" else _run_fused

    # Warmup.
    for _ in range(5):
        run_fn(inputs)
    torch.xpu.synchronize()

    peak_transient = _peak_transient_bytes(run_fn, inputs)

    ms, ms_min, ms_max = triton.testing.do_bench(
        lambda: run_fn(inputs),
        warmup=50,
        rep=200,
        quantiles=[0.5, 0.2, 0.8],
    )

    # Effective TFLOPS for the MoE (2 GEMMs per token-expert):
    #   flop ≈ num_tokens * topk * (2 * 2*intermediate*hidden + 2*hidden*intermediate)
    #        = num_tokens * topk * 6 * hidden * intermediate
    flop = num_tokens * topk * 6 * hidden * intermediate
    tflops = flop / (ms / 1e3) / 1e12

    # Static weight footprint (same for both — bf16 never resident).
    weights_packed_MB = (
        (
            num_experts
            * (2 * intermediate + intermediate)
            * (hidden // 2 + hidden // MXFP4_BLOCK_SIZE)
        )
        / 1024
        / 1024
    )

    ALL_RESULTS.append(
        {
            "shape": name,
            "provider": provider,
            "num_tokens": num_tokens,
            "num_experts": num_experts,
            "topk": topk,
            "hidden": hidden,
            "intermediate": intermediate,
            "ms": round(ms, 4),
            "ms_min": round(ms_min, 4),
            "ms_max": round(ms_max, 4),
            "tflops": round(tflops, 2),
            "peak_transient_MB": round(peak_transient / 1024 / 1024, 2),
            "weights_packed_MB": round(weights_packed_MB, 2),
        }
    )
    del inputs
    gc.collect()
    torch.xpu.empty_cache()
    return ms


if __name__ == "__main__":
    benchmark.run(print_data=False)
    print("\nBenchmark finished!\n")
    import pandas as pd

    df = pd.DataFrame(ALL_RESULTS)
    # Pivot for side-by-side view.
    pivot_cols = ["ms", "tflops", "peak_transient_MB"]
    pv = df.pivot_table(
        index=["shape", "num_tokens", "num_experts", "topk", "hidden", "intermediate"],
        columns="provider",
        values=pivot_cols,
        aggfunc="first",
    )
    pv["speedup"] = (pv[("ms", "bf16_dequant")] / pv[("ms", "mxfp4_fused")]).round(2)
    pv["transient_saved_MB"] = (
        pv[("peak_transient_MB", "bf16_dequant")]
        - pv[("peak_transient_MB", "mxfp4_fused")]
    ).round(2)
    print(pv.to_markdown())
