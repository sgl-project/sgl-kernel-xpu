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
# Shape families covered:
#   - DSv4-like prefill:  num_experts=256, hidden=7168, intermediate∈{512,1024,2048}
#   - DSv4-like decode :  num_tokens=1 slices of the above
#   - Mixtral-like     :  num_experts=8, hidden=4096, intermediate=3584
#   - Qwen-MoE-like    :  num_experts=64, hidden=3584, intermediate=1280

import sys
from pathlib import Path

import torch
import triton

import sgl_kernel  # noqa: F401 — registers torch.ops.sgl_kernel
from sgl_kernel import fused_experts

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from mxfp4_utils import MXFP4_BLOCK_SIZE  # noqa: E402
from mxfp4_utils import quantize_mxfp4_2d  # noqa: E402


ALL_RESULTS = []


def _quantize_weights_mxfp4_3d(w_bf16_cpu: torch.Tensor):
    E, rows, cols = w_bf16_cpu.shape
    flat = w_bf16_cpu.reshape(E * rows, cols).float().cpu()
    p, s = quantize_mxfp4_2d(flat, MXFP4_BLOCK_SIZE)
    return (
        p.reshape(E, rows, cols // 2),
        s.reshape(E, rows, cols // MXFP4_BLOCK_SIZE),
    )


# (config_name, num_tokens, num_experts, topk, hidden, intermediate).
# We keep num_tokens moderate so experts get a realistic avg_m distribution.
# Prefill/decode is expressed via num_tokens choice; topk and experts drive
# per-expert average M.
SHAPES = [
    # DSv4-like shapes (num_experts=256, topk=8, hidden=7168). intermediate
    # tracks tp-shard size.
    ("dsv4_prefill_tp8_i512",  512,  256, 8, 7168, 512),
    ("dsv4_prefill_tp4_i1024", 512,  256, 8, 7168, 1024),
    ("dsv4_prefill_tp2_i2048", 512,  256, 8, 7168, 2048),
    ("dsv4_decode_tp8_i512",   1,    256, 8, 7168, 512),
    ("dsv4_decode_tp4_i1024",  1,    256, 8, 7168, 1024),
    ("dsv4_decode_tp2_i2048",  1,    256, 8, 7168, 2048),

    # Mixtral tp=4 (num_experts=8, hidden=4096).
    ("mixtral_tp4_prefill",    512,  8,   2, 4096, 7168),
    ("mixtral_tp4_decode",     1,    8,   2, 4096, 7168),

    # Qwen2-MoE tp=4 (num_experts=64, hidden=3584).
    ("qwen_tp4_prefill",       512,  64,  8, 3584, 1280),
    ("qwen_tp4_decode",        1,    64,  8, 3584, 1280),
]


def _prepare_inputs(num_tokens, num_experts, topk, hidden, intermediate):
    torch.manual_seed(0)
    torch.xpu.manual_seed_all(0)

    a = torch.empty(num_tokens, hidden, dtype=torch.bfloat16, device="xpu").normal_(
        0, 0.01
    )

    # Build bf16 weights on CPU, quantize to MXFP4 there (avoids the
    # CPU-only quantize_mxfp4_2d from running on an XPU tensor).
    w1_bf16 = torch.empty(
        (num_experts, 2 * intermediate, hidden), dtype=torch.bfloat16
    ).normal_(0, 0.01)
    w2_bf16 = torch.empty(
        (num_experts, hidden, intermediate), dtype=torch.bfloat16
    ).normal_(0, 0.01)

    w1_packed_cpu, w1_scale_cpu = _quantize_weights_mxfp4_3d(w1_bf16)
    w2_packed_cpu, w2_scale_cpu = _quantize_weights_mxfp4_3d(w2_bf16)

    score = torch.randn(num_tokens, num_experts, dtype=torch.bfloat16, device="xpu")
    score = torch.softmax(score.float(), dim=-1)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.to(torch.bfloat16)

    return {
        "a": a,
        "w1_packed": w1_packed_cpu.to("xpu"),
        "w2_packed": w2_packed_cpu.to("xpu"),
        "w1_scale": w1_scale_cpu.to("xpu"),
        "w2_scale": w2_scale_cpu.to("xpu"),
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
        line_names=["bf16_dequant (Python dequant + bf16 GEMM)", "mxfp4_fused (tile-fused MXFP4)"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="Time (ms)",
        plot_name="fused-experts-mxfp4-e2e",
        args={},
    )
)
def benchmark(name, num_tokens, num_experts, topk, hidden, intermediate, provider):
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
        num_experts * (2 * intermediate + intermediate) * (hidden // 2 + hidden // MXFP4_BLOCK_SIZE)
    ) / 1024 / 1024

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
    pv["speedup"] = (
        pv[("ms", "bf16_dequant")] / pv[("ms", "mxfp4_fused")]
    ).round(2)
    pv["transient_saved_MB"] = (
        pv[("peak_transient_MB", "bf16_dequant")] - pv[("peak_transient_MB", "mxfp4_fused")]
    ).round(2)
    print(pv.to_markdown())
