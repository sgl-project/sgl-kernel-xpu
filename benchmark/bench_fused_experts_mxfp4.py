# End-to-end benchmark: fused_experts(use_mxfp4_w4a16=True) — two providers
# at MoE-layer granularity (both GEMMs + scatter + activation + combine):
#
#   mxfp4_fused     sgl_kernel fused_experts(use_mxfp4_w4a16=True):
#                   moe_grouped_mm_nt_xe20_mxfp4_w4a16 dequants per-tile in
#                   registers, no intermediate bf16 weight.
#   triton_fused    sglang Triton fused_experts_impl from the DeepSeek-V4 XPU
#                   support branch. Detects MXFP4-packed routed-expert weights
#                   (_is_mxfp4_xpu_packed), upcasts them to bf16 with a Triton
#                   dequant kernel (_upcast_mxfp4_triton), then runs the
#                   standard fused_moe_kernel. The bf16 weight is materialized
#                   as a transient, so its peak_transient_MB is higher than the
#                   tile-fused sgl_kernel path. Requires sglang to be installed
#                   in the environment (see project README for the XPU install
#                   recipe).
#
# Both compute the same result modulo MXFP4 rounding. ms and
# peak_transient_MB are directly comparable.
#
# Run:
#   python benchmark/bench_fused_experts_mxfp4.py                 # all providers
#   python benchmark/bench_fused_experts_mxfp4.py --cutlass-only  # skip Triton
#   python benchmark/bench_fused_experts_mxfp4.py --shape-filter <name> \
#       --json-out out.json                                       # single shape
#
# Per-shape + per-process invocation is sometimes necessary on BMG: both
# the sgl_kernel AOT libs and Triton JIT cache register modules with Level
# Zero, and at large N the driver's per-context module pool can run out
# mid-sweep. Fresh processes reset the pool.

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


def _import_triton_fused_experts_impl():
    # The triton_fused provider needs sglang's moe_runner/triton_utils path
    # with the DeepSeek-V4 XPU MXFP4 support: fused_experts_impl plus the
    # _is_mxfp4_xpu_packed detector that routes int8/uint8-packed weights
    # through the Triton MXFP4 upcast. Without it the packed weights would be
    # misread as real FP8 and the bench would compare against the wrong path,
    # so require this marker and refuse to fall back.
    from sglang.srt import server_args as _sa
    from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe as _fm
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        fused_experts_impl as _impl,
    )

    assert hasattr(
        _fm, "_is_mxfp4_xpu_packed"
    ), "Resolved sglang lacks MXFP4-XPU support (_is_mxfp4_xpu_packed) — wrong checkout"

    # fused_experts_impl + the Triton config picker read get_global_server_args()
    # at call time. Install a minimal stub holding only the flags this path reads
    # (a real ServerArgs would trigger HuggingFace model resolution).
    class _StubServerArgs:
        enable_deterministic_inference = False
        enable_fused_moe_sum_all_reduce = False

    _sa._global_server_args = _StubServerArgs()
    print(f"[triton_fused] using {_impl.__module__} from {_impl.__code__.co_filename}")
    return _impl


try:
    triton_fused_experts_impl = _import_triton_fused_experts_impl()
    TRITON_REF_AVAILABLE = True
except Exception as exc:
    triton_fused_experts_impl = None
    TRITON_REF_AVAILABLE = False
    print(f"[triton_fused provider disabled: {type(exc).__name__}: {exc}]")

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


# (name, num_tokens, num_experts, topk, hidden, intermediate). Only the
# sgl_kernel provider is tile-fused; the Triton reference upcasts to a bf16
# weight transient first (see provider notes at the top of the file).
#
# fused_experts picks a fused vs unfused GEMM1 activation by
#   use_unfused_act = avg_m <= 128 and (hidden*intermediate > 4096**2)
# where avg_m = (num_tokens*topk)//num_experts. Cover BOTH branches:
#   - dsv4_prefill_2k: real DSV4 prefill (E=256, H=4096, I=256) -> fused path
#   - dsv4_unfused:    small E + large H*I (E=8, H=4096, I=8192) -> unfused path
#     (M=128: avg_m = 128*6//8 = 96 <= 128, and H*I = 33.5M > 16.8M)
SHAPES = [
    ("dsv4_prefill_2k", 2048, 256, 6, 4096, 256),
    ("dsv4_unfused", 128, 8, 6, 4096, 8192),
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

    # Both sgl_kernel and the Triton reference accept N-outer scales [E, N, K/32].
    w1_scale_fp32 = torch.exp2(
        (w1_scale_cpu.to(torch.int32) - 127).to(torch.float32)
    ).contiguous()
    w2_scale_fp32 = torch.exp2(
        (w2_scale_cpu.to(torch.int32) - 127).to(torch.float32)
    ).contiguous()

    score = torch.randn(num_tokens, num_experts, dtype=torch.bfloat16, device="xpu")
    score = torch.softmax(score.float(), dim=-1)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.to(torch.bfloat16)
    # sglang's Triton fused_moe requires int32 topk_ids.
    topk_ids_i32 = topk_ids.to(torch.int32)

    return {
        "a": a,
        "w1_packed": w1_packed_cpu.view(torch.int8).to("xpu"),
        "w2_packed": w2_packed_cpu.view(torch.int8).to("xpu"),
        "w1_scale": w1_scale_fp32.to("xpu"),
        "w2_scale": w2_scale_fp32.to("xpu"),
        "topk_weight": topk_weight,
        "topk_ids": topk_ids,
        "topk_ids_i32": topk_ids_i32,
    }


# DSV4 swiglu clamp limit. Clamps the gate/up halves before silu_and_mul.
SWIGLU_LIMIT = 10


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
        w1_scale=inputs["w1_scale"],
        w2_scale=inputs["w2_scale"],
        swiglu_limit=SWIGLU_LIMIT,
    )


def _run_triton_fused(inputs):
    # sglang auto-detects MXFP4 inside fused_experts_impl from (w.dtype in
    # {uint8,int8}, w_scale present, not use_int4_w4a16) via _is_mxfp4_xpu_packed.
    # All quantization flags stay False — the packed E2M1 + E8M0 weights are
    # upcast to bf16 by a Triton dequant kernel before the standard fused_moe.
    return triton_fused_experts_impl(
        inputs["a"],
        inputs["w1_packed"].view(torch.uint8),
        inputs["w2_packed"].view(torch.uint8),
        inputs["topk_weight"],
        inputs["topk_ids_i32"],
        None,
        None,
        inplace=False,
        activation="silu",
        is_gated=True,
        apply_router_weight_on_input=False,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        w1_scale=inputs["w1_scale"],
        w2_scale=inputs["w2_scale"],
        swiglu_limit=SWIGLU_LIMIT,
    )


def _peak_transient_bytes(run_fn, inputs) -> int:
    torch.xpu.synchronize()
    torch.xpu.reset_peak_memory_stats()
    run_fn(inputs)
    torch.xpu.synchronize()
    return torch.xpu.max_memory_allocated()


_LINE_VALS = ["mxfp4_fused"]
_LINE_NAMES = ["mxfp4_fused (tile-fused MXFP4, CUTLASS)"]
_LINE_STYLES = [("green", "-")]
if TRITON_REF_AVAILABLE:
    _LINE_VALS.append("triton_fused")
    _LINE_NAMES.append(
        "triton_fused (Triton MXFP4 upcast + fused_moe, sglang DSV4-XPU)"
    )
    _LINE_STYLES.append(("red", "-"))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["name", "num_tokens", "num_experts", "topk", "hidden", "intermediate"],
        x_vals=SHAPES,
        line_arg="provider",
        line_vals=_LINE_VALS,
        line_names=_LINE_NAMES,
        styles=_LINE_STYLES,
        ylabel="Time (ms)",
        plot_name="fused-experts-mxfp4-e2e",
        args={},
    )
)
def benchmark(name, num_tokens, num_experts, topk, hidden, intermediate, provider):
    # Free any lingering tensors from the previous (shape, provider) run before
    # building a new set — otherwise triton_full's XPU bf16 weight copy can
    # stack on top of the previous run's allocations and OOM the device.
    gc.collect()
    torch.xpu.empty_cache()

    # Skip shapes that clearly won't fit before we try to run them.
    # BMG has 16 GB; leave headroom for activations + XPU runtime.
    RESIDENT_LIMIT_BYTES = 12 * 1024**3
    resident_bytes = (
        num_experts * 2 * intermediate * hidden  # packed w1 (int8)
        + num_experts * intermediate * hidden  # packed w2 (int8)
        + (num_experts * 2 * intermediate * hidden // 32) * 4  # w1 fp32 scales
        + (num_experts * intermediate * hidden // 32) * 4  # w2 fp32 scales
    )
    skip_reason = None
    if resident_bytes >= RESIDENT_LIMIT_BYTES:
        skip_reason = "resident weights too large"
    if skip_reason is not None:
        ALL_RESULTS.append(
            {
                "shape": name,
                "provider": provider,
                "num_tokens": num_tokens,
                "num_experts": num_experts,
                "topk": topk,
                "hidden": hidden,
                "intermediate": intermediate,
                "ms": float("nan"),
                "ms_min": float("nan"),
                "ms_max": float("nan"),
                "tflops": float("nan"),
                "peak_transient_MB": float("nan"),
                "weights_packed_MB": float("nan"),
            }
        )
        print(f"[skip {name}/{provider}: {skip_reason}]")
        return float("nan")

    inputs = _prepare_inputs(num_tokens, num_experts, topk, hidden, intermediate)
    if provider == "mxfp4_fused":
        run_fn = _run_fused
    elif provider == "triton_fused":
        run_fn = _run_triton_fused
    else:
        raise ValueError(f"unknown provider: {provider}")

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
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape-filter",
        default=None,
        help="Run only the shape with this name (see SHAPES).",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Dump per-row results as JSON instead of the markdown pivot.",
    )
    parser.add_argument(
        "--cutlass-only",
        action="store_true",
        help="Skip the Triton reference provider.",
    )
    args = parser.parse_args()

    if args.cutlass_only and "triton_fused" in _LINE_VALS:
        i = _LINE_VALS.index("triton_fused")
        _LINE_VALS.pop(i)
        _LINE_NAMES.pop(i)
        _LINE_STYLES.pop(i)

    if args.shape_filter is not None:
        filtered = [s for s in SHAPES if s[0] == args.shape_filter]
        if not filtered:
            raise SystemExit(f"no shape matches --shape-filter={args.shape_filter}")
        # perf_report captured x_vals=SHAPES by reference, so mutate in place.
        SHAPES[:] = filtered

    benchmark.run(print_data=False)

    if args.json_out is not None:
        with open(args.json_out, "w") as f:
            json.dump(ALL_RESULTS, f, indent=2)
        print(f"[wrote {len(ALL_RESULTS)} rows -> {args.json_out}]")
        raise SystemExit(0)

    print("\nBenchmark finished!\n")
    import pandas as pd

    df = pd.DataFrame(ALL_RESULTS)
    pv = df.pivot_table(
        index=["shape", "num_tokens", "num_experts", "topk", "hidden", "intermediate"],
        columns="provider",
        values=["ms", "tflops", "peak_transient_MB"],
        aggfunc="first",
    )
    # Compare the Triton reference against mxfp4_fused where both ran.
    if ("ms", "mxfp4_fused") in pv.columns and ("ms", "triton_fused") in pv.columns:
        pv["speedup_vs_triton_fused"] = (
            pv[("ms", "triton_fused")] / pv[("ms", "mxfp4_fused")]
        ).round(2)
        pv["transient_saved_vs_triton_fused_MB"] = (
            pv[("peak_transient_MB", "triton_fused")]
            - pv[("peak_transient_MB", "mxfp4_fused")]
        ).round(2)
    print(pv.to_markdown())
