# End-to-end benchmark: fused_experts(use_mxfp4_w4a16=True) — three providers
# at MoE-layer granularity (both GEMMs + scatter + activation + combine):
#
#   bf16_dequant    sgl_kernel path with use_fused_mxfp4_kernel=False:
#                   dequantize_mxfp4_weights materializes bf16 weights on
#                   XPU, then moe_grouped_mm_nt_xe20.
#   mxfp4_fused     sgl_kernel path with use_fused_mxfp4_kernel=True:
#                   moe_grouped_mm_nt_xe20_mxfp4 dequants per-tile in
#                   registers, no intermediate bf16 weight.
#   triton_full     sglang Triton fused_experts_impl (MXFP4-xpu branch):
#                   one Triton kernel per weight upcasts MXFP4 → bf16,
#                   then the Triton fused-moe GEMM runs on bf16.
#
# All three compute the same result modulo MXFP4 rounding. ms and
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
import importlib.machinery
import sys
import types
from pathlib import Path

import sgl_kernel  # noqa: F401 — registers torch.ops.sgl_kernel
import torch
import triton
from sgl_kernel import fused_experts

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from mxfp4_utils import MXFP4_BLOCK_SIZE  # noqa: E402
from mxfp4_utils import quantize_mxfp4_2d  # noqa: E402


# Importing sglang on an XPU-only host trips over flashinfer.comm (CUDA)
# and a DSV4 env default that asserts swiglu_limit. Stub both before the
# sglang import chain runs.
def _import_triton_fused_experts_impl():
    import os

    os.environ.setdefault("SGLANG_DSV4_2604_SUBMODE", "2604A")

    if "flashinfer.comm" not in sys.modules:
        fi = sys.modules.get("flashinfer") or types.ModuleType("flashinfer")
        fi.__spec__ = importlib.machinery.ModuleSpec(
            "flashinfer", loader=None, is_package=True
        )
        if not hasattr(fi, "__path__"):
            fi.__path__ = []
        fi_comm = types.ModuleType("flashinfer.comm")
        fi_comm.__spec__ = importlib.machinery.ModuleSpec(
            "flashinfer.comm", loader=None, is_package=False
        )

        class _Stub:
            def __init__(self, *a, **k):
                pass

            def __getattr__(self, n):
                return _Stub()

            def __call__(self, *a, **k):
                return _Stub()

        fi_comm.MoeAlltoAll = _Stub
        fi_comm.moe_a2a_get_workspace_size_per_rank = _Stub
        sys.modules["flashinfer"] = fi
        sys.modules["flashinfer.comm"] = fi_comm

    # The Triton MoE config picker consults a global ServerArgs. Install
    # a minimal stub (a real one would trigger HuggingFace model resolution).
    from sglang.srt import server_args as _sa
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
        fused_experts_impl as _impl,
    )

    class _StubServerArgs:
        enable_deterministic_inference = False

    _sa._global_server_args = _StubServerArgs()
    return _impl


try:
    triton_fused_experts_impl = _import_triton_fused_experts_impl()
    TRITON_REF_AVAILABLE = True
except Exception as exc:
    triton_fused_experts_impl = None
    TRITON_REF_AVAILABLE = False
    print(f"[triton_full provider disabled: {type(exc).__name__}: {exc}]")

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


# (name, num_tokens, num_experts, topk, hidden, intermediate). Default is a
# real DSV4 MXFP4 fused_experts call (E=256, H=4096, I=256, topk=6). Both
# the bf16_dequant and triton_full providers also need to fit a bf16
# weight transient (E*2I*H*2B); the fused path avoids it entirely.
SHAPES = [
    ("dsv4_prefill_2k", 2048, 256, 6, 4096, 256),
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


def _run_triton_full(inputs):
    # use_fp8_w8a8=True is how _is_mxfp4_xpu_packed detects MXFP4-packed
    # routed experts on this branch (see sglang fused_moe.py).
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
        use_fp8_w8a8=True,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        w1_scale=inputs["w1_scale"],
        w2_scale=inputs["w2_scale"],
    )


def _peak_transient_bytes(run_fn, inputs) -> int:
    torch.xpu.synchronize()
    torch.xpu.reset_peak_memory_stats()
    run_fn(inputs)
    torch.xpu.synchronize()
    return torch.xpu.max_memory_allocated()


# bf16_dequant is available but not enabled by default: at the shapes we
# care about it's orders of magnitude slower than the other two and not
# the comparison we care about. Re-add "bf16_dequant" here if you want it.
_LINE_VALS = ["mxfp4_fused"]
_LINE_NAMES = ["mxfp4_fused (tile-fused MXFP4, CUTLASS)"]
_LINE_STYLES = [("green", "-")]
if TRITON_REF_AVAILABLE:
    _LINE_VALS.append("triton_full")
    _LINE_NAMES.append("triton_full (Triton dequant + Triton fused_moe)")
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
    # building a new set — otherwise the unfused path's XPU bf16 weight copy
    # can stack on top of the previous run's allocations and OOM the device.
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
    if provider == "bf16_dequant":
        run_fn = _run_unfused
    elif provider == "mxfp4_fused":
        run_fn = _run_fused
    elif provider == "triton_full":
        run_fn = _run_triton_full
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

    if args.cutlass_only and "triton_full" in _LINE_VALS:
        i = _LINE_VALS.index("triton_full")
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
    # Compare each reference provider against mxfp4_fused where both ran.
    for ref in ("bf16_dequant", "triton_full"):
        if ("ms", "mxfp4_fused") in pv.columns and ("ms", ref) in pv.columns:
            pv[f"speedup_vs_{ref}"] = (
                pv[("ms", ref)] / pv[("ms", "mxfp4_fused")]
            ).round(2)
            pv[f"transient_saved_vs_{ref}_MB"] = (
                pv[("peak_transient_MB", ref)]
                - pv[("peak_transient_MB", "mxfp4_fused")]
            ).round(2)
    print(pv.to_markdown())
