# Op-level benchmark for the MXFP4 MoE grouped GEMM. Three providers:
#
#   bf16_dequant    Python-loop dequant (dequantize_mxfp4_weights) → bf16
#                   grouped GEMM (moe_grouped_mm_nt_xe20). The legacy
#                   sgl-kernel path in fused_experts(use_mxfp4_w4a16=True).
#   triton_full     sglang Triton path: _upcast_mxfp4_triton for dequant
#                   followed by the Triton fused_moe_kernel for the GEMM.
#                   Same code paths the sglang Triton runtime uses.
#   mxfp4_fused     moe_grouped_mm_nt_xe20_mxfp4_w4a16 directly on packed
#                   weights + fp32 scales — tile-level dequant, no bf16
#                   weight ever materialized.
#
# All three produce the same result modulo MXFP4 rounding. peak_transient
# captures the bf16 weight materialization the first two providers
# require and the fused path avoids.
#
# Run:
#   python benchmark/bench_moe_mxfp4_gemm.py

import importlib.machinery
import sys
import types
from pathlib import Path

import sgl_kernel  # noqa: F401 — registers the torch.ops.sgl_kernel namespace
import torch
import triton
import triton.language as tl
from sgl_kernel.moe import dequantize_mxfp4_weights

# The CPU MXFP4 quantize/dequantize helpers live next to the tests.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from mxfp4_utils import MXFP4_BLOCK_SIZE  # noqa: E402
from mxfp4_utils import quantize_mxfp4_2d  # noqa: E402


# Importing sglang on an XPU-only host trips over flashinfer.comm (CUDA).
# Stub it before the sglang import chain runs. Same trick as in
# bench_fused_experts_mxfp4.py.
def _import_triton_symbols():
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

    # The Triton MoE config picker reads a global ServerArgs. Install a stub.
    from sglang.srt import server_args as _sa
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import _upcast_mxfp4_triton
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_config import (
        try_get_optimal_moe_config,
    )
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_kernels import (
        invoke_fused_moe_kernel,
    )
    from sglang.srt.layers.moe.fused_moe_triton.moe_align_block_size import (
        moe_align_block_size,
    )

    class _StubServerArgs:
        enable_deterministic_inference = False

    _sa._global_server_args = _StubServerArgs()

    return (
        _upcast_mxfp4_triton,
        invoke_fused_moe_kernel,
        moe_align_block_size,
        try_get_optimal_moe_config,
    )


try:
    (
        _upcast_mxfp4_triton,
        _invoke_fused_moe_kernel,
        _moe_align_block_size,
        _try_get_optimal_moe_config,
    ) = _import_triton_symbols()
    TRITON_REF_AVAILABLE = True
except Exception as exc:
    _upcast_mxfp4_triton = None
    _invoke_fused_moe_kernel = None
    _moe_align_block_size = None
    _try_get_optimal_moe_config = None
    TRITON_REF_AVAILABLE = False
    print(f"[triton_full provider disabled: {type(exc).__name__}: {exc}]")

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
    scales_fp32 = torch.exp2(
        (scales_u8.to(torch.int32) - 127).to(torch.float32)
    ).reshape(E, N, K // MXFP4_BLOCK_SIZE)
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

    # Triton fused_moe_kernel expects MoE dispatch metadata. Synthesize
    # topk=1 ids that assign each contiguous avg_m-row block to one
    # expert — semantically the same dispatch moe_grouped_mm_nt_xe20 runs.
    topk_ids = torch.arange(NUM_EXPERTS, dtype=torch.int32, device="xpu")
    topk_ids = topk_ids.repeat_interleave(avg_m).reshape(total_m, 1)
    topk_weights = torch.ones((total_m, 1), dtype=torch.float32, device="xpu")

    return {
        "activations": a,
        "total_rows": total_rows,
        "packed_xpu": packed_xpu,
        "scales_xpu": scales_xpu,
        "output": output,
        "total_m": total_m,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "gemm_n": gemm_n,
        "gemm_k": gemm_k,
        "avg_m": avg_m,
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


def _run_triton_full(inputs):
    # Full sglang Triton path: Triton dequant → Triton fused_moe_kernel.
    # Each expert's `avg_m` rows land in one contiguous block of output,
    # which matches moe_grouped_mm_nt_xe20's semantics (topk=1, no combine).
    w_bf16 = _upcast_mxfp4_triton(
        inputs["packed_xpu"].view(torch.uint8),
        inputs["scales_xpu"],
        torch.bfloat16,
    )

    # Pick a Triton MoE config for this (E, N, K, M) shape.
    config, _ = _try_get_optimal_moe_config(
        w_bf16.shape,
        w_bf16.shape,
        top_k=1,
        dtype="bfloat16",
        M=inputs["total_m"],
        block_shape=None,
        per_channel_quant=False,
        return_down_config=True,
    )
    sorted_token_ids, expert_ids, num_tokens_post_padded = _moe_align_block_size(
        inputs["topk_ids"], config["BLOCK_SIZE_M"], NUM_EXPERTS
    )

    _invoke_fused_moe_kernel(
        inputs["activations"],
        w_bf16,
        None,  # bias
        inputs["output"].unsqueeze(0),  # (1, total_m, N) for topk=1 no_combine
        None,  # a_scale
        None,  # b_scale
        None,  # b_zp
        inputs["topk_weights"],
        inputs["topk_ids"],
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight=False,
        top_k=1,
        config=config,
        compute_type=tl.bfloat16,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
    )


def _run_mxfp4_fused(inputs):
    torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20_mxfp4_w4a16(
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


_LINE_VALS = ["bf16_dequant", "mxfp4_fused"]
_LINE_NAMES = ["bf16_dequant (Python dequant)", "mxfp4_fused (tile-level dequant)"]
_LINE_STYLES = [("blue", "-"), ("green", "-")]
if TRITON_REF_AVAILABLE:
    _LINE_VALS.insert(1, "triton_full")
    _LINE_NAMES.insert(1, "triton_full (Triton dequant + Triton fused_moe_kernel)")
    _LINE_STYLES.insert(1, ("red", "-"))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["avg_m", "gemm_n", "gemm_k"],
        x_vals=BENCH_SHAPES,
        line_arg="provider",
        line_vals=_LINE_VALS,
        line_names=_LINE_NAMES,
        styles=_LINE_STYLES,
        ylabel="Time (ms)",
        plot_name="moe-mxfp4-gemm-op-level",
        args={},
    )
)
def benchmark(avg_m, gemm_n, gemm_k, provider):
    inputs = _prepare_inputs(avg_m, gemm_n, gemm_k)
    if provider == "bf16_dequant":
        run_fn = _run_bf16_dequant
    elif provider == "triton_full":
        run_fn = _run_triton_full
    elif provider == "mxfp4_fused":
        run_fn = _run_mxfp4_fused
    else:
        raise ValueError(f"unknown provider: {provider}")

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

    # Effective B-side bandwidth the GEMM reads. bf16_dequant and
    # triton_dequant both hand a bf16 weight tensor to the GEMM, so the
    # GEMM's B-side read is 2 B/elem. mxfp4_fused consumes packed MXFP4
    # (0.5 B/elem) plus fp32 scales (4 B per MXFP4_BLOCK_SIZE elements).
    if provider == "mxfp4_fused":
        b_bytes = NUM_EXPERTS * gemm_n * gemm_k * (1 / 2 + 4 / MXFP4_BLOCK_SIZE)
    else:
        b_bytes = NUM_EXPERTS * gemm_n * gemm_k * 2

    # Both dequant-then-GEMM providers materialize the bf16 weight
    # transiently; mxfp4_fused never does.
    packed_bytes = (
        NUM_EXPERTS * gemm_n * (gemm_k // 2 + 4 * (gemm_k // MXFP4_BLOCK_SIZE))
    )
    bf16_transient = NUM_EXPERTS * gemm_n * gemm_k * 2
    weights_resident_bytes = packed_bytes
    transient_bf16_bytes = (
        bf16_transient if provider in ("bf16_dequant", "triton_full") else 0
    )

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
    # Compare each reference provider against mxfp4_fused where both ran.
    if ("ms", "mxfp4_fused") in pv.columns:
        for ref in ("bf16_dequant", "triton_full"):
            if ("ms", ref) in pv.columns:
                pv[f"speedup_fused_vs_{ref}"] = (
                    pv[("ms", ref)] / pv[("ms", "mxfp4_fused")]
                ).round(2)
    # Both dequant-then-GEMM providers materialize the bf16 weight tensor
    # transiently; mxfp4_fused never does. Report the savings once.
    if ("transient_bf16_MB", "bf16_dequant") in pv.columns:
        pv["transient_bf16_saved_MB"] = pv[("transient_bf16_MB", "bf16_dequant")].round(
            2
        )
    print(pv.to_markdown())
