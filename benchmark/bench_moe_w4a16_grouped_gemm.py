# Op-level benchmark for the 4-bit W4A16 MoE grouped GEMM. Compares SGLang's
# sgl_kernel and vLLM's XPU grouped-GEMM integrations on identical shapes and
# identical logical weights. Both use the Xe2 tiled CUTLASS/CuTe-style W4A16
# kernel family; the sgl_kernel path extends its quantized-weight conventions.
# Providers:
#
#   mxfp4_fused     sgl_kernel moe_grouped_mm_nt_xe20_w4a16 on MXFP4 packed
#                   weights + raw uint8 E8M0 scales.
#   int4_fused      Same-geometry alternative quantization kernel comparison:
#                   the sgl_kernel grouped-GEMM path on symmetric signed INT4
#                   packed weights + per-group bf16 scales.
#   vllm_mxfp4      vLLM cutlass_grouped_gemm_xe2 on the same MXFP4 packed
#                   weights (viewed as float4_e2m1fn_x2) + uint8 E8M0 scales.
#   vllm_int4       Same-geometry alternative quantization kernel comparison:
#                   vLLM cutlass_grouped_gemm_xe2 on equivalent INT4 weights
#                   in uint4 / zero-point-8 representation, folded to signed
#                   s4 via implement_zp, plus per-group bf16 scales.
#
# The sgl and vLLM providers of the same recipe consume the same logical
# weights, so the __main__ correctness pass cross-checks their outputs.
# Both packages must be importable in one env (vllm_xpu_kernels alongside
# sgl_kernel); see the repo setup notes.
#
# Model shapes represent one rank of an 8-device SGLang deployment with
# tp_size=8 and ep_size=8. In this mode moe_tp_size=1, so each rank keeps the
# full intermediate width and stores one eighth of the experts. The quick set
# includes GPT-OSS geometry (4 local experts, hidden=2880, intermediate=2880).
# Set SGL_MOE_BENCH_FULL_SHAPES=1 to also include DeepSeek-V4 geometry (32 local
# experts, hidden=4096, intermediate=2048) and higher-load model points.
#
# Run:
#   python benchmark/bench_moe_w4a16_grouped_gemm.py

import gc
import math
import os
import sys
from pathlib import Path

import sgl_kernel  # noqa: F401 — registers the torch.ops.sgl_kernel namespace
import torch
import triton

# The CPU MXFP4 quantize/dequantize helpers live next to the tests.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from mxfp4_utils import MXFP4_BLOCK_SIZE  # noqa: E402
from mxfp4_utils import quantize_mxfp4_2d  # noqa: E402


def _import_vllm_grouped_gemm():
    import vllm_xpu_kernels._moe_C  # noqa: F401 — registers torch.ops._moe_C
    import vllm_xpu_kernels._xpu_C  # noqa: F401 — registers torch.ops._xpu_C
    from vllm_xpu_kernels.fused_moe_interface import cutlass_grouped_gemm_xe2

    return cutlass_grouped_gemm_xe2


try:
    _cutlass_grouped_gemm_xe2 = _import_vllm_grouped_gemm()
    VLLM_REF_AVAILABLE = True
except Exception as exc:
    _cutlass_grouped_gemm_xe2 = None
    VLLM_REF_AVAILABLE = False
    print(f"[vllm provider disabled: {type(exc).__name__}: {exc}]")

QUICK_BENCH_SHAPES = [
    # (num_experts, avg_m_per_expert, gemm_n, gemm_k)  # total_m = experts*avg_m
    # Controlled sweep across the four avg_m dispatch policies.
    (8, 4, 1024, 1024),
    (8, 8, 1024, 1024),
    (8, 33, 1024, 1024),
    (8, 129, 1024, 1024),
    # GPT-OSS-20B, TP=8/EP=8. avg_m=4 corresponds to 32 tokens under
    # perfectly uniform routing: tokens * topk / global_experts = 32*4/32.
    (4, 4, 5760, 2880),  # gemm1: [E, 2*intermediate, hidden]
    (4, 4, 2880, 2880),  # gemm2: [E, hidden, intermediate]
]

FULL_BENCH_SHAPES = QUICK_BENCH_SHAPES + [
    # (num_experts, avg_m_per_expert, gemm_n, gemm_k)  # total_m = experts*avg_m
    # N/K sensitivity at representative medium and large avg_m policies.
    (8, 33, 2048, 1024),
    (8, 33, 1024, 2048),
    (8, 129, 2048, 2048),
    # DeepSeek-V4 routed-expert GEMMs (32 of 256 experts per TP=8/EP=8 rank,
    # intermediate=2048, topk=6). gemm1 weight is [E, 2*inter, hidden] ->
    # N=4096, K=4096; gemm2 weight is [E, hidden, inter] -> N=4096, K=2048.
    # avg_m is forced equal per expert: these are controlled load points, not
    # a real routing distribution. avg_m=1 approximates a uniformly routed
    # 43-token batch (43*6/256 ~= 1); avg_m=48 corresponds to 2048 tokens.
    (32, 1, 4096, 4096),  # dsv4 gemm1, EP=8 rank, avg_m=1
    (32, 1, 4096, 2048),  # dsv4 gemm2, EP=8 rank, avg_m=1
    (32, 48, 4096, 4096),  # dsv4 gemm1, EP=8 rank, avg_m=48
    (32, 48, 4096, 2048),  # dsv4 gemm2, EP=8 rank, avg_m=48
    # GPT-OSS-20B higher-load point. avg_m=256 corresponds to 2048 tokens under
    # uniform routing (2048*4/32 = 256), matching the fused benchmark's prefill.
    (4, 256, 5760, 2880),  # gpt-oss gemm1, EP=8 rank, avg_m=256
    (4, 256, 2880, 2880),  # gpt-oss gemm2, EP=8 rank, avg_m=256
]

RUN_FULL_SHAPES = os.environ.get("SGL_MOE_BENCH_FULL_SHAPES") == "1"
BENCH_SHAPES = FULL_BENCH_SHAPES if RUN_FULL_SHAPES else QUICK_BENCH_SHAPES
print(
    f"[config] grouped-GEMM shape set: "
    f"{'full' if RUN_FULL_SHAPES else 'quick'} ({len(BENCH_SHAPES)} shapes)",
    flush=True,
)


ALL_RESULTS = []
_MXFP4_CPU_WEIGHT_CACHE = {}


def _quantize_bf16_weights_mxfp4(num_experts: int, N: int, K: int):
    """Draw + quantize per expert: -> ([E, N, K/2] int8 packed,
    [E, N, K/32] uint8 E8M0).

    Quantizes one expert at a time. Building the whole [E, N, K] bf16 tensor
    and calling `.float()` on it at once needs tens of GB of host RAM at the
    real 256-expert / N=4096/K=4096 shapes and trips the OOM killer. Per-expert
    normal_ draws consume the RNG in the same row-major order as one big draw,
    so the resulting weights are bit-identical to the naive version.
    """
    packed = torch.empty((num_experts, N, K // 2), dtype=torch.int8)
    scales = torch.empty((num_experts, N, K // MXFP4_BLOCK_SIZE), dtype=torch.uint8)
    for e in range(num_experts):
        w_e = torch.empty((N, K), dtype=torch.bfloat16).normal_(0, 0.01)
        p_e, s_e = quantize_mxfp4_2d(w_e.float(), MXFP4_BLOCK_SIZE)
        packed[e].copy_(p_e.view(torch.int8))
        scales[e].copy_(s_e)
        del w_e, p_e, s_e
    return packed, scales


def _get_mxfp4_cpu_weights(num_experts: int, N: int, K: int):
    """Return one cached MXFP4 source for the current grouped-GEMM shape.

    The cache holds one shape only, so it removes duplicate quantization for
    adjacent SGL/vLLM MXFP4 providers without accumulating large model weights.
    """
    key = (num_experts, N, K)
    cached = _MXFP4_CPU_WEIGHT_CACHE.get(key)
    if cached is not None:
        print(f"[weights] MXFP4 CPU cache hit: E={num_experts}, N={N}, K={K}", flush=True)
        return cached
    print(
        f"[weights] MXFP4 CPU cache miss; quantizing: "
        f"E={num_experts}, N={N}, K={K}",
        flush=True,
    )
    _MXFP4_CPU_WEIGHT_CACHE.clear()
    cached = _quantize_bf16_weights_mxfp4(num_experts, N, K)
    _MXFP4_CPU_WEIGHT_CACHE[key] = cached
    return cached


# INT4 group size for the int4_fused provider. Matches the MXFP4 block size so
# both recipes read the same number of scale groups per row.
INT4_GROUP_SIZE = 32
CORRECTNESS_REL_TOL = 1e-2


def _quantize_bf16_weights_int4_symmetric(num_experts: int, N: int, K: int, group_size: int):
    """Draw + quantize per expert: -> ([E, N, K/2] int8 packed signed nibbles,
    [E, N, K/group_size] bf16 per-group scales).

    Symmetric INT4 (no zero-point): each group's scale is amax/7 and codes are
    signed 4-bit two's-complement nibbles in [-8, 7]. This is the layout the
    kernel expects when is_int4=True and zeros=None. Per-expert draws keep host
    RAM bounded (see _quantize_bf16_weights_mxfp4).
    """
    assert K % group_size == 0
    num_groups = K // group_size
    packed = torch.empty((num_experts, N, K // 2), dtype=torch.int8)
    scales = torch.empty((num_experts, N, num_groups), dtype=torch.bfloat16)
    for e in range(num_experts):
        w = (
            torch.empty((N, K), dtype=torch.bfloat16)
            .normal_(0, 0.01)
            .float()
            .reshape(N, num_groups, group_size)
        )
        amax = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        s = amax / 7.0
        codes = torch.round(w / s).clamp_(-8, 7).to(torch.int16).reshape(N, K)
        nibbles = torch.bitwise_and(codes, 0xF)
        p = (nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)).to(torch.uint8)
        packed[e].copy_(p.view(torch.int8))
        scales[e].copy_(s.reshape(N, num_groups).to(torch.bfloat16))
        del w, amax, s, codes, nibbles, p
    return packed, scales


def _vllm_implement_zp(qweight: torch.Tensor) -> torch.Tensor:
    """vLLM's uint4 -> signed-s4 fold (matches XpuFusedMoe.implement_zp).

    Converts unsigned nibble codes in [0, 15] (zero-point 8) into vLLM's
    sign-magnitude s4 packing that its grouped-GEMM kernel expects. Returns an
    int8 tensor with the same [.., K/2] shape.
    """
    assert qweight.dtype == torch.uint8
    high_u4 = (qweight >> 4) & 0x0F
    low_u4 = qweight & 0x0F
    high_s8 = high_u4.to(torch.int8) - 8
    low_s8 = low_u4.to(torch.int8) - 8

    def _pack(x):
        sign = (x < 0).to(torch.uint8)
        abs_low3 = (x.view(torch.uint8) & 0x7).to(torch.uint8)
        return (sign << 3) | abs_low3

    return ((_pack(high_s8) << 4) | _pack(low_s8)).view(torch.int8)


def _quantize_bf16_weights_int4_vllm(num_experts: int, N: int, K: int, group_size: int):
    """Draw + quantize per expert: -> ([E, N, K/2] int8 vLLM-folded s4 codes,
    [E, N, K/group_size] bf16 per-group scales).

    Same symmetric quantization as _quantize_bf16_weights_int4_symmetric
    (scale = amax/7, values in [-8, 7]) but re-encoded into vLLM's uint4 /
    zero-point-8 packing and then folded with implement_zp, so both the sgl
    and vLLM INT4 providers dequantize to the same weights. Per-expert draws
    keep host RAM bounded.
    """
    assert K % group_size == 0
    num_groups = K // group_size
    packed = torch.empty((num_experts, N, K // 2), dtype=torch.int8)
    scales = torch.empty((num_experts, N, num_groups), dtype=torch.bfloat16)
    for e in range(num_experts):
        w = (
            torch.empty((N, K), dtype=torch.bfloat16)
            .normal_(0, 0.01)
            .float()
            .reshape(N, num_groups, group_size)
        )
        amax = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        s = amax / 7.0
        # uint4 codes with zero-point 8: dequant = (code - 8) * scale.
        codes = (
            (torch.round(w / s).clamp_(-8, 7) + 8).to(torch.int32).reshape(N, K)
        )
        p_u8 = (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)
        packed[e].copy_(_vllm_implement_zp(p_u8))
        scales[e].copy_(s.reshape(N, num_groups).to(torch.bfloat16))
        del w, amax, s, codes, p_u8
    return packed, scales


def _prepare_inputs(
    num_experts: int,
    avg_m: int,
    gemm_n: int,
    gemm_k: int,
    recipe: str = "mxfp4",
    backend: str = "sgl",
):
    """Build the XPU tensors for one provider.

    `recipe` selects the weight encoding and `backend` selects the packing
    layout of that encoding:
      - recipe "mxfp4": E2M1 packed + uint8 E8M0 scales. The packed bytes and
        scales are identical for the sgl and vLLM backends (vLLM just views
        the weight as float4_e2m1fn_x2 at call time).
      - recipe "int4", backend "sgl":  signed INT4 packed + per-group bf16
        scales (symmetric, zeros=None).
      - recipe "int4", backend "vllm": the same weights re-encoded in vLLM's
        uint4 / zero-point-8 packing folded to s4 + per-group bf16 scales.
    The bf16 source weights are seeded identically, so the sgl and vLLM
    providers of a recipe operate on the same logical weights.
    """
    print(
        f"[setup] provider={backend}/{recipe}: "
        f"E={num_experts}, avg_m={avg_m}, N={gemm_n}, K={gemm_k}",
        flush=True,
    )
    torch.manual_seed(0)
    torch.xpu.manual_seed_all(0)

    total_m = num_experts * avg_m
    total_rows = torch.full((num_experts,), avg_m, dtype=torch.int32, device="xpu")

    a = torch.empty((total_m, gemm_k), dtype=torch.bfloat16, device="xpu").normal_(
        0, 0.01
    )

    if recipe == "int4":
        if backend == "vllm":
            packed_cpu, scales_cpu = _quantize_bf16_weights_int4_vllm(
                num_experts, gemm_n, gemm_k, INT4_GROUP_SIZE
            )
        else:
            packed_cpu, scales_cpu = _quantize_bf16_weights_int4_symmetric(
                num_experts, gemm_n, gemm_k, INT4_GROUP_SIZE
            )
        group_size = INT4_GROUP_SIZE
        is_int4 = True
    else:
        packed_cpu, scales_cpu = _get_mxfp4_cpu_weights(
            num_experts, gemm_n, gemm_k
        )
        group_size = MXFP4_BLOCK_SIZE
        is_int4 = False

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
        "gemm_n": gemm_n,
        "gemm_k": gemm_k,
        "avg_m": avg_m,
        "num_experts": num_experts,
        "group_size": group_size,
        "is_int4": is_int4,
        "recipe": recipe,
        "backend": backend,
    }


def _run_vllm_mxfp4(inputs):
    # vLLM CUTLASS grouped GEMM on the same MXFP4 packed weights as
    # mxfp4_fused. vLLM reads the packed bytes as float4_e2m1fn_x2 and the
    # uint8 E8M0 scales unchanged.
    _cutlass_grouped_gemm_xe2(
        inputs["activations"],
        inputs["packed_xpu"].view(torch.uint8).view(torch.float4_e2m1fn_x2),
        inputs["scales_xpu"],
        None,  # bias
        inputs["output"],
        inputs["total_rows"],
        inputs["gemm_n"],
        inputs["gemm_k"],
        inputs["num_experts"],
    )


def _run_vllm_int4(inputs):
    # vLLM CUTLASS grouped GEMM on INT4 weights in vLLM's folded s4 packing
    # (built in _prepare_inputs) + per-group bf16 scales.
    _cutlass_grouped_gemm_xe2(
        inputs["activations"],
        inputs["packed_xpu"],
        inputs["scales_xpu"],
        None,  # bias
        inputs["output"],
        inputs["total_rows"],
        inputs["gemm_n"],
        inputs["gemm_k"],
        inputs["num_experts"],
    )


def _run_mxfp4_fused(inputs):
    torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20_w4a16(
        inputs["output"],
        inputs["activations"],
        inputs["packed_xpu"],
        inputs["scales_xpu"],
        None,  # zeros
        None,  # bias
        inputs["total_rows"],
        inputs["num_experts"],
        False,  # is_int4
        MXFP4_BLOCK_SIZE,
    )


def _run_int4_fused(inputs):
    # Same grouped-GEMM kernel as mxfp4_fused, is_int4=True. Symmetric INT4:
    # signed packed nibbles + per-group bf16 scales, zeros=None. Like the mxfp4
    # path it dequants per tile and never materializes a bf16 weight.
    torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20_w4a16(
        inputs["output"],
        inputs["activations"],
        inputs["packed_xpu"],
        inputs["scales_xpu"],
        None,  # zeros (symmetric)
        None,  # bias
        inputs["total_rows"],
        inputs["num_experts"],
        True,  # is_int4
        inputs["group_size"],
    )


def _peak_extra_allocated_bytes(run_fn, inputs) -> int:
    """Measure the peak allocation increase above the prepared inputs."""
    torch.xpu.synchronize()
    baseline = torch.xpu.memory_allocated()
    torch.xpu.reset_peak_memory_stats()
    run_fn(inputs)
    torch.xpu.synchronize()
    return max(0, torch.xpu.max_memory_allocated() - baseline)


_LINE_VALS = ["mxfp4_fused", "int4_fused"]
_LINE_NAMES = [
    "mxfp4_fused (sgl_kernel Xe2 grouped GEMM)",
    "int4_fused (sgl_kernel Xe2 grouped GEMM, symmetric INT4)",
]
_LINE_STYLES = [("green", "-"), ("blue", "-")]
if VLLM_REF_AVAILABLE:
    _LINE_VALS = ["mxfp4_fused", "vllm_mxfp4", "int4_fused", "vllm_int4"]
    _LINE_NAMES += [
        "vllm_mxfp4 (vLLM Xe2 grouped GEMM)",
        "vllm_int4 (vLLM Xe2 grouped GEMM)",
    ]
    _LINE_NAMES = [
        "mxfp4_fused (sgl_kernel Xe2 grouped GEMM)",
        "vllm_mxfp4 (vLLM Xe2 grouped GEMM)",
        "int4_fused (sgl_kernel Xe2 grouped GEMM, symmetric INT4)",
        "vllm_int4 (vLLM Xe2 grouped GEMM)",
    ]
    _LINE_STYLES = [("green", "-"), ("red", "-"), ("blue", "-"), ("orange", "-")]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_experts", "avg_m", "gemm_n", "gemm_k"],
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
def benchmark(num_experts, avg_m, gemm_n, gemm_k, provider):
    recipe = "int4" if provider in ("int4_fused", "vllm_int4") else "mxfp4"
    backend = "vllm" if provider.startswith("vllm_") else "sgl"
    inputs = _prepare_inputs(num_experts, avg_m, gemm_n, gemm_k, recipe, backend)
    print(f"[run] timing provider={provider}", flush=True)
    if provider == "mxfp4_fused":
        run_fn = _run_mxfp4_fused
    elif provider == "int4_fused":
        run_fn = _run_int4_fused
    elif provider == "vllm_mxfp4":
        run_fn = _run_vllm_mxfp4
    elif provider == "vllm_int4":
        run_fn = _run_vllm_int4
    else:
        raise ValueError(f"unknown provider: {provider}")

    # Warm up.
    for _ in range(5):
        run_fn(inputs)
    torch.xpu.synchronize()

    # Peak allocator growth above the prepared inputs during one call. No
    # provider materializes a bf16 weight; this contrasts backend workspaces.
    peak_extra_allocated = _peak_extra_allocated_bytes(run_fn, inputs)

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

    # Effective B-side bandwidth the GEMM reads. Both backends read packed
    # 4-bit weights + per-group scales (no bf16 weight is ever materialized).
    # MXFP4: packed (0.5 B/elem) + uint8 E8M0 scales (1 B / MXFP4_BLOCK_SIZE).
    # INT4:  packed (0.5 B/elem) + bf16 scales (2 B / INT4_GROUP_SIZE).
    if recipe == "mxfp4":
        b_bytes = num_experts * gemm_n * gemm_k * (1 / 2 + 1 / MXFP4_BLOCK_SIZE)
    else:
        b_bytes = num_experts * gemm_n * gemm_k * (1 / 2 + 2 / INT4_GROUP_SIZE)

    if recipe == "int4":
        packed_bytes = (
            num_experts * gemm_n * (gemm_k // 2 + (gemm_k // INT4_GROUP_SIZE) * 2)
        )
    else:
        packed_bytes = (
            num_experts * gemm_n * (gemm_k // 2 + gemm_k // MXFP4_BLOCK_SIZE)
        )
    weights_resident_bytes = packed_bytes

    tflops = flop / (ms / 1e3) / 1e12
    b_gbps = b_bytes / (ms / 1e3) / 1e9

    ALL_RESULTS.append(
        {
            "provider": provider,
            "num_experts": num_experts,
            "avg_m": avg_m,
            "total_m": total_m,
            "gemm_n": gemm_n,
            "gemm_k": gemm_k,
            "ms": round(ms, 4),
            "ms_min": round(ms_min, 4),
            "ms_max": round(ms_max, 4),
            "tflops": round(tflops, 2),
            "b_gbps": round(b_gbps, 1),
            "peak_extra_allocated_MB": round(
                peak_extra_allocated / 1024 / 1024, 2
            ),
            "weights_resident_MB": round(weights_resident_bytes / 1024 / 1024, 2),
        }
    )
    return ms


def _correctness_check(rel_tol=CORRECTNESS_REL_TOL):
    """Cross-check sgl_kernel vs vLLM outputs on the same logical weights.

    Runs once per (shape, recipe) outside the timed loop and reports the
    aggregate L2 relative error — the two 4-bit kernels differ only by
    rounding, so a correct pair stays well under 1%.
    """
    if not VLLM_REF_AVAILABLE:
        print("[correctness skipped: vllm_xpu_kernels unavailable]")
        return
    rows = []
    failures = []
    sgl_run = {"mxfp4": _run_mxfp4_fused, "int4": _run_int4_fused}
    vllm_run = {"mxfp4": _run_vllm_mxfp4, "int4": _run_vllm_int4}
    for num_experts, avg_m, n, k in BENCH_SHAPES:
        for recipe in ("mxfp4", "int4"):
            sgl_in = _prepare_inputs(num_experts, avg_m, n, k, recipe, "sgl")
            sgl_run[recipe](sgl_in)
            torch.xpu.synchronize()
            sgl_out = sgl_in["output"].float()
            del sgl_in
            gc.collect()
            torch.xpu.empty_cache()

            vllm_in = _prepare_inputs(num_experts, avg_m, n, k, recipe, "vllm")
            vllm_run[recipe](vllm_in)
            torch.xpu.synchronize()
            vllm_out = vllm_in["output"].float()

            diff = (sgl_out - vllm_out).norm().item()
            denom = vllm_out.norm().clamp_min(1e-6).item()
            rel_l2_err = diff / denom
            if not math.isfinite(rel_l2_err) or rel_l2_err > rel_tol:
                failures.append(
                        f"E{num_experts}x{avg_m}x{n}x{k}/{recipe}: "
                    f"{rel_l2_err:.5g} exceeds {rel_tol}"
                )
            rows.append(
                {
                        "shape": f"E{num_experts}x{avg_m}x{n}x{k}",
                    "recipe": recipe,
                    "rel_l2_err": round(rel_l2_err, 5),
                }
            )
            del vllm_in, sgl_out, vllm_out
            gc.collect()
            torch.xpu.empty_cache()
    import pandas as pd

    print("\n[correctness: sgl_kernel vs vLLM (same logical weights)]")
    print(pd.DataFrame(rows).to_markdown(index=False))
    print()
    if failures:
        raise RuntimeError(
            "Correctness check failed; refusing to report benchmark timings: "
            + "; ".join(failures)
        )


if __name__ == "__main__":
    _correctness_check()
    benchmark.run(print_data=False)
    print("\nBenchmark finished!\n")
    import pandas as pd

    df = pd.DataFrame(ALL_RESULTS)
    # Pivot on provider so the comparison is one row per shape.
    pivot_cols = [
        "ms",
        "tflops",
        "b_gbps",
        "peak_extra_allocated_MB",
        "weights_resident_MB",
    ]
    pv = df.pivot_table(
        index=[
            "num_experts",
            "avg_m",
            "total_m",
            "gemm_n",
            "gemm_k",
        ],
        columns="provider",
        values=pivot_cols,
        aggfunc="first",
    )
    # sgl_kernel vs vLLM, per recipe. >1 means sgl_kernel is faster than vLLM.
    if ("ms", "mxfp4_fused") in pv.columns and ("ms", "vllm_mxfp4") in pv.columns:
        pv["speedup_mxfp4_sgl_vs_vllm"] = (
            pv[("ms", "vllm_mxfp4")] / pv[("ms", "mxfp4_fused")]
        ).round(2)
    if ("ms", "int4_fused") in pv.columns and ("ms", "vllm_int4") in pv.columns:
        pv["speedup_int4_sgl_vs_vllm"] = (
            pv[("ms", "vllm_int4")] / pv[("ms", "int4_fused")]
        ).round(2)
    # sgl_kernel INT4 vs MXFP4 head to head. >1 means int4_fused is faster.
    if ("ms", "mxfp4_fused") in pv.columns and ("ms", "int4_fused") in pv.columns:
        pv["speedup_int4_vs_mxfp4"] = (
            pv[("ms", "mxfp4_fused")] / pv[("ms", "int4_fused")]
        ).round(2)
    print(pv.to_markdown())
