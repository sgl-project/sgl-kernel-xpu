"""Benchmark the Xe2 FP8 W8A16 grouped GEMM against matched references.

This is the op-level companion to ``bench_fused_moe_fp8.py``, following the
same split as the W4A16 benchmarks. It compares scalar SGL/vLLM kernels on
identical logical FP8 weights and reports the SGL 128x128 block-scale path as
a separate quantization contract.

Run:
  python benchmark/bench_moe_fp8_w8a16_grouped_gemm.py
  SGL_MOE_BENCH_FULL_SHAPES=1 python benchmark/bench_moe_fp8_w8a16_grouped_gemm.py
"""

import os

import sgl_kernel  # noqa: F401 - registers torch.ops.sgl_kernel
import torch
import triton


def _import_vllm_grouped_gemm():
    import vllm_xpu_kernels._moe_C  # noqa: F401
    import vllm_xpu_kernels._xpu_C  # noqa: F401
    from vllm_xpu_kernels.fused_moe_interface import cutlass_grouped_gemm_xe2

    return cutlass_grouped_gemm_xe2


try:
    _vllm_grouped_gemm = _import_vllm_grouped_gemm()
    VLLM_AVAILABLE = True
except Exception as exc:
    _vllm_grouped_gemm = None
    VLLM_AVAILABLE = False
    print(f"[vLLM provider disabled: {type(exc).__name__}: {exc}]")


FP8_MAX = 448.0
BLOCK_SIZE = 128
QUICK_SHAPES = [
    # (experts, avg_m, N, K): controlled scalar dispatch boundaries.
    (8, 4, 1024, 1024),
    (8, 16, 1024, 1024),
    (8, 64, 1024, 2048),
    (8, 129, 1024, 1024),
]
FULL_SHAPES = QUICK_SHAPES + [
    # One-rank model GEMM1/GEMM2 shapes used during optimization.
    (8, 64, 2560, 3584),
    (8, 64, 3584, 1280),
    (8, 64, 1024, 7168),
    (8, 64, 7168, 512),
    (16, 64, 256, 2048),
    (16, 64, 2048, 128),
]
BENCH_SHAPES = (
    FULL_SHAPES if os.environ.get("SGL_MOE_BENCH_FULL_SHAPES") == "1" else QUICK_SHAPES
)


def _quantize_scalar(weight):
    scale = weight.float().abs().amax().clamp_min(1e-12) / FP8_MAX
    quantized = (
        (weight.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    )
    return quantized, scale


def _quantize_block(weight):
    experts, rows, columns = weight.shape
    assert rows % BLOCK_SIZE == 0 and columns % BLOCK_SIZE == 0
    blocked = weight.float().reshape(
        experts,
        rows // BLOCK_SIZE,
        BLOCK_SIZE,
        columns // BLOCK_SIZE,
        BLOCK_SIZE,
    )
    scales = blocked.abs().amax((2, 4), keepdim=True).clamp_min(1e-12) / FP8_MAX
    quantized = (blocked / scales).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return (
        quantized.reshape_as(weight),
        scales.reshape(experts, rows // BLOCK_SIZE, columns // BLOCK_SIZE),
    )


def _make_inputs(experts, avg_m, gemm_n, gemm_k, provider):
    torch.manual_seed(0)
    torch.xpu.manual_seed_all(0)
    total_m = experts * avg_m
    activations = (
        torch.randn((total_m, gemm_k), device="xpu", dtype=torch.bfloat16) / 16
    )
    weight_bf16 = (
        torch.randn((experts, gemm_n, gemm_k), device="xpu", dtype=torch.bfloat16) / 16
    )
    rows_per_expert = torch.full((experts,), avg_m, device="xpu", dtype=torch.int32)
    output = torch.empty((total_m, gemm_n), device="xpu", dtype=torch.bfloat16)

    if provider == "sgl_block":
        weights, scales = _quantize_block(weight_bf16)
    else:
        weights, scale = _quantize_scalar(weight_bf16)
        scales = torch.full(
            (experts, 1), scale.item(), device="xpu", dtype=torch.float32
        )
    del weight_bf16

    if provider == "vllm_scalar":
        weights = weights.transpose(-1, -2).contiguous()
        scales = scales.flatten()

    return activations, weights, scales, rows_per_expert, output


def _run_sgl(inputs, experts):
    activations, weights, scales, rows_per_expert, output = inputs
    torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20_fp8_w8a16(
        output,
        activations,
        weights,
        scales,
        None,
        rows_per_expert,
        experts,
    )
    return output


def _run_vllm(inputs, experts, gemm_n, gemm_k):
    activations, weights, scales, rows_per_expert, output = inputs
    _vllm_grouped_gemm(
        activations,
        weights,
        scales,
        None,
        output,
        rows_per_expert,
        gemm_n,
        gemm_k,
        experts,
    )
    return output


PROVIDERS = ["sgl_scalar", "sgl_block"]
PROVIDER_NAMES = ["SGL scalar W8A16", "SGL 128x128 block W8A16"]
STYLES = [("green", "-"), ("blue", "-")]
if VLLM_AVAILABLE:
    PROVIDERS.insert(1, "vllm_scalar")
    PROVIDER_NAMES.insert(1, "vLLM scalar W8A16")
    STYLES.insert(1, ("red", "-"))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["experts", "avg_m", "gemm_n", "gemm_k"],
        x_vals=BENCH_SHAPES,
        line_arg="provider",
        line_vals=PROVIDERS,
        line_names=PROVIDER_NAMES,
        styles=STYLES,
        ylabel="Time (ms)",
        plot_name="moe-fp8-w8a16-grouped-gemm",
        args={},
    )
)
def benchmark(experts, avg_m, gemm_n, gemm_k, provider):
    inputs = _make_inputs(experts, avg_m, gemm_n, gemm_k, provider)
    if provider == "vllm_scalar":
        run = lambda: _run_vllm(inputs, experts, gemm_n, gemm_k)
    else:
        run = lambda: _run_sgl(inputs, experts)
    return triton.testing.do_bench(run, warmup=100, rep=300)


def _check_scalar_reference():
    if not VLLM_AVAILABLE:
        return
    shape = QUICK_SHAPES[1]
    sgl_inputs = _make_inputs(*shape, "sgl_scalar")
    vllm_inputs = _make_inputs(*shape, "vllm_scalar")
    sgl_output = _run_sgl(sgl_inputs, shape[0])
    vllm_output = _run_vllm(vllm_inputs, shape[0], shape[2], shape[3])
    torch.xpu.synchronize()
    torch.testing.assert_close(sgl_output, vllm_output, rtol=5e-2, atol=5e-2)
    print("[correctness] SGL/vLLM scalar outputs match", flush=True)


if __name__ == "__main__":
    _check_scalar_reference()
    benchmark.run(print_data=True)
