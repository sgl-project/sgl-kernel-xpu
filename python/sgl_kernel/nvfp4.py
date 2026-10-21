"""NVFP4 weight-only-quantization (WoQ) helpers for Intel XPU.

Xe3P has no native NVFP4 MMA — the ``bdpas`` block-scaled instruction only
accepts UE8M0 scale factors with a fixed block size of 32, while NVFP4 uses
E4M3 scale factors with block size 16 plus one FP32 per-tensor global scale.
NVFP4 weights can therefore be emulated either by requantizing to MXFP8 or by
dequantizing directly into a BF16 MMA fragment. Both paths leave the packed
4-bit payload in memory.

``nvfp4_woq_prep_xe35`` rewrites only the scales into the ``(c, sfmx)`` pair consumed
by the fused WoQ mainloop, which keeps the 4-bit payload in memory and
requantizes inside the GEMM.  It adds ~1/32 byte per weight element on top of
NVFP4 and is the only weight preprocessing the path needs; the E2M1 payload is
consumed exactly as the checkpoint ships it.

See tests/nvfp4_utils.py for the reference implementation of the numerics and
src/sycl/xe35/nvfp4_woq.cpp for the device kernel.
"""

from typing import Tuple

import torch

NVFP4_BLOCK_SIZE = 16
MXFP8_BLOCK_SIZE = 32


def nvfp4_woq_prep_xe35(
    w_s: torch.Tensor,
    w_gs: torch.Tensor,
    out_c: torch.Tensor = None,
    out_sfmx: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One-time scale prep for the fused NVFP4 WoQ mainloop.

    Args:
        w_s: (E, N, K/16) float8_e4m3fn NVFP4 block scales.
        w_gs: (E,) float32 dequant multiplier.

    Returns:
        ``(c, sfmx)`` where ``c`` is (E, N, K/16) bfloat16 — the per-16
        multiplier with the global scale folded in, row-major so a lane reads
        its entries for one N contiguously — and ``sfmx`` is (E, K/32, N) uint8
        UE8M0, MN-major, the per-32 requantization scale handed straight to
        ``bdpas`` through the ordinary Xe block-2D scale load.
    """
    assert w_s.dim() == 3, f"w_s must be (E, N, K/16), got {tuple(w_s.shape)}"
    assert (
        w_s.dtype == torch.float8_e4m3fn
    ), f"w_s must be float8_e4m3fn, got {w_s.dtype}"
    assert w_gs.dtype == torch.float32, f"w_gs must be float32, got {w_gs.dtype}"
    e, n, sk16 = w_s.shape
    assert sk16 % 2 == 0, f"K/16 ({sk16}) must be even, i.e. K a multiple of 32"
    assert w_gs.numel() == e, f"w_gs must have {e} entries, got {w_gs.numel()}"

    if out_c is None:
        out_c = torch.empty((e, n, sk16), device=w_s.device, dtype=torch.bfloat16)
    if out_sfmx is None:
        out_sfmx = torch.empty((e, sk16 // 2, n), device=w_s.device, dtype=torch.uint8)
    torch.ops.sgl_kernel.nvfp4_woq_prep_xe35.default(out_c, out_sfmx, w_s, w_gs)
    return out_c, out_sfmx


def nvfp4_woq_mxfp8_scaled_grouped_mm_xe35(
    output: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scales_a: torch.Tensor,
    sfmx_b: torch.Tensor,
    c_b: torch.Tensor,
    problem_sizes: torch.Tensor,
    expert_offsets: torch.Tensor,
    workspace: torch.Tensor,
) -> None:
    """Fused NVFP4 weight-only-quantized MXFP8 grouped GEMM (Xe3P / CRI only).

    B stays NVFP4 in memory; the mainloop dequantizes it to BF16 and
    requantizes to MXFP8 in registers, then runs a normal MXFP8 ``bdpas``.

    Args:
        output: (sum_m, N) float32, ragged over experts. Zero-initialized.
        a: (sum_m, K) float8_e4m3fn MXFP8 activations.
        b: (E, N, K/2) uint8 packed E2M1 weights.
        scales_a: (sum_m, K/32) uint8 UE8M0 activation scales, row-major.
        sfmx_b: (E, K/32, N) uint8 UE8M0, MN-major — from :func:`nvfp4_woq_prep_xe35`.
        c_b: (E, N, K/16) bfloat16 per-16 multiplier — from :func:`nvfp4_woq_prep_xe35`.
        problem_sizes: (E, 3) int32 per-expert (M, N, K).
        expert_offsets: (E,) int32 cumulative row starts into the flat A/output.
        workspace: uint8 scratch for the grouped GEMM.

    Shape constraints (checked in C++): ``K`` must be a multiple of the 64-wide
    workgroup K tile, and ``N * (K // 32)`` must be a multiple of 64 so that each
    expert's slice of ``sfmx_b`` starts on a 64-byte boundary, which the Xe
    block-2D scale load requires.  Per-expert M is unconstrained.
    """
    from sgl_kernel import is_xe3_arch

    assert (
        is_xe3_arch()
    ), "nvfp4_woq_mxfp8_scaled_grouped_mm_xe35 is only supported on CRI (Xe3P) devices"
    torch.ops.sgl_kernel.nvfp4_woq_mxfp8_scaled_grouped_mm_xe35.default(
        output, a, b, scales_a, sfmx_b, c_b, problem_sizes, expert_offsets, workspace
    )


def nvfp4_woq_bf16_grouped_mm_xe35(
    output: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    sfmx_b: torch.Tensor,
    c_b: torch.Tensor,
    problem_sizes: torch.Tensor,
    expert_offsets: torch.Tensor,
    workspace: torch.Tensor,
) -> None:
    """NVFP4 weight-only-quantized BF16 grouped GEMM (Xe3P / CRI only).

    Activations remain BF16. Packed E2M1 weights are converted and multiplied
    by their reconstructed per-16 BF16 scale inside the mainloop, then consumed
    by BF16 DPAS with FP32 accumulation.

    Args:
        output: (sum_m, N) float32, ragged over experts.
        a: (sum_m, K) bfloat16 activations.
        b: (E, N, K/2) uint8 packed E2M1 weights.
        sfmx_b: (E, K/32, N) uint8 prepared exponent component.
        c_b: (E, N, K/16) bfloat16 prepared multiplier component.
        problem_sizes: (E, 3) int32 per-expert (M, N, K).
        expert_offsets: (E,) int32 cumulative row starts.
        workspace: uint8 scratch for the grouped GEMM.
    """
    from sgl_kernel import is_xe3_arch

    assert (
        is_xe3_arch()
    ), "nvfp4_woq_bf16_grouped_mm_xe35 is only supported on CRI (Xe3P) devices"
    torch.ops.sgl_kernel.nvfp4_woq_bf16_grouped_mm_xe35.default(
        output, a, b, sfmx_b, c_b, problem_sizes, expert_offsets, workspace
    )
