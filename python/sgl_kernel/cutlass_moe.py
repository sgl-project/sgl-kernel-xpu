"""CUTLASS-based fused MoE wrapper for Intel XPU.

Signature-compatible with sglang's CUDA cutlass_fused_experts_fp8. CUDA-only
args (*_strides, *_ptrs, enable_es) are unused on XPU. `use_mxfp8` selects
between DSV3-style FP8 (BS=128, fp32 scales) and OCP MXFP8 (BS=32, UE8M0
uint8 scales); C++ dispatches on scales_a dtype.
"""

import os
from typing import Optional, Tuple

import torch
from sgl_kernel.elementwise import silu_and_mul
from sgl_kernel.gemm import (
    sgl_per_token_group_quant_8bit,
    sgl_per_token_group_quant_fp4,
)
from sgl_kernel.moe import (
    apply_shuffle_mul_sum,
    fp8_blockwise_scaled_grouped_mm,
    mxfp4_blockwise_scaled_grouped_mm,
    prepare_moe_input,
    scatter_tokens_to_experts,
)

_FP8_E4M3_MIN = -448.0
_FP8_E4M3_MAX = 448.0

_MXFP8_BLOCK_SIZE = 32
_NVFP4_WOQ_COMPUTE_ENV = "SGLANG_NVFP4_WOQ_COMPUTE"


def _nvfp4_woq_compute_mode() -> str:
    mode = os.getenv(_NVFP4_WOQ_COMPUTE_ENV, "mxfp8").strip().lower()
    if mode not in ("mxfp8", "bf16"):
        raise ValueError(
            f"{_NVFP4_WOQ_COMPUTE_ENV} must be 'mxfp8' or 'bf16', got {mode!r}"
        )
    return mode


def _per_token_group_quant_fp8(x: torch.Tensor, group_size: int = 128):
    """Per-token group quant along last dim. Returns (x_q fp8_e4m3, x_s fp32)."""
    assert x.shape[-1] % group_size == 0
    out_q = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    out_s_shape = (*x.shape[:-1], x.shape[-1] // group_size)
    out_s = torch.empty(out_s_shape, device=x.device, dtype=torch.float32)
    sgl_per_token_group_quant_8bit(
        x,
        out_q,
        out_s,
        group_size,
        1e-10,
        _FP8_E4M3_MIN,
        _FP8_E4M3_MAX,
        False,
        False,
        None,
        False,
    )
    return out_q, out_s


def _per_token_group_quant_mxfp8(x: torch.Tensor, group_size: int = _MXFP8_BLOCK_SIZE):
    """MXFP8 per-token group quant. Returns (x_q float8_e4m3fn, x_s uint8 UE8M0).

    Uses v1 (enable_v2=False): v2 does not support row-major UE8M0 output.
    """
    assert x.shape[-1] % group_size == 0
    out_q = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    out_s_shape = (*x.shape[:-1], x.shape[-1] // group_size)
    out_s = torch.empty(out_s_shape, device=x.device, dtype=torch.uint8)
    sgl_per_token_group_quant_8bit(
        x,
        out_q,
        out_s,
        group_size,
        1e-10,
        _FP8_E4M3_MIN,
        _FP8_E4M3_MAX,
        True,  # scale_ue8m0
        False,
        None,
        False,  # enable_v2 (v1 supports row-major UE8M0; v2 does not)
    )
    return out_q, out_s


def cutlass_fused_experts_fp8(
    a: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    a1_strides: torch.Tensor,
    c1_strides: torch.Tensor,
    a2_strides: torch.Tensor,
    c2_strides: torch.Tensor,
    workspace: torch.Tensor,
    a_ptrs: torch.Tensor,
    b_ptrs: torch.Tensor,
    out_ptrs: torch.Tensor,
    a_scales_ptrs: torch.Tensor,
    b_scales_ptrs: torch.Tensor,
    expert_offsets: torch.Tensor,
    problem_sizes1: torch.Tensor,
    problem_sizes2: torch.Tensor,
    use_fp8_blockscale: bool = True,
    use_mxfp8: bool = False,
    output: Optional[torch.Tensor] = None,
    enable_es: Tuple[bool, bool] = (False, False),
) -> torch.Tensor:
    """Fused MoE on Intel XPU.

    Mirrors sglang's CUDA cutlass_fused_experts_fp8 signature; CUDA-only args
    (*_strides, *_ptrs, enable_es) are unused on XPU.

    Weights expected in (E, N, K) row-major: w1=(E, n*2, k), w2=(E, k, n).
    MXFP8 scales are (E, n*2, k/32) and (E, k, n/32) un-transposed row-major;
    the kernel transposes B-scales on device. Sglang's dispatcher applies
    .transpose(1, 2) before calling; auto-detected and undone below.
    """
    assert use_fp8_blockscale, "Only support fp8 blockscale on XPU"
    assert enable_es == (False, False), "enable_es is CUDA-only"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert w1_q.dtype == torch.float8_e4m3fn
    assert w2_q.dtype == torch.float8_e4m3fn
    assert w1_q.dim() == 3 and w2_q.dim() == 3, "Weights must be 3D"
    assert w1_q.shape[0] == w2_q.shape[0], "Expert count mismatch w1/w2"
    assert a.dtype in (torch.half, torch.bfloat16), "Invalid input dtype"
    if use_mxfp8:
        assert (
            w1_scale.dtype == torch.uint8 and w2_scale.dtype == torch.uint8
        ), "use_mxfp8=True requires uint8 UE8M0 weight scales"
    else:
        assert (
            w1_scale.dtype == torch.float32 and w2_scale.dtype == torch.float32
        ), "use_mxfp8=False requires fp32 weight scales"

    # Detect on w1 only — w2 is ambiguous when intermediate == k_hidden.
    # Sglang transposes both weights+scales together or not at all.
    hidden_size = a.shape[1]
    if w1_q.shape[2] == hidden_size:
        sglang_transposed = False
    elif w1_q.shape[1] == hidden_size:
        sglang_transposed = True
    else:
        raise AssertionError(
            f"w1_q shape {tuple(w1_q.shape)} incompatible with a.shape[1]={hidden_size}"
        )

    if sglang_transposed:
        w1_q = w1_q.transpose(1, 2).contiguous()
        w2_q = w2_q.transpose(1, 2).contiguous()
        w1_scale = w1_scale.transpose(1, 2).contiguous()
        w2_scale = w2_scale.transpose(1, 2).contiguous()

    assert w1_q.shape[2] == hidden_size
    assert w2_q.shape[1] == hidden_size
    assert w1_q.shape[1] == 2 * w2_q.shape[2]

    del a1_strides, c1_strides, a2_strides, c2_strides
    del a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs

    out_dtype = a.dtype
    num_experts = w1_q.size(0)
    m = a.size(0)
    k = a.size(1)
    n = w2_q.size(2)
    topk = topk_ids.size(1)
    device = a.device

    if use_mxfp8:
        # MXFP8 gating (mirrors CUDA use_mxfp8 checks).
        from sgl_kernel import is_xe3_arch

        assert is_xe3_arch(), "MXFP8 requires an Xe3P (CRI) XPU"
        assert (
            k % _MXFP8_BLOCK_SIZE == 0
        ), f"MXFP8 requires hidden size divisible by {_MXFP8_BLOCK_SIZE}, got k={k}"
        assert (
            n % _MXFP8_BLOCK_SIZE == 0
        ), f"MXFP8 requires intermediate size divisible by {_MXFP8_BLOCK_SIZE}, got n={n}"
        expected_w1_scale_shape = (
            num_experts,
            w1_q.shape[1],
            w1_q.shape[2] // _MXFP8_BLOCK_SIZE,
        )
        expected_w2_scale_shape = (
            num_experts,
            w2_q.shape[1],
            w2_q.shape[2] // _MXFP8_BLOCK_SIZE,
        )
        assert (
            w1_scale.shape == expected_w1_scale_shape
        ), f"MXFP8 w1_scale must be {expected_w1_scale_shape}, got {tuple(w1_scale.shape)}"
        assert (
            w2_scale.shape == expected_w2_scale_shape
        ), f"MXFP8 w2_scale must be {expected_w2_scale_shape}, got {tuple(w2_scale.shape)}"

    a_map = torch.empty((topk_ids.numel(),), dtype=torch.int32, device=device)
    c_map = torch.empty((topk_ids.numel(),), dtype=torch.int32, device=device)

    # sglang allocates expert_offsets as (E+1,) and slices [:-1]; XPU's
    # prepare_moe_input takes size E. Accept either.
    eo = (
        expert_offsets[:num_experts]
        if expert_offsets.numel() > num_experts
        else expert_offsets
    )

    # prepare_moe_input fills eo with per-expert M counts (not cumulative).
    prepare_moe_input(
        topk_ids,
        eo,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
        num_experts,
        n,
        k,
    )

    # Flat-2D kernel wants cumulative start offsets; exclusive-scan eo.
    expert_starts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    if num_experts > 1:
        expert_starts[1:] = torch.cumsum(eo[:-1], dim=0).to(torch.int32)

    # Pass all experts (including zero-M) directly. The Xe group tile
    # scheduler skips zero-M groups at runtime via ceil_div(0, BLK_M) = 0
    # (xe_tile_scheduler_group.hpp), so the mainloop never runs for them.
    # The driver bypasses sycl-tla's spurious M==0 reject in
    # can_implement — see blockwise_moe_runner.hpp for details.

    # Scatter then quantize. scatter_tokens_to_experts uses c_map (the
    # src->dst permutation) to gather per-expert rows.
    rep_a = torch.empty((m * topk, k), dtype=a.dtype, device=device)
    scatter_tokens_to_experts(a, c_map, rep_a)
    if use_mxfp8:
        rep_a_q, rep_a1_scales = _per_token_group_quant_mxfp8(
            rep_a, group_size=_MXFP8_BLOCK_SIZE
        )
    else:
        rep_a_q, rep_a1_scales = _per_token_group_quant_fp8(rep_a, group_size=128)

    c1 = torch.zeros((m * topk, n * 2), dtype=torch.float32, device=device)

    # Empty int64 sentinels select on-device pointer-table + scale-transpose.
    empty_ptrs = torch.empty((0,), dtype=torch.int64, device=device)
    zeros_stride = torch.zeros((num_experts,), dtype=torch.int64, device=device)
    zeros_layout = torch.zeros((num_experts, 5), dtype=torch.int32, device=device)

    fp8_blockwise_scaled_grouped_mm(
        c1,
        empty_ptrs,
        empty_ptrs,
        empty_ptrs,
        empty_ptrs,
        empty_ptrs,
        rep_a_q,
        w1_q,
        rep_a1_scales,
        w1_scale,
        zeros_stride,
        zeros_stride,
        zeros_stride,
        zeros_layout,
        zeros_layout,
        problem_sizes1,
        expert_starts,
        workspace,
    )

    intermediate = torch.empty((m * topk, n), dtype=out_dtype, device=device)
    silu_and_mul(c1.to(out_dtype), intermediate)

    if use_mxfp8:
        intermediate_q, a2_scale = _per_token_group_quant_mxfp8(
            intermediate, group_size=_MXFP8_BLOCK_SIZE
        )
    else:
        intermediate_q, a2_scale = _per_token_group_quant_fp8(
            intermediate, group_size=128
        )

    c2 = torch.zeros((m * topk, k), dtype=torch.float32, device=device)
    fp8_blockwise_scaled_grouped_mm(
        c2,
        empty_ptrs,
        empty_ptrs,
        empty_ptrs,
        empty_ptrs,
        empty_ptrs,
        intermediate_q,
        w2_q,
        a2_scale,
        w2_scale,
        zeros_stride,
        zeros_stride,
        zeros_stride,
        zeros_layout,
        zeros_layout,
        problem_sizes2,
        expert_starts,
        workspace,
    )

    if output is None:
        output = torch.empty((m, k), dtype=out_dtype, device=device)
    apply_shuffle_mul_sum(c2.to(out_dtype), output, c_map, topk_weights.to(out_dtype))
    return output


_MXFP4_BLOCK_SIZE = 32


def cutlass_fused_experts_mxfp4(
    a: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused MoE on XPU using mxfp4_blockwise_scaled_grouped_mm. Slim signature
    mirrors sglang's NVFP4 cutlass_moe_fp4 (no CUDA-only placeholders).
    Weights in (E, N, K/2) uint8 packed E2M1 with UE8M0 block-32 scales:
      w1_q (E, n*2, k/2), w2_q (E, k, n/2),
      w1_scale (E, n*2, k/32), w2_scale (E, k, n/32) (un-transposed).
    """
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert w1_q.dtype == torch.uint8, "w1_q must be uint8 (packed MXFP4)"
    assert w2_q.dtype == torch.uint8, "w2_q must be uint8 (packed MXFP4)"
    assert w1_scale.dtype == torch.uint8, "w1_scale must be uint8 (UE8M0)"
    assert w2_scale.dtype == torch.uint8, "w2_scale must be uint8 (UE8M0)"
    assert w1_q.dim() == 3 and w2_q.dim() == 3, "Weights must be 3D (E, N, K/2)"
    assert w1_q.shape[0] == w2_q.shape[0], "Expert count mismatch w1/w2"
    assert a.dtype in (torch.half, torch.bfloat16), "Invalid input dtype"

    out_dtype = a.dtype
    num_experts = w1_q.size(0)
    m = a.size(0)
    k = a.size(1)
    n = w2_q.size(2) * 2  # w2_q is (E, k, n/2)
    topk = topk_ids.size(1)
    device = a.device

    assert w1_q.size(2) * 2 == k
    assert w2_q.size(1) == k
    assert w1_q.size(1) == 2 * n

    a_map = torch.empty((topk_ids.numel(),), dtype=torch.int32, device=device)
    c_map = torch.empty((topk_ids.numel(),), dtype=torch.int32, device=device)

    # prepare_moe_input writes per-expert M counts (not cumulative starts).
    expert_offsets = torch.zeros(num_experts, dtype=torch.int32, device=device)
    problem_sizes1 = torch.zeros((num_experts, 3), dtype=torch.int32, device=device)
    problem_sizes2 = torch.zeros((num_experts, 3), dtype=torch.int32, device=device)

    prepare_moe_input(
        topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
        num_experts,
        n,
        k,
    )

    # Flat-2D kernel needs cumulative start offsets.
    expert_starts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    if num_experts > 1:
        expert_starts[1:] = torch.cumsum(expert_offsets[:-1], dim=0).to(torch.int32)

    # XPU scatter doesn't support uint8/fp4; scatter bf16 then quantize.
    rep_a = torch.empty((m * topk, k), dtype=a.dtype, device=device)
    scatter_tokens_to_experts(a, c_map, rep_a)
    rep_a_q, rep_a1_scales = sgl_per_token_group_quant_fp4(
        rep_a, group_size=_MXFP4_BLOCK_SIZE
    )

    c1 = torch.zeros((m * topk, n * 2), dtype=torch.float32, device=device)
    # Empty sentinels -> on-device ptr-table + scale-transpose.
    empty_ptrs = torch.empty((0,), dtype=torch.int64, device=device)
    workspace = torch.zeros((64 * 1024 * 1024,), dtype=torch.uint8, device=device)

    mxfp4_blockwise_scaled_grouped_mm(
        c1,
        empty_ptrs,
        empty_ptrs,
        empty_ptrs,
        empty_ptrs,
        empty_ptrs,
        rep_a_q,
        w1_q,
        rep_a1_scales,
        w1_scale,
        problem_sizes1,
        expert_starts,
        workspace,
    )

    intermediate = torch.empty((m * topk, n), dtype=out_dtype, device=device)
    silu_and_mul(c1.to(out_dtype), intermediate)

    intermediate_q, a2_scale = sgl_per_token_group_quant_fp4(
        intermediate, group_size=_MXFP4_BLOCK_SIZE
    )

    c2 = torch.zeros((m * topk, k), dtype=torch.float32, device=device)
    mxfp4_blockwise_scaled_grouped_mm(
        c2,
        empty_ptrs,
        empty_ptrs,
        empty_ptrs,
        empty_ptrs,
        empty_ptrs,
        intermediate_q,
        w2_q,
        a2_scale,
        w2_scale,
        problem_sizes2,
        expert_starts,
        workspace,
    )

    if output is None:
        output = torch.empty((m, k), dtype=out_dtype, device=device)
    apply_shuffle_mul_sum(c2.to(out_dtype), output, c_map, topk_weights.to(out_dtype))
    return output


# ---------------------------------------------------------------------------
# NVFP4 weight-only quantization (MXFP8 or BF16 compute)
# ---------------------------------------------------------------------------

def nvfp4_woq_prepare_weights(
    w1_scale: torch.Tensor,
    w1_gs: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_gs: torch.Tensor,
) -> dict:
    """One-time scale prep for :func:`cutlass_fused_experts_nvfp4_woq`.

    Call once when the checkpoint is loaded and keep the result; it replaces the
    NVFP4 E4M3 block scales and per-tensor global scales with the ``(c, sfmx)``
    pair the fused mainloop reads.  The 4-bit payloads are used as-is.
    ``w1_gs`` accepts one shared scale per expert or separate ``(gate, up)``
    scales in shape ``(E, 2)``. ``w2_gs`` accepts one scale per expert.
    """
    from sgl_kernel.nvfp4 import nvfp4_woq_prep

    num_experts = w1_scale.size(0)
    if w1_gs.dim() == 2 and w1_gs.shape == (num_experts, 2):
        assert (
            w1_scale.size(1) % 2 == 0
        ), "w1_scale output dimension must be even for separate gate/up scales"
        split_n = w1_scale.size(1) // 2
        gate_c, gate_sfmx = nvfp4_woq_prep(
            w1_scale[:, :split_n].contiguous(), w1_gs[:, 0].contiguous()
        )
        up_c, up_sfmx = nvfp4_woq_prep(
            w1_scale[:, split_n:].contiguous(), w1_gs[:, 1].contiguous()
        )
        w1_c = torch.cat((gate_c, up_c), dim=1)
        w1_sfmx = torch.cat((gate_sfmx, up_sfmx), dim=2)
    else:
        assert w1_gs.shape in (
            (num_experts,),
            (num_experts, 1),
        ), f"w1_gs must be (E,), (E, 1), or (E, 2), got {tuple(w1_gs.shape)}"
        w1_c, w1_sfmx = nvfp4_woq_prep(
            w1_scale, w1_gs.reshape(num_experts).contiguous()
        )

    w2_num_experts = w2_scale.size(0)
    assert w2_gs.shape in (
        (w2_num_experts,),
        (w2_num_experts, 1),
    ), f"w2_gs must be (E,) or (E, 1), got {tuple(w2_gs.shape)}"
    w2_c, w2_sfmx = nvfp4_woq_prep(w2_scale, w2_gs.reshape(w2_num_experts).contiguous())
    return {"w1_c": w1_c, "w1_sfmx": w1_sfmx, "w2_c": w2_c, "w2_sfmx": w2_sfmx}


def cutlass_fused_experts_nvfp4_woq(
    a: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    prepared: dict,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused MoE with NVFP4 weights and MXFP8 or BF16 computation.

    The weights stay 4-bit in memory. ``SGLANG_NVFP4_WOQ_COMPUTE`` selects
    ``mxfp8`` (default, in-register weight requantization) or ``bf16``
    (in-register weight dequantization with BF16 activations and MMA).

    Args:
        a: (M, K) activations, fp16 or bf16.
        w1_q: (E, 2N, K/2) uint8 packed E2M1 gate+up weights.
        w2_q: (E, K, N/2) uint8 packed E2M1 down weights.
        prepared: result of :func:`nvfp4_woq_prepare_weights`.
        topk_weights, topk_ids: routing results.
        output: optional (M, K) output buffer.
        workspace: optional uint8 scratch for the grouped GEMMs.

    Returns:
        (M, K) tensor in ``a.dtype``.
    """
    from sgl_kernel.nvfp4 import (
        nvfp4_woq_bf16_grouped_mm,
        nvfp4_woq_mxfp8_scaled_grouped_mm,
    )

    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert w1_q.dtype == torch.uint8 and w2_q.dtype == torch.uint8, "weights must be packed E2M1 uint8"
    assert w1_q.dim() == 3 and w2_q.dim() == 3, "weights must be 3D (E, N, K/2)"
    assert w1_q.shape[0] == w2_q.shape[0], "expert count mismatch w1/w2"
    assert a.dtype in (torch.half, torch.bfloat16), "invalid input dtype"
    compute_mode = _nvfp4_woq_compute_mode()

    out_dtype = a.dtype
    num_experts = w1_q.size(0)
    m = a.size(0)
    k = a.size(1)
    n = w2_q.size(2) * 2  # w2_q is (E, k, n/2)
    topk = topk_ids.size(1)
    device = a.device

    assert w1_q.size(2) * 2 == k, f"w1_q inner dim must be K/2={k // 2}, got {w1_q.size(2)}"
    assert w2_q.size(1) == k, f"w2_q must have K={k} rows, got {w2_q.size(1)}"
    assert w1_q.size(1) == 2 * n, f"w1_q must have 2N={2 * n} rows, got {w1_q.size(1)}"

    topk_ids = topk_ids.to(device=device, dtype=torch.int32).contiguous()
    a_map = torch.empty((topk_ids.numel(),), dtype=torch.int32, device=device)
    c_map = torch.empty((topk_ids.numel(),), dtype=torch.int32, device=device)
    expert_counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    problem_sizes1 = torch.zeros((num_experts, 3), dtype=torch.int32, device=device)
    problem_sizes2 = torch.zeros((num_experts, 3), dtype=torch.int32, device=device)
    prepare_moe_input(
        topk_ids, expert_counts, problem_sizes1, problem_sizes2, a_map, c_map,
        num_experts, n, k,
    )
    expert_starts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    if num_experts > 1:
        expert_starts[1:] = torch.cumsum(expert_counts[:-1], dim=0).to(torch.int32)

    if workspace is None:
        workspace = torch.zeros((64 * 1024 * 1024,), dtype=torch.uint8, device=device)

    c1 = torch.zeros((m * topk, n * 2), dtype=torch.float32, device=device)
    c2 = torch.zeros((m * topk, k), dtype=torch.float32, device=device)
    if compute_mode == "bf16":
        # Both GEMMs consume BF16 directly. Convert FP16 inputs once before the
        # routing scatter and keep the intermediate activation in BF16.
        a_bf16 = a if a.dtype == torch.bfloat16 else a.to(torch.bfloat16)
        rep_a = torch.empty((m * topk, k), dtype=torch.bfloat16, device=device)
        scatter_tokens_to_experts(a_bf16, c_map, rep_a)
        nvfp4_woq_bf16_grouped_mm(
            c1,
            rep_a,
            w1_q,
            prepared["w1_sfmx"],
            prepared["w1_c"],
            problem_sizes1,
            expert_starts,
            workspace,
        )

        intermediate = torch.empty((m * topk, n), dtype=torch.bfloat16, device=device)
        silu_and_mul(c1.to(torch.bfloat16), intermediate)
        nvfp4_woq_bf16_grouped_mm(
            c2,
            intermediate,
            w2_q,
            prepared["w2_sfmx"],
            prepared["w2_c"],
            problem_sizes2,
            expert_starts,
            workspace,
        )
    else:
        # XPU scatter doesn't support fp8; scatter in a.dtype then quantize.
        rep_a = torch.empty((m * topk, k), dtype=a.dtype, device=device)
        scatter_tokens_to_experts(a, c_map, rep_a)
        rep_a_q, rep_a_s = _per_token_group_quant_mxfp8(
            rep_a, group_size=_MXFP8_BLOCK_SIZE
        )
        nvfp4_woq_mxfp8_scaled_grouped_mm(
            c1,
            rep_a_q,
            w1_q,
            rep_a_s,
            prepared["w1_sfmx"],
            prepared["w1_c"],
            problem_sizes1,
            expert_starts,
            workspace,
        )

        intermediate = torch.empty((m * topk, n), dtype=out_dtype, device=device)
        silu_and_mul(c1.to(out_dtype), intermediate)
        inter_q, inter_s = _per_token_group_quant_mxfp8(
            intermediate, group_size=_MXFP8_BLOCK_SIZE
        )
        nvfp4_woq_mxfp8_scaled_grouped_mm(
            c2,
            inter_q,
            w2_q,
            inter_s,
            prepared["w2_sfmx"],
            prepared["w2_c"],
            problem_sizes2,
            expert_starts,
            workspace,
        )

    if output is None:
        output = torch.empty((m, k), dtype=out_dtype, device=device)
    apply_shuffle_mul_sum(c2.to(out_dtype), output, c_map, topk_weights.to(out_dtype))
    return output
