from typing import Optional, Tuple

import torch
from sgl_kernel.elementwise import silu_and_mul
from sgl_kernel.gemm import sgl_per_token_group_quant_8bit
from sgl_kernel.moe import apply_shuffle_mul_sum, prepare_moe_input, scatter_tokens_to_experts
from sgl_kernel import nvfp4
from sgl_kernel.nvfp4 import nvfp4_woq_bf16_grouped_mm_xe35, nvfp4_woq_mxfp8_scaled_grouped_mm_xe35

_FP8_E4M3_MIN = -448.0
_FP8_E4M3_MAX = 448.0
_MXFP8_BLOCK_SIZE = 32

def _nvfp4_woq_compute_mode_xe35(mode: str = "auto") -> str:
    mode = mode.strip().lower()
    if mode not in ("auto", "mxfp8", "bf16"):
        raise ValueError(
            "compute_mode must be 'auto', 'mxfp8', or 'bf16', "
            f"got {mode!r}"
        )
    return "bf16" if mode == "auto" else mode


def _per_token_group_quant_mxfp8_xe35(x: torch.Tensor, group_size: int = _MXFP8_BLOCK_SIZE):
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


def nvfp4_woq_prepare_weights_xe35(
    w1_scale: torch.Tensor,
    w1_gs: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_gs: torch.Tensor,
) -> dict:
    """One-time scale prep for :func:`cutlass_fused_experts_nvfp4_woq_xe35`.

    Call once when the checkpoint is loaded and keep the result. ``w1_gs`` and
    ``w2_gs`` are the FP32 checkpoint global scales used to quantize the block
    scales: dequantization multiplies by their reciprocals. This function
    converts them to the dequant multipliers required by ``nvfp4_woq_prep_xe35``.
    ``w1_gs`` accepts one shared scale per expert or separate ``(gate, up)``
    scales in shape ``(E, 2)``. ``w2_gs`` accepts one scale per expert.
    The 4-bit payloads are used as-is.
    """

    num_experts = w1_scale.size(0)
    if w1_gs.dim() == 2 and w1_gs.shape == (num_experts, 2):
        assert (
            w1_scale.size(1) % 2 == 0
        ), "w1_scale output dimension must be even for separate gate/up scales"
        split_n = w1_scale.size(1) // 2
        gate_c, gate_sfmx = nvfp4.nvfp4_woq_prep_xe35(
            w1_scale[:, :split_n].contiguous(),
            w1_gs[:, 0].reciprocal().contiguous(),
        )
        up_c, up_sfmx = nvfp4.nvfp4_woq_prep_xe35(
            w1_scale[:, split_n:].contiguous(),
            w1_gs[:, 1].reciprocal().contiguous(),
        )
        w1_c = torch.cat((gate_c, up_c), dim=1)
        w1_sfmx = torch.cat((gate_sfmx, up_sfmx), dim=2)
    else:
        assert w1_gs.shape in (
            (num_experts,),
            (num_experts, 1),
        ), f"w1_gs must be (E,), (E, 1), or (E, 2), got {tuple(w1_gs.shape)}"
        w1_c, w1_sfmx = nvfp4.nvfp4_woq_prep_xe35(
            w1_scale, w1_gs.reshape(num_experts).reciprocal().contiguous()
        )

    w2_num_experts = w2_scale.size(0)
    assert w2_gs.shape in (
        (w2_num_experts,),
        (w2_num_experts, 1),
    ), f"w2_gs must be (E,) or (E, 1), got {tuple(w2_gs.shape)}"
    w2_c, w2_sfmx = nvfp4.nvfp4_woq_prep_xe35(
        w2_scale, w2_gs.reshape(w2_num_experts).reciprocal().contiguous()
    )
    return {"w1_c": w1_c, "w1_sfmx": w1_sfmx, "w2_c": w2_c, "w2_sfmx": w2_sfmx}


def cutlass_fused_experts_nvfp4_woq_xe35(
    a: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    prepared: dict,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    compute_mode: str = "auto",
) -> torch.Tensor:
    """Fused MoE with NVFP4 weights and MXFP8 or BF16 computation.

    The weights stay 4-bit in memory. Auto mode uses BF16 computation.
    The explicit ``compute_mode`` argument selects ``auto``, ``mxfp8``, or ``bf16``.
    For MXFP8, each 32-wide MMA-K slice is converted immediately before BDPAS.

    Args:
        a: (M, K) bfloat16 activations. FP16 is unsupported because neither
            compute path preserves FP16 activation precision.
        w1_q: (E, 2N, K/2) uint8 packed E2M1 gate+up weights.
        w2_q: (E, K, N/2) uint8 packed E2M1 down weights.
        prepared: result of :func:`nvfp4_woq_prepare_weights_xe35`.
        topk_weights, topk_ids: routing results.
        output: optional (M, K) output buffer.
        workspace: optional uint8 scratch for the grouped GEMMs.
        compute_mode: computation precision; ``auto`` (default) uses BF16.

    Returns:
        (M, K) tensor in ``a.dtype``.
    """

    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert (
        w1_q.dtype == torch.uint8 and w2_q.dtype == torch.uint8
    ), "weights must be packed E2M1 uint8"
    assert w1_q.dim() == 3 and w2_q.dim() == 3, "weights must be 3D (E, N, K/2)"
    assert w1_q.shape[0] == w2_q.shape[0], "expert count mismatch w1/w2"
    if a.dtype != torch.bfloat16:
        raise TypeError(
            "NVFP4 WoQ fused experts requires BF16 activations; "
            f"got {a.dtype}. FP16 activation precision is unsupported."
        )
    out_dtype = a.dtype
    num_experts = w1_q.size(0)
    m = a.size(0)
    k = a.size(1)
    n = w2_q.size(2) * 2  # w2_q is (E, k, n/2)
    topk = topk_ids.size(1)
    device = a.device
    compute_mode = _nvfp4_woq_compute_mode_xe35(compute_mode)

    assert (
        w1_q.size(2) * 2 == k
    ), f"w1_q inner dim must be K/2={k // 2}, got {w1_q.size(2)}"
    assert w2_q.size(1) == k, f"w2_q must have K={k} rows, got {w2_q.size(1)}"
    assert w1_q.size(1) == 2 * n, f"w1_q must have 2N={2 * n} rows, got {w1_q.size(1)}"

    topk_ids = topk_ids.to(device=device, dtype=torch.int32).contiguous()
    a_map = torch.empty((topk_ids.numel(),), dtype=torch.int32, device=device)
    c_map = torch.empty((topk_ids.numel(),), dtype=torch.int32, device=device)
    expert_counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    problem_sizes1 = torch.zeros((num_experts, 3), dtype=torch.int32, device=device)
    problem_sizes2 = torch.zeros((num_experts, 3), dtype=torch.int32, device=device)
    prepare_moe_input(
        topk_ids,
        expert_counts,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
        num_experts,
        n,
        k,
    )
    expert_starts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    if num_experts > 1:
        expert_starts[1:] = torch.cumsum(expert_counts[:-1], dim=0).to(torch.int32)

    if workspace is None:
        workspace = torch.zeros((64 * 1024 * 1024,), dtype=torch.uint8, device=device)

    c1 = torch.zeros((m * topk, n * 2), dtype=torch.float32, device=device)
    c2 = torch.zeros((m * topk, k), dtype=torch.float32, device=device)
    if compute_mode == "bf16":
        # Both GEMMs consume BF16 directly.
        rep_a = torch.empty((m * topk, k), dtype=torch.bfloat16, device=device)
        scatter_tokens_to_experts(a, c_map, rep_a)
        nvfp4_woq_bf16_grouped_mm_xe35(
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
        nvfp4_woq_bf16_grouped_mm_xe35(
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
        rep_a_q, rep_a_s = _per_token_group_quant_mxfp8_xe35(
            rep_a, group_size=_MXFP8_BLOCK_SIZE
        )
        nvfp4_woq_mxfp8_scaled_grouped_mm_xe35(
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
        inter_q, inter_s = _per_token_group_quant_mxfp8_xe35(
            intermediate, group_size=_MXFP8_BLOCK_SIZE
        )
        nvfp4_woq_mxfp8_scaled_grouped_mm_xe35(
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
