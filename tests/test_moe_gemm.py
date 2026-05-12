import itertools
import sys
from typing import Callable

import pytest
import torch
import torch.nn.functional as F

# Shared MXFP4 helpers live in a dedicated module next to this file.
from mxfp4_utils import MXFP4_BLOCK_SIZE
from mxfp4_utils import dequantize_mxfp4_2d as _dequantize_mxfp4_2d
from mxfp4_utils import quantize_mxfp4_2d as _quantize_mxfp4_2d
from sgl_kernel import fused_experts


def apply_act_and_mul(
    x: torch.Tensor, act_func: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    d = x.shape[-1] // 2
    return act_func(x[..., :d]) * x[..., d:]


def create_random_xpu_tensor(shape, dtype, mean=0, std=0.01):
    return torch.empty(shape, dtype=dtype, device="xpu").normal_(mean, std)


def create_random_cpu_tensor(shape, dtype, mean=0, std=0.01):
    return torch.empty(shape, dtype=dtype, device="cpu").normal_(mean, std)


# GPT-OSS SwiGLU parameters (matches kernel defaults)
SWIGLU_ALPHA = 1.702
SWIGLU_LIMIT = 7.0


def swiglu_gpt_oss_sigmoid_alpha(
    x: torch.Tensor,
    alpha: float = SWIGLU_ALPHA,
    limit: float = SWIGLU_LIMIT,
) -> torch.Tensor:
    """Matches the kernel's swiglu_gpt_oss_sigmoid_alpha formula:
        gate = clamp(gate, -inf, limit)
        up   = clamp(up,   -limit, limit)
        out  = gate * sigmoid(gate * alpha) * (up + 1)

    Args:
        x: Input tensor of shape (..., 2*N).
           x is in [g0, u0, g1, u1, ...] layout
           (model weight format).
    Note: currently, only GPT-OSS uses this variant.
    """
    gate = x[..., 0::2].float()  # even columns
    up = x[..., 1::2].float()  # odd columns
    gate = gate.clamp(max=limit)
    up = up.clamp(-limit, limit)
    return (gate * torch.sigmoid(gate * alpha) * (up + 1.0)).to(x.dtype)


def torch_naive_moe(
    a,
    w1,
    w2,
    topk_ids,
    topk_weight,
    topk,
    b1,
    b2,
    activations="silu",
    gemm1_alpha: float = None,
    gemm1_limit: float = None,
    routed_scaling_factor=None,
):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    b1 = (
        b1
        if b1 is not None
        else torch.zeros(w1.shape[:2], dtype=a.dtype, device=a.device)
    )
    b2 = (
        b2
        if b2 is not None
        else torch.zeros(w2.shape[:2], dtype=a.dtype, device=a.device)
    )
    assert activations in [
        "silu",
        "gelu",
    ], "Only silu and gelu activations are supported."

    is_swiglu_gpt_oss = (
        activations == "silu" and gemm1_alpha is not None and gemm1_limit is not None
    )
    if is_swiglu_gpt_oss:
        # w1 is in interleaved layout [g0, u0, g1, u1, ...] (model weight format).
        # The GEMM output is therefore also interleaved along the N dimension.
        act_fn = lambda x: swiglu_gpt_oss_sigmoid_alpha(x, gemm1_alpha, gemm1_limit)
        for i in range(w1.shape[0]):
            mask = topk_ids == i
            if mask.sum():
                # Matches kernel behavior: accumulator is float32, bias is float32,
                gemm1 = (a[mask] @ w1[i].transpose(0, 1)).float() + b1[i].float()
                tmp = act_fn(gemm1).to(a.dtype)
                # Same for GEMM2.
                gemm2 = (tmp @ w2[i].transpose(0, 1)).float() + b2[i].float()
                out[mask] = gemm2.to(a.dtype)
    else:
        act_func = (
            F.silu if activations == "silu" else lambda x: F.gelu(x, approximate="tanh")
        )
        for i in range(w1.shape[0]):
            mask = topk_ids == i
            if mask.sum():
                gemm1 = (a[mask] @ w1[i].transpose(0, 1)).float() + b1[i].float()
                tmp = apply_act_and_mul(gemm1.to(a.dtype), act_func)
                gemm2 = (tmp @ w2[i].transpose(0, 1)).float() + b2[i].float()
                out[mask] = gemm2.to(a.dtype)

    result = (
        out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)

    if routed_scaling_factor is not None:
        result = result * routed_scaling_factor

    return result


@pytest.mark.parametrize(
    "num_tokens,topk,num_experts,hidden_size,intermediate_size,bias_dtype,act,routed_scaling_factor",
    list(
        itertools.product(
            [1, 4, 33, 64, 222],  # num_tokens
            [1, 2, 6],  # topk
            [8, 64],  #  num_experts
            [1024, 4096],  # hidden_size
            [512, 1024, 4096],  # intermediate_size
            [False, "bfloat16", "float32"],  # bias_dtype
            [
                ("silu", None, None),
                ("gelu", None, None),
                ("silu", SWIGLU_ALPHA, SWIGLU_LIMIT),  # swiglu_gpt_oss
            ],  # (act_type, gemm1_alpha, gemm1_limit)
            [2.5],
        )
    ),
)
def test_moe_gemm(
    num_tokens,
    topk,
    num_experts,
    hidden_size,
    intermediate_size,
    bias_dtype,
    act,
    routed_scaling_factor,
):
    act_type, gemm1_alpha, gemm1_limit = act
    torch.xpu.manual_seed_all(0)

    rtol, atol = 1e-4, 1e-3
    a = create_random_xpu_tensor((num_tokens, hidden_size), torch.bfloat16)
    w1 = create_random_xpu_tensor(
        (num_experts, 2 * intermediate_size, hidden_size), torch.bfloat16
    )
    w2 = create_random_xpu_tensor(
        (num_experts, hidden_size, intermediate_size), torch.bfloat16
    )
    b1, b2 = None, None
    if bias_dtype:
        dtype = torch.bfloat16 if bias_dtype == "bfloat16" else torch.float32
        b1 = create_random_xpu_tensor(
            (num_experts, 2 * intermediate_size), dtype, std=0.005
        )
        b2 = create_random_xpu_tensor((num_experts, hidden_size), dtype, std=0.005)
    score = torch.randn([num_tokens, num_experts], dtype=torch.bfloat16).to("xpu")

    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    torch_output = torch_naive_moe(
        a,
        w1,
        w2,
        topk_ids,
        topk_weight,
        topk,
        b1,
        b2,
        activations=act_type,
        gemm1_alpha=gemm1_alpha,
        gemm1_limit=gemm1_limit,
        routed_scaling_factor=routed_scaling_factor,
    )
    sglang_output = fused_experts(
        a,
        w1,
        w2,
        topk_weight,
        topk_ids,
        b1,
        b2,
        activation=act_type,
        gemm1_alpha=gemm1_alpha,
        gemm1_limit=gemm1_limit,
        routed_scaling_factor=routed_scaling_factor,
    )
    torch.testing.assert_close(torch_output, sglang_output, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# MXFP4 expert-weight helpers (W4A16)
# ---------------------------------------------------------------------------


def _quantize_weights_mxfp4(
    w: torch.Tensor,
    block_size: int = MXFP4_BLOCK_SIZE,
):
    """Quantize a 3-D expert weight tensor [E, rows, cols] to MXFP4 on CPU.

    The last dimension is quantised in blocks of *block_size* elements.
    Both *cols* and *block_size* must be compatible with MXFP4 packing
    (cols divisible by block_size and by 2).

    Returns:
        packed  – [E, rows, cols // 2] uint8, two E2M1 nibbles per byte
                  (low nibble = first element, matching pack_fp4 convention).
        scales  – [E, rows, cols // block_size] uint8, UE8M0 format
                  (stored_byte = biased_exp + 127).
    """
    E, rows, cols = w.shape
    assert (
        cols % block_size == 0
    ), f"last dim {cols} must be divisible by block_size {block_size}"
    flat = w.reshape(E * rows, cols).float().cpu()
    packed_flat, scales_flat = _quantize_mxfp4_2d(flat, block_size)
    return (
        packed_flat.reshape(E, rows, cols // 2),
        scales_flat.reshape(E, rows, cols // block_size),
    )


def _dequantize_weights_mxfp4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    block_size: int = MXFP4_BLOCK_SIZE,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize 3-D packed MXFP4 weights [E, rows, packed_cols] to BF16 on CPU.

    Returns a [E, rows, cols] tensor where cols = packed_cols * 2.
    """
    E, rows, packed_cols = packed.shape
    cols = packed_cols * 2
    flat_packed = packed.reshape(E * rows, packed_cols).cpu()
    flat_scales = scales.reshape(E * rows, cols // block_size).cpu()
    flat_dq = _dequantize_mxfp4_2d(
        flat_packed, flat_scales, dtype=dtype, block_size=block_size
    )
    return flat_dq.reshape(E, rows, cols)


# ---------------------------------------------------------------------------
# MXFP4 expert-weight test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_tokens,topk,num_experts,hidden_size,intermediate_size,bias_dtype,activation,use_fused_kernel",
    list(
        itertools.product(
            [1, 33, 222],  # num_tokens
            [1, 2, 6],  # topk
            [8, 64],  # num_experts
            [128, 1024],  # hidden_size  – must be a multiple of MXFP4_BLOCK_SIZE
            [128, 512],  # intermediate_size – must be a multiple of MXFP4_BLOCK_SIZE
            [False, "bfloat16", "float32"],  # bias_dtype
            ["silu", "gelu"],  # activation type
            [False, True],  # use_fused_mxfp4_kernel
        )
    ),
)
def test_moe_gemm_mxfp4_weights(
    num_tokens,
    topk,
    num_experts,
    hidden_size,
    intermediate_size,
    bias_dtype,
    activation,
    use_fused_kernel,
):
    # The tile-fused kernel is built with a pruned template matrix
    # (ActType=0 silu, WithBias=false) to keep Level Zero module pressure
    # in budget under TP>1. See src/GroupGemmMxfp4Xe20.cmake.
    if use_fused_kernel and (activation != "silu" or bias_dtype):
        pytest.skip("fused MXFP4 kernel currently built silu + no-bias only")
    """Test fused_experts with MXFP4-packed expert weights (W4A16).

    Weights are quantized to MXFP4 on CPU and passed to fused_experts as packed
    uint8 tensors together with their UE8M0 block scales via the
    ``use_mxfp4_w4a16=True`` flag.  Activations remain in BF16 throughout.

    The reference is torch_naive_moe run with the *dequantised* BF16 weights
    so that both code paths see identical effective weights; any numerical
    difference is purely from the BF16 grouped GeMM arithmetic, not from
    quantisation, and should be within the same tolerances as the BF16 test.

    Parametrised over use_fused_mxfp4_kernel so both the legacy path
    (Python dequant + bf16 GEMM) and the tile-fused path
    (moe_grouped_mm_nt_xe20_mxfp4) are covered at every shape.
    """
    act_type = activation
    torch.manual_seed(0)
    torch.xpu.manual_seed_all(0)

    rtol, atol = 1e-1, 1e-2

    a = create_random_cpu_tensor((num_tokens, hidden_size), torch.bfloat16)
    # w1: gate+up projection  [E, 2*I, H];  w2: down projection  [E, H, I]
    w1_bf16 = create_random_cpu_tensor(
        (num_experts, 2 * intermediate_size, hidden_size), torch.bfloat16
    )
    w2_bf16 = create_random_cpu_tensor(
        (num_experts, hidden_size, intermediate_size), torch.bfloat16
    )
    b1, b2 = None, None
    if bias_dtype:
        dtype = torch.bfloat16 if bias_dtype == "bfloat16" else torch.float32
        b1 = create_random_cpu_tensor(
            (num_experts, 2 * intermediate_size), dtype, std=0.005
        )
        b2 = create_random_cpu_tensor((num_experts, hidden_size), dtype, std=0.005)

    score = torch.randn([num_tokens, num_experts], dtype=torch.bfloat16)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)

    # ---- Reference: quantise w1/w2 → dequantise to get MXFP4-rounded BF16 ----
    # Both the kernel and the reference operate on these rounded weights, so any
    # discrepancy is purely arithmetic (not quantisation error).
    w1_packed, w1_scale = _quantize_weights_mxfp4(w1_bf16)
    w2_packed, w2_scale = _quantize_weights_mxfp4(w2_bf16)
    w1_dq = _dequantize_weights_mxfp4(w1_packed, w1_scale)
    w2_dq = _dequantize_weights_mxfp4(w2_packed, w2_scale)

    torch_output = torch_naive_moe(
        a,
        w1_dq,
        w2_dq,
        topk_ids,
        topk_weight,
        topk,
        b1,
        b2,
        activations=act_type,
    )

    # ---- fused_experts with packed MXFP4 weights on XPU ----
    # fused_experts expects packed weights as int8 (bitwise identical to the
    # uint8 reference packing) and scales as a fp32 direct multiplier
    # (decoded from UE8M0).
    device = "xpu"
    a_xpu = a.clone().to(device)
    w1_packed_xpu = w1_packed.view(torch.int8).to(device)
    w2_packed_xpu = w2_packed.view(torch.int8).to(device)
    # Decode UE8M0 bytes to fp32 direct multipliers and transpose to K-outer
    # [E, K/32, N] layout for the kernel's coalesced scale load.
    w1_scale_xpu = (
        torch.exp2((w1_scale.to(torch.int32) - 127).to(torch.float32))
        .transpose(1, 2)
        .contiguous()
        .to(device)
    )
    w2_scale_xpu = (
        torch.exp2((w2_scale.to(torch.int32) - 127).to(torch.float32))
        .transpose(1, 2)
        .contiguous()
        .to(device)
    )
    topk_weight_xpu = topk_weight.clone().to(device)
    topk_ids_xpu = topk_ids.clone().to(device)
    b1_xpu = b1.clone().to(device) if b1 is not None else None
    b2_xpu = b2.clone().to(device) if b2 is not None else None

    sglang_output = fused_experts(
        a_xpu,
        w1_packed_xpu,
        w2_packed_xpu,
        topk_weight_xpu,
        topk_ids_xpu,
        b1_xpu,
        b2_xpu,
        activation=act_type,
        use_mxfp4_w4a16=True,
        use_fused_mxfp4_kernel=use_fused_kernel,
        w1_scale=w1_scale_xpu,
        w2_scale=w2_scale_xpu,
    )

    torch.testing.assert_close(
        torch_output, sglang_output.to("cpu"), rtol=rtol, atol=atol
    )


# ---------------------------------------------------------------------------
# Op-level test: moe_grouped_mm_nt_xe20_mxfp4 vs. moe_grouped_mm_nt_xe20(dequant)
# ---------------------------------------------------------------------------
#
# Exercises the tile-fused MXFP4 grouped GEMM op directly (no fused_experts
# orchestrator). Compares against running the non-quantized bf16 grouped GEMM
# on the dequantized weights — both paths see the same MXFP4-rounded weight
# values, so any difference is bf16 GEMM arithmetic noise, not quantization.


def _build_moe_gemm_inputs(
    num_experts: int,
    avg_m_per_expert: int,
    gemm_n: int,
    gemm_k: int,
    with_bias: bool,
    fuse_act: bool,
    seed: int = 0,
):
    """Construct (activations, bf16_weights, mxfp4_packed, mxfp4_scales,
    total_rows_for_experts, bias_or_none) on XPU for the op-level test."""
    torch.manual_seed(seed)
    torch.xpu.manual_seed_all(seed)

    # Equal rows per expert for simplicity.
    total_m = num_experts * avg_m_per_expert
    total_rows = torch.full(
        (num_experts,), avg_m_per_expert, dtype=torch.int32, device="xpu"
    )

    activations = create_random_xpu_tensor((total_m, gemm_k), torch.bfloat16)

    # Build bf16 weights on CPU, quantize to mxfp4 there, then move to XPU.
    w_bf16_cpu = create_random_cpu_tensor((num_experts, gemm_n, gemm_k), torch.bfloat16)
    w_packed_cpu, w_scale_cpu = _quantize_weights_mxfp4(w_bf16_cpu)
    w_dq_cpu = _dequantize_weights_mxfp4(w_packed_cpu, w_scale_cpu)

    # Fused op contract: int8 packed weights, fp32 direct-multiplier scales
    # in K-outer [E, K/32, N] layout.
    w_dq_xpu = w_dq_cpu.to("xpu")
    w_packed_xpu = w_packed_cpu.view(torch.int8).to("xpu")
    w_scale_xpu = (
        torch.exp2((w_scale_cpu.to(torch.int32) - 127).to(torch.float32))
        .transpose(1, 2)
        .contiguous()
        .to("xpu")
    )

    bias = None
    if with_bias:
        bias = create_random_xpu_tensor((num_experts, gemm_n), torch.float32, std=0.005)

    out_cols = gemm_n // 2 if fuse_act else gemm_n
    output_bf16 = torch.empty((total_m, out_cols), dtype=torch.bfloat16, device="xpu")
    output_mxfp4 = torch.empty((total_m, out_cols), dtype=torch.bfloat16, device="xpu")

    return {
        "activations": activations,
        "w_dq": w_dq_xpu,
        "w_packed": w_packed_xpu,
        "w_scale": w_scale_xpu,
        "total_rows": total_rows,
        "bias": bias,
        "output_bf16": output_bf16,
        "output_mxfp4": output_mxfp4,
    }


@pytest.mark.parametrize("num_tokens_per_expert", [1, 33, 222])
@pytest.mark.parametrize("num_experts", [8])
@pytest.mark.parametrize("hidden_size", [1024])
@pytest.mark.parametrize("intermediate_size", [512])
@pytest.mark.parametrize("activation_type", [0, 1, 2])  # silu, gelu, swiglu_gpt_oss
@pytest.mark.parametrize("fuse_act", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
def test_moe_grouped_mm_nt_xe20_mxfp4_op(
    num_tokens_per_expert,
    num_experts,
    hidden_size,
    intermediate_size,
    activation_type,
    fuse_act,
    with_bias,
):
    """Direct op-level comparison: mxfp4 fused op vs. bf16 op on dequant weights.

    gemm_k = hidden_size (activation's inner dim)
    gemm_n = 2*intermediate_size (w1 style) — we pick one shape for simplicity
    For fuse_act=True the output has N/2 cols, so gemm_n must be even.
    """
    # See src/GroupGemmMxfp4Xe20.cmake: the fused kernel is pruned to
    # ActType=0 silu and WithBias=false to keep L0 module pressure sane.
    if activation_type != 0 or with_bias:
        pytest.skip("fused MXFP4 kernel currently built silu + no-bias only")
    gemm_k = hidden_size
    gemm_n = 2 * intermediate_size
    assert gemm_n % 2 == 0
    assert gemm_k % 32 == 0, "gemm_k must be a multiple of MXFP4 group size"

    inputs = _build_moe_gemm_inputs(
        num_experts=num_experts,
        avg_m_per_expert=num_tokens_per_expert,
        gemm_n=gemm_n,
        gemm_k=gemm_k,
        with_bias=with_bias,
        fuse_act=fuse_act,
    )

    # Baseline: bf16 op on the dequantised weights.
    torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20(
        inputs["output_bf16"],
        inputs["activations"],
        inputs["w_dq"],
        inputs["bias"],
        inputs["total_rows"],
        num_experts,
        activation_type,
        fuse_act,
        1.702,
        7.0,
    )

    # Fused MXFP4 path.
    torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20_mxfp4(
        inputs["output_mxfp4"],
        inputs["activations"],
        inputs["w_packed"],
        inputs["w_scale"],
        inputs["bias"],
        inputs["total_rows"],
        num_experts,
        activation_type,
        fuse_act,
        1.702,
        7.0,
    )

    torch.testing.assert_close(
        inputs["output_bf16"], inputs["output_mxfp4"], rtol=1e-1, atol=1e-2
    )


# ---------------------------------------------------------------------------
# End-to-end: fused_experts with use_fused_mxfp4_kernel=True
# ---------------------------------------------------------------------------
# Same reference pipeline as test_moe_gemm_mxfp4_weights (torch_naive_moe on
# dequantised weights); the sglang side routes through
# moe_grouped_mm_nt_xe20_mxfp4 and skips the intermediate bf16 weight.


@pytest.mark.parametrize(
    "num_tokens,topk,num_experts,hidden_size,intermediate_size",
    list(
        itertools.product(
            [33, 128],  # avg_m per expert ∈ (8, 128] with num_experts=8
            [1, 2],
            [8],
            [1024],  # small enough to stay in the dispatcher's small-weight branch
            [512, 1024],
        )
    ),
)
def test_fused_experts_mxfp4_fused_kernel(
    num_tokens, topk, num_experts, hidden_size, intermediate_size
):
    """End-to-end fused_experts(use_mxfp4_w4a16=True, use_fused_mxfp4_kernel=True).

    Reference is torch_naive_moe on dequantised BF16 weights.  The sglang
    path runs the tile-fused MXFP4 grouped-GEMM directly (no intermediate
    bf16 weight materialisation in Python).
    """
    torch.manual_seed(0)
    torch.xpu.manual_seed_all(0)
    rtol, atol = 1e-1, 1e-2

    a = create_random_cpu_tensor((num_tokens, hidden_size), torch.bfloat16)
    w1_bf16 = create_random_cpu_tensor(
        (num_experts, 2 * intermediate_size, hidden_size), torch.bfloat16
    )
    w2_bf16 = create_random_cpu_tensor(
        (num_experts, hidden_size, intermediate_size), torch.bfloat16
    )

    score = torch.randn([num_tokens, num_experts], dtype=torch.bfloat16)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)

    w1_packed, w1_scale = _quantize_weights_mxfp4(w1_bf16)
    w2_packed, w2_scale = _quantize_weights_mxfp4(w2_bf16)
    w1_dq = _dequantize_weights_mxfp4(w1_packed, w1_scale)
    w2_dq = _dequantize_weights_mxfp4(w2_packed, w2_scale)

    torch_output = torch_naive_moe(
        a, w1_dq, w2_dq, topk_ids, topk_weight, topk, None, None, activations="silu"
    )

    # fused_experts expects int8 packed weights and fp32 K-outer scales.
    device = "xpu"
    w1_scale_fp32 = (
        torch.exp2((w1_scale.to(torch.int32) - 127).to(torch.float32))
        .transpose(1, 2)
        .contiguous()
    )
    w2_scale_fp32 = (
        torch.exp2((w2_scale.to(torch.int32) - 127).to(torch.float32))
        .transpose(1, 2)
        .contiguous()
    )
    sglang_output = fused_experts(
        a.to(device),
        w1_packed.view(torch.int8).to(device),
        w2_packed.view(torch.int8).to(device),
        topk_weight.to(device),
        topk_ids.to(device),
        None,
        None,
        activation="silu",
        use_mxfp4_w4a16=True,
        use_fused_mxfp4_kernel=True,
        w1_scale=w1_scale_fp32.to(device),
        w2_scale=w2_scale_fp32.to(device),
    )

    torch.testing.assert_close(
        torch_output, sglang_output.to("cpu"), rtol=rtol, atol=atol
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
