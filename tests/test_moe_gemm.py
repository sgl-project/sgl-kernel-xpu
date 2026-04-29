import itertools
import os
import sys
from typing import Callable

import pytest
import torch
import torch.nn.functional as F
from sgl_kernel import fused_experts

# Reuse the reference MXFP4 quantisation/dequantisation helpers that live in
# the dedicated unit-test file next to this one.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_per_token_group_quant_mxfp4 import MXFP4_BLOCK_SIZE
from test_per_token_group_quant_mxfp4 import dequantize_mxfp4 as _dequantize_mxfp4_2d
from test_per_token_group_quant_mxfp4 import quantize_to_mxfp4 as _quantize_mxfp4_2d


def apply_act_and_mul(
    x: torch.Tensor, act_func: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    d = x.shape[-1] // 2
    return act_func(x[..., :d]) * x[..., d:]


def create_random_xpu_tensor(shape, dtype, mean=0, std=0.01):
    """Create a random xpu tensor

    Args:
        shape: Tensor shape
        dtype: Data type
        mean: Mean value
        std: Standard deviation

    Returns:
        torch.Tensor: Randomly initialized xpu tensor
    """
    return torch.empty(shape, dtype=dtype, device="xpu").normal_(mean, std)


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
    "num_tokens,topk,num_experts,hidden_size,intermediate_size,bias_dtype,act",
    list(
        itertools.product(
            [1, 33, 222],  # num_tokens
            [1, 2, 6],  # topk
            [8, 64],  # num_experts
            [128, 1024],  # hidden_size  – must be a multiple of MXFP4_BLOCK_SIZE
            [128, 512],  # intermediate_size – must be a multiple of MXFP4_BLOCK_SIZE
            [False, "bfloat16", "float32"],  # bias_dtype
            [
                ("silu", None, None),
                ("gelu", None, None),
            ],  # (act_type, gemm1_alpha, gemm1_limit)
        )
    ),
)
def test_moe_gemm_mxfp4_weights(
    num_tokens, topk, num_experts, hidden_size, intermediate_size, bias_dtype, act
):
    """Test fused_experts with MXFP4-packed expert weights (W4A16).

    Weights are quantized to MXFP4 on CPU and passed to fused_experts as packed
    uint8 tensors together with their UE8M0 block scales via the
    ``use_mxfp4_w4a16=True`` flag.  Activations remain in BF16 throughout.

    The reference is torch_naive_moe run with the *dequantised* BF16 weights
    so that both code paths see identical effective weights; any numerical
    difference is purely from the BF16 grouped GeMM arithmetic, not from
    quantisation, and should be within the same tolerances as the BF16 test.
    """
    act_type, gemm1_alpha, gemm1_limit = act
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
    device = "xpu"
    a_xpu = a.clone().to(device)
    w1_packed_xpu = w1_packed.to(device)
    w2_packed_xpu = w2_packed.to(device)
    w1_scale_xpu = w1_scale.to(device)
    w2_scale_xpu = w2_scale.to(device)
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
        w1_scale=w1_scale_xpu,
        w2_scale=w2_scale_xpu,
    )

    torch.testing.assert_close(
        torch_output, sglang_output.to("cpu"), rtol=rtol, atol=atol
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
