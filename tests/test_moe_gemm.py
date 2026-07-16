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
        "relu2",
    ], "Only silu, gelu and relu2 activations are supported."

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
    elif activations == "relu2":
        act_fn = lambda x: F.relu(x) ** 2
        for i in range(w1.shape[0]):
            mask = topk_ids == i
            if mask.sum():
                gemm1 = (a[mask] @ w1[i].transpose(0, 1)).float() + b1[i].float()
                tmp = act_fn(gemm1).to(a.dtype)
                gemm2 = (tmp @ w2[i].transpose(0, 1)).float() + b2[i].float()
                out[mask] = gemm2.to(a.dtype)
    else:
        act_fn = (
            F.silu if activations == "silu" else lambda x: F.gelu(x, approximate="tanh")
        )
        for i in range(w1.shape[0]):
            mask = topk_ids == i
            if mask.sum():
                gemm1 = (a[mask] @ w1[i].transpose(0, 1)).float() + b1[i].float()
                tmp = apply_act_and_mul(gemm1.to(a.dtype), act_fn)
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
                ("relu2", None, None),
            ],  # (act_type, gemm1_alpha, gemm1_limit)
            [2.5],
        )
    )
    # Gemma4-26B-A4B TP=4 shapes: hidden=2816, intermediate=176 (shard=352=2×176).
    # GEMM1: K=2816, N=352, fuse_act=True  → narrow_n_fused branch (N≤512, avg_m>128)
    # GEMM2: K=176,  N=2816, fuse_act=False → narrow_k branch (K≤256, avg_m>128)
    # num_tokens=[1,64,256] covers avg_m≤8/16/128 branches; 1024 hits the new branches.
    + [
        (num_tokens, 8, 128, 2816, 176, False, ("silu", None, None), 2.5)
        for num_tokens in [1, 64, 256, 1024]
    ],
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

    # For relu2 activation, only test bias_dtype=False
    if act_type == "relu2" and bias_dtype != False:
        pytest.skip("relu2 only supports bias_dtype=False")

    torch.xpu.manual_seed_all(0)

    # NOTE: Nemotron3 Nano is using a non-gated MoE w/ activation type ReLU2
    gating_factor = 1 if act_type == "relu2" else 2

    rtol, atol = 1e-4, 1e-3
    a = create_random_xpu_tensor((num_tokens, hidden_size), torch.bfloat16)
    w1 = create_random_xpu_tensor(
        (num_experts, gating_factor * intermediate_size, hidden_size), torch.bfloat16
    )
    w2 = create_random_xpu_tensor(
        (num_experts, hidden_size, intermediate_size), torch.bfloat16
    )
    b1, b2 = None, None
    if bias_dtype:
        dtype = torch.bfloat16 if bias_dtype == "bfloat16" else torch.float32
        b1 = create_random_xpu_tensor(
            (num_experts, gating_factor * intermediate_size), dtype, std=0.005
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


def _pack_int4_codes(codes: torch.Tensor) -> torch.Tensor:
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)


@pytest.mark.parametrize("explicit_zero", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_moe_grouped_mm_nt_xe20_int4_zero_point(explicit_zero, dtype):
    torch.manual_seed(0)
    num_experts, rows_per_expert, gemm_n, gemm_k, group_size = 8, 2, 128, 128, 32
    total_m = num_experts * rows_per_expert

    activations = torch.randn(total_m, gemm_k, dtype=dtype) * 0.1
    codes = torch.randint(0, 16, (num_experts, gemm_n, gemm_k), dtype=torch.uint8)
    scales = torch.rand(
        num_experts, gemm_n, gemm_k // group_size, dtype=dtype
    ) * 0.02
    if explicit_zero:
        zeros = torch.randint(
            3,
            13,
            (num_experts, gemm_n, gemm_k // group_size),
            dtype=torch.int32,
        ).to(dtype)
        packed = _pack_int4_codes(codes)
    else:
        zeros = None
        packed = torch.bitwise_xor(_pack_int4_codes(codes), 0x88)

    expanded_scales = scales.repeat_interleave(group_size, dim=-1).float()
    expanded_zeros = (
        zeros.repeat_interleave(group_size, dim=-1).float()
        if zeros is not None
        else 8.0
    )
    weights = (codes.float() - expanded_zeros) * expanded_scales
    expected = torch.cat(
        [
            activations[
                expert * rows_per_expert : (expert + 1) * rows_per_expert
            ].float()
            @ weights[expert].transpose(0, 1)
            for expert in range(num_experts)
        ]
    ).to(dtype)

    output = torch.empty(total_m, gemm_n, dtype=dtype, device="xpu")
    torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20_w4a16(
        output,
        activations.to("xpu"),
        packed.view(torch.int8).to("xpu"),
        scales.to("xpu"),
        zeros.to("xpu") if zeros is not None else None,
        None,
        torch.full(
            (num_experts,), rows_per_expert, dtype=torch.int32, device="xpu"
        ),
        num_experts,
        True,
        group_size,
    )

    torch.testing.assert_close(output.cpu(), expected, rtol=5e-2, atol=2e-2)


# ---------------------------------------------------------------------------
# MXFP4 expert-weight test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_tokens,topk,num_experts,hidden_size,intermediate_size",
    list(
        itertools.product(
            [1, 33, 222],  # num_tokens
            [1, 2, 6],  # topk
            [8, 64],  # num_experts
            [128, 1024],  # hidden_size  – must be a multiple of MXFP4_BLOCK_SIZE
            [128, 512],  # intermediate_size – must be a multiple of MXFP4_BLOCK_SIZE
        )
    ),
)
def test_moe_gemm_mxfp4_weights(
    num_tokens,
    topk,
    num_experts,
    hidden_size,
    intermediate_size,
):
    """Test fused_experts with MXFP4-packed expert weights (W4A16).

    Weights are quantized to MXFP4 on CPU and passed to fused_experts as packed
    uint8 tensors together with their UE8M0 block scales via the
    ``use_mxfp4_w4a16=True`` flag.  Activations remain in BF16 throughout.

    The reference is torch_naive_moe run with the *dequantised* BF16 weights
    so that both code paths see identical effective weights; any numerical
    difference is purely from the BF16 grouped GeMM arithmetic, not from
    quantisation, and should be within the same tolerances as the BF16 test.

    activation=silu runs GEMM1 followed by the dedicated silu_and_mul kernel.
    Bias is left None here; the gpt-oss coverage below exercises per-expert
    MLP biases.
    """
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
        None,
        None,
        activations="silu",
    )

    # ---- fused_experts with packed MXFP4 weights on XPU ----
    # fused_experts expects packed weights as int8 (bitwise identical to the
    # uint8 reference packing) and raw uint8 E8M0 block scales.
    device = "xpu"
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
        w1_scale=w1_scale.to(device),
        w2_scale=w2_scale.to(device),
    )

    torch.testing.assert_close(
        torch_output, sglang_output.to("cpu"), rtol=rtol, atol=atol
    )

# ---------------------------------------------------------------------------
# MXFP4 expert-weight test — gpt-oss swiglu (ActType=2) + per-expert bias
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_tokens,topk,num_experts,hidden_size,intermediate_size,with_bias",
    list(
        itertools.product(
            [1, 33, 222],  # num_tokens
            [1, 2, 6],  # topk
            [8, 64],  # num_experts
            [128, 1024],  # hidden_size       – multiple of MXFP4_BLOCK_SIZE
            [128, 512],  # intermediate_size  – multiple of MXFP4_BLOCK_SIZE
            [False, True],  # with_bias
        )
    ),
)
def test_moe_gemm_mxfp4_weights_gpt_oss(
    num_tokens,
    topk,
    num_experts,
    hidden_size,
    intermediate_size,
    with_bias,
):
    """MXFP4-packed expert weights (W4A16) with the gpt-oss gated activation
    (swiglu_gpt_oss, ActType=2) and optional per-channel mlp1/mlp2 biases.

    This is the combination gpt-oss-20b (GptOssForCausalLM) needs: unified
    W4A16 GEMM followed by the swiglu_gpt_oss activation and optional
    per-channel mlp1/mlp2 bias. This test guards that path so a regression
    fails loudly at test time instead of at the first gpt-oss MoE forward.

    Weights are quantised to MXFP4 then dequantised for the reference, so both
    paths see identical MXFP4-rounded weights and any diff is bf16 GEMM noise.
    """
    torch.manual_seed(0)
    torch.xpu.manual_seed_all(0)

    rtol, atol = 1e-1, 1e-2

    a = create_random_cpu_tensor((num_tokens, hidden_size), torch.bfloat16)
    # w1: gate+up projection [E, 2*I, H] (interleaved g0,u0,g1,u1,... for gpt-oss);
    # w2: down projection [E, H, I].
    w1_bf16 = create_random_cpu_tensor(
        (num_experts, 2 * intermediate_size, hidden_size), torch.bfloat16
    )
    w2_bf16 = create_random_cpu_tensor(
        (num_experts, hidden_size, intermediate_size), torch.bfloat16
    )

    # Per-channel biases (float32, matching the kernel's fp32 bias accumulate).
    b1, b2 = None, None
    if with_bias:
        b1 = create_random_cpu_tensor(
            (num_experts, 2 * intermediate_size), torch.float32, std=0.005
        )
        b2 = create_random_cpu_tensor(
            (num_experts, hidden_size), torch.float32, std=0.005
        )

    score = torch.randn([num_tokens, num_experts], dtype=torch.bfloat16)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)

    # ---- quantise → dequantise so kernel + reference see the same weights ----
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
        activations="silu",
        gemm1_alpha=SWIGLU_ALPHA,
        gemm1_limit=SWIGLU_LIMIT,
    )

    device = "xpu"
    sglang_output = fused_experts(
        a.to(device),
        w1_packed.view(torch.int8).to(device),
        w2_packed.view(torch.int8).to(device),
        topk_weight.to(device),
        topk_ids.to(device),
        b1.to(device) if b1 is not None else None,
        b2.to(device) if b2 is not None else None,
        activation="silu",
        use_mxfp4_w4a16=True,
        w1_scale=w1_scale.to(device),
        w2_scale=w2_scale.to(device),
        gemm1_alpha=SWIGLU_ALPHA,
        gemm1_limit=SWIGLU_LIMIT,
    )

    torch.testing.assert_close(
        torch_output, sglang_output.to("cpu"), rtol=rtol, atol=atol
    )


# ---------------------------------------------------------------------------
# Op-level test: unified W4A16 MXFP4 vs. moe_grouped_mm_nt_xe20(dequant)
# ---------------------------------------------------------------------------
#
# Exercises the unified MXFP4 grouped GEMM op directly (no fused_experts
# orchestrator). Compares against running the non-quantized bf16 grouped GEMM
# on the dequantized weights — both paths see the same MXFP4-rounded weight
# values, so any difference is bf16 GEMM arithmetic noise, not quantization.


def _build_moe_gemm_inputs(
    num_experts: int,
    avg_m_per_expert: int,
    gemm_n: int,
    gemm_k: int,
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

    # Unified W4A16 contract: int8 packed weights and raw uint8 E8M0 scales.
    w_dq_xpu = w_dq_cpu.to("xpu")
    w_packed_xpu = w_packed_cpu.view(torch.int8).to("xpu")
    w_scale_xpu = w_scale_cpu.to("xpu")

    output_bf16 = torch.empty((total_m, gemm_n), dtype=torch.bfloat16, device="xpu")
    output_mxfp4 = torch.empty((total_m, gemm_n), dtype=torch.bfloat16, device="xpu")

    return {
        "activations": activations,
        "w_dq": w_dq_xpu,
        "w_packed": w_packed_xpu,
        "w_scale": w_scale_xpu,
        "total_rows": total_rows,
        "output_bf16": output_bf16,
        "output_mxfp4": output_mxfp4,
    }


@pytest.mark.parametrize("num_tokens_per_expert", [1, 33, 222])
@pytest.mark.parametrize("num_experts", [8])
@pytest.mark.parametrize("hidden_size", [1024])
@pytest.mark.parametrize("intermediate_size", [512])
def test_moe_grouped_mm_nt_xe20_w4a16_mxfp4_op(
    num_tokens_per_expert,
    num_experts,
    hidden_size,
    intermediate_size,
):
    """Compare the unified MXFP4 op with BF16 GEMM on dequantized weights.

    gemm_k = hidden_size (activation's inner dim)
    gemm_n = 2*intermediate_size (w1 style).
    """
    gemm_k = hidden_size
    gemm_n = 2 * intermediate_size
    assert gemm_n % 2 == 0
    assert gemm_k % 32 == 0, "gemm_k must be a multiple of MXFP4 group size"

    inputs = _build_moe_gemm_inputs(
        num_experts=num_experts,
        avg_m_per_expert=num_tokens_per_expert,
        gemm_n=gemm_n,
        gemm_k=gemm_k,
    )

    # Baseline: bf16 op on the dequantised weights.
    torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20(
        inputs["output_bf16"],
        inputs["activations"],
        inputs["w_dq"],
        None,
        inputs["total_rows"],
        num_experts,
        0,
        False,
        1.702,
        7.0,
    )

    torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20_w4a16(
        inputs["output_mxfp4"],
        inputs["activations"],
        inputs["w_packed"],
        inputs["w_scale"],
        None,
        None,
        inputs["total_rows"],
        num_experts,
        False,
        32,
    )

    torch.testing.assert_close(
        inputs["output_bf16"], inputs["output_mxfp4"], rtol=1e-1, atol=1e-2
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
