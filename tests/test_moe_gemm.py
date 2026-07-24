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

    The tile-fused MXFP4 kernel is currently built silu + no-bias only
    (see src/GroupGemmMxfp4W4A16Xe20.cmake — pruned to keep L0 module
    pressure sane under TP>1), so this test pins activation=silu and bias=None.
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
    # uint8 reference packing) and scales as a fp32 direct multiplier
    # (decoded from UE8M0).
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
        w1_scale=torch.exp2((w1_scale.to(torch.int32) - 127).to(torch.float32)).to(
            device
        ),
        w2_scale=torch.exp2((w2_scale.to(torch.int32) - 127).to(torch.float32)).to(
            device
        ),
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

    This is the combination gpt-oss-20b (GptOssForCausalLM) needs: the tile-
    fused MXFP4 grouped-GEMM must dispatch activation_type=2 + with_bias=true.
    The instantiation matrix in src/GroupGemmMxfp4W4A16Xe20.cmake was originally
    pruned to {silu, deepseek_v4} x {no-bias}; this test guards the re-enabled
    ActType=2 / WithBias path so a future re-prune that drops it fails loudly
    here instead of aborting at the first gpt-oss MoE forward with:
        RuntimeError: mxfp4 fused kernel built with ActType=0 (silu) and
        ActType=4 (swiglu_deepseek_v4) only; got ActType=2.

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
        w1_scale=torch.exp2((w1_scale.to(torch.int32) - 127).to(torch.float32)).to(
            device
        ),
        w2_scale=torch.exp2((w2_scale.to(torch.int32) - 127).to(torch.float32)).to(
            device
        ),
        gemm1_alpha=SWIGLU_ALPHA,
        gemm1_limit=SWIGLU_LIMIT,
    )

    torch.testing.assert_close(
        torch_output, sglang_output.to("cpu"), rtol=rtol, atol=atol
    )


# ---------------------------------------------------------------------------
# Op-level test: moe_grouped_mm_nt_xe20_mxfp4_w4a16 vs. moe_grouped_mm_nt_xe20(dequant)
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

    # Fused op contract: int8 packed weights, fp32 direct-multiplier scales.
    w_dq_xpu = w_dq_cpu.to("xpu")
    w_packed_xpu = w_packed_cpu.view(torch.int8).to("xpu")
    w_scale_xpu = torch.exp2((w_scale_cpu.to(torch.int32) - 127).to(torch.float32)).to(
        "xpu"
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
@pytest.mark.parametrize("fuse_act", [False, True])
def test_moe_grouped_mm_nt_xe20_mxfp4_w4a16_op(
    num_tokens_per_expert,
    num_experts,
    hidden_size,
    intermediate_size,
    fuse_act,
):
    """Direct op-level comparison: mxfp4 fused op vs. bf16 op on dequant weights.

    gemm_k = hidden_size (activation's inner dim)
    gemm_n = 2*intermediate_size (w1 style) — we pick one shape for simplicity
    For fuse_act=True the output has N/2 cols, so gemm_n must be even.

    The fused MXFP4 kernel is built silu + no-bias only (see
    src/GroupGemmMxfp4W4A16Xe20.cmake — pruned to keep L0 module pressure
    sane under TP>1), so this op-level test pins activation=silu and
    bias=None.
    """
    activation_type = 0  # silu
    gemm_k = hidden_size
    gemm_n = 2 * intermediate_size
    assert gemm_n % 2 == 0
    assert gemm_k % 32 == 0, "gemm_k must be a multiple of MXFP4 group size"

    inputs = _build_moe_gemm_inputs(
        num_experts=num_experts,
        avg_m_per_expert=num_tokens_per_expert,
        gemm_n=gemm_n,
        gemm_k=gemm_k,
        with_bias=False,
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
    torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20_mxfp4_w4a16(
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
# FP8 (E4M3) W8A8 expert-weight helpers
# ---------------------------------------------------------------------------
#
# Quantization is done with plain torch.float8_e4m3fn casts (a numeric cast,
# not a call into the kernel-under-test), so these helpers are independent of
# sgl_kernel and can run entirely on CPU. This mirrors the MXFP4 test's
# philosophy: both the kernel and the reference see the *same rounded*
# values, so any remaining numerical difference is GEMM arithmetic noise
# (bf16 reference vs. the kernel's internal fp16 compute - see
# moe_mainloop.hpp for why fp16), not quantization error.

FP8_E4M3_MAX = 448.0
FP8_BLOCK_SIZE = 128  # matches FP8_GROUP_SIZE_K in moe_mainloop.hpp


def _quant_dequant_fp8_per_token(x: torch.Tensor):
    """Per-token (per-row) fp8 e4m3 quantize + dequantize, matching the
    per_token_quant_fp8 kernel's scale formula (rowmax(|x|) / 448).

    Returns (scale [tokens], dequantized tensor in x's original dtype).
    """
    x_f32 = x.float()
    amax = x_f32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / FP8_E4M3_MAX
    q = (x_f32 / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    dq = (q.float() * scale).to(x.dtype)
    return scale.squeeze(-1), dq


def _quant_dequant_fp8_block(w: torch.Tensor, block_size: int = FP8_BLOCK_SIZE):
    """2-D block (e.g. DeepSeek-style 128x128) fp8 e4m3 quantize + dequantize
    for a 3-D expert weight tensor [E, N, K]. N and K must be multiples of
    block_size.

    Returns (scale [E, N/block_size, K/block_size] fp32, dequantized tensor
    in w's original dtype).
    """
    E, N, K = w.shape
    assert (
        N % block_size == 0 and K % block_size == 0
    ), f"N={N} and K={K} must both be multiples of block_size={block_size}"
    w_f32 = w.float().reshape(
        E, N // block_size, block_size, K // block_size, block_size
    )
    amax = w_f32.abs().amax(dim=(2, 4), keepdim=True).clamp(min=1e-12)
    scale = amax / FP8_E4M3_MAX
    q = (w_f32 / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    dq = (q * scale).reshape(E, N, K).to(w.dtype)
    return scale.reshape(E, N // block_size, K // block_size), dq


def torch_naive_moe_fp8_w8a8(
    a,
    w1_dq,
    w2_dq,
    topk_ids,
    topk_weight,
    topk,
    b1,
    b2,
    routed_scaling_factor=None,
):
    """Reference for the fp8 W8A8 path. Unlike torch_naive_moe, this models
    the *actual* two-stage quantization pipeline fused_experts(use_fp8_w8a8=
    True) runs: activations are fp8-quantized per-token before GEMM1, AND the
    post-silu intermediate is fp8-quantized per-token again before GEMM2 (the
    kernel's activation scale is per-token and applied once per GEMM - see
    moe_mainloop.hpp). w1_dq/w2_dq must already be the dequantized (rounded)
    weights so weight-quantization noise is factored out the same way the
    MXFP4 reference does it.
    """
    B, D = a.shape
    a_rep = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    _, a_dq = _quant_dequant_fp8_per_token(a_rep)

    out = torch.zeros(B * topk, w2_dq.shape[1], dtype=a.dtype, device=a.device)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    b1 = (
        b1
        if b1 is not None
        else torch.zeros(w1_dq.shape[:2], dtype=torch.float32, device=a.device)
    )
    b2 = (
        b2
        if b2 is not None
        else torch.zeros(w2_dq.shape[:2], dtype=torch.float32, device=a.device)
    )

    for i in range(w1_dq.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            gemm1 = (a_dq[mask].float() @ w1_dq[i].float().transpose(0, 1)) + b1[
                i
            ].float()
            tmp = apply_act_and_mul(gemm1.to(a.dtype), F.silu)
            _, tmp_dq = _quant_dequant_fp8_per_token(tmp)
            gemm2 = (tmp_dq.float() @ w2_dq[i].float().transpose(0, 1)) + b2[i].float()
            out[mask] = gemm2.to(a.dtype)

    result = (
        out.view(B, -1, w2_dq.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)
    if routed_scaling_factor is not None:
        result = result * routed_scaling_factor
    return result


# ---------------------------------------------------------------------------
# FP8 W8A8 expert-weight test (fused_experts level)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_tokens,topk,num_experts,hidden_size,intermediate_size,with_bias",
    list(
        itertools.product(
            [1, 8, 17, 130],  # num_tokens - covers avg_m<=8/16/32/128 tile tiers
            [1, 2],  # topk
            [8, 16],  # num_experts (kernel requires a multiple of 8)
            [256, 512],  # hidden_size - must be a multiple of FP8_BLOCK_SIZE
            [256],  # intermediate_size - must be a multiple of FP8_BLOCK_SIZE
            [False, True],  # with_bias
        )
    ),
)
def test_moe_gemm_fp8_w8a8_weights(
    num_tokens,
    topk,
    num_experts,
    hidden_size,
    intermediate_size,
    with_bias,
):
    """Test fused_experts with fp8 e4m3 W8A8 expert weights.

    Weights are block-quantized (128x128, DeepSeek-style) then dequantized
    for the reference so both paths see identical fp8-rounded weights; the
    reference (torch_naive_moe_fp8_w8a8) also replicates the kernel's
    per-token activation quantization before each of the two expert GEMMs.
    Any remaining numerical difference should be GEMM arithmetic noise only.

    v1 of the fp8 W8A8 kernel only supports activation=silu and 2-D
    block-quant weight scales (see GroupGemmFp8W8A8Xe20.cpp/.cmake and
    xpu_fp8_moe_minimal_plan.md), so this test pins those.
    """
    torch.manual_seed(0)
    torch.xpu.manual_seed_all(0)

    rtol, atol = 1e-1, 1e-2

    a = create_random_cpu_tensor((num_tokens, hidden_size), torch.bfloat16)
    # w1: gate+up projection [E, 2*I, H]; w2: down projection [E, H, I].
    w1_bf16 = create_random_cpu_tensor(
        (num_experts, 2 * intermediate_size, hidden_size), torch.bfloat16
    )
    w2_bf16 = create_random_cpu_tensor(
        (num_experts, hidden_size, intermediate_size), torch.bfloat16
    )

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

    w1_scale, w1_dq = _quant_dequant_fp8_block(w1_bf16)
    w2_scale, w2_dq = _quant_dequant_fp8_block(w2_bf16)

    torch_output = torch_naive_moe_fp8_w8a8(
        a,
        w1_dq,
        w2_dq,
        topk_ids,
        topk_weight,
        topk,
        b1,
        b2,
    )

    device = "xpu"
    sglang_output = fused_experts(
        a.to(device),
        w1_dq.to(torch.float8_e4m3fn).to(device),
        w2_dq.to(torch.float8_e4m3fn).to(device),
        topk_weight.to(device),
        topk_ids.to(device),
        b1.to(device) if b1 is not None else None,
        b2.to(device) if b2 is not None else None,
        activation="silu",
        use_fp8_w8a8=True,
        w1_scale=w1_scale.to(device),
        w2_scale=w2_scale.to(device),
    )

    torch.testing.assert_close(
        torch_output, sglang_output.to("cpu"), rtol=rtol, atol=atol
    )


# ---------------------------------------------------------------------------
# Op-level test: moe_grouped_mm_nt_xe20_fp8_w8a8 vs. moe_grouped_mm_nt_xe20(dequant)
# ---------------------------------------------------------------------------
#
# Exercises the fp8 W8A8 grouped GEMM op directly (single GEMM, no fused
# silu-mul in between two stages), so both paths can be fed the exact same
# already-quantized-and-dequantized activations/weights and compared
# directly - no need to model a mid-network requantization step like the
# fused_experts-level test above.


def _build_moe_gemm_inputs_fp8(
    num_experts: int,
    avg_m_per_expert: int,
    gemm_n: int,
    gemm_k: int,
    with_bias: bool,
    fuse_act: bool,
    seed: int = 0,
):
    torch.manual_seed(seed)
    torch.xpu.manual_seed_all(seed)

    total_m = num_experts * avg_m_per_expert
    total_rows = torch.full(
        (num_experts,), avg_m_per_expert, dtype=torch.int32, device="xpu"
    )

    a_bf16_cpu = create_random_cpu_tensor((total_m, gemm_k), torch.bfloat16)
    a_scale_cpu, a_dq_cpu = _quant_dequant_fp8_per_token(a_bf16_cpu)

    w_bf16_cpu = create_random_cpu_tensor((num_experts, gemm_n, gemm_k), torch.bfloat16)
    w_scale_cpu, w_dq_cpu = _quant_dequant_fp8_block(w_bf16_cpu)
    # The op expects one weight-scale row per N (see
    # moe_mainloop.hpp/_expand_fp8_block_scale_to_per_row); expand the 2-D
    # block scale the same way python/sgl_kernel/moe.py does.
    w_scale_per_row_cpu = w_scale_cpu.repeat_interleave(FP8_BLOCK_SIZE, dim=1)

    bias = None
    if with_bias:
        bias = create_random_xpu_tensor((num_experts, gemm_n), torch.float32, std=0.005)

    out_cols = gemm_n // 2 if fuse_act else gemm_n
    output_bf16 = torch.empty((total_m, out_cols), dtype=torch.bfloat16, device="xpu")
    output_fp8 = torch.empty((total_m, out_cols), dtype=torch.bfloat16, device="xpu")

    return {
        "a_dq": a_dq_cpu.to("xpu"),
        "a_fp8": a_dq_cpu.to(torch.float8_e4m3fn).to("xpu"),
        "a_scale": a_scale_cpu.to("xpu"),
        "w_dq": w_dq_cpu.to("xpu"),
        "w_fp8": w_dq_cpu.to(torch.float8_e4m3fn).to("xpu"),
        "w_scale_per_row": w_scale_per_row_cpu.to("xpu"),
        "total_rows": total_rows,
        "bias": bias,
        "output_bf16": output_bf16,
        "output_fp8": output_fp8,
    }


@pytest.mark.parametrize("num_tokens_per_expert", [1, 17, 130])
@pytest.mark.parametrize("num_experts", [8])
@pytest.mark.parametrize("hidden_size", [256])
@pytest.mark.parametrize("intermediate_size", [256])
@pytest.mark.parametrize("fuse_act", [False, True])
def test_moe_grouped_mm_nt_xe20_fp8_w8a8_op(
    num_tokens_per_expert,
    num_experts,
    hidden_size,
    intermediate_size,
    fuse_act,
):
    """Direct op-level comparison: fp8 W8A8 fused op vs. bf16 op, both fed
    the same fp8-rounded (quantize+dequantize) activations/weights.

    v1 of the fp8 W8A8 kernel is silu + no-unfused-heuristic only (see
    GroupGemmFp8W8A8Xe20.cpp/.cmake), so this op-level test pins
    activation_type=0 and does not exercise a bias-varying matrix (bias
    support is exercised by the fused_experts-level test above).
    """
    activation_type = 0  # silu
    gemm_k = hidden_size
    gemm_n = 2 * intermediate_size
    assert gemm_n % 2 == 0
    assert gemm_k % FP8_BLOCK_SIZE == 0, "gemm_k must be a multiple of FP8_GROUP_SIZE_K"

    inputs = _build_moe_gemm_inputs_fp8(
        num_experts=num_experts,
        avg_m_per_expert=num_tokens_per_expert,
        gemm_n=gemm_n,
        gemm_k=gemm_k,
        with_bias=False,
        fuse_act=fuse_act,
    )

    # Baseline: bf16 op on the fp8-rounded (dequantized) activations/weights.
    torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20(
        inputs["output_bf16"],
        inputs["a_dq"],
        inputs["w_dq"],
        inputs["bias"],
        inputs["total_rows"],
        num_experts,
        activation_type,
        fuse_act,
        1.702,
        7.0,
    )

    # Fused fp8 W8A8 path, fed the genuine fp8 tensors + scales.
    torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20_fp8_w8a8(
        inputs["output_fp8"],
        inputs["a_fp8"],
        inputs["a_scale"],
        inputs["w_fp8"],
        inputs["w_scale_per_row"],
        inputs["bias"],
        inputs["total_rows"],
        num_experts,
        activation_type,
        fuse_act,
        1.702,
        7.0,
    )

    torch.testing.assert_close(
        inputs["output_bf16"], inputs["output_fp8"], rtol=1e-1, atol=1e-2
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
