"""DSV4 fused_experts shape regression test.

Exercises fused_experts at the exact shape a real DSV4 decode step
produced (47 tokens, E=256, topk=6, H=4096, I=256), with N-outer
[E, N, K/32] scales matching the sglang checkpoint convention.

The golden reference dequantizes and computes one active expert at a time,
keeping the large-shape peak memory bounded.

The shapes are large enough (~800 MB resident weights), so run this file
directly when a targeted DSV4 regression check is needed:

    pytest tests/test_fused_experts_mxfp4_dsv4_shapes.py -v
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mxfp4_utils import MXFP4_BLOCK_SIZE  # noqa: E402
from mxfp4_utils import dequantize_mxfp4_2d  # noqa: E402
from mxfp4_utils import quantize_mxfp4_2d  # noqa: E402
from sgl_kernel import fused_experts  # noqa: E402

def _torch_mxfp4_moe_reference(
    hidden_states,
    w1_packed,
    w1_scale,
    w2_packed,
    w2_scale,
    topk_weights,
    topk_ids,
    routed_scaling_factor,
    swiglu_limit,
):
    """Compute the reference while materializing only one expert at a time."""
    num_tokens, hidden = hidden_states.shape
    topk = topk_ids.shape[1]
    routed_inputs = (
        hidden_states[:, None, :].expand(-1, topk, -1).reshape(-1, hidden)
    )
    flat_ids = topk_ids.reshape(-1)
    routed_outputs = torch.zeros_like(routed_inputs)

    for expert in flat_ids.unique().cpu().tolist():
        mask = flat_ids == expert
        expert_w1 = dequantize_mxfp4_2d(
            w1_packed[expert], w1_scale[expert], dtype=hidden_states.dtype
        )
        gate_up = routed_inputs[mask] @ expert_w1.transpose(0, 1)
        del expert_w1

        gate, up = gate_up.chunk(2, dim=-1)
        if swiglu_limit is not None:
            gate = gate.to(torch.bfloat16).clamp(max=swiglu_limit)
            up = up.to(torch.bfloat16).clamp(-swiglu_limit, swiglu_limit)
            intermediate = (F.silu(gate.float()) * up.float()).to(
                hidden_states.dtype
            )
        else:
            intermediate = F.silu(gate) * up

        expert_w2 = dequantize_mxfp4_2d(
            w2_packed[expert], w2_scale[expert], dtype=hidden_states.dtype
        )
        routed_outputs[mask] = intermediate @ expert_w2.transpose(0, 1)
        del expert_w2

    return (
        routed_outputs.view(num_tokens, topk, hidden)
        * topk_weights[..., None].to(routed_outputs.dtype)
    ).sum(dim=1) * routed_scaling_factor


def _build_packed_weights(E, N, K):
    # Quantize one expert at a time to avoid materializing a full
    # (E, N, K) bf16 tensor plus its fp32 copy on the host.
    packed = torch.empty((E, N, K // 2), dtype=torch.uint8)
    scales = torch.empty((E, N, K // MXFP4_BLOCK_SIZE), dtype=torch.uint8)
    for e in range(E):
        w_e = torch.empty((N, K), dtype=torch.bfloat16).normal_(0, 0.01)
        p_e, s_e = quantize_mxfp4_2d(w_e.float(), MXFP4_BLOCK_SIZE)
        packed[e].copy_(p_e)
        scales[e].copy_(s_e)
        del w_e, p_e, s_e
    return packed, scales


# Cover both the real DSV4 decode shape and a larger-intermediate shape. Both
# use the W4A16 two-GEMM path and the standalone fused activation kernel.
# Each case keeps packed resident weights around 400 MB.
_DSV4_SHAPES = [
    # (name, num_tokens, num_experts, topk, hidden, intermediate)
    ("decode", 47, 256, 6, 4096, 256),
    ("large-intermediate", 47, 8, 6, 4096, 8192),
]


# swiglu_limit selects whether the DeepSeek-V4 clamp runs at all:
#   - 10:   activation_type=4 (swiglu_deepseek_v4) — clamp gate/up then silu*up
#   - None: activation_type=0 (plain silu_and_mul, no clamp)
# Both values must match the independent reference, so cross both with the two
# activation paths (4 cases total).
_SWIGLU_LIMITS = [10, None]


@pytest.mark.parametrize(
    "swiglu_limit",
    _SWIGLU_LIMITS,
    ids=[f"limit{lim}" for lim in _SWIGLU_LIMITS],
)
@pytest.mark.parametrize(
    "name,num_tokens,num_experts,topk,hidden,intermediate",
    _DSV4_SHAPES,
    ids=[s[0] for s in _DSV4_SHAPES],
)
def test_fused_experts_dsv4_shape(
    name, num_tokens, num_experts, topk, hidden, intermediate, swiglu_limit
):
    """fused_experts with/without the DeepSeek-V4 swiglu clamp, compared
    against an independently dequantized PyTorch reference.

    Runs at two shapes that take the two distinct activation paths in
    fused_experts. The "fused" shape reproduces the exact call sglang's fp8
    apply path was making when the scale-layout mismatch first surfaced:

      x.shape                 = [47, 4096]       bfloat16
      w13_weight.shape        = [256, 512, 2048] int8   (E, 2*I, H/2)
      w2_weight.shape         = [256, 4096, 128] int8   (E, H, I/2)
      w13_weight_scale        = [256, 512, 128]  uint8 E8M0
      w2_weight_scale         = [256, 4096, 8]   uint8 E8M0
      topk_weights.shape      = [47, 6]          float32
      topk_ids.shape          = [47, 6]          int64
      activation='silu', routed_scaling_factor=1.5, swiglu_limit=10

    The "large-intermediate" shape (E=8, H=4096, I=8192) exercises the same
    activation path with a substantially larger intermediate projection.

    swiglu_limit is crossed over {10, None}: 10 exercises the DSV4 clamp
    (activation_type=4), None falls back to plain silu (activation_type=0).
    """
    torch.manual_seed(0)
    torch.xpu.manual_seed_all(0)

    a = torch.empty(num_tokens, hidden, dtype=torch.bfloat16, device="xpu").normal_(
        0, 0.01
    )

    w1_packed, w1_scale_u8 = _build_packed_weights(
        num_experts, 2 * intermediate, hidden
    )
    w2_packed, w2_scale_u8 = _build_packed_weights(num_experts, hidden, intermediate)

    # Keep raw E8M0 scales for both the W4A16 kernel and reference decoder.
    w1_scale = w1_scale_u8.to("xpu")
    w2_scale = w2_scale_u8.to("xpu")
    w1_packed_xpu = w1_packed.view(torch.int8).to("xpu")
    w2_packed_xpu = w2_packed.view(torch.int8).to("xpu")

    score = torch.randn(num_tokens, num_experts, dtype=torch.float32, device="xpu")
    score = torch.softmax(score, dim=-1)
    topk_weights_f32, topk_ids_i64 = torch.topk(score, topk)
    topk_ids_i64 = topk_ids_i64.to(torch.int64)

    routed_scaling_factor = 1.5

    # sgl_kernel path.
    out_sgl = fused_experts(
        a,
        w1_packed_xpu,
        w2_packed_xpu,
        topk_weights_f32,
        topk_ids_i64,
        None,
        None,
        activation="silu",
        use_mxfp4_w4a16=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        routed_scaling_factor=routed_scaling_factor,
        swiglu_limit=swiglu_limit,
    )

    assert out_sgl.shape == (num_tokens, hidden)
    assert out_sgl.dtype == torch.bfloat16
    assert torch.isfinite(out_sgl).all(), "sgl_kernel output has non-finite values"

    out_reference = _torch_mxfp4_moe_reference(
        a,
        w1_packed_xpu.view(torch.uint8),
        w1_scale,
        w2_packed_xpu.view(torch.uint8),
        w2_scale,
        topk_weights_f32,
        topk_ids_i64,
        routed_scaling_factor,
        swiglu_limit,
    )

    assert out_reference.shape == out_sgl.shape
    assert out_reference.dtype == out_sgl.dtype

    # Both paths compute MXFP4-rounded weights times bf16 activations, so any
    # numerical delta comes from GEMM arithmetic ordering. Use
    # the same tolerances as the existing test_moe_gemm_mxfp4_weights
    # suite (rtol=1e-1, atol=1e-2) — MXFP4's 4-bit mantissa plus bf16
    # activations means the absolute error can be up to a few percent of
    # the dynamic range on individual outputs.
    torch.testing.assert_close(
        out_sgl.float(), out_reference.float(), rtol=1e-1, atol=1e-2
    )
