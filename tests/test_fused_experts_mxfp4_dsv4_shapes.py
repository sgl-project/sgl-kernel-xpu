"""DSV4 fused_experts shape regression test.

Exercises fused_experts at the exact shape a real DSV4 decode step
produced (47 tokens, E=256, topk=6, H=4096, I=256), with N-outer
[E, N, K/32] scales matching the sglang checkpoint convention.

Golden reference is sglang's Triton fused_experts_impl on the DeepSeek-V4
XPU support branch — the upstream path the model is already known to run
correctly with. Requires sglang to be installed in the environment (see the
project README for the XPU install recipe).

Not part of the `per-commit` CI suite: the shapes are large enough
(~800 MB resident weights) that they would slow the gate meaningfully.
Run manually:

    pytest tests/test_fused_experts_mxfp4_dsv4_shapes.py -v

Or force-include via SGL_RUN_DSV4_SHAPE_TEST=1 when running the suite.
"""

import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mxfp4_utils import MXFP4_BLOCK_SIZE  # noqa: E402
from mxfp4_utils import quantize_mxfp4_2d  # noqa: E402
from sgl_kernel import fused_experts  # noqa: E402

# Skip on CI unless explicitly requested. tests/run_suite.py does not
# list this file, so `per-commit` won't pick it up either way — this is
# a second safety net when the file is discovered by bare `pytest tests/`.
pytestmark = pytest.mark.skipif(
    os.environ.get("SGL_RUN_DSV4_SHAPE_TEST", "0") != "1",
    reason="DSV4 full-shape regression — set SGL_RUN_DSV4_SHAPE_TEST=1 to run",
)


def _import_triton_fused_experts_impl():
    """Import sglang's Triton fused_experts_impl as the golden reference.

    Requires sglang (upstream main) to be installed in the environment — see
    the project README for the XPU install recipe. The DeepSeek-V4 swiglu_limit
    clamp lives in moe_runner/triton_utils/fused_moe.py; the older
    fused_moe_triton tree lacks it, so this import path is required.

    fused_experts_impl reads get_global_server_args() at call time but the
    reference never runs under a real server, so install a minimal stub holding
    only the flags this code path reads.
    """
    from sglang.srt import server_args as _sa
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        fused_experts_impl as _impl,
    )

    class _StubServerArgs:
        enable_deterministic_inference = False
        enable_fused_moe_sum_all_reduce = False

    _sa._global_server_args = _StubServerArgs()
    return _impl


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
# Both values must match the Triton reference, so cross both with the two
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
    against the sglang Triton reference.

    Runs at two shapes that take the two distinct activation paths in
    fused_experts. The "fused" shape reproduces the exact call sglang's fp8
    apply path was making when the scale-layout mismatch first surfaced:

      x.shape                 = [47, 4096]       bfloat16
      w13_weight.shape        = [256, 512, 2048] int8   (E, 2*I, H/2)
      w2_weight.shape         = [256, 4096, 128] int8   (E, H, I/2)
      w13_weight_scale_inv    = [256, 512, 128]  float32
      w2_weight_scale_inv     = [256, 4096, 8]   float32
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

    # UE8M0 → fp32 direct multiplier. N-outer [E, N, K/32] layout.
    w1_scale = torch.exp2((w1_scale_u8.to(torch.int32) - 127).to(torch.float32)).to(
        "xpu"
    )
    w2_scale = torch.exp2((w2_scale_u8.to(torch.int32) - 127).to(torch.float32)).to(
        "xpu"
    )
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

    # Golden reference: sglang Triton fused_experts_impl on the DeepSeek-V4
    # XPU support branch. use_fp8_w8a8=True is what the DSv4 fp8 loader passes
    # for MXFP4 routed experts; _is_mxfp4_xpu_packed then detects the packed
    # weights (uint8/int8 + scales, last dim == hidden/2) and routes them
    # through the Triton MXFP4 upcast-then-fused-moe path. Triton's fused_moe
    # kernel expects int32 topk_ids.
    triton_fused_experts_impl = _import_triton_fused_experts_impl()
    out_triton = triton_fused_experts_impl(
        a,
        w1_packed_xpu.view(torch.uint8),
        w2_packed_xpu.view(torch.uint8),
        topk_weights_f32,
        topk_ids_i64.to(torch.int32),
        None,
        None,
        inplace=False,
        activation="silu",
        is_gated=True,
        apply_router_weight_on_input=False,
        use_fp8_w8a8=True,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        routed_scaling_factor=routed_scaling_factor,
        swiglu_limit=swiglu_limit,
    )

    assert out_triton.shape == out_sgl.shape
    assert out_triton.dtype == out_sgl.dtype

    # Both paths compute MXFP4-rounded weights × bf16 activations, so any
    # numerical delta comes from GEMM arithmetic ordering (bf16 accumulate
    # within the tile, fp32 across tiles differs between the two). Use
    # the same tolerances as the existing test_moe_gemm_mxfp4_weights
    # suite (rtol=1e-1, atol=1e-2) — MXFP4's 4-bit mantissa plus bf16
    # activations means the absolute error can be up to a few percent of
    # the dynamic range on individual outputs.
    torch.testing.assert_close(
        out_sgl.float(), out_triton.float(), rtol=1e-1, atol=1e-2
    )
