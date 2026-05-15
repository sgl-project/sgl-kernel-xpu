"""DSV4 fused_experts shape regression test.

Exercises fused_experts at the exact shape a real DSV4 decode step
produced (47 tokens, E=256, topk=6, H=4096, I=256), with N-outer
[E, N, K/32] scales matching the sglang checkpoint convention.

Golden reference is sglang's Triton fused_experts_impl on the MXFP4-xpu
branch — the upstream path the model is already known to run correctly
with.

Not part of the `per-commit` CI suite: the shapes are large enough
(~800 MB resident weights) that they would slow the gate meaningfully.
Run manually:

    pytest tests/test_fused_experts_mxfp4_dsv4_shapes.py -v

Or force-include via SGL_RUN_DSV4_SHAPE_TEST=1 when running the suite.
"""

import importlib.machinery
import os
import sys
import types
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
    # sglang's package chain pulls flashinfer.comm (CUDA) and a DSV4 env
    # default that asserts swiglu_limit. Stub both before importing so
    # the reference loads on an XPU-only host. Same preamble as in
    # benchmark/bench_fused_experts_mxfp4.py.
    os.environ.setdefault("SGLANG_DSV4_2604_SUBMODE", "2604A")

    if "flashinfer.comm" not in sys.modules:
        fi = sys.modules.get("flashinfer") or types.ModuleType("flashinfer")
        fi.__spec__ = importlib.machinery.ModuleSpec(
            "flashinfer", loader=None, is_package=True
        )
        if not hasattr(fi, "__path__"):
            fi.__path__ = []
        fi_comm = types.ModuleType("flashinfer.comm")
        fi_comm.__spec__ = importlib.machinery.ModuleSpec(
            "flashinfer.comm", loader=None, is_package=False
        )

        class _Stub:
            def __init__(self, *a, **k):
                pass

            def __getattr__(self, n):
                return _Stub()

            def __call__(self, *a, **k):
                return _Stub()

        fi_comm.MoeAlltoAll = _Stub
        fi_comm.moe_a2a_get_workspace_size_per_rank = _Stub
        sys.modules["flashinfer"] = fi
        sys.modules["flashinfer.comm"] = fi_comm

    from sglang.srt import server_args as _sa
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
        fused_experts_impl as _impl,
    )

    class _StubServerArgs:
        enable_deterministic_inference = False

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


def test_fused_experts_dsv4_shape():
    """fused_experts at the DSV4 decode shape, compared against the
    sglang Triton reference.

    Reproduces the exact call sglang's fp8 apply path was making when
    the scale-layout mismatch first surfaced. Inputs replicate the
    observed shapes and dtypes:

      x.shape                 = [47, 4096]       bfloat16
      w13_weight.shape        = [256, 512, 2048] int8   (E, 2*I, H/2)
      w2_weight.shape         = [256, 4096, 128] int8   (E, H, I/2)
      w13_weight_scale_inv    = [256, 512, 128]  float32
      w2_weight_scale_inv     = [256, 4096, 8]   float32
      topk_weights.shape      = [47, 6]          float32
      topk_ids.shape          = [47, 6]          int64
      activation='silu', routed_scaling_factor=1.5,
      gemm1_alpha=None, gemm1_clamp_limit=None
    """
    num_tokens, num_experts, topk, hidden, intermediate = 47, 256, 6, 4096, 256
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
    )

    assert out_sgl.shape == (num_tokens, hidden)
    assert out_sgl.dtype == torch.bfloat16
    assert torch.isfinite(out_sgl).all(), "sgl_kernel output has non-finite values"

    # Golden reference: sglang Triton fused_experts_impl on the MXFP4-xpu
    # branch. use_fp8_w8a8=True triggers _is_mxfp4_xpu_packed which routes
    # the int8-packed weights through the MXFP4 dequant-then-fused-moe path.
    # Triton fused_moe kernel expects int32 topk_ids.
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
