"""DSV4 fused_experts shape regression test.

Exercises fused_experts at the exact shape a real DSV4 decode step
produced (47 tokens, E=256, topk=6, H=4096, I=256), with N-outer
[E, N, K/32] scales matching the sglang checkpoint convention.

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
    """fused_experts at the DSV4 decode shape with N-outer scales.

    Shape taken from a real DSV4 decode step:
      x.shape       = [47, 4096]
      w13_weight    = [256, 512, 2048]    (E, 2*I, H/2)  -> I=256, H=4096
      w2_weight     = [256, 4096, 128]    (E, H, I/2)
      topk_ids      = [47, 6]             -> topk=6
    """
    num_tokens, num_experts, topk, hidden, intermediate = 47, 256, 6, 4096, 256
    torch.manual_seed(0)
    torch.xpu.manual_seed_all(0)

    a = torch.empty(
        num_tokens, hidden, dtype=torch.bfloat16, device="xpu"
    ).normal_(0, 0.01)

    w1_packed, w1_scale_u8 = _build_packed_weights(
        num_experts, 2 * intermediate, hidden
    )
    w2_packed, w2_scale_u8 = _build_packed_weights(num_experts, hidden, intermediate)

    # UE8M0 → fp32 direct multiplier. N-outer [E, N, K/32] layout.
    w1_scale = torch.exp2((w1_scale_u8.to(torch.int32) - 127).to(torch.float32))
    w2_scale = torch.exp2((w2_scale_u8.to(torch.int32) - 127).to(torch.float32))

    score = torch.randn(
        num_tokens, num_experts, dtype=torch.bfloat16, device="xpu"
    )
    score = torch.softmax(score.float(), dim=-1)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.to(torch.bfloat16)

    output = fused_experts(
        a,
        w1_packed.view(torch.int8).to("xpu"),
        w2_packed.view(torch.int8).to("xpu"),
        topk_weight,
        topk_ids,
        None,
        None,
        activation="silu",
        use_mxfp4_w4a16=True,
        use_fused_mxfp4_kernel=True,
        w1_scale=w1_scale.to("xpu"),
        w2_scale=w2_scale.to("xpu"),
    )

    assert output.shape == (num_tokens, hidden)
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all(), "fused_experts output has non-finite values"
