"""Minimal-shape MOE smoke tests for Xe3 (Xe35/CRI).

Intended to run on a (very slow) Xe3 hardware simulator, where the full
parametrize matrices in test_moe_gemm.py etc. are infeasible. This file only
exercises the two Xe3-specific grouped-GEMM dispatch paths
(``moe_grouped_mm_nt_xe35`` and ``moe_grouped_mm_nt_xe35_mxfp4_w4a16``) with
the smallest shapes known to satisfy the kernels' own tiling constraints.

Coverage of the remaining, arch-independent MOE ops (``moe_fused_gate``,
``moe_align_block_size``, ``moe_sum``, ``prepare_moe_input[_small]``,
``scatter_tokens_to_experts``, ``apply_shuffle_mul_sum``, ``topk_softmax``,
``topk_sigmoid``, ``hash_topk``, ``biased_topk``) is intentionally NOT
duplicated here: their existing test files already contain small-shape
parametrize cases suitable for a slow simulator. See the curated list of
minimal pytest node IDs in
``~/frameworks.ai.pytorch.sgl-kernel-xpu/MOE_XE3_SIMULATOR_TEST_PLAN.md``.

Explicitly out of scope (kernels not compiled in for Xe3, already handled by
``is_op_kernel_built``-gated skips in their own test files):
  - mxfp4_blockwise_scaled_grouped_mm  (test_mxfp4_blockwise_moe.py)
  - fp8_blockwise_scaled_grouped_mm    (test_mxfp8_blockwise_moe.py / test_cutlass_moe.py)

Also out of scope: ``moe_grouped_mm_nt_xe20_fp8_w8a16``. Despite its
unconditional torch-op registration and lack of arch-gating in its CMake
inclusion, its .cpp source hardcodes ``#define SYCL_INTEL_TARGET 20`` and the
Python-level ``fused_experts`` still blocks ``use_fp8_w8a8`` on Xe3
(see moe.py). It has not been validated/ported for Xe3 and is treated as
Xe2-only pending a real port.
"""

import pytest
import torch
from sgl_kernel import fused_experts
from sgl_kernel.utils import is_xe3_arch
from test_moe_gemm import _mxfp4_expert_weights, torch_naive_moe


def _is_xpu_available():
    return hasattr(torch, "xpu") and torch.xpu.is_available()


pytestmark = pytest.mark.skipif(
    not (_is_xpu_available() and is_xe3_arch()),
    reason="Xe3 (CRI) smoke tests require running on Xe3/CRI XPU hardware (or its simulator)",
)


def test_xe35_bf16_grouped_gemm_smoke():
    """Smallest shape exercising moe_grouped_mm_nt_xe35 via fused_experts."""
    torch.manual_seed(0)
    torch.xpu.manual_seed_all(0)

    num_tokens, topk, num_experts, hidden_size, intermediate_size = 2, 1, 2, 64, 32

    a = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16)
    w1 = (
        torch.randn(
            (num_experts, 2 * intermediate_size, hidden_size), dtype=torch.bfloat16
        )
        * 0.01
    )
    w2 = (
        torch.randn((num_experts, hidden_size, intermediate_size), dtype=torch.bfloat16)
        * 0.01
    )

    score = torch.softmax(
        torch.randn([num_tokens, num_experts], dtype=torch.bfloat16),
        dim=-1,
        dtype=torch.float32,
    )
    topk_weight, topk_ids = torch.topk(score, topk)

    torch_output = torch_naive_moe(
        a, w1, w2, topk_ids, topk_weight, topk, None, None, activations="silu"
    )

    device = "xpu"
    kernel_output = fused_experts(
        a.to(device),
        w1.to(device),
        w2.to(device),
        topk_weight.to(device),
        topk_ids.to(device),
        None,
        None,
        activation="silu",
    )

    torch.testing.assert_close(
        torch_output, kernel_output.to("cpu"), rtol=1e-1, atol=1e-2
    )


def test_xe35_mxfp4_w4a16_grouped_gemm_smoke():
    """Smallest shape exercising moe_grouped_mm_nt_xe35_mxfp4_w4a16 via
    fused_experts' Xe3 mxfp4 dispatch path (silu, no bias, group_size=32).

    hidden_size/intermediate_size = 128 is the smallest value used by the
    existing (Xe2/Xe3-agnostic) test_moe_gemm_mxfp4_weights parametrize
    matrix and is known to satisfy the kernel's tiling constraints; going
    lower is not validated.
    """
    torch.manual_seed(0)
    torch.xpu.manual_seed_all(0)

    num_tokens, topk, num_experts, hidden_size, intermediate_size = 1, 1, 8, 128, 128

    a = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16)
    score = torch.softmax(
        torch.randn([num_tokens, num_experts], dtype=torch.bfloat16),
        dim=-1,
        dtype=torch.float32,
    )
    topk_weight, topk_ids = torch.topk(score, topk)

    w1_packed, w1_scale, w1_dq = _mxfp4_expert_weights(
        num_experts, 2 * intermediate_size, hidden_size
    )
    w2_packed, w2_scale, w2_dq = _mxfp4_expert_weights(
        num_experts, hidden_size, intermediate_size
    )

    torch_output = torch_naive_moe(
        a, w1_dq, w2_dq, topk_ids, topk_weight, topk, None, None, activations="silu"
    )

    device = "xpu"
    kernel_output = fused_experts(
        a.to(device),
        w1_packed.to(device),
        w2_packed.to(device),
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
        torch_output, kernel_output.to("cpu"), rtol=1e-1, atol=1e-2
    )
