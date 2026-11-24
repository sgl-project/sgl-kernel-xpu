# SPDX-License-Identifier: Apache-2.0
"""Tests for mxfp8_quant_rows (Xe3P row-major MXFP8 activation quant).

The plain mode must be bit-identical to the generic per-token-group quant
(sgl_per_token_group_quant_8bit, group 32, UE8M0 row-major) it replaces on the
MoE path. The fused silu_and_mul mode is checked against a float64 SiLU
reference quantized with the same rules.
"""

import pytest
import torch

MX = 32


def _skip_if_unavailable():
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        pytest.skip("Intel XPU not available")
    import sgl_kernel  # noqa: F401
    from sgl_kernel import is_xe3_arch

    if not is_xe3_arch() or not hasattr(torch.ops.sgl_kernel, "mxfp8_quant_rows"):
        pytest.skip("mxfp8_quant_rows requires an Xe3P (CRI) build")


def _quant_rows(x, fuse):
    m, kin = x.shape
    k = kin // 2 if fuse else kin
    q = torch.empty((m, k), device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty((m, k // MX), device=x.device, dtype=torch.uint8)
    torch.ops.sgl_kernel.mxfp8_quant_rows(x, q, s, fuse)
    return q, s


def _ref_quant(v):
    """Host MXFP8 quant with the kernel's rules, v float32 (M, K)."""
    m, k = v.shape
    blocks = v.reshape(m, k // MX, MX)
    amax = blocks.abs().amax(-1).clamp_min(1e-10)
    e = torch.ceil(torch.log2((amax.double() / 448.0).clamp_min(1e-10))).float()
    q = (blocks / torch.exp2(e).unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.reshape(m, k), (e + 127).to(torch.uint8)


@pytest.mark.parametrize("m,k", [(1, 32), (7, 64), (333, 2048), (64, 2880), (513, 768)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_plain_matches_generic_quant(m, k, dtype):
    _skip_if_unavailable()
    from sgl_kernel.gemm import sgl_per_token_group_quant_8bit

    torch.manual_seed(m * 7 + k)
    x = torch.randn(m, k, device="xpu") * torch.exp2(
        torch.randint(-8, 8, (m, 1), device="xpu").float()
    )
    x = x.to(dtype)
    x[0, :MX] = 0  # all-zero group -> eps floor
    q, s = _quant_rows(x, False)

    q0 = torch.empty_like(q)
    s0 = torch.empty_like(s)
    sgl_per_token_group_quant_8bit(
        x, q0, s0, MX, 1e-10, -448.0, 448.0, True, False, None, False
    )
    assert torch.equal(s, s0)
    assert torch.equal(q.view(torch.uint8), q0.view(torch.uint8))


@pytest.mark.parametrize("m,k", [(1, 32), (65, 768), (300, 1536)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fused_silu_and_mul(m, k, dtype):
    _skip_if_unavailable()
    torch.manual_seed(m + k)
    x = (torch.randn(m, 2 * k, device="xpu") * 2).to(dtype)
    q, s = _quant_rows(x, True)

    xb = x.to(torch.bfloat16).double().cpu()
    g, u = xb[:, :k], xb[:, k:]
    act = (g / (1 + torch.exp(-g)) * u).to(torch.bfloat16).float()
    q_ref, s_ref = _ref_quant(act)
    # native exp/recip may move the bf16 activation by one ulp, which can flip
    # an E4M3 rounding or (rarely) a block exponent.
    assert (s.cpu() != s_ref).float().mean().item() < 0.01
    deq = q.cpu().float().reshape(m, -1, MX) * torch.exp2(s.cpu().float() - 127).unsqueeze(-1)
    deq_ref = q_ref.float().reshape(m, -1, MX) * torch.exp2(s_ref.float() - 127).unsqueeze(-1)
    scale = act.abs().amax().item()
    assert (deq - deq_ref).abs().max().item() <= 0.13 * scale


def test_rejects_bad_shapes():
    _skip_if_unavailable()
    x = torch.randn(4, 48, device="xpu", dtype=torch.bfloat16)
    q = torch.empty((4, 48), device="xpu", dtype=torch.float8_e4m3fn)
    s = torch.empty((4, 1), device="xpu", dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="multiple of 32"):
        torch.ops.sgl_kernel.mxfp8_quant_rows(x, q, s, False)


@pytest.mark.parametrize("m,k,topk", [(1, 2048, 8), (37, 768, 4), (300, 2880, 8)])
def test_scatter_matches_scatter_then_quant(m, k, topk):
    _skip_if_unavailable()
    from sgl_kernel.moe import scatter_tokens_to_experts

    torch.manual_seed(m + k + topk)
    a = torch.randn(m, k, device="xpu", dtype=torch.bfloat16)
    src2dst = torch.randperm(m * topk, device="xpu").to(torch.int32)
    q = torch.empty((m * topk, k), device="xpu", dtype=torch.float8_e4m3fn)
    s = torch.empty((m * topk, k // MX), device="xpu", dtype=torch.uint8)
    torch.ops.sgl_kernel.mxfp8_quant_rows(a, q, s, False, src2dst)

    rep = torch.empty((m * topk, k), device="xpu", dtype=torch.bfloat16)
    scatter_tokens_to_experts(a, src2dst, rep)
    q0, s0 = _quant_rows(rep, False)
    assert torch.equal(s, s0)
    assert torch.equal(q.view(torch.uint8), q0.view(torch.uint8))
