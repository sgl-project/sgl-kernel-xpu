# SPDX-License-Identifier: Apache-2.0
"""Tests for the software-emulated NVFP4 MoE path on Intel XPU (Xe3P / CRI).

NVFP4 has no native MMA on Xe3P: ``bdpas`` takes UE8M0 scale factors with a
fixed block size of 32, while NVFP4 uses E4M3 scale factors with block size 16
plus a per-tensor global scale.  The emulation is weight-only-quantized:
``nvfp4_woq_mxfp8_scaled_grouped_mm`` dequantizes the FP4 weights to BF16 and
requantizes them to MXFP8 inside the mainloop, so the weights stay 4-bit in
memory, and the MMA runs as an ordinary MXFP8 block-scaled GEMM.

The oracle is ``nvfp4_to_mxfp8_ref`` (torch, in nvfp4_utils.py): requantizing on
the host and feeding the result to the existing MXFP8 grouped GEMM must match the
fused kernel *bit-for-bit*, because both then multiply the same E4M3 values by
the same UE8M0 scales.  Any difference is a layout or race bug, not rounding.
"""

import os

import pytest
import torch
from nvfp4_utils import (
    NVFP4_BLOCK_SIZE,
    dequantize_mxfp8,
    dequantize_nvfp4,
    dequantize_nvfp4_via_mxfp8,
    e2m1_lut,
    nvfp4_to_mxfp8_ref,
    nvfp4_woq_prep_ref,
    quantize_mxfp8,
    quantize_nvfp4,
)

# The 512-N / 1024-K shapes take tens of minutes on the CRI simulator.
LONG_TESTS = os.getenv("LONG_TESTS") == "1"

WORKSPACE_BYTES = 64 * 1024 * 1024


def is_xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def is_cri_device() -> bool:
    if not is_xpu_available():
        return False
    try:
        from sgl_kernel import is_xe3_arch

        return is_xe3_arch()
    except ImportError:
        return False


def skip_if_no_cri():
    if not is_xpu_available():
        pytest.skip("Intel XPU not available")
    if not is_cri_device():
        pytest.skip("NVFP4 WoQ kernels require an Xe3P (CRI) XPU")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_nvfp4_weights(e: int, n: int, k: int, device, seed: int = 0, scale: float = 0.05):
    """Quantize a random (E, N, K) weight tensor to NVFP4.

    Returns ``(packed, sf, gs_dq, w_fp32)`` where ``gs_dq`` is the per-expert
    dequant multiplier (reciprocal of the usual global scale).
    """
    torch.manual_seed(seed)
    w = torch.randn(e, n, k) * scale
    qs, sfs, gss = [], [], []
    for i in range(e):
        q, sf, gs = quantize_nvfp4(w[i])
        qs.append(q)
        sfs.append(sf)
        gss.append(1.0 / gs)
    return (
        torch.stack(qs).to(device),
        torch.stack(sfs).to(device),
        torch.stack(gss).float().to(device),
        w,
    )


def make_grouped_layout(ms, n, k, device):
    problem_sizes = torch.tensor([[m, n, k] for m in ms], dtype=torch.int32, device=device)
    starts, acc = [], 0
    for m in ms:
        starts.append(acc)
        acc += m
    expert_offsets = torch.tensor(starts, dtype=torch.int32, device=device)
    return problem_sizes, expert_offsets


def run_mxfp8_reference(a_q, a_s, b_mx, b_smx, problem_sizes, expert_offsets, workspace):
    """Plain MXFP8 grouped GEMM over host-requantized weights — the oracle."""
    from sgl_kernel import fp8_blockwise_scaled_grouped_mm

    device = a_q.device
    e = problem_sizes.size(0)
    empty = torch.empty((0,), dtype=torch.int64, device=device)
    zeros_stride = torch.zeros((e,), dtype=torch.int64, device=device)
    zeros_layout = torch.zeros((e, 5), dtype=torch.int32, device=device)
    out = torch.zeros((a_q.size(0), b_mx.size(1)), dtype=torch.float32, device=device)
    fp8_blockwise_scaled_grouped_mm(
        out, empty, empty, empty, empty, empty,
        a_q, b_mx, a_s, b_smx,
        zeros_stride, zeros_stride, zeros_stride, zeros_layout, zeros_layout,
        problem_sizes, expert_offsets, workspace,
    )
    return out


# ---------------------------------------------------------------------------
# Reference numerics (CPU only)
# ---------------------------------------------------------------------------


def test_nvfp4_roundtrip_on_grid():
    """Values already on the E2M1 grid with an exactly representable scale
    survive a quantize/dequantize round trip."""
    lut = e2m1_lut()
    torch.manual_seed(0)
    x = lut[torch.randint(0, 16, (8, 64))] * 0.25
    q, sf, gs = quantize_nvfp4(x)
    # amax = 1.5 -> gs = 2688/1.5 = 1792, sf = 448 (exact in E4M3) for every
    # block that contains a +-6 entry.
    assert torch.equal(sf.float(), torch.full_like(sf.float(), 448.0))
    assert torch.equal(dequantize_nvfp4(q, sf, gs), x)


@pytest.mark.parametrize("adversarial", [False, True])
def test_requantization_error_is_small(adversarial):
    """Emulating NVFP4 through MXFP8 must not meaningfully degrade the weights.

    The extra error comes from one E4M3 rounding per element; NVFP4's own error
    (1 mantissa bit per value) dominates it.
    """
    torch.manual_seed(0)
    n, k = 64, 512
    w = torch.randn(n, k)
    if adversarial:
        # Neighbouring 16-blocks differing by 2**12 stress the shared per-32
        # UE8M0 exponent.
        blocks = w.reshape(n, k // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE)
        blocks[:, 1::2, :] *= 2.0**-12
        w = blocks.reshape(n, k)
    else:
        w = w * 0.05

    q, sf, gs = quantize_nvfp4(w)
    gs_dq = (1.0 / gs).reshape(1)
    w_nvfp4 = dequantize_nvfp4(q, sf, gs)
    w_emul = dequantize_nvfp4_via_mxfp8(q, sf, gs_dq)

    def rel(x, y):
        return ((x - y).norm() / y.norm().clamp_min(1e-30)).item()

    err_nvfp4 = rel(w_nvfp4, w)
    err_extra = rel(w_emul, w_nvfp4)
    assert err_extra < 0.05, f"requantization error too large: {err_extra}"
    assert rel(w_emul, w) < err_nvfp4 * 1.15, "emulation degrades NVFP4 by more than 15%"

    # Requantization must never clip: c * 6 stays inside E4M3's range.
    c, _ = nvfp4_woq_prep_ref(sf, gs_dq, transpose=False)
    assert (c.float() * 6.0).max().item() <= 448.0


# ---------------------------------------------------------------------------
# Device kernels
# ---------------------------------------------------------------------------


def test_prep_kernel_matches_reference():
    """The device scale-prep kernel is bit-exact against the Python reference."""
    skip_if_no_cri()
    from sgl_kernel import nvfp4_woq_prep

    e, n, k = 3, 64, 256
    dev = "xpu"
    _, sf, gs_dq, _ = make_nvfp4_weights(e, n, k, dev, seed=1)

    c_dev, sfmx_dev = nvfp4_woq_prep(sf, gs_dq)
    c_ref, sfmx_ref = nvfp4_woq_prep_ref(sf.cpu(), gs_dq.cpu(), transpose=True)
    assert torch.equal(sfmx_dev.cpu(), sfmx_ref)
    # Compare BF16 bit patterns so a rounding difference cannot hide.
    assert torch.equal(c_dev.cpu().view(torch.int16), c_ref.view(torch.int16))


def test_prep_kernel_rejects_noncontiguous_global_scales():
    """The prep kernel reads global scales as a contiguous pointer."""
    skip_if_no_cri()
    from sgl_kernel import nvfp4_woq_prep

    e, n, k = 3, 64, 256
    dev = "xpu"
    _, sf, gs_dq, _ = make_nvfp4_weights(e, n, k, dev, seed=1)
    strided_gs = torch.stack((gs_dq, gs_dq), dim=1)[:, 0]
    assert not strided_gs.is_contiguous()

    with pytest.raises(RuntimeError, match="w_gs must be contiguous"):
        nvfp4_woq_prep(sf, strided_gs)


@pytest.mark.parametrize(
    "ms,n,k",
    [
        ([64, 64], 128, 256),
        ([64, 64], 128, 64),      # single K tile
        ([32, 96], 128, 256),     # ragged M
        ([33, 95], 128, 256),     # M not 4-aligned (scale buffer is padded)
        ([64, 64], 192, 256),     # N not a multiple of the workgroup N tile
        pytest.param([256, 256], 512, 512, marks=pytest.mark.skipif(
            not LONG_TESTS, reason="set LONG_TESTS=1")),
        pytest.param([64, 64], 128, 1024, marks=pytest.mark.skipif(
            not LONG_TESTS, reason="set LONG_TESTS=1")),
    ],
)
def test_fused_woq_matches_host_requantized_mxfp8(ms, n, k):
    """The fused WoQ GEMM must reproduce host-side requantization exactly.

    Both paths multiply the same E4M3 values by the same UE8M0 scales, so any
    difference is a layout or indexing bug rather than rounding.
    """
    skip_if_no_cri()
    from sgl_kernel import nvfp4_woq_mxfp8_scaled_grouped_mm, nvfp4_woq_prep

    dev = "xpu"
    m_total = sum(ms)
    packed, sf, gs_dq, _ = make_nvfp4_weights(len(ms), n, k, dev, seed=2)

    torch.manual_seed(3)
    a_q, a_s = quantize_mxfp8(torch.randn(m_total, k) * 0.5)
    a_q, a_s = a_q.to(dev), a_s.to(dev)

    problem_sizes, expert_offsets = make_grouped_layout(ms, n, k, dev)
    workspace = torch.zeros((WORKSPACE_BYTES,), dtype=torch.uint8, device=dev)

    b_mx, b_smx = nvfp4_to_mxfp8_ref(packed.cpu(), sf.cpu(), gs_dq.cpu())
    ref = run_mxfp8_reference(
        a_q, a_s, b_mx.to(dev), b_smx.to(dev), problem_sizes, expert_offsets, workspace
    )

    c_b, sfmx_b = nvfp4_woq_prep(sf, gs_dq)
    out = torch.zeros((m_total, n), dtype=torch.float32, device=dev)
    nvfp4_woq_mxfp8_scaled_grouped_mm(
        out, a_q, packed, a_s, sfmx_b, c_b, problem_sizes, expert_offsets, workspace
    )
    torch.xpu.synchronize()
    assert torch.equal(out.cpu(), ref.cpu()), (
        f"fused WoQ != host-requantized MXFP8; max diff "
        f"{(out.cpu() - ref.cpu()).abs().max().item()}"
    )


def test_unaligned_scale_stride_is_rejected():
    """Shapes whose per-expert B-scale slice isn't 64-byte aligned must raise.

    The Xe block-2D scale load needs a 64-byte aligned base address, so
    N * (K/32) has to be a multiple of 64. Fail on the host with a readable
    message instead of letting the device-side assert fire per work-item.
    """
    skip_if_no_cri()
    from sgl_kernel import nvfp4_woq_mxfp8_scaled_grouped_mm, nvfp4_woq_prep

    dev = "xpu"
    ms, n, k = [64, 64], 132, 256  # 132 * (256/32) = 1056, not a multiple of 64
    m_total = sum(ms)
    packed, sf, gs_dq, _ = make_nvfp4_weights(len(ms), n, k, dev, seed=4)
    a_q, a_s = quantize_mxfp8(torch.randn(m_total, k))
    c_b, sfmx_b = nvfp4_woq_prep(sf, gs_dq)
    problem_sizes, expert_offsets = make_grouped_layout(ms, n, k, dev)
    workspace = torch.zeros((WORKSPACE_BYTES,), dtype=torch.uint8, device=dev)
    out = torch.zeros((m_total, n), dtype=torch.float32, device=dev)

    with pytest.raises(RuntimeError, match="multiple of 64"):
        nvfp4_woq_mxfp8_scaled_grouped_mm(
            out, a_q.to(dev), packed, a_s.to(dev), sfmx_b, c_b,
            problem_sizes, expert_offsets, workspace,
        )


# ---------------------------------------------------------------------------
# End-to-end MoE
# ---------------------------------------------------------------------------


def _moe_reference(a, w1_eff, w2_eff, topk_weights, topk_ids, n_inter):
    """Fused-MoE reference over already-dequantized (emulated) weights.

    Mirrors the kernel path: MXFP8-quantize the activations, SiLU-and-mul in the
    activation dtype, MXFP8-quantize the intermediate, accumulate in fp32.
    """
    m, _ = a.shape
    dt = a.dtype
    a_dq = dequantize_mxfp8(*quantize_mxfp8(a.float()))
    out = torch.zeros(m, w2_eff.shape[1], dtype=torch.float32)
    for t in range(m):
        for j in range(topk_ids.shape[1]):
            e = int(topk_ids[t, j])
            h = a_dq[t] @ w1_eff[e].T
            gate, up = h[:n_inter], h[n_inter:]
            inter = torch.nn.functional.silu(gate.to(dt).float()) * up.to(dt).float()
            inter_dq = dequantize_mxfp8(*quantize_mxfp8(inter.unsqueeze(0).to(dt).float()))[0]
            out[t] += float(topk_weights[t, j]) * (inter_dq @ w2_eff[e].T)
    return out


def test_fused_experts_nvfp4_woq():
    """End-to-end MoE with NVFP4 weights through the fused WoQ path."""
    skip_if_no_cri()
    from sgl_kernel import cutlass_fused_experts_nvfp4_woq, nvfp4_woq_prepare_weights

    dev = "xpu"
    dt = torch.bfloat16
    e, m, k, n_inter, topk = 4, 32, 256, 64, 2

    torch.manual_seed(7)
    a = (torch.randn(m, k) * 0.5).to(dt)
    w1_q, w1_sf, w1_gs, _ = make_nvfp4_weights(e, 2 * n_inter, k, dev, seed=11)
    w2_q, w2_sf, w2_gs, _ = make_nvfp4_weights(e, k, n_inter, dev, seed=12)
    topk_weights, topk_ids = torch.topk(torch.softmax(torch.randn(m, e), -1), topk, dim=-1)

    prepared = nvfp4_woq_prepare_weights(w1_sf, w1_gs, w2_sf, w2_gs)
    out = cutlass_fused_experts_nvfp4_woq(
        a.to(dev), w1_q, w2_q, prepared,
        topk_weights.to(dt).to(dev), topk_ids.int().to(dev),
    )
    torch.xpu.synchronize()

    w1_eff = torch.stack(
        [dequantize_nvfp4_via_mxfp8(w1_q[i].cpu(), w1_sf[i].cpu(), w1_gs[i].cpu()) for i in range(e)]
    )
    w2_eff = torch.stack(
        [dequantize_nvfp4_via_mxfp8(w2_q[i].cpu(), w2_sf[i].cpu(), w2_gs[i].cpu()) for i in range(e)]
    )
    ref = _moe_reference(a, w1_eff, w2_eff, topk_weights, topk_ids, n_inter)

    rel = ((out.float().cpu() - ref).norm() / ref.norm()).item()
    # Residual is BF16 output rounding plus accumulation order; the requantization
    # itself is covered bit-exactly by test_fused_woq_matches_host_requantized_mxfp8.
    assert rel < 0.02, f"relative error {rel} too large"
