# SPDX-License-Identifier: Apache-2.0
"""Tests for the software-emulated NVFP4 MoE path on Intel XPU (Xe3P / CRI).

NVFP4 has no native MMA on Xe3P: ``bdpas`` takes UE8M0 scale factors with a
fixed block size of 32, while NVFP4 uses E4M3 scale factors with block size 16
plus a per-tensor global scale.  The emulation is weight-only-quantized:
``nvfp4_woq_mxfp8_scaled_grouped_mm_xe35`` dequantizes the FP4 weights to BF16 and
requantizes them to MXFP8 inside the mainloop, so the weights stay 4-bit in
memory, and the MMA runs as an ordinary MXFP8 block-scaled GEMM.

The oracle is ``nvfp4_to_mxfp8_ref_xe35`` (torch, in nvfp4_utils.py): requantizing on
the host and feeding the result to the existing MXFP8 grouped GEMM must match the
fused kernel *bit-for-bit*, because both then multiply the same E4M3 values by
the same UE8M0 scales.  Any difference is a layout or race bug, not rounding.
"""

import importlib.util
import pathlib
import time

import pytest
import torch
from sgl_kernel import is_xe3_arch
from nvfp4_utils import (
    NVFP4_BLOCK_SIZE,
    dequantize_mxfp8_xe35,
    dequantize_nvfp4_xe35,
    dequantize_nvfp4_via_bf16_prepared_xe35,
    dequantize_nvfp4_via_mxfp8_xe35,
    e2m1_lut_xe35,
    nvfp4_to_mxfp8_ref_xe35,
    nvfp4_woq_prep_ref_xe35,
    quantize_mxfp8_xe35,
    quantize_nvfp4_xe35,
)

WORKSPACE_BYTES = 64 * 1024 * 1024
pytestmark = [
    pytest.mark.arch("xe35"),
    pytest.mark.skipif(
        not torch.xpu.is_available() or not is_xe3_arch(),
        reason="NVFP4 WoQ requires Xe35 (compute capability 3.5)",
    ),
]


@pytest.fixture(autouse=True)
def cooldown_xe35():
    yield
    if torch.xpu.is_available():
        torch.xpu.synchronize()
        torch.xpu.empty_cache()
        time.sleep(2)


def load_accuracy_reference_xe35():
    source = (
        pathlib.Path(__file__).resolve().parents[1] / "benchmark/accuracy_reference.py"
    )
    specification = importlib.util.spec_from_file_location(
        "nvfp4_accuracy_reference", source
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_accuracy_gate_rejects_corrupt_zero_xe35():
    oracle = load_accuracy_reference_xe35()
    with pytest.raises(AssertionError):
        oracle.verify_grouped_output_xe35(
            torch.ones(1, 32),
            torch.ones(1, 64, dtype=torch.bfloat16),
            torch.zeros(1, 32, 32, dtype=torch.uint8),
            torch.ones(1, 32, 4, dtype=torch.bfloat16),
            torch.full((1, 2, 32), 127, dtype=torch.uint8),
            [1],
            "bf16",
        )


@pytest.mark.long_test
@pytest.mark.parametrize("mode", ["bf16", "mxfp8"])
@pytest.mark.parametrize(
    "routed_rows,n,k",
    [
        (64, 8192, 8192),  # Muse B8 topk8 gate/up: 64 routed rows, hidden K8192.
        (64, 8192, 4096),  # Muse B8 topk8 down: intermediate K4096, hidden N8192.
        (65, 8192, 4096),  # Generic routed-row tail, not another serving batch.
        (262144, 8192, 4096),  # Muse B8 ISL4096 topk8 prefill down projection.
    ],
)
def test_grouped_gemm_production_and_tails_xe35(mode, routed_rows, n, k):
    skip_if_no_cri_xe35()
    from sgl_kernel import nvfp4_woq_bf16_grouped_mm_xe35, nvfp4_woq_mxfp8_scaled_grouped_mm_xe35

    torch.manual_seed(42)
    experts = 256
    base, extra = divmod(routed_rows, experts)
    counts = [base] * experts
    for expert in torch.randperm(experts)[:extra].tolist():
        counts[expert] += 1
    problem_sizes, offsets = make_grouped_layout_xe35(counts, n, k, "xpu")
    weights = torch.randint(
        0, 256, (experts, n, k // 2), dtype=torch.uint8, device="xpu"
    )
    scales = torch.randint(
        124, 130, (experts, k // 32, n), dtype=torch.uint8, device="xpu"
    )
    multipliers = torch.ones((experts, n, k // 16), dtype=torch.bfloat16, device="xpu")
    dtype = torch.bfloat16 if mode == "bf16" else torch.float8_e4m3fn
    activations = torch.randint(
        -4, 5, (routed_rows, k), dtype=torch.int8, device="xpu"
    ).to(dtype)
    activation_scales = None
    output = torch.empty((routed_rows, n), dtype=torch.float32, device="xpu")
    workspace = torch.empty((WORKSPACE_BYTES,), dtype=torch.uint8, device="xpu")
    if mode == "bf16":
        nvfp4_woq_bf16_grouped_mm_xe35(
            output,
            activations,
            weights,
            scales,
            multipliers,
            problem_sizes,
            offsets,
            workspace,
        )
    else:
        activation_scales = torch.randint(
            125, 130, (routed_rows, k // 32), dtype=torch.uint8, device="xpu"
        )
        nvfp4_woq_mxfp8_scaled_grouped_mm_xe35(
            output,
            activations,
            weights,
            activation_scales,
            scales,
            multipliers,
            problem_sizes,
            offsets,
            workspace,
        )
    torch.xpu.synchronize()
    load_accuracy_reference_xe35().verify_grouped_output_xe35(
        output,
        activations,
        weights,
        multipliers,
        scales,
        counts,
        mode,
        activation_scales,
        max_rows=3,
    )


@pytest.mark.parametrize(
    "mode,expected", [("auto", "bf16"), (" BF16 ", "bf16"), (" MXFP8 ", "mxfp8")]
)
def test_compute_mode_argument_xe35(mode, expected):
    from sgl_kernel.nvfp4_moe import _nvfp4_woq_compute_mode_xe35

    assert _nvfp4_woq_compute_mode_xe35() == "bf16"
    assert _nvfp4_woq_compute_mode_xe35(mode) == expected


def test_compute_mode_rejects_invalid_argument_xe35():
    from sgl_kernel.nvfp4_moe import _nvfp4_woq_compute_mode_xe35

    with pytest.raises(ValueError, match="must be 'auto', 'mxfp8', or 'bf16'"):
        _nvfp4_woq_compute_mode_xe35("invalid")


@pytest.mark.parametrize("separate_gate_up", [False, True])
def test_prepare_weights_accepts_checkpoint_global_scales_xe35(
    monkeypatch, separate_gate_up
):
    """The high-level API takes quantizer scales, while prep takes reciprocals."""
    import sgl_kernel.nvfp4 as nvfp4
    from sgl_kernel import nvfp4_woq_prepare_weights_xe35

    monkeypatch.setattr(nvfp4, "nvfp4_woq_prep_xe35", nvfp4_woq_prep_ref_xe35)
    w1_scale = torch.tensor(
        [
            [[8.0, 4.0, 2.0, 1.0]] * 8,
            [[4.0, 2.0, 1.0, 0.5]] * 8,
        ],
        dtype=torch.float32,
    ).to(torch.float8_e4m3fn)
    w2_scale = w1_scale[:, :4].contiguous()
    if separate_gate_up:
        w1_gs = torch.tensor([[8.0, 16.0], [16.0, 32.0]])
    else:
        w1_gs = torch.tensor([[8.0], [16.0]])
    w2_gs = torch.tensor([32.0, 64.0])

    prepared = nvfp4_woq_prepare_weights_xe35(w1_scale, w1_gs, w2_scale, w2_gs)
    if separate_gate_up:
        gate = nvfp4_woq_prep_ref_xe35(w1_scale[:, :4], w1_gs[:, 0].reciprocal())
        up = nvfp4_woq_prep_ref_xe35(w1_scale[:, 4:], w1_gs[:, 1].reciprocal())
        expected_w1_c = torch.cat((gate[0], up[0]), dim=1)
        expected_w1_sfmx = torch.cat((gate[1], up[1]), dim=2)
    else:
        expected_w1_c, expected_w1_sfmx = nvfp4_woq_prep_ref_xe35(
            w1_scale, w1_gs.reshape(-1).reciprocal()
        )
    expected_w2_c, expected_w2_sfmx = nvfp4_woq_prep_ref_xe35(w2_scale, w2_gs.reciprocal())

    assert torch.equal(prepared["w1_c"], expected_w1_c)
    assert torch.equal(prepared["w1_sfmx"], expected_w1_sfmx)
    assert torch.equal(prepared["w2_c"], expected_w2_c)
    assert torch.equal(prepared["w2_sfmx"], expected_w2_sfmx)
    _, wrong_w2_sfmx = nvfp4_woq_prep_ref_xe35(w2_scale, w2_gs)
    assert not torch.equal(prepared["w2_sfmx"], wrong_w2_sfmx)


@pytest.mark.parametrize("mode", ["auto", "bf16", "mxfp8"])
def test_fused_experts_rejects_fp16_before_device_work_xe35(mode):
    from sgl_kernel import cutlass_fused_experts_nvfp4_woq_xe35

    a = torch.empty((1, 64), dtype=torch.float16)
    w1_q = torch.empty((1, 8, 32), dtype=torch.uint8)
    w2_q = torch.empty((1, 64, 2), dtype=torch.uint8)
    topk_weights = torch.ones((1, 1), dtype=torch.float16)
    topk_ids = torch.zeros((1, 1), dtype=torch.int32)
    with pytest.raises(TypeError, match="FP16 activation precision is unsupported"):
        cutlass_fused_experts_nvfp4_woq_xe35(
            a, w1_q, w2_q, {}, topk_weights, topk_ids, compute_mode=mode
        )


def is_xpu_available_xe35() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def is_cri_device_xe35() -> bool:
    if not is_xpu_available_xe35():
        return False
    try:
        from sgl_kernel import is_xe3_arch

        return is_xe3_arch()
    except ImportError:
        return False


def skip_if_no_cri_xe35():
    if not torch.xpu.is_available() or not is_xe3_arch():
        pytest.skip("NVFP4 WoQ requires Xe35 (compute capability 3.5)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_nvfp4_weights_xe35(
    e: int, n: int, k: int, device, seed: int = 0, scale: float = 0.05
):
    """Quantize a random (E, N, K) weight tensor to NVFP4.

    Returns ``(packed, sf, gs_dq, w_fp32)`` where ``gs_dq`` is the per-expert
    dequant multiplier (reciprocal of the usual global scale).
    """
    torch.manual_seed(seed)
    w = torch.randn(e, n, k) * scale
    qs, sfs, gss = [], [], []
    for i in range(e):
        q, sf, gs = quantize_nvfp4_xe35(w[i])
        qs.append(q)
        sfs.append(sf)
        gss.append(1.0 / gs)
    return (
        torch.stack(qs).to(device),
        torch.stack(sfs).to(device),
        torch.stack(gss).float().to(device),
        w,
    )


def make_grouped_layout_xe35(ms, n, k, device):
    problem_sizes = torch.tensor(
        [[m, n, k] for m in ms], dtype=torch.int32, device=device
    )
    starts, acc = [], 0
    for m in ms:
        starts.append(acc)
        acc += m
    expert_offsets = torch.tensor(starts, dtype=torch.int32, device=device)
    return problem_sizes, expert_offsets


def run_mxfp8_reference_xe35(
    a_q, a_s, b_mx, b_smx, problem_sizes, expert_offsets, workspace
):
    """Plain MXFP8 grouped GEMM over host-requantized weights — the oracle."""
    from sgl_kernel import fp8_blockwise_scaled_grouped_mm

    device = a_q.device
    e = problem_sizes.size(0)
    empty = torch.empty((0,), dtype=torch.int64, device=device)
    zeros_stride = torch.zeros((e,), dtype=torch.int64, device=device)
    zeros_layout = torch.zeros((e, 5), dtype=torch.int32, device=device)
    out = torch.zeros((a_q.size(0), b_mx.size(1)), dtype=torch.float32, device=device)
    fp8_blockwise_scaled_grouped_mm(
        out,
        empty,
        empty,
        empty,
        empty,
        empty,
        a_q,
        b_mx,
        a_s,
        b_smx,
        zeros_stride,
        zeros_stride,
        zeros_stride,
        zeros_layout,
        zeros_layout,
        problem_sizes,
        expert_offsets,
        workspace,
    )
    return out


# ---------------------------------------------------------------------------
# Reference numerics (CPU only)
# ---------------------------------------------------------------------------


def test_nvfp4_roundtrip_on_grid_xe35():
    """Values already on the E2M1 grid with an exactly representable scale
    survive a quantize/dequantize round trip."""
    lut = e2m1_lut_xe35()
    torch.manual_seed(0)
    x = lut[torch.randint(0, 16, (8, 64))] * 0.25
    x.reshape(8, 4, NVFP4_BLOCK_SIZE)[:, :, 0] = 1.5
    q, sf, gs = quantize_nvfp4_xe35(x)
    # amax = 1.5 -> gs = 2688/1.5 = 1792, sf = 448 (exact in E4M3).
    assert torch.equal(sf.float(), torch.full_like(sf.float(), 448.0))
    assert torch.equal(dequantize_nvfp4_xe35(q, sf, gs), x)


@pytest.mark.parametrize("adversarial", [False, True])
def test_requantization_error_is_small_xe35(adversarial):
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

    q, sf, gs = quantize_nvfp4_xe35(w)
    gs_dq = (1.0 / gs).reshape(1)
    w_nvfp4 = dequantize_nvfp4_xe35(q, sf, gs)
    w_emul = dequantize_nvfp4_via_mxfp8_xe35(q, sf, gs_dq)

    def rel(x, y):
        return ((x - y).norm() / y.norm().clamp_min(1e-30)).item()

    err_nvfp4 = rel(w_nvfp4, w)
    err_extra = rel(w_emul, w_nvfp4)
    assert err_extra < 0.05, f"requantization error too large: {err_extra}"
    assert (
        rel(w_emul, w) < err_nvfp4 * 1.15
    ), "emulation degrades NVFP4 by more than 15%"

    # Requantization must never clip: c * 6 stays inside E4M3's range.
    c, _ = nvfp4_woq_prep_ref_xe35(sf, gs_dq, transpose=False)
    assert (c.float() * 6.0).max().item() <= 448.0


# ---------------------------------------------------------------------------
# Device kernels
# ---------------------------------------------------------------------------


def test_prep_kernel_matches_reference_xe35():
    """The device scale-prep kernel is bit-exact against the Python reference."""
    skip_if_no_cri_xe35()
    from sgl_kernel import nvfp4_woq_prep_xe35

    e, n, k = 3, 64, 256
    dev = "xpu"
    _, sf, gs_dq, _ = make_nvfp4_weights_xe35(e, n, k, dev, seed=1)

    c_dev, sfmx_dev = nvfp4_woq_prep_xe35(sf, gs_dq)
    c_ref, sfmx_ref = nvfp4_woq_prep_ref_xe35(sf.cpu(), gs_dq.cpu(), transpose=True)
    assert torch.equal(sfmx_dev.cpu(), sfmx_ref)
    # Compare BF16 bit patterns so a rounding difference cannot hide.
    assert torch.equal(c_dev.cpu().view(torch.int16), c_ref.view(torch.int16))


def test_prep_kernel_rejects_noncontiguous_global_scales_xe35():
    """The prep kernel reads global scales as a contiguous pointer."""
    skip_if_no_cri_xe35()
    from sgl_kernel import nvfp4_woq_prep_xe35

    e, n, k = 3, 64, 256
    dev = "xpu"
    _, sf, gs_dq, _ = make_nvfp4_weights_xe35(e, n, k, dev, seed=1)
    strided_gs = torch.stack((gs_dq, gs_dq), dim=1)[:, 0]
    assert not strided_gs.is_contiguous()

    with pytest.raises(RuntimeError, match="w_gs must be contiguous"):
        nvfp4_woq_prep_xe35(sf, strided_gs)


def test_prep_kernel_rejects_cpu_global_scales_xe35():
    skip_if_no_cri_xe35()
    from sgl_kernel import nvfp4_woq_prep_xe35

    e, n, k = 2, 64, 256
    _, sf, gs_dq, _ = make_nvfp4_weights_xe35(e, n, k, "xpu", seed=2)

    with pytest.raises(
        RuntimeError, match="w_gs must be on the same XPU device as w_s"
    ):
        nvfp4_woq_prep_xe35(sf, gs_dq.cpu())


@pytest.mark.parametrize(
    "ms,n,k",
    [
        ([64, 64], 128, 256),
        ([64, 64], 128, 64),
        ([1, 1], 128, 64),  # M8N128 tile
        ([1] * 256, 256, 64),  # M8N256 tile
        ([16, 16], 128, 64),  # M32N256 tile
        ([65, 65], 128, 64),  # M256N256 tile
        ([32, 96], 128, 256),  # ragged M
        ([33, 95], 128, 256),  # M not 4-aligned (scale buffer is padded)
        ([0, 1, 7, 0, 33], 128, 64),  # empty experts and compact-scale offsets
        ([64, 64], 192, 256),  # N not a multiple of the workgroup N tile
        pytest.param(
            [256, 256],
            512,
            512,
            marks=pytest.mark.long_test,
        ),
        pytest.param(
            [64, 64],
            128,
            1024,
            marks=pytest.mark.long_test,
        ),
    ],
)
def test_fused_woq_matches_host_requantized_mxfp8_xe35(ms, n, k):
    """The fused WoQ GEMM must reproduce host-side requantization exactly.

    Both paths multiply the same E4M3 values by the same UE8M0 scales, so any
    difference is a layout or indexing bug rather than rounding.
    """
    skip_if_no_cri_xe35()
    from sgl_kernel import nvfp4_woq_mxfp8_scaled_grouped_mm_xe35, nvfp4_woq_prep_xe35

    dev = "xpu"
    m_total = sum(ms)
    packed, sf, gs_dq, _ = make_nvfp4_weights_xe35(len(ms), n, k, dev, seed=2)

    torch.manual_seed(3)
    a_q, a_s = quantize_mxfp8_xe35(torch.randn(m_total, k) * 0.5)
    a_q, a_s = a_q.to(dev), a_s.to(dev)

    problem_sizes, expert_offsets = make_grouped_layout_xe35(ms, n, k, dev)
    workspace = torch.zeros((WORKSPACE_BYTES,), dtype=torch.uint8, device=dev)

    b_mx, b_smx = nvfp4_to_mxfp8_ref_xe35(packed.cpu(), sf.cpu(), gs_dq.cpu())
    ref = run_mxfp8_reference_xe35(
        a_q, a_s, b_mx.to(dev), b_smx.to(dev), problem_sizes, expert_offsets, workspace
    )
    torch.xpu.synchronize()

    c_b, sfmx_b = nvfp4_woq_prep_xe35(sf, gs_dq)
    out = torch.zeros((m_total, n), dtype=torch.float32, device=dev)
    nvfp4_woq_mxfp8_scaled_grouped_mm_xe35(
        out, a_q, packed, a_s, sfmx_b, c_b, problem_sizes, expert_offsets, workspace
    )
    torch.xpu.synchronize()
    actual = out.cpu()
    oracle = ref.cpu()
    a_dq = dequantize_mxfp8_xe35(a_q.cpu(), a_s.cpu())
    b_dq = dequantize_mxfp8_xe35(b_mx, b_smx)
    direct = torch.empty_like(actual)
    row = 0
    for expert, expert_m in enumerate(ms):
        direct[row : row + expert_m] = (
            a_dq[row : row + expert_m].float() @ b_dq[expert].float().T
        )
        row += expert_m
    mismatch_count = (actual != oracle).sum().item()
    torch.testing.assert_close(actual, direct, rtol=1e-5, atol=1e-6)
    assert torch.equal(actual, oracle), (
        f"fused WoQ != host-requantized MXFP8; max diff "
        f"{(actual - oracle).abs().max().item()}, mismatches {mismatch_count}; "
        f"fused/direct max diff {(actual - direct).abs().max().item()}, "
        f"oracle/direct max diff {(oracle - direct).abs().max().item()}"
    )


def test_mxfp8_gemm_covers_all_e2m1_codes_xe35():
    """Every nibble and packed-byte position must reach the right MXFP8 lane."""
    skip_if_no_cri_xe35()
    from sgl_kernel import nvfp4_woq_mxfp8_scaled_grouped_mm_xe35

    dev = "xpu"
    e, m, n, k = 1, 64, 128, 64
    torch.manual_seed(11)
    codes = torch.randint(0, 16, (n, k), dtype=torch.uint8)
    assert torch.all(torch.bincount(codes.flatten().long(), minlength=16) > 0)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).unsqueeze(0).to(dev)
    a = torch.eye(m, k, dtype=torch.float8_e4m3fn, device=dev)
    scales_a = torch.full((m, k // 32), 127, dtype=torch.uint8, device=dev)
    c_b = torch.ones((e, n, k // 16), dtype=torch.bfloat16, device=dev)
    sfmx_b = torch.full((e, k // 32, n), 127, dtype=torch.uint8, device=dev)
    problem_sizes, expert_offsets = make_grouped_layout_xe35([m], n, k, dev)
    workspace = torch.empty((WORKSPACE_BYTES,), dtype=torch.uint8, device=dev)
    out = torch.zeros((m, n), dtype=torch.float32, device=dev)

    nvfp4_woq_mxfp8_scaled_grouped_mm_xe35(
        out,
        a,
        packed,
        scales_a,
        sfmx_b,
        c_b,
        problem_sizes,
        expert_offsets,
        workspace,
    )
    torch.xpu.synchronize()
    expected = e2m1_lut_xe35()[codes.long()].T.float()
    actual = out.cpu()
    mismatch = actual != expected
    mismatch_indices = mismatch.nonzero()
    assert torch.equal(actual, expected), (
        f"MXFP8 E2M1 payload mismatch at {mismatch_indices.shape[0]} entries; "
        f"first indices {mismatch_indices[:16].tolist()}, "
        f"actual {actual[mismatch][:16].tolist()}, "
        f"expected {expected[mismatch][:16].tolist()}"
    )


def test_mxfp8_gemm_scale_layouts_xe35():
    """All scale axes and per-expert pointers use their declared coordinates."""
    skip_if_no_cri_xe35()
    from sgl_kernel import nvfp4_woq_mxfp8_scaled_grouped_mm_xe35

    dev = "xpu"
    e, expert_m, n, k = 2, 64, 128, 64
    m_total = e * expert_m
    base_codes = torch.arange(n * k, dtype=torch.uint8).reshape(n, k)
    codes = torch.stack([(base_codes + 3 * i).remainder(16) for i in range(e)])
    packed = (codes[:, :, 0::2] | (codes[:, :, 1::2] << 4)).to(dev)
    a = torch.eye(expert_m, k).to(torch.float8_e4m3fn).repeat(e, 1).to(dev)

    row = torch.arange(m_total).unsqueeze(1)
    row_expert = row // expert_m
    k32 = torch.arange(k // 32).unsqueeze(0)
    a_exp = (row_expert + row + 2 * k32).remainder(4) - 10
    scales_a = (a_exp + 127).to(torch.uint8).to(dev)

    expert = torch.arange(e).view(e, 1, 1)
    col = torch.arange(n).view(1, n, 1)
    k32 = torch.arange(k // 32).view(1, 1, -1)
    b_exp = (expert + col + 2 * k32).remainder(4) - 14
    sfmx_b = (b_exp.permute(0, 2, 1) + 127).to(torch.uint8).contiguous().to(dev)

    k16 = torch.arange(k // 16).view(1, 1, -1)
    c_index = 17 * expert + 5 * col + 11 * k16
    c_cpu = ((c_index.remainder(251) + 1).float() / 4).to(torch.bfloat16)
    c_b = c_cpu.to(dev)

    problem_sizes, expert_offsets = make_grouped_layout_xe35([expert_m] * e, n, k, dev)
    workspace = torch.empty((WORKSPACE_BYTES,), dtype=torch.uint8, device=dev)
    out = torch.zeros((m_total, n), dtype=torch.float32, device=dev)
    nvfp4_woq_mxfp8_scaled_grouped_mm_xe35(
        out,
        a,
        packed,
        scales_a,
        sfmx_b,
        c_b,
        problem_sizes,
        expert_offsets,
        workspace,
    )
    torch.xpu.synchronize()

    expected = torch.empty((m_total, n), dtype=torch.float32)
    local_rows = torch.arange(expert_m)
    for expert_idx in range(e):
        flat_rows = expert_idx * expert_m + local_rows
        diag_codes = codes[expert_idx, :, local_rows].T.long()
        diag_b_exp = b_exp[expert_idx, :, local_rows // 32].T
        diag_a_exp = a_exp[flat_rows, local_rows // 32].unsqueeze(1)
        requantized = (
            (e2m1_lut_xe35()[diag_codes] * c_cpu[expert_idx, :, local_rows // 16].T.float())
            .to(torch.float8_e4m3fn)
            .float()
        )
        expected[flat_rows] = requantized * torch.exp2(
            (diag_b_exp + diag_a_exp).float()
        )
    actual = out.cpu()
    mismatch = actual != expected
    mismatch_indices = mismatch.nonzero()
    assert torch.equal(actual, expected), (
        f"MXFP8 scale-layout mismatch at {mismatch_indices.shape[0]} entries; "
        f"first indices {mismatch_indices[:16].tolist()}, "
        f"actual {actual[mismatch][:16].tolist()}, "
        f"expected {expected[mismatch][:16].tolist()}"
    )


def test_mxfp8_gemm_random_a_against_identity_b_xe35():
    skip_if_no_cri_xe35()
    from sgl_kernel import nvfp4_woq_mxfp8_scaled_grouped_mm_xe35

    dev = "xpu"
    e, m, n, k = 1, 64, 128, 64
    codes = torch.zeros((n, k), dtype=torch.uint8)
    codes[torch.arange(k), torch.arange(k)] = 2  # E2M1 value 1.0
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).unsqueeze(0).to(dev)
    torch.manual_seed(3)
    a_q_cpu, a_s_cpu = quantize_mxfp8_xe35(torch.randn(m, k) * 0.5)
    a_q, a_s = a_q_cpu.to(dev), a_s_cpu.to(dev)
    c_b = torch.ones((e, n, k // 16), dtype=torch.bfloat16, device=dev)
    sfmx_b = torch.full((e, k // 32, n), 127, dtype=torch.uint8, device=dev)
    problem_sizes, expert_offsets = make_grouped_layout_xe35([m], n, k, dev)
    workspace = torch.zeros((WORKSPACE_BYTES,), dtype=torch.uint8, device=dev)
    out = torch.zeros((m, n), dtype=torch.float32, device=dev)

    nvfp4_woq_mxfp8_scaled_grouped_mm_xe35(
        out,
        a_q,
        packed,
        a_s,
        sfmx_b,
        c_b,
        problem_sizes,
        expert_offsets,
        workspace,
    )
    torch.xpu.synchronize()

    actual = out.cpu()[:, :k]
    expected = dequantize_mxfp8_xe35(a_q_cpu, a_s_cpu)
    assert torch.equal(
        actual, expected
    ), f"random A payload mismatch; max diff {(actual - expected).abs().max().item()}"


def test_mxfp8_gemm_prepared_b_against_identity_a_xe35():
    skip_if_no_cri_xe35()
    from sgl_kernel import nvfp4_woq_mxfp8_scaled_grouped_mm_xe35, nvfp4_woq_prep_xe35

    dev = "xpu"
    e, m, n, k = 1, 64, 128, 64
    packed, sf, gs_dq, _ = make_nvfp4_weights_xe35(e, n, k, dev, seed=2)
    c_b, sfmx_b = nvfp4_woq_prep_xe35(sf, gs_dq)
    a = torch.eye(m, k, dtype=torch.float8_e4m3fn, device=dev)
    scales_a = torch.full((m, k // 32), 127, dtype=torch.uint8, device=dev)
    problem_sizes, expert_offsets = make_grouped_layout_xe35([m], n, k, dev)
    workspace = torch.zeros((WORKSPACE_BYTES,), dtype=torch.uint8, device=dev)
    out = torch.zeros((m, n), dtype=torch.float32, device=dev)

    nvfp4_woq_mxfp8_scaled_grouped_mm_xe35(
        out,
        a,
        packed,
        scales_a,
        sfmx_b,
        c_b,
        problem_sizes,
        expert_offsets,
        workspace,
    )
    torch.xpu.synchronize()

    b_mx, b_smx = nvfp4_to_mxfp8_ref_xe35(packed.cpu(), sf.cpu(), gs_dq.cpu())
    expected = dequantize_mxfp8_xe35(b_mx, b_smx)[0].T
    actual = out.cpu()
    mismatch = actual != expected
    mismatch_indices = mismatch.nonzero()
    assert torch.equal(actual, expected), (
        f"prepared B mismatch at {mismatch_indices.shape[0]} entries; "
        f"first indices {mismatch_indices[:16].tolist()}"
    )


def test_mxfp8_dense_single_expert_matches_direct_xe35():
    skip_if_no_cri_xe35()
    from sgl_kernel import nvfp4_woq_mxfp8_scaled_grouped_mm_xe35, nvfp4_woq_prep_xe35

    dev = "xpu"
    e, m, n, k = 1, 64, 128, 64
    packed, sf, gs_dq, _ = make_nvfp4_weights_xe35(e, n, k, dev, seed=2)
    torch.manual_seed(3)
    a_q_cpu, a_s_cpu = quantize_mxfp8_xe35(torch.randn(m, k) * 0.5)
    a_q, a_s = a_q_cpu.to(dev), a_s_cpu.to(dev)
    c_b, sfmx_b = nvfp4_woq_prep_xe35(sf, gs_dq)
    problem_sizes, expert_offsets = make_grouped_layout_xe35([m], n, k, dev)
    workspace = torch.zeros((WORKSPACE_BYTES,), dtype=torch.uint8, device=dev)
    out = torch.zeros((m, n), dtype=torch.float32, device=dev)

    nvfp4_woq_mxfp8_scaled_grouped_mm_xe35(
        out,
        a_q,
        packed,
        a_s,
        sfmx_b,
        c_b,
        problem_sizes,
        expert_offsets,
        workspace,
    )
    torch.xpu.synchronize()

    b_mx, b_smx = nvfp4_to_mxfp8_ref_xe35(packed.cpu(), sf.cpu(), gs_dq.cpu())
    a_dq = dequantize_mxfp8_xe35(a_q_cpu, a_s_cpu)
    b_dq = dequantize_mxfp8_xe35(b_mx, b_smx)[0]
    expected = a_dq @ b_dq.T
    actual = out.cpu()
    no_a_scale = a_q_cpu.float() @ b_dq.T
    no_b_scale = a_dq @ b_mx[0].float().T
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6), (
        f"dense fused/direct max diff {(actual - expected).abs().max().item()}; "
        f"actual rms {actual.square().mean().sqrt().item()}, "
        f"expected rms {expected.square().mean().sqrt().item()}, "
        f"no-A-scale max diff {(actual - no_a_scale).abs().max().item()}, "
        f"no-B-scale max diff {(actual - no_b_scale).abs().max().item()}"
    )


def test_bf16_gemm_covers_all_e2m1_codes_xe35():
    """Every nibble and packed-byte position must reach the right BF16 lane."""
    skip_if_no_cri_xe35()
    from sgl_kernel import nvfp4_woq_bf16_grouped_mm_xe35

    dev = "xpu"
    e, m, n, k = 1, 32, 64, 32
    codes = torch.arange(n * k, dtype=torch.uint8).remainder(16).reshape(n, k)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).unsqueeze(0).to(dev)
    a = torch.eye(m, k, dtype=torch.bfloat16, device=dev)
    c_b = torch.ones((e, n, k // 16), dtype=torch.bfloat16, device=dev)
    sfmx_b = torch.full((e, k // 32, n), 127, dtype=torch.uint8, device=dev)
    problem_sizes, expert_offsets = make_grouped_layout_xe35([m], n, k, dev)
    workspace = torch.empty((WORKSPACE_BYTES,), dtype=torch.uint8, device=dev)
    out = torch.zeros((m, n), dtype=torch.float32, device=dev)

    nvfp4_woq_bf16_grouped_mm_xe35(
        out, a, packed, sfmx_b, c_b, problem_sizes, expert_offsets, workspace
    )
    torch.xpu.synchronize()
    expected = e2m1_lut_xe35()[codes.long()].T.float()
    assert torch.equal(out.cpu(), expected)


@pytest.mark.parametrize(
    "ms,n,k",
    [
        ([32, 96], 128, 256),
        ([33, 31], 192, 64),
        ([1, 7], 128, 32),
        pytest.param(
            [64, 64],
            128,
            1024,
            marks=pytest.mark.long_test,
        ),
    ],
)
def test_bf16_grouped_gemm_matches_direct_dequant_xe35(ms, n, k):
    skip_if_no_cri_xe35()
    from sgl_kernel import nvfp4_woq_bf16_grouped_mm_xe35, nvfp4_woq_prep_xe35

    dev = "xpu"
    m_total = sum(ms)
    packed, sf, gs_dq, _ = make_nvfp4_weights_xe35(len(ms), n, k, dev, seed=21)
    torch.manual_seed(22)
    a_cpu = (torch.randn(m_total, k) * 0.5).to(torch.bfloat16)
    a = a_cpu.to(dev)
    c_b, sfmx_b = nvfp4_woq_prep_xe35(sf, gs_dq)
    problem_sizes, expert_offsets = make_grouped_layout_xe35(ms, n, k, dev)
    workspace = torch.empty((WORKSPACE_BYTES,), dtype=torch.uint8, device=dev)
    out = torch.zeros((m_total, n), dtype=torch.float32, device=dev)

    nvfp4_woq_bf16_grouped_mm_xe35(
        out, a, packed, sfmx_b, c_b, problem_sizes, expert_offsets, workspace
    )
    torch.xpu.synchronize()

    weights = dequantize_nvfp4_via_bf16_prepared_xe35(packed.cpu(), c_b.cpu(), sfmx_b.cpu())
    ref = torch.zeros((m_total, n), dtype=torch.float32)
    start = 0
    for expert, expert_m in enumerate(ms):
        ref[start : start + expert_m] = (
            a_cpu[start : start + expert_m].float() @ weights[expert].float().T
        )
        start += expert_m
    torch.testing.assert_close(out.cpu(), ref, rtol=5e-3, atol=5e-3)


def test_unaligned_scale_stride_is_rejected_xe35():
    """Shapes whose per-expert B-scale slice isn't 64-byte aligned must raise.

    The Xe block-2D scale load needs a 64-byte aligned base address, so
    N * (K/32) has to be a multiple of 64. Fail on the host with a readable
    message instead of letting the device-side assert fire per work-item.
    """
    skip_if_no_cri_xe35()
    from sgl_kernel import nvfp4_woq_mxfp8_scaled_grouped_mm_xe35, nvfp4_woq_prep_xe35

    dev = "xpu"
    ms, n, k = [64, 64], 132, 256  # 132 * (256/32) = 1056, not a multiple of 64
    m_total = sum(ms)
    packed, sf, gs_dq, _ = make_nvfp4_weights_xe35(len(ms), n, k, dev, seed=4)
    a_q, a_s = quantize_mxfp8_xe35(torch.randn(m_total, k))
    c_b, sfmx_b = nvfp4_woq_prep_xe35(sf, gs_dq)
    problem_sizes, expert_offsets = make_grouped_layout_xe35(ms, n, k, dev)
    workspace = torch.zeros((WORKSPACE_BYTES,), dtype=torch.uint8, device=dev)
    out = torch.zeros((m_total, n), dtype=torch.float32, device=dev)

    with pytest.raises(RuntimeError, match="multiple of 64"):
        nvfp4_woq_mxfp8_scaled_grouped_mm_xe35(
            out,
            a_q.to(dev),
            packed,
            a_s.to(dev),
            sfmx_b,
            c_b,
            problem_sizes,
            expert_offsets,
            workspace,
        )


def test_grouped_gemms_reject_cpu_problem_sizes_xe35():
    skip_if_no_cri_xe35()
    from sgl_kernel import (
        nvfp4_woq_bf16_grouped_mm_xe35,
        nvfp4_woq_mxfp8_scaled_grouped_mm_xe35,
        nvfp4_woq_prep_xe35,
    )

    dev = "xpu"
    e, m, n, k = 1, 32, 64, 64
    packed, sf, gs_dq, _ = make_nvfp4_weights_xe35(e, n, k, dev, seed=5)
    c_b, sfmx_b = nvfp4_woq_prep_xe35(sf, gs_dq)
    problem_sizes, expert_offsets = make_grouped_layout_xe35([m], n, k, dev)
    workspace = torch.empty((WORKSPACE_BYTES,), dtype=torch.uint8, device=dev)
    out = torch.zeros((m, n), dtype=torch.float32, device=dev)

    a_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device=dev)
    with pytest.raises(RuntimeError, match="problem_sizes must be on the same XPU"):
        nvfp4_woq_bf16_grouped_mm_xe35(
            out,
            a_bf16,
            packed,
            sfmx_b,
            c_b,
            problem_sizes.cpu(),
            expert_offsets,
            workspace,
        )

    a_q, a_s = quantize_mxfp8_xe35(torch.randn(m, k))
    with pytest.raises(RuntimeError, match="problem_sizes must be on the same XPU"):
        nvfp4_woq_mxfp8_scaled_grouped_mm_xe35(
            out,
            a_q.to(dev),
            packed,
            a_s.to(dev),
            sfmx_b,
            c_b,
            problem_sizes.cpu(),
            expert_offsets,
            workspace,
        )


# ---------------------------------------------------------------------------
# End-to-end MoE
# ---------------------------------------------------------------------------


def _moe_reference_xe35(a, w1_eff, w2_eff, topk_weights, topk_ids, n_inter, mode):
    """Fused-MoE reference over already-dequantized (emulated) weights.

    Mirrors the production BF16 or MXFP8 operand precision, SiLU-and-mul in
    the activation dtype, and unrounded fp32 output accumulation.
    """
    m, _ = a.shape
    dt = a.dtype
    if mode == "bf16":
        a_dq = a.to(torch.bfloat16).float()
    else:
        a_dq = dequantize_mxfp8_xe35(*quantize_mxfp8_xe35(a.float()))
    out = torch.zeros(m, w2_eff.shape[1], dtype=torch.float32)
    for t in range(m):
        for j in range(topk_ids.shape[1]):
            e = int(topk_ids[t, j])
            h = a_dq[t] @ w1_eff[e].T
            gate, up = h[:n_inter], h[n_inter:]
            inter = torch.nn.functional.silu(gate.to(dt).float()) * up.to(dt).float()
            if mode == "bf16":
                inter_dq = inter.to(torch.bfloat16).float()
            else:
                inter_dq = dequantize_mxfp8_xe35(
                    *quantize_mxfp8_xe35(inter.unsqueeze(0).to(dt).float())
                )[0]
            out[t] += float(topk_weights[t, j]) * (inter_dq @ w2_eff[e].T)
    return out


@pytest.mark.parametrize("mode", ["mxfp8", "bf16"])
def test_fused_experts_nvfp4_woq_xe35(mode, monkeypatch):
    """End-to-end MoE with NVFP4 weights through the fused WoQ path."""
    skip_if_no_cri_xe35()
    import sgl_kernel.nvfp4_moe as cutlass_moe
    from sgl_kernel import cutlass_fused_experts_nvfp4_woq_xe35, nvfp4_woq_prepare_weights_xe35

    if mode == "bf16":

        def unexpected_quantization(*args, **kwargs):
            raise AssertionError("BF16 dispatch must not quantize activations to MXFP8")

        monkeypatch.setattr(
            cutlass_moe, "_per_token_group_quant_mxfp8_xe35", unexpected_quantization
        )

    dev = "xpu"
    dt = torch.bfloat16
    e, m, k, n_inter, topk = 4, 32, 256, 64, 2

    torch.manual_seed(7)
    a = (torch.randn(m, k) * 0.5).to(dt)
    gate_q, gate_sf, gate_gs, _ = make_nvfp4_weights_xe35(e, n_inter, k, dev, seed=11)
    up_q, up_sf, up_gs, _ = make_nvfp4_weights_xe35(e, n_inter, k, dev, seed=13)
    w1_q = torch.cat((gate_q, up_q), dim=1)
    w1_sf = torch.cat((gate_sf, up_sf), dim=1)
    w1_gs = torch.stack((gate_gs, up_gs), dim=1)
    w2_q, w2_sf, w2_gs, _ = make_nvfp4_weights_xe35(e, k, n_inter, dev, seed=12)
    topk_weights, topk_ids = torch.topk(
        torch.softmax(torch.randn(m, e), -1), topk, dim=-1
    )
    assert topk_ids.dtype == torch.int64

    # This API takes checkpoint quantizer global scales, not dequant multipliers.
    prepared = nvfp4_woq_prepare_weights_xe35(
        w1_sf, w1_gs.reciprocal(), w2_sf, w2_gs.reciprocal()
    )
    out = cutlass_fused_experts_nvfp4_woq_xe35(
        a.to(dev),
        w1_q,
        w2_q,
        prepared,
        topk_weights.to(dt).to(dev),
        topk_ids.to(dev),
        compute_mode=mode,
    )
    torch.xpu.synchronize()

    if mode == "bf16":
        w1_eff = dequantize_nvfp4_via_bf16_prepared_xe35(
            w1_q.cpu(), prepared["w1_c"].cpu(), prepared["w1_sfmx"].cpu()
        ).float()
        w2_eff = dequantize_nvfp4_via_bf16_prepared_xe35(
            w2_q.cpu(), prepared["w2_c"].cpu(), prepared["w2_sfmx"].cpu()
        ).float()
    else:
        w1_eff = torch.stack(
            [
                torch.cat(
                    (
                        dequantize_nvfp4_via_mxfp8_xe35(
                            gate_q[i].cpu(), gate_sf[i].cpu(), gate_gs[i].cpu()
                        ),
                        dequantize_nvfp4_via_mxfp8_xe35(
                            up_q[i].cpu(), up_sf[i].cpu(), up_gs[i].cpu()
                        ),
                    ),
                    dim=0,
                )
                for i in range(e)
            ]
        )
        w2_eff = torch.stack(
            [
                dequantize_nvfp4_via_mxfp8_xe35(
                    w2_q[i].cpu(), w2_sf[i].cpu(), w2_gs[i].cpu()
                )
                for i in range(e)
            ]
        )
    ref = _moe_reference_xe35(a, w1_eff, w2_eff, topk_weights, topk_ids, n_inter, mode)

    torch.testing.assert_close(out.float().cpu(), ref, rtol=5e-3, atol=5e-3)
