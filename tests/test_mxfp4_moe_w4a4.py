# SPDX-License-Identifier: Apache-2.0
"""Tests for the MXFP4 W4A4 MoE grouped GEMM on Intel Xe3.5 (CRI) --
`moe_grouped_mm_nt_mxfp4_w4a4_xe35` -- and for the activation-scale relayout it
consumes, `mxfp4_moe_scale_to_mn_major_xe35`.

Layout contract under test (see src/sycl/xe35/moe_grouped_mm_mxfp4_w4a4.cpp):

    a        [total_M, K/2]    packed E2M1, row-major, rows grouped by expert
    a_scale  [padded_M, K/32]  UE8M0; each expert's block is written MN-major with
                               its row count rounded up to 4 -- exactly what
                               mxfp4_moe_scale_to_mn_major_xe35 emits
    b        [E, N, K/2]       packed E2M1, contiguous
    b_scale  [E, K/32, N]      UE8M0, contiguous, or its [E, N, K/32] transposed
                               view: N must be the dense extent
    out      [total_M, N]      bfloat16
    rows_per_expert [E]        int32, on device

This extends the existing MXFP4 grouped GEMM coverage with the W4A4 path.
"""

import importlib.util
import math
import pathlib

import pytest
import torch
from mxfp4_utils import MXFP4_BLOCK_SIZE, dequantize_mxfp4_2d, quantize_mxfp4_2d

# The 64-byte base-pointer rule below makes the per-expert row counts, and with them
# total_M, grow fast; keep the wider shapes out of the default (simulator) run.

SCALE_ALIGN_ROWS = 4
PTR_ALIGN_BYTES = 64


def is_xpu_available_xe35() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def is_cri_device_xe35() -> bool:
    if not is_xpu_available_xe35():
        return False
    from sgl_kernel import is_xe3_arch

    return is_xe3_arch()


requires_cri = pytest.mark.skipif(
    not is_xpu_available_xe35() or not is_cri_device_xe35(),
    reason="the MXFP4 W4A4 kernels require Intel Xe3.5",
)


def load_accuracy_reference_xe35():
    source = (
        pathlib.Path(__file__).resolve().parents[1] / "benchmark/accuracy_reference.py"
    )
    specification = importlib.util.spec_from_file_location(
        "mxfp4_accuracy_reference", source
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_accuracy_gate_rejects_corrupt_zero_xe35():
    with pytest.raises(AssertionError):
        load_accuracy_reference_xe35().verify_grouped_output_xe35(
            torch.ones(1, 32),
            torch.zeros(1, 64, dtype=torch.uint8),
            torch.full((4, 4), 127, dtype=torch.uint8),
            torch.zeros(1, 32, 64, dtype=torch.uint8),
            torch.full((1, 4, 32), 127, dtype=torch.uint8),
            [1],
        )


@requires_cri
@pytest.mark.long_test
@pytest.mark.parametrize(
    "allocated_rows,n",
    [
        (256, 8192),  # Muse B8 topk8 compressed gate/up: K4096, N8192.
        (257, 4096),  # Generic allocated-row tail, not a serving batch.
        (262144, 4096),  # Muse B8 ISL4096 topk8 prefill compressed down projection.
    ],
)
def test_grouped_gemm_production_and_tails_xe35(allocated_rows, n):
    if not is_cri_device_xe35():
        pytest.skip("MXFP4 W4A4 requires Xe3.5 capability")
    from sgl_kernel import moe_grouped_mm_nt_mxfp4_w4a4_xe35

    torch.manual_seed(42)
    experts, k = 256, 4096
    if allocated_rows <= 257:
        counts = [0] * experts
        for expert in torch.randperm(experts)[: 8 * 8].tolist():
            counts[expert] = 4
    else:
        counts = [allocated_rows // experts] * experts
    rows = torch.tensor(counts, dtype=torch.int32, device="xpu")
    activations = torch.randint(
        0, 256, (allocated_rows, k // 2), dtype=torch.uint8, device="xpu"
    )
    weights = torch.randint(
        0, 256, (experts, n, k // 2), dtype=torch.uint8, device="xpu"
    )
    activation_scales = torch.randint(
        124, 130, (allocated_rows, k // 32), dtype=torch.uint8, device="xpu"
    )
    weight_scales = torch.randint(
        124, 130, (experts, k // 32, n), dtype=torch.uint8, device="xpu"
    )
    output = torch.empty((allocated_rows, n), dtype=torch.bfloat16, device="xpu")
    moe_grouped_mm_nt_mxfp4_w4a4_xe35(
        output,
        activations,
        activation_scales,
        weights,
        weight_scales,
        rows,
        n,
        k,
        experts,
    )
    torch.xpu.synchronize()
    load_accuracy_reference_xe35().verify_grouped_output_xe35(
        output,
        activations,
        activation_scales,
        weights,
        weight_scales,
        counts,
        max_rows=4,
    )


def row_alignment_xe35(k: int) -> int:
    """Smallest per-expert row multiple that keeps every A and A-scale base pointer
    64-byte aligned, mirroring sglang's `_xpu_w4a4_row_alignment`. A row is K/2
    bytes of packed E2M1 and one byte per K/32 scale group.
    """
    alignment = SCALE_ALIGN_ROWS
    for elems_per_row in (k // 2, k // MXFP4_BLOCK_SIZE):
        needed = PTR_ALIGN_BYTES // math.gcd(PTR_ALIGN_BYTES, elems_per_row)
        alignment = math.lcm(alignment, needed)
    return alignment


def padded_xe35(rows: int) -> int:
    return (rows + SCALE_ALIGN_ROWS - 1) // SCALE_ALIGN_ROWS * SCALE_ALIGN_ROWS


def mn_major_reference_xe35(scales_rm: torch.Tensor, rows_per_expert) -> torch.Tensor:
    """Host reference for mxfp4_moe_scale_to_mn_major_xe35.

    Row-major [sum(rows), G] -> [sum(padded_xe35(rows)), G], where each expert's block of
    `G * padded_xe35(rows_e)` bytes holds that expert's scales MN-major (M fastest) and the
    padding rows are zero.
    """
    groups = scales_rm.shape[1]
    out = torch.zeros(
        sum(padded_xe35(r) for r in rows_per_expert), groups, dtype=torch.uint8
    )
    src = dst = 0
    for rows in rows_per_expert:
        block = out.view(-1)[dst * groups : (dst + padded_xe35(rows)) * groups].view(
            groups, padded_xe35(rows)
        )
        block[:, :rows] = scales_rm[src : src + rows].transpose(0, 1)
        src += rows
        dst += padded_xe35(rows)
    return out


def make_expert_operands_xe35(rows_per_expert, n: int, k: int, seed: int = 0):
    """Quantize per-expert A and B on the host and return the kernel's operands
    alongside the dequantized fp32 reference output.
    """
    torch.manual_seed(seed)
    a_packed, a_scales_rm, b_packed, b_scales, ref_rows = [], [], [], [], []

    for e, rows in enumerate(rows_per_expert):
        b_q, b_s = quantize_mxfp4_2d(torch.randn(n, k, dtype=torch.float32) * 2.0)
        b_packed.append(b_q)
        # b_scale must be dense along N: [G, N] per expert.
        b_scales.append(b_s.transpose(0, 1).contiguous())
        b_deq = dequantize_mxfp4_2d(b_q, b_s, torch.float32)

        if rows == 0:
            continue
        a_q, a_s = quantize_mxfp4_2d(torch.randn(rows, k, dtype=torch.float32) * 2.0)
        a_packed.append(a_q)
        a_scales_rm.append(a_s)
        ref_rows.append(
            dequantize_mxfp4_2d(a_q, a_s, torch.float32) @ b_deq.transpose(0, 1)
        )

    return {
        "a": torch.cat(a_packed, dim=0),
        "a_scale": mn_major_reference_xe35(
            torch.cat(a_scales_rm, dim=0), [r for r in rows_per_expert if r]
        ),
        "b": torch.stack(b_packed, dim=0).contiguous(),
        "b_scale": torch.stack(b_scales, dim=0).contiguous(),
        "reference": torch.cat(ref_rows, dim=0),
    }


def run_grouped_mm_xe35(
    operands, rows_per_expert, n, k, *, transposed_b_scale=False, bias=None
):
    from sgl_kernel import moe_grouped_mm_nt_mxfp4_w4a4_xe35

    device = "xpu"
    total_m = operands["a"].shape[0]
    b_scale = operands["b_scale"].to(device)
    if transposed_b_scale:
        # [E, N, G] view of the same bytes; still N-dense, and accepted as such.
        b_scale = b_scale.transpose(1, 2)

    out = torch.empty(total_m, n, dtype=torch.bfloat16, device=device)
    moe_grouped_mm_nt_mxfp4_w4a4_xe35(
        out,
        operands["a"].to(device),
        operands["a_scale"].to(device),
        operands["b"].to(device),
        b_scale,
        torch.tensor(rows_per_expert, dtype=torch.int32, device=device),
        n,
        k,
        len(rows_per_expert),
        bias=None if bias is None else bias.to(device),
    )
    return out


def assert_moe_close_xe35(got: torch.Tensor, want: torch.Tensor):
    tolerance = 1e-3 if got.dtype == torch.float16 else 5e-3
    got = got.float().cpu()
    want = want.float().cpu()
    assert got.shape == want.shape, f"{got.shape} vs {want.shape}"
    assert not torch.isnan(got).any(), "kernel produced NaN"
    assert not torch.isinf(got).any(), "kernel produced Inf"
    torch.testing.assert_close(got, want, rtol=tolerance, atol=tolerance)
    if want.abs().max() > 0:
        correlation = torch.corrcoef(torch.stack([got.flatten(), want.flatten()]))[0, 1]
        assert correlation > 0.99, f"correlation too low: {correlation}"


# ---------------------------------------------------------------------------
# mxfp4_moe_scale_to_mn_major_xe35
# ---------------------------------------------------------------------------


@requires_cri
class TestMxfp4MoeScaleToMnMajor_xe35:
    @pytest.mark.parametrize(
        "rows_per_expert",
        [
            [4, 4],  # already 4-row aligned: a pure per-expert transpose
            [3, 5],  # ragged: both blocks grow to the next multiple of 4
            [16, 0, 8],  # an unrouted expert contributes no rows at all
            [1],
            [64, 4],  # more than one 64-row tile in the first expert
        ],
        ids=lambda r: "x".join(str(x) for x in r),
    )
    # groups % 4 == 0 takes the vectorized (32-bit quad) path, the rest the
    # byte-granular one; K = 32 * groups.
    @pytest.mark.parametrize("groups", [2, 4])
    def test_matches_reference_xe35(self, rows_per_expert, groups):
        from sgl_kernel import mxfp4_moe_scale_to_mn_major_xe35

        torch.manual_seed(len(rows_per_expert) * 100 + groups)
        rows = sum(rows_per_expert)
        scales_rm = torch.randint(100, 160, (rows, groups), dtype=torch.uint8)
        want = mn_major_reference_xe35(scales_rm, rows_per_expert)

        # Zeroed, as the op's docstring requires whenever a row count is not a
        # multiple of 4 (the padding rows are read by the GEMM, never written here).
        got = torch.zeros(want.shape[0], groups, dtype=torch.uint8, device="xpu")
        mxfp4_moe_scale_to_mn_major_xe35(
            got,
            scales_rm.to("xpu"),
            torch.tensor(rows_per_expert, dtype=torch.int32, device="xpu"),
        )
        assert torch.equal(got.cpu(), want)

    def test_slack_rows_are_left_alone_xe35(self):
        """Callers size the surface from a host-side upper bound, so rows past
        sum(padded_xe35(rows_e)) must not be touched -- sglang relies on that to pass an
        `empty_like` buffer.
        """
        from sgl_kernel import mxfp4_moe_scale_to_mn_major_xe35

        rows_per_expert = [4, 8]
        groups = 4
        used = sum(padded_xe35(r) for r in rows_per_expert)
        scales_rm = torch.randint(
            100, 160, (sum(rows_per_expert), groups), dtype=torch.uint8
        )

        got = torch.full((used + 5, groups), 0xAA, dtype=torch.uint8, device="xpu")
        mxfp4_moe_scale_to_mn_major_xe35(
            got,
            scales_rm.to("xpu"),
            torch.tensor(rows_per_expert, dtype=torch.int32, device="xpu"),
        )
        got = got.cpu()
        assert torch.equal(got[:used], mn_major_reference_xe35(scales_rm, rows_per_expert))
        assert (got[used:] == 0xAA).all(), "slack rows were overwritten"

    def test_rejects_out_shorter_than_input_xe35(self):
        from sgl_kernel import mxfp4_moe_scale_to_mn_major_xe35

        rows_per_expert = torch.tensor([4, 4], dtype=torch.int32, device="xpu")
        scales = torch.zeros(8, 4, dtype=torch.uint8, device="xpu")
        out = torch.zeros(7, 4, dtype=torch.uint8, device="xpu")
        with pytest.raises(RuntimeError, match="at least as many rows"):
            mxfp4_moe_scale_to_mn_major_xe35(out, scales, rows_per_expert)

    def test_accepts_ragged_counts_with_padded_output_xe35(self):
        from sgl_kernel import mxfp4_moe_scale_to_mn_major_xe35

        counts = [3, 5]
        scales = torch.arange(32, dtype=torch.uint8).reshape(8, 4)
        out = torch.zeros(12, 4, dtype=torch.uint8, device="xpu")
        mxfp4_moe_scale_to_mn_major_xe35(
            out,
            scales.to("xpu"),
            torch.tensor(counts, dtype=torch.int32, device="xpu"),
        )
        assert torch.equal(out.cpu(), mn_major_reference_xe35(scales, counts))

    def test_rejects_noncontiguous_row_counts_xe35(self):
        from sgl_kernel import mxfp4_moe_scale_to_mn_major_xe35

        rows = torch.tensor([4, 0, 4, 0], dtype=torch.int32, device="xpu")[::2]
        scales = torch.zeros(8, 4, dtype=torch.uint8, device="xpu")
        with pytest.raises(RuntimeError, match="rows_per_expert must be contiguous"):
            mxfp4_moe_scale_to_mn_major_xe35(torch.zeros_like(scales), scales, rows)


# ---------------------------------------------------------------------------
# moe_grouped_mm_nt_mxfp4_w4a4_xe35
# ---------------------------------------------------------------------------


@requires_cri
class TestMoeGroupedMmNtXe35Mxfp4W4A4_xe35:
    @pytest.mark.parametrize("negative", [False, True])
    @pytest.mark.parametrize("lower_odd", [False, True])
    @torch.inference_mode()
    def test_source_bf16_midpoint_semantics_xe35(self, negative, lower_odd):
        rows_per_expert = [row_alignment_xe35(128)]
        activations = torch.zeros(rows_per_expert[0], 64, dtype=torch.uint8)
        weights = torch.zeros(1, 32, 64, dtype=torch.uint8)
        activations[:, 0] = 10 if negative else 2
        weights[:, :, 0] = 2
        activations[:, 16] = 9 if negative else 1
        weights[:, :, 16] = 1
        if lower_odd:
            activations[:, 32] = 9 if negative else 1
            weights[:, :, 32] = 1
        scales = torch.full((rows_per_expert[0], 4), 127, dtype=torch.uint8)
        scales[:, 1] = 121
        scales[:, 2] = 122
        operands = {
            "a": activations,
            "a_scale": mn_major_reference_xe35(scales, rows_per_expert),
            "b": weights,
            "b_scale": torch.full((1, 4, 32), 127, dtype=torch.uint8),
        }
        output = run_grouped_mm_xe35(operands, rows_per_expert, 32, 128)
        magnitude = 1.015625 if lower_odd else 1.0078125
        expected = torch.full(
            output.shape,
            -magnitude if negative else magnitude,
            dtype=torch.bfloat16,
        )
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        load_accuracy_reference_xe35().verify_grouped_output_xe35(
            output,
            operands["a"].to("xpu"),
            operands["a_scale"].to("xpu"),
            operands["b"].to("xpu"),
            operands["b_scale"].to("xpu"),
            rows_per_expert,
        )

    @pytest.mark.parametrize("transposed_b_scale", [False, True])
    @torch.inference_mode()
    def test_kernel_vs_reference_xe35(self, transposed_b_scale):
        n, k = 32, 128
        rows_per_expert = [row_alignment_xe35(k)] * 2
        operands = make_expert_operands_xe35(rows_per_expert, n, k, seed=1)
        got = run_grouped_mm_xe35(
            operands, rows_per_expert, n, k, transposed_b_scale=transposed_b_scale
        )
        assert_moe_close_xe35(got, operands["reference"])

    @torch.inference_mode()
    def test_ragged_rows_per_expert_xe35(self):
        """Per-expert row counts differ, so every expert's A, A-scale and D base
        pointer moves by a different amount.
        """
        n, k = 32, 128
        step = row_alignment_xe35(k)
        rows_per_expert = [step, 2 * step, step]
        operands = make_expert_operands_xe35(rows_per_expert, n, k, seed=2)
        assert_moe_close_xe35(
            run_grouped_mm_xe35(operands, rows_per_expert, n, k), operands["reference"]
        )

    @torch.inference_mode()
    def test_unrouted_experts_xe35(self):
        """Experts with no rows must be skipped. With 8 experts and 2 routed, the
        average token count also drops to 4, which selects the decode tile policy.
        """
        n, k = 32, 128
        step = row_alignment_xe35(k)
        rows_per_expert = [step, 0, 0, step, 0, 0, 0, 0]
        operands = make_expert_operands_xe35(rows_per_expert, n, k, seed=3)
        assert_moe_close_xe35(
            run_grouped_mm_xe35(operands, rows_per_expert, n, k), operands["reference"]
        )

    @torch.inference_mode()
    @pytest.mark.long_test
    @pytest.mark.parametrize(
        "n,allocated_rows",
        [(n, rows) for n in (4096, 8192) for rows in (256, 257, 384, 1279, 1280)]
        + [(4096, 2880), (4096, 2881), (8192, 3840), (8192, 3841)],
    )
    def test_decode_tile_boundary_xe35(self, n, allocated_rows):
        """The small-row tile and the mid-row fallback preserve used rows."""

        from sgl_kernel import moe_grouped_mm_nt_mxfp4_w4a4_xe35

        experts, k = 256, 4096
        torch.manual_seed(7)
        a_q, a_s = quantize_mxfp4_2d(torch.randn(4, k))
        b_q, b_s = quantize_mxfp4_2d(torch.randn(n, k))
        expected = (
            dequantize_mxfp4_2d(a_q, a_s, torch.float32)
            @ dequantize_mxfp4_2d(b_q, b_s, torch.float32).T
        )
        a = torch.empty(allocated_rows, k // 2, dtype=torch.uint8, device="xpu")
        a[:4].copy_(a_q.to("xpu"))
        a_scale = torch.empty(allocated_rows, k // 32, dtype=torch.uint8, device="xpu")
        a_scale[:4].view(-1).copy_(a_s.T.contiguous().view(-1).to("xpu"))
        b = torch.empty(experts, n, k // 2, dtype=torch.uint8, device="xpu")
        b[0].copy_(b_q.to("xpu"))
        b_scale = torch.empty(experts, k // 32, n, dtype=torch.uint8, device="xpu")
        b_scale[0].copy_(b_s.T.contiguous().to("xpu"))
        rows = torch.zeros(experts, dtype=torch.int32, device="xpu")
        rows[0] = 4
        out = torch.empty(allocated_rows, n, dtype=torch.bfloat16, device="xpu")
        moe_grouped_mm_nt_mxfp4_w4a4_xe35(
            out, a, a_scale, b, b_scale, rows, n, k, experts
        )
        torch.xpu.synchronize()
        assert_moe_close_xe35(out[:4], expected)

    @torch.inference_mode()
    def test_bias_xe35(self):
        """A non-null C turns on the epilogue's beta=1 path; the bias is per row,
        already expanded by the caller.
        """
        n, k = 32, 128
        rows_per_expert = [row_alignment_xe35(k)] * 2
        operands = make_expert_operands_xe35(rows_per_expert, n, k, seed=4)
        bias = torch.randn(operands["a"].shape[0], n, dtype=torch.float32) * 10.0
        got = run_grouped_mm_xe35(operands, rows_per_expert, n, k, bias=bias)
        assert_moe_close_xe35(got, operands["reference"] + bias)

    @pytest.mark.long_test
    @torch.inference_mode()
    def test_prefill_tile_dispatch_xe35(self):
        """avg_tokens_cnt > 32 selects the prefill tile family instead of the
        mid/decode ones.
        """
        n, k = 256, 256
        rows_per_expert = [max(64, row_alignment_xe35(k))] * 2
        operands = make_expert_operands_xe35(rows_per_expert, n, k, seed=5)
        assert_moe_close_xe35(
            run_grouped_mm_xe35(operands, rows_per_expert, n, k), operands["reference"]
        )

    @torch.inference_mode()
    def test_end_to_end_quant_relayout_gemm_xe35(self):
        """The way sglang drives these two ops: quantize the permuted activations,
        relayout their scales on device, then run the GEMM.
        """
        from sgl_kernel import (
            mxfp4_moe_scale_to_mn_major_xe35,
            sgl_per_token_group_quant_fp4,
        )

        device = "xpu"
        n, k = 32, 128
        rows_per_expert = [row_alignment_xe35(k)] * 2
        total_m = sum(rows_per_expert)
        torch.manual_seed(6)

        activations = torch.randn(total_m, k, dtype=torch.bfloat16, device=device)
        a_q, a_s = sgl_per_token_group_quant_fp4(
            x=activations, group_size=MXFP4_BLOCK_SIZE
        )

        a_scale = torch.zeros_like(a_s)
        rows = torch.tensor(rows_per_expert, dtype=torch.int32, device=device)
        mxfp4_moe_scale_to_mn_major_xe35(a_scale, a_s, rows)

        b_packed, b_scales, b_deq = [], [], []
        for _ in rows_per_expert:
            b_q, b_s = quantize_mxfp4_2d(torch.randn(n, k, dtype=torch.float32) * 2.0)
            b_packed.append(b_q)
            b_scales.append(b_s.transpose(0, 1).contiguous())
            b_deq.append(dequantize_mxfp4_2d(b_q, b_s, torch.float32))

        operands = {
            "a": a_q.cpu(),
            "a_scale": a_scale.cpu(),
            "b": torch.stack(b_packed).contiguous(),
            "b_scale": torch.stack(b_scales).contiguous(),
        }
        got = run_grouped_mm_xe35(operands, rows_per_expert, n, k)

        # Reference from the same quantized bytes the kernel was given.
        a_deq = dequantize_mxfp4_2d(a_q.cpu(), a_s.cpu(), torch.float32)
        want, row = [], 0
        for e, m_i in enumerate(rows_per_expert):
            want.append(a_deq[row : row + m_i] @ b_deq[e].transpose(0, 1))
            row += m_i
        assert_moe_close_xe35(got, torch.cat(want, dim=0))

    def test_argument_validation_xe35(self):
        from sgl_kernel import moe_grouped_mm_nt_mxfp4_w4a4_xe35

        device = "xpu"
        n, k, e = 32, 128, 2
        rows_per_expert = [4, 4]
        total_m = sum(rows_per_expert)
        g = k // MXFP4_BLOCK_SIZE

        def operand(shape, dtype=torch.uint8):
            return torch.zeros(shape, dtype=dtype, device=device)

        rows = torch.tensor(rows_per_expert, dtype=torch.int32, device=device)
        a = operand((total_m, k // 2))
        a_scale = operand((total_m, g))
        b = operand((e, n, k // 2))
        b_scale = operand((e, g, n)).transpose(1, 2)
        out = operand((total_m, n), torch.bfloat16)

        def call(**overrides):
            kwargs = dict(
                output=out,
                a=a,
                a_scale=a_scale,
                b=b,
                b_scale=b_scale,
                rows_per_expert=rows,
                n=n,
                k=k,
                num_experts=e,
            )
            kwargs.update(overrides)
            moe_grouped_mm_nt_mxfp4_w4a4_xe35(**kwargs)

        with pytest.raises(RuntimeError, match="multiple of 32"):
            call(n=n + 16, b=operand((e, n + 16, k // 2)))
        with pytest.raises(RuntimeError, match="multiple of 64"):
            call(k=k + 32)
        with pytest.raises(RuntimeError, match="bfloat16"):
            call(output=operand((total_m, n), torch.float32))
        with pytest.raises(RuntimeError, match="dense along N"):
            call(b_scale=operand((e, n, g)))  # K-dense: neither accepted form
        with pytest.raises(RuntimeError, match="at least as many rows"):
            call(a_scale=operand((total_m - 4, g)))
        with pytest.raises(RuntimeError, match="rows_per_expert must be contiguous"):
            call(
                rows_per_expert=torch.tensor(
                    [4, 0, 4, 0], dtype=torch.int32, device=device
                )[::2]
            )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
