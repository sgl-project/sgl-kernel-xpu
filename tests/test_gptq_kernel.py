import pytest
import torch

if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
    pytest.skip("XPU not available", allow_module_level=True)

from sgl_kernel import gptq_gemm, gptq_repack_weights

DEVICE = torch.device("xpu")


def pack_rows(q_uint8):
    k, n = q_uint8.shape
    out = torch.zeros(k // 8, n, dtype=torch.int32)
    for i in range(8):
        out |= q_uint8[i::8].to(torch.int32) << (i * 4)
    return out


def pack_cols(z_uint8):
    groups, n = z_uint8.shape
    out = torch.zeros(groups, n // 8, dtype=torch.int32)
    for i in range(8):
        out |= z_uint8[:, i::8].to(torch.int32) << (i * 4)
    return out


def quantize(weight, g_idx, zero_format):
    groups = int(g_idx.max().item()) + 1
    n = weight.shape[1]
    scales = torch.empty(groups, n)
    stored = torch.empty(groups, n, dtype=torch.int32)
    q = torch.empty_like(weight)
    for gi in range(groups):
        rows = weight[g_idx == gi]
        wmin = rows.min(dim=0).values
        wmax = rows.max(dim=0).values
        scale = ((wmax - wmin) / 15.0).clamp(min=1e-4)
        eff = (-wmin / scale).round()
        if zero_format == 1:
            eff = eff.clamp(1, 15)
            stored[gi] = (eff - 1).to(torch.int32)
        else:
            eff = eff.clamp(0, 15)
            stored[gi] = eff.to(torch.int32)
        scales[gi] = scale
        sel = g_idx == gi
        q[sel] = (weight[sel] / scale + eff).round().clamp(0, 15)
    return q, stored, scales


def reference(act, q, stored, scales, g_idx, zero_format):
    eff = stored[g_idx].to(torch.float32)
    if zero_format == 1:
        eff = eff + 1
    return act @ ((q - eff) * scales[g_idx])


def make_g_idx(kind, k):
    groups = k // 128
    contiguous = torch.arange(k) // 128
    if kind == "contiguous":
        return contiguous
    if kind == "reversed":
        return (groups - 1) - contiguous
    if kind == "interleaved":
        return torch.arange(k) % groups
    raise ValueError(kind)


def run_case(m, k, n, dtype, zero_format, g_idx_kind, packed):
    g_idx = make_g_idx(g_idx_kind, k)
    weight = torch.randn(k, n)
    act = torch.randn(m, k)
    q, stored, scales = quantize(weight, g_idx, zero_format)
    act_q = act.to(dtype)
    scales_q = scales.to(dtype)
    ref = reference(act_q.float(), q, stored, scales_q.float(), g_idx, zero_format).to(dtype)
    qweight = pack_rows(q.to(torch.uint8)).to(DEVICE)
    qweight_before = qweight.clone()
    if packed:
        packed_qweight = gptq_repack_weights(
            qweight, g_idx.to(device=DEVICE, dtype=torch.int32)
        )
        # The repack must never modify the caller's tensor.
        assert torch.equal(qweight, qweight_before)
        qweight = packed_qweight
    out = gptq_gemm(
        act_q.to(DEVICE),
        qweight,
        pack_cols(stored.to(torch.uint8)).to(DEVICE),
        scales_q.to(DEVICE),
        g_idx.to(device=DEVICE, dtype=torch.int32),
        False,
        4,
        zero_format,
        packed,
    )
    torch.xpu.synchronize()
    if not packed:
        # The unpacked path must never modify the caller's weights either.
        assert torch.equal(qweight, qweight_before)
    diff = (out.float().cpu() - ref.float()).abs()
    if packed and dtype == torch.bfloat16 and m >= 8:
        # The bf16 XMX path (DPAS) accumulates at bf16-grade precision:
        # measured on a B70, its outputs deviate from the fp32 reference by
        # up to ~0.3% of the output full scale, roughly independent of K,
        # while single-row probes show the staged weights are exact. Assert
        # that full-scale bound instead of the elementwise one.
        assert diff.max() <= 0.01 * ref.float().abs().max()
        return
    limit = torch.maximum(0.01 * ref.float().abs(), torch.full_like(ref.float(), 0.02))
    assert torch.all(diff <= limit)


@pytest.mark.parametrize("m", [1, 2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("zero_format", [1, 2])
@pytest.mark.parametrize("g_idx_kind", ["contiguous", "reversed", "interleaved"])
def test_gptq_gemm_small(m, dtype, zero_format, g_idx_kind):
    run_case(m, 256, 32, dtype, zero_format, g_idx_kind, packed=False)


# The larger GEMM shapes this file carried before the packed work: they
# exercise the unrolled loops and, for bf16, the aligned XMX path.
@pytest.mark.parametrize(
    "m,k,n",
    [
        (1, 2048, 2048),
        (8, 2048, 4096),
        (128, 4096, 2048),
        (128, 2048, 2048),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gptq_gemm_large(m, k, n, dtype):
    run_case(m, k, n, dtype, 1, "contiguous", packed=False)


@pytest.mark.parametrize("m", [1, 2, 4, 8, 128])
@pytest.mark.parametrize("zero_format", [1, 2])
def test_gptq_packed(m, zero_format):
    run_case(m, 4096, 1024, torch.bfloat16, zero_format, "contiguous", packed=True)


@pytest.mark.parametrize("k,n", [(4096, 1024), (1024, 4096)])
def test_gptq_packed_m4_column_layouts(k, n):
    # M=4 dispatches to the pair kernel when K >= N and the quad kernel
    # when K < N; cover both.
    run_case(4, k, n, torch.float16, 1, "contiguous", packed=True)


# Regression for the tail bug the review caught: with K=1664 the packed
# kernels have 208 words, a count where the old tail loop restarted at the
# block boundary and processed the main loop's words a second time.
@pytest.mark.parametrize("m", [1, 2, 4])
@pytest.mark.parametrize("zero_format", [1, 2])
def test_gptq_packed_tail_regression(m, zero_format):
    run_case(m, 1664, 512, torch.float16, zero_format, "contiguous", packed=True)


def test_gptq_repack_rejects_permuted_g_idx():
    k, n = 256, 32
    g_idx = make_g_idx("reversed", k)
    qweight = torch.zeros(k // 8, n, dtype=torch.int32, device=DEVICE)
    with pytest.raises(RuntimeError):
        gptq_repack_weights(qweight, g_idx.to(device=DEVICE, dtype=torch.int32))
