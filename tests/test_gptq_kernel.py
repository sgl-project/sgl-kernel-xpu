import pytest
import torch

if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
    pytest.skip("XPU not available", allow_module_level=True)

from sgl_kernel import gptq_gemm


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


@pytest.mark.parametrize("m", [1, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("zero_format", [1, 2])
@pytest.mark.parametrize("index_name", ["contiguous", "noncontiguous"])
def test_gptq_gemm(m, dtype, zero_format, index_name):
    k, n = 256, 32
    groups = k // 128
    contiguous = torch.arange(k) // 128
    g_idx = contiguous if index_name == "contiguous" else (groups - 1) - contiguous
    weight = torch.randn(k, n)
    act = torch.randn(m, k)
    q, stored, scales = quantize(weight, g_idx, zero_format)
    act_q = act.to(dtype)
    scales_q = scales.to(dtype)
    ref = reference(act_q.float(), q, stored, scales_q.float(), g_idx, zero_format).to(dtype)
    device = torch.device("xpu")
    out = gptq_gemm(
        act_q.to(device),
        pack_rows(q.to(torch.uint8)).to(device),
        pack_cols(stored.to(torch.uint8)).to(device),
        scales_q.to(device),
        g_idx.to(device=device, dtype=torch.int32),
        False,
        4,
        zero_format,
    )
    torch.xpu.synchronize()
    diff = (out.float().cpu() - ref.float()).abs()
    limit = torch.maximum(0.01 * ref.float().abs(), torch.full_like(ref.float(), 0.02))
    assert torch.all(diff <= limit)
