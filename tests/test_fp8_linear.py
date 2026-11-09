from unittest import mock

import pytest
import sgl_kernel.gemm as gemm
import torch
from sgl_kernel import fp8_linear, fp8_linear_supported

FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
BLOCK = 128

# Per-rank (N, K) of the dense projections of Qwen3.6-35B-A3B: full-attention
# qkv / o_proj, linear-attention in_proj_qkvz / out_proj, shared-expert
# gate_up / down.
SHAPES_TP2 = [(4608, 2048), (2048, 2048), (6144, 2048), (512, 2048), (2048, 256)]
SHAPES_TP4 = [(2560, 2048), (2048, 1024), (3072, 2048), (256, 2048), (2048, 128)]
SHAPES = SHAPES_TP2 + SHAPES_TP4
# Decode (split-K) tiles up to M = 128, 2D-tiled prefill tiles above.
M_VALUES = [1, 3, 16, 64, 128, 1000]


def _quant_per_tensor(w: torch.Tensor):
    scale = (w.abs().max().float() / FP8_E4M3_MAX).clamp(min=1e-12)
    w_fp8 = (w.float() / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    w_fp8 = w_fp8.to(torch.float8_e4m3fn)
    return w_fp8, scale.reshape(1), w_fp8.float() * scale


def _quant_block(w: torch.Tensor):
    n, k = w.shape
    n_blocks = (n + BLOCK - 1) // BLOCK
    padded = torch.zeros(n_blocks * BLOCK, k, dtype=torch.float32, device=w.device)
    padded[:n] = w.float()
    blocks = padded.view(n_blocks, BLOCK, k // BLOCK, BLOCK)
    scale = (blocks.abs().amax(dim=(1, 3)) / FP8_E4M3_MAX).clamp(min=1e-12)
    expanded = scale[:, None, :, None]
    w_fp8 = (blocks / expanded).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    w_fp8 = w_fp8.to(torch.float8_e4m3fn)
    w_dq = (w_fp8.float() * expanded).view(n_blocks * BLOCK, k)[:n]
    w_fp8 = w_fp8.view(n_blocks * BLOCK, k)[:n].contiguous()
    return w_fp8, scale.contiguous(), w_dq


def _quant_per_channel(w: torch.Tensor):
    scale = (w.abs().amax(dim=1).float() / FP8_E4M3_MAX).clamp(min=1e-12)
    w_fp8 = (w.float() / scale[:, None]).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    w_fp8 = w_fp8.to(torch.float8_e4m3fn)
    return w_fp8, scale, w_fp8.float() * scale[:, None]


QUANT = {
    "per_tensor": _quant_per_tensor,
    "per_channel": _quant_per_channel,
    "block": _quant_block,
}


def _check(out: torch.Tensor, ref: torch.Tensor):
    # Bounds relative to the output magnitude, as in the grouped W8A16 block-scale
    # test: the bf16 output alone rounds by up to 2^-8 of |ref|.
    assert out.dtype == torch.bfloat16
    error = (out.float() - ref).abs()
    assert error.mean() <= ref.abs().mean() * 1e-2
    assert error.max() <= ref.abs().max() * 2e-2


def _problem(m, n, k, scale_kind, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="xpu")
    w = torch.randn(n, k, dtype=torch.bfloat16, device="xpu") * 0.05
    w_fp8, scale, w_dq = QUANT[scale_kind](w)
    return x, w_fp8, scale, w_dq


@pytest.mark.arch("xe20")
@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("m", M_VALUES)
@pytest.mark.parametrize("scale_kind", ["per_tensor", "per_channel", "block"])
def test_fp8_linear(m, n, k, scale_kind):
    if not fp8_linear_supported(n, k, block_scale=scale_kind == "block"):
        pytest.skip("128x128 block scales need K % 128 == 0")
    x, w_fp8, scale, w_dq = _problem(m, n, k, scale_kind)
    _check(fp8_linear(x, w_fp8, scale), x.float() @ w_dq.t())


@pytest.mark.arch("xe20")
@pytest.mark.parametrize("scale_kind", ["per_tensor", "per_channel", "block"])
def test_fp8_linear_bias_and_out(scale_kind):
    m, n, k = 7, 4608, 2048
    x, w_fp8, scale, w_dq = _problem(m, n, k, scale_kind)
    bias = torch.randn(n, dtype=torch.bfloat16, device="xpu")
    out = torch.empty(m, n, dtype=torch.bfloat16, device="xpu")
    ret = fp8_linear(x, w_fp8, scale, bias=bias, out=out)
    assert ret.data_ptr() == out.data_ptr()
    _check(out, x.float() @ w_dq.t() + bias.float())


@pytest.mark.arch("xe20")
def test_fp8_linear_unaligned_views():
    # Views at a 2-byte offset: fp8_linear copies them to aligned buffers.
    m, n, k = 5, 2048, 1024
    x, w_fp8, scale, w_dq = _problem(m, n, k, "per_channel")
    x_view = torch.empty(m * k + 1, dtype=torch.bfloat16, device="xpu")[1:].view(m, k)
    x_view.copy_(x)
    out = torch.empty(m * n + 1, dtype=torch.bfloat16, device="xpu")[1:].view(m, n)
    ret = fp8_linear(x_view, w_fp8, scale, out=out)
    assert ret.data_ptr() == out.data_ptr()
    _check(out, x.float() @ w_dq.t())


@pytest.mark.arch("xe20")
def test_fp8_linear_empty_input():
    _, w_fp8, scale, _ = _problem(1, 256, 256, "per_tensor")
    x = torch.empty(0, 256, dtype=torch.bfloat16, device="xpu")
    assert fp8_linear(x, w_fp8, scale).shape == (0, 256)


@pytest.mark.parametrize(
    "xe2,n,k,block_scale,expected",
    [
        (True, None, None, False, True),  # device only
        (True, 4608, 2048, True, True),
        (True, 4608, 2048, False, True),
        (True, 4608, 2112, False, True),  # K % 64 for per-tensor / per-channel
        (True, 4608, 2112, True, False),  # block scales need K % 128
        (True, 4608, 2080, False, False),
        (True, 4600, 2048, False, False),  # N % 64
        # Any other device (CRI included) keeps its existing FP8 path.
        (False, None, None, False, False),
        (False, 4608, 2048, True, False),
        (False, 4608, 2048, False, False),
    ],
)
def test_fp8_linear_supported(xe2, n, k, block_scale, expected):
    with mock.patch.object(gemm, "is_xe2_arch", return_value=xe2):
        assert fp8_linear_supported(n, k, block_scale) is expected


def test_fp8_linear_rejects_other_devices():
    x = torch.empty(4, 256, dtype=torch.bfloat16)
    w_fp8 = torch.empty(256, 256, dtype=torch.float8_e4m3fn)
    with mock.patch.object(gemm, "is_xe2_arch", return_value=False):
        with pytest.raises(AssertionError, match="BMG \\(Xe2\\)"):
            fp8_linear(x, w_fp8, torch.ones(1))


if __name__ == "__main__":
    pytest.main([__file__])
