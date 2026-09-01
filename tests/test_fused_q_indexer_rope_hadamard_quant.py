import math

import pytest
import torch
from sgl_kernel import fused_q_indexer_rope_hadamard_quant
from utils import FP8_E4M3_MAX, HEAD_DIM, NOPE_DIM, ROPE_DIM

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available(), reason="XPU not available"
)


def _fwht128(x: torch.Tensor) -> torch.Tensor:
    """Unnormalised 128-point Walsh-Hadamard Transform (last dimension)."""
    N = x.shape[-1]
    leading = x.shape[:-1]
    h = 1
    while h < N:
        x = x.reshape(*leading, N // (2 * h), 2, h)
        u, v = x[..., 0, :], x[..., 1, :]
        x = torch.stack([u + v, u - v], dim=-2).reshape(*leading, N)
        h *= 2
    return x


def _ref_torch_impl(
    q_input: torch.Tensor,
    q_fp8: torch.Tensor,
    weight: torch.Tensor,
    weights_out: torch.Tensor,
    weight_scale: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    B, H, head_dim = q_input.shape
    assert head_dim == HEAD_DIM
    assert freqs_cis.shape[-1] == ROPE_DIM
    assert q_fp8.shape == (B, H, HEAD_DIM)
    assert weight.shape == (B, H)
    assert weights_out.shape == (B, H, 1)

    x = q_input.float()

    freq = freqs_cis[positions].unsqueeze(1)  # (B, 1, 64)
    freq_re, freq_im = freq[..., 0::2], freq[..., 1::2]  # (B, 1, 32) each
    x_nope = x[..., :NOPE_DIM]
    x_rope = x[..., NOPE_DIM:]
    xr, xi = x_rope[..., 0::2], x_rope[..., 1::2]
    rotated = torch.stack(
        [xr * freq_re - xi * freq_im, xr * freq_im + xi * freq_re], dim=-1
    ).flatten(-2)
    x = torch.cat([x_nope, rotated], dim=-1)

    x = _fwht128(x) * (1.0 / math.sqrt(HEAD_DIM))

    abs_max = x.abs().amax(dim=-1, keepdim=True)  # (B, H, 1)
    scale = abs_max.clamp(min=1e-4) / FP8_E4M3_MAX
    q_fp8.copy_((x / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn))

    weights_out.copy_(
        (weight.float() * float(weight_scale) * scale.squeeze(-1)).unsqueeze(-1)
    )


@pytest.mark.parametrize("batch_size", [1, 8])
def test_fused_q_indexer_rope_hadamard_quant(batch_size):
    torch.manual_seed(42)
    num_heads = 4
    head_dim = 128
    rope_dim = 64
    max_pos = 256
    device = torch.device("xpu")

    q_input = torch.randn(batch_size, num_heads, head_dim, dtype=torch.bfloat16)
    q_input_xpu = q_input.to(device=device)
    weight = torch.randn(batch_size, num_heads, dtype=torch.bfloat16)
    weight_xpu = weight.to(device=device)
    freqs_cis = torch.randn(max_pos, rope_dim // 2, dtype=torch.complex64)
    freqs_cis_xpu = freqs_cis.to(device=device)
    positions = torch.randint(0, max_pos, (batch_size,), dtype=torch.int64)
    positions_xpu = positions.to(device=device)
    weight_scale = 0.5

    ref_q_fp8 = torch.empty(
        q_input.shape, dtype=torch.float8_e4m3fn, device=q_input.device
    )
    q_fp8 = torch.empty_like(ref_q_fp8).to(device=device)
    ref_weights_out = torch.empty(
        (*q_input.shape[:-1], 1), dtype=torch.float32, device=q_input.device
    )
    weights_out = torch.empty_like(ref_weights_out).to(device=device)

    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
    _ref_torch_impl(
        q_input,
        ref_q_fp8,
        weight,
        ref_weights_out,
        float(weight_scale),
        freqs_real,
        positions,
    )

    freqs_real_xpu = torch.view_as_real(freqs_cis_xpu).flatten(-2)
    fused_q_indexer_rope_hadamard_quant(
        q_input_xpu,
        q_fp8,
        weight_xpu,
        weights_out,
        float(weight_scale),
        freqs_real_xpu,
        positions_xpu,
    )

    assert torch.isfinite(weights_out).all(), "weights_out contains non-finite values"
    assert torch.isfinite(
        ref_weights_out
    ).all(), "ref_weights_out contains non-finite values"
    torch.testing.assert_close(q_fp8.cpu(), ref_q_fp8)
    torch.testing.assert_close(weights_out.cpu(), ref_weights_out)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
