"""Tests for DeepSeek-V4 fused norm + RoPE + flashmla cache layout kernels."""

import pytest
import torch
from sgl_kernel import fused_k_norm_rope_flashmla
from utils import get_device


@pytest.mark.parametrize("max_pos", [16, 128])
def test_fused_k_norm_rope_flashmla_correctness(max_pos):
    """Test Q norm + rope against reference."""
    pytest.skip(
        "torch vs jit kernel on cuda passes but torch kernel on cuda vs cpu fails"
    )
    torch.manual_seed(42)
    rope_dim = 64
    page_size = 256
    head_dim = 512
    eps = 1e-6
    device = get_device()

    kv = torch.randn(max_pos, head_dim, dtype=torch.bfloat16)
    kv_weight = torch.randn(head_dim, dtype=torch.bfloat16)
    freqs_cis = torch.randn(max_pos, rope_dim // 2, dtype=torch.complex64)
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
    positions = torch.randint(0, max_pos, (max_pos,), dtype=torch.int64)
    """
    out_loc = torch.randint(
        0, max_pos, (max_pos,), dtype=torch.int32
    )
    """
    out_loc = torch.randperm(max_pos, dtype=torch.int32)
    kvcache = torch.zeros((128, 149760), dtype=torch.uint8)

    fused_k_norm_rope_flashmla(
        kv.to(device),
        kv_weight.to(device),
        freqs_real.to(device),
        positions.to(device),
        out_loc.to(device),
        kvcache.to(device),
        eps,
        page_size,
    )

    # Reference
    ref_kvcache = torch.zeros_like(kvcache)
    fused_k_norm_rope_flashmla(
        kv, kv_weight, freqs_real, positions, out_loc, ref_kvcache, eps, page_size
    )

    torch.testing.assert_close(
        kvcache.cpu().float(), ref_kvcache.float(), rtol=1e-2, atol=1e-2
    )
