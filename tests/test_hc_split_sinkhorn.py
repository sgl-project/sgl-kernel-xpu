import pytest
import torch

from sgl_kernel import hc_split_sinkhorn

def _hc_split_sinkhorn_torch(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
):
    """Pure-torch implementation of hc_split_sinkhorn (tilelang-free fallback)."""
    b, s, _ = mixes.shape
    hc = hc_mult
    flat = mixes.reshape(b * s, (2 + hc) * hc)

    pre = torch.sigmoid(flat[:, :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2.0 * torch.sigmoid(
        flat[:, hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc]
    )
    comb = (
        flat[:, 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]
    ).reshape(b * s, hc, hc)

    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    return (
        pre.view(b, s, hc),
        post.view(b, s, hc),
        comb.view(b, s, hc, hc),
    )

def _make_inputs(b, s, device="cpu", seed=42):
    """Return (mixes, hc_scale, hc_base) for hc=4 on the given device."""
    hc = 4
    col_size = (2 + hc) * hc  # 24
    torch.manual_seed(seed)
    mixes = torch.randn(b, s, col_size, dtype=torch.float32, device=device)
    hc_scale = torch.rand(3, dtype=torch.float32, device=device) * 0.5 + 0.5
    hc_base = torch.randn(col_size, dtype=torch.float32, device=device) * 0.1
    return mixes, hc_scale, hc_base

@pytest.mark.parametrize("b", [7, 384, 512])
@pytest.mark.parametrize("s", [1])
@pytest.mark.parametrize("sinkhorn_iters", [20])
def test_hc_split_sinkhorn(b, s, sinkhorn_iters):
    hc = 4
    eps = 1e-6
    mixes_cpu, hc_scale_cpu, hc_base_cpu = _make_inputs(b, s)

    pre_ref, post_ref, comb_ref = _hc_split_sinkhorn_torch(
        mixes_cpu, hc_scale_cpu, hc_base_cpu, hc, sinkhorn_iters, eps
    )
    pre_xpu, post_xpu, comb_xpu = hc_split_sinkhorn(
        mixes_cpu.to("xpu"),
        hc_scale_cpu.to("xpu"),
        hc_base_cpu.to("xpu"),
        hc, sinkhorn_iters, eps,
    )

    atol = 1e-4
    assert torch.allclose(pre_xpu.cpu(), pre_ref, atol=atol), \
        f"pre mismatch: max={(pre_xpu.cpu() - pre_ref).abs().max():.2e}"
    assert torch.allclose(post_xpu.cpu(), post_ref, atol=atol), \
        f"post mismatch: max={(post_xpu.cpu() - post_ref).abs().max():.2e}"
    assert torch.allclose(comb_xpu.cpu(), comb_ref, atol=atol), \
        f"comb mismatch: max={(comb_xpu.cpu() - comb_ref).abs().max():.2e}"