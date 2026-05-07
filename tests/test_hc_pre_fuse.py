import pytest
import torch
import utils
from sgl_kernel import hc_pre_fuse

device = utils.get_device()


def _hc_pre_fuse_torch(
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    residual: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
):
    """
    Pure-torch reference implementation of hc_pre_fuse based on TileLang code.
    """
    n_splits, T, hc_mult3 = gemm_out_mul.shape
    _, hidden_size = residual.shape[0], residual.shape[2]
    hc = hc_mult

    # Phase 1: RMS normalization with n_splits accumulation
    rms = gemm_out_sqrsum.sum(dim=0)  # [T]
    rms = torch.rsqrt(rms / (hc * hidden_size) + rms_eps)  # [T]

    mixes = gemm_out_mul.sum(dim=0)  # [T, 24]
    mixes = mixes * rms.unsqueeze(-1)  # [T, 24]

    # Phase 2a: post_mix computation
    post_logits = mixes[:, hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc]
    post_mix = torch.sigmoid(post_logits) * hc_post_mult_value  # [T, 4]

    # Phase 2a: Sinkhorn on comb matrix
    comb = (mixes[:, 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).reshape(T, hc, hc)

    # Sinkhorn iterations
    comb = torch.softmax(comb, dim=-1) + hc_sinkhorn_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)

    comb_mix = comb.reshape(T, hc * hc)  # [T, 16]

    # Phase 2b: pre_mix computation (internal only)
    pre_logits = mixes[:, :hc] * hc_scale[0] + hc_base[:hc]
    pre_mix = torch.sigmoid(pre_logits) + hc_pre_eps  # [T, 4]

    # Weighted sum: layer_input = sum_k(pre_mix[k] * residual[:, k, :])
    # residual: [T, 4, D], pre_mix: [T, 4]
    # layer_input[t, h] = sum_k(pre_mix[t, k] * residual[t, k, h])
    layer_input = torch.einsum("tk,tkh->th", pre_mix, residual.float())  # [T, D]

    return post_mix, comb_mix, layer_input


def _make_inputs(T, hidden_size, n_splits, device, seed=42):
    """Generate test inputs for hc_pre_fuse with hc=4."""
    hc = 4
    hc_mult3 = (2 + hc) * hc  # 24

    torch.manual_seed(seed)

    # GEMM outputs
    gemm_out_mul = torch.randn(
        n_splits, T, hc_mult3, dtype=torch.float32, device=device
    )
    gemm_out_sqrsum = (
        torch.rand(n_splits, T, dtype=torch.float32, device=device) * 100 + 10
    )

    # Hyperconnection parameters
    hc_scale = torch.rand(3, dtype=torch.float32, device=device) * 0.5 + 0.5
    hc_base = torch.randn(hc_mult3, dtype=torch.float32, device=device) * 0.1

    # Residual from previous layer
    residual = torch.randn(T, hc, hidden_size, dtype=torch.bfloat16, device=device)

    # Output tensors (allocated by caller)
    post_mix = torch.empty(T, hc, dtype=torch.float32, device=device)
    comb_mix = torch.empty(T, hc * hc, dtype=torch.float32, device=device)
    layer_input = torch.empty(T, hidden_size, dtype=torch.bfloat16, device=device)

    return (
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual,
        post_mix,
        comb_mix,
        layer_input,
    )


@pytest.mark.parametrize("T", [1, 16, 128])
@pytest.mark.parametrize(
    "hidden_size", [7168, 512]
)  # 7168 is model default, 512 for faster testing
@pytest.mark.parametrize("n_splits", [1, 4])
def test_hc_pre_fuse(T, hidden_size, n_splits):
    hc = 4
    sinkhorn_iters = 20
    rms_eps = 1e-5
    hc_pre_eps = 1e-6
    hc_sinkhorn_eps = 1e-6
    hc_post_mult_value = 2.0

    (
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual,
        post_mix,
        comb_mix,
        layer_input,
    ) = _make_inputs(T, hidden_size, n_splits, device=f"{device}:0")

    # Reference implementation
    post_mix_ref, comb_mix_ref, layer_input_ref = _hc_pre_fuse_torch(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual,
        hc,
        sinkhorn_iters,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
    )

    # XPU kernel
    hc_pre_fuse(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual,
        post_mix,
        comb_mix,
        layer_input,
        hc,
        sinkhorn_iters,
        n_splits,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
    )

    # Validate outputs
    atol_fp32 = 1e-3
    atol_bf16 = 1e-2

    assert torch.allclose(
        post_mix, post_mix_ref, atol=atol_fp32, rtol=1e-3
    ), f"post_mix mismatch: max={(post_mix - post_mix_ref).abs().max():.2e}"

    assert torch.allclose(
        comb_mix, comb_mix_ref, atol=atol_fp32, rtol=1e-3
    ), f"comb_mix mismatch: max={(comb_mix - comb_mix_ref).abs().max():.2e}"

    assert torch.allclose(
        layer_input.float(), layer_input_ref, atol=atol_bf16, rtol=1e-2
    ), f"layer_input mismatch: max={(layer_input.float() - layer_input_ref).abs().max():.2e}"

    print(f"✓ T={T}, hidden_size={hidden_size}, n_splits={n_splits} passed")


if __name__ == "__main__":
    # Quick smoke test
    test_hc_pre_fuse(T=16, hidden_size=512, n_splits=1)
    print("All tests passed!")
