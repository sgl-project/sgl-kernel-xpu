"""End-to-end test for the mhc_pre pipeline (GEMM+sqrsum -> hc_pre_big_fuse).

mhc_pre replaces the two non-prenorm TileLang paths with the CUTLASS gemm_sqrsum
kernel (writing K-split partials) followed by hc_pre_big_fuse (which reduces the
split axis, then does RMS / Sinkhorn / mix). This exercises BOTH kernels wired
together, against a pure-torch reference of the whole pipeline.

The fuse reference is imported from test_hc_pre_fuse to avoid re-deriving the
Sinkhorn math; we only add the GEMM+sqrsum reference here.
"""

import pytest
import torch
import utils
from sgl_kernel import mhc_pre
from test_hc_pre_fuse import _hc_pre_big_fuse_torch

device = utils.get_device()

# Production shapes (from the real mhc_pre call log): D=4096, hc_mult=4,
# hc_hidden=16384, fn=[24,16384], norm_weight present. Token counts span the
# ragged set; all are <= 2048, so all take the split-k path (n_splits_pre=32).
PROD_D = 4096
PROD_T = [16, 48, 128, 512, 896, 1021, 1024, 1034, 1038, 1518, 2048]

HC_MULT = 4
HC_MULT3 = (2 + HC_MULT) * HC_MULT  # 24
SINKHORN_REPEAT = 20
RMS_EPS = 1e-6
HC_PRE_EPS = 1e-6
HC_SINKHORN_EPS = 1e-6
HC_POST_MULT_VALUE = 2.0
NORM_EPS = 1e-6


def _make_inputs(T, D, device, seed=42):
    torch.manual_seed(seed)
    hc_hidden = HC_MULT * D
    residual = torch.randn(T, HC_MULT, D, dtype=torch.bfloat16, device=device)
    fn = torch.randn(HC_MULT3, hc_hidden, dtype=torch.float32, device=device)
    hc_scale = torch.rand(3, dtype=torch.float32, device=device) * 0.5 + 0.5
    hc_base = torch.randn(HC_MULT3, dtype=torch.float32, device=device) * 0.1
    # RMSNorm weights in practice cluster near 1 (they scale a normalized signal),
    # not centered at 0 with unit variance. A randn*0.5+1 weight is both realistic
    # and avoids adversarially amplifying the tf32 GEMM error in layer_input.
    norm_weight = (
        torch.randn(D, dtype=torch.float32, device=device) * 0.5 + 1.0
    ).to(torch.bfloat16)
    return residual, fn, hc_scale, hc_base, norm_weight


def _mhc_pre_torch(residual, fn, hc_scale, hc_base, norm_weight):
    """Pure-torch reference: GEMM+sqrsum (single split) -> fuse reference.

    norm_weight is the bf16 tensor or None (plain fuse / no RMSNorm of layer_input).
    """
    T = residual.size(0)
    hc_hidden = HC_MULT * residual.size(2)
    A = residual.reshape(T, hc_hidden).float()
    B = fn.float()  # [24, hc_hidden]

    # gemm_out_mul = A @ Bᵀ, gemm_out_sqrsum = row sum of squares. As a single
    # "split" slab ([1, T, *]) — the fuse reduces a 1-deep axis trivially, and a
    # sum over n_splits real slabs equals the un-split slab (reduction is sum).
    gemm_out_mul = (A @ B.t()).unsqueeze(0)  # [1, T, 24]
    gemm_out_sqrsum = (A * A).sum(dim=1).unsqueeze(0)  # [1, T]

    return _hc_pre_big_fuse_torch(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual,
        HC_MULT,
        SINKHORN_REPEAT,
        RMS_EPS,
        HC_PRE_EPS,
        HC_SINKHORN_EPS,
        HC_POST_MULT_VALUE,
        norm_weight=norm_weight.float() if norm_weight is not None else None,
        norm_eps=NORM_EPS,
    )


@pytest.mark.parametrize("T", PROD_T)
@pytest.mark.parametrize("with_norm", [True, False])
def test_mhc_pre(T, with_norm):
    """Full mhc_pre pipeline vs pure-torch reference, production shapes."""
    residual, fn, hc_scale, hc_base, norm_weight = _make_inputs(
        T, PROD_D, device=f"{device}:0"
    )
    nw = norm_weight if with_norm else None

    post_mix_ref, comb_mix_ref, layer_input_ref = _mhc_pre_torch(
        residual, fn, hc_scale, hc_base, nw
    )

    post_mix, comb_mix, layer_input = mhc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps=RMS_EPS,
        hc_pre_eps=HC_PRE_EPS,
        hc_sinkhorn_eps=HC_SINKHORN_EPS,
        hc_post_mult_value=HC_POST_MULT_VALUE,
        sinkhorn_repeat=SINKHORN_REPEAT,
        norm_weight=nw,
        norm_eps=NORM_EPS,
    )

    # post_mix / comb_mix depend only on the GEMM mixes (tf32) through RMS+Sinkhorn,
    # which is well-conditioned (sigmoid / softmax bounded) -> keep these strict so
    # real bugs surface.
    atol_mix = 5e-2
    # layer_input is the bf16-quantized weighted sum (optionally RMS-normed). Its
    # error budget is larger than the pure-fuse test (test_hc_pre_fuse uses 1e-2)
    # because: (a) bf16 output quantization is ~0.016 at magnitude ~2, and (b) the
    # tf32 GEMM error propagates through pre_mix into the sum. This is a max-abs
    # over T*D (~8.4M) elements, so the worst single element sets the bound.
    atol_bf16 = 1.5e-1

    assert torch.allclose(
        post_mix, post_mix_ref, atol=atol_mix, rtol=2e-2
    ), f"post_mix mismatch (T={T}): max={(post_mix - post_mix_ref).abs().max():.3e}"

    assert torch.allclose(
        comb_mix.reshape(T, HC_MULT, HC_MULT),
        comb_mix_ref.reshape(T, HC_MULT, HC_MULT),
        atol=atol_mix,
        rtol=2e-2,
    ), f"comb_mix mismatch (T={T}): max={(comb_mix.reshape(-1) - comb_mix_ref.reshape(-1)).abs().max():.3e}"

    assert torch.allclose(
        layer_input.float(), layer_input_ref, atol=atol_bf16, rtol=2e-2
    ), f"layer_input mismatch (T={T}): max={(layer_input.float() - layer_input_ref).abs().max():.3e}"
