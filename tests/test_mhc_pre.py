import pytest
import torch
import utils
from sgl_kernel import mhc_pre
from test_hc_pre_fuse import _hc_pre_big_fuse_torch

device = utils.get_device()

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
    norm_weight = (torch.randn(D, dtype=torch.float32, device=device) * 0.5 + 1.0).to(
        torch.bfloat16
    )
    return residual, fn, hc_scale, hc_base, norm_weight


def _mhc_pre_torch(residual, fn, hc_scale, hc_base, norm_weight):
    T = residual.size(0)
    hc_hidden = HC_MULT * residual.size(2)
    A = residual.reshape(T, hc_hidden).float()
    B = fn.float()  # [24, hc_hidden]

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


@pytest.mark.parametrize(
    "T", [16, 48, 128, 512, 896, 1021, 1024, 1034, 1038, 1518, 2048]
)
@pytest.mark.parametrize("with_norm", [True, False])
def test_mhc_pre(T, with_norm):
    """Full mhc_pre pipeline vs pure-torch reference, production shapes."""
    D = 4096
    residual, fn, hc_scale, hc_base, norm_weight = _make_inputs(
        T, D, device=f"{device}:0"
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

    assert torch.allclose(
        post_mix, post_mix_ref, atol=2e-2, rtol=2e-2
    ), f"post_mix mismatch (T={T}): max={(post_mix - post_mix_ref).abs().max():.3e}"

    assert torch.allclose(
        comb_mix.reshape(T, HC_MULT, HC_MULT),
        comb_mix_ref.reshape(T, HC_MULT, HC_MULT),
        atol=2e-2,
        rtol=2e-2,
    ), f"comb_mix mismatch (T={T}): max={(comb_mix.reshape(-1) - comb_mix_ref.reshape(-1)).abs().max():.3e}"

    assert torch.allclose(
        layer_input.float(), layer_input_ref, atol=2e-2, rtol=2e-2
    ), f"layer_input mismatch (T={T}): max={(layer_input.float() - layer_input_ref).abs().max():.3e}"
