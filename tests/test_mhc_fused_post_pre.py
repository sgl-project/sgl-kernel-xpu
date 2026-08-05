import pytest
import torch
from sgl_kernel import hc_post, mhc_fused_post_pre, mhc_pre

HC_MULT = 4
HC_MULT3 = (2 + HC_MULT) * HC_MULT  # 24
SINKHORN_REPEAT = 20
RMS_EPS = 1e-6
HC_PRE_EPS = 1e-6
HC_SINKHORN_EPS = 1e-6
HC_POST_MULT_VALUE = 2.0
NORM_EPS = 1e-6


@pytest.fixture(autouse=True)
def skip_if_no_xpu():
    if not torch.xpu.is_available():
        pytest.skip("XPU not available")


def _make_inputs(t, d, device, seed=42):
    torch.manual_seed(seed)
    hc_hidden = HC_MULT * d

    x = torch.randn(t, d, dtype=torch.bfloat16, device=device)
    residual = torch.randn(t, HC_MULT, d, dtype=torch.bfloat16, device=device)
    post = torch.rand(t, HC_MULT, dtype=torch.float32, device=device) * 2.0
    comb = torch.rand(t, HC_MULT, HC_MULT, dtype=torch.float32, device=device)
    comb = comb / comb.sum(dim=-1, keepdim=True)

    fn = torch.randn(HC_MULT3, hc_hidden, dtype=torch.float32, device=device)
    hc_scale = torch.rand(3, dtype=torch.float32, device=device) * 0.5 + 0.5
    hc_base = torch.randn(HC_MULT3, dtype=torch.float32, device=device) * 0.1
    norm_weight = (torch.randn(d, dtype=torch.float32, device=device) * 0.5 + 1.0).to(
        torch.bfloat16
    )

    return x, residual, post, comb, fn, hc_scale, hc_base, norm_weight


@pytest.mark.parametrize("d", [4096, 7168])
@pytest.mark.parametrize("t", [0, 1, 8, 17, 32, 64])
@pytest.mark.parametrize("with_norm", [False, True])
def test_mhc_fused_post_pre(t, d, with_norm):
    x, residual, post, comb, fn, hc_scale, hc_base, norm_weight = _make_inputs(
        t, d, device="xpu:0"
    )

    nw = norm_weight if with_norm else None

    residual_cur, post_cur, comb_cur, layer_input_cur = mhc_fused_post_pre(
        x,
        residual,
        post,
        comb,
        fn,
        hc_scale,
        hc_base,
        rms_eps=RMS_EPS,
        hc_pre_eps=HC_PRE_EPS,
        hc_sinkhorn_eps=HC_SINKHORN_EPS,
        hc_post_mult_value=HC_POST_MULT_VALUE,
        sinkhorn_repeat=SINKHORN_REPEAT,
        norm_weight=nw,
        norm_eps=NORM_EPS if with_norm else None,
    )

    residual_ref = hc_post(x, residual, post, comb)
    post_ref, comb_ref, layer_input_ref = mhc_pre(
        residual_ref,
        fn,
        hc_scale,
        hc_base,
        rms_eps=RMS_EPS,
        hc_pre_eps=HC_PRE_EPS,
        hc_sinkhorn_eps=HC_SINKHORN_EPS,
        hc_post_mult_value=HC_POST_MULT_VALUE,
        sinkhorn_repeat=SINKHORN_REPEAT,
        norm_weight=nw,
        norm_eps=NORM_EPS if with_norm else None,
    )

    if t == 0:
        assert residual_cur.shape == residual.shape
        assert post_cur.shape == (0, HC_MULT, 1)
        assert comb_cur.shape == (0, HC_MULT, HC_MULT)
        assert layer_input_cur.shape == (0, d)
        assert residual_cur.dtype == torch.bfloat16
        assert post_cur.dtype == torch.float32
        assert comb_cur.dtype == torch.float32
        assert layer_input_cur.dtype == torch.bfloat16
        assert residual_ref.shape == residual_cur.shape
        assert post_ref.shape == post_cur.squeeze(-1).shape
        assert comb_ref.shape == comb_cur.shape
        assert layer_input_ref.shape == layer_input_cur.shape
        assert residual_ref.dtype == residual_cur.dtype
        assert post_ref.dtype == post_cur.dtype
        assert comb_ref.dtype == comb_cur.dtype
        assert layer_input_ref.dtype == layer_input_cur.dtype
        return

    torch.testing.assert_close(
        residual_cur,
        residual_ref,
        atol=1e-2,
        rtol=1e-2,
        msg=f"residual_cur mismatch (T={t}, D={d}, norm={with_norm})",
    )

    torch.testing.assert_close(
        post_cur.squeeze(-1),
        post_ref,
        atol=2e-2,
        rtol=2e-2,
        msg=f"post_mix mismatch (T={t}, D={d}, norm={with_norm})",
    )

    torch.testing.assert_close(
        comb_cur,
        comb_ref,
        atol=2e-2,
        rtol=2e-2,
        msg=f"comb_mix mismatch (T={t}, D={d}, norm={with_norm})",
    )

    torch.testing.assert_close(
        layer_input_cur,
        layer_input_ref,
        atol=2e-2,
        rtol=2e-2,
        msg=f"layer_input mismatch (T={t}, D={d}, norm={with_norm})",
    )
