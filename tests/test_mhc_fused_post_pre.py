import pytest
import torch
import utils
from sgl_kernel import hc_post, mhc_fused_post_pre, mhc_pre
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


def _hc_post_torch_impl(x, residual, post, comb):
    return (
        post.unsqueeze(-1) * x.unsqueeze(1)
        + (comb.unsqueeze(-1) * residual.unsqueeze(2)).sum(dim=1)
    ).type_as(x)


def _mhc_pre_torch(residual, fn, hc_scale, hc_base, norm_weight):
    t = residual.size(0)
    hc_hidden = HC_MULT * residual.size(2)
    a = residual.reshape(t, hc_hidden).float()
    b = fn.float()  # [24, hc_hidden]

    gemm_out_mul = (a @ b.t()).unsqueeze(0)  # [1, T, 24]
    gemm_out_sqrsum = (a * a).sum(dim=1).unsqueeze(0)  # [1, T]

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


def _bench_xpu(fn, *, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()

    times = []
    for _ in range(iters):
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.xpu.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def _bench_xpu_pair(fn_a, fn_b, *, warmup=15, iters=30):
    for _ in range(warmup):
        fn_a()
        fn_b()
    torch.xpu.synchronize()

    a_times = []
    b_times = []
    for _ in range(iters):
        sa = torch.xpu.Event(enable_timing=True)
        ea = torch.xpu.Event(enable_timing=True)
        sa.record()
        fn_a()
        ea.record()
        sb = torch.xpu.Event(enable_timing=True)
        eb = torch.xpu.Event(enable_timing=True)
        sb.record()
        fn_b()
        eb.record()
        torch.xpu.synchronize()
        a_times.append(sa.elapsed_time(ea))
        b_times.append(sb.elapsed_time(eb))
    return min(a_times), min(b_times)


@pytest.mark.parametrize("d", [4096, 7168])
@pytest.mark.parametrize("t", [0, 1, 8, 17, 32, 64])
@pytest.mark.parametrize("with_norm", [False, True])
def test_mhc_fused_post_pre(t, d, with_norm):
    x, residual, post, comb, fn, hc_scale, hc_base, norm_weight = _make_inputs(
        t, d, device=f"{device}:0"
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

    if t == 0:
        assert residual_cur.shape == residual.shape
        assert post_cur.shape == (0, HC_MULT, 1)
        assert comb_cur.shape == (0, HC_MULT, HC_MULT)
        assert layer_input_cur.shape == (0, d)
        assert residual_cur.dtype == torch.bfloat16
        assert post_cur.dtype == torch.float32
        assert comb_cur.dtype == torch.float32
        assert layer_input_cur.dtype == torch.bfloat16
        return

    residual_ref = _hc_post_torch_impl(x, residual, post, comb)
    post_ref, comb_ref, layer_input_ref = _mhc_pre_torch(
        residual_ref, fn, hc_scale, hc_base, nw
    )

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
        comb_ref.reshape(t, HC_MULT, HC_MULT),
        atol=2e-2,
        rtol=2e-2,
        msg=f"comb_mix mismatch (T={t}, D={d}, norm={with_norm})",
    )

    torch.testing.assert_close(
        layer_input_cur.float(),
        layer_input_ref,
        atol=2e-2,
        rtol=2e-2,
        msg=f"layer_input mismatch (T={t}, D={d}, norm={with_norm})",
    )


@torch.inference_mode()
@pytest.mark.parametrize("d", [4096, 7168])
@pytest.mark.parametrize("t", [1, 8, 17, 32, 64])
def test_mhc_fused_post_pre_perf(t, d):
    if not torch.xpu.is_available():
        pytest.skip("XPU is required for perf test")

    x, residual, post, comb, fn, hc_scale, hc_base, norm_weight = _make_inputs(
        t, d, device=f"{device}:0", seed=123
    )

    def run_fused():
        mhc_fused_post_pre(
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
            norm_weight=norm_weight,
            norm_eps=NORM_EPS,
        )

    def run_split():
        residual_cur = hc_post(x, residual, post, comb)
        mhc_pre(
            residual_cur,
            fn,
            hc_scale,
            hc_base,
            rms_eps=RMS_EPS,
            hc_pre_eps=HC_PRE_EPS,
            hc_sinkhorn_eps=HC_SINKHORN_EPS,
            hc_post_mult_value=HC_POST_MULT_VALUE,
            sinkhorn_repeat=SINKHORN_REPEAT,
            norm_weight=norm_weight,
            norm_eps=NORM_EPS,
        )

    fused_ms, split_ms = _bench_xpu_pair(run_fused, run_split)

    assert fused_ms < split_ms, (
        f"fused op is not faster: " f"fused={fused_ms:.3f} ms, split={split_ms:.3f} ms"
    )
