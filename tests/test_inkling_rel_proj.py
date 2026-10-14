import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="Inkling relative projection is XPU-only",
)


def make_packed_r(t, h, kv_heads, d):
    q_width = h * 128
    kv_width = 2 * kv_heads * 128
    packed = torch.randn(
        (t, q_width + kv_width + h * d), device="xpu", dtype=torch.bfloat16
    )
    return packed[:, q_width + kv_width :].view(t, h, d)


def rel_proj_ref(r, proj, tau):
    r = r.detach().cpu().float()
    proj = proj.detach().cpu().float()
    tau = tau.detach().cpu().float()
    r = (r * tau.view(-1, 1, 1)).bfloat16().float()
    return torch.einsum("thd,de->the", r, proj).bfloat16().float()


@pytest.mark.parametrize(
    "t,h,kv_heads",
    [
        (1, 24, 2),
        (9, 12, 1),
        (32, 6, 1),
    ],
)
def test_rel_proj_small_t_matches_inkling_shapes(t, h, kv_heads):
    from sgl_kernel import rel_proj_small_t

    torch.manual_seed(7)
    d, e = 16, 1024
    r = make_packed_r(t, h, kv_heads, d)
    proj = torch.randn(d, e, device="xpu", dtype=torch.bfloat16) * 0.1
    tau = 1.0 + 0.1 * torch.rand(t, device="xpu", dtype=torch.float32)
    out = torch.empty(t, h, e, device="xpu", dtype=torch.bfloat16)

    returned = rel_proj_small_t(r, proj, tau, out)
    reference = rel_proj_ref(r, proj, tau)

    assert returned.data_ptr() == out.data_ptr()
    assert out.is_contiguous()
    assert out.shape == (t, h, e)
    torch.testing.assert_close(
        out.detach().cpu().float(), reference, atol=1e-6, rtol=2.0**-7
    )


def test_rel_proj_small_t_rejects_contiguous_r():
    from sgl_kernel import rel_proj_small_t

    t, h, d, e = 5, 12, 16, 1024
    r = torch.randn(t, h, d, device="xpu", dtype=torch.bfloat16)
    proj = torch.randn(d, e, device="xpu", dtype=torch.bfloat16)
    tau = torch.ones(t, device="xpu", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="packed qkvr"):
        rel_proj_small_t(r, proj, tau)


@pytest.mark.parametrize("d,e", [(13, 1024), (16, 65)])
def test_rel_proj_small_t_rejects_nonproduction_projection_shape(d, e):
    from sgl_kernel import rel_proj_small_t

    t, h = 5, 3
    packed = torch.randn(t, h * d + 19, device="xpu", dtype=torch.bfloat16)
    r = packed[:, : h * d].view(t, h, d)
    proj = torch.randn(d, e, device="xpu", dtype=torch.bfloat16)
    tau = torch.ones(t, device="xpu", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="only production"):
        rel_proj_small_t(r, proj, tau)
