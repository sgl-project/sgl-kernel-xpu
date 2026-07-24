import pytest
import torch

from sgl_kernel import rel_proj_small_t, row_compact_bf16, row_scale_bf16

try:
    HAS_XPU = torch.xpu.is_available()
except (ImportError, AttributeError):
    HAS_XPU = False

pytestmark = pytest.mark.skipif(not HAS_XPU, reason="Inkling helper tests require XPU")


def _make_row_input(rows: int, inner: int, strided: bool) -> torch.Tensor:
    torch.manual_seed(rows * 1009 + inner * 17 + int(strided))
    if strided:
        offset = 8
        packed = torch.randn(
            rows, inner + offset + 40, device="xpu", dtype=torch.bfloat16
        )
        return packed[:, offset : offset + inner]
    return torch.randn(rows, inner, device="xpu", dtype=torch.bfloat16)


@pytest.mark.parametrize("rows", [1, 3, 37, 512])
@pytest.mark.parametrize("inner", [5, 8, 17, 256])
@pytest.mark.parametrize("strided", [False, True])
def test_row_scale_bf16_matches_reference(rows, inner, strided):
    x = _make_row_input(rows, inner, strided)
    tau = 1.0 + 0.1 * torch.rand(rows, device="xpu", dtype=torch.float32)

    out = row_scale_bf16(x, tau)
    ref = (x.float() * tau.view(-1, 1)).bfloat16()

    assert out.is_contiguous()
    assert out.shape == x.shape
    assert torch.equal(out, ref)


@pytest.mark.parametrize(
    ("rows", "inner", "strided"),
    [
        (4096, 256, True),
        (512, 16_384, False),
    ],
)
def test_row_scale_bf16_production_shapes_match_reference(rows, inner, strided):
    x = _make_row_input(rows, inner, strided)
    tau = 1.0 + 0.1 * torch.rand(rows, device="xpu", dtype=torch.float32)

    out = row_scale_bf16(x, tau)
    ref = (x.float() * tau.view(-1, 1)).bfloat16()

    assert out.is_contiguous()
    assert torch.equal(out, ref)


@pytest.mark.parametrize("rows", [1, 3, 41, 512])
@pytest.mark.parametrize("inner", [4, 8, 19, 256])
@pytest.mark.parametrize("strided", [False, True])
def test_row_compact_bf16_matches_contiguous(rows, inner, strided):
    x = _make_row_input(rows, inner, strided)

    out = row_compact_bf16(x)

    assert out.is_contiguous()
    assert torch.equal(out, x.contiguous())


@pytest.mark.parametrize(
    ("rows", "inner", "strided"),
    [
        (4096, 256, True),
        (512, 16_384, False),
    ],
)
def test_row_compact_bf16_production_shapes_match_contiguous(rows, inner, strided):
    x = _make_row_input(rows, inner, strided)

    out = row_compact_bf16(x)

    assert out.is_contiguous()
    assert torch.equal(out, x.contiguous())


def _make_r(t: int, h: int, d: int, strided: bool) -> torch.Tensor:
    torch.manual_seed(t * 101 + h * 17 + d * 13 + int(strided))
    if strided:
        offset = 8
        row = offset + h * d + 40
        packed = torch.randn(t, row, device="xpu", dtype=torch.bfloat16)
        return packed[:, offset : offset + h * d].view(t, h, d)
    return torch.randn(t, h, d, device="xpu", dtype=torch.bfloat16)


def _rel_proj_ref(
    r: torch.Tensor, proj: torch.Tensor, tau: torch.Tensor | None
) -> torch.Tensor:
    rf = r.float()
    if tau is not None:
        rf = (rf * tau.view(-1, 1, 1)).bfloat16().float()
    return torch.einsum("thd,de->the", rf, proj.float())


@pytest.mark.parametrize("t", [1, 5, 32])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("with_tau", [False, True])
def test_rel_proj_small_t_matches_reference(t, strided, with_tau):
    h, d, e = 6, 16, 65
    r = _make_r(t, h, d, strided)
    proj = torch.randn(d, e, device="xpu", dtype=torch.bfloat16) * 0.1
    tau = (
        1.0 + 0.1 * torch.rand(t, device="xpu", dtype=torch.float32)
        if with_tau
        else None
    )

    out = rel_proj_small_t(r, proj, tau)
    ref = _rel_proj_ref(r, proj, tau)

    assert out.is_contiguous()
    assert out.shape == (t, h, e)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("t", [1, 2, 4])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("with_tau", [False, True])
def test_rel_proj_small_t_production_esimd_matches_reference(t, strided, with_tau):
    h, d, e = 16, 16, 1024
    r = _make_r(t, h, d, strided)
    proj = torch.randn(d, e, device="xpu", dtype=torch.bfloat16) * 0.1
    tau = (
        1.0 + 0.1 * torch.rand(t, device="xpu", dtype=torch.float32)
        if with_tau
        else None
    )

    out = rel_proj_small_t(r, proj, tau)
    ref = _rel_proj_ref(r, proj, tau)

    assert out.is_contiguous()
    assert out.shape == (t, h, e)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


def test_rel_proj_tau_prescale_rounding_isolated():
    t, h, d, e = 8, 6, 16, 64
    r = _make_r(t, h, d, True)
    proj = torch.randn(d, e, device="xpu", dtype=torch.bfloat16) * 0.1
    tau = 1.0 + 0.5 * torch.rand(t, device="xpu", dtype=torch.float32)

    out = rel_proj_small_t(r, proj, tau)
    r_pre = (r.float() * tau.view(-1, 1, 1)).bfloat16()

    assert torch.equal(out, rel_proj_small_t(r_pre.contiguous(), proj))
