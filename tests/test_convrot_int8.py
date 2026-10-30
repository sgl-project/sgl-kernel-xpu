import pytest
import torch
from sgl_kernel.convrot import convrot_int8_linear_prequant, convrot_rotate_quantize

pytestmark = pytest.mark.arch("xe20")


def reference_rotate_quantize(x):
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=torch.float32,
        device=x.device,
    )
    h = h4
    for _ in range(3):
        h = torch.kron(h, h4)
    rotated = (x.float().reshape(-1, 256) @ (h / 16)).to(x.dtype)
    rotated = rotated.reshape_as(x)
    scale = (rotated.abs().amax(-1).float() / 127).clamp_min(1e-30)
    math_scale = scale.to(x.dtype)
    math_scale = torch.where(math_scale == 0, torch.finfo(x.dtype).tiny, math_scale)
    q = (rotated / math_scale[:, None]).round().clamp(-128, 127).to(torch.int8)
    return q, scale


def reference_linear(q, weight, scale, weight_scale, bias=None):
    acc = torch._int_mm(q, weight.T.contiguous())
    out = (acc.float() * (scale[:, None] * weight_scale.reshape(1, -1))).bfloat16()
    return out if bias is None else out + bias


@pytest.mark.parametrize(
    "k",
    [
        256,
        512,
        768,
        1024,
        1280,
        1792,
        2048,
        2304,
        3584,
        4096,
        4352,
        5376,
        6144,
        6400,
        8192,
    ],
)
@pytest.mark.parametrize("m", [0, 1, 17])
def test_rotate_quantize(m, k):
    torch.manual_seed(23)
    x = torch.randn(m, k, device="xpu", dtype=torch.bfloat16)
    q, scale = convrot_rotate_quantize(x)
    expected_q, expected_scale = reference_rotate_quantize(x)
    assert q.shape == x.shape and q.dtype == torch.int8
    assert scale.shape == (m,) and scale.dtype == torch.float32
    torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(q, expected_q, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["zero", "constant", "basis", "ties", "small"])
def test_rotation_and_quantization_edges(kind):
    x = torch.zeros(256, 512, device="xpu", dtype=torch.bfloat16)
    if kind == "constant":
        x.fill_(1)
    elif kind == "basis":
        x[:, :256] = torch.eye(256, device="xpu").bfloat16()
    elif kind == "ties":
        x.copy_(torch.arange(512, device="xpu") % 19 - 9)
    elif kind == "small":
        x.fill_(1e-30)
    expected_q, expected_scale = reference_rotate_quantize(x)
    q, scale = convrot_rotate_quantize(x)
    torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(q, expected_q, rtol=0, atol=0)


@pytest.mark.parametrize("k", range(256, 8193, 256))
def test_rotation_butterfly_order(k):
    torch.manual_seed(131 + k)
    x = torch.randn(17, k, dtype=torch.bfloat16)
    values = x.float()
    for stride in (1, 4, 16, 64):
        blocks = values.reshape(17, k // (4 * stride), 4, stride)
        a, b, c, d = blocks.unbind(2)
        values = torch.stack(
            (
                ((a + b) + c) - d,
                ((a + b) - c) + d,
                ((a - b) + c) + d,
                ((-a + b) + c) + d,
            ),
            dim=2,
        ).reshape_as(values)
    rotated = (values * 0.0625).bfloat16()
    scale = (rotated.abs().amax(-1).float() * (1.0 / 127.0)).clamp_min(1e-30)
    quotient = (rotated.float() / scale.bfloat16()[:, None].float()).bfloat16()
    expected_q = quotient.float().round().clamp(-128, 127).to(torch.int8)
    actual_q, actual_scale = convrot_rotate_quantize(x.to("xpu"))
    torch.testing.assert_close(actual_scale.cpu(), scale, rtol=0, atol=0)
    torch.testing.assert_close(actual_q.cpu(), expected_q, rtol=0, atol=0)


@pytest.mark.parametrize(
    "m,n,k",
    [
        (0, 32, 256),
        (1, 32, 256),
        (17, 96, 256),
        (257, 288, 512),
        (128, 5376, 1792),
        (2305, 288, 256),
        (4097, 608, 512),
        (8193, 96, 256),
    ],
)
@pytest.mark.parametrize("with_bias", [False, True])
def test_prequant_linear(m, n, k, with_bias):
    torch.manual_seed(41)
    q = torch.randint(-128, 128, (m, k), device="xpu", dtype=torch.int8)
    weight = torch.randint(-128, 128, (n, k), device="xpu", dtype=torch.int8)
    scale = torch.rand(m, device="xpu") / 128
    weight_scale = torch.rand(n, 1, device="xpu") / 128
    bias = torch.randn(n, device="xpu", dtype=torch.bfloat16) if with_bias else None
    out = convrot_int8_linear_prequant(q, weight, scale, weight_scale, bias)
    if m:
        expected = reference_linear(q, weight, scale, weight_scale, bias)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert out.shape == (m, n) and out.dtype == torch.bfloat16


@pytest.mark.parametrize("device", range(4))
@pytest.mark.parametrize("m", [16385, 16641, 16897])
@pytest.mark.parametrize("with_bias", [False, True])
def test_grouped_gemm_partial_band(device, m, with_bias):
    if torch.xpu.device_count() <= device:
        pytest.skip("requires additional visible XPU")
    with torch.xpu.device(device):
        torch.manual_seed(227 + m)
        n, k = 7168, 5376
        q = torch.randint(-128, 128, (m, k), device=f"xpu:{device}", dtype=torch.int8)
        weight = torch.randint(-128, 128, (n, k), device=q.device, dtype=torch.int8)
        scale = torch.rand(m, device=q.device) / 128
        weight_scale = torch.rand(n, device=q.device) / 128
        bias = (
            torch.randn(n, device=q.device, dtype=torch.bfloat16) if with_bias else None
        )
        out = convrot_int8_linear_prequant(q, weight, scale, weight_scale, bias)
        expected = reference_linear(q, weight, scale, weight_scale, bias)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "k,n", [(5376, 5376), (1792, 5376), (5376, 7168), (3584, 5376)]
)
def test_two_stage_projection_widths(k, n):
    torch.manual_seed(59)
    x = torch.randn(32, k, device="xpu", dtype=torch.bfloat16)
    weight = torch.randint(-127, 128, (n, k), device="xpu", dtype=torch.int8)
    weight_scale = torch.rand(n, device="xpu") / 128
    q, scale = convrot_rotate_quantize(x)
    expected_q, expected_scale = reference_rotate_quantize(x)
    out = convrot_int8_linear_prequant(q, weight, scale, weight_scale)
    expected = reference_linear(expected_q, weight, expected_scale, weight_scale)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_invalid_rotate_inputs():
    x = torch.ones(2, 256, device="xpu", dtype=torch.bfloat16)
    for invalid, message in [
        (x.float(), "dtype"),
        (x[:, ::2], "contiguous"),
        (x.reshape(-1), "2D"),
        (x[:, :0], "K must"),
        (torch.ones(1, 8448, device="xpu", dtype=torch.bfloat16), "K must"),
    ]:
        with pytest.raises(RuntimeError, match=message):
            convrot_rotate_quantize(invalid)
    with pytest.raises(RuntimeError, match="group_size"):
        convrot_rotate_quantize(x, 64)


def test_invalid_gemm_inputs():
    q = torch.ones(2, 256, device="xpu", dtype=torch.int8)
    weight = torch.ones(32, 256, device="xpu", dtype=torch.int8)
    scale = torch.ones(2, device="xpu")
    weight_scale = torch.ones(32, device="xpu")
    for args, message in [
        ((q.float(), weight, scale, weight_scale), "dtype"),
        ((q, weight.cpu(), scale, weight_scale), "device"),
        ((q, weight[:, :128].contiguous(), scale, weight_scale), "K must match"),
        ((q, weight[:31], scale, weight_scale[:31]), "N must"),
        ((q, weight, scale[:, None], weight_scale), "input_scale"),
        ((q, weight, scale, weight_scale[:1]), "weight_scale"),
        (
            (q, weight, scale, weight_scale, torch.zeros(32, device="xpu")),
            "bias.*dtype",
        ),
    ]:
        with pytest.raises(RuntimeError, match=message):
            convrot_int8_linear_prequant(*args)


@pytest.mark.parametrize("k", [256, 1792, 3584, 5376])
def test_non_default_stream(k):
    stream = torch.xpu.Stream()
    stream.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(stream):
        x = torch.ones(17, k, device="xpu", dtype=torch.bfloat16)
        weight = torch.ones(32, k, device="xpu", dtype=torch.int8)
        weight_scale = torch.ones(32, device="xpu")
        q, scale = convrot_rotate_quantize(x)
        out = convrot_int8_linear_prequant(q, weight, scale, weight_scale)
        expected = reference_linear(q, weight, scale, weight_scale)
    torch.xpu.current_stream().wait_stream(stream)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_zero_scale_preserves_sign_without_bias():
    q = torch.ones(2, 256, device="xpu", dtype=torch.int8)
    weight = -torch.ones(32, 256, device="xpu", dtype=torch.int8)
    scale = torch.zeros(2, device="xpu")
    weight_scale = torch.ones(32, device="xpu")
    out = convrot_int8_linear_prequant(q, weight, scale, weight_scale)
    expected = reference_linear(q, weight, scale, weight_scale)
    torch.testing.assert_close(
        out.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
    )


@pytest.mark.parametrize("device", [1, 2, 3])
@pytest.mark.parametrize("k", [256, 1792, 3584, 5376])
def test_other_devices(device, k):
    if torch.xpu.device_count() <= device:
        pytest.skip("requires additional visible XPU")
    with torch.xpu.device(device):
        x = torch.ones(17, k, device=f"xpu:{device}", dtype=torch.bfloat16)
        weight = torch.ones(32, k, device=x.device, dtype=torch.int8)
        weight_scale = torch.ones(32, device=x.device)
        q, scale = convrot_rotate_quantize(x)
        out = convrot_int8_linear_prequant(q, weight, scale, weight_scale)
        expected = reference_linear(q, weight, scale, weight_scale)
        assert out.device == x.device
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
