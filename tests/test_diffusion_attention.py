"""Numerical contracts for dense diffusion attention on one XPU."""

import pytest
import torch
import torch.nn.functional as F
from sgl_kernel import diffusion
from sgl_kernel.flash_attn import flash_attn_varlen_func

pytestmark = [
    pytest.mark.arch("xe20"),
    pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU required"),
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "length,heads",
    [
        (1, 1),
        (17, 3),
        (31, 3),
        (32, 3),
        (33, 3),
        (255, 3),
        (256, 3),
        (257, 3),
        (511, 3),
        (512, 3),
        (513, 3),
        (4128, 30),
        (4160, 24),
    ],
)
def test_attention(dtype, length, heads):
    torch.manual_seed(1013 + length)
    shape = (1, length, heads, 128)
    q = torch.randn(shape, device="xpu", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    actual = diffusion.sdp(q, k, v)
    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)

    assert actual.shape == shape and actual.is_contiguous()
    assert torch.isfinite(actual).all()
    tolerance = 0.00390625 if dtype == torch.bfloat16 else 0.002
    assert (actual.float() - expected.float()).abs().max().item() <= tolerance
    if length <= 513:
        # An independent FP32 path can round BF16 at a different final ULP.
        expected_fp32 = (
            F.scaled_dot_product_attention(
                q.transpose(1, 2).float(),
                k.transpose(1, 2).float(),
                v.transpose(1, 2).float(),
            )
            .transpose(1, 2)
            .to(dtype)
        )
        fp32_tolerance = 0.015625 if dtype == torch.bfloat16 else 0.002
        assert (
            actual.float() - expected_fp32.float()
        ).abs().max().item() <= fp32_tolerance


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("length,heads", [(257, 3), (4160, 24)])
def test_attention_packed(dtype, length, heads):
    q, k, v = torch.randn(1, length, 3, heads, 128, device="xpu", dtype=dtype).unbind(2)
    torch.testing.assert_close(
        diffusion.sdp(q, k, v),
        diffusion.sdp(q.contiguous(), k.contiguous(), v.contiguous()),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_h3_packed_against_flash_attention(dtype):
    length = 16464
    q, k, v = torch.randn(1, length, 3, 7, 128, device="xpu", dtype=dtype).unbind(2)
    cu_seqlens = torch.tensor((0, length), device="xpu", dtype=torch.int32)
    expected = flash_attn_varlen_func(
        q=q[0],
        k=k[0],
        v=v[0],
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=length,
        max_seqlen_k=length,
        softmax_scale=128**-0.5,
        causal=False,
    )
    if isinstance(expected, tuple):
        expected = expected[0]
    actual = diffusion.sdp(q, k, v)[0]
    assert torch.isfinite(actual).all()
    tolerance = 0.00390625 if dtype == torch.bfloat16 else 0.002
    assert (actual.float() - expected.float()).abs().max().item() <= tolerance


def test_current_stream():
    stream = torch.xpu.Stream()
    with torch.xpu.stream(stream):
        q = torch.ones((1, 17, 1, 128), device="xpu", dtype=torch.bfloat16) * 3
        result = diffusion.sdp(q, q, q)
        delta = (result.float() - q.float()).abs().max()
    stream.synchronize()
    assert delta.item() < 0.01


def test_invalid_contracts():
    q = torch.ones((1, 17, 1, 128), device="xpu", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="B==1"):
        diffusion.sdp(q.expand(2, -1, -1, -1), q, q)
    with pytest.raises(RuntimeError, match="head_dim"):
        diffusion.sdp(q[..., :64], q[..., :64], q[..., :64])
    with pytest.raises(RuntimeError, match="dtype"):
        diffusion.sdp(q, q.float(), q)
    with pytest.raises(RuntimeError, match="self-attention"):
        diffusion.sdp(q, q[:, :16], q[:, :16])
    with pytest.raises(RuntimeError, match="XPU"):
        diffusion.sdp(q, q.cpu(), q)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_sdp_schema_and_stream(dtype):
    assert diffusion.is_available()
    names = [
        schema.name
        for schema in torch._C._jit_get_all_schemas()
        if schema.name.startswith("sgl_kernel::diffusion_sdp")
    ]
    assert names == ["sgl_kernel::diffusion_sdp"]

    stream = torch.xpu.Stream()
    with torch.xpu.stream(stream):
        q = torch.ones((1, 33, 3, 128), device="xpu", dtype=dtype)
        v = q * 3
        output = diffusion.sdp(q, q, v)
        delta = (output.float() - v.float()).abs().max()
    stream.synchronize()
    assert output.dtype == dtype and output.shape == q.shape
    assert delta.item() < 0.01


def test_native_sdp_rejects_unsupported_dtype():
    assert diffusion.is_available()
    q = torch.ones((1, 17, 1, 128), device="xpu", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="only fp16/bf16"):
        diffusion.sdp(q, q, q)
