import pytest
import torch
from sgl_kernel import index_k_norm_rope_pack_store, index_q_rope_pack_weights

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available(), reason="XPU not available"
)

HEAD_DIM = 128
ROPE_DIM = 64
GROUP_SIZE = 32
NUM_GROUPS = HEAD_DIM // GROUP_SIZE
PACKED_DIM = HEAD_DIM // 2
SLOT_BYTES = PACKED_DIM + NUM_GROUPS


def _freqs(max_pos: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(max_pos, device=device, dtype=torch.float32)
    frequencies = torch.arange(ROPE_DIM // 2, device=device, dtype=torch.float32)
    angles = positions[:, None] / (10000.0 ** (frequencies[None, :] / (ROPE_DIM // 2)))
    return torch.view_as_real(torch.polar(torch.ones_like(angles), angles)).flatten(-2)


def _rope(x: torch.Tensor, freqs_cis: torch.Tensor, positions: torch.Tensor):
    pairs = x[..., -ROPE_DIM:].float().view(*x.shape[:-1], ROPE_DIM // 2, 2)
    freqs = freqs_cis[positions].view(
        positions.numel(), *([1] * (x.ndim - 2)), ROPE_DIM // 2, 2
    )
    real = pairs[..., 0] * freqs[..., 0] - pairs[..., 1] * freqs[..., 1]
    imag = pairs[..., 0] * freqs[..., 1] + pairs[..., 1] * freqs[..., 0]
    rotated = torch.stack((real, imag), dim=-1).flatten(-2).to(x.dtype)
    return torch.cat((x[..., :-ROPE_DIM], rotated), dim=-1)


def _scale_from_exponent(exponent: torch.Tensor) -> torch.Tensor:
    return (exponent << 23).contiguous().view(torch.float32)


def _ceil_ue8m0_exponent(value: torch.Tensor) -> torch.Tensor:
    bits = value.contiguous().view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    return (exponent + (mantissa != 0).to(torch.int32)).clamp(1, 254)


def _e2m1_code_rne(value: torch.Tensor) -> torch.Tensor:
    magnitude = value.abs().clamp_max(6.0)
    thresholds = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        device=value.device,
    )
    code = (magnitude.unsqueeze(-1) >= thresholds).sum(-1).to(torch.uint8)
    boundary = (magnitude.unsqueeze(-1) == thresholds).any(-1)
    code = torch.where(boundary & ((code & 1) != 0), code - 1, code)
    sign = ((value < 0) & (code != 0)).to(torch.uint8) * 8
    return code | sign


def _fake_quant_fp4(x: torch.Tensor) -> torch.Tensor:
    groups = x.float().view(-1, NUM_GROUPS, GROUP_SIZE)
    amax = groups.abs().amax(-1).clamp_min(6.0 * 2.0**-126)
    exponent = _ceil_ue8m0_exponent(amax * (1.0 / 6.0))
    scale = _scale_from_exponent(exponent)
    normalized = groups / scale.unsqueeze(-1)
    magnitude = normalized.abs()
    step = torch.where(magnitude < 2.0, 0.5, torch.where(magnitude < 4.0, 1.0, 2.0))
    dequantized = torch.round(magnitude / step) * step * torch.sign(normalized)
    return (dequantized * scale.unsqueeze(-1)).flatten(-2).to(x.dtype)


def _pack_fp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows = x.float().view(-1, HEAD_DIM)
    groups = rows.view(-1, NUM_GROUPS, GROUP_SIZE)
    exponent = _ceil_ue8m0_exponent((groups.abs().amax(-1) / 6.0).clamp_min(1.0e-4))
    scale = _scale_from_exponent(exponent)
    codes = _e2m1_code_rne(groups / scale.unsqueeze(-1)).view(-1, HEAD_DIM)
    packed = (
        codes[:, 0::2].to(torch.int16) | (codes[:, 1::2].to(torch.int16) << 4)
    ).to(torch.int8)
    packed_scale = (
        exponent[:, 0]
        | (exponent[:, 1] << 8)
        | (exponent[:, 2] << 16)
        | (exponent[:, 3] << 24)
    )
    return packed, packed_scale.to(torch.int32)


def _reference_query(q, freqs_cis, positions, head_weights, weight_scale):
    rotated = _rope(q, freqs_cis, positions)
    fake_quantized = _fake_quant_fp4(rotated)
    payload, scale = _pack_fp4(fake_quantized)
    weights = (head_weights.float() * weight_scale).to(torch.bfloat16).float()
    return payload, scale, weights


def _reference_key(input, norm_weight, eps, freqs_cis, positions):
    x = input.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    x = (norm_weight.float() * x).to(input.dtype)
    rotated = _rope(x, freqs_cis, positions)
    return _pack_fp4(_fake_quant_fp4(rotated))


@pytest.mark.parametrize("batch,heads", [(3, 5), (64, 8), (64, 64)])
def test_index_q_rope_pack_weights(batch, heads):
    device = torch.device("xpu")
    torch.manual_seed(2026)
    q = torch.randn(batch, heads, HEAD_DIM, device=device, dtype=torch.bfloat16)
    weights = torch.randn(batch, heads, device=device, dtype=torch.bfloat16)
    positions = torch.arange(batch, device=device, dtype=torch.int32) % 32
    freqs_cis = _freqs(32, device)
    weight_scale = 0.137

    got = index_q_rope_pack_weights(q, freqs_cis, positions, weights, weight_scale)
    expected = _reference_query(q, freqs_cis, positions, weights, weight_scale)
    for actual, reference in zip(got, expected):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("norm_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("ratio", [1, 2, 4])
def test_index_k_norm_rope_pack_store(norm_dtype, ratio):
    device = torch.device("xpu")
    torch.manual_seed(2110 + ratio)
    num_tokens = 4
    input = torch.randn(num_tokens, HEAD_DIM, device=device, dtype=torch.bfloat16)
    norm_weight = torch.randn(HEAD_DIM, device=device, dtype=norm_dtype)
    positions = torch.tensor([0, 3, 7, 11], device=device, dtype=torch.int64)
    loc = torch.tensor([0, 2, 5, 11], device=device, dtype=torch.int64)
    freqs_cis = _freqs(16, device)
    page_size = 4
    cache = torch.full(
        (4, page_size * SLOT_BYTES), 0xA5, device=device, dtype=torch.uint8
    )
    eps = 1.0e-6

    index_k_norm_rope_pack_store(
        input,
        norm_weight,
        eps,
        freqs_cis,
        positions,
        loc,
        cache,
        ratio=ratio,
    )

    group_positions = positions & ~(ratio - 1)
    payload, scale = _reference_key(input, norm_weight, eps, freqs_cis, group_positions)
    payload = payload.view(num_tokens, PACKED_DIM).view(torch.uint8)
    scale_bytes = scale.contiguous().view(torch.uint8).view(num_tokens, NUM_GROUPS)
    for row, slot in enumerate(loc.tolist()):
        if slot == 0:
            continue
        page, offset = divmod(slot, page_size)
        page_cache = cache[page]
        torch.testing.assert_close(
            page_cache[offset * PACKED_DIM : (offset + 1) * PACKED_DIM],
            payload[row],
            rtol=0,
            atol=0,
        )
        scale_start = page_size * PACKED_DIM + offset * NUM_GROUPS
        torch.testing.assert_close(
            page_cache[scale_start : scale_start + NUM_GROUPS],
            scale_bytes[row],
            rtol=0,
            atol=0,
        )
    assert torch.all(cache[0, :SLOT_BYTES] == 0xA5)
