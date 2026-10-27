import pytest
import torch
from sgl_kernel import fused_q_indexer_rope_hadamard_fp4_quant

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available(), reason="XPU not available"
)


HEAD_DIM = 128
FP4_DIM = HEAD_DIM // 2
GROUP_SIZE = 32
SCALE_GROUPS = HEAD_DIM // GROUP_SIZE
E2M1_MAX = 6.0


def _fwht128(x: torch.Tensor) -> torch.Tensor:
    leading = x.shape[:-1]
    transformed = x
    width = 1
    while width < HEAD_DIM:
        transformed = transformed.reshape(*leading, HEAD_DIM // (2 * width), 2, width)
        even, odd = transformed[..., 0, :], transformed[..., 1, :]
        transformed = torch.stack([even + odd, even - odd], dim=-2).reshape(
            *leading, HEAD_DIM
        )
        width *= 2
    return transformed


def _ceil_ue8m0_exp_ref(x: torch.Tensor) -> torch.Tensor:
    bits = x.to(torch.float32).contiguous().view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    return (exponent + (mantissa != 0).to(torch.int32)).clamp(1, 254)


def _fp4_e2m1_code_ref(x: torch.Tensor) -> torch.Tensor:
    absolute = torch.minimum(x.abs(), torch.tensor(E2M1_MAX, device=x.device))
    code = torch.zeros_like(absolute, dtype=torch.uint8)
    for threshold in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        comparison = (
            absolute >= threshold
            if threshold in (0.75, 1.75, 3.5)
            else absolute > threshold
        )
        code += comparison.to(torch.uint8)
    sign = ((x < 0) & (code != 0)).to(torch.uint8) * 8
    return code | sign


def test_fp4_e2m1_midpoint_reference():
    midpoints = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device="xpu")
    expected = torch.tensor([0, 2, 2, 4, 4, 6, 6], device="xpu", dtype=torch.uint8)
    torch.testing.assert_close(_fp4_e2m1_code_ref(midpoints), expected, rtol=0, atol=0)
    negative_expected = torch.tensor(
        [0, 10, 10, 12, 12, 14, 14], device="xpu", dtype=torch.uint8
    )
    torch.testing.assert_close(
        _fp4_e2m1_code_ref(-midpoints), negative_expected, rtol=0, atol=0
    )


@pytest.mark.parametrize("midpoint,expected_code", [(1.75, 3), (3.5, 6)])
def test_fp4_fused_q_midpoint_rounding(midpoint, expected_code):
    device = torch.device("xpu")
    q = torch.zeros(1, 1, HEAD_DIM, device=device, dtype=torch.bfloat16)
    q[..., 64] = 1
    q[..., 66] = 1
    rope_cache = torch.zeros(1, 64, device=device)
    # Two impulses exercise FP4 thresholds after RoPE rounds back to BF16.
    scale = torch.tensor(HEAD_DIM**-0.5, device=device)
    rope_cache[0, 0] = (4 + midpoint) / (2 * scale)
    rope_cache[0, 2] = (4 - midpoint) / (2 * scale)
    positions = torch.zeros(1, device=device, dtype=torch.int32)
    weight = torch.ones(1, 1, device=device, dtype=torch.bfloat16)
    payload = torch.empty(1, 1, FP4_DIM, device=device, dtype=torch.uint8)
    scales = torch.empty(1, 1, SCALE_GROUPS, device=device, dtype=torch.uint8)
    weights_out = torch.empty(1, 1, 1, device=device)
    fused_q_indexer_rope_hadamard_fp4_quant(
        q, payload, scales, weight, weights_out, 1.0, rope_cache, positions
    )
    indices = torch.arange(HEAD_DIM, device=device)
    codes = torch.where((indices & 2) == 0, 6, expected_code)
    codes = (codes | torch.where((indices & 64) != 0, 8, 0)).to(torch.uint8)
    expected = (codes[0::2] | (codes[1::2] << 4)).view(1, 1, FP4_DIM)
    torch.testing.assert_close(payload, expected, rtol=0, atol=0)
    torch.testing.assert_close(scales, torch.full_like(scales, 127), rtol=0, atol=0)


def _ref_quantize_fp4_indexer(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = x.contiguous().view(-1, HEAD_DIM).float()
    groups = values.view(-1, SCALE_GROUPS, GROUP_SIZE)
    scale_raw = groups.abs().amax(dim=-1).clamp_min(1.0e-4) / E2M1_MAX
    scale_exp = _ceil_ue8m0_exp_ref(scale_raw)
    scale = (scale_exp << 23).contiguous().view(torch.float32)
    codes = _fp4_e2m1_code_ref((groups / scale.unsqueeze(-1)).view(-1, HEAD_DIM))
    packed = (
        codes[:, 0::2].to(torch.int16) | (codes[:, 1::2].to(torch.int16) << 4)
    ).to(torch.uint8)
    packed_scale = scale_exp[:, 0].clone()
    for group_id in range(1, SCALE_GROUPS):
        packed_scale |= scale_exp[:, group_id] << (8 * group_id)
    return packed, packed_scale


def _make_freqs_cis(max_pos: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(max_pos, device=device, dtype=torch.float32)
    frequencies = torch.arange(32, device=device, dtype=torch.float32)
    angles = positions[:, None] / (10000.0 ** (frequencies[None, :] / 32.0))
    return torch.complex(torch.cos(angles), torch.sin(angles))


@pytest.mark.parametrize("batch_size", [1, 5, 17])
@pytest.mark.parametrize("input_scale", [0.0, 1.0e-6, 1.0e-4, 1.0])
def test_fp4_fused_q_indexer_rope_hadamard_quant(batch_size: int, input_scale: float):
    device = torch.device("xpu")
    torch.manual_seed(batch_size + 200)
    num_heads = 8
    weight_scale = HEAD_DIM**-0.5 * num_heads**-0.5
    q_input = torch.randn(
        batch_size, num_heads, HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    q_input.mul_(input_scale)
    weight = torch.randn(batch_size, num_heads, device=device, dtype=torch.bfloat16)
    positions = (torch.arange(batch_size, device=device, dtype=torch.int32) * 7) % 63
    rope_cache = torch.view_as_real(_make_freqs_cis(64, device)).flatten(-2)
    q_fp4 = torch.empty(
        (batch_size, num_heads, FP4_DIM), device=device, dtype=torch.uint8
    )
    q_scale = torch.empty(
        (batch_size, num_heads, SCALE_GROUPS), device=device, dtype=torch.uint8
    )
    weights_out = torch.empty(
        (batch_size, num_heads, 1), device=device, dtype=torch.float32
    )

    fused_q_indexer_rope_hadamard_fp4_quant(
        q_input,
        q_fp4,
        q_scale,
        weight,
        weights_out,
        weight_scale,
        rope_cache,
        positions,
    )
    torch.xpu.synchronize()

    reference = q_input.float()
    frequency = rope_cache[positions].view(batch_size, 1, 32, 2)
    rope = reference[..., -64:].view(batch_size, num_heads, 32, 2)
    real, imaginary = rope[..., 0], rope[..., 1]
    rotated = (
        torch.stack(
            [
                real * frequency[..., 0] - imaginary * frequency[..., 1],
                real * frequency[..., 1] + imaginary * frequency[..., 0],
            ],
            dim=-1,
        )
        .flatten(-2)
        .to(q_input.dtype)
        .float()
    )
    reference[..., -64:] = rotated
    reference = (_fwht128(reference) * (HEAD_DIM**-0.5)).to(q_input.dtype).float()
    ref_fp4, ref_scale = _ref_quantize_fp4_indexer(reference)
    ref_fp4 = ref_fp4.view(batch_size, num_heads, FP4_DIM)
    ref_scale = (
        ref_scale.view(batch_size, num_heads)
        .view(torch.uint8)
        .view(batch_size, num_heads, SCALE_GROUPS)
    )

    torch.testing.assert_close(q_fp4.cpu(), ref_fp4.cpu())
    torch.testing.assert_close(q_scale.cpu(), ref_scale.cpu())
    torch.testing.assert_close(
        weights_out.cpu(), weight.cpu().float().unsqueeze(-1) * weight_scale
    )
