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
        code += (absolute > threshold).to(torch.uint8)
    sign = ((x < 0) & (code != 0)).to(torch.uint8) * 8
    return code | sign


def _ref_quantize_fp4_indexer(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = x.contiguous().view(-1, HEAD_DIM).float()
    groups = values.view(-1, SCALE_GROUPS, GROUP_SIZE)
    scale_raw = (groups.abs().amax(dim=-1) / E2M1_MAX).clamp_min(1.0e-4)
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
def test_fp4_fused_q_indexer_rope_hadamard_quant(batch_size: int):
    device = torch.device("xpu")
    torch.manual_seed(batch_size + 200)
    num_heads = 8
    weight_scale = HEAD_DIM**-0.5 * num_heads**-0.5
    q_input = torch.randn(
        batch_size, num_heads, HEAD_DIM, device=device, dtype=torch.bfloat16
    )
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
    rotated = torch.stack(
        [
            real * frequency[..., 0] - imaginary * frequency[..., 1],
            real * frequency[..., 1] + imaginary * frequency[..., 0],
        ],
        dim=-1,
    ).flatten(-2)
    reference[..., -64:] = rotated
    reference = _fwht128(reference) * (HEAD_DIM**-0.5)
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
