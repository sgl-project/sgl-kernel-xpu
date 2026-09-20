import pytest
import torch

from sgl_kernel import fp4_paged_mqa_logits


pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available(), reason="XPU not available"
)


HEAD_DIM = 128
PACKED_DIM = 64
GROUPS = 4
PAGE_SIZE = 4


def _decode_fp4(code: torch.Tensor) -> torch.Tensor:
    magnitudes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=code.device,
        dtype=torch.float32,
    )
    value = magnitudes[(code & 0x7).long()]
    return torch.where((code & 0x8) != 0, -value, value)


def _decode_scale(scale: torch.Tensor) -> torch.Tensor:
    return torch.pow(2.0, scale.to(torch.float32) - 127.0)


def _reference(q, q_scale, cache, weights, seq_lens, block_tables, max_seq_len, weight_scale):
    batch, heads = q.shape[:2]
    output = torch.zeros((batch, max_seq_len), device=q.device, dtype=torch.float32)
    for b in range(batch):
        for token in range(int(seq_lens[b].item())):
            page = int(block_tables[b, token // PAGE_SIZE].item())
            offset = token % PAGE_SIZE
            k_token = cache[page, offset, 0]
            k_payload = k_token[:PACKED_DIM]
            k_scales = _decode_scale(k_token[PACKED_DIM : PACKED_DIM + GROUPS])
            score = 0.0
            for head in range(heads):
                q_code = torch.stack(
                    ((q[b, head] & 0x0F), (q[b, head] >> 4) & 0x0F), dim=-1
                ).reshape(-1)
                k_code = torch.stack(
                    ((k_payload & 0x0F), (k_payload >> 4) & 0x0F), dim=-1
                ).reshape(-1)
                dot = (_decode_fp4(q_code) * _decode_scale(q_scale[b, head]).repeat_interleave(32) *
                       _decode_fp4(k_code) * k_scales.repeat_interleave(32)).sum()
                score += torch.relu(dot) * weights[b, head]
            output[b, token] = score * weight_scale
    return output


@pytest.mark.parametrize("batch,heads", [(1, 2), (2, 4)])
def test_fp4_paged_mqa_logits(batch, heads):
    device = torch.device("xpu")
    torch.manual_seed(batch + heads)
    q = torch.randint(0, 256, (batch, heads, PACKED_DIM), device=device, dtype=torch.uint8)
    q_scale = torch.tensor([127, 128, 126, 127], device=device, dtype=torch.uint8).expand(
        batch, heads, GROUPS
    ).contiguous()
    cache = torch.randint(
        0, 128, (4, PAGE_SIZE, 1, PACKED_DIM + GROUPS), device=device, dtype=torch.uint8
    )
    cache[..., PACKED_DIM:] = torch.tensor(
        [127, 128, 126, 127], device=device, dtype=torch.uint8
    )
    weight_scale = 0.75
    weights = torch.randn(batch, heads, device=device, dtype=torch.float32) * weight_scale
    seq_lens = torch.tensor([5, 3][:batch], device=device, dtype=torch.int32)
    block_tables = torch.tensor(
        [[0, 1], [2, 3]][:batch], device=device, dtype=torch.int32
    )
    max_seq_len = 5
    got = fp4_paged_mqa_logits(
        (q, q_scale), cache, weights, seq_lens, block_tables, None, max_seq_len
    )
    ref = _reference(q, q_scale, cache, weights, seq_lens, block_tables, max_seq_len, 1.0)
    torch.testing.assert_close(got.cpu(), ref.cpu(), rtol=1e-4, atol=1e-4)