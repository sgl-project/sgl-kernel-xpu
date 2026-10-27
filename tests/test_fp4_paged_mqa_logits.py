import pytest
import torch
from sgl_kernel import fp4_paged_mqa_logits

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available(), reason="XPU not available"
)


HEAD_DIM = 128
PACKED_DIM = 64
GROUPS = 4
PAGE_SIZE = 64


def _decode_fp4(code: torch.Tensor) -> torch.Tensor:
    code = code.to(torch.int16)
    code = torch.stack((code & 0xF, (code >> 4) & 0xF), dim=-1).flatten(-2)
    magnitudes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=code.device,
        dtype=torch.float32,
    )
    value = magnitudes[(code & 0x7).long()]
    return torch.where((code & 0x8) != 0, -value, value)


def _decode_scale(scale: torch.Tensor) -> torch.Tensor:
    return torch.pow(2.0, scale.to(torch.float32) - 127.0)


def _reference(q, q_scale, cache, weights, seq_lens, block_tables, max_seq_len):
    batch, heads = q.shape[:2]
    device = q.device
    cache_flat = cache.reshape(cache.shape[0], -1)
    positions = torch.arange(max_seq_len, device=device, dtype=torch.int64)
    page_ids = block_tables[:, positions // PAGE_SIZE].long()
    offsets = positions % PAGE_SIZE

    payload_indices = offsets.view(1, -1, 1) * PACKED_DIM + torch.arange(
        PACKED_DIM, device=device
    ).view(1, 1, -1)
    k_packed = cache_flat[
        page_ids.unsqueeze(-1),
        payload_indices.expand(batch, -1, -1),
    ]
    scale_indices = (
        PAGE_SIZE * PACKED_DIM
        + offsets.view(1, -1, 1) * GROUPS
        + torch.arange(GROUPS, device=device).view(1, 1, -1)
    )
    k_scales = cache_flat[
        page_ids.unsqueeze(-1),
        scale_indices.expand(batch, -1, -1),
    ]

    q_values = _decode_fp4(q).float()
    k_values = _decode_fp4(k_packed).float()
    q_scale_values = _decode_scale(q_scale).repeat_interleave(32, dim=-1)
    k_scale_values = _decode_scale(k_scales).repeat_interleave(32, dim=-1)
    dots = (
        q_values.unsqueeze(2)
        * q_scale_values.unsqueeze(2)
        * k_values.unsqueeze(1)
        * k_scale_values.unsqueeze(1)
    ).sum(dim=-1)
    logits = (torch.relu(dots) * weights.unsqueeze(-1)).sum(dim=1)
    valid = positions.unsqueeze(0) < seq_lens.unsqueeze(1)
    return torch.where(valid, logits, torch.zeros_like(logits))


@pytest.mark.parametrize("batch", [1, 6, 13])
@pytest.mark.parametrize("heads", [2, 8, 32, 64])
@pytest.mark.parametrize("max_seq_len", [1, 63, 64, 65, 129, 512])
@pytest.mark.parametrize("packed_query_layout", [False, True])
def test_fp4_paged_mqa_logits(batch, heads, max_seq_len, packed_query_layout):
    device = torch.device("xpu")
    torch.manual_seed(batch * 1000 + heads * 10 + max_seq_len)
    pages_per_batch = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    num_pages = batch * pages_per_batch
    q = torch.randint(
        0, 256, (batch, heads, PACKED_DIM), device=device, dtype=torch.uint8
    )
    q_scale = torch.randint(
        126, 129, (batch, heads, GROUPS), device=device, dtype=torch.uint8
    )
    cache = torch.randint(
        0,
        256,
        (num_pages, PAGE_SIZE, 1, PACKED_DIM + GROUPS),
        device=device,
        dtype=torch.uint8,
    )
    # The cache stores all packed payloads before per-token group scales.
    cache.view(num_pages, -1)[:, PAGE_SIZE * PACKED_DIM :] = torch.randint(
        126,
        129,
        (num_pages, PAGE_SIZE * GROUPS),
        device=device,
        dtype=torch.uint8,
    )
    weights = torch.randint(-8, 9, (batch, heads), device=device).to(torch.float32) / 8
    lengths = [0, 1, 63, 64, 65, max_seq_len - 1, max_seq_len]
    seq_lens = torch.tensor(
        [min(max_seq_len, lengths[i % len(lengths)]) for i in range(batch)],
        device=device,
        dtype=torch.int32,
    )
    seq_lens[-1] = max_seq_len
    page_order = torch.randperm(num_pages, device=device, dtype=torch.int32)
    block_tables = page_order.view(batch, pages_per_batch)
    if packed_query_layout:
        q_for_kernel = q.unsqueeze(1)
        q_scale_for_kernel = (
            q_scale.contiguous().view(torch.int32).squeeze(-1).unsqueeze(1)
        )
    else:
        q_for_kernel = q
        q_scale_for_kernel = q_scale
    got = fp4_paged_mqa_logits(
        (q_for_kernel, q_scale_for_kernel),
        cache,
        weights,
        seq_lens,
        block_tables,
        None,
        max_seq_len,
    )
    ref = _reference(q, q_scale, cache, weights, seq_lens, block_tables, max_seq_len)
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)
