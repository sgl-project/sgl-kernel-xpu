from typing import Any

import torch
import torch.nn.functional as F

FP8_DTYPE = torch.float8_e4m3fn
_arange_cache = {}


def fp8_paged_mqa_logits_torch(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata: Any,
    max_seq_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Vectorized implementation compatible with CUDA graph capture."""
    _ = deep_gemm_metadata
    batch_size, _, num_heads, head_dim = q_fp8.shape
    block_size = kvcache_fp8.shape[1]

    assert head_dim == 128
    assert block_size == 64
    assert q_fp8.shape == (batch_size, 1, num_heads, head_dim)
    assert kvcache_fp8.shape[1:] == (block_size, 1, head_dim + 4)
    assert weight.shape == (batch_size, num_heads)
    assert seq_lens.shape == (batch_size,)
    assert page_table.shape[0] == batch_size
    assert clean_logits == False

    max_num_pages = page_table.shape[1]
    SCALE_OFFSET = block_size * head_dim
    total_dim = block_size * (head_dim + 4)

    kvcache_flat = kvcache_fp8.view(-1, total_dim)

    pages_clamped = page_table.clamp(min=0)
    kvcache_gathered = kvcache_flat[pages_clamped]

    kv_values_raw = kvcache_gathered[..., :SCALE_OFFSET].contiguous()
    kv_values_fp8 = kv_values_raw.view(dtype=FP8_DTYPE)
    kv_values = kv_values_fp8.to(torch.float32)
    kv_values = kv_values.reshape(batch_size, max_num_pages * block_size, head_dim)

    kv_scales_raw = kvcache_gathered[..., SCALE_OFFSET:].contiguous()
    kv_scales = kv_scales_raw.view(dtype=torch.float32)
    kv_scales = kv_scales.reshape(batch_size, max_num_pages * block_size)

    q_float = q_fp8[:, 0].to(torch.float32)
    scores = torch.bmm(kv_values, q_float.transpose(1, 2))
    scores = F.relu(scores)
    scores = scores * weight.unsqueeze(1)
    scores = scores.sum(dim=2)
    scores = scores * kv_scales

    padded_seq_len = max_num_pages * block_size
    cache = _arange_cache
    arange_key = f"arange_{padded_seq_len}_{scores.device}"
    if arange_key not in cache:
        cache[arange_key] = torch.arange(padded_seq_len, device=scores.device)
    positions = cache[arange_key].unsqueeze(0)
    valid_mask = positions < seq_lens.unsqueeze(1)
    scores = scores.masked_fill(~valid_mask, 0.0)

    if padded_seq_len < max_seq_len:
        scores = F.pad(scores, (0, max_seq_len - padded_seq_len), value=0.0)
    else:
        scores = scores[:, :max_seq_len]

    return scores
