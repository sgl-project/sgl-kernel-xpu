"""Correctness tests for sparse MLA fp8 decode.

Run: pytest tests/test_flash_mla_sparse_decode.py
"""

import pytest
import torch
from sgl_kernel import flash_mla_sparse_decode

NOPE_DIM = 448
ROPE_DIM = 64
D_QK = NOPE_DIM + ROPE_DIM
D_V = 512
DATA_BYTES_PER_TOKEN = NOPE_DIM + ROPE_DIM * 2
SCALE_BYTES_PER_TOKEN = 8
HEAD_BYTES = DATA_BYTES_PER_TOKEN + SCALE_BYTES_PER_TOKEN

default_params = {
    "is_fp8_query": [False, True],
    "has_attn_sink": [False, True],
    "has_extra": [False, True],
    "extra_topk": [1024],
    "topk": [6, 512],
    "h_q": [6, 96],
    "s_q": [1, 128],
}


def _pack_sparse_fp8_kv_deepseek_v4(kv: torch.Tensor):
    """Pack logical bf16 KV into sparse fp8 cache layout.

    Physical block layout is:
      [block_size * 576B token data][block_size * 8B token scales]

    The returned 4D tensor is an as_strided view over that physical storage so
    the extension can still receive shape [num_blocks, block_size, 1, 584].
    """
    num_blocks, block_size, num_heads, d_qk = kv.shape
    assert num_heads == 1
    assert d_qk == D_QK

    packed_storage = torch.empty(
        (num_blocks, block_size * HEAD_BYTES), device=kv.device, dtype=torch.uint8
    )
    dequant = torch.empty_like(kv, dtype=torch.float32)

    nope = kv[:, :, 0, :NOPE_DIM].float()
    rope = kv[:, :, 0, NOPE_DIM:].contiguous()
    scale_bytes = torch.full(
        (num_blocks, block_size, SCALE_BYTES_PER_TOKEN),
        127,
        device=kv.device,
        dtype=torch.uint8,
    )
    scale_bytes[..., 7] = 0

    nope_fp8 = nope.to(torch.float8_e4m3fn)
    dequant[:, :, 0, :NOPE_DIM] = nope_fp8.float()
    dequant[:, :, 0, NOPE_DIM:] = rope.float()

    data_region = packed_storage[:, : block_size * DATA_BYTES_PER_TOKEN].view(
        num_blocks, block_size, DATA_BYTES_PER_TOKEN
    )
    scale_region = packed_storage[:, block_size * DATA_BYTES_PER_TOKEN :].view(
        num_blocks, block_size, SCALE_BYTES_PER_TOKEN
    )

    data_region[..., :NOPE_DIM] = nope_fp8.contiguous().view(torch.uint8)
    data_region[..., NOPE_DIM:] = rope.view(torch.uint8).view(
        num_blocks, block_size, ROPE_DIM * 2
    )
    scale_region.copy_(scale_bytes)

    packed = packed_storage.as_strided(
        (num_blocks, block_size, num_heads, HEAD_BYTES),
        (block_size * HEAD_BYTES, DATA_BYTES_PER_TOKEN, HEAD_BYTES, 1),
    ).view(torch.float8_e4m3fn)
    return packed, dequant


def _reference_sparse_decode(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    topk_length,
    attn_sink,
    extra_kv=None,
    extra_indices=None,
    extra_topk_length=None,
):
    b, s_q, h_q, d_qk = q.shape
    assert d_qk == D_QK

    q_f = q.float()
    kv_flat = kv.view(-1, 1, D_QK).squeeze(1).float()
    all_logits = []
    all_values = []
    all_valid = []

    def normalize_topk_length(active_topk_length: torch.Tensor) -> torch.Tensor:
        if active_topk_length.dim() == 1:
            assert active_topk_length.shape == (b,)
            return active_topk_length.view(b, 1, 1)
        assert active_topk_length.shape == (b, s_q)
        return active_topk_length.unsqueeze(-1)

    def append_range(active_kv, active_indices, active_topk_length):
        active_flat = active_kv.view(-1, 1, D_QK).squeeze(1).float()
        active_indices = active_indices.clone()
        topk = active_indices.shape[-1]
        if active_topk_length is not None:
            normalized_topk_length = normalize_topk_length(active_topk_length)
            mask = (
                torch.arange(topk, device=q.device).view(1, 1, topk)
                >= normalized_topk_length
            )
            active_indices[mask.expand_as(active_indices)] = -1

        valid = (active_indices >= 0) & (active_indices < active_flat.shape[0])
        gather_indices = active_indices.masked_fill(~valid, 0)
        gathered = active_flat.index_select(0, gather_indices.reshape(-1)).view(
            b, s_q, topk, D_QK
        )
        logits = torch.einsum("bshd,bskd->bshk", q_f, gathered) * sm_scale
        logits = logits.masked_fill(~valid.unsqueeze(2), float("-inf"))
        all_logits.append(logits)
        all_values.append(gathered[..., :D_V])
        all_valid.append(valid)

    append_range(kv_flat.view_as(kv), indices, topk_length)
    if extra_kv is not None:
        assert extra_indices is not None
        append_range(extra_kv, extra_indices, extra_topk_length)

    logits = torch.cat(all_logits, dim=-1)
    values = torch.cat(all_values, dim=2)
    lse = torch.logsumexp(logits, dim=-1)
    lse_for_out = lse
    if attn_sink is not None:
        lse_for_out = torch.logsumexp(
            torch.stack([lse, attn_sink.view(1, 1, h_q).expand_as(lse)], dim=0),
            dim=0,
        )
    lse_for_out = lse_for_out.masked_fill(lse_for_out == float("-inf"), float("inf"))
    probs = torch.exp(logits - lse_for_out.unsqueeze(-1))
    out = torch.einsum("bshk,bskd->bshd", probs, values)
    lse = lse.masked_fill(lse == float("-inf"), float("inf"))
    return out.to(torch.bfloat16), lse


@pytest.mark.parametrize("is_fp8_query", default_params["is_fp8_query"])
@pytest.mark.parametrize("has_attn_sink", default_params["has_attn_sink"])
@pytest.mark.parametrize("has_extra", default_params["has_extra"])
@pytest.mark.parametrize("extra_topk", default_params["extra_topk"])
@pytest.mark.parametrize("topk", default_params["topk"])
@pytest.mark.parametrize("h_q", default_params["h_q"])
@pytest.mark.parametrize("s_q", default_params["s_q"])
def test_flash_mla_sparse_decode_fp8_kvcache(
    s_q, h_q, topk, extra_topk, is_fp8_query, has_attn_sink, has_extra
):
    device = "xpu"
    torch.manual_seed(1234)

    # simulate vllm behavior: batch = s_q, single decode query per batch entry.
    b = s_q
    s_q = 1
    d_qk = D_QK
    num_blocks, block_size = 128, 64
    extra_num_blocks, extra_block_size = 256, 64
    sm_scale = d_qk**-0.5

    q_ref = torch.randn((b, s_q, h_q, d_qk), device=device, dtype=torch.float32) * 0.5
    q_scale = None
    if is_fp8_query:
        q_scale = torch.tensor(0.25, device=device, dtype=torch.float32)
        q = (q_ref / q_scale).to(torch.float8_e4m3fn)
        q_for_ref = q.float() * q_scale
    else:
        q = q_ref.to(torch.bfloat16)
        q_for_ref = q.float()

    logical_kv = (
        torch.randn(
            (num_blocks, block_size, 1, d_qk), device=device, dtype=torch.bfloat16
        )
        * 0.5
    )
    packed_kv, dequant_kv = _pack_sparse_fp8_kv_deepseek_v4(logical_kv)

    indices = torch.randint(
        0, num_blocks * block_size, (b, s_q, topk), device=device, dtype=torch.int32
    )
    indices[0, 0, -1] = num_blocks * block_size + 11
    topk_length = torch.randint(1, topk + 1, (b,), device=device, dtype=torch.int32)

    attn_sink = None
    if has_attn_sink:
        attn_sink = torch.randn((h_q,), device=device, dtype=torch.float32) * 0.25

    packed_extra_kv = None
    dequant_extra_kv = None
    extra_indices = None
    extra_topk_length = None
    if has_extra:
        logical_extra_kv = (
            torch.randn(
                (extra_num_blocks, extra_block_size, 1, d_qk),
                device=device,
                dtype=torch.bfloat16,
            )
            * 0.5
        )
        packed_extra_kv, dequant_extra_kv = _pack_sparse_fp8_kv_deepseek_v4(
            logical_extra_kv
        )
        extra_indices = torch.randint(
            0,
            extra_num_blocks * extra_block_size,
            (b, s_q, extra_topk),
            device=device,
            dtype=torch.int32,
        )
        extra_topk_length = torch.randint(
            1, extra_topk + 1, (b,), device=device, dtype=torch.int32
        )

    out, lse = flash_mla_sparse_decode(
        q,
        packed_kv,
        indices,
        sm_scale=sm_scale,
        d_v=D_V,
        topk_length=topk_length,
        attn_sink=attn_sink,
        extra_kv=packed_extra_kv,
        extra_indices=extra_indices,
        extra_topk_length=extra_topk_length,
        q_scale=q_scale,
        is_fp8_query=is_fp8_query,
        return_softmax_lse=True,
    )

    ref_out, ref_lse = _reference_sparse_decode(
        q_for_ref,
        dequant_kv,
        indices,
        sm_scale,
        topk_length,
        attn_sink,
        dequant_extra_kv,
        extra_indices,
        extra_topk_length,
    )

    assert out.shape == (b, s_q, h_q, D_V)
    assert lse is not None
    assert lse.shape == (b, s_q, h_q)
    torch.testing.assert_close(lse, ref_lse, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
