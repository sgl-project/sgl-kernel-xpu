from typing import Any

import torch
import triton
import triton.language as tl

FP8_DTYPE = torch.float8_e4m3fn
_MIN_PROGRAMS = 2048
_MAX_PAGES_PER_PROG = 32


@triton.jit
def _paged_mqa_logits_kernel(
    q_ptr,
    kv_value_ptr,
    kv_scale_ptr,
    weight_ptr,
    seq_lens_ptr,
    page_table_ptr,
    logits_ptr,
    num_pages,
    num_pages_req,
    max_seq_len,
    q_stride_b,
    q_stride_h,
    kv_value_stride_p,
    kv_scale_stride_p,
    weight_stride_b,
    page_table_stride_b,
    logits_stride_b,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SZ: tl.constexpr,
    BLOCK_H: tl.constexpr,
    SCALE_COL: tl.constexpr,
    PAGES_PER_PROG: tl.constexpr,
):
    """logits[b, s] = kv_scale[b, s] * sum_h relu(q[b, h] . k[b, s]) * weight[b, h]

    One program per (batch row, page block). BLOCK_SZ equals the KV page size, so
    each iteration covers exactly one page and the gather is a single base offset.
    q is loaded once per program and reused across the page loop.

    The QK product is a bf16 tl.dot with an fp32 accumulator. e4m3 widens
    exactly to bf16, so the operands are unchanged; the reduction order differs
    from an eager fp32 matmul.
    """
    pid_b = tl.program_id(0)
    pid_page_block = tl.program_id(1)

    s_local = tl.arange(0, BLOCK_SZ)
    d_offs = tl.arange(0, HEAD_DIM)
    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < NUM_HEADS

    q_t_ptrs = (
        q_ptr + pid_b * q_stride_b + h_offs[None, :] * q_stride_h + d_offs[:, None]
    )
    q_t = tl.load(q_t_ptrs, mask=h_mask[None, :], other=0.0).to(tl.bfloat16)

    weight = tl.load(
        weight_ptr + pid_b * weight_stride_b + h_offs, mask=h_mask, other=0.0
    ).to(tl.float32)
    seq_len = tl.load(seq_lens_ptr + pid_b).to(tl.int32)

    for i in tl.static_range(PAGES_PER_PROG):
        pid_page = pid_page_block * PAGES_PER_PROG + i
        if pid_page < num_pages_req:
            page = tl.load(
                page_table_ptr + pid_b * page_table_stride_b + pid_page
            ).to(tl.int32)
            page = tl.minimum(tl.maximum(page, 0), num_pages - 1)

            k = tl.load(
                kv_value_ptr
                + page * kv_value_stride_p
                + s_local[:, None] * HEAD_DIM
                + d_offs[None, :]
            ).to(tl.bfloat16)

            score = tl.maximum(tl.dot(k, q_t), 0.0)
            acc = tl.sum(score * weight[None, :], axis=1)

            scale = tl.load(
                kv_scale_ptr + page * kv_scale_stride_p + SCALE_COL + s_local
            )
            acc = acc * scale

            s_global = pid_page * BLOCK_SZ + s_local
            valid = (s_global < seq_len) & (s_global < max_seq_len)
            tl.store(
                logits_ptr + pid_b * logits_stride_b + s_global,
                tl.where(valid, acc, 0.0),
                mask=s_global < max_seq_len,
            )


def _pages_per_prog(batch_size: int, num_pages_req: int) -> int:
    pages_per_prog = _MAX_PAGES_PER_PROG
    if batch_size * num_pages_req < _MIN_PROGRAMS * pages_per_prog:
        pages_per_prog = max(1, (batch_size * num_pages_req) // _MIN_PROGRAMS)
        pages_per_prog = 1 << (pages_per_prog.bit_length() - 1)
    return pages_per_prog


def fp8_paged_mqa_logits_triton(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata: Any,
    max_seq_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Triton fp8_paged_mqa_logits.

    The paged gather, the fp8 dequantization, the QK product and the
    relu/weight/head-reduction/scale/mask pipeline all run in one kernel. The
    product is taken in bf16 so it maps to the XPU matrix engines; keeping it in
    fp32 leaves it on the scalar units, since Xe2 has no fp32 DPAS path.
    """
    _ = deep_gemm_metadata
    batch_size, _, num_heads, head_dim = q_fp8.shape
    block_size = kvcache_fp8.shape[1]

    assert head_dim == 128
    assert block_size == 64
    assert q_fp8.shape == (batch_size, 1, num_heads, head_dim)
    assert kvcache_fp8.shape[1:] == (block_size, 1, head_dim + 4)
    assert weight.shape == (batch_size, num_heads)
    if seq_lens.dim() > 1:
        seq_lens = seq_lens.squeeze(-1)
    assert seq_lens.shape == (batch_size,)
    assert page_table.shape[0] == batch_size
    assert clean_logits == False

    head_dim_with_sf = head_dim + 4

    max_pages_eff = (max_seq_len + block_size - 1) // block_size
    num_pages_req = min(page_table.shape[1], max_pages_eff)
    padded_seq_len = num_pages_req * block_size

    logits = page_table.new_empty((batch_size, max_seq_len), dtype=torch.float32)

    kv_bytes = kvcache_fp8.reshape(kvcache_fp8.shape[0], -1)
    kv_value = kv_bytes.view(FP8_DTYPE)
    kv_scale = kv_bytes.view(torch.float32)

    q = q_fp8[:, 0]
    if not q.is_contiguous():
        q = q.contiguous()
    if not page_table.is_contiguous():
        page_table = page_table.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if not seq_lens.is_contiguous():
        seq_lens = seq_lens.contiguous()

    pages_per_prog = _pages_per_prog(batch_size, num_pages_req)
    grid = (batch_size, triton.cdiv(num_pages_req, pages_per_prog))

    _paged_mqa_logits_kernel[grid](
        q,
        kv_value,
        kv_scale,
        weight,
        seq_lens,
        page_table,
        logits,
        kvcache_fp8.shape[0],
        num_pages_req,
        max_seq_len,
        q.stride(0),
        q.stride(1),
        kv_value.stride(0),
        kv_scale.stride(0),
        weight.stride(0),
        page_table.stride(0),
        logits.stride(0),
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        BLOCK_SZ=block_size,
        BLOCK_H=max(16, triton.next_power_of_2(num_heads)),
        SCALE_COL=(block_size * head_dim) // 4,
        PAGES_PER_PROG=pages_per_prog,
        num_warps=8,
        num_stages=4,
    )

    if padded_seq_len < max_seq_len:
        logits[:, padded_seq_len:] = 0

    return logits
