"""Sparse attention with vertical+slash sparsity patterns for DeepSeek V3.

Supports both native CUDA/SYCL kernels (when available) and a pure PyTorch
fallback that works on any device (CPU, XPU, CUDA). The vertical+slash
pattern is described in Appendix C.4.2 of https://arxiv.org/abs/2407.02490.

The sparsity pattern consists of:
- **Vertical (column)** patterns: specific KV columns attended to by all queries.
- **Slash (diagonal block)** patterns: contiguous KV blocks derived from
  diagonal stripes in the attention matrix.
"""

import math
from typing import Optional, Tuple

import torch


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _has_native_sparse_ops() -> bool:
    """Check if native sparse attention ops are available."""
    try:
        torch.ops.sgl_kernel.fwd_sparse  # noqa: B018
        return True
    except (AttributeError, RuntimeError):
        return False


def _has_native_convert_ops() -> bool:
    """Check if native convert_vertical_slash_indexes ops are available."""
    try:
        torch.ops.sgl_kernel.convert_vertical_slash_indexes  # noqa: B018
        return True
    except (AttributeError, RuntimeError):
        return False


# ---------------------------------------------------------------------------
# Pure PyTorch implementation of convert_vertical_slash_indexes
# ---------------------------------------------------------------------------


def _save_blocks(
    block_offset, b, h, row, range_start, range_end, block_size_N, kv_len, blk_cnt
):
    """Save block-aligned offsets within [range_start, range_end)."""
    idx = range_start
    while idx < range_end and idx < kv_len:
        block_offset[b, h, row, blk_cnt] = idx
        blk_cnt += 1
        idx += block_size_N
    return blk_cnt


def _convert_vertical_slash_indexes_torch(
    q_seqlens,
    kv_seqlens,
    vertical_indexes,
    slash_indexes,
    context_size,
    block_size_M,
    block_size_N,
    causal=True,
):
    """Pure PyTorch implementation of convert_vertical_slash_indexes.

    Mirrors the CUDA kernel two-pointer merge algorithm:
    1. For each query block row, convert slash diagonal indices into KV block
       ranges (block_count / block_offset).
    2. Vertical column indices that fall outside any slash block range are
       stored separately (column_count / column_index).
    """
    B = slash_indexes.size(0)
    H = slash_indexes.size(1)
    NNZ_S = slash_indexes.size(2)
    NNZ_V = vertical_indexes.size(2)
    num_rows = (context_size + block_size_M - 1) // block_size_M

    device = q_seqlens.device
    dtype = q_seqlens.dtype

    block_count = torch.zeros(B, H, num_rows, dtype=dtype, device=device)
    block_offset = torch.zeros(B, H, num_rows, NNZ_S, dtype=dtype, device=device)
    column_count = torch.zeros(B, H, num_rows, dtype=dtype, device=device)
    column_index = torch.zeros(B, H, num_rows, NNZ_V, dtype=dtype, device=device)

    for b in range(B):
        q_len = q_seqlens[b].item()
        kv_len = kv_seqlens[b].item()

        for h in range(H):
            v_list = vertical_indexes[b, h].tolist()  # sorted ascending
            s_list = slash_indexes[b, h].tolist()  # sorted descending

            for row in range(num_rows):
                start_m = row * block_size_M
                end_m = start_m + block_size_M

                # --- Phase 1: find the first valid slash for this row ---
                s_ptr = 0
                has_slash = NNZ_S > 0
                s_val = s_list[0] if has_slash else 0

                if causal and has_slash:
                    # Skip slashes whose diagonal is beyond the causal boundary
                    causal_limit = end_m + (kv_len - q_len)
                    while s_val >= causal_limit and s_ptr + 1 < NNZ_S:
                        s_ptr += 1
                        s_val = s_list[s_ptr]
                    if s_val >= causal_limit:
                        has_slash = False
                s_ptr += 1  # consumed one slash index

                if has_slash:
                    if causal:
                        s_conv = max((kv_len - q_len) + end_m - s_val, block_size_M)
                    else:
                        s_conv = kv_len + end_m - s_val
                    range_start = s_conv - block_size_M
                    range_end = s_conv
                else:
                    range_start = 0
                    range_end = 0

                tmp_blk_cnt = 0
                tmp_col_cnt = 0
                v_ptr = 0

                # --- Phase 2: two-pointer merge scan ---
                if has_slash:
                    while True:
                        if v_ptr < NNZ_V and v_list[v_ptr] < range_end:
                            v_idx = v_list[v_ptr]
                            if v_idx < range_start:
                                # Vertical column before the current slash block
                                if not (causal and v_idx >= end_m + (kv_len - q_len)):
                                    column_index[b, h, row, tmp_col_cnt] = v_idx
                                    tmp_col_cnt += 1
                            # else: inside block range, implicitly covered
                            v_ptr += 1
                        else:
                            # Advance slash or flush
                            got_next = False
                            while s_ptr < NNZ_S:
                                next_s = s_list[s_ptr]
                                s_ptr += 1
                                if causal and next_s >= end_m + (kv_len - q_len):
                                    continue
                                if causal:
                                    next_conv = max(
                                        (kv_len - q_len) + end_m - next_s,
                                        block_size_M,
                                    )
                                else:
                                    next_conv = kv_len + end_m - next_s

                                if next_conv > range_end + block_size_M:
                                    # Gap: flush current range, start new
                                    tmp_blk_cnt = _save_blocks(
                                        block_offset,
                                        b,
                                        h,
                                        row,
                                        range_start,
                                        range_end,
                                        block_size_N,
                                        kv_len,
                                        tmp_blk_cnt,
                                    )
                                    range_start = next_conv - block_size_M
                                    range_end = next_conv
                                elif next_conv > range_end:
                                    range_end += block_size_M
                                got_next = True
                                break

                            if not got_next:
                                # No more slashes: flush and collect rest
                                tmp_blk_cnt = _save_blocks(
                                    block_offset,
                                    b,
                                    h,
                                    row,
                                    range_start,
                                    range_end,
                                    block_size_N,
                                    kv_len,
                                    tmp_blk_cnt,
                                )
                                while v_ptr < NNZ_V:
                                    v_idx = v_list[v_ptr]
                                    v_ptr += 1
                                    if causal and v_idx >= end_m - 1 + (kv_len - q_len):
                                        break
                                    if v_idx >= range_end:
                                        column_index[b, h, row, tmp_col_cnt] = v_idx
                                        tmp_col_cnt += 1
                                break
                else:
                    # No slashes: all verticals become columns
                    for vi in range(NNZ_V):
                        v_idx = v_list[vi]
                        if causal and v_idx >= end_m - 1 + (kv_len - q_len):
                            break
                        if 0 <= v_idx < kv_len:
                            column_index[b, h, row, tmp_col_cnt] = v_idx
                            tmp_col_cnt += 1

                block_count[b, h, row] = tmp_blk_cnt
                column_count[b, h, row] = tmp_col_cnt

    return block_count, block_offset, column_count, column_index


def _convert_vertical_slash_indexes_mergehead_torch(
    q_seqlens,
    kv_seqlens,
    vertical_indexes,
    slash_indexes,
    vertical_indices_count,
    slash_indices_count,
    context_size,
    block_size_M,
    block_size_N,
    causal=True,
):
    """Pure PyTorch implementation of convert_vertical_slash_indexes_mergehead.

    Same algorithm as the standard version, but each head uses only
    ``vertical_indices_count[h]`` / ``slash_indices_count[h]`` entries from
    the respective index arrays.
    """
    B = slash_indexes.size(0)
    H = slash_indexes.size(1)
    NNZ_S = slash_indexes.size(2)
    NNZ_V = vertical_indexes.size(2)
    num_rows = (context_size + block_size_M - 1) // block_size_M

    device = q_seqlens.device
    dtype = q_seqlens.dtype

    block_count = torch.empty(B, H, num_rows, dtype=dtype, device=device)
    block_offset = torch.empty(B, H, num_rows, NNZ_S, dtype=dtype, device=device)
    column_count = torch.empty(B, H, num_rows, dtype=dtype, device=device)
    column_index = torch.empty(B, H, num_rows, NNZ_V, dtype=dtype, device=device)

    for b in range(B):
        q_len = q_seqlens[b].item()
        kv_len = kv_seqlens[b].item()

        for h in range(H):
            nnz_v_h = vertical_indices_count[h].item()
            nnz_s_h = slash_indices_count[h].item()
            v_list = vertical_indexes[b, h, :nnz_v_h].tolist()
            s_list = slash_indexes[b, h, :nnz_s_h].tolist()

            for row in range(num_rows):
                start_m = row * block_size_M
                end_m = start_m + block_size_M

                s_ptr = 0
                has_slash = nnz_s_h > 0
                s_val = s_list[0] if has_slash else 0

                if causal and has_slash:
                    causal_limit = end_m + (kv_len - q_len)
                    while s_val >= causal_limit and s_ptr + 1 < nnz_s_h:
                        s_ptr += 1
                        s_val = s_list[s_ptr]
                    if s_val >= causal_limit:
                        has_slash = False
                s_ptr += 1

                if has_slash:
                    if causal:
                        s_conv = max((kv_len - q_len) + end_m - s_val, block_size_M)
                    else:
                        s_conv = kv_len + end_m - s_val
                    range_start = s_conv - block_size_M
                    range_end = s_conv
                else:
                    range_start = 0
                    range_end = 0

                tmp_blk_cnt = 0
                tmp_col_cnt = 0
                v_ptr = 0

                if has_slash:
                    while True:
                        if v_ptr < nnz_v_h and v_list[v_ptr] < range_end:
                            v_idx = v_list[v_ptr]
                            if v_idx < range_start:
                                if not (causal and v_idx >= end_m + (kv_len - q_len)):
                                    column_index[b, h, row, tmp_col_cnt] = v_idx
                                    tmp_col_cnt += 1
                            v_ptr += 1
                        else:
                            got_next = False
                            while s_ptr < nnz_s_h:
                                next_s = s_list[s_ptr]
                                s_ptr += 1
                                if causal and next_s >= end_m + (kv_len - q_len):
                                    continue
                                if causal:
                                    next_conv = max(
                                        (kv_len - q_len) + end_m - next_s,
                                        block_size_M,
                                    )
                                else:
                                    next_conv = kv_len + end_m - next_s
                                if next_conv > range_end + block_size_M:
                                    tmp_blk_cnt = _save_blocks(
                                        block_offset,
                                        b,
                                        h,
                                        row,
                                        range_start,
                                        range_end,
                                        block_size_N,
                                        kv_len,
                                        tmp_blk_cnt,
                                    )
                                    range_start = next_conv - block_size_M
                                    range_end = next_conv
                                elif next_conv > range_end:
                                    range_end += block_size_M
                                got_next = True
                                break
                            if not got_next:
                                tmp_blk_cnt = _save_blocks(
                                    block_offset,
                                    b,
                                    h,
                                    row,
                                    range_start,
                                    range_end,
                                    block_size_N,
                                    kv_len,
                                    tmp_blk_cnt,
                                )
                                while v_ptr < nnz_v_h:
                                    v_idx = v_list[v_ptr]
                                    v_ptr += 1
                                    if causal and v_idx >= end_m - 1 + (kv_len - q_len):
                                        break
                                    if v_idx >= range_end:
                                        column_index[b, h, row, tmp_col_cnt] = v_idx
                                        tmp_col_cnt += 1
                                break
                else:
                    for vi in range(nnz_v_h):
                        v_idx = v_list[vi]
                        if causal and v_idx >= end_m - 1 + (kv_len - q_len):
                            break
                        if 0 <= v_idx < kv_len:
                            column_index[b, h, row, tmp_col_cnt] = v_idx
                            tmp_col_cnt += 1

                block_count[b, h, row] = tmp_blk_cnt
                column_count[b, h, row] = tmp_col_cnt

    return block_count, block_offset, column_count, column_index


# ---------------------------------------------------------------------------
# Pure PyTorch implementation of sparse attention forward
# ---------------------------------------------------------------------------


def _sparse_attn_func_torch(
    q,
    k,
    v,
    block_count,
    block_offset,
    column_count,
    column_index,
    softmax_scale,
    causal=False,
    softcap=0.0,
    block_size_M=64,
    block_size_N=64,
):
    """Compute sparse attention using pure PyTorch operations.

    This is a reference / fallback implementation that works on any device.
    The inner operations (matmul, softmax, scatter) are fully vectorized;
    the only Python-level loop is over query block rows.

    Args:
        q: (batch_size, seqlen_q, nheads, headdim)
        k: (batch_size, seqlen_k, nheads_k, headdim)
        v: (batch_size, seqlen_k, nheads_k, headdim)
        block_count: (batch_size, nheads, num_rows)
        block_offset: (batch_size, nheads, num_rows, NNZ_S)
        column_count: (batch_size, nheads, num_rows)
        column_index: (batch_size, nheads, num_rows, NNZ_V)
        softmax_scale: float
        causal: bool
        softcap: float (0.0 = deactivated)
        block_size_M: int, query block size
        block_size_N: int, key block size

    Returns:
        out: (batch_size, seqlen_q, nheads, headdim)
        softmax_lse: (batch_size, nheads, seqlen_q)
    """
    B, Sq, Hq, D = q.shape
    _, Sk, Hk, _ = k.shape

    # Handle GQA: expand K/V heads to match Q heads
    ngroups = Hq // Hk
    if ngroups > 1:
        k = k[:, :, :, None, :].expand(B, Sk, Hk, ngroups, D).reshape(B, Sk, Hq, D)
        v = v[:, :, :, None, :].expand(B, Sk, Hk, ngroups, D).reshape(B, Sk, Hq, D)

    num_rows = block_count.shape[2]
    NNZ_S = block_offset.shape[3] if block_offset.dim() > 3 else 0
    NNZ_V = column_index.shape[3] if column_index.dim() > 3 else 0
    device = q.device

    # Transpose to (B, H, S, D), compute in float32 for numerical stability
    q_f = q.transpose(1, 2).float()
    k_f = k.transpose(1, 2).float()
    v_f = v.transpose(1, 2).float()

    out = torch.zeros_like(q_f)
    lse = torch.full((B, Hq, Sq), float("-inf"), device=device, dtype=torch.float32)

    for row_idx in range(num_rows):
        q_start = row_idx * block_size_M
        q_end = min(q_start + block_size_M, Sq)
        q_block = q_f[:, :, q_start:q_end, :]  # (B, H, bm, D)

        # ----- Build KV mask for this row: (B, H, Sk) -----
        mask = torch.zeros(B, Hq, Sk, dtype=torch.bool, device=device)

        # Slash blocks (vectorized)
        if NNZ_S > 0:
            bc = block_count[:, :, row_idx]  # (B, H)
            bo = block_offset[:, :, row_idx, :]  # (B, H, NNZ_S)
            kv_range = torch.arange(block_size_N, device=device)  # (BN,)

            # (B, H, NNZ_S, BN)
            bo_exp = bo.unsqueeze(-1).long() + kv_range
            j_range = torch.arange(NNZ_S, device=device)
            valid_j = j_range < bc.unsqueeze(-1)  # (B, H, NNZ_S)
            valid_kv = (bo_exp >= 0) & (bo_exp < Sk)
            valid = valid_j.unsqueeze(-1) & valid_kv  # (B, H, NNZ_S, BN)

            bo_flat = bo_exp.reshape(B, Hq, -1).clamp(0, Sk - 1)
            valid_flat = valid.reshape(B, Hq, -1)
            mask.scatter_(-1, bo_flat, valid_flat)

        # Vertical columns (vectorized)
        if NNZ_V > 0:
            cc = column_count[:, :, row_idx]  # (B, H)
            ci = column_index[:, :, row_idx, :].long()  # (B, H, NNZ_V)

            j_range_v = torch.arange(NNZ_V, device=device)
            valid_j_v = j_range_v < cc.unsqueeze(-1)  # (B, H, NNZ_V)
            valid_kv_v = (ci >= 0) & (ci < Sk)
            valid_v = valid_j_v & valid_kv_v

            ci_clamped = ci.clamp(0, Sk - 1)
            mask.scatter_(-1, ci_clamped, valid_v)

        # ----- Compute attention scores -----
        scores = (
            torch.matmul(q_block, k_f.transpose(-2, -1)) * softmax_scale
        )  # (B, H, bm, Sk)

        if softcap > 0:
            scores = torch.tanh(scores / softcap) * softcap

        # Apply causal mask
        if causal:
            q_pos = torch.arange(q_start, q_end, device=device).unsqueeze(1)
            k_pos = torch.arange(Sk, device=device).unsqueeze(0)
            causal_mask = (q_pos + Sk - Sq) >= k_pos
            scores = scores.masked_fill(
                ~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        # Apply sparse mask
        scores = scores.masked_fill(~mask.unsqueeze(2), float("-inf"))

        # Softmax and weighted sum
        block_lse = scores.logsumexp(dim=-1)  # (B, H, bm)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = attn_weights.nan_to_num(0.0)

        block_out = torch.matmul(attn_weights, v_f)  # (B, H, bm, D)

        out[:, :, q_start:q_end, :] = block_out
        lse[:, :, q_start:q_end] = block_lse

    # Transpose back to (B, Sq, H, D)
    out = out.transpose(1, 2).to(q.dtype)
    return out, lse


def _sparse_attn_varlen_func_torch(
    q,
    k,
    v,
    block_count,
    block_offset,
    column_count,
    column_index,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    causal=False,
    softcap=0.0,
    block_size_M=64,
    block_size_N=64,
):
    """Pure PyTorch implementation of variable-length sparse attention.

    Processes each batch element separately using padded sub-tensors,
    then maps results back to the packed (varlen) format.

    Args:
        q: (total_q, nheads, headdim)
        k: (total_k, nheads_k, headdim)
        v: (total_k, nheads_k, headdim)
        cu_seqlens_q: (batch_size + 1,), int32
        cu_seqlens_k: (batch_size + 1,), int32
        max_seqlen_q: int
        max_seqlen_k: int
        (Other args same as _sparse_attn_func_torch)

    Returns:
        out: (total_q, nheads, headdim)
        softmax_lse: (nheads, total_q)
    """
    total_q, Hq, D = q.shape
    _, Hk, _ = k.shape
    batch_size = cu_seqlens_q.shape[0] - 1
    device = q.device

    out = torch.zeros(total_q, Hq, D, device=device, dtype=q.dtype)
    lse = torch.full((Hq, total_q), float("-inf"), device=device, dtype=torch.float32)

    for b in range(batch_size):
        q_start = cu_seqlens_q[b].item()
        q_end = cu_seqlens_q[b + 1].item()
        k_start = cu_seqlens_k[b].item()
        k_end = cu_seqlens_k[b + 1].item()
        sq = q_end - q_start
        sk = k_end - k_start

        # Extract and reshape to (1, seqlen, nheads, headdim)
        q_b = q[q_start:q_end].unsqueeze(0)  # (1, sq, H, D)
        k_b = k[k_start:k_end].unsqueeze(0)  # (1, sk, Hk, D)
        v_b = v[k_start:k_end].unsqueeze(0)  # (1, sk, Hk, D)

        # Per-batch sparse indices (already indexed by batch dim)
        num_rows_b = block_count.shape[2]
        bc_b = block_count[b : b + 1]  # (1, H, num_rows)
        bo_b = block_offset[b : b + 1]
        cc_b = column_count[b : b + 1]
        ci_b = column_index[b : b + 1]

        out_b, lse_b = _sparse_attn_func_torch(
            q_b,
            k_b,
            v_b,
            bc_b,
            bo_b,
            cc_b,
            ci_b,
            softmax_scale,
            causal=causal,
            softcap=softcap,
            block_size_M=block_size_M,
            block_size_N=block_size_N,
        )

        out[q_start:q_end] = out_b.squeeze(0)
        lse[:, q_start:q_end] = lse_b.squeeze(0)

    return out, lse


# ---------------------------------------------------------------------------
# Public API — dispatches to native kernels or PyTorch fallback
# ---------------------------------------------------------------------------


def convert_vertical_slash_indexes(
    q_seqlens: torch.Tensor,  # [BATCH, ]
    kv_seqlens: torch.Tensor,  # [BATCH, ]
    vertical_indexes: torch.Tensor,  # [BATCH, N_HEADS, NNZ_V]
    slash_indexes: torch.Tensor,  # [BATCH, N_HEADS, NNZ_S]
    context_size: int,
    block_size_M: int,
    block_size_N: int,
    causal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if _has_native_convert_ops():
        batch_size = slash_indexes.size(0)
        num_heads = slash_indexes.size(1)
        nnz_slash = slash_indexes.size(2)
        nnz_vertical = vertical_indexes.size(2)
        num_rows = (context_size + block_size_M - 1) // block_size_M

        block_count = torch.zeros(
            batch_size,
            num_heads,
            num_rows,
            dtype=q_seqlens.dtype,
            device=q_seqlens.device,
        )
        block_offset = torch.zeros(
            batch_size,
            num_heads,
            num_rows,
            nnz_slash,
            dtype=q_seqlens.dtype,
            device=q_seqlens.device,
        )
        column_count = torch.zeros(
            batch_size,
            num_heads,
            num_rows,
            dtype=q_seqlens.dtype,
            device=q_seqlens.device,
        )
        column_index = torch.zeros(
            batch_size,
            num_heads,
            num_rows,
            nnz_vertical,
            dtype=q_seqlens.dtype,
            device=q_seqlens.device,
        )

        torch.ops.sgl_kernel.convert_vertical_slash_indexes.default(
            block_count,
            block_offset,
            column_count,
            column_index,
            q_seqlens,
            kv_seqlens,
            vertical_indexes,
            slash_indexes,
            context_size,
            block_size_M,
            block_size_N,
            causal,
        )
        return block_count, block_offset, column_count, column_index
    else:
        return _convert_vertical_slash_indexes_torch(
            q_seqlens,
            kv_seqlens,
            vertical_indexes,
            slash_indexes,
            context_size,
            block_size_M,
            block_size_N,
            causal,
        )


def convert_vertical_slash_indexes_mergehead(
    q_seqlens: torch.Tensor,  # [BATCH, ]
    kv_seqlens: torch.Tensor,  # [BATCH, ]
    vertical_indexes: torch.Tensor,  # [BATCH, N_HEADS, NNZ_V]
    slash_indexes: torch.Tensor,  # [BATCH, N_HEADS, NNZ_S]
    # [N_HEADS] : different head use different number of indices
    vertical_indices_count: torch.Tensor,
    slash_indices_count: torch.Tensor,
    context_size: int,
    block_size_M: int,
    block_size_N: int,
    causal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if _has_native_convert_ops():
        batch_size = slash_indexes.size(0)
        num_heads = slash_indexes.size(1)
        nnz_slash = slash_indexes.size(2)
        nnz_vertical = vertical_indexes.size(2)
        num_rows = (context_size + block_size_M - 1) // block_size_M

        block_count = torch.empty(
            batch_size,
            num_heads,
            num_rows,
            dtype=q_seqlens.dtype,
            device=q_seqlens.device,
        )
        block_offset = torch.empty(
            batch_size,
            num_heads,
            num_rows,
            nnz_slash,
            dtype=q_seqlens.dtype,
            device=q_seqlens.device,
        )
        column_count = torch.empty(
            batch_size,
            num_heads,
            num_rows,
            dtype=q_seqlens.dtype,
            device=q_seqlens.device,
        )
        column_index = torch.empty(
            batch_size,
            num_heads,
            num_rows,
            nnz_vertical,
            dtype=q_seqlens.dtype,
            device=q_seqlens.device,
        )

        torch.ops.sgl_kernel.convert_vertical_slash_indexes_mergehead.default(
            block_count,
            block_offset,
            column_count,
            column_index,
            q_seqlens,
            kv_seqlens,
            vertical_indexes,
            slash_indexes,
            vertical_indices_count,
            slash_indices_count,
            context_size,
            block_size_M,
            block_size_N,
            causal,
        )
        return block_count, block_offset, column_count, column_index
    else:
        return _convert_vertical_slash_indexes_mergehead_torch(
            q_seqlens,
            kv_seqlens,
            vertical_indexes,
            slash_indexes,
            vertical_indices_count,
            slash_indices_count,
            context_size,
            block_size_M,
            block_size_N,
            causal,
        )


def sparse_attn_func(
    q,
    k,
    v,
    block_count,
    block_offset,
    column_count,
    column_index,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    *,
    return_softmax_lse=False,
    out=None,
    block_size_M=64,
    block_size_N=64,
):
    """Compute attention with vertical and slash sparsity patterns.

    Most arguments are the same as the flash_attn_func interface, except for
    4 extra args: block_count and block_offset for slash sparsity patterns, and
    column_count and column_index for vertical sparsity patterns.
    For more details please refer to Appendix C.4.2 of paper
    https://arxiv.org/abs/2407.02490.

    Uses native CUDA/SYCL kernels when available, otherwise falls back to
    a pure PyTorch implementation that works on any device including XPU.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        block_count: (batch_size, nheads, cdiv(seqlen, BLOCK_M))
        block_offset: (batch_size, nheads, cdiv(seqlen, BLOCK_M), NNZ_S)
        column_count: (batch_size, nheads, cdiv(seqlen, BLOCK_M))
        column_index: (batch_size, nheads, cdiv(seqlen, BLOCK_M), NNZ_V)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32 (native only).
        deterministic: bool (native only).
        return_attn_probs: bool (native only).
        block_size_M: int. Query block size for the PyTorch fallback (default 64).
        block_size_N: int. Key block size for the PyTorch fallback (default 64).

    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]:
            (batch_size, nheads, seqlen).
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    if _has_native_sparse_ops():
        q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
        out, softmax_lse = torch.ops.sgl_kernel.fwd_sparse.default(
            q,
            k,
            v,
            block_count,
            block_offset,
            column_count,
            column_index,
            out,
            alibi_slopes,
            dropout_p,
            softmax_scale,
            causal,
            softcap,
            return_attn_probs and dropout_p > 0,
            None,
        )
    else:
        out, softmax_lse = _sparse_attn_func_torch(
            q,
            k,
            v,
            block_count,
            block_offset,
            column_count,
            column_index,
            softmax_scale,
            causal=causal,
            softcap=softcap,
            block_size_M=block_size_M,
            block_size_N=block_size_N,
        )

    return (out, softmax_lse) if return_softmax_lse else out


def sparse_attn_varlen_func(
    q,
    k,
    v,
    block_count,
    block_offset,
    column_count,
    column_index,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    *,
    return_softmax_lse=False,
    out=None,
    block_size_M=64,
    block_size_N=64,
):
    """Compute variable-length attention with vertical+slash sparsity.

    Most arguments are the same as the flash_attn_varlen_func interface,
    except for the 4 sparse pattern args.

    Uses native kernels when available, otherwise falls back to PyTorch.

    Arguments:
        q: (total_q, nheads, headdim)
        k: (total_k, nheads_k, headdim)
        v: (total_k, nheads_k, headdim)
        block_count: (batch_size, nheads, cdiv(seqlen, BLOCK_M))
        block_offset: (batch_size, nheads, cdiv(seqlen, BLOCK_M), NNZ_S)
        column_count: (batch_size, nheads, cdiv(seqlen, BLOCK_M))
        column_index: (batch_size, nheads, cdiv(seqlen, BLOCK_M), NNZ_V)
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32
        max_seqlen_q: int
        max_seqlen_k: int
        dropout_p: float
        softmax_scale: float
        causal: bool
        softcap: float
        block_size_M: int. Query block size for PyTorch fallback (default 64).
        block_size_N: int. Key block size for PyTorch fallback (default 64).

    Return:
        out: (total_q, nheads, headdim).
        softmax_lse [optional]: (nheads, total_q).
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    if _has_native_sparse_ops():
        q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
        out, softmax_lse = torch.ops.sgl_kernel.varlen_fwd_sparse.default(
            q,
            k,
            v,
            block_count,
            block_offset,
            column_count,
            column_index,
            out,
            cu_seqlens_q,
            cu_seqlens_k,
            None,
            alibi_slopes,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            False,
            causal,
            softcap,
            return_attn_probs and dropout_p > 0,
            None,
        )
    else:
        out, softmax_lse = _sparse_attn_varlen_func_torch(
            q,
            k,
            v,
            block_count,
            block_offset,
            column_count,
            column_index,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale,
            causal=causal,
            softcap=softcap,
            block_size_M=block_size_M,
            block_size_N=block_size_N,
        )

    return (out, softmax_lse) if return_softmax_lse else out
