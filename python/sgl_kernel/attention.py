from typing import Optional, Tuple

import torch


def lightning_attention_decode(q, k, v, past_kv, slope, output, new_kv):
    torch.ops.sgl_kernel.lightning_attention_decode.default(
        q, k, v, past_kv, slope, output, new_kv
    )


def merge_state(
    v_a: torch.Tensor,
    s_a: torch.Tensor,
    v_b: torch.Tensor,
    s_b: torch.Tensor,
    v_merged: Optional[torch.Tensor] = None,
    s_merged: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    s_a = s_a.to(torch.float32)
    s_b = s_b.to(torch.float32)
    # Avoid creating new tensors if they are already provided
    if v_merged is None:
        v_merged = torch.empty_like(v_a)
    if s_merged is None:
        s_merged = torch.empty_like(s_a)
    torch.ops.sgl_kernel.merge_state.default(v_a, s_a, v_b, s_b, v_merged, s_merged)
    return v_merged, s_merged


def merge_state_v2(
    v_a: torch.Tensor,
    s_a: torch.Tensor,
    v_b: torch.Tensor,
    s_b: torch.Tensor,
    v_merged: Optional[torch.Tensor] = None,
    s_merged: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    s_a = s_a.to(torch.float32)
    s_b = s_b.to(torch.float32)
    # TODO(DefTruth): Currently, the custom merge_attn_states kernel
    # does not support the FP8 data type and non - CUDA devices.
    # It may be necessary to fall back to using the Triton kernel.

    # Avoid creating new tensors if they are already provided
    if v_merged is None:
        v_merged = torch.empty_like(v_a)
    if s_merged is None:
        s_merged = torch.empty_like(s_a)
    torch.ops.sgl_kernel.merge_state_v2.default(v_a, s_a, v_b, s_b, v_merged, s_merged)
    return v_merged, s_merged


def flash_mla_decode(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_c_and_k_pe_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    workspace: torch.Tensor,
    sm_scale: float,
    num_kv_splits: int = 1,
) -> torch.Tensor:
    assert q_nope.ndim == 3, f"q_nope must be a 3D tensor, but got {q_nope.ndim}"
    assert q_pe.ndim == 3, f"q_pe must be a 3D tensor, but got {q_pe.ndim}"
    assert (
        kv_c_and_k_pe_cache.ndim == 3
    ), f"kv_c_and_k_pe_cache must be a 3D tensor, but got {kv_c_and_k_pe_cache.ndim}"

    device_type = q_nope.device.type
    B_q, H, D_q_nope = q_nope.shape
    B_q_2, H_2, D_q_pe = q_pe.shape
    assert (B_q == B_q_2) and (H == H_2)

    _, PAGE_SIZE, D_ckv = kv_c_and_k_pe_cache.shape

    D_latent = 512
    D_rope = 64
    assert D_q_nope == D_latent
    assert D_q_pe == D_rope
    assert D_ckv == D_latent + D_rope

    MAX_HEADS = 128
    assert H <= MAX_HEADS, f"H must be <= {MAX_HEADS}, but got {H}"
    if H < MAX_HEADS and device_type != "xpu":
        q_nope_padded = q_nope.new_empty((B_q, MAX_HEADS, D_q_nope))
        q_nope_padded[:, :H] = q_nope
        q_nope = q_nope_padded

        q_pe_padded = q_pe.new_empty((B_q, MAX_HEADS, D_q_pe))
        q_pe_padded[:, :H] = q_pe
        q_pe = q_pe_padded
    elif device_type == "xpu":
        q_nope = q_nope.contiguous()
        q_pe = q_pe.contiguous()

    assert len(page_table.shape) == 2
    B_block_table, block_num = page_table.shape
    assert B_block_table == B_q
    assert block_num > 0, f"block num must be greater than 0, got {block_num}"
    assert block_num % (128 / PAGE_SIZE) == 0

    assert q_nope.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"q_nope.dtype needs to be fp16 or bf16 but got {q_nope.dtype}."
    assert q_nope.dtype == q_pe.dtype == kv_c_and_k_pe_cache.dtype
    assert (
        seq_lens.dtype == torch.int32
    ), f"seq_lens.dtype needs to be int32 but got {seq_lens.dtype}."
    assert (
        page_table.dtype == torch.int32
    ), f"page_table.dtype needs to be int32 but got {page_table.dtype}."

    out = (
        q_nope.new_empty((B_q, H, D_latent))
        if device_type == "xpu"
        else q_nope.new_empty((B_q, MAX_HEADS, D_latent))
    )

    torch.ops.sgl_kernel.flash_mla_decode.default(
        out,
        q_nope,
        q_pe,
        kv_c_and_k_pe_cache,
        seq_lens,
        page_table,
        workspace,
        sm_scale,
        num_kv_splits,
    )
    return out if device_type == "xpu" else out[:, :H].contiguous()


def flash_mla_get_workspace_size(
    max_seq_len: int,
    num_batches: int,
    num_heads: int = 0,
    page_size: int = 0,
    num_kv_splits: int = -1,
) -> int:
    assert max_seq_len > 0, f"max_seq_len must be greater than 0, got {max_seq_len}"
    assert num_batches > 0, f"num_batches must be greater than 0, got {num_batches}"
    return torch.ops.sgl_kernel.flash_mla_get_workspace_size.default(
        max_seq_len, num_batches, num_heads, page_size, num_kv_splits
    )


def flash_mla_prefill(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_c_and_k_pe_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens_k: torch.Tensor,
    max_seqlen_q: int,
    page_table: torch.Tensor,
    workspace: torch.Tensor,
    sm_scale: float,
    causal: bool = True,
    num_kv_splits: int = -1,
) -> torch.Tensor:
    """MLA prefill with varlen/ragged Q and causal masking.

    Supports full prefill (seqlen_q == seqlen_k) and incremental prefill
    (seqlen_q < seqlen_k). Prefix tokens are unmasked; only new prompt
    tokens get triangular causal mask.

    Args:
        q_nope: (total_q, num_heads, latent_dim)  fp16/bf16, ragged
        q_pe:   (total_q, num_heads, rope_dim)    fp16/bf16, ragged
        kv_c_and_k_pe_cache: (total_pages, page_size, latent_dim + rope_dim)
        cu_seqlens_q: (batch + 1,) int32  cumulative Q lengths
        seq_lens_k:   (batch,) int32  KV sequence length per request
        max_seqlen_q: int  max Q length across batch
        page_table:   (batch, max_pages_per_seq)  int32
        workspace:    uint8 workspace tensor
        sm_scale:     softmax scale factor
        causal:       apply causal masking (default True)
        num_kv_splits: KV split count. -1 = auto-select. Split-KV is not yet
                       implemented for MLA prefill; reserved for future use.

    Returns:
        out: (total_q, num_heads, latent_dim)  ragged, same layout as q_nope
    """
    assert (
        q_nope.ndim == 3
    ), f"q_nope must be 3D (total_q, heads, dim), got {q_nope.ndim}"
    assert q_pe.ndim == 3, f"q_pe must be 3D (total_q, heads, dim), got {q_pe.ndim}"
    assert (
        kv_c_and_k_pe_cache.ndim == 3
    ), f"kv_c_and_k_pe_cache must be 3D (pages, page_size, dim), got {kv_c_and_k_pe_cache.ndim}"

    total_q, H, D_latent = q_nope.shape
    _, _, D_rope = q_pe.shape
    _, PAGE_SIZE, D_ckv = kv_c_and_k_pe_cache.shape

    assert (
        D_ckv == D_latent + D_rope
    ), f"kv dim {D_ckv} must equal D_latent({D_latent}) + D_rope({D_rope})"
    assert q_nope.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"q_nope.dtype must be fp16 or bf16, got {q_nope.dtype}"
    assert q_nope.dtype == q_pe.dtype == kv_c_and_k_pe_cache.dtype
    assert cu_seqlens_q.dtype == torch.int32
    assert seq_lens_k.dtype == torch.int32
    assert page_table.dtype == torch.int32

    batch_size = cu_seqlens_q.shape[0] - 1
    assert seq_lens_k.shape[0] == batch_size

    # The kernel epilogue writes Q_TILE_M rows per tile without bounds-checking.
    # Pad the output to the next multiple of Q_TILE_M (max 256 for the Large
    # bucket) so partial last tiles don't cause OOB writes and device-lost.
    _Q_TILE_MAX = 256
    total_q_padded = (total_q + _Q_TILE_MAX - 1) // _Q_TILE_MAX * _Q_TILE_MAX
    out = q_nope.new_empty((total_q_padded, H, D_latent))

    torch.ops.sgl_kernel.flash_mla_prefill.default(
        out,
        q_nope.contiguous(),
        q_pe.contiguous(),
        kv_c_and_k_pe_cache,
        cu_seqlens_q,
        seq_lens_k.contiguous(),
        max_seqlen_q,
        page_table,
        workspace,
        sm_scale,
        causal,
        num_kv_splits,
    )
    return out[:total_q]


def flash_mla_prefill_get_workspace_size(
    max_seq_len: int,
    num_batches: int,
    num_heads: int = 0,
    page_size: int = 0,
    num_kv_splits: int = -1,
) -> int:
    assert max_seq_len > 0, f"max_seq_len must be > 0, got {max_seq_len}"
    assert num_batches > 0, f"num_batches must be > 0, got {num_batches}"
    return torch.ops.sgl_kernel.flash_mla_prefill_get_workspace_size.default(
        max_seq_len, num_batches, num_heads, page_size, num_kv_splits
    )


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    head_dim_v: int = 512,
    tile_scheduler_metadata: Optional[torch.Tensor] = None,
    num_splits: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DeepSeek V4 MLA Decode Attention with KVCache.

    When block_table is None: Performs sparse attention via indices.
    When block_table is provided: Performs dense attention via block_table.

    Returns:
        out: [B, s_q, H, head_dim_v]  bf16 — final output
        lse: [B, H, s_q]              fp32 — log-sum-exp
    """
    assert (
        tile_scheduler_metadata is None
    ), "tile_scheduler_metadata is not supported for xpu"
    assert num_splits is None, "num_splits is not supported for xpu"
    assert not causal, "causal attention is not supported for xpu"
    assert q.ndim == 4, f"q must be 4D [B, s_q, H, D_qk], got {q.ndim}D"
    assert (
        k_cache.ndim == 4
    ), f"k_cache must be 4D [num_pages, P, 1, D], got {k_cache.ndim}D"

    B, s_q, H, D_qk = q.shape

    if softmax_scale is None:
        softmax_scale = D_qk ** (-0.5)

    assert q.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"q.dtype must be fp16 or bf16, got {q.dtype}"

    # Allocate outputs
    out = q.new_empty((B, s_q, H, head_dim_v))
    lse = torch.empty((B, H, s_q), dtype=torch.float32, device=q.device)

    if block_table is None:
        assert indices is not None, "indices must be provided for sparse decode path"
        if topk_length is not None:
            assert (
                topk_length.device == q.device
                and topk_length.dtype == torch.int32
                and topk_length.shape == (B,)
            ), "topk_length must be int32 on the same device with shape [B]"
        if attn_sink is not None:
            assert (
                attn_sink.device == q.device
                and attn_sink.dtype == torch.float32
                and attn_sink.shape == (H,)
            ), "attn_sink must be float32 on the same device with shape [H]"
        if (extra_k_cache is None) ^ (extra_indices_in_kvcache is None):
            raise AssertionError(
                "extra_k_cache and extra_indices_in_kvcache must be provided together"
            )
        if extra_indices_in_kvcache is not None:
            assert (
                extra_indices_in_kvcache.dtype == torch.int32
            ), "extra_indices_in_kvcache.dtype must be int32"
            assert extra_indices_in_kvcache.device == q.device and extra_indices_in_kvcache.shape[
                :2
            ] == (
                B,
                s_q,
            ), "extra_indices_in_kvcache must be on the same device with shape [B, s_q, extra_topk]"
        if extra_topk_length is not None:
            assert (
                extra_topk_length.device == q.device
                and extra_topk_length.dtype == torch.int32
                and extra_topk_length.shape == (B,)
            ), "extra_topk_length must be int32 on the same device with shape [B]"
        torch.ops.sgl_kernel.flash_mla_sparse_decode.default(
            out,
            lse,
            q,
            k_cache,
            indices,
            topk_length,
            extra_k_cache,
            extra_indices_in_kvcache,
            extra_topk_length,
            attn_sink,
            softmax_scale,
            head_dim_v,
            is_fp8_kvcache,
        )
    else:
        assert (
            block_table is not None and cache_seqlens is not None
        ), "block_table path is not enabled yet for xpu"

    return out, lse


def flash_mla_sparse_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: Optional[float] = None,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    return_softmax_lse: bool = False,
):
    """DeepSeek V4 sparse MLA prefill (dense gather + fused attention).

    Args:
        q: [s_q, h_q, d_qk] bfloat16 query.
        kv: [s_kv, h_kv, d_qk] bfloat16 key/value (h_kv must be 1).
        indices: [s_q, h_kv, topk] int32 gathered token indices.
        sm_scale: softmax scale. Defaults to d_qk ** -0.5.
        d_v: value head dim (must be 512).
        attn_sink: optional [h_q] float32 attention sink logits.
        topk_length: optional [s_q] int32 valid topk length per query.
        return_softmax_lse: if True, also return (max_logits, lse).

    Returns:
        out [s_q, h_q, d_v] if return_softmax_lse is False, else
        (out, max_logits, lse) with max_logits/lse shaped [s_q, h_q].
    """
    assert q.ndim == 3, f"q must be 3D [s_q, h_q, d_qk], got {q.ndim}D"
    assert kv.ndim == 3, f"kv must be 3D [s_kv, h_kv, d_qk], got {kv.ndim}D"
    assert (
        indices.ndim == 3
    ), f"indices must be 3D [s_q, h_kv, topk], got {indices.ndim}D"

    d_qk = q.shape[2]
    if sm_scale is None:
        sm_scale = d_qk ** (-0.5)

    outs = torch.ops.sgl_kernel.flash_mla_sparse_prefill.default(
        q,
        kv,
        indices,
        sm_scale,
        d_v,
        attn_sink,
        topk_length,
        return_softmax_lse,
    )
    if return_softmax_lse:
        out, max_logits, lse = outs
        return out, max_logits, lse
    return outs[0]
