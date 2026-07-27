from typing import Optional, Tuple

import torch

_VALID_FP8_DTYPES = (torch.uint8, torch.float8_e4m3fn)

# Minimum sizes for torch._scaled_mm alignment requirement:
# mat1 rows, mat2 cols, and K must all be divisible by 16.
_SCALED_MM_ALIGN = 16

# Minimum output elements (M*Nk) above which heads are processed one at a time
# to avoid allocating a large (M, Nk) intermediate tensor.  128M fp32 ≈ 512MB.
_HEAD_LOOP_THRESHOLD = 1 << 27


def _check_fp8_dtype(t: torch.Tensor, name: str) -> None:
    if t.dtype not in _VALID_FP8_DTYPES:
        raise TypeError(f"{name} must be uint8 or float8_e4m3fn, got {t.dtype}")


def _fp8_mqa_logits_impl(
    q_fp8: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
) -> torch.Tensor:
    """Core FP8 MQA logits computation (pure Python/PyTorch).

    Math: score[i,j] = k_scale[j] * Σ_h ReLU( q[i,h,:] · k[j,:] ) * weights[i,h]
    Only positions j in [ks[i], ke[i]) are non-zero.

    Args:
        q_fp8: (Nq, H, D) — fp8_e4m3 queries (as uint8 or float8_e4m3fn)
        k_fp8: (Nk, D) — fp8_e4m3 keys (as uint8 or float8_e4m3fn)
        k_scale: (Nk,) float32 — per-token dequant scales
        weights: (Nq, H) float32 — combined head gate weights
        ks: (Nq,) int32 — start index per query
        ke: (Nq,) int32 — end index (exclusive) per query

    Returns:
        logits: (Nq, Nk) float32
    """
    q_fp8 = q_fp8.view(torch.float8_e4m3fn) if q_fp8.dtype == torch.uint8 else q_fp8
    k_fp8 = k_fp8.view(torch.float8_e4m3fn) if k_fp8.dtype == torch.uint8 else k_fp8
    Nq, H, D = q_fp8.shape
    Nk = k_fp8.shape[0]

    if Nq == 0 or Nk == 0:
        return torch.zeros(Nq, Nk, dtype=torch.float32, device=q_fp8.device)

    M = Nq * H
    q_2d = q_fp8.contiguous().reshape(M, D)
    k_2d = k_fp8.contiguous()

    # torch._scaled_mm requires alignment: both dims of mat2 (Nk, D) and M divisible by 16.
    use_scaled_mm = (
        hasattr(torch, "_scaled_mm")
        and Nk % _SCALED_MM_ALIGN == 0
        and D % _SCALED_MM_ALIGN == 0
    )

    # For large prefill batches, M*Nk can be very large (e.g. 4096*32*4096 = 2GB).
    # Process heads one at a time to avoid OOM.  Convert fp8→bf16 ONCE (not per-head)
    # because the cast is expensive on some devices (e.g. XPU ~15ms/head).
    if M * Nk > _HEAD_LOOP_THRESHOLD:
        score = torch.zeros(Nq, Nk, dtype=torch.float32, device=q_fp8.device)
        if use_scaled_mm and Nq % _SCALED_MM_ALIGN == 0:
            # _scaled_mm path: convert k once; q_h is passed as fp8 per head
            k_t = k_2d.t().contiguous()
            one = torch.ones(1, dtype=torch.float32, device=q_fp8.device)
            for h in range(H):
                q_h = q_fp8[:, h, :].contiguous()
                dots_h = torch._scaled_mm(
                    q_h, k_t, scale_a=one, scale_b=one, out_dtype=torch.float32
                )
                score += dots_h.relu_() * weights[:, h : h + 1]
        else:
            # bf16 path: convert q and k to bf16 once, then slice per head
            q_bf16 = q_fp8.to(torch.bfloat16)  # (Nq, H, D)
            k_bf16 = k_2d.to(torch.bfloat16)  # (Nk, D)
            k_bf16_t = k_bf16.t().contiguous()
            for h in range(H):
                dots_h = torch.mm(q_bf16[:, h, :].contiguous(), k_bf16_t).float()
                score += dots_h.relu_() * weights[:, h : h + 1]
        score *= k_scale
    else:
        if use_scaled_mm and M % _SCALED_MM_ALIGN == 0:
            one = torch.ones(1, dtype=torch.float32, device=q_fp8.device)
            # dots: (M, Nk) = (Nq*H, Nk)
            dots = torch._scaled_mm(
                q_2d, k_2d.t(), scale_a=one, scale_b=one, out_dtype=torch.float32
            )
        else:
            dots = torch.mm(
                q_2d.to(torch.bfloat16), k_2d.to(torch.bfloat16).t()
            ).float()

        # Reduce across heads: score[i,j] = Σ_h ReLU(dots[i*H+h, j]) * weights[i,h]
        # then multiply by k_scale[j]
        score = (dots.view(Nq, H, Nk).relu() * weights.unsqueeze(-1)).sum(
            dim=1
        ) * k_scale

    # Mask out positions outside [ks[i], ke[i])
    j = torch.arange(Nk, device=q_fp8.device, dtype=torch.int32).unsqueeze(0)  # (1, Nk)
    mask = (j >= ks.unsqueeze(1)) & (j < ke.unsqueeze(1))
    return score * mask


def fp8_mqa_logits(
    q_fp8: torch.Tensor,
    kv_fp8: Tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    clean_logits: bool = False,
) -> torch.Tensor:
    """FP8 MQA logits for ragged (prefill/extend) path.

    Computes indexer scores: for each query, dot-product with K tokens
    in range [ks, ke), apply ReLU, weight by head gates, reduce across
    heads, and scale by per-token K scale.

    Args:
        q_fp8: (Nq, H, D) fp8 e4m3 queries (stored as uint8 or float8_e4m3fn)
        kv_fp8: tuple of (k_fp8, k_scale) where
            k_fp8: (Nk, D) fp8 e4m3 keys
            k_scale: (Nk,) float32 per-token dequant scales
        weights: (Nq, H) float32 combined gate weights
        ks: (Nq,) int32 start indices
        ke: (Nq,) int32 end indices
        clean_logits: unused, kept for API compatibility

    Returns:
        logits: (Nq, Nk) float32
    """
    k_fp8, k_scale = kv_fp8
    _check_fp8_dtype(q_fp8, "q_fp8")
    _check_fp8_dtype(k_fp8, "k_fp8")
    return _fp8_mqa_logits_impl(q_fp8, k_fp8, k_scale, weights, ks, ke)


def fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: Optional[torch.Tensor],
    max_seq_len: int,
    clean_logits: bool = False,
) -> torch.Tensor:
    """FP8 MQA logits for paged (decode) path.

    Args:
        q_fp8: (B, 1, H, D) or (B, H, D) fp8 e4m3 queries (stored as uint8)
        kv_cache: (num_pages, page_size, 1, D+4) uint8 paged KV cache
        weights: (B, H) float32 combined gate weights
        seq_lens: (B,) or (B,1) int32 actual sequence lengths
        block_tables: (B, max_num_blocks) int32 page table
        schedule_metadata: optional scheduling metadata (ignored on XPU)
        max_seq_len: maximum sequence length
        clean_logits: unused on XPU (output is always zero-initialized), kept for API compatibility

    Returns:
        logits: (B, max_seq_len) float32
    """
    _check_fp8_dtype(q_fp8, "q_fp8")
    # The SYCL kernel expects q_fp8 to be 4D (B, 1, H, D). Add the next_n=1 dimension
    # if a 3D tensor (B, H, D) is passed (e.g. from _get_topk_paged in decode mode).
    if q_fp8.dim() == 3:
        q_fp8 = q_fp8.unsqueeze(1)
    q_input = q_fp8.view(torch.uint8) if q_fp8.dtype != torch.uint8 else q_fp8
    return torch.ops.sgl_kernel.fp8_paged_mqa_logits.default(
        q_input,
        kv_cache,
        weights,
        seq_lens,
        block_tables,
        schedule_metadata,
        max_seq_len,
        clean_logits,
    )
