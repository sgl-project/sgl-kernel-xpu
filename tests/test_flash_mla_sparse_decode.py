"""
Tests for V4 Sparse MLA Decode — matches DeepSeek V4 actual parameters.

Reference implementation: _sm120_sparse_decode_fwd inlined from
https://github.com/AliceChenyy/sglang/blob/7cc3aa4819525d9d95f048786eb21853b08cbade/
  python/sglang/srt/layers/attention/flash_mla_sm120_fallback.py

DeepSeek V4 KV cache layout (FP8 packed, PAGE-END scales):
  Physical page structure (total_page_bytes = page_size*576 + ceil_576(page_size*8)):
    [0 .. page_size*576)           Token data section
      Per token (576 bytes):
        bytes   0-447: K_nope  FP8_E4M3  (448B = 7 tiles × 64)
        bytes 448-575: K_rope  BF16      (128B = 64 dims × 2 bytes)
    [page_size*576 .. end)         Scale section (padded to 576-byte boundary)
      Per token (8 bytes):
        bytes 0-6: 7 nope tile scales UE8M0
        byte  7:   1 reserved

  as_strided view (as seen by kernel):
    shape:  (num_pages, page_size, 1, 584)
    stride: (total_page_bytes, 584, 584, 1)
    dtype:  float8_e4m3fn

  stride(1) = 584 is a metadata value (= shape[3]), NOT physical token spacing.
  stride(0) = total_page_bytes encodes the real page stride for raw byte access.
  The kernel uses stride(0) to locate pages and computes internal offsets manually.

DeepSeek V4 constants (from production trace):
  D_QK = 512, D_NOPE = 448, D_ROPE = 64, D_V = 512
  num_heads = 64, num_kv_heads = 1, page_size = 256
  SWA_WINDOW = 128 tokens, extra_topk ∈ {64, 512, 8256}
  sm_scale = 1/sqrt(512) = 0.04419417382415922
"""

import gc
import math
import os
import sys
from typing import Optional

import pytest
import torch
from sgl_kernel import flash_mla_sparse_decode
from torch import Tensor

device = torch.device("xpu")

if not torch.xpu.is_available():
    pytest.skip(
        reason="V4 Sparse MLA Decode requires XPU device.",
        allow_module_level=True,
    )

# V4 production kernel requires FP8 packed KV cache (FP8_E4M3 nope + BF16 rope + UE8M0 scales)
_HAS_FP8 = hasattr(torch, "float8_e4m3fn") and hasattr(torch, "float8_e8m0fnu")
if not _HAS_FP8:
    pytest.skip(
        reason="V4 Sparse MLA requires torch.float8_e4m3fn + torch.float8_e8m0fnu. "
        "Upgrade PyTorch to a version with FP8 support.",
        allow_module_level=True,
    )

# Page layout constants for DSv4-Flash (MODEL1):
#   nope_dim = 448, rope_dim = 64, quantize_block_size = 64
#   nope_rope_stride = 448 + 64*2 = 576 bytes per token
#   scale_stride = ceil(448/64) + 1 = 8 bytes per token (7 scales + 1 pad)
#   bytes_per_token = 448 + 128 + 8 = 584
#   page_bytes = ceil_div(page_size * 584, 576) * 576

# ── DeepSeek V4 fixed constants (from production trace) ──
_NOPE_DIM = 448
_ROPE_DIM = 64
D_QK = 512  # D_NOPE + D_ROPE
D_V = 512
H_Q = 64
H_KV = 1
SWA_WINDOW = 128
SM_SCALE = 1.0 / math.sqrt(D_QK)  # 1/sqrt(512) = 0.04419...

# FP8 packed layout constants (page-end scales)
_NOPE_ROPE_STRIDE = _NOPE_DIM + _ROPE_DIM * 2  # 576 bytes per token in data section
_TILE_SIZE = 64
_NUM_TILES = _NOPE_DIM // _TILE_SIZE  # 7
_SCALE_STRIDE = _NUM_TILES + 1  # 8 bytes per token in scale section
_D = _NOPE_DIM + _ROPE_DIM  # 512 (dequantized output dim)

# Default page size from trace
PAGE_SIZE = 256

_GATHER_CHUNK = 16384  # tokens per chunk; ~16k * 1024 B ≈ 16 MiB output per chunk

# Per-chunk peak-memory budget for the sparse decode fallback (MiB).  Read
# once at import time so the forward path doesn't pay an os.environ lookup
# per layer per decode step.
_SM120_SPARSE_CHUNK_MIB = int(os.environ.get("SGLANG_SM120_SPARSE_CHUNK_MIB", "256"))


def _ceil_to_576(x):
    """Align x up to the next multiple of 576."""
    return ((x + 575) // 576) * 576


def _total_page_bytes(page_size):
    """Compute total page bytes matching production allocation.

    total = page_size * 576 (data) + ceil_576(page_size * 8) (scales)
    Verified against trace stride(0) for page_size = 256, 64, 2.
    """
    return page_size * _NOPE_ROPE_STRIDE + _ceil_to_576(page_size * _SCALE_STRIDE)


def clear_memory():
    gc.collect()
    if torch.xpu.is_available():
        torch.xpu.empty_cache()
        torch.xpu.synchronize()


@pytest.fixture(autouse=True)
def reset_torch_defaults():
    yield
    clear_memory()


# ===========================================================================
# Reference: _gather_and_dequant + _sm120_sparse_decode_fwd
# ===========================================================================


def _gather_and_dequant(k_cache, indices, page_size):
    """Gather KV entries from the paged buffer using correct page-internal addressing.

    Args:
        k_cache: (num_pages, page_size, 1, bytes_per_token) float8_e4m3fn
                 Non-contiguous view of the raw page buffer.
        indices: (...) int32/int64, token-level indices. Invalid indices are
                 expected to already be clamped into [0, num_pages*page_size).
        page_size: tokens per page (e.g. 256, 64, 2)

    Returns:
        kv: (..., _D) bfloat16, dequantized KV vectors
    """
    idx_shape = indices.shape
    flat_idx = indices.reshape(-1)  # (N,)
    N = flat_idx.shape[0]
    device = k_cache.device

    page_bytes = k_cache.stride(0)  # actual byte stride between pages
    num_pages = k_cache.shape[0]

    # Flatten the raw byte buffer so we can gather with a single int64 index
    # per byte instead of paying for a full (N, 448) int64 index tensor up
    # front. flat_buf has nelems = num_pages * page_bytes uint8.
    raw_pages = k_cache.as_strided(
        (num_pages, page_bytes),
        (page_bytes, 1),
    ).view(torch.uint8)
    flat_buf = raw_pages.reshape(-1)

    scale_section_offset = page_size * _NOPE_ROPE_STRIDE

    nope_arange = torch.arange(_NOPE_DIM, device=device, dtype=torch.long)
    rope_arange = torch.arange(_ROPE_DIM * 2, device=device, dtype=torch.long)
    scale_arange = torch.arange(_NUM_TILES, device=device, dtype=torch.long)

    result = torch.empty(N, _D, dtype=torch.bfloat16, device=device)

    # Process in chunks to bound peak memory of the int64 advanced-index
    # tensors (which would otherwise be N * 448 * 8 bytes — multiple GB on
    # long-context prefills with large topk).
    for start in range(0, N, _GATHER_CHUNK):
        end = min(start + _GATHER_CHUNK, N)
        chunk = flat_idx[start:end]
        n = end - start

        pages = chunk // page_size
        offsets = chunk % page_size

        # Per-token base byte offset into the flat raw buffer.
        page_base = pages.to(torch.long) * page_bytes  # (n,)
        nope_base = page_base + offsets.to(torch.long) * _NOPE_ROPE_STRIDE  # (n,)

        nope_idx = nope_base.unsqueeze(-1) + nope_arange  # (n, 448)
        rope_idx = nope_base.unsqueeze(-1) + (_NOPE_DIM + rope_arange)  # (n, 128)
        scale_idx = (
            page_base.unsqueeze(-1)
            + scale_section_offset
            + offsets.to(torch.long).unsqueeze(-1) * _SCALE_STRIDE
            + scale_arange
        )  # (n, 7)

        nope_bytes = flat_buf[nope_idx.reshape(-1)].view(n, _NOPE_DIM)
        rope_bytes = flat_buf[rope_idx.reshape(-1)].view(n, _ROPE_DIM * 2)
        scale_bytes = flat_buf[scale_idx.reshape(-1)].view(n, _NUM_TILES)

        nope_fp8 = nope_bytes.view(torch.float8_e4m3fn)  # (n, 448)
        rope_bf16 = rope_bytes.contiguous().view(torch.bfloat16)  # (n, 64)
        scale_e8m0 = scale_bytes.view(torch.float8_e8m0fnu)  # (n, 7)

        result[start:end, :_NOPE_DIM] = (
            (
                nope_fp8.view(n, _NUM_TILES, _TILE_SIZE).float()
                * scale_e8m0.view(n, _NUM_TILES, 1).float()
            )
            .view(n, _NOPE_DIM)
            .to(torch.bfloat16)
        )
        result[start:end, _NOPE_DIM:] = rope_bf16

    return result.reshape(*idx_shape, _D)


def _sm120_sparse_decode_fwd(
    q,
    k_cache,
    indices,
    topk_length,
    attn_sink,
    head_dim_v,
    softmax_scale,
    extra_k_cache=None,
    extra_indices=None,
    extra_topk_length=None,
):
    B, s_q, H_q, D_qk = q.shape
    num_pages, page_size, H_k, bpt = k_cache.shape
    topk = indices.shape[-1]
    device = q.device

    # FlashMLA kernel treats `index == -1` as invalid; we additionally treat
    # any index outside [0, num_pages*page_size) as invalid because the CUDA
    # tile scheduler would simply never visit those slots, whereas this
    # PyTorch fallback gathers them eagerly.
    max_valid = num_pages * page_size
    invalid_mask = (indices < 0) | (indices >= max_valid)
    safe_indices = indices.clamp(min=0, max=max_valid - 1)
    if topk_length is not None:
        topk_range = torch.arange(topk, device=topk_length.device).view(1, 1, topk)
        invalid_mask = invalid_mask | (topk_range >= topk_length.view(B, 1, 1))

    have_extra = extra_k_cache is not None and extra_indices is not None
    if have_extra:
        extra_topk = extra_indices.shape[-1]
        extra_num_pages, extra_page_size = (
            extra_k_cache.shape[0],
            extra_k_cache.shape[1],
        )
        extra_max_valid = extra_num_pages * extra_page_size
        extra_invalid = (extra_indices < 0) | (extra_indices >= extra_max_valid)
        extra_safe = extra_indices.clamp(min=0, max=extra_max_valid - 1)
        if extra_topk_length is not None:
            extra_range = torch.arange(
                extra_topk, device=extra_topk_length.device
            ).view(1, 1, extra_topk)
            extra_invalid = extra_invalid | (
                extra_range >= extra_topk_length.view(B, 1, 1)
            )
    else:
        extra_topk = 0

    total_topk = topk + extra_topk
    # Flatten the (B, s_q) row dimension so we can chunk easily.
    R = B * s_q  # number of query rows
    q_rows = q.reshape(R, H_q, D_qk)
    safe_indices_rows = safe_indices.reshape(R, topk)
    invalid_rows = invalid_mask.reshape(R, topk)
    if have_extra:
        extra_safe_rows = extra_safe.reshape(R, extra_topk)
        extra_invalid_rows = extra_invalid.reshape(R, extra_topk)

    out_rows = torch.empty(R, H_q, head_dim_v, dtype=torch.bfloat16, device=device)
    lse_rows = torch.empty(R, H_q, dtype=torch.float32, device=device)

    # Bound per-chunk peak memory. Dominant bf16 tensor is gathered KV:
    # chunk * total_topk * _D * 2 bytes; fp32 working set adds ~3x on top.
    # On Intel L0, per-launch overhead is high (~hundreds of us), so prefer
    # fewer/larger chunks. Target 256 MiB peak (override via
    # SGLANG_SM120_SPARSE_CHUNK_MIB at import time).
    bytes_per_row = total_topk * _D * 2
    chunk_rows = max(
        1, min(R, (_SM120_SPARSE_CHUNK_MIB * 1024 * 1024) // max(1, bytes_per_row))
    )

    for start in range(0, R, chunk_rows):
        end = min(start + chunk_rows, R)
        n = end - start

        # Gather KV for this chunk only.
        kv_chunk = _gather_and_dequant(
            k_cache, safe_indices_rows[start:end], page_size
        )  # (n, topk, _D)
        inv_chunk = invalid_rows[start:end]  # (n, topk)
        if have_extra:
            extra_kv_chunk = _gather_and_dequant(
                extra_k_cache, extra_safe_rows[start:end], extra_page_size
            )  # (n, extra_topk, _D)
            kv_chunk = torch.cat([kv_chunk, extra_kv_chunk], dim=1)
            inv_chunk = torch.cat([inv_chunk, extra_invalid_rows[start:end]], dim=1)
            del extra_kv_chunk

        q_chunk = q_rows[start:end].float()  # (n, H_q, D_qk)
        # Scrub NaN from invalid-index dequant so the value reduction is not
        # polluted by 0 * NaN = NaN. Done in-place after the float upcast to
        # avoid a separate allocation; ``scores`` is masked to ``-inf`` below
        # which gives invalid positions exactly zero weight.
        kv_f = kv_chunk.float().nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        kv_d = kv_f.shape[-1]
        if D_qk != kv_d:
            q_chunk = q_chunk[..., :kv_d]

        # scores: (n, H_q, T)
        scores = torch.einsum("nhd,ntd->nht", q_chunk, kv_f) * softmax_scale
        scores.masked_fill_(inv_chunk.unsqueeze(1).expand_as(scores), float("-inf"))

        lse = torch.logsumexp(scores, dim=-1)  # (n, H_q)

        if attn_sink is not None:
            lse_for_out = torch.logsumexp(
                torch.stack([lse, attn_sink.view(1, H_q).expand_as(lse)], dim=0),
                dim=0,
            )
        else:
            lse_for_out = lse.clone()

        lonely = lse == float("-inf")
        lse_for_out[lonely] = float("inf")
        weights = torch.exp(scores - lse_for_out.unsqueeze(-1))
        out_chunk = torch.einsum("nht,ntv->nhv", weights, kv_f[..., :head_dim_v])
        out_chunk[lonely.unsqueeze(-1).expand_as(out_chunk)] = 0.0

        out_rows[start:end] = out_chunk.to(torch.bfloat16)
        lse_rows[start:end] = lse

        del (
            kv_chunk,
            kv_f,
            q_chunk,
            scores,
            weights,
            out_chunk,
            lse,
            lse_for_out,
            lonely,
        )

    out = out_rows.reshape(B, s_q, H_q, head_dim_v)
    lse = lse_rows.reshape(B, s_q, H_q).permute(0, 2, 1)
    return out, lse


# ===========================================================================
# Kernel under test
# ===========================================================================


def call_kernel(
    q,
    k_cache,
    indices,
    attn_sink=None,
    extra_k_cache=None,
    extra_indices=None,
    topk_length=None,
    extra_topk_length=None,
):
    """Calls flash_mla_sparse_decode with FP8 packed KV cache (page-end scales)."""
    return flash_mla_sparse_decode(
        q=q,
        k_cache=k_cache,
        indices=indices,
        topk_length=topk_length,
        attn_sink=attn_sink,
        head_dim_v=D_V,
        softmax_scale=SM_SCALE,
        extra_k_cache=extra_k_cache,
        extra_indices=extra_indices,
        extra_topk_length=extra_topk_length,
    )


# ===========================================================================
# Helpers — create FP8 packed KV cache matching DeepSeek V4 production format
# ===========================================================================


def make_fp8_kv_cache(num_pages, page_size=PAGE_SIZE):
    """
    Create FP8 packed KV cache matching DeepSeek V4 production format.

    Page-end scales layout:
      [0 .. page_size*576)    Token data: [nope0|rope0|nope1|rope1|...|nopeN|ropeN]
      [page_size*576 .. end)  Scale section: [scale0|scale1|...|scaleN] (padded to 576)

    Returns k_cache as_strided view:
      shape:  (num_pages, page_size, 1, 584)
      stride: (total_page_bytes, 584, 584, 1)
      dtype:  float8_e4m3fn
    """
    total_pb = _total_page_bytes(page_size)

    # Allocate raw uint8 buffer
    raw = torch.zeros(num_pages, total_pb, dtype=torch.uint8, device=device)

    # Fill token data section: [nope_i(448B) | rope_i(128B)] × page_size
    for t in range(page_size):
        t_offset = t * _NOPE_ROPE_STRIDE
        # Nope: bf16 randn → fp8 → store bytes
        nope_bf16 = torch.randn(
            num_pages, _NOPE_DIM, dtype=torch.bfloat16, device=device
        )
        nope_fp8 = nope_bf16.to(torch.float8_e4m3fn)
        raw[:, t_offset : t_offset + _NOPE_DIM] = nope_fp8.view(torch.uint8)
        # Rope: bf16 randn → store bytes
        rope_bf16 = torch.randn(
            num_pages, _ROPE_DIM, dtype=torch.bfloat16, device=device
        )
        raw[:, t_offset + _NOPE_DIM : t_offset + _NOPE_DIM + _ROPE_DIM * 2] = (
            rope_bf16.view(torch.uint8)
        )

    # Fill scale section at page end: [scale_i(8B)] × page_size
    scale_section_start = page_size * _NOPE_ROPE_STRIDE
    for t in range(page_size):
        s_offset = scale_section_start + t * _SCALE_STRIDE
        # Set 7 tile scales = 127 → exp2(127-127) = 1.0 (unit scale)
        raw[:, s_offset : s_offset + _NUM_TILES] = 127

    # Create as_strided view matching production format
    k_cache = raw.as_strided(
        (num_pages, page_size, 1, 584),
        (total_pb, 584, 584, 1),
    ).view(torch.float8_e4m3fn)

    return k_cache


def make_indices(B, topk, n_valid_list, num_pages, page_size=PAGE_SIZE, s_q=1):
    """Generate token-level indices [B, s_q, topk] with -1 padding."""
    total_tokens = num_pages * page_size
    idx = torch.full((B, s_q, topk), -1, dtype=torch.int32, device=device)
    for b in range(B):
        n = min(n_valid_list[b], topk)
        if n > 0:
            idx[b, 0, :n] = torch.randperm(total_tokens, device=device)[:n].to(
                torch.int32
            )
    return idx


def make_attn_sink(h_q=H_Q):
    """Generate attn_sink [H_q] fp32 with some extreme values."""
    sink = torch.randn(h_q, dtype=torch.float32, device=device)
    mask = torch.randn(h_q, device=device)
    sink[mask > 1.5] = float("inf")
    sink[mask < -1.5] = float("-inf")
    return sink


# ===========================================================================
# Tests — DeepSeek V4 parameters
# ===========================================================================


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("bs", [7, 384, 512])
@pytest.mark.parametrize("num_heads", [64])
@pytest.mark.parametrize("have_extra", [False, True])
@pytest.mark.parametrize("have_attn_sink", [False, True])
@pytest.mark.parametrize("have_topk_length", [False, True])
@pytest.mark.parametrize(
    "page_size,extra_page_size,extra_topk",
    [
        (256, 64, 512),  # DeepSeek V4 default (from trace)
        (256, 2, 64),  # V4 small extra page variant (from trace)
        (256, 2, 8256),  # V4 large extra_topk variant (from trace)
    ],
)
def test_dsv4_sparse_decode_correctness(
    dtype,
    bs,
    num_heads,
    have_extra,
    have_attn_sink,
    have_topk_length,
    page_size,
    extra_page_size,
    extra_topk,
):
    """Correctness test covering all V4 sparse decode parameter combinations."""
    torch.manual_seed(42)

    num_swa_pages = 64
    num_ext_pages = 640

    q = torch.randn(bs, 1, num_heads, D_QK, dtype=dtype, device=device)
    k_cache = make_fp8_kv_cache(num_swa_pages, page_size=page_size)
    indices = make_indices(
        bs, SWA_WINDOW, [SWA_WINDOW] * bs, num_swa_pages, page_size=page_size
    )

    extra_k_cache = (
        make_fp8_kv_cache(num_ext_pages, page_size=extra_page_size)
        if have_extra
        else None
    )
    extra_indices = (
        make_indices(
            bs,
            extra_topk,
            [min(256, extra_topk)] * bs,
            num_ext_pages,
            page_size=extra_page_size,
        )
        if have_extra
        else None
    )
    attn_sink = make_attn_sink(num_heads) if have_attn_sink else None

    topk_length = None
    extra_topk_length = None
    if have_topk_length:
        topk_length = torch.tensor(
            [min(64 + b * 20, SWA_WINDOW) for b in range(bs)],
            dtype=torch.int32,
            device=device,
        )
        if have_extra:
            extra_topk_length = torch.tensor(
                [min(100 + b * 50, extra_topk) for b in range(bs)],
                dtype=torch.int32,
                device=device,
            )
    print(f"\nq shape : {q.shape}, stride : {q.stride()}")
    print(f"k_cache shape : {k_cache.shape}, stride: {k_cache.stride()}")
    print(f"indices shape : {indices.shape}")
    if extra_k_cache is not None:
        print(
            f"extra_k_cache shape : {extra_k_cache.shape}, stride : {extra_k_cache.stride()}"
        )
        print(f"extra_indices shape : {extra_indices.shape}")
    if attn_sink is not None:
        print(f"attn_sink shape : {attn_sink.shape}")
    print(f"topk_length : {topk_length}")
    print(f"extra_topk_length : {extra_topk_length}")
    # print(f"q_nope : {q[0, 0, 1, :448]}")
    # print(f"q_pe : {q[0, 0, 1, 448:]}")
    # For the FP8 packed cache (584 bytes/token layout):
    # k_bytes = k_cache.view(torch.uint8)
    # # Primary K_nope: FP8 nope (first 448 bytes of token 0, page 0)
    # print(f"k_nope: {k_bytes[0, 0, 0, :448].view(torch.float8_e4m3fn)}")
    # # Primary K_pe: bf16 rope (bytes 448..576 of token 0, page 0)
    # k_rope = k_cache.view(torch.uint8)[0, 0, 0, 448 : 448 + 128].view(torch.bfloat16)
    # print(f"k_rope : {k_rope}")
    # # Primary KV_scale
    # raw_flat = k_cache.as_strided(
    #     (k_cache.shape[0], k_cache.stride(0)), (k_cache.stride(0), 1)
    # ).view(torch.uint8)
    # scale_start = page_size * 576
    # print("Scales page0 token0:", raw_flat[0, scale_start : scale_start + 8])
    # # Primary V_nope = K_nope (shared latent in MLA)
    # print(f"v_nope: {k_bytes[0, 0, 0, :448].view(torch.float8_e4m3fn)}")
    # # Primary V_pe = K_pe (shared latent in MLA)
    # print(f"v_rope : {k_rope}")
    # # Extra K_nope
    # extra_bytes = extra_k_cache.view(torch.uint8)
    # print(f"k_nope_extra: {extra_bytes[0, 0, 0, :448].view(torch.float8_e4m3fn)}")
    # # Extra K_pe
    # extra_rope = extra_k_cache.view(torch.uint8)[0, 0, 0, 448 : 448 + 128].view(
    #     torch.bfloat16
    # )
    # print(f"k_rope_extra : {extra_rope}")
    # # Extra KV_scale
    # extra_raw_flat = extra_k_cache.as_strided(
    #     (extra_k_cache.shape[0], extra_k_cache.stride(0)), (extra_k_cache.stride(0), 1)
    # ).view(torch.uint8)
    # extra_scale_start = extra_page_size * 576
    # print(
    #     "Scales_extra page0 token0:",
    #     extra_raw_flat[0, extra_scale_start : extra_scale_start + 8],
    # )
    # # Extra V_nope = Extra K_nope (shared latent)
    # print(f"v_nope_extra: {extra_bytes[0, 0, 0, :448].view(torch.float8_e4m3fn)}")
    # # Extra V_pe = Extra K_pe (shared latent)
    # print(f"v_rope_extra : {extra_rope}")
    out, lse = call_kernel(
        q,
        k_cache,
        indices,
        attn_sink,
        extra_k_cache,
        extra_indices,
        topk_length=topk_length,
        extra_topk_length=extra_topk_length,
    )
    ref_out, ref_lse = _sm120_sparse_decode_fwd(
        q,
        k_cache,
        indices,
        topk_length=topk_length,
        attn_sink=attn_sink,
        head_dim_v=D_V,
        softmax_scale=SM_SCALE,
        extra_k_cache=extra_k_cache,
        extra_indices=extra_indices,
        extra_topk_length=extra_topk_length,
    )

    torch.testing.assert_close(out.float(), ref_out.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("bs", [7, 384])
@pytest.mark.parametrize("extra_topk_valid", [0, 50, 512])
def test_dsv4_variable_extra_topk(bs, extra_topk_valid):
    """Variable number of valid extra tokens per batch."""
    torch.manual_seed(42)

    num_swa_pages = 64
    num_ext_pages = 640
    extra_topk = 512

    q = torch.randn(bs, 1, H_Q, D_QK, dtype=torch.bfloat16, device=device).clamp(-1, 1)
    k_cache = make_fp8_kv_cache(num_swa_pages, page_size=PAGE_SIZE)
    indices = make_indices(
        bs, SWA_WINDOW, [SWA_WINDOW] * bs, num_swa_pages, page_size=PAGE_SIZE
    )

    extra_k_cache = make_fp8_kv_cache(num_ext_pages, page_size=64)
    valid_per_batch = [min(extra_topk_valid + b * 10, extra_topk) for b in range(bs)]
    extra_indices = make_indices(
        bs, extra_topk, valid_per_batch, num_ext_pages, page_size=64
    )

    out, lse = call_kernel(q, k_cache, indices, None, extra_k_cache, extra_indices)
    ref_out, ref_lse = _sm120_sparse_decode_fwd(
        q,
        k_cache,
        indices,
        topk_length=None,
        attn_sink=None,
        head_dim_v=D_V,
        softmax_scale=SM_SCALE,
        extra_k_cache=extra_k_cache,
        extra_indices=extra_indices,
        extra_topk_length=None,
    )

    torch.testing.assert_close(out.float(), ref_out.float(), atol=2e-2, rtol=2e-2)


def test_dsv4_attn_sink_dampens_output():
    """Large positive attn_sink should scale output toward zero."""
    torch.manual_seed(42)
    bs = 7

    q = torch.randn(bs, 1, H_Q, D_QK, dtype=torch.bfloat16, device=device).clamp(-1, 1)
    k_cache = make_fp8_kv_cache(64, page_size=PAGE_SIZE)
    indices = make_indices(bs, SWA_WINDOW, [SWA_WINDOW] * bs, 64, page_size=PAGE_SIZE)

    large_sink = torch.full((H_Q,), 100.0, dtype=torch.float32, device=device)
    out, _ = call_kernel(q, k_cache, indices, large_sink)

    assert (
        out.abs().max() < 1e-3
    ), f"Large attn_sink should zero output, got max={out.abs().max()}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
