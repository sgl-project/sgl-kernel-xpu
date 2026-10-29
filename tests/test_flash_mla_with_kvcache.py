"""
Tests for Sparse MLA Decode.

Reference implementation: _sm120_sparse_decode_fwd inlined from
https://github.com/AliceChenyy/sglang/blob/7cc3aa4819525d9d95f048786eb21853b08cbade/
  python/sglang/srt/layers/attention/flash_mla_sm120_fallback.py

(A) DeepSeek V4 — d_qk=512, 584 B/token, page-internal SECTIONS:
  Physical page (total_page_bytes = page_size*576 + ceil_576(page_size*8)):
    [0 .. page_size*576)           Token data section
      Per token (576 bytes):
        bytes   0-447: K_nope  FP8_E4M3  (448B = 7 tiles × 64)
        bytes 448-575: K_rope  BF16      (128B = 64 dims × 2 bytes)
    [page_size*576 .. end)         Scale section (padded to 576-byte boundary)
      Per token (8 bytes):
        bytes 0-6: 7 nope tile scales UE8M0
        byte  7:   1 reserved

(B) DeepSeek V3.2 / GLM-DSA — d_qk=576, 656 B/token, one CONTIGUOUS record per token:
    bytes   0-511: K_nope  FP8_E4M3  (512B = 4 tiles × 128)
    bytes 512-527: 4 nope tile scales FP32
    bytes 528-655: K_rope  BF16      (128B = 64 dims × 2 bytes)

"""

import gc
import math
import os
import sys
from typing import NamedTuple, Optional

import pytest
import torch
from sgl_kernel import flash_mla_with_kvcache
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

# ── Layout-independent constants (from the DeepSeek V4 production trace) ──
D_V = 512
H_Q = 64
H_KV = 1
SWA_WINDOW = 128
PAGE_SIZE = 256
SMOKE_POOL_TOKENS = 16384
_GATHER_CHUNK = 16384  # tokens per chunk; ~16k * 1024 B ≈ 16 MiB output per chunk

# Per-chunk peak-memory budget for the sparse decode fallback (MiB).  Read
# once at import time so the forward path doesn't pay an os.environ lookup
# per layer per decode step.
_SM120_SPARSE_CHUNK_MIB = int(os.environ.get("SGLANG_SM120_SPARSE_CHUNK_MIB", "256"))


class KvLayout(NamedTuple):
    """Byte geometry of one packed FP8 KV cache layout, keyed by d_qk.

    This class encapsulates the layout of the KV cache for a given d_qk, including
    the dimensions of the NoPE and RoPE components, the tile size for NoPE, and
    whether scales are stored inline and in FP32 format.
    """

    name: str
    nope_dim: int
    rope_dim: int
    tile_size: int  # nope values sharing one scale
    scales_inline: bool  # inline in the token record vs a separate page-end section
    scales_fp32: bool  # fp32 scales vs UE8M0 exponent bytes

    @property
    def d_qk(self) -> int:
        return self.nope_dim + self.rope_dim

    @property
    def num_tiles(self) -> int:
        return self.nope_dim // self.tile_size

    @property
    def scale_bytes(self) -> int:
        """Per-token scale footprint: 4 B per fp32 tile scale, or 1 B per UE8M0 + 1 pad."""
        return self.num_tiles * 4 if self.scales_fp32 else self.num_tiles + 1

    @property
    def token_stride(self) -> int:
        """Spacing between consecutive token records in the data region."""
        inline = self.scale_bytes if self.scales_inline else 0
        return self.nope_dim + inline + self.rope_dim * 2

    @property
    def rope_byte_offset(self) -> int:
        """Where RoPE starts in a token record (after NoPE, and after inline scales)."""
        return self.nope_dim + (self.scale_bytes if self.scales_inline else 0)

    @property
    def head_bytes(self) -> int:
        """k_cache.shape[-1] -- the per-token width the kernel validates d_qk against."""
        return self.token_stride + (0 if self.scales_inline else self.scale_bytes)

    @property
    def sm_scale(self) -> float:
        return 1.0 / math.sqrt(self.d_qk)

    def total_page_bytes(self, page_size: int) -> int:
        """Real byte stride between pages (k_cache.stride(0))."""
        data = page_size * self.token_stride
        if self.scales_inline:
            # Records are self-contained, so a page is exactly its data region.
            return data
        # DSv4: the page-end scale section is padded up to a 576-byte boundary.
        scale_section = page_size * self.scale_bytes
        return data + ((scale_section + 575) // 576) * 576


DSV4_LAYOUT = KvLayout(
    name="dsv4-584",
    nope_dim=448,
    rope_dim=64,
    tile_size=64,
    scales_inline=False,
    scales_fp32=False,
)
DSA_LAYOUT = KvLayout(
    name="dsa-656",
    nope_dim=512,
    rope_dim=64,
    tile_size=128,
    scales_inline=True,
    scales_fp32=True,
)
ALL_LAYOUTS = [DSV4_LAYOUT, DSA_LAYOUT]

# Pin the derived geometry to the two production widths the kernel dispatches on.
assert (DSV4_LAYOUT.d_qk, DSV4_LAYOUT.head_bytes) == (512, 584), DSV4_LAYOUT
assert (DSA_LAYOUT.d_qk, DSA_LAYOUT.head_bytes) == (576, 656), DSA_LAYOUT
assert (DSA_LAYOUT.rope_byte_offset, DSA_LAYOUT.num_tiles) == (528, 4), DSA_LAYOUT


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
def _gather_and_dequant(k_cache, indices, page_size, layout):
    """Gather KV entries from the paged buffer using correct page-internal addressing.

    Args:
        k_cache: (num_pages, page_size, 1, layout.head_bytes) float8_e4m3fn.
                 May be a non-contiguous view of the raw page buffer (DSv4).
        indices: (...) int32/int64, token-level indices. Invalid indices are
                 expected to already be clamped into [0, num_pages*page_size).
        page_size: tokens per page (e.g. 256, 64, 2)
        layout: KvLayout deciding where a token's scale bytes live and how they decode.

    Returns:
        kv: (..., layout.d_qk) bfloat16, dequantized KV vectors
    """
    idx_shape = indices.shape
    flat_idx = indices.reshape(-1)  # (N,)
    N = flat_idx.shape[0]
    device = k_cache.device

    page_bytes = k_cache.stride(0)  # actual byte stride between pages
    num_pages = k_cache.shape[0]

    # Flatten the raw byte buffer so we can gather with a single int64 index
    # per byte instead of paying for a full (N, nope_dim) int64 index tensor up
    # front. flat_buf has nelems = num_pages * page_bytes uint8.
    raw_pages = k_cache.as_strided(
        (num_pages, page_bytes),
        (page_bytes, 1),
    ).view(torch.uint8)
    flat_buf = raw_pages.reshape(-1)

    scale_section_offset = page_size * layout.token_stride

    nope_arange = torch.arange(layout.nope_dim, device=device, dtype=torch.long)
    rope_arange = torch.arange(layout.rope_dim * 2, device=device, dtype=torch.long)
    # UE8M0 scales are 1 byte each (the pad byte is skipped); fp32 scales are 4.
    scale_byte_width = 4 if layout.scales_fp32 else 1
    scale_arange = torch.arange(
        layout.num_tiles * scale_byte_width, device=device, dtype=torch.long
    )

    result = torch.empty(N, layout.d_qk, dtype=torch.bfloat16, device=device)

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
        rec_base = page_base + offsets.to(torch.long) * layout.token_stride  # (n,)

        nope_idx = rec_base.unsqueeze(-1) + nope_arange  # (n, nope_dim)
        rope_idx = rec_base.unsqueeze(-1) + (
            layout.rope_byte_offset + rope_arange
        )  # (n, 2*rope_dim)
        if layout.scales_inline:
            # DSv3.2 / GLM-DSA: scales sit inside the record, right after NoPE.
            scale_idx = rec_base.unsqueeze(-1) + layout.nope_dim + scale_arange
        else:
            # DSv4: scales live in a page-end section indexed by in-page token offset.
            scale_idx = (
                page_base.unsqueeze(-1)
                + scale_section_offset
                + offsets.to(torch.long).unsqueeze(-1) * layout.scale_bytes
                + scale_arange
            )

        nope_bytes = flat_buf[nope_idx.reshape(-1)].view(n, layout.nope_dim)
        rope_bytes = flat_buf[rope_idx.reshape(-1)].view(n, layout.rope_dim * 2)
        scale_bytes = flat_buf[scale_idx.reshape(-1)].view(
            n, layout.num_tiles * scale_byte_width
        )

        nope_fp8 = nope_bytes.view(torch.float8_e4m3fn)  # (n, nope_dim)
        rope_bf16 = rope_bytes.contiguous().view(torch.bfloat16)  # (n, rope_dim)
        scales = scale_bytes.contiguous().view(
            torch.float32 if layout.scales_fp32 else torch.float8_e8m0fnu
        )  # (n, num_tiles)

        result[start:end, : layout.nope_dim] = (
            (
                nope_fp8.view(n, layout.num_tiles, layout.tile_size).float()
                * scales.view(n, layout.num_tiles, 1).float()
            )
            .view(n, layout.nope_dim)
            .to(torch.bfloat16)
        )
        result[start:end, layout.nope_dim :] = rope_bf16

    return result.reshape(*idx_shape, layout.d_qk)


def _sm120_sparse_decode_fwd(
    q,
    k_cache,
    indices,
    topk_length,
    attn_sink,
    head_dim_v,
    softmax_scale,
    layout,
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
    # chunk * total_topk * d_qk * 2 bytes; fp32 working set adds ~3x on top.
    # On Intel L0, per-launch overhead is high (~hundreds of us), so prefer
    # fewer/larger chunks. Target 256 MiB peak (override via
    # SGLANG_SM120_SPARSE_CHUNK_MIB at import time).
    bytes_per_row = total_topk * layout.d_qk * 2
    chunk_rows = max(
        1, min(R, (_SM120_SPARSE_CHUNK_MIB * 1024 * 1024) // max(1, bytes_per_row))
    )

    for start in range(0, R, chunk_rows):
        end = min(start + chunk_rows, R)
        n = end - start

        # Gather KV for this chunk only.
        kv_chunk = _gather_and_dequant(
            k_cache, safe_indices_rows[start:end], page_size, layout
        )  # (n, topk, d_qk)
        inv_chunk = invalid_rows[start:end]  # (n, topk)
        if have_extra:
            extra_kv_chunk = _gather_and_dequant(
                extra_k_cache, extra_safe_rows[start:end], extra_page_size, layout
            )  # (n, extra_topk, d_qk)
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
    *,
    layout,
):
    """Calls flash_mla_sparse_decode with a packed FP8 KV cache in `layout`'s format."""
    return flash_mla_with_kvcache(
        q=q,
        k_cache=k_cache,
        block_table=None,
        cache_seqlens=None,
        head_dim_v=D_V,
        tile_scheduler_metadata=None,
        num_splits=None,
        softmax_scale=layout.sm_scale,
        causal=False,
        is_fp8_kvcache=True,
        indices=indices,
        attn_sink=attn_sink,
        extra_k_cache=extra_k_cache,
        extra_indices_in_kvcache=extra_indices,
        topk_length=topk_length,
        extra_topk_length=extra_topk_length,
    )


def _tile_scale_values(num_tiles, varied):
    """Per-tile NoPE scale multipliers. Powers of two so UE8M0 can represent them exactly.

    Unit scales (varied=False) keep the historical DSv4 fixture behavior. Varied scales
    make a wrong scale-group index or a mis-decoded scale section observable, which unit
    scales cannot.
    """
    if not varied:
        return [1.0] * num_tiles
    return [2.0 ** ((t % 3) - 1) for t in range(num_tiles)]  # 0.5, 1.0, 2.0, ...


def make_fp8_kv_cache(num_pages, layout, page_size=PAGE_SIZE, varied_scales=False):
    """Create a packed FP8 KV cache in `layout`'s production byte format.

    DSv4 (page-end scales):
      [0 .. page_size*576)    Token data: [nope0|rope0|nope1|rope1|...|nopeN|ropeN]
      [page_size*576 .. end)  Scale section: [scale0|...|scaleN] UE8M0 (padded to 576)

    DSv3.2 / GLM-DSA (inline scales): page_size contiguous self-contained records
      [nope(512B) | scales(16B fp32) | rope(128B)]
    """
    total_pb = layout.total_page_bytes(page_size)
    raw = torch.zeros(num_pages, total_pb, dtype=torch.uint8, device=device)
    scale_values = _tile_scale_values(layout.num_tiles, varied_scales)

    # Token data records: NoPE, then RoPE at the layout's offset (past inline scales).
    for t in range(page_size):
        rec = t * layout.token_stride
        nope_bf16 = torch.randn(
            num_pages, layout.nope_dim, dtype=torch.bfloat16, device=device
        )
        nope_fp8 = nope_bf16.to(torch.float8_e4m3fn)
        raw[:, rec : rec + layout.nope_dim] = nope_fp8.view(torch.uint8)

        rope_bf16 = torch.randn(
            num_pages, layout.rope_dim, dtype=torch.bfloat16, device=device
        )
        r0 = rec + layout.rope_byte_offset
        raw[:, r0 : r0 + layout.rope_dim * 2] = rope_bf16.view(torch.uint8)

        if layout.scales_inline:
            # fp32 tile scales, inline immediately after NoPE.
            s0 = rec + layout.nope_dim
            scales = torch.tensor(
                scale_values, dtype=torch.float32, device=device
            ).expand(num_pages, layout.num_tiles)
            raw[:, s0 : s0 + layout.scale_bytes] = scales.contiguous().view(torch.uint8)

    if not layout.scales_inline:
        # UE8M0 page-end section: byte b encodes 2^(b-127), so 127 is a unit scale.
        scale_bytes = torch.tensor(
            [127 + int(math.log2(v)) for v in scale_values],
            dtype=torch.uint8,
            device=device,
        )
        scale_section_start = page_size * layout.token_stride
        for t in range(page_size):
            s0 = scale_section_start + t * layout.scale_bytes
            raw[:, s0 : s0 + layout.num_tiles] = scale_bytes

    hb = layout.head_bytes
    return raw.as_strided(
        (num_pages, page_size, 1, hb),
        (total_pb, hb, hb, 1),
    ).view(torch.float8_e4m3fn)


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


@pytest.mark.arch("xe20", "xe35")
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("layout", ALL_LAYOUTS, ids=lambda lo: lo.name)
@pytest.mark.parametrize("bs", [7, 384, 512])
@pytest.mark.parametrize("num_heads", [16, 32, 64])
@pytest.mark.parametrize("have_extra", [False, True])
@pytest.mark.parametrize("have_attn_sink", [False, True])
@pytest.mark.parametrize("have_topk_length", [False, True])
@pytest.mark.parametrize("variable_topk", [False, True])
@pytest.mark.parametrize(
    "page_size,num_swa_pages,extra_page_size,extra_topk,num_ext_pages",
    [
        (256, 64, 64, 512, 640),
        (256, 64, 2, 64, 640),
        (256, 64, 2, 8256, 640),
        (64, 256, 64, 512, 160),
    ],
)
def test_sparse_decode_correctness(
    dtype,
    layout,
    bs,
    num_heads,
    have_extra,
    have_attn_sink,
    have_topk_length,
    variable_topk,
    page_size,
    num_swa_pages,
    extra_page_size,
    extra_topk,
    num_ext_pages,
):
    torch.manual_seed(42)
    assert num_swa_pages * page_size >= SWA_WINDOW

    q = torch.randn(bs, 1, num_heads, layout.d_qk, dtype=dtype, device=device)
    k_cache = make_fp8_kv_cache(
        num_swa_pages, layout, page_size=page_size, varied_scales=True
    )
    assert k_cache.shape[-1] == layout.head_bytes
    indices = make_indices(
        bs, SWA_WINDOW, [SWA_WINDOW] * bs, num_swa_pages, page_size=page_size
    )

    extra_k_cache = (
        make_fp8_kv_cache(
            num_ext_pages, layout, page_size=extra_page_size, varied_scales=True
        )
        if have_extra
        else None
    )
    extra_max_valid = min(extra_topk, num_ext_pages * extra_page_size)
    if variable_topk:
        extra_valid_list = [min(b * 10, extra_max_valid) for b in range(bs)]
    else:
        extra_valid_list = [min(256, extra_max_valid)] * bs
    extra_indices = (
        make_indices(
            bs,
            extra_topk,
            extra_valid_list,
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
    out, lse = call_kernel(
        q,
        k_cache,
        indices,
        attn_sink,
        extra_k_cache,
        extra_indices,
        topk_length=topk_length,
        extra_topk_length=extra_topk_length,
        layout=layout,
    )
    ref_out, ref_lse = _sm120_sparse_decode_fwd(
        q,
        k_cache,
        indices,
        topk_length=topk_length,
        attn_sink=attn_sink,
        head_dim_v=D_V,
        softmax_scale=layout.sm_scale,
        layout=layout,
        extra_k_cache=extra_k_cache,
        extra_indices=extra_indices,
        extra_topk_length=extra_topk_length,
    )

    torch.testing.assert_close(out.float(), ref_out.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse, ref_lse, atol=1e-3, rtol=1e-3)


@pytest.mark.arch("xe20", "xe35")
@pytest.mark.parametrize("layout", ALL_LAYOUTS, ids=lambda lo: lo.name)
def test_attn_sink_dampens_output(layout):
    """Large positive attn_sink should scale output toward zero."""
    torch.manual_seed(42)
    bs = 7

    q = torch.randn(bs, 1, H_Q, layout.d_qk, dtype=torch.bfloat16, device=device)
    num_pages = SMOKE_POOL_TOKENS // PAGE_SIZE
    k_cache = make_fp8_kv_cache(num_pages, layout, page_size=PAGE_SIZE)
    indices = make_indices(
        bs, SWA_WINDOW, [SWA_WINDOW] * bs, num_pages, page_size=PAGE_SIZE
    )

    large_sink = torch.full((H_Q,), 100.0, dtype=torch.float32, device=device)
    out, _ = call_kernel(q, k_cache, indices, large_sink, layout=layout)

    assert (
        out.abs().max() < 1e-3
    ), f"Large attn_sink should zero output, got max={out.abs().max()}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
