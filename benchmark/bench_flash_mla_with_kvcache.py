"""Benchmark: flash_mla_sparse_decode — Triton V4 vs sgl_kernel.

Compares execution time and effective bandwidth of both implementations
across DeepSeek-V4 production shapes (varying B, topk, extra_topk).

Usage:
  python benchmark/bench_flash_mla_sparse_decode.py
"""

import math
from itertools import product
from typing import NamedTuple, Optional, Tuple

import torch
import triton
import triton.language as tl
from _bench import BenchSpec, ProblemShape, format_tables, make_row, require_arch
from sgl_kernel import flash_mla_with_kvcache

# ── Layout-independent constants (DeepSeek V4 production trace) ──
D_V = 512  # value head dim (output width); identical for both layouts
H_PER_RANK = 16
PAGE_SIZE = 256


class KvLayout(NamedTuple):
    """Byte geometry of one packed FP8 KV cache layout, keyed by d_qk.

    DSv4 : d_qk=512, 584 B/token, page-end UE8M0 scale section.
    DSA  : d_qk=576, 656 B/token, inline fp32 scales per token record.
    """

    name: str
    nope_dim: int
    rope_dim: int
    tile_size: int  # nope values sharing one scale
    scales_inline: bool  # inline in record vs separate page-end section
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
            return data
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
LAYOUTS = {lo.d_qk: lo for lo in ALL_LAYOUTS}

# Pin the derived geometry to the two production widths the kernel dispatches on.
assert (DSV4_LAYOUT.d_qk, DSV4_LAYOUT.head_bytes) == (512, 584), DSV4_LAYOUT
assert (DSA_LAYOUT.d_qk, DSA_LAYOUT.head_bytes) == (576, 656), DSA_LAYOUT
assert (DSA_LAYOUT.rope_byte_offset, DSA_LAYOUT.num_tiles) == (528, 4), DSA_LAYOUT

SPEC = BenchSpec(
    kernel="flash_mla_sparse_decode",
    shape=ProblemShape(
        (
            "b",
            "h_q",
            "d_qk",
            "topk",
            "extra_topk",
            "num_pages",
            "page_size",
            "extra_num_pages",
            "extra_page_size",
        ),
        description="fp8 paged sparse MLA decode; d_qk ∈ {512, 576}, extra_* is the optional second KV pool",
    ),
    metrics=("time_us", "bandwidth_gbs"),
)


# ============================================================================
# Triton V4: Gather + dequant kernel
# ============================================================================
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_T": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_T": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_T": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_T": 128}, num_warps=8, num_stages=2),
    ],
    key=["topk"],
)
@triton.jit
def _gather_dequant_kernel(
    cache_fp8_ptr,
    cache_uint8_ptr,
    cache_bf16_ptr,
    cache_fp32_ptr,
    indices_ptr,
    out_ptr,
    page_size: tl.int32,
    page_bytes: tl.int64,
    token_stride: tl.int64,
    scale_section_off: tl.int64,
    rope_byte_offset: tl.int32,
    scale_stride: tl.int32,
    topk: tl.int32,
    stride_ib: tl.int32,
    stride_ob: tl.int32,
    stride_ot: tl.int32,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    NUM_TILES: tl.constexpr,
    SCALES_INLINE: tl.constexpr,
    SCALES_FP32: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    bid = tl.program_id(0)
    tile_id = tl.program_id(1)

    t_offs = tile_id * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_offs < topk

    raw_indices = tl.load(
        indices_ptr + bid * stride_ib + t_offs,
        mask=t_mask,
        other=-1,
    )
    idx_valid = t_mask & (raw_indices >= 0)
    safe_indices = tl.where(idx_valid, raw_indices, tl.zeros_like(raw_indices))

    page_ids = (safe_indices // page_size).to(tl.int64)
    page_offs = (safe_indices % page_size).to(tl.int64)
    token_data_bases = page_ids * page_bytes + page_offs * token_stride

    if SCALES_INLINE:
        # DSv3.2 / GLM-DSA: fp32 scales sit inside the record, right after NoPE.
        scale_byte_bases = token_data_bases + NOPE_DIM
    else:
        # DSv4: UE8M0 scales live in a page-end section indexed by in-page offset.
        scale_byte_bases = (
            page_ids * page_bytes + scale_section_off + page_offs * scale_stride
        )

    for g in tl.static_range(NUM_TILES):
        d_start = g * TILE_SIZE
        d_offs = tl.arange(0, TILE_SIZE)
        d_abs = d_start + d_offs
        d_mask = d_abs < NOPE_DIM

        fp8_addrs = token_data_bases[:, None] + (d_abs[None, :]).to(tl.int64)
        load_mask = idx_valid[:, None] & d_mask[None, :]
        fp8_vals = tl.load(cache_fp8_ptr + fp8_addrs, mask=load_mask, other=0.0)

        if SCALES_FP32:
            scale_elem = (scale_byte_bases + g * 4) // 4
            scale_f32 = tl.load(cache_fp32_ptr + scale_elem, mask=idx_valid, other=1.0)
        else:
            scale_val = tl.load(
                cache_uint8_ptr + scale_byte_bases + g,
                mask=idx_valid,
                other=127,
            )
            scale_f32 = tl.math.exp2(scale_val.to(tl.float32) - 127.0)

        bf16_vals = (fp8_vals.to(tl.float32) * scale_f32[:, None]).to(tl.bfloat16)
        bf16_vals = tl.where(load_mask, bf16_vals, tl.zeros_like(bf16_vals))

        out_addrs = bid * stride_ob + t_offs[:, None] * stride_ot + d_abs[None, :]
        tl.store(out_ptr + out_addrs, bf16_vals, mask=load_mask)

    rope_offs = tl.arange(0, ROPE_DIM)
    rope_byte_bases = token_data_bases + rope_byte_offset
    rope_elem_bases = (rope_byte_bases // 2).to(tl.int64)
    rope_addrs = rope_elem_bases[:, None] + rope_offs[None, :].to(tl.int64)
    rope_vals = tl.load(
        cache_bf16_ptr + rope_addrs,
        mask=idx_valid[:, None],
        other=0.0,
    )

    out_rope_addrs = (
        bid * stride_ob + t_offs[:, None] * stride_ot + (NOPE_DIM + rope_offs)[None, :]
    )
    tl.store(
        out_ptr + out_rope_addrs,
        rope_vals.to(tl.bfloat16),
        mask=idx_valid[:, None],
    )


# ============================================================================
# Triton V4: Python helpers
# ============================================================================
def _gather_kv_pages(
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    layout: "KvLayout",
) -> torch.Tensor:
    B = indices.shape[0]
    topk = indices.shape[1]
    num_pages = k_cache.shape[0]
    page_size = k_cache.shape[1]
    page_bytes = k_cache.stride(0)

    total_elems = num_pages * page_bytes
    raw_fp8 = k_cache.as_strided((total_elems,), (1,))
    raw_uint8 = raw_fp8.view(torch.uint8)
    raw_bf16 = raw_uint8.view(torch.bfloat16)
    raw_fp32 = raw_uint8.view(torch.float32)

    kv_dense = torch.zeros(
        B, topk, layout.d_qk, dtype=torch.bfloat16, device=k_cache.device
    )

    if topk_length is not None:
        arange = torch.arange(topk, device=indices.device)
        invalid = arange.unsqueeze(0) >= topk_length.unsqueeze(1)
        indices = indices.clone()
        indices[invalid] = -1

    grid = lambda meta: (B, triton.cdiv(topk, meta["BLOCK_T"]))
    _gather_dequant_kernel[grid](
        raw_fp8,
        raw_uint8,
        raw_bf16,
        raw_fp32,
        indices,
        kv_dense,
        page_size,
        int(page_bytes),
        layout.token_stride,
        int(page_size * layout.token_stride),
        layout.rope_byte_offset,
        layout.scale_bytes,
        topk,
        indices.stride(0),
        kv_dense.stride(0),
        kv_dense.stride(1),
        NOPE_DIM=layout.nope_dim,
        ROPE_DIM=layout.rope_dim,
        TILE_SIZE=layout.tile_size,
        NUM_TILES=layout.num_tiles,
        SCALES_INLINE=layout.scales_inline,
        SCALES_FP32=layout.scales_fp32,
    )

    return kv_dense


def _build_invalid_mask(
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
) -> torch.Tensor:
    B, topk = indices.shape
    mask = indices < 0
    if topk_length is not None:
        arange = torch.arange(topk, device=indices.device)
        mask = mask | (arange.unsqueeze(0) >= topk_length.unsqueeze(1))
    return mask.unsqueeze(1)


def _compute_attention(
    q_3d: torch.Tensor,
    kv_dense: torch.Tensor,
    invalid_mask: torch.Tensor,
    softmax_scale: float,
    head_dim_v: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scores = torch.matmul(q_3d, kv_dense.transpose(-1, -2))
    scores = scores.float() * softmax_scale
    scores.masked_fill_(invalid_mask, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    p = torch.softmax(scores, dim=-1)
    del scores
    p = torch.nan_to_num(p, 0.0)
    out = torch.matmul(p.to(torch.bfloat16), kv_dense[..., :head_dim_v])
    del p
    return out, lse


def _merge_partial_attn(
    out1: torch.Tensor,
    lse1: torch.Tensor,
    out2: torch.Tensor,
    lse2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_lse = torch.maximum(lse1, lse2)
    exp1 = torch.exp(lse1 - max_lse)
    exp2 = torch.exp(lse2 - max_lse)
    exp1 = torch.where(lse1 > -1e20, exp1, torch.zeros_like(exp1))
    exp2 = torch.where(lse2 > -1e20, exp2, torch.zeros_like(exp2))
    total = (exp1 + exp2).clamp_(min=1e-20)

    merged = out1.float()
    del out1
    merged.mul_(exp1.unsqueeze(-1))

    tmp = out2.float()
    del out2
    tmp.mul_(exp2.unsqueeze(-1))
    merged.add_(tmp)
    del tmp

    merged.div_(total.unsqueeze(-1))
    merged_lse = max_lse + torch.log(total)
    return merged.to(torch.bfloat16), merged_lse


def flash_mla_sparse_decode_triton(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    attn_sink: Optional[torch.Tensor],
    head_dim_v: int,
    softmax_scale: float,
    layout: "KvLayout",
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    B, _, H, D = q.shape
    q_3d = q.squeeze(1)
    if not q_3d.is_contiguous():
        q_3d = q_3d.contiguous()

    flat_indices = indices.reshape(B, -1).contiguous()

    kv_dense = _gather_kv_pages(k_cache, flat_indices, topk_length, layout)
    invalid_mask = _build_invalid_mask(flat_indices, topk_length)

    out, lse = _compute_attention(
        q_3d, kv_dense, invalid_mask, softmax_scale, head_dim_v
    )
    del kv_dense, invalid_mask

    if extra_k_cache is not None and extra_indices is not None:
        extra_flat = extra_indices.reshape(B, -1).contiguous()
        kv_extra = _gather_kv_pages(
            extra_k_cache, extra_flat, extra_topk_length, layout
        )
        extra_mask = _build_invalid_mask(extra_flat, extra_topk_length)

        out_extra, lse_extra = _compute_attention(
            q_3d,
            kv_extra,
            extra_mask,
            softmax_scale,
            head_dim_v,
        )
        del kv_extra, extra_mask, extra_flat

        out, lse = _merge_partial_attn(out, lse, out_extra, lse_extra)
        del out_extra, lse_extra

    if attn_sink is not None:
        lse_f32 = lse.float() if lse.dtype != torch.float32 else lse
        w = 1.0 / (1.0 + torch.exp(attn_sink.view(1, -1) - lse_f32))
        out = out.float().mul_(w.unsqueeze(-1)).to(torch.bfloat16)

    lonely = lse == float("-inf")
    if lonely.any():
        out = out.masked_fill(lonely.unsqueeze(-1), 0.0)
    lse = lse.masked_fill(lonely, float("+inf"))

    out = out.to(torch.bfloat16).unsqueeze(1)
    lse = lse.unsqueeze(1)
    return out, lse.permute(0, 2, 1)


# ============================================================================
# KV cache construction
# ============================================================================
def _tile_scale_values(num_tiles, varied):
    """Per-tile NoPE scale multipliers. Powers of two so UE8M0 can represent them exactly."""
    if not varied:
        return [1.0] * num_tiles
    return [2.0 ** ((t % 3) - 1) for t in range(num_tiles)]  # 0.5, 1.0, 2.0, ...


def make_fp8_kv_cache(
    num_pages, layout, device, page_size=PAGE_SIZE, varied_scales=False
):
    """Create a packed FP8 KV cache in `layout`'s production byte format."""
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


def make_indices(B, topk, num_pages, device, page_size=PAGE_SIZE, s_q=1):
    """Token-level indices [B, s_q, topk] drawn without replacement per row."""
    total_tokens = num_pages * page_size
    idx = torch.full((B, s_q, topk), -1, dtype=torch.int32, device=device)
    n = min(topk, total_tokens)
    for b in range(B):
        idx[b, 0, :n] = torch.randperm(total_tokens, device=device)[:n].to(torch.int32)
    return idx


# ============================================================================
# Input construction
# ============================================================================
def build_inputs(
    B,
    topk,
    extra_topk,
    num_pages,
    page_size,
    H,
    layout,
    device,
    extra_num_pages=0,
    extra_page_size=64,
):
    k_cache = make_fp8_kv_cache(
        num_pages, layout, device, page_size=page_size, varied_scales=True
    )
    indices = make_indices(B, topk, num_pages, device, page_size=page_size)
    q = torch.randn(B, 1, H, layout.d_qk, dtype=torch.bfloat16, device=device)
    topk_length = torch.full((B,), topk, dtype=torch.int32, device=device)
    attn_sink = torch.randn(H, dtype=torch.float32, device=device) * 0.1

    inputs = {
        "q": q,
        "k_cache": k_cache,
        "indices": indices,
        "topk_length": topk_length,
        "attn_sink": attn_sink,
        "head_dim_v": D_V,
        "layout": layout,
    }

    if extra_topk > 0:
        # extra pool defaults to the main pool's sizing unless overridden.
        extra_pages = extra_num_pages or num_pages
        extra_cache = make_fp8_kv_cache(
            extra_pages, layout, device, page_size=extra_page_size, varied_scales=True
        )
        extra_indices = make_indices(
            B, extra_topk, extra_pages, device, page_size=extra_page_size
        )
        extra_topk_length = torch.full(
            (B,), extra_topk, dtype=torch.int32, device=device
        )
        inputs["extra_k_cache"] = extra_cache
        inputs["extra_indices"] = extra_indices
        inputs["extra_topk_length"] = extra_topk_length

    return inputs


# ============================================================================
# Bandwidth calculation
# ============================================================================
def _compute_total_bytes(B, topk, extra_topk, H, layout):
    read_q = B * H * layout.d_qk * 2
    read_kv = B * topk * layout.head_bytes
    read_extra_kv = B * extra_topk * layout.head_bytes if extra_topk > 0 else 0
    read_indices = B * (topk + extra_topk) * 4
    write_out = B * H * D_V * 2
    write_lse = B * H * 4
    return read_q + read_kv + read_extra_kv + read_indices + write_out + write_lse


# ============================================================================
# Benchmark configuration
# ============================================================================
class DecodeConfig(NamedTuple):
    b: int
    h_q: int
    topk: int
    extra_topk: int
    num_pages: int
    page_size: int = PAGE_SIZE
    extra_num_pages: int = 0
    extra_page_size: int = 0
    sm_scale: float | None = None
    d_qk: int = 512  # 512 -> DSv4 (584 B/token); 576 -> DSv3.2/GLM-DSA (656 B/token)


batch_size_range = [1, 8, 42, 128, 256]
topk_range = [64, 128, 512]
extra_topk_range = [0, 512]

# ---- DECODE-stage on synthetic data ----
configs = [
    DecodeConfig(
        b,
        H_PER_RANK,
        topk,
        extra_topk,
        num_pages=512,
        page_size=PAGE_SIZE,
        extra_num_pages=512,
        extra_page_size=64,
    )
    for b, topk, extra_topk in product(batch_size_range, topk_range, extra_topk_range)
]

# ---- DECODE-stage for DEEPSEEK V4 flash ----
configs += [
    # conc=2
    DecodeConfig(
        2,
        64,
        128,
        128,
        num_pages=103,
        page_size=PAGE_SIZE,
        extra_num_pages=1025,
        extra_page_size=256,
    ),
    DecodeConfig(
        2,
        64,
        128,
        512,
        num_pages=103,
        page_size=PAGE_SIZE,
        extra_num_pages=1025,
        extra_page_size=256,
    ),
    # conc=8, steady-state, no extra pool
    DecodeConfig(8, 64, 128, 0, num_pages=103, page_size=PAGE_SIZE),
    # conc=32
    DecodeConfig(
        32,
        64,
        128,
        128,
        num_pages=129,
        page_size=PAGE_SIZE,
        extra_num_pages=1281,
        extra_page_size=256,
    ),
    DecodeConfig(
        32,
        64,
        128,
        128,
        num_pages=129,
        page_size=PAGE_SIZE,
        extra_num_pages=1281,
        extra_page_size=64,
    ),
    DecodeConfig(
        32,
        64,
        128,
        128,
        num_pages=129,
        page_size=PAGE_SIZE,
        extra_num_pages=1281,
        extra_page_size=2,
    ),
    DecodeConfig(
        32,
        64,
        128,
        512,
        num_pages=129,
        page_size=PAGE_SIZE,
        extra_num_pages=1281,
        extra_page_size=256,
    ),
    # conc=128
    DecodeConfig(
        128,
        64,
        128,
        128,
        num_pages=513,
        page_size=PAGE_SIZE,
        extra_num_pages=5121,
        extra_page_size=256,
    ),
    DecodeConfig(
        128,
        64,
        128,
        128,
        num_pages=513,
        page_size=PAGE_SIZE,
        extra_num_pages=5121,
        extra_page_size=64,
    ),
    DecodeConfig(
        128,
        64,
        128,
        128,
        num_pages=513,
        page_size=PAGE_SIZE,
        extra_num_pages=5121,
        extra_page_size=2,
    ),
    DecodeConfig(
        128,
        64,
        128,
        512,
        num_pages=513,
        page_size=PAGE_SIZE,
        extra_num_pages=5121,
        extra_page_size=256,
    ),
]
# ---- DECODE-stage for DEEPSEEK V4 pro ----
configs += [
    DecodeConfig(1, 64, 128, 0, num_pages=513, page_size=PAGE_SIZE),
    DecodeConfig(2, 64, 128, 0, num_pages=513, page_size=PAGE_SIZE),
    DecodeConfig(8, 64, 128, 0, num_pages=513, page_size=PAGE_SIZE),
    DecodeConfig(16, 64, 128, 0, num_pages=513, page_size=PAGE_SIZE),
    DecodeConfig(32, 64, 128, 0, num_pages=513, page_size=PAGE_SIZE),
]

# ---- DECODE-stage for DEEPSEEK V3.2 / GLM-DSA (d_qk=576, 656 B/token) ----
configs += [
    # steady-state, no extra pool
    DecodeConfig(1, 64, 128, 0, num_pages=513, page_size=PAGE_SIZE, d_qk=576),
    DecodeConfig(2, 64, 128, 0, num_pages=513, page_size=PAGE_SIZE, d_qk=576),
    DecodeConfig(8, 64, 128, 0, num_pages=513, page_size=PAGE_SIZE, d_qk=576),
    DecodeConfig(16, 64, 128, 0, num_pages=513, page_size=PAGE_SIZE, d_qk=576),
    DecodeConfig(32, 64, 128, 0, num_pages=513, page_size=PAGE_SIZE, d_qk=576),
    # with extra KV pool
    DecodeConfig(
        32,
        64,
        128,
        128,
        num_pages=129,
        page_size=PAGE_SIZE,
        extra_num_pages=1281,
        extra_page_size=256,
        d_qk=576,
    ),
    DecodeConfig(
        128,
        64,
        128,
        512,
        num_pages=513,
        page_size=PAGE_SIZE,
        extra_num_pages=5121,
        extra_page_size=256,
        d_qk=576,
    ),
]

# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    require_arch("xe20", "xe35")
    device = torch.device("xpu")

    torch.manual_seed(42)
    if hasattr(torch.xpu, "manual_seed_all"):
        torch.xpu.manual_seed_all(42)

    results = []

    for cfg in configs:
        layout = LAYOUTS[cfg.d_qk]
        inputs = build_inputs(
            cfg.b,
            cfg.topk,
            cfg.extra_topk,
            cfg.num_pages,
            cfg.page_size,
            cfg.h_q,
            layout,
            device,
            extra_num_pages=cfg.extra_num_pages,
            extra_page_size=cfg.extra_page_size,
        )
        sm_scale = cfg.sm_scale if cfg.sm_scale is not None else layout.sm_scale
        total_bytes = _compute_total_bytes(
            cfg.b, cfg.topk, cfg.extra_topk, cfg.h_q, layout
        )

        # Triton reference (Triton gather+dequant -> PyTorch attention)
        fn_triton = lambda: flash_mla_sparse_decode_triton(
            **inputs, softmax_scale=sm_scale
        )
        ms_triton, _, _ = triton.testing.do_bench(fn_triton, quantiles=[0.5, 0.2, 0.8])
        torch.xpu.synchronize()

        # SGL Kernel
        fn_sgl = lambda: flash_mla_with_kvcache(
            q=inputs["q"],
            k_cache=inputs["k_cache"],
            block_table=None,
            cache_seqlens=None,
            head_dim_v=inputs["head_dim_v"],
            tile_scheduler_metadata=None,
            num_splits=None,
            softmax_scale=sm_scale,
            causal=False,
            is_fp8_kvcache=True,
            indices=inputs["indices"],
            attn_sink=inputs["attn_sink"],
            extra_k_cache=inputs.get("extra_k_cache"),
            extra_indices_in_kvcache=inputs.get("extra_indices"),
            topk_length=inputs["topk_length"],
            extra_topk_length=inputs.get("extra_topk_length"),
        )
        ms_sgl, _, _ = triton.testing.do_bench(fn_sgl, quantiles=[0.5, 0.2, 0.8])
        bw_sgl = total_bytes / (ms_sgl / 1e3) / 1e9
        torch.xpu.synchronize()

        results.append(
            make_row(
                SPEC,
                dtype=torch.bfloat16,
                size=(
                    cfg.b,
                    cfg.h_q,
                    cfg.d_qk,
                    cfg.topk,
                    cfg.extra_topk,
                    cfg.num_pages,
                    cfg.page_size,
                    cfg.extra_num_pages,
                    cfg.extra_page_size,
                ),
                time_us=ms_sgl * 1e3,
                bandwidth_gbs=bw_sgl,
                provider="sglang",
                reference_provider="triton",
                reference_time_us=ms_triton * 1e3,
            )
        )

    print()
    print(format_tables(results))
    print()
