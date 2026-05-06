"""Benchmark: flash_mla_sparse_decode — Triton V4 vs sgl_kernel.

Compares execution time and effective bandwidth of both implementations
across DeepSeek-V4 production shapes (varying B, topk, extra_topk).

Usage:
  python benchmark/bench_flash_mla_sparse_decode.py
"""

import math
from itertools import product
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from sgl_kernel import flash_mla_sparse_decode

# ── DeepSeek V4 constants ──
NOPE_DIM_VAL = 448
ROPE_DIM_VAL = 64
D_VAL = 512
TOKEN_DATA_STRIDE_VAL = 576
SCALE_STRIDE_VAL = 8

_TOKEN_DATA_STRIDE = tl.constexpr(TOKEN_DATA_STRIDE_VAL)
_SCALE_STRIDE = tl.constexpr(SCALE_STRIDE_VAL)

H_PER_RANK = 16
SM_SCALE = 1.0 / math.sqrt(D_VAL)
PAGE_SIZE = 256


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
    indices_ptr,
    out_ptr,
    page_size: tl.int32,
    page_bytes: tl.int64,
    scale_section_off: tl.int64,
    topk: tl.int32,
    stride_ib: tl.int32,
    stride_ob: tl.int32,
    stride_ot: tl.int32,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    D_OUT: tl.constexpr,
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
    token_data_bases = page_ids * page_bytes + page_offs * _TOKEN_DATA_STRIDE
    scale_bases = page_ids * page_bytes + scale_section_off + page_offs * _SCALE_STRIDE

    for g in tl.static_range(7):
        d_start = g * 64
        d_offs = tl.arange(0, 64)
        d_abs = d_start + d_offs
        d_mask = d_abs < NOPE_DIM

        fp8_addrs = token_data_bases[:, None] + (d_abs[None, :]).to(tl.int64)
        load_mask = idx_valid[:, None] & d_mask[None, :]
        fp8_vals = tl.load(cache_fp8_ptr + fp8_addrs, mask=load_mask, other=0.0)

        scale_val = tl.load(
            cache_uint8_ptr + scale_bases + g,
            mask=idx_valid,
            other=127,
        )
        scale_f32 = tl.math.exp2(scale_val.to(tl.float32) - 127.0)

        bf16_vals = (fp8_vals.to(tl.float32) * scale_f32[:, None]).to(tl.bfloat16)
        bf16_vals = tl.where(load_mask, bf16_vals, tl.zeros_like(bf16_vals))

        out_addrs = bid * stride_ob + t_offs[:, None] * stride_ot + d_abs[None, :]
        tl.store(out_ptr + out_addrs, bf16_vals, mask=load_mask)

    rope_offs = tl.arange(0, ROPE_DIM)
    rope_byte_bases = token_data_bases + NOPE_DIM
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

    kv_dense = torch.zeros(B, topk, D_VAL, dtype=torch.bfloat16, device=k_cache.device)

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
        indices,
        kv_dense,
        page_size,
        int(page_bytes),
        int(page_size * TOKEN_DATA_STRIDE_VAL),
        topk,
        indices.stride(0),
        kv_dense.stride(0),
        kv_dense.stride(1),
        NOPE_DIM=NOPE_DIM_VAL,
        ROPE_DIM=ROPE_DIM_VAL,
        D_OUT=D_VAL,
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    scores = torch.matmul(q_3d, kv_dense.transpose(-1, -2))
    scores = scores.float() * softmax_scale
    scores.masked_fill_(invalid_mask, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    p = torch.softmax(scores, dim=-1)
    del scores
    p = torch.nan_to_num(p, 0.0)
    out = torch.matmul(p.to(torch.bfloat16), kv_dense)
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

    kv_dense = _gather_kv_pages(k_cache, flat_indices, topk_length)
    invalid_mask = _build_invalid_mask(flat_indices, topk_length)

    out, lse = _compute_attention(q_3d, kv_dense, invalid_mask, softmax_scale)
    del kv_dense, invalid_mask

    if extra_k_cache is not None and extra_indices is not None:
        extra_flat = extra_indices.reshape(B, -1).contiguous()
        kv_extra = _gather_kv_pages(extra_k_cache, extra_flat, extra_topk_length)
        extra_mask = _build_invalid_mask(extra_flat, extra_topk_length)

        out_extra, lse_extra = _compute_attention(
            q_3d,
            kv_extra,
            extra_mask,
            softmax_scale,
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
def _ceil_to_576(x):
    return ((x + 575) // 576) * 576


def _total_page_bytes(page_size):
    return page_size * TOKEN_DATA_STRIDE_VAL + _ceil_to_576(
        page_size * SCALE_STRIDE_VAL
    )


def make_fp8_kv_cache(num_pages, page_size, device):
    total_pb = _total_page_bytes(page_size)
    raw = torch.zeros(num_pages, total_pb, dtype=torch.uint8, device=device)

    fill_pages = min(num_pages, 8)
    for p in range(fill_pages):
        for t in range(min(page_size, 4)):
            t_offset = t * TOKEN_DATA_STRIDE_VAL
            raw[p, t_offset : t_offset + NOPE_DIM_VAL] = torch.randint(
                0, 200, (NOPE_DIM_VAL,), dtype=torch.uint8, device=device
            )
            rope_bf16 = torch.randn(ROPE_DIM_VAL, dtype=torch.bfloat16, device=device)
            raw[
                p, t_offset + NOPE_DIM_VAL : t_offset + NOPE_DIM_VAL + ROPE_DIM_VAL * 2
            ] = rope_bf16.view(torch.uint8)

    scale_section_start = page_size * TOKEN_DATA_STRIDE_VAL
    for p in range(fill_pages):
        for t in range(min(page_size, 4)):
            s_offset = scale_section_start + t * SCALE_STRIDE_VAL
            raw[p, s_offset : s_offset + 7] = 127

    k_cache = raw.as_strided(
        (num_pages, page_size, 1, 584),
        (total_pb, 584, 584, 1),
    ).view(torch.float8_e4m3fn)
    return k_cache


def make_indices(B, topk, num_pages, page_size, device):
    total_tokens = num_pages * page_size
    return torch.randint(
        0, total_tokens, (B, 1, topk), dtype=torch.int32, device=device
    )


def build_inputs(B, topk, extra_topk, num_pages, page_size, H, device):
    k_cache = make_fp8_kv_cache(num_pages, page_size, device)
    indices = make_indices(B, topk, num_pages, page_size, device)
    q = torch.randn(B, 1, H, D_VAL, dtype=torch.bfloat16, device=device)
    topk_length = torch.full((B,), topk, dtype=torch.int32, device=device)
    attn_sink = torch.randn(H, dtype=torch.float32, device=device) * 0.1

    inputs = {
        "q": q,
        "k_cache": k_cache,
        "indices": indices,
        "topk_length": topk_length,
        "attn_sink": attn_sink,
        "head_dim_v": D_VAL,
        "softmax_scale": SM_SCALE,
    }

    if extra_topk > 0:
        extra_cache = make_fp8_kv_cache(num_pages, 64, device)
        extra_indices = make_indices(B, extra_topk, num_pages, 64, device)
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
def _compute_total_bytes(B, topk, extra_topk, H):
    read_q = B * H * D_VAL * 2
    read_kv = B * topk * (TOKEN_DATA_STRIDE_VAL + SCALE_STRIDE_VAL)
    read_extra_kv = (
        B * extra_topk * (TOKEN_DATA_STRIDE_VAL + SCALE_STRIDE_VAL)
        if extra_topk > 0
        else 0
    )
    read_indices = B * (topk + extra_topk) * 4
    write_out = B * H * D_VAL * 2
    write_lse = B * H * 4
    return read_q + read_kv + read_extra_kv + read_indices + write_out + write_lse


# ============================================================================
# Benchmark configuration
# ============================================================================
batch_size_range = [8, 48, 128, 256]
topk_range = [64, 128, 512]
extra_topk_range = [0, 512]

MAX_TOKENS = 256 * 640

configs = [
    (b, topk, extra)
    for b, topk, extra in product(batch_size_range, topk_range, extra_topk_range)
    if b * (topk + extra) <= MAX_TOKENS
]


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    device = torch.device("xpu")
    H = H_PER_RANK
    num_pages = 512

    torch.manual_seed(42)
    if hasattr(torch.xpu, "manual_seed_all"):
        torch.xpu.manual_seed_all(42)

    results = []

    for b, topk, extra_topk in configs:
        inputs = build_inputs(b, topk, extra_topk, num_pages, PAGE_SIZE, H, device)
        total_bytes = _compute_total_bytes(b, topk, extra_topk, H)

        # Triton V4
        fn_triton = lambda: flash_mla_sparse_decode_triton(**inputs)
        ms_triton, _, _ = triton.testing.do_bench(fn_triton, quantiles=[0.5, 0.2, 0.8])
        bw_triton = total_bytes / (ms_triton / 1e3) / 1e9

        # SGL Kernel
        fn_sgl = lambda: flash_mla_sparse_decode(**inputs)
        ms_sgl, _, _ = triton.testing.do_bench(fn_sgl, quantiles=[0.5, 0.2, 0.8])
        bw_sgl = total_bytes / (ms_sgl / 1e3) / 1e9

        results.append((b, topk, extra_topk, ms_triton, ms_sgl, bw_triton, bw_sgl))

    # Print table with borders
    hdr = (
        "| batch_size | topk | extra_topk | Triton V4 (ms) | SGL Kernel (ms) "
        "| Triton BW (GB/s) | SGL Kernel BW (GB/s) |"
    )
    sep = (
        "|------------|------|------------|----------------|-----------------|"
        "------------------|----------------------|"
    )

    print()
    print(sep)
    print(hdr)
    print(sep)
    for b, topk, extra_topk, ms_t, ms_s, bw_t, bw_s in results:
        print(
            f"| {b:>10} | {topk:>4} | {extra_topk:>10} "
            f"| {ms_t:>14.4f} | {ms_s:>15.4f} "
            f"| {bw_t:>16.2f} | {bw_s:>20.2f} |"
        )
    print(sep)
    print()
