"""Benchmark: 2-stage sparse MLA prefill — Triton reference vs sgl_kernel.

Compares execution time and effective bandwidth of both implementations across
prefill shapes. The kernel reuses the 2-stage sparse MLA decode device stack
(gather -> dense flash); prefill supports dense bf16 KV with d_qk in {512, 576}
(512 latent, or 576 = nope-512 + rope-64); d_v == 512.

Mirrors bench_flash_mla_with_kvcache.py: a Triton reference (Triton gather ->
PyTorch attention) vs the SGL kernel, timed with triton.testing.do_bench and
printed as a bordered markdown table. Unlike the decode benchmark's fp8 packed
gather, prefill KV is dense bf16, so the gather here is a plain index_select-style
Triton kernel (no page/scale/dequant math).

Usage:
  python benchmark/bench_flash_mla_sparse_fwd.py
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from sgl_kernel import flash_mla_sparse_fwd

# ── constants ──
D_V = 512
H_KV = 1
S_KV = 16384


# ============================================================================
# Triton reference: dense bf16 gather kernel
# ============================================================================
@triton.jit
def _gather_dense_kernel(
    kv_ptr,  # [s_kv, d_qk] bf16 (h_kv==1 squeezed)
    indices_ptr,  # [s_q, topk] int32 (topk_length masking pre-applied -> -1)
    out_ptr,  # [s_q, topk, d_qk] bf16
    s_kv: tl.int32,
    topk: tl.int32,
    stride_kv_s: tl.int64,
    stride_ib: tl.int32,
    stride_ob: tl.int64,
    stride_ot: tl.int32,
    D_QK: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)  # s_q index
    tile = tl.program_id(1)  # topk tile

    t_offs = tile * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_offs < topk

    idx = tl.load(indices_ptr + row * stride_ib + t_offs, mask=t_mask, other=-1)
    valid = t_mask & (idx >= 0) & (idx < s_kv)
    safe_idx = tl.where(valid, idx, 0).to(tl.int64)

    for d_start in tl.static_range(0, D_QK, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D_QK
        kv_addrs = safe_idx[:, None] * stride_kv_s + d_offs[None, :]
        load_mask = valid[:, None] & d_mask[None, :]
        vals = tl.load(kv_ptr + kv_addrs, mask=load_mask, other=0.0)

        out_addrs = row * stride_ob + t_offs[:, None] * stride_ot + d_offs[None, :]
        tl.store(out_ptr + out_addrs, vals, mask=load_mask)


# ============================================================================
# Triton reference: Python helpers
# ============================================================================
def _gather_dense(
    kv: torch.Tensor,  # [s_kv, h_kv, d_qk] bf16
    indices: torch.Tensor,  # [s_q, topk] int32 (topk_length masking pre-applied)
    d_qk: int,
) -> torch.Tensor:
    s_kv = kv.shape[0]
    s_q, topk = indices.shape
    kv_2d = kv.reshape(s_kv, d_qk)

    out = torch.zeros(s_q, topk, d_qk, dtype=torch.bfloat16, device=kv.device)

    block_d = 64  # d_qk in {512, 576} -> both divisible by 64
    grid = lambda meta: (s_q, triton.cdiv(topk, meta["BLOCK_T"]))
    _gather_dense_kernel[grid](
        kv_2d,
        indices,
        out,
        s_kv,
        topk,
        kv_2d.stride(0),
        indices.stride(0),
        out.stride(0),
        out.stride(1),
        D_QK=d_qk,
        BLOCK_T=64,
        BLOCK_D=block_d,
    )
    return out


def _compute_attention(
    q: torch.Tensor,  # [s_q, h_q, d_qk]
    gathered_kv: torch.Tensor,  # [s_q, topk, d_qk]
    invalid_mask: torch.Tensor,  # [s_q, topk] bool
    sm_scale: float,
    d_v: int,
    attn_sink: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    s_q, h_q, _ = q.shape
    gathered_f32 = gathered_kv.float()
    P = (q.float() @ gathered_f32.transpose(1, 2)) * sm_scale
    P.masked_fill_(invalid_mask.unsqueeze(1).broadcast_to(P.shape), float("-inf"))

    orig_lse = torch.logsumexp(P, dim=-1)

    lse_for_o = orig_lse
    if attn_sink is not None:
        lse_for_o = torch.logsumexp(
            torch.stack(
                [orig_lse.view(s_q, h_q), attn_sink.broadcast_to(s_q, h_q)], dim=0
            ),
            dim=0,
        )
    lse_for_o = lse_for_o.clone()
    lse_for_o[lse_for_o == float("-inf")] = float("+inf")

    s_for_o = torch.exp(P - lse_for_o.unsqueeze(-1))
    out = s_for_o @ gathered_f32[..., :d_v]

    lonely = orig_lse == float("-inf")
    orig_lse = orig_lse.masked_fill(lonely, float("+inf"))
    return out.to(torch.bfloat16), orig_lse


def flash_mla_sparse_prefill_triton(
    q: torch.Tensor,  # [s_q, h_q, d_qk]
    kv: torch.Tensor,  # [s_kv, h_kv, d_qk]
    indices: torch.Tensor,  # [s_q, h_kv, topk]
    sm_scale: float,
    d_v: int,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    s_q, h_q, d_qk = q.shape
    s_kv = kv.shape[0]
    topk = indices.shape[-1]

    flat_indices = indices.reshape(s_q, topk).clone()
    if topk_length is not None:
        arange = torch.arange(topk, device=flat_indices.device)
        pad = arange.unsqueeze(0) >= topk_length.unsqueeze(1)
        flat_indices[pad] = -1

    invalid_mask = (flat_indices < 0) | (flat_indices >= s_kv)

    gathered_kv = _gather_dense(kv, flat_indices, d_qk)
    out, lse = _compute_attention(
        q, gathered_kv, invalid_mask, sm_scale, d_v, attn_sink
    )
    return out, lse


# ============================================================================
# Input construction
# ============================================================================
def build_inputs(s_q, h_q, topk, d_qk, device="xpu", dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    q = torch.randn((s_q, h_q, d_qk), device=device, dtype=dtype)
    kv = torch.randn((S_KV, H_KV, d_qk), device=device, dtype=dtype)
    indices = torch.full((s_q, H_KV, topk), S_KV, dtype=torch.int32, device=device)
    for t in range(s_q):
        n = min(topk, max(1, S_KV))
        i_i = torch.randperm(S_KV, device=device)[:n].to(torch.int32)
        indices[t, 0, : len(i_i)] = i_i
    return q, kv, indices


# ============================================================================
# Bandwidth calculation
# ============================================================================
def effective_bytes(s_q, h_q, topk, d_qk):
    # q read + gathered kv read + out write (bf16 = 2 bytes; fp32 lse/max negligible)
    q_bytes = s_q * h_q * d_qk * 2
    kv_bytes = s_q * topk * d_qk * 2
    out_bytes = s_q * h_q * D_V * 2
    return q_bytes + kv_bytes + out_bytes


# ============================================================================
# Benchmark configuration
# ============================================================================
# (s_q, h_q, topk, d_qk); d_qk in {512, 576}
configs = [
    (512, 16, 2048, 512),
    (512, 32, 2048, 512),
    (512, 128, 2048, 512),
    (2048, 16, 512, 512),
    (2048, 128, 512, 512),
    (512, 16, 2048, 576),
    (512, 32, 2048, 576),
    (512, 128, 2048, 576),
    (2048, 16, 512, 576),
    (2048, 128, 512, 576),
]


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    device = torch.device("xpu")

    torch.manual_seed(42)
    if hasattr(torch.xpu, "manual_seed_all"):
        torch.xpu.manual_seed_all(42)

    results = []

    for s_q, h_q, topk, d_qk in configs:
        q, kv, indices = build_inputs(s_q, h_q, topk, d_qk, device=device)
        sm_scale = d_qk**-0.5
        total_bytes = effective_bytes(s_q, h_q, topk, d_qk)

        # Triton reference (Triton gather -> PyTorch attention)
        fn_triton = lambda: flash_mla_sparse_prefill_triton(
            q, kv, indices, sm_scale, D_V
        )
        ms_triton, _, _ = triton.testing.do_bench(fn_triton, quantiles=[0.5, 0.2, 0.8])
        bw_triton = total_bytes / (ms_triton / 1e3) / 1e9

        # SGL Kernel
        fn_sgl = lambda: flash_mla_sparse_fwd(
            q, kv, indices, sm_scale=sm_scale, d_v=D_V
        )
        ms_sgl, _, _ = triton.testing.do_bench(fn_sgl, quantiles=[0.5, 0.2, 0.8])
        bw_sgl = total_bytes / (ms_sgl / 1e3) / 1e9

        results.append((s_q, h_q, topk, d_qk, ms_triton, ms_sgl, bw_triton, bw_sgl))

    # Print table with borders
    hdr = (
        "| s_q  | head_q | topk | d_qk | Triton Ref (ms) | SGL Kernel (ms) "
        "| Triton Ref BW (GB/s) | SGL Kernel BW (GB/s) |"
    )
    sep = (
        "|------|--------|------|------|-----------------|-----------------|"
        "----------------------|----------------------|"
    )

    print()
    print(sep)
    print(hdr)
    print(sep)
    for s_q, h_q, topk, d_qk, ms_t, ms_s, bw_t, bw_s in results:
        print(
            f"| {s_q:>4} | {h_q:>6} | {topk:>4} | {d_qk:>4} "
            f"| {ms_t:>15.4f} | {ms_s:>15.4f} "
            f"| {bw_t:>20.2f} | {bw_s:>20.2f} |"
        )
    print(sep)
    print()
