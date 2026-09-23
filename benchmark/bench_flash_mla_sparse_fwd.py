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

from typing import NamedTuple, Optional, Tuple

import torch
import triton
import triton.language as tl
from _bench import BenchSpec, ProblemShape, format_tables, make_row, require_arch
from sgl_kernel import flash_mla_sparse_fwd

# ── constants ──
D_V = 512
H_KV = 1
S_KV = 16384

SPEC = BenchSpec(
    kernel="flash_mla_sparse_prefill",
    shape=ProblemShape(
        ("s_q", "h_q", "topk", "d_qk", "s_kv"),
        description="2-stage sparse MLA prefill; dense bf16 KV, d_qk ∈ {512, 576}, d_v=512",
    ),
    metrics=("time_us", "bandwidth_gbs"),
)


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


_REF_CHUNK_SQ = 256


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

    outs, lses = [], []
    for start in range(0, s_q, _REF_CHUNK_SQ):
        end = min(start + _REF_CHUNK_SQ, s_q)
        gathered_kv = _gather_dense(kv, flat_indices[start:end], d_qk)
        out_chunk, lse_chunk = _compute_attention(
            q[start:end], gathered_kv, invalid_mask[start:end], sm_scale, d_v, attn_sink
        )
        outs.append(out_chunk)
        lses.append(lse_chunk)

    out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)
    lse = lses[0] if len(lses) == 1 else torch.cat(lses, dim=0)
    return out, lse


# ============================================================================
# Input construction
# ============================================================================
def build_inputs(
    s_q,
    h_q,
    topk,
    d_qk,
    s_kv,
    use_topk_length=False,
    device="xpu",
    dtype=torch.bfloat16,
    seed=0,
):
    torch.manual_seed(seed)
    q = torch.randn((s_q, h_q, d_qk), device=device, dtype=dtype)
    kv = torch.randn((s_kv, H_KV, d_qk), device=device, dtype=dtype)

    n = min(topk, max(1, s_kv))
    # per-row random permutation via argsort of random keys (vectorized -- avoids
    # an s_q-iteration Python loop, needed since chunked-prefill s_q can be 8192+)
    perm = torch.argsort(torch.rand((s_q, s_kv), device=device), dim=-1)[:, :n]
    indices = torch.full((s_q, H_KV, topk), s_kv, dtype=torch.int32, device=device)
    indices[:, 0, :n] = perm.to(torch.int32)

    topk_length = None
    if use_topk_length:
        topk_length = torch.randint(
            1, topk + 1, (s_q,), device=device, dtype=torch.int32
        )

    return q, kv, indices, topk_length


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
class PrefillConfig(NamedTuple):
    s_q: int
    h_q: int
    topk: int
    d_qk: int
    s_kv: int
    use_topk_length: bool = False


# ---- PREFILL-stage on synthetic data ----
configs = [
    PrefillConfig(512, 16, 2048, 512, S_KV),
    PrefillConfig(512, 32, 2048, 512, S_KV),
    PrefillConfig(512, 128, 2048, 512, S_KV),
    PrefillConfig(2048, 16, 512, 512, S_KV),
    PrefillConfig(2048, 128, 512, 512, S_KV),
    PrefillConfig(512, 16, 2048, 576, S_KV),
    PrefillConfig(512, 32, 2048, 576, S_KV),
    PrefillConfig(512, 128, 2048, 576, S_KV),
    PrefillConfig(2048, 16, 512, 576, S_KV),
    PrefillConfig(2048, 128, 512, 576, S_KV),
]

# ---- PREFILL-stage for DEEPSEEK V4 flash ----
configs += [
    PrefillConfig(8192, 64, 128, 512, 8192, use_topk_length=True),  # conc=2/32/128
    PrefillConfig(8192, 64, 256, 512, 8256, use_topk_length=True),  # conc=2/32/128
    PrefillConfig(8192, 64, 640, 512, 10240, use_topk_length=True),  # conc=2/32/128
    PrefillConfig(8192, 64, 128, 512, 8113, use_topk_length=True),  # conc=32
    PrefillConfig(8192, 64, 256, 512, 8113, use_topk_length=True),  # conc=8
    PrefillConfig(8192, 64, 256, 512, 8176, use_topk_length=True),  # conc=32
    PrefillConfig(8192, 64, 640, 512, 10141, use_topk_length=True),  # conc=32
]

# ---- PREFILL-stage for DEEPSEEK V4 Pro ----
configs += [
    PrefillConfig(1024, 32, 1024, 512, 1024),
    PrefillConfig(8192, 32, 1024, 512, 8192),
    PrefillConfig(10240, 32, 1024, 512, 10240),
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
        q, kv, indices, topk_length = build_inputs(
            cfg.s_q,
            cfg.h_q,
            cfg.topk,
            cfg.d_qk,
            cfg.s_kv,
            use_topk_length=cfg.use_topk_length,
            device=device,
        )
        sm_scale = cfg.d_qk**-0.5
        total_bytes = effective_bytes(cfg.s_q, cfg.h_q, cfg.topk, cfg.d_qk)

        # Triton reference (Triton gather -> PyTorch attention)
        fn_triton = lambda: flash_mla_sparse_prefill_triton(
            q, kv, indices, sm_scale, D_V, topk_length=topk_length
        )
        ms_triton, _, _ = triton.testing.do_bench(fn_triton, quantiles=[0.5, 0.2, 0.8])
        torch.xpu.synchronize()

        # SGL Kernel
        fn_sgl = lambda: flash_mla_sparse_fwd(
            q, kv, indices, sm_scale=sm_scale, d_v=D_V, topk_length=topk_length
        )
        ms_sgl, _, _ = triton.testing.do_bench(fn_sgl, quantiles=[0.5, 0.2, 0.8])
        bw_sgl = total_bytes / (ms_sgl / 1e3) / 1e9
        torch.xpu.synchronize()
        results.append(
            make_row(
                SPEC,
                dtype=torch.bfloat16,
                size=(cfg.s_q, cfg.h_q, cfg.topk, cfg.d_qk, cfg.s_kv),
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
