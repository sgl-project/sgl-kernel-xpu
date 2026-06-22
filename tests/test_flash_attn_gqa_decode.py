"""
Regression repro for SGLANGT-1286:
  xe_fmha_fwd_decode produces NaN output for GQA + MHA configs with paged KV cache.

Affected configs (both yield 0% accuracy on intel_xpu; triton gives correct results):
  LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct : 32Q / 8KV heads, head_dim=128, tp=1
    → dispatches xe_fmha_fwd_decode_kernel_4_128_128 (GQA_ratio=4, HD=128, page_size=128)
  meta-llama/Llama-2-13b-chat-hf        : 20Q / 20KV heads/TP, head_dim=128, tp=2
    → dispatches xe_fmha_fwd_decode_kernel_1_128_128 (GQA_ratio=1, HD=128, page_size=128)

Kernel dispatch: xe_fmha_fwd_decode_kernel_{GQA_ratio}_{head_dim}_{page_size}
  - GQA_ratio = nq_heads / nkv_heads, must be in {1, 2, 4, 8, 16}
  - head_dim  must be in {64, 72, 96, 128, 192, 256, 512}
  - page_size must be in {64, 128}  (sglang intel_xpu default: 128)

This file tests the kernel directly — no model download needed, runs on ~200 MB GPU.

Run:
    cd sgl-kernel-xpu/tests
    python -m pytest test_flash_attn_gqa_decode.py -v
"""

import pytest
import torch
import utils

device = utils.get_device()


def _skip_msg():
    try:
        if not torch.xpu.is_available():
            return "XPU not available"
        from sgl_kernel.flash_attn import flash_attn_with_kvcache  # noqa: F401
    except (ImportError, AttributeError) as e:
        return str(e)
    return None


SKIP = _skip_msg()


# ---------------------------------------------------------------------------
# Reference: plain PyTorch GQA attention (on CPU in fp32 for accuracy)
# ---------------------------------------------------------------------------

def _ref_attn_gqa(q_bshd, k_bshd, v_bshd, cache_seqlens):
    """
    q_bshd : (batch, seqlen_q, nq_heads, head_dim)  fp32 CPU
    k_bshd : (batch, seqlen_k, nkv_heads, head_dim) fp32 CPU
    v_bshd : same as k
    cache_seqlens : (batch,) int  – how many KV slots are valid
    returns  (batch, seqlen_q, nq_heads, head_dim)
    """
    batch, seqlen_q, nq, d = q_bshd.shape
    _, seqlen_k, nkv, _ = k_bshd.shape
    ratio = nq // nkv
    assert nq % nkv == 0

    # Expand KV heads to match Q heads
    k = k_bshd.repeat_interleave(ratio, dim=2)  # (batch, seqlen_k, nq, d)
    v = v_bshd.repeat_interleave(ratio, dim=2)

    scale = d ** -0.5
    # (batch, nq, seqlen_q, seqlen_k)
    scores = torch.einsum("bqhd,bkhd->bhqk", q_bshd * scale, k)

    # Mask out positions beyond cache_seqlens
    for b in range(batch):
        valid = cache_seqlens[b].item()
        if valid < seqlen_k:
            scores[b, :, :, valid:] = float("-inf")

    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhqk,bkhd->bqhd", attn, v)
    return out


# ---------------------------------------------------------------------------
# Core test helper
# ---------------------------------------------------------------------------

def _run(batch, nq_heads, nkv_heads, head_dim, page_size, seqlen, dtype=torch.float16):
    """
    Allocate a paged KV cache and run one decode step (seqlen_q=1 per batch item).
    Asserts:
      1. Output contains no NaN (SGLANGT-1286 symptom)
      2. Output is close to PyTorch reference attention

    page_size must be 64 or 128 (kernel constraint for intel_xpu decode).
    GQA ratio (nq_heads // nkv_heads) must be in {1, 2, 4, 8, 16}.
    """
    from sgl_kernel.flash_attn import flash_attn_with_kvcache

    assert page_size in (64, 128), f"page_size must be 64 or 128, got {page_size}"
    assert seqlen % page_size == 0, f"seqlen must be multiple of page_size"
    gqa_ratio = nq_heads // nkv_heads
    assert nq_heads % nkv_heads == 0 and gqa_ratio in (1, 2, 4, 8, 16), (
        f"nq/nkv must give GQA ratio in {{1,2,4,8,16}}, got {gqa_ratio}"
    )

    pages_per_seq = seqlen // page_size
    num_pages = batch * pages_per_seq

    torch.manual_seed(42)

    # Paged KV cache: (num_pages, page_size, nkv_heads, head_dim)
    k_cache = torch.randn(num_pages, page_size, nkv_heads, head_dim,
                          dtype=dtype, device=device)
    v_cache = torch.randn(num_pages, page_size, nkv_heads, head_dim,
                          dtype=dtype, device=device)

    # Decode query: one token per batch item → (batch, 1, nq_heads, head_dim)
    q = torch.randn(batch, 1, nq_heads, head_dim, dtype=dtype, device=device)

    # Linear page assignment: seq i → pages [i*P .. (i+1)*P)
    page_table = torch.arange(num_pages, dtype=torch.int32, device=device).view(
        batch, pages_per_seq
    )
    cache_seqlens = torch.full((batch,), seqlen, dtype=torch.int32, device=device)

    # --- kernel under test ---
    out = flash_attn_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        softmax_scale=head_dim ** -0.5,
        causal=True,
    )
    # out shape: (batch, 1, nq_heads, head_dim)

    nan_count = out.isnan().sum().item()
    assert nan_count == 0, (
        f"flash_attn_with_kvcache produced {nan_count}/{out.numel()} NaN values "
        f"[batch={batch}, nq={nq_heads}, nkv={nkv_heads}, head_dim={head_dim}, "
        f"page_size={page_size}, seqlen={seqlen}] — SGLANGT-1286"
    )

    assert out.abs().max().item() > 0.0, "output is all-zero (silent failure)"

    # --- reference on CPU ---
    # Reconstruct dense KV from paged layout (batch, seqlen, nkv_heads, head_dim)
    k_dense = k_cache[page_table.cpu().flatten()].view(
        batch, pages_per_seq * page_size, nkv_heads, head_dim
    )[:, :seqlen].float().cpu()
    v_dense = v_cache[page_table.cpu().flatten()].view(
        batch, pages_per_seq * page_size, nkv_heads, head_dim
    )[:, :seqlen].float().cpu()
    q_cpu = q.float().cpu()

    ref = _ref_attn_gqa(q_cpu, k_dense, v_dense,
                        cache_seqlens.cpu())  # (batch, 1, nq, d)

    out_cpu = out.float().cpu()
    atol = 1e-2 if dtype == torch.float16 else 5e-3
    max_diff = (out_cpu - ref).abs().max().item()
    assert max_diff < atol, (
        f"Max abs diff {max_diff:.4f} exceeds {atol} "
        f"[nq={nq_heads}, nkv={nkv_heads}] — kernel output likely incorrect"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(SKIP, reason=SKIP or "xpu unavailable")
def test_exaone_gqa_config():
    """EXAONE-3.5-7.8B-Instruct: 32Q/8KV, head_dim=128 → kernel _4_128_128."""
    _run(batch=2, nq_heads=32, nkv_heads=8, head_dim=128, page_size=128, seqlen=128)


@pytest.mark.skipif(SKIP, reason=SKIP or "xpu unavailable")
def test_llama2_13b_tp2_mha_config():
    """Llama-2-13b TP=2 per-rank: 20Q/20KV, head_dim=128 → kernel _1_128_128."""
    _run(batch=2, nq_heads=20, nkv_heads=20, head_dim=128, page_size=128, seqlen=128)


@pytest.mark.skipif(SKIP, reason=SKIP or "xpu unavailable")
@pytest.mark.parametrize("nq,nkv", [
    (32, 8),   # GQA ratio=4  (EXAONE / Mistral-class)
    (20, 20),  # MHA ratio=1  (Llama-2-13b TP=2)
    (32, 32),  # MHA ratio=1  (Llama-2-7b TP=1)
    (8, 2),    # GQA ratio=4  (smoke, small heads)
    (8, 4),    # GQA ratio=2
    (8, 1),    # MQA ratio=8
])
@pytest.mark.parametrize("page_size", [64, 128])
@pytest.mark.parametrize("seqlen_pages", [1, 4])  # seqlen = page_size * seqlen_pages
def test_gqa_paged_decode_no_nan(nq, nkv, page_size, seqlen_pages):
    """Parametrized: decode step must not produce NaN for any supported GQA/MHA config."""
    seqlen = page_size * seqlen_pages
    _run(batch=2, nq_heads=nq, nkv_heads=nkv, head_dim=128,
         page_size=page_size, seqlen=seqlen)


@pytest.mark.skipif(SKIP, reason=SKIP or "xpu unavailable")
def test_minimal_repro_sglangt_1286():
    """Smallest possible SGLANGT-1286 repro: batch=2, 32Q/8KV, page_size=128, seqlen=128.

    Bug symptom: kernel produces grossly wrong outputs (max_diff ~4), not NaN.
    The wrong outputs cause 0% model accuracy when using intel_xpu backend.
    Kernel dispatched: xe_fmha_fwd_decode_kernel_4_128_128 (GQA=4, HD=128, PS=128).
    """
    from sgl_kernel.flash_attn import flash_attn_with_kvcache

    dev = device
    torch.manual_seed(0)

    # 2 pages × page_size=128 × 8 KV heads × head_dim=128  (~0.5 MB per K/V)
    k = torch.randn(2, 128, 8, 128, dtype=torch.float16, device=dev)
    v = torch.randn(2, 128, 8, 128, dtype=torch.float16, device=dev)
    q = torch.randn(2, 1, 32, 128, dtype=torch.float16, device=dev)

    out = flash_attn_with_kvcache(
        q=q,
        k_cache=k,
        v_cache=v,
        page_table=torch.tensor([[0], [1]], dtype=torch.int32, device=dev),
        cache_seqlens=torch.tensor([128, 128], dtype=torch.int32, device=dev),
        softmax_scale=128 ** -0.5,
        causal=True,
    )

    # Symptom 1: no NaN in output
    nan_count = out.isnan().sum().item()
    assert nan_count == 0, (
        f"Minimal SGLANGT-1286 repro: {nan_count}/{out.numel()} NaN elements"
    )

    # Symptom 2 (primary): output must be close to reference — this is what causes 0% accuracy
    k_dense = k.float().cpu().view(2, 128, 8, 128)   # (batch, seqlen, nkv, d)
    v_dense = v.float().cpu().view(2, 128, 8, 128)
    ref = _ref_attn_gqa(q.float().cpu(), k_dense, v_dense,
                        torch.tensor([128, 128]))  # (2, 1, 32, 128)
    max_diff = (out.float().cpu() - ref).abs().max().item()
    assert max_diff < 0.01, (
        f"SGLANGT-1286 repro CONFIRMED: max_diff={max_diff:.4f} (expected < 0.01)\n"
        f"  xe_fmha_fwd_decode_kernel_4_128_128 produces wrong outputs on this GPU.\n"
        f"  This matches the 0% accuracy observed on EXAONE/Llama with intel_xpu backend."
    )