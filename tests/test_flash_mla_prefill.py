import gc
import sys

import pytest
import torch
import torch.nn.functional as F
from sgl_kernel import flash_mla_prefill, flash_mla_prefill_get_workspace_size
from torch import Tensor

device = torch.device("xpu")

if not torch.xpu.is_available():
    pytest.skip(
        reason="MLA Prefill requires XPU device.",
        allow_module_level=True,
    )


def clear_memory():
    gc.collect()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
        torch.xpu.synchronize()


@pytest.fixture(autouse=True)
def reset_torch_defaults():
    yield
    clear_memory()


def ref_mla_prefill_varlen(
    q_nope: Tensor,  # (total_q, H, D_latent)
    q_pe: Tensor,  # (total_q, H, D_rope)
    kv_cache: Tensor,  # (total_pages, page_size, D_ckv)
    scale: float,
    block_tables: Tensor,  # (B, max_pages) int32
    cu_seqlens_q: Tensor,  # (B+1,) int32
    seq_lens_k: Tensor,  # (B,) int32
    causal: bool = True,
) -> Tensor:
    """Pure-PyTorch reference for varlen MLA prefill with causal mask."""
    batch_size = seq_lens_k.shape[0]
    H = q_nope.shape[1]
    D_latent = q_nope.shape[2]
    D_rope = q_pe.shape[2]
    D_ckv = D_latent + D_rope
    total_q = q_nope.shape[0]

    out = torch.zeros(total_q, H, D_latent, dtype=q_nope.dtype)

    for b in range(batch_size):
        q_start = cu_seqlens_q[b].item()
        q_end = cu_seqlens_q[b + 1].item()
        seqlen_q = q_end - q_start
        seqlen_k = seq_lens_k[b].item()

        # Get Q for this batch
        q_n = q_nope[q_start:q_end]  # (seqlen_q, H, D_latent)
        q_p = q_pe[q_start:q_end]  # (seqlen_q, H, D_rope)

        # Gather KV from paged cache
        kv = kv_cache[block_tables[b]].reshape(-1, D_ckv)[:seqlen_k]

        # q_full = cat(q_nope, q_pe) -> (seqlen_q, H, D_ckv)
        q_full = torch.cat([q_n, q_p], dim=-1)  # (seqlen_q, H, D_ckv)
        q_full = q_full.permute(1, 0, 2)  # (H, seqlen_q, D_ckv)

        k_full = kv.unsqueeze(0).expand(H, -1, -1)  # (H, seqlen_k, D_ckv)
        v_full = kv[:, :D_latent].unsqueeze(0).expand(H, -1, -1)

        if causal:
            # Causal mask: k_idx <= (seqlen_k - seqlen_q) + q_idx
            causal_offset = seqlen_k - seqlen_q
            q_idx = torch.arange(seqlen_q).unsqueeze(1)
            k_idx = torch.arange(seqlen_k).unsqueeze(0)
            mask = k_idx > (causal_offset + q_idx)
            attn_mask = torch.zeros(seqlen_q, seqlen_k, dtype=torch.float32)
            attn_mask.masked_fill_(mask, float("-inf"))
            attn_mask = attn_mask.unsqueeze(0).expand(H, -1, -1)

            o = F.scaled_dot_product_attention(
                q_full, k_full, v_full, scale=scale, attn_mask=attn_mask
            )
        else:
            o = F.scaled_dot_product_attention(q_full, k_full, v_full, scale=scale)

        out[q_start:q_end] = o.permute(1, 0, 2)

    return out


# ============================================================================
# Test: MLA prefill
# Primarily exercises incremental prefill (seqlen_q < seqlen_k); a handful
# of full-prefill cases (seqlen_q == seqlen_k) are included for coverage.
# ============================================================================
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("block_size", [16, 32, 64, 128])
@pytest.mark.parametrize("num_heads", [16, 128])
@pytest.mark.parametrize(
    "seqlens_q,seqlens_k",
    [
        # Single-sequence (bs=1) cases — aligned and unaligned Q lengths
        ([16], [128]),
        ([17], [128]),
        ([33], [256]),
        ([64], [256]),
        # Uniform multi-batch cases
        ([32, 32], [256, 256]),
        ([33, 33, 33, 33], [256, 256, 256, 256]),
        # Different Q lengths, same K length (non-monotonic Q)
        ([64, 16, 32], [256, 256, 256]),
        # Different Q and K lengths (randomized order, sq <= sk per index)
        ([32, 16, 48], [128, 64, 256]),
        ([48, 16, 32], [256, 128, 512]),
        # Minimal Q (single tile) with large K (randomized K)
        ([16, 16, 16, 16], [256, 64, 512, 128]),
        # Large Q spread (non-monotonic)
        ([128, 16, 64], [256, 256, 256]),
        # Q lengths NOT multiples of Q_TILE_M=16 (randomized; exercises partial-tile mask)
        ([50, 17, 33], [512, 128, 256]),
        ([31, 1, 47, 15], [256, 64, 512, 128]),
        # K lengths NOT multiples of 16 or 128 (randomized; exercises K remainder mask)
        ([32, 16], [200, 100]),
        ([33, 17], [137, 71]),
        ([31, 1, 15], [250, 50, 150]),
        # Full prefill (seqlen_q == seqlen_k) — a few cases for coverage
        ([64], [64]),
        ([17, 17], [17, 17]),
        ([128, 16, 50], [128, 16, 50]),
        # Medium-bucket-targeting cases. These verify correctness of the
        # Q_TILE_M=128/NumSGM=16 kernel; under the production predicate they
        # currently route to small (Q < kQTileThreshold=192 in the first two
        # cases) or large (Q=511 case below). Build with MLA_FORCE_DISPATCH=3
        # to exercise the medium kernel on these shapes.
        ([256, 256, 256, 256], [2048, 2048, 2048, 2048]),
        ([200, 200], [1536, 1536]),
        ([511], [2048]),
    ],
    ids=[
        "incremental_prefill_bs1_aligned_16",
        "incremental_prefill_bs1_unaligned_17",
        "incremental_prefill_bs1_unaligned_33",
        "incremental_prefill_bs1_aligned_64",
        "incremental_prefill_bs2_uniform_32",
        "incremental_prefill_bs4_uniform_unaligned_33",
        "incremental_prefill_diff_q_same_k",
        "incremental_prefill_diff_q_diff_k_small",
        "incremental_prefill_diff_q_diff_k",
        "incremental_prefill_min_q_large_k",
        "incremental_prefill_large_q_spread",
        "incremental_prefill_unaligned_q",
        "incremental_prefill_unaligned_q_tiny",
        "incremental_prefill_unaligned_k",
        "incremental_prefill_unaligned_qk",
        "incremental_prefill_unaligned_qk_tiny",
        "full_prefill_bs1",
        "full_prefill_bs2_unaligned",
        "full_prefill_mixed",
        "medium_bs4_q256_k2048",
        "medium_partial_q200_k1536",
        "medium_q511_k2048",
    ],
)
def test_mla_prefill(dtype, block_size, num_heads, seqlens_q, seqlens_k):
    """MLA prefill: mostly incremental (seqlen_q < seqlen_k), some full prefill."""
    for sq, sk in zip(seqlens_q, seqlens_k):
        if sq > sk:
            pytest.skip("seqlen_q must be <= seqlen_k for prefill")

    torch.random.manual_seed(42)

    D_latent = 512
    D_rope = 64
    D_ckv = D_latent + D_rope
    scale = (128 + D_rope) ** (-0.5)

    bs = len(seqlens_q)
    total_q = sum(seqlens_q)
    cu_seqlens_q = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(seqlens_q), 0).tolist()), dtype=torch.int32
    )
    seq_lens_k = torch.tensor(seqlens_k, dtype=torch.int32)

    block_num = (max(seqlens_k) + block_size - 1) // block_size
    pack_factor = 128 // block_size
    block_num = ((block_num + pack_factor - 1) // pack_factor) * pack_factor

    q_nope_cpu = torch.randn(total_q, num_heads, D_latent, dtype=dtype)
    q_pe_cpu = torch.randn(total_q, num_heads, D_rope, dtype=dtype)
    block_table_cpu = torch.randint(
        0, bs * block_num, (bs, block_num), dtype=torch.int32
    )
    kv_cache_cpu = torch.randn(
        block_table_cpu.max().item() + 1, block_size, D_ckv, dtype=dtype
    )

    out_ref = ref_mla_prefill_varlen(
        q_nope_cpu,
        q_pe_cpu,
        kv_cache_cpu,
        scale,
        block_table_cpu,
        cu_seqlens_q,
        seq_lens_k,
        causal=True,
    )

    q_nope_xpu = q_nope_cpu.to(device).contiguous()
    q_pe_xpu = q_pe_cpu.to(device).contiguous()
    kv_cache_xpu = kv_cache_cpu.to(device)
    block_table_xpu = block_table_cpu.to(device)
    cu_seqlens_q_xpu = cu_seqlens_q.to(device)
    seq_lens_k_xpu = seq_lens_k.to(device)

    ws_size = flash_mla_prefill_get_workspace_size(block_num * block_size, bs)
    workspace = torch.empty(ws_size, device=device, dtype=torch.uint8)

    out = flash_mla_prefill(
        q_nope_xpu,
        q_pe_xpu,
        kv_cache_xpu,
        cu_seqlens_q_xpu,
        seq_lens_k_xpu,
        max(seqlens_q),
        block_table_xpu,
        workspace,
        scale,
        causal=True,
        num_kv_splits=1,
    )
    torch.xpu.synchronize()

    atol, rtol = (1e-2, 1e-2) if dtype == torch.bfloat16 else (1e-3, 1e-3)
    torch.testing.assert_close(out_ref.float(), out.cpu().float(), atol=atol, rtol=rtol)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
