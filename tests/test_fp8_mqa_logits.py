"""Unit tests for FP8 MQA logits kernels."""

import struct
import sys

import pytest
import sgl_kernel  # noqa: F401 — triggers op registration
import torch


def make_fp8_tensor(shape, device="xpu"):
    """Create a random FP8 e4m3 tensor by quantizing random floats."""
    t = torch.randn(shape, dtype=torch.float32, device=device) * 0.5
    t_fp8 = t.to(torch.float8_e4m3fn)
    return t_fp8


def reference_fp8_mqa_logits(q_fp8, k_fp8, k_scale, weights, ks, ke):
    """PyTorch reference for fp8_mqa_logits."""
    q = q_fp8.to(torch.float32)  # (Nq, H, D)
    k = k_fp8.to(torch.float32)  # (Nk, D)

    # dots[i,h,j] = q[i,h,:] · k[j,:]
    dots = torch.einsum("ihd,jd->ihj", q, k)  # (Nq, H, Nk)
    dots = torch.relu(dots)
    # Weighted sum over heads: weights is (Nq, H)
    logits = torch.einsum("ihn,ih->in", dots, weights)  # (Nq, Nk)
    logits = logits * k_scale.unsqueeze(0)  # broadcast k_scale (Nk,)

    # Zero out positions outside [ks, ke) range
    Nq, Nk = logits.shape
    j_idx = torch.arange(Nk, device=logits.device).unsqueeze(0)  # (1, Nk)
    mask = (j_idx >= ks.unsqueeze(1)) & (j_idx < ke.unsqueeze(1))  # (Nq, Nk)
    logits = logits * mask.float()

    return logits


def reference_fp8_paged_mqa_logits(
    q_fp8, kv_cache_uint8, weights, seq_lens, block_tables, max_seq_len, page_size, D
):
    """PyTorch reference for fp8_paged_mqa_logits."""
    B = q_fp8.shape[0]
    H = q_fp8.shape[2]
    q = q_fp8.cpu().to(torch.float32).reshape(B, H, D)

    head_dim_with_sf = D + 4
    seq_lens_cpu = seq_lens.cpu().flatten().int()
    block_tables_cpu = block_tables.cpu().int()
    kv_flat = kv_cache_uint8.cpu().flatten()

    logits = torch.zeros(B, max_seq_len, dtype=torch.float32)
    for b in range(B):
        sl = seq_lens_cpu[b].item()
        for kj in range(sl):
            page_idx = block_tables_cpu[b, kj // page_size].item()
            token_in_page = kj % page_size
            offset = (
                page_idx * page_size * head_dim_with_sf
                + token_in_page * head_dim_with_sf
            )

            k_bytes = kv_flat[offset : offset + D].to(torch.uint8)
            k_fp8 = k_bytes.view(torch.float8_e4m3fn).to(torch.float32)
            scale_bytes = kv_flat[offset + D : offset + D + 4].tolist()
            k_sc = struct.unpack("<f", bytes(int(x) for x in scale_bytes))[0]

            # dots[h] = q[b,h,:] · k[:]
            dots = torch.mv(q[b], k_fp8)  # (H,)
            dots = torch.relu(dots)
            score = (dots * weights[b].cpu()).sum().item() * k_sc
            logits[b, kj] = score

    return logits


def make_kv_cache(num_pages, page_size, D, device="cpu"):
    """Build a KV cache with random FP8 keys and float32 scales.
    KV cache is uint8 because each token packs 128 bytes of FP8 key data + 4 bytes of float32 scale (132 bytes total).
    Since it mixes two dtypes, uint8 is the only representation — this matches the real NSA KV cache format.
    """
    head_dim_with_sf = D + 4
    kv_cache = torch.zeros(
        num_pages, page_size, 1, head_dim_with_sf, dtype=torch.uint8, device="cpu"
    )
    for p in range(num_pages):
        for t in range(page_size):
            k_float = torch.randn(D, dtype=torch.float32) * 0.5
            k_fp8 = k_float.to(torch.float8_e4m3fn)
            kv_cache[p, t, 0, :D] = k_fp8.view(torch.uint8)
            scale = 0.5 + torch.rand(1).item()
            scale_bytes = struct.pack("<f", scale)
            for i, byte_val in enumerate(scale_bytes):
                kv_cache[p, t, 0, D + i] = byte_val
    return kv_cache.to(device)


@pytest.mark.parametrize(
    "Nq,H,D,Nk",
    [
        (2, 4, 128, 8),
        (1, 4, 128, 16),
        (4, 8, 128, 32),
    ],
)
def test_fp8_mqa_logits(Nq, H, D, Nk):
    device = "xpu"
    q = make_fp8_tensor((Nq, H, D), device)
    k = make_fp8_tensor((Nk, D), device)
    k_scale = torch.rand(Nk, dtype=torch.float32, device=device) + 0.5
    weights = torch.rand(Nq, H, dtype=torch.float32, device=device)
    ks = torch.zeros(Nq, dtype=torch.int32, device=device)
    ke = torch.full((Nq,), Nk, dtype=torch.int32, device=device)

    logits = torch.ops.sgl_kernel.fp8_mqa_logits.default(
        q.view(torch.uint8), k.view(torch.uint8), k_scale, weights, ks, ke
    )
    ref = reference_fp8_mqa_logits(q, k, k_scale, weights, ks, ke)

    torch.testing.assert_close(logits.cpu(), ref.cpu(), rtol=1e-3, atol=1e-3)


def test_fp8_mqa_logits_masking():
    """Verify that positions outside [ks, ke) are zeroed."""
    device = "xpu"
    Nq, H, D, Nk = 2, 4, 128, 8
    q = make_fp8_tensor((Nq, H, D), device)
    k = make_fp8_tensor((Nk, D), device)
    k_scale = torch.ones(Nk, dtype=torch.float32, device=device)
    weights = torch.ones(Nq, H, dtype=torch.float32, device=device)
    ks = torch.tensor([2, 4], dtype=torch.int32, device=device)
    ke = torch.tensor([5, 7], dtype=torch.int32, device=device)

    logits = torch.ops.sgl_kernel.fp8_mqa_logits.default(
        q.view(torch.uint8), k.view(torch.uint8), k_scale, weights, ks, ke
    )
    ref = reference_fp8_mqa_logits(q, k, k_scale, weights, ks, ke)

    logits_cpu = logits.cpu()
    # Invalid positions should be zero
    assert logits_cpu[0, :2].abs().max().item() == 0.0
    assert logits_cpu[0, 5:].abs().max().item() == 0.0
    assert logits_cpu[1, :4].abs().max().item() == 0.0
    assert logits_cpu[1, 7:].abs().max().item() == 0.0
    torch.testing.assert_close(logits_cpu, ref.cpu(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("page_size", [4, 8])
def test_fp8_paged_mqa_logits(page_size):
    device = "xpu"
    B, H, D = 1, 4, 128
    num_pages = 8
    seq_len = page_size * 3  # 3 pages
    max_num_blocks = 4
    max_seq_len = max_num_blocks * page_size

    kv_cache = make_kv_cache(num_pages, page_size, D, device)
    q = make_fp8_tensor((B, 1, H, D), device)
    weights = torch.rand(B, H, dtype=torch.float32, device=device)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    block_tables = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32, device=device)

    logits = torch.ops.sgl_kernel.fp8_paged_mqa_logits.default(
        q.view(torch.uint8),
        kv_cache,
        weights,
        seq_lens,
        block_tables,
        None,
        max_seq_len,
        False,
    )
    ref = reference_fp8_paged_mqa_logits(
        q,
        kv_cache,
        weights,
        seq_lens,
        block_tables,
        max_seq_len,
        page_size,
        D,
    )

    logits_cpu = logits.cpu()
    torch.testing.assert_close(logits_cpu, ref, rtol=1e-3, atol=1e-3)

    if seq_len < max_seq_len:
        assert logits_cpu[0, seq_len:].abs().max().item() == 0.0


def test_fp8_paged_mqa_logits_noncontiguous_pages():
    """Test with non-contiguous page mapping."""
    device = "xpu"
    B, H, D = 1, 4, 128
    page_size = 4
    num_pages = 8
    seq_len = 8  # 2 pages
    max_num_blocks = 4
    max_seq_len = max_num_blocks * page_size

    kv_cache = make_kv_cache(num_pages, page_size, D, device)
    q = make_fp8_tensor((B, 1, H, D), device)
    weights = torch.rand(B, H, dtype=torch.float32, device=device)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    block_tables = torch.tensor([[3, 7, 0, 0]], dtype=torch.int32, device=device)

    logits = torch.ops.sgl_kernel.fp8_paged_mqa_logits.default(
        q.view(torch.uint8),
        kv_cache,
        weights,
        seq_lens,
        block_tables,
        None,
        max_seq_len,
        False,
    )
    ref = reference_fp8_paged_mqa_logits(
        q,
        kv_cache,
        weights,
        seq_lens,
        block_tables,
        max_seq_len,
        page_size,
        D,
    )

    torch.testing.assert_close(logits.cpu(), ref, rtol=1e-3, atol=1e-3)


def test_python_wrapper():
    """Test the Python wrapper (nsa.py) interface."""
    from sgl_kernel.nsa import fp8_mqa_logits, fp8_paged_mqa_logits  # noqa: F811

    device = "xpu"
    Nq, H, D, Nk = 2, 4, 128, 8

    q = make_fp8_tensor((Nq, H, D), device)
    k = make_fp8_tensor((Nk, D), device)
    k_scale = torch.ones(Nk, dtype=torch.float32, device=device)
    weights = torch.ones(Nq, H, dtype=torch.float32, device=device)
    ks = torch.zeros(Nq, dtype=torch.int32, device=device)
    ke = torch.full((Nq,), Nk, dtype=torch.int32, device=device)

    logits = fp8_mqa_logits(
        q.view(torch.uint8), (k.view(torch.uint8), k_scale), weights, ks, ke
    )
    assert logits.shape == (Nq, Nk)
    assert logits.dtype == torch.float32


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
