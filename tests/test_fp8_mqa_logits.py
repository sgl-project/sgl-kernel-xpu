"""Unit tests for FP8 MQA logits kernels."""

import struct
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="XPU not available",
)

import sgl_kernel  # noqa: F401, E402 — triggers op registration
from sgl_kernel.nsa import _fp8_mqa_logits_impl


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
        # Larger sizes that exercise SYCL-TLA GEMM path (M=Nq*H >= 32, Nk >= 128)
        (1, 64, 128, 128),
        (1, 64, 128, 256),
        (4, 64, 128, 512),
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

    logits = _fp8_mqa_logits_impl(
        q.view(torch.uint8), k.view(torch.uint8), k_scale, weights, ks, ke
    )
    ref = reference_fp8_mqa_logits(q, k, k_scale, weights, ks, ke)

    torch.testing.assert_close(logits.cpu(), ref.cpu(), rtol=2e-3, atol=0.1)


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

    logits = _fp8_mqa_logits_impl(
        q.view(torch.uint8), k.view(torch.uint8), k_scale, weights, ks, ke
    )
    ref = reference_fp8_mqa_logits(q, k, k_scale, weights, ks, ke)

    logits_cpu = logits.cpu()
    # Invalid positions should be zero
    assert logits_cpu[0, :2].abs().max().item() == 0.0
    assert logits_cpu[0, 5:].abs().max().item() == 0.0
    assert logits_cpu[1, :4].abs().max().item() == 0.0
    assert logits_cpu[1, 7:].abs().max().item() == 0.0
    torch.testing.assert_close(logits_cpu, ref.cpu(), rtol=2e-3, atol=0.1)


@pytest.mark.parametrize(
    "B,H,D,page_size",
    [
        (1, 4, 128, 4),  # small H, page_size=4
        (1, 4, 128, 8),  # small H, page_size=8
        (1, 64, 128, 64),  # large H / SYCL-TLA (xe20) path
    ],
)
def test_fp8_paged_mqa_logits(B, H, D, page_size):
    device = "xpu"
    num_pages = 8
    seq_len = page_size * 3  # 3 pages used, 1 page padding
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
        True,  # clean_logits: ensure out-of-range positions are zeroed
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
    torch.testing.assert_close(logits_cpu, ref, rtol=2e-3, atol=0.1)
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
        True,
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

    torch.testing.assert_close(logits.cpu(), ref, rtol=2e-3, atol=0.1)


def test_python_wrapper():
    """Test the Python wrapper (nsa.py) interface with numerical correctness."""
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

    # Verify numerical correctness against reference
    ref = reference_fp8_mqa_logits(q, k, k_scale, weights, ks, ke)
    torch.testing.assert_close(logits.cpu(), ref.cpu(), rtol=2e-3, atol=0.1)

    # Also test paged wrapper shape/dtype (full correctness tested in paged tests)
    page_size = 1
    num_pages = Nk
    B = 1
    kv_cache = make_kv_cache(num_pages, page_size, D, device)
    seq_lens = torch.tensor([Nk], dtype=torch.int32, device=device)
    block_tables = torch.arange(num_pages, dtype=torch.int32, device=device).unsqueeze(
        0
    )
    q_paged = make_fp8_tensor((B, 1, H, D), device)
    w_paged = torch.ones(B, H, dtype=torch.float32, device=device)
    paged_logits = fp8_paged_mqa_logits(
        q_paged.view(torch.uint8),
        kv_cache,
        w_paged,
        seq_lens,
        block_tables,
        None,
        Nk,
    )
    assert paged_logits.shape == (B, Nk)
    assert paged_logits.dtype == torch.float32


@pytest.mark.parametrize(
    "Nq,H,D,Nk,aligned",
    [
        # aligned → head-loop uses _scaled_mm (Nq, Nk, D all divisible by 16)
        (16, 4, 128, 16, True),
        # unaligned → head-loop uses bf16 fallback
        (3, 4, 128, 15, False),
    ],
    ids=["head_loop_scaled_mm", "head_loop_bf16"],
)
def test_fp8_mqa_logits_head_loop(Nq, H, D, Nk, aligned):
    """Head-loop OOM-avoidance path is numerically correct.

    The threshold is patched to 0 so that even small tensors take the head-loop
    branch, exercising both the _scaled_mm sub-path (aligned sizes) and the bf16
    sub-path (unaligned sizes).
    """
    import sgl_kernel.nsa as nsa_mod

    device = "xpu"
    q = make_fp8_tensor((Nq, H, D), device)
    k = make_fp8_tensor((Nk, D), device)
    k_scale = torch.rand(Nk, dtype=torch.float32, device=device) + 0.5
    weights = torch.rand(Nq, H, dtype=torch.float32, device=device)
    ks = torch.zeros(Nq, dtype=torch.int32, device=device)
    ke = torch.full((Nq,), Nk, dtype=torch.int32, device=device)

    ref = reference_fp8_mqa_logits(q, k, k_scale, weights, ks, ke)

    # Patch threshold to 0 so every call takes the head-loop branch.
    original = nsa_mod._HEAD_LOOP_THRESHOLD
    nsa_mod._HEAD_LOOP_THRESHOLD = 0
    try:
        logits = nsa_mod._fp8_mqa_logits_impl(
            q.view(torch.uint8), k.view(torch.uint8), k_scale, weights, ks, ke
        )
    finally:
        nsa_mod._HEAD_LOOP_THRESHOLD = original

    assert logits.shape == (Nq, Nk)
    torch.testing.assert_close(logits.cpu(), ref.cpu(), rtol=2e-3, atol=0.1)


def test_fp8_paged_mqa_logits_3d_input():
    """fp8_paged_mqa_logits accepts a 3D (B, H, D) query and matches the 4D result.

    The DSA indexer passes q_fp8 as (B, H, D) in decode mode; the wrapper must
    unsqueeze dim 1 before calling the SYCL kernel.
    """
    from sgl_kernel.nsa import fp8_paged_mqa_logits

    device = "xpu"
    B, H, D = 2, 4, 128
    page_size = 4
    seq_len = 16
    num_pages = seq_len // page_size

    kv_cache = make_kv_cache(num_pages, page_size, D, device)
    seq_lens = torch.full((B,), seq_len, dtype=torch.int32, device=device)
    block_tables = torch.stack(
        [torch.arange(num_pages, dtype=torch.int32, device=device)] * B
    )
    weights = torch.rand(B, H, dtype=torch.float32, device=device)

    q_4d = make_fp8_tensor((B, 1, H, D), device)
    q_3d = q_4d.squeeze(1)  # (B, H, D)

    common_kwargs = dict(
        kv_cache=kv_cache,
        weights=weights,
        seq_lens=seq_lens,
        block_tables=block_tables,
        schedule_metadata=None,
        max_seq_len=seq_len,
    )

    out_4d = fp8_paged_mqa_logits(q_4d.view(torch.uint8), **common_kwargs)
    out_3d = fp8_paged_mqa_logits(q_3d.view(torch.uint8), **common_kwargs)

    assert out_3d.shape == (B, seq_len)
    torch.testing.assert_close(out_3d, out_4d)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
