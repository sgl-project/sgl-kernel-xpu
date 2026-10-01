import gc
import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F
from sgl_kernel import flash_mla_decode, flash_mla_decode_get_workspace_size
from torch import Tensor

LONG_TESTS = os.getenv("LONG_TESTS") == "1"
device = torch.device("xpu")

if not torch.xpu.is_available():
    pytest.skip(
        reason="Cutlass MLA Requires xpu device only.",
        allow_module_level=True,
    )


def clear_memory():
    """Clear GPU memory between tests to avoid OOM errors."""
    gc.collect()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
        torch.xpu.synchronize()


@pytest.fixture(autouse=True)
def reset_torch_defaults():
    """Reset torch defaults before and after each test to ensure isolation."""

    yield  # Run the test

    clear_memory()


def ref_mla(
    out: Tensor,  # (bs, num_heads, v_head_dim)
    query: Tensor,  # (bs, num_heads, head_dim)
    kv_cache: Tensor,  # (num_blocks, block_size, head_dim)
    scale: float,
    block_tables: Tensor,  # (bs, max_num_blocks)
    seq_lens: Tensor,  # (bs,)
    lse: Tensor = None,  # (bs, num_heads) fp32, log2-domain log-sum-exp
):
    bs, num_heads, v_head_dim = out.shape
    head_dim = query.shape[2]

    # Decode has a single query position, so attention reduces to two matmuls
    # over all heads at once. Doing this instead of one SDPA call per head
    # keeps the reference off a 128-iteration Python loop (the dominant cost of
    # this file) without materializing any per-head expansion of the KV cache.
    for i in range(bs):
        kv = kv_cache[block_tables[i]]  # (max_num_blocks, block_size, head_dim)
        kv = kv.view(-1, head_dim)[: seq_lens[i]].float()  # (seq_len, head_dim)
        v = kv[:, :v_head_dim]  # (seq_len, v_head_dim)

        # (num_heads, head_dim) @ (head_dim, seq_len) -> (num_heads, seq_len)
        scores = (query[i].float() @ kv.transpose(0, 1)) * scale
        probs = scores.softmax(dim=-1)
        out[i] = (probs @ v).to(out.dtype)  # (num_heads, v_head_dim)
        if lse is not None:
            # The kernel emits log2(sum_j exp2(score_j)) = logsumexp / ln(2).
            lse[i] = torch.logsumexp(scores, dim=-1) / math.log(2)

    return out


@pytest.mark.arch("xe20", "xe35")
@pytest.mark.parametrize("return_lse", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "mean_seq_len", [128, 1024, 4096] + ([8192, 16384, 32768] if LONG_TESTS else [])
)
@pytest.mark.parametrize("bs", [1, 2, 4])
@pytest.mark.parametrize("varlen", [True, False])
# TODO: enable block_size 1
@pytest.mark.parametrize("block_size", [16, 32, 64, 128])
@pytest.mark.parametrize("num_heads", [16, 32, 64, 128])
@pytest.mark.parametrize("num_kv_splits", [-1, 1])
@pytest.mark.parametrize(
    "dv, q_pe_dim, q_nope_dim",
    [(512, 64, 128), (256, 32, 64)],
    ids=["deepseek", "minicpm3"],
)
def test_flash_mla_decode(
    return_lse: bool,
    dtype: torch.dtype,
    mean_seq_len: int,
    bs: int,
    varlen: bool,
    block_size: int,
    num_heads: int,
    num_kv_splits: int,
    dv: int,
    q_pe_dim: int,
    q_nope_dim: int,
):

    torch.random.manual_seed(42)

    d = dv + q_pe_dim
    h_q = num_heads
    scale = (q_nope_dim + q_pe_dim) ** (-0.5)
    if varlen:
        seq_lens_cpu = torch.empty(bs, dtype=dtype).normal_(
            mean_seq_len, mean_seq_len / 2
        )
        seq_lens_cpu = seq_lens_cpu.clip(2).to(torch.int32)
    else:
        seq_lens_cpu = torch.full((bs,), mean_seq_len, dtype=torch.int32)
    max_seq_len = seq_lens_cpu.max().item()
    block_num = (max_seq_len + block_size - 1) // block_size

    # Pad block_num so that small blocks can be packed into full 128-sized CUTLASS tiles.
    # One 128-wide tile can hold (128 // block_size) small blocks.
    pack_factor = 128 // block_size
    block_num = ((block_num + pack_factor - 1) // pack_factor) * pack_factor

    q_cpu = torch.randn(bs, h_q, d, dtype=dtype, device="cpu") * 100
    block_table_cpu = torch.randint(
        0, bs * block_num, (bs, block_num), dtype=torch.int32, device="cpu"
    )
    kv_cache_cpu = torch.randn(
        block_table_cpu.numel(), block_size, d, dtype=dtype, device="cpu"
    )

    # --- Reference: run on CPU ---
    out_ref = torch.zeros(bs, h_q, dv, dtype=dtype, device="cpu")
    lse_ref = (
        torch.zeros(bs, h_q, dtype=torch.float32, device="cpu") if return_lse else None
    )
    ref_mla(out_ref, q_cpu, kv_cache_cpu, scale, block_table_cpu, seq_lens_cpu, lse_ref)

    # --- Kernel under test: run on XPU ---
    q_xpu = q_cpu.to(device=device)
    kv_cache_xpu = kv_cache_cpu.to(device=device)
    block_table_xpu = block_table_cpu.to(device=device)
    seq_lens_xpu = seq_lens_cpu.to(device=device)
    del q_cpu, kv_cache_cpu, block_table_cpu, seq_lens_cpu

    workspace_size = flash_mla_decode_get_workspace_size(
        block_num * block_size, bs, h_q, block_size, num_kv_splits=num_kv_splits
    )
    workspace = torch.empty(workspace_size, device=device, dtype=torch.uint8)

    q_nope = torch.empty((h_q, bs, dv), dtype=dtype, device=device).transpose(0, 1)
    q_nope.copy_(q_xpu[:, :, :dv])
    q_pe = q_xpu[:, :, dv:].clone()
    del q_xpu
    ret = flash_mla_decode(
        q_nope,
        q_pe,
        kv_cache_xpu,
        seq_lens_xpu,
        block_table_xpu,
        workspace,
        scale,
        num_kv_splits,
        return_lse=return_lse,
    )
    out, lse = ret if return_lse else (ret, None)
    torch.xpu.synchronize()
    atol, rtol = (1e-2, 1e-2) if dtype == torch.bfloat16 else (1e-3, 1e-3)
    torch.testing.assert_close(out_ref.float(), out.cpu().float(), atol=atol, rtol=rtol)

    if return_lse:
        assert lse.shape == (bs, h_q)
        assert lse.dtype == torch.float32
        lse_atol, lse_rtol = (2e-2, 2e-2) if dtype == torch.bfloat16 else (5e-3, 5e-3)
        torch.testing.assert_close(lse_ref, lse.cpu(), atol=lse_atol, rtol=lse_rtol)

    del ret, out, lse, out_ref, lse_ref, q_nope, q_pe, kv_cache_xpu, block_table_xpu
    del workspace, seq_lens_xpu


# --------------------------------------------------------------------------- #
# Multi-token decode: s_q in {2, 4, 8, 16} query tokens per request (MTP /
# speculative decode). q is 4D (bs, s_q, num_heads, dim); the s_q tokens are the
# last s_q positions of the sequence and are causally masked among themselves.
# --------------------------------------------------------------------------- #


def ref_mla_multi_token(
    query: Tensor,  # (bs, s_q, num_heads, head_dim)
    kv_cache: Tensor,  # (num_blocks, block_size, head_dim)
    scale: float,
    block_tables: Tensor,  # (bs, max_num_blocks)
    seq_lens: Tensor,  # (bs,), includes the s_q new tokens
    dv: int,
):
    """Causal multi-token decode reference.

    Token t of request i attends to KV positions [0, seq_lens[i] - s_q + t].
    Returns out (bs, s_q, num_heads, dv) fp32 and lse (bs, s_q, num_heads) fp32
    in the log2 domain, matching the kernel.
    """
    bs, s_q, num_heads, head_dim = query.shape
    out = torch.empty(bs, s_q, num_heads, dv, dtype=torch.float32)
    lse = torch.empty(bs, s_q, num_heads, dtype=torch.float32)
    for i in range(bs):
        seq_len = int(seq_lens[i])
        kv = kv_cache[block_tables[i]].view(-1, head_dim)[:seq_len].float()
        v = kv[:, :dv]

        # (s_q, num_heads, head_dim) x (seq_len, head_dim) -> (s_q, num_heads, seq_len)
        scores = torch.einsum("shd,kd->shk", query[i].float(), kv) * scale
        q_pos = torch.arange(s_q).view(s_q, 1, 1)
        k_pos = torch.arange(seq_len).view(1, 1, seq_len)
        scores = scores.masked_fill(k_pos > (seq_len - s_q) + q_pos, float("-inf"))

        out[i] = scores.softmax(dim=-1) @ v
        lse[i] = torch.logsumexp(scores, dim=-1) / math.log(2)
    return out, lse


def _make_decode_inputs(bs, s_q, h_q, d, mean_seq_len, varlen, block_size, dtype):
    if varlen:
        seq_lens = torch.empty(bs, dtype=torch.float32).normal_(
            mean_seq_len, mean_seq_len / 2
        )
        # Every sequence must hold at least its s_q new tokens.
        seq_lens = seq_lens.clip(min=max(2, s_q)).to(torch.int32)
    else:
        seq_lens = torch.full((bs,), max(mean_seq_len, s_q), dtype=torch.int32)
    max_seq_len = seq_lens.max().item()
    block_num = (max_seq_len + block_size - 1) // block_size
    pack_factor = 128 // block_size
    block_num = ((block_num + pack_factor - 1) // pack_factor) * pack_factor

    q = torch.randn(bs, s_q, h_q, d, dtype=dtype) * 100
    block_table = torch.randint(0, bs * block_num, (bs, block_num), dtype=torch.int32)
    kv_cache = torch.randn(block_table.numel(), block_size, d, dtype=dtype)
    return q, kv_cache, block_table, seq_lens, block_num


@pytest.mark.arch("xe20", "xe35")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("s_q", [2, 4, 8, 16])
# 20 is shorter than one 128-wide split, so num_kv_splits=4 leaves empty splits,
# and with s_q=16 the causal boundary falls inside the only (partial) KV tile.
@pytest.mark.parametrize("mean_seq_len", [20, 1024])
@pytest.mark.parametrize("bs", [1, 3])
@pytest.mark.parametrize("varlen", [True, False])
# 16: one subgroup along KV; 64/128: sub-page KV tiles for s_q >= 8 / >= 4.
@pytest.mark.parametrize("block_size", [16, 64, 128])
@pytest.mark.parametrize("num_heads", [16, 128])
@pytest.mark.parametrize("num_kv_splits", [-1, 1, 4])
def test_flash_mla_decode_multi_token(
    dtype: torch.dtype,
    s_q: int,
    mean_seq_len: int,
    bs: int,
    varlen: bool,
    block_size: int,
    num_heads: int,
    num_kv_splits: int,
):
    torch.random.manual_seed(42)

    dv, q_pe_dim = 512, 64
    d = dv + q_pe_dim
    scale = d ** (-0.5)

    q, kv_cache, block_table, seq_lens, block_num = _make_decode_inputs(
        bs, s_q, num_heads, d, mean_seq_len, varlen, block_size, dtype
    )
    out_ref, lse_ref = ref_mla_multi_token(
        q, kv_cache, scale, block_table, seq_lens, dv
    )

    workspace_size = flash_mla_decode_get_workspace_size(
        block_num * block_size,
        bs,
        num_heads,
        block_size,
        num_kv_splits=num_kv_splits,
        seq_len_q=s_q,
    )
    workspace = torch.empty(workspace_size, device=device, dtype=torch.uint8)

    q_xpu = q.to(device)
    out, lse = flash_mla_decode(
        q_xpu[..., :dv].contiguous(),
        q_xpu[..., dv:].contiguous(),
        kv_cache.to(device),
        seq_lens.to(device),
        block_table.to(device),
        workspace,
        scale,
        num_kv_splits,
        return_lse=True,
    )
    torch.xpu.synchronize()

    assert out.shape == (bs, s_q, num_heads, dv)
    assert lse.shape == (bs, s_q, num_heads)
    assert lse.dtype == torch.float32
    atol, rtol = (1e-2, 1e-2) if dtype == torch.bfloat16 else (1e-3, 1e-3)
    torch.testing.assert_close(out_ref, out.cpu().float(), atol=atol, rtol=rtol)
    lse_atol, lse_rtol = (2e-2, 2e-2) if dtype == torch.bfloat16 else (5e-3, 5e-3)
    torch.testing.assert_close(lse_ref, lse.cpu(), atol=lse_atol, rtol=lse_rtol)

    del out, lse, out_ref, lse_ref, q, q_xpu, kv_cache, block_table, workspace


@pytest.mark.arch("xe20", "xe35")
@pytest.mark.parametrize("num_kv_splits", [-1, 1, 4])
@pytest.mark.parametrize("block_size", [16, 64])
def test_flash_mla_decode_4d_s_q_1_matches_3d(num_kv_splits: int, block_size: int):
    """(bs, 1, H, D) must give exactly the (bs, H, D) result: same kernel, no causal mask."""
    torch.random.manual_seed(0)
    bs, num_heads, dv, q_pe_dim = 2, 32, 512, 64
    d = dv + q_pe_dim
    scale = d ** (-0.5)

    q, kv_cache, block_table, seq_lens, block_num = _make_decode_inputs(
        bs, 1, num_heads, d, 512, True, block_size, torch.bfloat16
    )
    workspace_size = flash_mla_decode_get_workspace_size(
        block_num * block_size, bs, num_heads, block_size, num_kv_splits=num_kv_splits
    )
    workspace = torch.empty(workspace_size, device=device, dtype=torch.uint8)
    q_xpu = q.to(device)
    args = (
        kv_cache.to(device),
        seq_lens.to(device),
        block_table.to(device),
        workspace,
        scale,
        num_kv_splits,
    )

    out_4d, lse_4d = flash_mla_decode(
        q_xpu[..., :dv].contiguous(),
        q_xpu[..., dv:].contiguous(),
        *args,
        return_lse=True,
    )
    out_3d, lse_3d = flash_mla_decode(
        q_xpu[:, 0, :, :dv].contiguous(),
        q_xpu[:, 0, :, dv:].contiguous(),
        *args,
        return_lse=True,
    )
    torch.xpu.synchronize()

    assert out_4d.shape == (bs, 1, num_heads, dv)
    assert lse_4d.shape == (bs, 1, num_heads)
    torch.testing.assert_close(out_4d[:, 0], out_3d, atol=0, rtol=0)
    torch.testing.assert_close(lse_4d[:, 0], lse_3d, atol=0, rtol=0)


@pytest.mark.arch("xe20", "xe35")
def test_flash_mla_decode_workspace_scales_with_s_q():
    args = (4096, 4, 128, 64)
    ws_1 = flash_mla_decode_get_workspace_size(*args, num_kv_splits=8)
    ws_8 = flash_mla_decode_get_workspace_size(*args, num_kv_splits=8, seq_len_q=8)
    assert ws_1 > 0
    # Each of the three sub-buffers is padded to 256 B, so allow that slack.
    assert ws_8 >= 8 * ws_1 - 8 * 3 * 256
    assert ws_8 <= 8 * ws_1 + 3 * 256


@pytest.mark.arch("xe20", "xe35")
def test_flash_mla_decode_rejects_unsupported_s_q():
    bs, s_q, num_heads, dv, q_pe_dim, block_size = 1, 3, 16, 512, 64, 64
    q_nope = torch.randn(bs, s_q, num_heads, dv, dtype=torch.bfloat16, device=device)
    q_pe = torch.randn(
        bs, s_q, num_heads, q_pe_dim, dtype=torch.bfloat16, device=device
    )
    kv_cache = torch.randn(
        2, block_size, dv + q_pe_dim, dtype=torch.bfloat16, device=device
    )
    seq_lens = torch.full((bs,), 64, dtype=torch.int32, device=device)
    block_table = torch.zeros(bs, 2, dtype=torch.int32, device=device)
    workspace = torch.empty(0, dtype=torch.uint8, device=device)
    with pytest.raises(AssertionError, match="s_q must be"):
        flash_mla_decode(
            q_nope, q_pe, kv_cache, seq_lens, block_table, workspace, 1.0, 1
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
