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


@pytest.mark.parametrize("num_kv_splits", [-1, 1])
def test_flash_mla_decode_lse_optional(num_kv_splits: int):
    """Skipping the LSE must be bit-identical on O, and return_lse defaults to False.

    Correctness of O and of the LSE values is already covered per-mode by
    test_flash_mla_decode's return_lse parametrization; what that cannot check is
    agreement *between* the two modes, since it compares each against the CPU
    reference at 1e-2 in separate invocations. Omitting the LSE passes a null LSE
    pointer that the epilogue branches on at runtime, so this pins O to be
    unperturbed by taking that branch. Both KV-split modes are covered: 1 split
    writes the LSE from the fused epilogue, auto (-1) may pick more and write it
    from the split-KV reduction kernel.
    """
    torch.random.manual_seed(42)

    dtype = torch.bfloat16
    bs, h_q, dv, q_pe_dim = 2, 16, 512, 64
    d = dv + q_pe_dim
    block_size, seq_len = 64, 256
    scale = (128 + q_pe_dim) ** (-0.5)

    seq_lens_cpu = torch.full((bs,), seq_len, dtype=torch.int32)
    block_num = seq_len // block_size
    pack_factor = 128 // block_size
    block_num = ((block_num + pack_factor - 1) // pack_factor) * pack_factor

    q_cpu = torch.randn(bs, h_q, d, dtype=dtype, device="cpu")
    block_table_cpu = torch.randint(
        0, bs * block_num, (bs, block_num), dtype=torch.int32, device="cpu"
    )
    kv_cache_cpu = torch.randn(
        block_table_cpu.numel(), block_size, d, dtype=dtype, device="cpu"
    )

    workspace = torch.empty(
        flash_mla_decode_get_workspace_size(
            block_num * block_size, bs, h_q, block_size, num_kv_splits=num_kv_splits
        ),
        device=device,
        dtype=torch.uint8,
    )
    args = (
        q_cpu[:, :, :dv].to(device).contiguous(),
        q_cpu[:, :, dv:].to(device).contiguous(),
        kv_cache_cpu.to(device),
        seq_lens_cpu.to(device),
        block_table_cpu.to(device),
        workspace,
        scale,
        num_kv_splits,
    )

    # out_default omits return_lse, so it takes the default (False) path and passes
    # no LSE tensor; out_lse asks for one on otherwise identical inputs.
    out_default = flash_mla_decode(*args)
    out_lse, _ = flash_mla_decode(*args, return_lse=True)
    torch.xpu.synchronize()

    assert isinstance(out_default, torch.Tensor)
    torch.testing.assert_close(out_default.cpu(), out_lse.cpu(), atol=0, rtol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
