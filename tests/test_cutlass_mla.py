import gc
import sys

import pytest
import torch
import torch.nn.functional as F
from sgl_kernel import cutlass_mla_decode, cutlass_mla_get_workspace_size
from torch import Tensor

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
):
    bs, num_heads, v_head_dim = out.shape
    head_dim = query.shape[2]

    for i in range(bs):
        # gather and flatten KV-cache
        kv = kv_cache[block_tables[i]]  # (max_num_blocks, block_size, head_dim)
        kv = kv.view(1, -1, head_dim)[:, : seq_lens[i]]  # (1, seq_len, head_dim)
        v = kv[:, :, :v_head_dim]

        q = query[i].view(num_heads, 1, head_dim)
        o = F.scaled_dot_product_attention(q, kv, v, scale=scale, enable_gqa=True)
        out[i] = o.view(num_heads, v_head_dim)

    return out


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mean_seq_len", [128, 1024, 4096])
@pytest.mark.parametrize("bs", [1, 2, 4])
@pytest.mark.parametrize("varlen", [True, False])
# TODO: enable block_size 1
@pytest.mark.parametrize("block_size", [16, 32, 64, 128])
@pytest.mark.parametrize("num_heads", [16, 32, 64, 128])
# TODO: enable num_kv_splits >1
@pytest.mark.parametrize("num_kv_splits", [-1, 1])
def test_cutlass_mla_decode(
    dtype: torch.dtype,
    mean_seq_len: int,
    bs: int,
    varlen: bool,
    block_size: int,
    num_heads: int,
    num_kv_splits: int,
):

    torch.random.manual_seed(42)

    d = 576
    h_q = num_heads
    dv = 512

    q_nope_dim = 128
    q_pe_dim = 64
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
    ref_mla(out_ref, q_cpu, kv_cache_cpu, scale, block_table_cpu, seq_lens_cpu)
    # out_ref = out_ref.to(device=device)

    # --- Kernel under test: run on XPU ---
    q_xpu = q_cpu.to(device=device)
    kv_cache_xpu = kv_cache_cpu.to(device=device)
    block_table_xpu = block_table_cpu.to(device=device)
    seq_lens_xpu = seq_lens_cpu.to(device=device)
    del q_cpu, kv_cache_cpu, block_table_cpu, seq_lens_cpu

    workspace_size = cutlass_mla_get_workspace_size(
        block_num * block_size, bs, num_kv_splits=num_kv_splits
    )
    workspace = torch.empty(workspace_size, device=device, dtype=torch.uint8)

    q_nope = torch.empty((h_q, bs, dv), dtype=dtype, device=device).transpose(0, 1)
    q_nope.copy_(q_xpu[:, :, :dv])
    q_pe = q_xpu[:, :, dv:].clone()
    del q_xpu
    out = cutlass_mla_decode(
        q_nope,
        q_pe,
        kv_cache_xpu,
        seq_lens_xpu,
        block_table_xpu,
        workspace,
        scale,
        num_kv_splits,
    )
    torch.xpu.synchronize()
    atol, rtol = (1e-2, 1e-2) if dtype == torch.bfloat16 else (1e-3, 1e-3)
    torch.testing.assert_close(out_ref.float(), out.cpu().float(), atol=atol, rtol=rtol)

    del out, out_ref, q_nope, q_pe, kv_cache_xpu, block_table_xpu
    del workspace, seq_lens_xpu


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
