"""Tests for the AOT SYCL Mamba HiCache transfer kernels.

    load  (pf->lf): dst[dst_idx[i]]        <- src[src_idx[i], layer_id]
    backup(lf->pf): dst[dst_idx[i], layer] <- src[layer, src_idx[i]]  (all layers)

These kernels are pure indexed block copies, so a correct copy is bit-exact:
every comparison here uses rtol=0, atol=0.

Layouts (all contiguous):
    page_first   [pool, num_layers, item_elems]
    layer_first  [num_layers, pool, item_elems]
    single-layer [pool, item_elems]
"""

import pytest
import torch
from sgl_kernel import transfer_kv_mamba_lf_pf, transfer_kv_mamba_pf_lf


@pytest.fixture(autouse=True)
def skip_if_no_xpu():
    if not torch.xpu.is_available():
        pytest.skip("XPU not available")


def _indices(pool, num_items):
    """Two independent permutations, so src and dst slots never line up."""
    src_idx = torch.randperm(pool, device="xpu")[:num_items].to(torch.int64)
    dst_idx = torch.randperm(pool, device="xpu")[:num_items].to(torch.int64)
    return src_idx, dst_idx


def _rand(shape, dtype):
    if dtype in (torch.bfloat16, torch.float16, torch.float32):
        return torch.randn(*shape, device="xpu", dtype=dtype)
    # Integer dtypes exercise the same byte copy with a different element width.
    return torch.randint(-128, 127, shape, device="xpu", dtype=dtype)


def _ref_load(src, dst, src_idx, dst_idx, layer_id, item_elems, num_layers):
    ref = dst.clone()
    ref.view(-1, item_elems)[dst_idx] = src.view(-1, num_layers, item_elems)[
        src_idx, layer_id
    ]
    return ref


def _ref_backup(src_layers, dst, src_idx, dst_idx, item_elems, num_layers):
    ref = dst.clone()
    s = src_layers.view(num_layers, -1, item_elems)
    ref.view(-1, num_layers, item_elems)[dst_idx] = s[:, src_idx, :].permute(1, 0, 2)
    return ref


@pytest.mark.parametrize("num_layers", [1, 8, 32])
@pytest.mark.parametrize("item_elems", [16, 128, 513, 2048])
@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16, torch.float32, torch.int8]
)
def test_load_pf_lf(dtype, item_elems, num_layers):
    """page_first -> single-layer, one layer slot."""
    torch.manual_seed(0)
    pool, num_items = 64, 48
    layer_id = num_layers // 2

    src = _rand((pool, num_layers, item_elems), dtype)
    dst = torch.zeros(pool, item_elems, device="xpu", dtype=dtype)
    src_idx, dst_idx = _indices(pool, num_items)

    ref = _ref_load(src, dst, src_idx, dst_idx, layer_id, item_elems, num_layers)

    item_size = item_elems * src.element_size()
    transfer_kv_mamba_pf_lf(
        src, dst, src_idx, dst_idx, layer_id, item_size, item_size * num_layers
    )
    torch.xpu.synchronize()

    torch.testing.assert_close(dst, ref, rtol=0, atol=0)


@pytest.mark.parametrize("num_layers", [1, 8, 32])
@pytest.mark.parametrize("item_elems", [16, 128, 513, 2048])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.int8])
def test_backup_lf_pf(dtype, item_elems, num_layers):
    """layer_first -> page_first, all layers at once."""
    torch.manual_seed(0)
    pool, num_items = 64, 48

    src_layers = _rand((num_layers, pool, item_elems), dtype)
    dst = torch.zeros(pool, num_layers, item_elems, device="xpu", dtype=dtype)
    src_idx, dst_idx = _indices(pool, num_items)

    ref = _ref_backup(src_layers, dst, src_idx, dst_idx, item_elems, num_layers)

    item_size = item_elems * src_layers.element_size()
    transfer_kv_mamba_lf_pf(
        src_layers, dst, src_idx, dst_idx, item_size, item_size * num_layers, num_layers
    )
    torch.xpu.synchronize()

    torch.testing.assert_close(dst, ref, rtol=0, atol=0)


@pytest.mark.parametrize("num_items", [1, 4])
@pytest.mark.parametrize("item_elems", [65536, 70000])
def test_chunked_grid_large_items(num_items, item_elems):
    """Few, very large items: the only shapes that take the multi-chunk grid path.

    Every other test here has enough copy units to fill the device, so the
    planner returns chunks == 1 and this splitting logic goes unexercised.
    70000 is deliberately not a whole multiple of the chunk stride, so the tail
    chunk is short.
    """
    torch.manual_seed(0)
    pool, num_layers, layer_id = num_items, 2, 1
    dtype = torch.float32

    src = _rand((pool, num_layers, item_elems), dtype)
    dst = torch.zeros(pool, item_elems, device="xpu", dtype=dtype)
    idx = torch.randperm(pool, device="xpu")[:num_items].to(torch.int64)
    rev = idx.flip(0).contiguous()
    item_size = item_elems * src.element_size()

    ref = _ref_load(src, dst, idx, rev, layer_id, item_elems, num_layers)
    transfer_kv_mamba_pf_lf(
        src, dst, idx, rev, layer_id, item_size, item_size * num_layers
    )
    torch.xpu.synchronize()
    torch.testing.assert_close(dst, ref, rtol=0, atol=0)

    src_layers = _rand((num_layers, pool, item_elems), dtype)
    dst_pf = torch.zeros(pool, num_layers, item_elems, device="xpu", dtype=dtype)
    ref_pf = _ref_backup(src_layers, dst_pf, idx, rev, item_elems, num_layers)
    transfer_kv_mamba_lf_pf(
        src_layers, dst_pf, idx, rev, item_size, item_size * num_layers, num_layers
    )
    torch.xpu.synchronize()
    torch.testing.assert_close(dst_pf, ref_pf, rtol=0, atol=0)


def test_odd_item_size_falls_back_to_narrow_copy():
    """An item size that is not 8-byte aligned must still copy exactly.

    The kernel picks its copy unit from the alignment of the pointers and byte
    strides; int8 with an odd item_elems forces the 1-byte unit.
    """
    torch.manual_seed(0)
    pool, num_items, num_layers, item_elems = 32, 32, 3, 45  # 45 B item, layer_id*45
    dtype = torch.int8

    src = _rand((pool, num_layers, item_elems), dtype)
    dst = torch.zeros(pool, item_elems, device="xpu", dtype=dtype)
    src_idx, dst_idx = _indices(pool, num_items)

    ref = _ref_load(src, dst, src_idx, dst_idx, 2, item_elems, num_layers)
    transfer_kv_mamba_pf_lf(
        src, dst, src_idx, dst_idx, 2, item_elems, item_elems * num_layers
    )
    torch.xpu.synchronize()
    torch.testing.assert_close(dst, ref, rtol=0, atol=0)


def test_empty_indices_is_noop():
    pool, num_layers, item_elems = 8, 4, 64
    src = _rand((pool, num_layers, item_elems), torch.bfloat16)
    dst = torch.zeros(pool, item_elems, device="xpu", dtype=torch.bfloat16)
    empty = torch.empty(0, device="xpu", dtype=torch.int64)
    item_size = item_elems * src.element_size()

    transfer_kv_mamba_pf_lf(
        src, dst, empty, empty, 0, item_size, item_size * num_layers
    )
    torch.xpu.synchronize()
    assert torch.count_nonzero(dst) == 0

    dst_pf = torch.zeros(
        pool, num_layers, item_elems, device="xpu", dtype=torch.bfloat16
    )
    src_layers = _rand((num_layers, pool, item_elems), torch.bfloat16)
    transfer_kv_mamba_lf_pf(
        src_layers, dst_pf, empty, empty, item_size, item_size * num_layers, num_layers
    )
    torch.xpu.synchronize()
    assert torch.count_nonzero(dst_pf) == 0


def test_load_then_backup_roundtrip():
    """backup(load(x)) must reproduce the original page-first pool."""
    torch.manual_seed(0)
    pool, num_layers, item_elems = 32, 4, 256
    dtype = torch.bfloat16
    item_size = item_elems * 2

    pf = _rand((pool, num_layers, item_elems), dtype)
    idx = torch.arange(pool, device="xpu", dtype=torch.int64)

    # Pull every layer out of the pf pool into a layer-first staging buffer...
    lf = torch.zeros(num_layers, pool, item_elems, device="xpu", dtype=dtype)
    for layer in range(num_layers):
        transfer_kv_mamba_pf_lf(
            pf, lf[layer], idx, idx, layer, item_size, item_size * num_layers
        )

    # ...then push all layers back into a fresh pf pool in one call.
    out = torch.zeros_like(pf)
    transfer_kv_mamba_lf_pf(
        lf, out, idx, idx, item_size, item_size * num_layers, num_layers
    )
    torch.xpu.synchronize()

    torch.testing.assert_close(out, pf, rtol=0, atol=0)


if __name__ == "__main__":
    pytest.main([__file__])
