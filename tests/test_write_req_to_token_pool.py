import time

import pytest
import sgl_kernel
import torch

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available() or not sgl_kernel.is_xe3_arch(),
    reason="Xe35 kernels require compute capability 3.5",
)


@pytest.fixture(autouse=True)
def cool_between_shapes():
    yield
    torch.xpu.synchronize()
    time.sleep(2)


def test_write_req_to_token_pool_prefix_and_extension():
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    device = "xpu"
    pool = torch.full((5, 8), -1, device=device, dtype=torch.int32)
    req_indices = torch.tensor([3, 1, 4], device=device, dtype=torch.int64)
    prefixes = [
        torch.tensor([20, 21], device=device, dtype=torch.int64),
        torch.empty(0, device=device, dtype=torch.int64),
        torch.tensor([30, 31, 32], device=device, dtype=torch.int64),
    ]
    pointers = torch.tensor(
        [p.data_ptr() for p in prefixes], device=device, dtype=torch.uint64
    )
    prefix_lens = torch.tensor([2, 0, 3], device=device, dtype=torch.int64)
    seq_lens = torch.tensor([4, 2, 4], device=device, dtype=torch.int64)
    extend_lens = torch.tensor([2, 2, 1], device=device, dtype=torch.int64)
    out_cache_loc = torch.tensor([10, 11, 12, 13, 14], device=device, dtype=torch.int64)
    torch.ops.sgl_kernel.write_req_to_token_pool_xe35(
        pool,
        req_indices,
        pointers,
        prefix_lens,
        seq_lens,
        extend_lens,
        out_cache_loc,
    )
    torch.xpu.synchronize()
    pool_cpu = pool.cpu()
    torch.testing.assert_close(
        pool_cpu[3],
        torch.tensor([20, 21, 10, 11, -1, -1, -1, -1], dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        pool_cpu[1],
        torch.tensor([12, 13, -1, -1, -1, -1, -1, -1], dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        pool_cpu[4],
        torch.tensor([30, 31, 32, 14, -1, -1, -1, -1], dtype=torch.int32),
        rtol=0,
        atol=0,
    )


def test_write_req_to_token_pool_multigroup_irregular_prefix():
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    pool = torch.full((3, 5200), -1, device="xpu", dtype=torch.int32)
    prefix = torch.arange(3001, device="xpu", dtype=torch.int64) + 20
    empty_prefix = torch.empty(0, device="xpu", dtype=torch.int64)
    pointers = torch.tensor(
        [prefix.data_ptr(), empty_prefix.data_ptr()], device="xpu", dtype=torch.uint64
    )
    req_indices = torch.tensor([2, 0], device="xpu", dtype=torch.int64)
    prefix_lens = torch.tensor([3001, 0], device="xpu", dtype=torch.int64)
    seq_lens = torch.tensor([5050, 1], device="xpu", dtype=torch.int64)
    extend_lens = torch.tensor([2049, 1], device="xpu", dtype=torch.int64)
    loc = torch.arange(2050, device="xpu", dtype=torch.int64) + 10000
    torch.ops.sgl_kernel.write_req_to_token_pool_xe35(
        pool, req_indices, pointers, prefix_lens, seq_lens, extend_lens, loc
    )
    torch.xpu.synchronize()
    pool_cpu = pool.cpu()
    expected = torch.cat((prefix.cpu(), loc[:2049].cpu())).int()
    torch.testing.assert_close(pool_cpu[2, :5050], expected, rtol=0, atol=0)
    torch.testing.assert_close(
        pool_cpu[2, 5050:], torch.full((150,), -1, dtype=torch.int32), rtol=0, atol=0
    )
    torch.testing.assert_close(pool_cpu[0, 0], loc[-1].cpu().int(), rtol=0, atol=0)
    torch.testing.assert_close(
        pool_cpu[0, 1:], torch.full((5199,), -1, dtype=torch.int32), rtol=0, atol=0
    )
