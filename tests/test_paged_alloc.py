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


@pytest.mark.parametrize("last_dtype", [torch.int32, torch.int64])
def test_paged_alloc_extend_crosses_page_boundaries(last_dtype):
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    device = "xpu"
    pre = torch.tensor([2, 4, 7, 1], device=device, dtype=torch.int64)
    seq = torch.tensor([5, 6, 10, 3], device=device, dtype=torch.int64)
    last = torch.tensor([101, -1, 202, 400], device=device, dtype=last_dtype)
    free_pages = torch.tensor([10, 20, 30], device=device, dtype=torch.int64)
    out = torch.empty(10, device=device, dtype=torch.int64)
    torch.ops.sgl_kernel.paged_alloc_extend_xe35(pre, seq, last, free_pages, out, 4)
    expected = torch.tensor(
        [102, 103, 40, 80, 81, 203, 120, 121, 401, 402],
        device=device,
        dtype=torch.int64,
    )
    torch.xpu.synchronize()
    torch.testing.assert_close(out.cpu(), expected.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("last_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("page_size", [4, 7, 32, 64, 128, 256])
def test_paged_alloc_decode_reuses_or_opens_page(last_dtype, page_size):
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    device = "xpu"
    seq = torch.tensor(
        [page_size, page_size + 1, 2 * page_size, 2 * page_size + 1],
        device=device,
        dtype=torch.int64,
    )
    last = torch.tensor([23, 31, 39, 47], device=device, dtype=last_dtype)
    free_pages = torch.tensor([11, 12], device=device, dtype=torch.int64)
    out = torch.empty_like(seq)
    torch.ops.sgl_kernel.paged_alloc_decode_xe35(seq, last, free_pages, out, page_size)
    expected = torch.tensor(
        [24, 11 * page_size, 40, 12 * page_size], device=device, dtype=torch.int64
    )
    torch.xpu.synchronize()
    torch.testing.assert_close(out.cpu(), expected.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("last_dtype", [torch.int32, torch.int64])
def test_paged_alloc_extend_page64_partial_and_new_pages(last_dtype):
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    pre = torch.tensor([62, 64, 127], device="xpu", dtype=torch.int64)
    seq = torch.tensor([66, 130, 130], device="xpu", dtype=torch.int64)
    last = torch.tensor([300, -1, 511], device="xpu", dtype=last_dtype)
    free_pages = torch.tensor([10, 20, 30, 40], device="xpu", dtype=torch.int64)
    out = torch.empty(73, device="xpu", dtype=torch.int64)
    torch.ops.sgl_kernel.paged_alloc_extend_xe35(pre, seq, last, free_pages, out, 64)
    expected = torch.tensor(
        [301, 302, 640, 641] + list(range(1280, 1344)) + [1920, 1921, 512, 2560, 2561],
        device="xpu",
        dtype=torch.int64,
    )
    torch.xpu.synchronize()
    torch.testing.assert_close(out.cpu(), expected.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("page_size", [7, 32, 64, 128])
@pytest.mark.parametrize("last_dtype", [torch.int32, torch.int64])
def test_paged_alloc_extend_and_write_matches_split_with_prefix(page_size, last_dtype):
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    prefix_lens = torch.tensor(
        [2, page_size, page_size - 1], device="xpu", dtype=torch.int64
    )
    seq_lens = torch.tensor(
        [page_size + 2, page_size + 3, page_size + 2], device="xpu", dtype=torch.int64
    )
    extend_lens = seq_lens - prefix_lens
    last = torch.tensor([101, 200, 301], device="xpu", dtype=last_dtype)
    free_pages = torch.arange(10, 16, device="xpu", dtype=torch.int64)
    total_extend = int(extend_lens.sum().item())
    split_out = torch.empty(total_extend, device="xpu", dtype=torch.int64)
    fused_out = torch.empty_like(split_out)
    pool_width = int(seq_lens.max().item())
    split_pool = torch.full((3, pool_width), -1, device="xpu", dtype=torch.int32)
    fused_pool = split_pool.clone()
    req_indices = torch.tensor([2, 0, 1], device="xpu", dtype=torch.int64)
    prefixes = [
        torch.arange(int(length), device="xpu", dtype=torch.int64) + 1000 * (index + 1)
        for index, length in enumerate(prefix_lens.tolist())
    ]
    pointers = torch.tensor(
        [tensor.data_ptr() for tensor in prefixes], device="xpu", dtype=torch.uint64
    )
    torch.ops.sgl_kernel.paged_alloc_extend_xe35(
        prefix_lens, seq_lens, last, free_pages, split_out, page_size
    )
    torch.ops.sgl_kernel.write_req_to_token_pool_xe35(
        split_pool, req_indices, pointers, prefix_lens, seq_lens, extend_lens, split_out
    )
    torch.ops.sgl_kernel.paged_alloc_extend_and_write_xe35(
        prefix_lens,
        seq_lens,
        last,
        free_pages,
        fused_out,
        fused_pool,
        req_indices,
        pointers,
        page_size,
    )
    torch.xpu.synchronize()
    torch.testing.assert_close(fused_out.cpu(), split_out.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(fused_pool.cpu(), split_pool.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("last_dtype", [torch.int32, torch.int64])
def test_paged_alloc_prefill_and_decode_exact(last_dtype):
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    batch, input_length, output_length, page_size = 8, 4096, 1024, 64
    prefix_lens = torch.zeros(batch, device="xpu", dtype=torch.int64)
    seq_lens = torch.full_like(prefix_lens, input_length)
    last = torch.full((batch,), -1, device="xpu", dtype=last_dtype)
    free_pages = torch.arange(1, batch * input_length // page_size + 1, device="xpu")
    locations = torch.empty(batch * input_length, device="xpu", dtype=torch.int64)
    pool = torch.full(
        (batch, input_length + output_length), -1, device="xpu", dtype=torch.int32
    )
    req_indices = torch.arange(batch - 1, -1, -1, device="xpu", dtype=torch.int64)
    empty_prefix = torch.zeros(1, device="xpu", dtype=torch.int64)
    pointers = torch.full(
        (batch,), empty_prefix.data_ptr(), device="xpu", dtype=torch.uint64
    )
    torch.ops.sgl_kernel.paged_alloc_extend_and_write_xe35(
        prefix_lens,
        seq_lens,
        last,
        free_pages,
        locations,
        pool,
        req_indices,
        pointers,
        page_size,
    )
    expected_locations = torch.arange(
        page_size, page_size + batch * input_length, dtype=torch.int64
    )
    torch.testing.assert_close(locations.cpu(), expected_locations, rtol=0, atol=0)
    expected_pool = torch.full(
        (batch, input_length + output_length), -1, dtype=torch.int32
    )
    expected_pool[:, :input_length] = (
        expected_locations.reshape(batch, input_length).flip(0).int()
    )
    torch.testing.assert_close(pool.cpu(), expected_pool, rtol=0, atol=0)

    last = locations.reshape(batch, input_length)[:, -1].to(last_dtype).contiguous()
    decode_pages = torch.arange(1000, 1000 + batch, device="xpu", dtype=torch.int64)
    decode_locations = torch.empty(batch, device="xpu", dtype=torch.int64)
    for step in range(output_length):
        first_page = 1000 + step // page_size * batch
        if step % page_size == 0:
            decode_pages = torch.arange(
                first_page, first_page + batch, device="xpu", dtype=torch.int64
            )
        seq_lens.fill_(input_length + step + 1)
        torch.ops.sgl_kernel.paged_alloc_decode_xe35(
            seq_lens, last, decode_pages, decode_locations, page_size
        )
        expected_decode = (
            torch.arange(first_page, first_page + batch, dtype=torch.int64) * page_size
            + step % page_size
        )
        torch.testing.assert_close(
            decode_locations.cpu(), expected_decode, rtol=0, atol=0
        )
        last = decode_locations.to(last_dtype)


@pytest.mark.parametrize("page_size", [1, 7, 32, 64, 128, 256])
@pytest.mark.parametrize("last_dtype", [torch.int32, torch.int64])
def test_paged_alloc_irregular_extend_matches_cpu(page_size, last_dtype):
    prefixes = [0, 2, page_size, 2 * page_size - 1, 3 * page_size + 1]
    extensions = [0, 1, page_size - 1, page_size, page_size + 1]
    sequences = [prefix + extension for prefix, extension in zip(prefixes, extensions)]
    last_values = [10000 + index * 1000 for index in range(len(prefixes))]
    free_values = list(range(100, 120))
    expected_values = []
    page_start = 0
    for prefix, sequence, last_value in zip(prefixes, sequences, last_values):
        old_pages = (prefix + page_size - 1) // page_size
        for position in range(prefix, sequence):
            if position < old_pages * page_size:
                expected_values.append(last_value + 1 + position - prefix)
            else:
                page = free_values[page_start + position // page_size - old_pages]
                expected_values.append(page * page_size + position % page_size)
        page_start += (sequence + page_size - 1) // page_size - old_pages

    prefix_lens = torch.tensor(prefixes, device="xpu", dtype=torch.int64)
    seq_lens = torch.tensor(sequences, device="xpu", dtype=torch.int64)
    last = torch.tensor(last_values, device="xpu", dtype=last_dtype)
    free_pages = torch.tensor(free_values, device="xpu", dtype=torch.int64)
    output = torch.empty(sum(extensions), device="xpu", dtype=torch.int64)
    torch.ops.sgl_kernel.paged_alloc_extend_xe35(
        prefix_lens, seq_lens, last, free_pages, output, page_size
    )
    expected = torch.tensor(expected_values, dtype=torch.int64)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)

    width = max(sequences) + 2
    pool = torch.full((len(prefixes), width), -1, device="xpu", dtype=torch.int32)
    req_indices = torch.arange(
        len(prefixes) - 1, -1, -1, device="xpu", dtype=torch.int64
    )
    prefix_tensors = [
        torch.arange(prefix, device="xpu", dtype=torch.int64) + 1000 * index
        for index, prefix in enumerate(prefixes)
    ]
    pointers = torch.tensor(
        [prefix.data_ptr() for prefix in prefix_tensors],
        device="xpu",
        dtype=torch.uint64,
    )
    output.fill_(-1)
    torch.ops.sgl_kernel.paged_alloc_extend_and_write_xe35(
        prefix_lens,
        seq_lens,
        last,
        free_pages,
        output,
        pool,
        req_indices,
        pointers,
        page_size,
    )
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    expected_pool = torch.full(pool.shape, -1, dtype=torch.int32)
    output_start = 0
    for index, (prefix, sequence, extension) in enumerate(
        zip(prefixes, sequences, extensions)
    ):
        row = len(prefixes) - 1 - index
        expected_pool[row, :prefix] = torch.arange(prefix) + 1000 * index
        expected_pool[row, prefix:sequence] = expected[
            output_start : output_start + extension
        ].int()
        output_start += extension
    torch.testing.assert_close(pool.cpu(), expected_pool, rtol=0, atol=0)


@pytest.mark.parametrize("last_dtype", [torch.int32, torch.int64])
def test_paged_alloc_empty_batch(last_dtype):
    metadata = torch.empty(0, device="xpu", dtype=torch.int64)
    last = torch.empty(0, device="xpu", dtype=last_dtype)
    pointers = torch.empty(0, device="xpu", dtype=torch.uint64)
    output = torch.full((1,), -1, device="xpu", dtype=torch.int64)
    pool = torch.full((1, 1), -1, device="xpu", dtype=torch.int32)
    torch.ops.sgl_kernel.paged_alloc_extend_xe35(
        metadata, metadata, last, metadata, output[:0], 64
    )
    torch.ops.sgl_kernel.paged_alloc_decode_xe35(
        metadata, last, metadata, output[:0], 64
    )
    torch.ops.sgl_kernel.paged_alloc_extend_and_write_xe35(
        metadata, metadata, last, metadata, output[:0], pool, metadata, pointers, 64
    )
    torch.ops.sgl_kernel.write_req_to_token_pool_xe35(
        pool, metadata, pointers, metadata, metadata, metadata, output[:0]
    )
    torch.testing.assert_close(output.cpu(), torch.tensor([-1]), rtol=0, atol=0)
    torch.testing.assert_close(
        pool.cpu(), torch.tensor([[-1]], dtype=torch.int32), rtol=0, atol=0
    )
