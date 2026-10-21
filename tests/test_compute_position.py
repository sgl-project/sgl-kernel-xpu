import pytest
import sgl_kernel
import torch

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available() or not sgl_kernel.is_xe3_arch(),
    reason="Xe35 kernels require compute capability 3.5",
)


@pytest.mark.parametrize("length_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("with_prefix", [False, True])
def test_compute_position_production_exact(length_dtype, with_prefix):
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    lengths = torch.full((8,), 4096, dtype=length_dtype, device="xpu")
    prefix = torch.zeros(8 if with_prefix else 0, dtype=length_dtype, device="xpu")
    positions = torch.empty(8 * 4096, dtype=torch.int64, device="xpu")
    starts = torch.empty(8, dtype=torch.int32, device="xpu")
    torch.ops.sgl_kernel.compute_position_xe35(positions, starts, prefix, lengths)
    torch.testing.assert_close(
        positions.cpu(), torch.arange(4096).repeat(8), atol=0, rtol=0
    )
    torch.testing.assert_close(
        starts.cpu(), torch.arange(8, dtype=torch.int32) * 4096, atol=0, rtol=0
    )


def test_compute_position_with_prefix():
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    seq = torch.tensor([2, 0, 3], device="xpu", dtype=torch.int32)
    prefix = torch.tensor([4, 2, 7], device="xpu", dtype=torch.int32)
    positions = torch.empty(5, device="xpu", dtype=torch.int64)
    starts = torch.empty(3, device="xpu", dtype=torch.int32)
    torch.ops.sgl_kernel.compute_position_xe35(positions, starts, prefix, seq)
    torch.testing.assert_close(
        positions, torch.tensor([4, 5, 7, 8, 9], device="xpu", dtype=torch.int64)
    )
    torch.testing.assert_close(
        starts, torch.tensor([0, 2, 2], device="xpu", dtype=torch.int32)
    )


def test_compute_position_multiple_workgroups_irregular():
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    lengths = torch.tensor([1, 4097, 33], device="xpu", dtype=torch.int64)
    prefix = torch.tensor([2, 4, 6], device="xpu", dtype=torch.int64)
    positions = torch.empty(4131, device="xpu", dtype=torch.int64)
    starts = torch.empty(3, device="xpu", dtype=torch.int32)
    torch.ops.sgl_kernel.compute_position_xe35(positions, starts, prefix, lengths)
    expected = torch.cat(
        [
            torch.arange(int(length), device="xpu") + int(base)
            for length, base in zip(lengths.cpu(), prefix.cpu())
        ]
    )
    torch.testing.assert_close(positions, expected)
    torch.testing.assert_close(
        starts, torch.tensor([0, 1, 4098], device="xpu", dtype=torch.int32)
    )


def test_compute_position_multiple_workgroups_no_prefix():
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    lengths = torch.tensor([2049, 1], device="xpu", dtype=torch.int32)
    prefix = torch.empty(0, device="xpu", dtype=torch.int32)
    positions = torch.empty(2050, device="xpu", dtype=torch.int64)
    starts = torch.empty(2, device="xpu", dtype=torch.int32)
    torch.ops.sgl_kernel.compute_position_xe35(positions, starts, prefix, lengths)
    torch.testing.assert_close(
        positions,
        torch.cat([torch.arange(2049, device="xpu"), torch.arange(1, device="xpu")]),
    )
    torch.testing.assert_close(
        starts, torch.tensor([0, 2049], device="xpu", dtype=torch.int32)
    )


@pytest.mark.parametrize("long_request", [4096, 8193])
def test_compute_position_with_host_max_hint_skew_b8(long_request):
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    lengths = torch.tensor([long_request] + [0] * 7, device="xpu", dtype=torch.int32)
    prefix = torch.arange(8, device="xpu", dtype=torch.int32) * 3
    positions = torch.empty(long_request, device="xpu", dtype=torch.int64)
    starts = torch.empty(8, device="xpu", dtype=torch.int32)
    torch.ops.sgl_kernel.compute_position_with_hint_xe35(
        positions, starts, prefix, lengths, long_request
    )
    torch.testing.assert_close(positions, torch.arange(long_request, device="xpu"))
    torch.testing.assert_close(
        starts,
        torch.tensor(
            [0] + [long_request] * 7,
            device="xpu",
            dtype=torch.int32,
        ),
    )
    # A deliberately low hint must remain correct; it only affects occupancy.
    torch.ops.sgl_kernel.compute_position_with_hint_xe35(
        positions, starts, prefix, lengths, 1
    )
    torch.testing.assert_close(positions, torch.arange(long_request, device="xpu"))
    with pytest.raises(RuntimeError, match="nonnegative"):
        torch.ops.sgl_kernel.compute_position_with_hint_xe35(
            positions, starts, prefix, lengths, -1
        )


@pytest.mark.parametrize("prefix_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("length_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("lengths", [[], [0] * 8, [1, 0, 257, 511, 513, 0, 7, 1025]])
def test_compute_position_empty_and_mixed_lengths(prefix_dtype, length_dtype, lengths):
    sequence = torch.tensor(lengths, dtype=length_dtype, device="xpu")
    prefix = torch.arange(len(lengths), dtype=prefix_dtype, device="xpu") * 3
    positions = torch.empty(sum(lengths), dtype=torch.int64, device="xpu")
    starts = torch.empty(len(lengths), dtype=torch.int32, device="xpu")
    torch.ops.sgl_kernel.compute_position_xe35(positions, starts, prefix, sequence)
    expected = torch.tensor(
        [
            base * 3 + offset
            for base, size in enumerate(lengths)
            for offset in range(size)
        ],
        dtype=torch.int64,
    )
    expected_starts = torch.tensor(
        [sum(lengths[:request]) for request in range(len(lengths))], dtype=torch.int32
    )
    torch.testing.assert_close(positions.cpu(), expected, atol=0, rtol=0)
    torch.testing.assert_close(starts.cpu(), expected_starts, atol=0, rtol=0)


@pytest.mark.parametrize("length_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("prefix_position", [4096, 4097, 5119, 5120])
def test_compute_position_production_decode_exact(length_dtype, prefix_position):
    lengths = torch.ones(8, dtype=length_dtype, device="xpu")
    prefix = torch.full((8,), prefix_position, dtype=length_dtype, device="xpu")
    positions = torch.empty(8, dtype=torch.int64, device="xpu")
    starts = torch.empty(8, dtype=torch.int32, device="xpu")
    torch.ops.sgl_kernel.compute_position_xe35(positions, starts, prefix, lengths)
    torch.testing.assert_close(
        positions.cpu(),
        torch.full((8,), prefix_position, dtype=torch.int64),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        starts.cpu(), torch.arange(8, dtype=torch.int32), atol=0, rtol=0
    )
