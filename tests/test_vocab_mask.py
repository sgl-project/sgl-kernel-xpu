import pytest
import sgl_kernel
import torch

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available() or not sgl_kernel.is_xe3_arch(),
    reason="Xe35 kernels require compute capability 3.5",
)


@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("vocab,hidden", [(128, 33), (202048, 4096), (50512, 8192)])
def test_vocab_parallel_embedding_independent_exact(vocab, hidden, dtype, id_dtype):
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    ids_cpu = torch.tensor(
        [-1, 0, 1, vocab // 2, vocab - 2, vocab - 1, vocab, vocab + 1], dtype=id_dtype
    )
    columns_cpu = (torch.arange(hidden, dtype=torch.float64) % 16) * 0.0625
    rows = torch.arange(vocab, device="xpu", dtype=torch.float32).view(-1, 1) * 0.125
    weight = (rows + columns_cpu.float().to("xpu")).to(dtype)
    output = torch.empty(8, hidden, device="xpu", dtype=dtype)
    torch.ops.sgl_kernel.vocab_parallel_embedding_xe35(
        ids_cpu.to("xpu"), weight, output, 0, vocab, 0, vocab, vocab
    )
    invalid = (ids_cpu < 0) | (ids_cpu >= vocab)
    expected = (
        ids_cpu.clamp(0, vocab - 1).double().view(-1, 1) * 0.125 + columns_cpu
    ).to(dtype)
    expected[invalid] = 0
    torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize("hidden", [6144, 8192])
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_vocab_parallel_embedding_bandwidth_layout(hidden, id_dtype, dtype):
    vocab = 8193
    ids_cpu = (torch.arange(257, dtype=id_dtype) * 31) % vocab
    ids_cpu[:4] = torch.tensor([-1, 0, vocab - 1, vocab], dtype=id_dtype)
    columns_cpu = (torch.arange(hidden) % 32).float() * 0.0625
    rows = (torch.arange(vocab, device="xpu") % 128).float().view(-1, 1) * 0.125
    weight = (rows + columns_cpu.to("xpu")).to(dtype)
    storage = torch.full((258, hidden), 17, dtype=dtype, device="xpu")
    output = storage[:257]
    torch.ops.sgl_kernel.vocab_parallel_embedding_xe35(
        ids_cpu.to("xpu"), weight, output, 0, vocab, 0, vocab, vocab
    )
    expected = (
        (ids_cpu.clamp(0, vocab - 1) % 128).float().view(-1, 1) * 0.125
        + columns_cpu
    ).to(dtype)
    expected[(ids_cpu < 0) | (ids_cpu >= vocab)] = 0
    torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)
    torch.testing.assert_close(storage[-1], torch.full_like(storage[-1], 17), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_vocab_mask_boundaries(dtype):
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    org_start, org_end, padding = 100, 151, 13
    added_start, added_end = 200, 207
    values = [
        -1,
        0,
        org_start - 1,
        org_start,
        org_end - 1,
        org_end,
        added_start - 1,
        added_start,
        added_end - 1,
        added_end,
        100000,
    ]
    input_ids = torch.tensor(values * 3, dtype=dtype, device="xpu")
    mapped, invalid = torch.ops.sgl_kernel.vocab_mask_xe35(
        input_ids, org_start, org_end, padding, added_start, added_end
    )
    org_valid = (input_ids >= org_start) & (input_ids < org_end)
    added_valid = (input_ids >= added_start) & (input_ids < added_end)
    offset = (
        org_start * org_valid
        + (added_start - (org_end - org_start) - padding) * added_valid
    )
    valid = org_valid | added_valid
    torch.testing.assert_close(mapped, (input_ids - offset) * valid)
    torch.testing.assert_close(invalid, ~valid)


@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_vocab_mask_tp2_shards(tp_rank, dtype):
    """Check both TP2 shard index mappings on one XPU without multi-GPU collectives."""
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")

    device = torch.device("xpu:0")
    ids = torch.tensor(
        [0, 511, 512, 1009, 1010, 1025, 1026, -1],
        device=device,
        dtype=dtype,
    )
    if tp_rank == 0:
        # 1010 original tokens are padded to 1024, then 16 added tokens.
        args = (0, 512, 0, 1010, 1026)
        mapped = [0, 511, 0, 0, 512, 527, 0, 0]
        invalid = [False, False, True, True, False, False, True, True]
    else:
        args = (512, 1010, 14, 1026, 1026)
        mapped = [0, 0, 0, 497, 0, 0, 0, 0]
        invalid = [True, True, False, False, True, True, True, True]

    actual_mapped, actual_invalid = torch.ops.sgl_kernel.vocab_mask_xe35(ids, *args)
    torch.testing.assert_close(
        actual_mapped, torch.tensor(mapped, device=device, dtype=torch.int64)
    )
    torch.testing.assert_close(
        actual_invalid, torch.tensor(invalid, device=device, dtype=torch.bool)
    )
    assert actual_mapped.device == device
    assert actual_invalid.device == device


@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_vocab_mask_tp2_framework_dispatch(tp_rank, dtype):
    """The sglang layer helper must route XPU inputs to the Xe35 kernel."""
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    vpe = pytest.importorskip("sglang.srt.layers.vocab_parallel_embedding")

    device = torch.device("xpu:0")
    ids = torch.tensor(
        [0, 511, 512, 1009, 1010, 1025, 1026, -1],
        device=device,
        dtype=dtype,
    )
    if tp_rank == 0:
        args = (0, 512, 0, 1010, 1026)
    else:
        args = (512, 1010, 14, 1026, 1026)

    actual_mapped, actual_invalid = torch.ops.sgl_kernel.vocab_mask_xe35(ids, *args)
    mapped_from_layer, invalid_from_layer = vpe.get_masked_input_and_mask(ids, *args)
    torch.testing.assert_close(mapped_from_layer, actual_mapped)
    torch.testing.assert_close(invalid_from_layer, actual_invalid)


def test_vocab_mask_subgroup_size_boundary():
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    ids = torch.tensor(
        [0, 511, 512, 1009, 1010, 1025, 1026, -1] * 4, device="xpu", dtype=torch.int64
    )
    mapped, invalid = torch.ops.sgl_kernel.vocab_mask_xe35(ids, 0, 512, 0, 1010, 1026)
    expected_mapped = torch.tensor(
        [0, 511, 0, 0, 512, 527, 0, 0] * 4, device="xpu", dtype=torch.int64
    )
    expected_invalid = torch.tensor(
        [False, False, True, True, False, False, True, True] * 4, device="xpu"
    )
    torch.testing.assert_close(mapped, expected_mapped)
    torch.testing.assert_close(invalid, expected_invalid)


@pytest.mark.parametrize("hidden", [8, 128, 2040, 2048, 2056, 4080, 4096, 4097, 4112])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_vocab_embedding_partial_workgroup(hidden, dtype, id_dtype):
    ids = torch.tensor([0, 1, -1, 3], device="xpu", dtype=id_dtype)
    weight = torch.randn(3, hidden, device="xpu", dtype=dtype)
    storage = torch.full((4 * hidden + 4096,), 17, device="xpu", dtype=dtype)
    output = storage[: 4 * hidden].view(4, hidden)
    torch.ops.sgl_kernel.vocab_parallel_embedding_xe35(
        ids, weight, output, 0, 3, 0, 3, 3
    )
    expected = torch.stack(
        [weight[0], weight[1], torch.zeros_like(weight[0]), torch.zeros_like(weight[0])]
    )
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    torch.testing.assert_close(
        storage[4 * hidden :],
        torch.full_like(storage[4 * hidden :], 17),
        atol=0,
        rtol=0,
    )
