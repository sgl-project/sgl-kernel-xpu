import os
import time

if os.environ.get("BATCHES", "8") != "8":
    raise SystemExit("BATCHES must be 8; rejected before GPU work")

import pytest
import sgl_kernel  # noqa: F401
import torch


@pytest.fixture(autouse=True)
def cooldown_xe35():
    if torch.xpu.is_available() and sgl_kernel.is_xe3_arch():
        torch.xpu.synchronize()
        time.sleep(2)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "experts,alignment", [(1, 1), (2, 3), (128, 4), (256, 4), (512, 16), (1024, 64)]
)
@pytest.mark.parametrize(
    "tokens,topk",
    [
        (0, 8),
        (1, 8),
        (7, 8),
        (8, 8),
        (9, 8),
        (16, 8),
        (32, 8),
        (1, 1),
        (3, 7),
        (1, 256),
    ],
)
@pytest.mark.parametrize("routing", ["low", "high", "colliding"])
def test_mxfp4_small_permutation_xe35(
    index_dtype, experts, alignment, tokens, topk, routing
):
    if not torch.xpu.is_available() or not sgl_kernel.is_xe3_arch():
        pytest.skip("Xe35 XPU is required")

    ids_cpu = (
        torch.arange(tokens * topk, dtype=torch.int64).reshape(tokens, topk) * 7
    ) % min(23, experts)
    if routing == "high":
        ids_cpu = experts - 1 - ids_cpu
    elif routing == "colliding":
        ids_cpu.fill_(experts - 1)
    ids = ids_cpu.to(device="xpu", dtype=index_dtype)
    pairs = tokens * topk
    upper_rows = pairs + (alignment - 1) * pairs
    counts = torch.empty(experts, device="xpu", dtype=torch.int32)
    perm_src = torch.empty(upper_rows, device="xpu", dtype=torch.int32)
    expanded_to_permuted = torch.empty(pairs, device="xpu", dtype=torch.int32)
    scale_map = torch.empty(upper_rows, 2, device="xpu", dtype=torch.int32)

    sgl_kernel.mxfp4_small_permutation_xe35(
        ids, counts, perm_src, expanded_to_permuted, scale_map, alignment
    )

    flat_ids = ids_cpu.flatten().tolist()
    raw_counts = [flat_ids.count(expert) for expert in range(experts)]
    expected_counts = [
        (count + alignment - 1) // alignment * alignment for count in raw_counts
    ]
    expected_perm_src = [0] * upper_rows
    expected_expanded_to_permuted = [0] * pairs
    expected_scale_map = [[row, 0] for row in range(upper_rows)]
    start = 0
    for expert, padded in enumerate(expected_counts):
        indices = [index for index, value in enumerate(flat_ids) if value == expert]
        for rank, index in enumerate(indices):
            row = start + rank
            expected_perm_src[row] = index // topk
            expected_expanded_to_permuted[index] = row
            expected_scale_map[row] = [start, padded]
        for row in range(start + len(indices), start + padded):
            expected_scale_map[row] = [start, -padded]
        start += padded

    assert counts.cpu().tolist() == expected_counts
    assert perm_src.cpu().tolist() == expected_perm_src
    assert expanded_to_permuted.cpu().tolist() == expected_expanded_to_permuted
    assert scale_map.cpu().tolist() == expected_scale_map


@pytest.mark.parametrize(
    "tokens,experts,alignment,capacity,dtype",
    [
        (33, 256, 4, 1056, torch.int32),
        (8, 3, 4, 256, torch.int32),
        (8, 256, 0, 256, torch.int32),
        (8, 256, 65, 4160, torch.int32),
        (8, 256, 4, 255, torch.int32),
        (8, 256, 4, 256, torch.float32),
    ],
)
def test_mxfp4_small_permutation_invalid_dimensions(
    tokens, experts, alignment, capacity, dtype
):
    if not torch.xpu.is_available() or not sgl_kernel.is_xe3_arch():
        pytest.skip("Xe35 XPU is required")
    ids = torch.zeros(tokens, 8, device="xpu", dtype=dtype)
    counts = torch.empty(experts, device="xpu", dtype=torch.int32)
    perm_src = torch.empty(capacity, device="xpu", dtype=torch.int32)
    expanded = torch.empty(tokens * 8, device="xpu", dtype=torch.int32)
    scale_map = torch.empty(capacity, 2, device="xpu", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="topk_ids|invalid small permutation"):
        sgl_kernel.mxfp4_small_permutation_xe35(
            ids, counts, perm_src, expanded, scale_map, alignment
        )
