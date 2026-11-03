import itertools
import sys

import pytest
import torch
from sgl_kernel import (
    apply_shuffle_mul_sum,
    prepare_moe_input,
    scatter_tokens_to_experts,
)
from sgl_kernel.moe import _should_use_small_moe_prepare


@pytest.mark.parametrize(
    "num_tokens,top_k,hidden_dim,num_experts,expected",
    [
        (1, 0, 2048, 64, False),
        (10, 4, 2048, 64, True),
        (1, 16, 2048, 64, True),
        (1, 17, 2048, 64, False),
        (11, 4, 2048, 64, True),
        (16, 4, 1024, 64, True),
        (17, 4, 1024, 128, False),
        (8, 8, 1024, 32, True),
        (8, 8, 2560, 32, True),
        (8, 8, 2561, 32, False),
    ],
)
def test_should_use_small_moe_prepare(
    num_tokens, top_k, hidden_dim, num_experts, expected
):
    assert (
        _should_use_small_moe_prepare(num_tokens, top_k, hidden_dim, num_experts)
        is expected
    )


@pytest.mark.parametrize(
    "num_tokens,top_k,hidden_dim,index_dtype,routing",
    [
        (1, 16, 15, torch.int32, "random"),
        (1, 8, 16, torch.int64, "all_same"),
        (3, 3, 33, torch.int64, "random"),
        (5, 7, 16, torch.int32, "skewed"),
        (8, 8, 33, torch.int32, "all_same"),
    ],
)
def test_prepare_moe_input_small(num_tokens, top_k, hidden_dim, index_dtype, routing):
    num_experts = 32
    torch.manual_seed(41)
    input_tensor = torch.randn(
        num_tokens, hidden_dim, dtype=torch.bfloat16, device="xpu"
    )
    if routing == "random":
        topk_ids = torch.stack(
            [torch.randperm(num_experts)[:top_k] for _ in range(num_tokens)]
        )
    elif routing == "skewed":
        topk_ids = torch.arange(top_k).repeat(num_tokens, 1)
    else:
        topk_ids = torch.zeros((num_tokens, top_k), dtype=torch.int64)
    topk_ids = topk_ids.to(device="xpu", dtype=index_dtype)
    expert_counts = torch.full((num_experts,), 123, dtype=torch.int32, device="xpu")
    output_permutation = torch.empty(
        num_tokens * top_k, dtype=torch.int32, device="xpu"
    )
    output = torch.empty(
        num_tokens * top_k, hidden_dim, dtype=torch.bfloat16, device="xpu"
    )

    torch.ops.sgl_kernel.prepare_moe_input_small.default(
        input_tensor, topk_ids, expert_counts, output_permutation, output
    )

    flat_ids = topk_ids.cpu().flatten()
    sorted_routes = torch.argsort(flat_ids, stable=True)
    expected_permutation = torch.empty_like(sorted_routes, dtype=torch.int32)
    expected_permutation[sorted_routes] = torch.arange(
        sorted_routes.numel(), dtype=torch.int32
    )
    expected_output = input_tensor.cpu()[sorted_routes // top_k]
    expected_counts = torch.bincount(flat_ids, minlength=num_experts).to(torch.int32)

    torch.testing.assert_close(expert_counts.cpu(), expected_counts)
    torch.testing.assert_close(output_permutation.cpu(), expected_permutation)
    torch.testing.assert_close(output.cpu(), expected_output)


@pytest.mark.parametrize(
    "num_tokens,top_k",
    [(1, 1), (1, 8), (1, 16), (3, 3), (8, 8), (4, 16)],
)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("routing", ["local_shard", "all_dropped"])
@pytest.mark.parametrize("hidden_dim", [32, 33])
def test_prepare_moe_input_small_drops_negative_ids(
    num_tokens, top_k, index_dtype, routing, hidden_dim
):
    # Expert-parallel callers mark experts owned by other ranks with -1.
    num_local_experts, num_global_experts = 16, 64
    torch.manual_seed(41)
    noise = torch.rand(num_tokens, num_global_experts)
    topk_ids = torch.argsort(noise, dim=1)[:, :top_k]
    if routing == "all_dropped":
        topk_ids.fill_(-1)
    else:
        topk_ids[topk_ids >= num_local_experts] = -1
    # Keep one local route so the single-route local_shard case is not all dropped.
    if routing == "local_shard" and top_k == 1:
        topk_ids[0, 0] = 3

    input_tensor = torch.randn(
        num_tokens, hidden_dim, dtype=torch.bfloat16, device="xpu"
    )
    expert_counts = torch.full(
        (num_local_experts,), 123, dtype=torch.int32, device="xpu"
    )
    output_permutation = torch.full(
        (num_tokens * top_k,), 123, dtype=torch.int32, device="xpu"
    )
    output = torch.zeros(
        num_tokens * top_k, hidden_dim, dtype=torch.bfloat16, device="xpu"
    )
    torch.ops.sgl_kernel.prepare_moe_input_small.default(
        input_tensor,
        topk_ids.to(device="xpu", dtype=index_dtype),
        expert_counts,
        output_permutation,
        output,
    )

    flat_ids = topk_ids.flatten()
    valid = flat_ids >= 0
    num_valid = int(valid.sum())
    # The kernel is stable: routes of one expert keep their original order.
    order = torch.argsort(flat_ids[valid], stable=True)
    valid_routes = torch.arange(flat_ids.numel())[valid]
    expected_permutation = torch.full((flat_ids.numel(),), -1, dtype=torch.int32)
    expected_permutation[valid_routes[order]] = torch.arange(
        num_valid, dtype=torch.int32
    )
    expected_counts = torch.bincount(flat_ids[valid], minlength=num_local_experts).to(
        torch.int32
    )
    expected_output = torch.zeros(num_tokens * top_k, hidden_dim, dtype=torch.bfloat16)
    expected_output[:num_valid] = input_tensor.cpu()[valid_routes[order] // top_k]

    torch.testing.assert_close(expert_counts.cpu(), expected_counts)
    torch.testing.assert_close(output_permutation.cpu(), expected_permutation)
    torch.testing.assert_close(output.cpu(), expected_output, rtol=0, atol=0)


@pytest.mark.parametrize(
    "num_tokens,top_k,error",
    [(1, 17, "topk must be in"), (13, 5, "routed rows must be")],
)
def test_prepare_moe_input_small_rejects_capacity_overflow(num_tokens, top_k, error):
    hidden_dim = 16
    input_tensor = torch.empty(
        (num_tokens, hidden_dim), dtype=torch.bfloat16, device="xpu"
    )
    topk_ids = torch.zeros((num_tokens, top_k), dtype=torch.int32, device="xpu")
    route_count = topk_ids.numel()

    with pytest.raises(RuntimeError, match=error):
        torch.ops.sgl_kernel.prepare_moe_input_small.default(
            input_tensor,
            topk_ids,
            torch.empty(32, dtype=torch.int32, device="xpu"),
            torch.empty(route_count, dtype=torch.int32, device="xpu"),
            torch.empty((route_count, hidden_dim), dtype=torch.bfloat16, device="xpu"),
        )


@pytest.mark.parametrize("num_tokens", [1, 2, 5, 16, 64, 128, 224, 1024])
@pytest.mark.parametrize("num_experts", [1, 4, 8, 32, 40, 64, 128, 256])
@pytest.mark.parametrize("top_k", [1, 2, 4, 6, 8])
@pytest.mark.parametrize("hidden_dims", [16, 32, 64, 128, 1024, 1536, 2048])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("use_factors", [True, False])
def test_prepare_input_moe(
    num_tokens, num_experts, top_k, hidden_dims, dtype, use_factors
):
    if num_experts < top_k:
        pytest.skip("invalid combination")
    if not use_factors:
        # Keep a single representative config for the null-factors path.
        keep_null_factors_case = (
            num_tokens == 16
            and num_experts == 8
            and top_k == 4
            and hidden_dims == 128
            and dtype == torch.float32
        )
        if not keep_null_factors_case:
            pytest.skip("reduce parameter space for null-factors coverage")
    torch.manual_seed(41)

    # Generate unique token
    def generate_unique_topk_ids(tokens, top_k, num_experts):
        # One randperm per token, batched: argsort of uniform noise per row is
        # an independent permutation of [0, num_experts) per token, so each row
        # still has no duplicate experts.
        noise = torch.rand(tokens, num_experts)
        return torch.argsort(noise, dim=1)[:, :top_k].to(torch.int32).contiguous()

    def prepare_input_moe_ref(
        topk_ids,
        expert_offsets,
        blockscale_offsets,
        problem_sizes1,
        problem_sizes2,
        input_permutation,
        output_permutation,
        num_experts,
        hidden_dim,
        top_k,
    ):
        top_k = topk_ids.shape[1]
        expert_cnt = torch.bincount(
            topk_ids.flatten().to(torch.int64), minlength=num_experts
        ).to(torch.int32)
        expert_offsets.copy_(expert_cnt)

        problem_sizes1[0::3] = expert_cnt
        problem_sizes1[1::3] = hidden_dim * 2
        problem_sizes1[2::3] = top_k
        problem_sizes2[0::3] = expert_cnt
        problem_sizes2[1::3] = top_k
        problem_sizes2[2::3] = hidden_dim

        # compute input/output permutes
        #
        # The loop this replaces walks the flattened topk_ids in order and
        # appends each route to its expert's run, where the runs are laid out
        # by ascending expert id. A stable argsort by expert id reproduces
        # exactly that assignment: destination slot i of the sorted order is
        # the i'th route of its expert, in original order.
        flat_topk = topk_ids.flatten()
        topk_length = flat_topk.numel()
        order = torch.argsort(flat_topk.to(torch.int64), stable=True)
        output_permutation[order] = torch.arange(
            topk_length, dtype=output_permutation.dtype
        )
        input_permutation.copy_((order // top_k).to(input_permutation.dtype))

    # routing that generate unique tokens
    topk_ids = generate_unique_topk_ids(num_tokens, top_k, num_experts)
    expert_offsets = torch.zeros(num_experts, dtype=torch.int32)
    my_atoimic_buffer = torch.zeros(num_experts, dtype=torch.int32)
    problem_sizes1 = torch.zeros(num_experts * 3, dtype=torch.int32)
    problem_sizes2 = torch.zeros(num_experts * 3, dtype=torch.int32)

    flat_topk = topk_ids.flatten()
    input_permutation = torch.empty_like(flat_topk)
    output_permutation = torch.empty_like(flat_topk)
    blocksclae_offset = None

    device = "xpu"
    topk_ids_xpu = topk_ids.clone().to(device)
    expert_offsets_xpu = expert_offsets.clone().to(device)
    problem_sizes1_xpu = problem_sizes1.clone().to(device)
    problem_sizes2_xpu = problem_sizes2.clone().to(device)
    input_permutation_xpu = torch.empty_like(flat_topk).to(device)
    output_permutation_xpu = torch.empty_like(flat_topk).to(device)

    # generate reference permutations on cpu
    prepare_input_moe_ref(
        topk_ids,
        expert_offsets,
        blocksclae_offset,
        problem_sizes1,
        problem_sizes2,
        input_permutation,
        output_permutation,
        num_experts,
        hidden_dims,
        top_k,
    )

    # prepare moe inputs on xpu
    prepare_moe_input(
        topk_ids_xpu,
        expert_offsets_xpu,
        problem_sizes1_xpu,
        problem_sizes2_xpu,
        input_permutation_xpu,
        output_permutation_xpu,
        num_experts,
        hidden_dims,
        top_k,
        blocksclae_offset,
    )

    # validate expert offsets
    torch.testing.assert_close(expert_offsets, expert_offsets_xpu.to("cpu"))
    input_tensor = torch.randn(num_tokens, hidden_dims, dtype=dtype)
    input_tensor_xpu = input_tensor.clone().to(device)
    output_tensor_xpu = torch.empty(
        (num_tokens * top_k, hidden_dims), dtype=dtype, device=device
    )
    scatter_tokens_to_experts(
        input_tensor_xpu, output_permutation_xpu, output_tensor_xpu
    )
    input_merge_xpu = torch.empty((num_tokens, hidden_dims), dtype=dtype, device=device)
    if use_factors:
        # Explicit per-route averaging weights.
        factors = torch.ones(
            top_k * num_tokens, dtype=torch.float32, device=device
        ).fill_(1 / top_k)
        apply_shuffle_mul_sum(
            output_tensor_xpu, input_merge_xpu, output_permutation_xpu, factors
        )
    else:
        # Null factors should default to unit weights in kernel.
        apply_shuffle_mul_sum(
            output_tensor_xpu, input_merge_xpu, output_permutation_xpu, None
        )

    expected = input_tensor if use_factors else input_tensor * top_k
    torch.testing.assert_allclose(input_merge_xpu.to("cpu"), expected)


@pytest.mark.parametrize("num_tokens", [1, 37, 300])
@pytest.mark.parametrize("routing", ["local_shard", "all_dropped"])
def test_prepare_moe_input_drops_negative_ids(num_tokens, routing):
    # Expert-parallel callers mark experts owned by other ranks with -1.
    # 8 * 300 routes exceed the single-kernel limit and use the two-kernel path.
    num_local_experts, num_global_experts, top_k, hidden_dim = 16, 64, 8, 128
    torch.manual_seed(41)
    noise = torch.rand(num_tokens, num_global_experts)
    topk_ids = torch.argsort(noise, dim=1)[:, :top_k].to(torch.int32).contiguous()
    if routing == "all_dropped":
        topk_ids.fill_(-1)
    else:
        topk_ids[topk_ids >= num_local_experts] = -1

    flat_ids = topk_ids.flatten()
    valid = flat_ids >= 0
    num_valid = int(valid.sum())
    expected_counts = torch.bincount(
        flat_ids[valid].to(torch.int64), minlength=num_local_experts
    ).to(torch.int32)

    device = "xpu"
    expert_offsets = torch.zeros(num_local_experts, dtype=torch.int32, device=device)
    problem_sizes1 = torch.zeros(
        num_local_experts * 3, dtype=torch.int32, device=device
    )
    problem_sizes2 = torch.zeros(
        num_local_experts * 3, dtype=torch.int32, device=device
    )
    input_permutation = torch.empty(flat_ids.numel(), dtype=torch.int32, device=device)
    output_permutation = torch.empty(flat_ids.numel(), dtype=torch.int32, device=device)
    prepare_moe_input(
        topk_ids.to(device),
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        input_permutation,
        output_permutation,
        num_local_experts,
        hidden_dim,
        top_k,
    )
    torch.xpu.synchronize()

    torch.testing.assert_close(expert_offsets.cpu(), expected_counts)
    perm = output_permutation.cpu()
    assert torch.equal(perm[~valid], torch.full_like(perm[~valid], -1))

    # Valid routes map one-to-one onto rows grouped by ascending expert id.
    dst = perm[valid].to(torch.int64)
    assert torch.equal(torch.sort(dst).values, torch.arange(num_valid))
    expert_ends = torch.cumsum(expected_counts, 0).to(torch.int64)
    expert_starts = expert_ends - expected_counts
    ids = flat_ids[valid].to(torch.int64)
    assert bool(((dst >= expert_starts[ids]) & (dst < expert_ends[ids])).all())

    # Dropped routes neither write rows nor contribute to the combined output.
    input_tensor = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16)
    scattered = torch.zeros(
        num_tokens * top_k, hidden_dim, dtype=torch.bfloat16, device=device
    )
    scatter_tokens_to_experts(input_tensor.to(device), output_permutation, scattered)
    route_tokens = torch.arange(flat_ids.numel())[valid] // top_k
    torch.testing.assert_close(
        scattered.cpu()[dst], input_tensor[route_tokens], rtol=0, atol=0
    )

    combined = torch.empty(num_tokens, hidden_dim, dtype=torch.bfloat16, device=device)
    factors = torch.full(
        (flat_ids.numel(),), 1 / top_k, dtype=torch.float32, device=device
    )
    apply_shuffle_mul_sum(scattered, combined, output_permutation, factors)
    valid_per_token = valid.view(num_tokens, top_k).sum(dim=1, keepdim=True)
    expected = (input_tensor.float() * valid_per_token / top_k).to(torch.bfloat16)
    torch.testing.assert_close(combined.cpu(), expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
