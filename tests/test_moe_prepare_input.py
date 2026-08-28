import itertools
import sys

import pytest
import torch
from sgl_kernel import (
    apply_shuffle_mul_sum,
    prepare_moe_input,
    scatter_tokens_to_experts,
)


@pytest.mark.parametrize("num_tokens", [1, 2, 5, 16, 64, 128, 224, 1024])
@pytest.mark.parametrize("num_experts", [1, 4, 8, 32, 40, 64, 128, 256])
@pytest.mark.parametrize("top_k", [1, 2, 4, 6, 8])
@pytest.mark.parametrize("hidden_dims", [16, 32, 64, 128, 1024, 1536])
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
