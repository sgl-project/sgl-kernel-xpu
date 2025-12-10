import itertools

import pytest
import torch
from sgl_kernel import apply_shuffle_mul_sum, prepare_moe_input, shuffle_rows


@pytest.mark.parametrize("num_tokens", [1, 5, 16, 128, 1024])
@pytest.mark.parametrize("num_experts", [1, 4, 8, 32, 64, 128])
@pytest.mark.parametrize("top_k", [1, 2, 4, 8])
@pytest.mark.parametrize("hidden_dims", [16, 32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_prepare_input_moe(num_tokens, num_experts, top_k, hidden_dims, dtype):
    if num_experts < top_k:
        pytest.skip("invalid combination")
    torch.manual_seed(41)

    # Generate unique token
    def generate_unique_topk_ids(tokens, top_k, num_experts):
        topk_ids = torch.empty((tokens, top_k), dtype=torch.int32)
        # avoid duplicate tokens
        for T in range(tokens):
            topk_ids[T] = torch.randperm(num_experts, dtype=torch.int32)[:top_k]
        return topk_ids

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
        tokens, top_k = topk_ids.shape
        expert_cnt = torch.zeros(num_experts, dtype=torch.int32)
        for e in range(num_experts):
            expert_cnt[e] = (topk_ids == e).sum()
            expert_offsets[e] = expert_cnt[e]

        for e in range(num_experts):
            r = expert_cnt[e].item()
            c = hidden_dim
            problem_sizes1[e * 3 + 0] = r
            problem_sizes1[e * 3 + 1] = c * 2
            problem_sizes1[e * 3 + 2] = top_k
            problem_sizes2[e * 3 + 0] = r
            problem_sizes2[e * 3 + 1] = top_k
            problem_sizes2[e * 3 + 2] = c

        # compute offsets
        atomic_buffer = torch.zeros(num_experts, dtype=torch.int32)
        tot_offset = 0
        # expert_offsets[0] = 0
        for i in range(num_experts):
            atomic_buffer[i] = tot_offset
            tot_offset += problem_sizes1[i * 3].item()
            # expert_offsets[i + 1] = tot_offset

        # compute input/output permutes
        num_tokens = topk_ids.size(0)
        flat_topk = topk_ids.flatten()
        topk_length = num_tokens * top_k

        for i in range(topk_length):
            expert_id = int(flat_topk[i])
            start = int(atomic_buffer[expert_id].item())
            atomic_buffer[expert_id] += 1

            input_permutation[start] = i // top_k
            output_permutation[i] = start

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
    output_tensor_xpu = shuffle_rows(
        input_tensor_xpu, input_permutation_xpu, (num_tokens * top_k, hidden_dims)
    )
    input_merge_xpu = torch.empty((num_tokens, hidden_dims), dtype=dtype, device=device)
    # apply weights
    factors = torch.ones(top_k * num_tokens, dtype=torch.float32, device=device).fill_(
        1 / top_k
    )
    apply_shuffle_mul_sum(
        output_tensor_xpu, input_merge_xpu, output_permutation_xpu, factors
    )
    # of same order as in input
    torch.testing.assert_allclose(input_merge_xpu.to("cpu"), input_tensor)
