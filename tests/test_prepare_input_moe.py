import itertools

import pytest
import torch
from sgl_kernel import prepare_moe_input


@pytest.mark.parametrize("n_token", [8])
@pytest.mark.parametrize("n_expert", [4])
@pytest.mark.parametrize("top-k", [2])
@pytest.mark.parametrize("hidden_dims", [16])
def test_prepare_input_moe():

    # Generate unique token
    def generate_unique_topk_ids(tokens, top_k, num_experts):
        topk_ids = torch.empty((tokens, top_k), dtype=torch.int32)
        #avoid duplicate tokens
        for T in range(tokens):
            topk_ids[T] = torch.randperm(num_experts, dtype=torch.int32)[:top_k]
        return topk_ids

    # compute problem sizes and expert offsets
    def compute_problem_sizes(topk_ids, num_experts, hidden_dim):
        tokens, top_k = topk_ids.shape
        expert_cnt = torch.zeros(num_experts, dtype=torch.int32)
        for e in range(num_experts):
            expert_cnt[e] = (topk_ids == e).sum()

        expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=topk_ids.device)
        expert_offsets[1:] = torch.cumsum(expert_cnt, dim=0)

        problem_sizes1 = torch.zeros(num_experts * 3, dtype=torch.int32, device=topk_ids.device)
        problem_sizes2 = torch.zeros(num_experts * 3, dtype=torch.int32, device=topk_ids.device)
        for e in range(num_experts):
            r = expert_cnt[e].item()
            c = hidden_dim
            problem_sizes1[e * 3 + 0] = r
            problem_sizes1[e * 3 + 1] = c * 2
            problem_sizes1[e * 3 + 2] = top_k
            problem_sizes2[e * 3 + 0] = r
            problem_sizes2[e * 3 + 1] = top_k
            problem_sizes2[e * 3 + 2] = c

        return expert_offsets, problem_sizes1, problem_sizes2


    topk_ids = generate_unique_topk_ids(n_token, top-k, n_expert)
    # gen Ref
    expert_offsets, problem_sizes1, problem_sizes1 = compute_problem_sizes_sim(topk_ids, n_expert, hidden_dims)


if __name__ == "__main__":
    pytest.main([__file__])
