import sys
from typing import (
    TYPE_CHECKING,
    Callable,
    NamedTuple,
    Optional,
    Protocol,
    TypeGuard,
    runtime_checkable,
)

import pytest
import torch
from sgl_kernel import moe_fused_gate


def assert_equal(
    score: torch.Tensor,
    indices_ref: torch.Tensor,
    indices_our: torch.Tensor,
    output_ref: torch.Tensor,
    output_our: torch.Tensor,
    bs: int,
    rtol: float = 1e-2,
    atol: float = 1e-3,
    max_permit_error: int = 0,
):
    indices_our_cpu = indices_our.cpu().tolist()
    indices_ref_cpu = indices_ref.cpu().tolist()
    output_our_cpu = output_our.cpu().tolist()
    output_ref_cpu = output_ref.cpu().tolist()

    wrong_values = 0
    for token_idx in range(bs):
        indices_ref_set = set(indices_ref_cpu[token_idx])
        indices_our_set = set(indices_our_cpu[token_idx])
        more = indices_our_set - indices_ref_set
        less = indices_ref_set - indices_our_set
        if more or less:
            more_values = sorted(score[token_idx, index].item() for index in more)
            less_values = sorted(score[token_idx, index].item() for index in less)
            if more_values != less_values:
                wrong_values += len(more)
                print(
                    f"{token_idx=}, {more=}, {less=} failed, with "
                    f"{more_values=}, {less_values=}"
                )
        assert wrong_values <= max_permit_error, f"{wrong_values=}, {max_permit_error=}"

        ref_output_by_index = dict(
            zip(indices_ref_cpu[token_idx], output_ref_cpu[token_idx])
        )
        output_by_index = dict(zip(indices_our_cpu[token_idx], output_our_cpu[token_idx]))
        for index in indices_ref_set & indices_our_set:
            expected = ref_output_by_index[index]
            actual = output_by_index[index]
            tolerance = atol + rtol * abs(expected)
            assert abs(actual - expected) <= tolerance, (
                f"Output mismatch at token {token_idx}, expert {index}: "
                f"expected {expected}, got {actual}, tolerance {tolerance}"
            )


def biased_grouped_topk_native(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: Optional[torch.Tensor],
    topk: int,
    renormalize: bool,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
    scoring_func: str = "sigmoid",
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    if scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    elif scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    else:
        raise ValueError(f"Unknown scoring_func: {scoring_func}")

    num_token = scores.shape[0]
    num_experts = scores.shape[1]
    scores_for_choice = scores.view(num_token, -1)
    if correction_bias is not None:
        scores_for_choice = scores_for_choice + correction_bias.unsqueeze(0)
    group_sum_count = 1 if scoring_func == "softmax" else 2
    group_scores = (
        scores_for_choice.view(num_token, num_expert_group, -1)
        .topk(group_sum_count, dim=-1)[0]
        .sum(dim=-1)
    )  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores_for_choice.masked_fill(
        ~score_mask.bool(), float("-inf")
    )  # [n, e]
    # TODO: NPU can't support directly evaluating a comparison for now
    _, topk_ids = torch.topk(
        tmp_scores,
        k=topk,
        dim=-1,
        sorted=(True if num_fused_shared_experts > 0 else False),
    )
    topk_weights = scores.gather(1, topk_ids)

    if num_fused_shared_experts:
        topk_ids[:, -1] = torch.randint(
            low=num_experts,
            high=num_experts + num_fused_shared_experts,
            size=(topk_ids.size(0),),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        topk_weights[:, -1] = topk_weights[:, :-1].sum(dim=-1) / routed_scaling_factor

    if renormalize:
        topk_weights_sum = (
            topk_weights.sum(dim=-1, keepdim=True)
            if num_fused_shared_experts == 0
            else topk_weights[:, :-1].sum(dim=-1, keepdim=True)
        )
        topk_weights = topk_weights / topk_weights_sum
        if apply_routed_scaling_factor_on_output:
            topk_weights *= routed_scaling_factor

    topk_weights, topk_ids = topk_weights.to(torch.float32), topk_ids.to(torch.int32)
    return topk_weights, topk_ids


@pytest.mark.parametrize(
    "seq_length",
    list(range(1, 10))
    + [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
)
@pytest.mark.parametrize(
    "params",
    [
        (128, 4, 2, 4),
        (256, 8, 4, 8),  # deepseek v3
        (512, 16, 8, 16),
    ],
)
# @pytest.mark.parametrize("num_fused_shared_experts", [0, 1, 2])
@pytest.mark.parametrize("num_fused_shared_experts", [0])
@pytest.mark.parametrize("apply_routed_scaling_factor_on_output", [False, True])
@pytest.mark.parametrize("scoring_func", ["sigmoid", "softmax"])
@pytest.mark.parametrize("renormalize", [False, True])
def test_moe_fused_gate_combined(
    seq_length,
    params,
    num_fused_shared_experts,
    apply_routed_scaling_factor_on_output,
    scoring_func,
    renormalize,
):
    num_experts, num_expert_group, topk_group, topk = params
    dtype = torch.float32

    torch.manual_seed(seq_length)
    tensor = torch.rand((seq_length, num_experts), dtype=dtype, device="xpu")
    scores = tensor.clone()
    if scoring_func == "softmax":
        # grouped_topk with softmax activation does not use correction bias.
        bias = None
    else:
        bias = torch.rand(num_experts, dtype=dtype, device="xpu")
    if scoring_func == "sigmoid":
        selection_scores = tensor.sigmoid()
    else:
        selection_scores = torch.softmax(tensor, dim=-1)
    if bias is not None:
        selection_scores = selection_scores + bias.unsqueeze(0)
    topk = topk + num_fused_shared_experts
    output, indices = moe_fused_gate(
        tensor,
        bias,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        topk=topk,
        renormalize=renormalize,
        scoring_func=scoring_func,
        num_fused_shared_experts=num_fused_shared_experts,
        routed_scaling_factor=2.5,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
    )

    ref_output, ref_indices = biased_grouped_topk_native(
        scores,
        scores,
        bias,
        topk=topk,
        renormalize=renormalize,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        num_fused_shared_experts=num_fused_shared_experts,
        routed_scaling_factor=2.5,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
        scoring_func=scoring_func,
    )

    # When num_fused_shared_experts > 0, ignore the comparison of the last topk dimension
    if num_fused_shared_experts > 0:
        original_indices = indices.clone()
        original_ref_indices = ref_indices.clone()

        indices = indices[:, :-1]
        ref_indices = ref_indices[:, :-1]

        valid_min = num_experts
        valid_max = num_experts + num_fused_shared_experts
        shared_indices = original_indices[:, -1]
        shared_ref_indices = original_ref_indices[:, -1]
        if shared_indices is not None:
            assert torch.all(
                (shared_indices >= valid_min) & (shared_indices < valid_max)
            ), f"Shared expert indices out of range: found values outside [{valid_min}, {valid_max})"
        if shared_ref_indices is not None:
            assert torch.all(
                (shared_ref_indices >= valid_min) & (shared_ref_indices < valid_max)
            ), f"Shared expert reference indices out of range: found values outside [{valid_min}, {valid_max})"

    assert_equal(
        selection_scores,
        ref_indices,
        indices,
        ref_output,
        output,
        seq_length,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
