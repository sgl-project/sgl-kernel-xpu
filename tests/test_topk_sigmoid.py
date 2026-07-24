import sys
from typing import Optional

import pytest
import sgl_kernel
import torch
import torch.nn.functional as F
import utils

device = utils.get_device()


def assert_equal(
    score: torch.Tensor,
    ref_indices: torch.Tensor,
    indices: torch.Tensor,
    bs: int,
    max_permit_error: int = 0,
):
    indices_ref_cpu = ref_indices.cpu().tolist()
    indices_our_cpu = indices.cpu().tolist()

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


def fused_topk_sigmoid_torch_native(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    correction_bias: Optional[torch.Tensor],
    routed_scaling_factor: float = 1.0,
    num_fused_shared_experts: int = 0,
):
    assert (
        hidden_states.shape[0] == gating_output.shape[0]
    ), f"Number of tokens mismatch, {hidden_states.shape=} vs {gating_output.shape=}"
    routed_topk = topk - num_fused_shared_experts
    assert routed_topk > 0, "routed_topk must be positive"

    if correction_bias is not None:
        n_routed_experts = gating_output.shape[-1]
        scores = F.sigmoid(gating_output)
        scores_for_choice = scores.view(
            -1, n_routed_experts
        ) + correction_bias.unsqueeze(0)
        topk_ids = torch.topk(scores_for_choice, k=routed_topk, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_ids)
    else:
        topk_weights = F.sigmoid(gating_output.float())
        topk_weights, topk_ids = torch.topk(topk_weights, routed_topk, dim=-1)

    row_sum = topk_weights.sum(dim=-1, keepdim=True)

    if renormalize:
        topk_weights = topk_weights * (routed_scaling_factor / (row_sum + 1e-20))

    if num_fused_shared_experts > 0:
        shared_id = torch.full(
            (topk_ids.shape[0], 1),
            gating_output.shape[-1],
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        if renormalize:
            shared_weight = torch.ones(
                (topk_weights.shape[0], 1),
                dtype=topk_weights.dtype,
                device=topk_weights.device,
            )
        else:
            shared_weight = row_sum / routed_scaling_factor

        topk_ids = torch.cat([topk_ids, shared_id], dim=-1)
        topk_weights = torch.cat([topk_weights, shared_weight], dim=-1)

    return topk_weights, topk_ids


# float32 covers the fp32 gating-logits path: some models (e.g.
# Nemotron-3-Nano MoE) emit float32 router logits. Before the kernel dispatch
# was widened from AT_DISPATCH_REDUCED_FLOATING_TYPES to also cover Float, this
# call raised "not implemented for 'Float'". The kernel upcasts to float
# internally, so fp32 in must match the float reference within the same
# tolerance as the reduced-float path.
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("n_token", [2, 32, 4096])
@pytest.mark.parametrize("n_expert", [8, 32, 256])
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("with_correction_bias", [False, True])
@pytest.mark.parametrize("routed_scaling_factor", [1.0, 2.5])
@pytest.mark.parametrize(
    "n_topk,num_fused_shared_experts",
    [(1, 0), (2, 0), (2, 1), (4, 0), (4, 1)],
)
def test_topk_sigmoid(
    dtype,
    n_token,
    n_topk,
    n_expert,
    renormalize,
    with_correction_bias,
    routed_scaling_factor,
    num_fused_shared_experts,
):
    torch.manual_seed(1024)

    # expand gating_output by M, otherwise bfloat16 fall into same value after truncating
    hidden_states = torch.randn(n_token, 100, device=device, dtype=dtype)
    gating_output = torch.randn(n_token, n_expert, device=device, dtype=dtype)
    correction_bias = None
    if with_correction_bias:
        correction_bias = torch.randn((n_expert), dtype=torch.float32, device=device)

    selection_scores = F.sigmoid(gating_output.float())
    if correction_bias is not None:
        selection_scores += correction_bias.unsqueeze(0)

    ref_token_weights, ref_topk_indices = fused_topk_sigmoid_torch_native(
        hidden_states.float(),
        gating_output.float(),
        n_topk,
        renormalize,
        correction_bias,
        routed_scaling_factor=routed_scaling_factor,
        num_fused_shared_experts=num_fused_shared_experts,
    )

    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    M, _ = hidden_states.shape

    topk_weights = torch.empty(
        M, n_topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_indices = torch.empty(
        M, n_topk, dtype=torch.int32, device=hidden_states.device
    )

    sgl_kernel.topk_sigmoid(
        topk_weights,
        topk_indices,
        gating_output,
        renormalize,
        correction_bias,
        routed_scaling_factor,
        num_fused_shared_experts,
    )

    assert_equal(
        selection_scores,
        ref_topk_indices,
        topk_indices,
        n_token,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
