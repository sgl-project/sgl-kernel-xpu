import sys
from typing import Optional

import pytest
import torch
import utils
from sgl_kernel import biased_topk

device = utils.get_device()


def biased_topk_torch_native(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    scoring_func: str = "sigmoid",
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    if scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    elif scoring_func == "sqrtsoftplus":
        scores = torch.nn.functional.softplus(gating_output).sqrt()

    num_token = scores.shape[0]
    num_experts = scores.shape[1]

    scores_for_choice = scores.view(num_token, -1) + correction_bias.unsqueeze(0)
    _, topk_ids = torch.topk(
        scores_for_choice,
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
        if routed_scaling_factor is not None:
            topk_weights[:, -1] = (
                topk_weights[:, :-1].sum(dim=-1) / routed_scaling_factor
            )

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


def _make_inputs(M: int, num_experts: int, seed: int):
    torch.manual_seed(seed)
    scores = torch.randn(M, num_experts, dtype=torch.float32, device=device) * 2.0
    bias = torch.randn(num_experts, dtype=torch.float32, device=device) * 0.5
    return scores, bias


def _scatter_by_expert(
    weights: torch.Tensor, indices: torch.Tensor, num_columns: int
) -> torch.Tensor:
    dense = torch.zeros(
        (weights.shape[0], num_columns), dtype=torch.float32, device=weights.device
    )
    dense.scatter_(1, indices.long(), weights.float())
    return dense


@pytest.mark.parametrize("M", [1, 64, 1024])
@pytest.mark.parametrize("num_experts", [128, 384, 512])
@pytest.mark.parametrize("topk", [4, 6, 8])
@pytest.mark.parametrize("scoring_func", ["sigmoid", "sqrtsoftplus"])
@pytest.mark.parametrize("num_shared", [0, 1])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("apply_scale", [True, False])
def test_biased_topk(
    M: int,
    num_experts: int,
    topk: int,
    scoring_func: str,
    num_shared: int,
    renormalize: bool,
    apply_scale: bool,
) -> None:
    hidden_states = torch.randn(M, 100, device=device, dtype=torch.float32)
    scores, bias = _make_inputs(M, num_experts, seed=num_experts * 100 + topk)
    scale = 2.5

    ref_weights, ref_indices = biased_topk_torch_native(
        hidden_states,
        scores,
        bias,
        topk,
        renormalize,
        scoring_func,
        num_shared,
        scale,
        apply_scale,
    )

    topk_weights = torch.empty(M, topk, dtype=torch.float32, device=device)
    topk_indices = torch.empty(M, topk, dtype=torch.int32, device=device)

    biased_topk(
        scores,
        bias,
        topk_weights,
        topk_indices,
        topk,
        scoring_func,
        num_shared,
        renormalize,
        scale,
        apply_scale,
    )

    num_columns = num_experts + num_shared
    torch.testing.assert_close(
        _scatter_by_expert(topk_weights, topk_indices, num_columns),
        _scatter_by_expert(ref_weights, ref_indices, num_columns),
        rtol=1e-4,
        atol=1e-5,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M", [1, 64, 1024])
@pytest.mark.parametrize("num_experts", [128, 384])
@pytest.mark.parametrize("scoring_func", ["sigmoid", "sqrtsoftplus"])
def test_biased_topk_reduced_precision_input(
    dtype: torch.dtype,
    M: int,
    num_experts: int,
    scoring_func: str,
) -> None:

    # configs from deepseek-v4
    topk = 6
    num_shared = 1
    renormalize = True
    apply_scale = True
    scale = 2.5

    hidden_states = torch.randn(M, 100, device=device, dtype=dtype)
    scores_fp32, bias = _make_inputs(M, num_experts, seed=num_experts * 100 + topk)
    scores = scores_fp32.to(dtype)

    ref_weights, ref_indices = biased_topk_torch_native(
        hidden_states,
        scores.float(),
        bias,
        topk,
        renormalize,
        scoring_func,
        num_shared,
        scale,
        apply_scale,
    )

    topk_weights = torch.empty(M, topk, dtype=torch.float32, device=device)
    topk_indices = torch.empty(M, topk, dtype=torch.int32, device=device)

    biased_topk(
        scores,
        bias,
        topk_weights,
        topk_indices,
        topk,
        scoring_func,
        num_shared,
        renormalize,
        scale,
        apply_scale,
    )

    num_columns = num_experts + num_shared
    torch.testing.assert_close(
        _scatter_by_expert(topk_weights, topk_indices, num_columns),
        _scatter_by_expert(ref_weights, ref_indices, num_columns),
        rtol=1e-4,
        atol=1e-5,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
