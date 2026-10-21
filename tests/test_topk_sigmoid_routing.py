import sys
from typing import Optional

import pytest
import sgl_kernel
import torch
import torch.nn.functional as F
import utils

device = utils.get_device()
requires_xe35 = pytest.mark.skipif(
    not torch.xpu.is_available() or not sgl_kernel.is_xe3_arch(),
    reason="routing topk Xe35 kernels require Intel Xe3.5",
)


def fused_topk_sigmoid_torch_native_xe35(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    correction_bias: Optional[torch.Tensor],
):
    assert (
        hidden_states.shape[0] == gating_output.shape[0]
    ), f"Number of tokens mismatch, {hidden_states.shape=} vs {gating_output.shape=}"
    if correction_bias is not None:
        n_routed_experts = gating_output.shape[-1]
        scores = F.sigmoid(gating_output)
        scores_for_choice = scores.view(
            -1, n_routed_experts
        ) + correction_bias.unsqueeze(0)
        topk_ids = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_ids)
    else:
        M, _ = hidden_states.shape
        topk_weights = torch.empty(
            M, topk, dtype=torch.float32, device=hidden_states.device
        )
        topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
        topk_weights = F.sigmoid(gating_output.float())
        topk_weights, topk_ids = torch.topk(topk_weights, topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids


def assert_topk_sigmoid_contract_xe35(
    weights, indices, gating_output, renormalize, correction_bias, *, atol, rtol
):
    scores = torch.sigmoid(gating_output.float())
    ranking_scores = scores if correction_bias is None else scores + correction_bias
    assert indices.dtype == torch.int32
    assert weights.shape == indices.shape
    assert bool(((indices >= 0) & (indices < gating_output.size(1))).all())
    sorted_indices = torch.sort(indices, dim=-1).values
    assert bool((torch.diff(sorted_indices, dim=-1) != 0).all())
    selected_ranking_scores = ranking_scores.gather(1, indices.long())
    expected_ranking_scores = torch.topk(
        ranking_scores, indices.size(1), dim=-1, sorted=True
    ).values
    torch.testing.assert_close(
        selected_ranking_scores, expected_ranking_scores, atol=0, rtol=0
    )
    expected_weights = scores.gather(1, indices.long())
    if renormalize:
        expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(weights, expected_weights, atol=atol, rtol=rtol)


# float32 covers the fp32 gating-logits path: some models (e.g.
# Nemotron-3-Nano MoE) emit float32 router logits. Before the kernel dispatch
# was widened from AT_DISPATCH_REDUCED_FLOATING_TYPES to also cover Float, this
# call raised "not implemented for 'Float'". The kernel upcasts to float
# internally, so fp32 in must match the float reference within the same
# tolerance as the reduced-float path.
# n_token / n_expert / n_topk lists expanded to cover PO topk_sigmoid_xe35 CFGs.
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("n_token", [1, 2, 32, 128, 4096])
@pytest.mark.parametrize("n_expert", [8, 32, 128, 256])
@pytest.mark.parametrize("n_topk", [1, 2, 4, 8])
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("with_correction_bias", [False, True])
@requires_xe35
def test_topk_sigmoid_xe35(
    dtype, n_token, n_topk, n_expert, renormalize, with_correction_bias
):
    torch.manual_seed(1024)

    # expand gating_output by M, otherwise bfloat16 fall into same value after truncating
    hidden_states = torch.randn(n_token, 100, device=device, dtype=dtype)
    gating_output = torch.randn(n_token, n_expert, device=device, dtype=dtype)
    correction_bias = None
    if with_correction_bias:
        correction_bias = torch.randn((n_expert), dtype=torch.float32, device=device)

    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    M, _ = hidden_states.shape

    topk_weights = torch.empty(
        M, n_topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_indices = torch.empty(
        M, n_topk, dtype=torch.int32, device=hidden_states.device
    )

    sgl_kernel.topk_sigmoid_xe35(
        topk_weights,
        topk_indices,
        gating_output,
        renormalize,
        correction_bias,
    )

    assert_topk_sigmoid_contract_xe35(
        topk_weights,
        topk_indices,
        gating_output,
        renormalize,
        correction_bias,
        atol=1e-3,
        rtol=1e-3,
    )


@pytest.mark.parametrize("renormalize", [False, True])
def test_topk_sigmoid_oracle_accepts_equal_score_alternatives_xe35(renormalize):
    logits = torch.tensor([[-2.0, -1.0, 0.0, 1.0]])
    scores = torch.sigmoid(logits)
    bias = 1.0 - scores[0]
    for indices in (torch.tensor([[0, 2]]), torch.tensor([[3, 1]])):
        weights = scores.gather(1, indices)
        if renormalize:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        assert_topk_sigmoid_contract_xe35(
            weights, indices.int(), logits, renormalize, bias, atol=1e-3, rtol=1e-3
        )


@pytest.mark.parametrize(
    "invalid", ["negative", "out_of_range", "duplicate", "non_topk", "weight"]
)
def test_topk_sigmoid_oracle_rejects_invalid_outputs_xe35(invalid):
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    indices = torch.tensor([[3, 2]], dtype=torch.int32)
    weights = torch.sigmoid(logits).gather(1, indices.long())
    if invalid == "negative":
        indices[0, 1] = -1
    elif invalid == "out_of_range":
        indices[0, 1] = 4
    elif invalid == "duplicate":
        indices[0, 1] = 3
    elif invalid == "non_topk":
        indices[0, 1] = 1
    else:
        weights[0, 1] = 0.0
    with pytest.raises(AssertionError):
        assert_topk_sigmoid_contract_xe35(
            weights, indices, logits, False, None, atol=1e-3, rtol=1e-3
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("n_token", [1, 7, 9, 17, 33, 129, 512, 2048, 4096])
@pytest.mark.parametrize("n_expert", [32, 64, 128, 256])
@requires_xe35
def test_topk8_fast_path_xe35(dtype, n_token, n_expert):
    """Check all full subgroup vector tiles, allowing equal-score ID ties."""
    torch.manual_seed(20260916 + n_token + n_expert)
    gating_output = torch.empty(n_token, n_expert, dtype=dtype, device=device).uniform_(
        -7.0, 7.0
    )
    correction_bias = torch.empty(
        n_expert, dtype=torch.float32, device=device
    ).uniform_(-0.15, 0.15)

    expected_scores = torch.sigmoid(gating_output.float())
    ranking_scores = expected_scores + correction_bias.unsqueeze(0)
    expected_rank_scores = torch.topk(
        ranking_scores,
        k=8,
        dim=-1,
        sorted=True,
    ).values

    actual_weights = torch.empty(n_token, 8, dtype=torch.float32, device=device)
    actual_ids = torch.empty(n_token, 8, dtype=torch.int32, device=device)
    sgl_kernel.topk_sigmoid_xe35(
        actual_weights,
        actual_ids,
        gating_output,
        True,
        correction_bias,
    )

    assert bool(((actual_ids >= 0) & (actual_ids < n_expert)).all())
    assert bool((torch.diff(torch.sort(actual_ids, dim=-1).values, dim=-1) != 0).all())
    torch.testing.assert_close(
        ranking_scores.gather(1, actual_ids.long()),
        expected_rank_scores,
        atol=0,
        rtol=0,
    )
    selected_scores = expected_scores.gather(1, actual_ids.long())
    expected_weights = selected_scores / selected_scores.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(actual_weights, expected_weights, atol=2.5e-4, rtol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("n_expert", [32, 64, 128, 256])
@requires_xe35
def test_topk8_ties_choose_lowest_expert_xe35(dtype, n_expert):
    gating_output = torch.zeros(9, n_expert, dtype=dtype, device=device)
    correction_bias = torch.zeros(n_expert, dtype=torch.float32, device=device)
    actual_weights = torch.empty(9, 8, dtype=torch.float32, device=device)
    actual_ids = torch.empty(9, 8, dtype=torch.int32, device=device)

    sgl_kernel.topk_sigmoid_xe35(
        actual_weights,
        actual_ids,
        gating_output,
        True,
        correction_bias,
    )

    expected_ids = torch.arange(8, dtype=torch.int32, device=device).expand(9, 8)
    torch.testing.assert_close(actual_ids, expected_ids)
    torch.testing.assert_close(
        actual_weights,
        torch.full_like(actual_weights, 0.125),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("n_expert", [32, 64, 128, 256])
@pytest.mark.parametrize("n_token", [1, 129])
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
@requires_xe35
def test_full_tile_topk8_options_xe35(dtype, n_expert, n_token, renormalize, with_bias):
    torch.manual_seed(31000 + n_expert + n_token)
    logits = torch.randn(n_token, n_expert, dtype=dtype, device=device)
    bias = (
        torch.randn(n_expert, dtype=torch.float32, device=device) if with_bias else None
    )
    weights = torch.empty(n_token, 8, dtype=torch.float32, device=device)
    ids = torch.empty(n_token, 8, dtype=torch.int32, device=device)
    sgl_kernel.topk_sigmoid_xe35(weights, ids, logits, renormalize, bias)

    scores = torch.sigmoid(logits.float())
    rank = scores if bias is None else scores + bias
    torch.testing.assert_close(
        rank.gather(1, ids.long()),
        torch.topk(rank, 8, dim=-1, sorted=True).values,
        rtol=0,
        atol=0,
    )
    selected = scores.gather(1, ids.long())
    expected = selected / selected.sum(-1, keepdim=True) if renormalize else selected
    torch.testing.assert_close(weights, expected, rtol=0, atol=2.5e-4)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
