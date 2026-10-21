"""Accuracy tests for fused MoE routing.

The reference follows native routing dtype transitions: multiply logits by the metap
multiplier, sigmoid, add the fp32 bias for selection only, take the top-k, then
emit the unbiased scores as `score * route_scale / (sum + eps)`. Padded rows
carry expert id -1, and the emitted ids pass through the expert-parallel table.

The router runs in fp32 in the production config (`moe_router_in_fp32`). The
reference keeps the native path's multiplication dtype and float32 sigmoid.
"""

from typing import Optional

import pytest
import sgl_kernel
import torch
import utils

device = utils.get_device()
pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available() or not sgl_kernel.is_xe3_arch(),
    reason="routing Xe35 kernels require Intel Xe3.5",
)


def routing_torch_native_xe35(
    gating_output: torch.Tensor,
    bias: Optional[torch.Tensor],
    topk: int,
    route_scale: float,
    router_output_multiplier: float,
    division_eps: float,
):
    """Mirror the native routing dtype transitions."""
    logits = gating_output * router_output_multiplier
    scores = torch.sigmoid(logits.float())
    rank_keys = scores if bias is None else scores + bias.unsqueeze(0)
    ids = torch.topk(rank_keys, k=topk, dim=-1, sorted=True)[1]
    weights = scores.gather(1, ids)
    weights = weights * (
        route_scale / (weights.sum(dim=-1, keepdim=True) + division_eps)
    )
    return weights, ids, scores, rank_keys


def check_routing_xe35(
    gating_output,
    bias,
    topk,
    route_scale,
    router_output_multiplier,
    division_eps,
    local_expert_mapping=None,
    num_valid=None,
):
    num_tokens, num_experts = gating_output.shape
    topk_weights = torch.empty(num_tokens, topk, dtype=torch.float32, device=device)
    topk_ids = torch.empty(num_tokens, topk, dtype=torch.int32, device=device)
    num_valid_tensor = (
        None
        if num_valid is None
        else torch.tensor([num_valid], dtype=torch.int32, device=device)
    )

    sgl_kernel.muse_spark_moe_routing_xe35(
        topk_weights,
        topk_ids,
        gating_output,
        bias,
        router_output_multiplier,
        route_scale,
        division_eps,
        local_expert_mapping,
        num_valid_tensor,
    )

    _, ref_ids, scores, rank_keys = routing_torch_native_xe35(
        gating_output, bias, topk, route_scale, router_output_multiplier, division_eps
    )
    scores = scores.cpu()
    ref_ids = ref_ids.cpu()
    rank_keys = rank_keys.cpu()
    got_ids = topk_ids.cpu()
    got_weights = topk_weights.cpu()

    inverse = None
    if local_expert_mapping is not None:
        mapping = local_expert_mapping.cpu().tolist()
        inverse = {local: glob for glob, local in enumerate(mapping)}

    valid = num_tokens if num_valid is None else num_valid
    for token in range(num_tokens):
        if token >= valid:
            assert torch.equal(
                got_ids[token], torch.full((topk,), -1, dtype=torch.int32)
            ), f"padded row {token} ids {got_ids[token].tolist()}"
            assert torch.all(got_weights[token] == 0.0), f"padded row {token} weights"
            continue

        selected = []
        for k in range(topk):
            expert = int(got_ids[token, k])
            if inverse is not None:
                assert (
                    expert in inverse
                ), f"token {token} rank {k}: id {expert} not local"
                expert = inverse[expert]
            assert 0 <= expert < num_experts, f"token {token} rank {k}: id {expert}"
            assert expert not in selected, f"token {token} rank {k}: duplicate {expert}"
            selected.append(expert)

        assert torch.equal(
            rank_keys[token, selected],
            rank_keys[token, ref_ids[token]],
        ), f"token {token}: rank scores differ for {selected} vs {ref_ids[token].tolist()}"

        total = sum(float(scores[token, e]) for e in selected)
        for k in range(topk):
            want = (
                float(scores[token, selected[k]]) * route_scale / (total + division_eps)
            )
            got = float(got_weights[token, k])
            assert abs(got - want) <= 2.5e-4 * max(
                1.0, abs(want)
            ), f"token {token} rank {k}: weight {got} vs {want}"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("num_tokens", [1, 7, 8, 33, 129, 1024])
@pytest.mark.parametrize("num_experts", [128, 256])
def test_moe_routing_production_shape_xe35(dtype, num_tokens, num_experts):
    """256 experts / top-k 8 is the production shape; 128 uses the same path."""
    # Muse decode is B8/E256/topk8; other rows and E128 are generic edge cases.
    torch.manual_seed(1024)
    gating_output = torch.randn(num_tokens, num_experts, dtype=dtype, device=device)
    bias = torch.randn(num_experts, dtype=torch.float32, device=device)
    check_routing_xe35(
        gating_output,
        bias,
        topk=8,
        route_scale=1.0,
        # 32 ** -0.5, the metap multiplier of Muse Spark 1.2 L3.
        router_output_multiplier=0.3535533905932738,
        division_eps=1e-15,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("num_experts", [2, 8, 33, 96, 255])
@pytest.mark.parametrize("topk", [1, 4, 8])
def test_moe_routing_generic_shapes_xe35(dtype, num_experts, topk):
    """Cover partial subgroup tiles, different top-k values and dtypes."""
    if topk > num_experts:
        pytest.skip("topk must not exceed the expert count")
    torch.manual_seed(7)
    gating_output = torch.randn(17, num_experts, dtype=dtype, device=device)
    bias = torch.randn(num_experts, dtype=torch.float32, device=device)
    check_routing_xe35(
        gating_output,
        bias,
        topk=topk,
        route_scale=2.5,
        router_output_multiplier=1.0,
        division_eps=0.0,
    )


def test_moe_routing_prefill_xe35():
    # Muse prefill is B8/ISL4096/E256/topk8/TP1/EP1 with an FP32 router.
    torch.manual_seed(20261016)
    tokens, experts, topk = 8 * 4096, 256, 8
    logits = torch.randn(tokens, experts, dtype=torch.float32, device=device)
    bias = torch.randn(experts, dtype=torch.float32, device=device)
    weights = torch.empty(tokens, topk, dtype=torch.float32, device=device)
    indices = torch.empty(tokens, topk, dtype=torch.int32, device=device)
    multiplier = 0.3535533905932738
    sgl_kernel.muse_spark_moe_routing_xe35(
        weights, indices, logits, bias, multiplier, 1.0, 1e-15
    )
    _, _, scores, ranking_scores = routing_torch_native_xe35(
        logits, bias, topk, 1.0, multiplier, 1e-15
    )
    assert bool(((indices >= 0) & (indices < experts)).all())
    assert bool((torch.diff(torch.sort(indices, dim=-1).values, dim=-1) != 0).all())
    torch.testing.assert_close(
        ranking_scores.gather(1, indices.long()),
        torch.topk(ranking_scores, topk, dim=-1, sorted=True).values,
        rtol=0,
        atol=0,
    )
    selected = scores.gather(1, indices.long())
    expected_weights = selected / (selected.sum(dim=-1, keepdim=True) + 1e-15)
    torch.testing.assert_close(weights, expected_weights, rtol=0, atol=2.5e-4)


@pytest.mark.parametrize("num_experts", [96, 256])
def test_moe_routing_no_bias_xe35(num_experts):
    torch.manual_seed(11)
    gating_output = torch.randn(9, num_experts, dtype=torch.float32, device=device)
    check_routing_xe35(
        gating_output,
        None,
        topk=8,
        route_scale=1.0,
        router_output_multiplier=0.5,
        division_eps=1e-15,
    )


@pytest.mark.parametrize("num_experts", [96, 256])
@pytest.mark.parametrize("num_valid", [0, 1, 11])
def test_moe_routing_padded_rows_xe35(num_experts, num_valid):
    """Rows past the device-scalar valid count get id -1 and weight 0."""
    torch.manual_seed(13)
    gating_output = torch.randn(20, num_experts, dtype=torch.float32, device=device)
    bias = torch.randn(num_experts, dtype=torch.float32, device=device)
    check_routing_xe35(
        gating_output,
        bias,
        topk=8,
        route_scale=1.0,
        router_output_multiplier=0.3535533905932738,
        division_eps=1e-15,
        num_valid=num_valid,
    )


@pytest.mark.parametrize("num_experts", [48, 256])
@pytest.mark.parametrize("num_valid", [None, 11])
def test_moe_routing_local_expert_mapping_xe35(num_experts, num_valid):
    """Expert-parallel remap: only the emitted id changes, never the selection."""
    torch.manual_seed(17)
    topk = 8 if num_experts == 256 else 6
    gating_output = torch.randn(20, num_experts, dtype=torch.float32, device=device)
    bias = torch.randn(num_experts, dtype=torch.float32, device=device)
    # A reversing permutation: distinguishable from the identity at every id.
    mapping = torch.arange(num_experts - 1, -1, -1, dtype=torch.int32, device=device)
    check_routing_xe35(
        gating_output,
        bias,
        topk=topk,
        route_scale=1.0,
        router_output_multiplier=0.3535533905932738,
        division_eps=1e-15,
        local_expert_mapping=mapping,
        num_valid=num_valid,
    )


def test_moe_routing_zero_multiplier_xe35():
    """A zero multiplier makes every sigmoid exactly 0.5, so bias alone ranks."""
    torch.manual_seed(19)
    gating_output = torch.randn(5, 256, dtype=torch.float32, device=device)
    bias = torch.randn(256, dtype=torch.float32, device=device)
    topk_weights = torch.empty(5, 8, dtype=torch.float32, device=device)
    topk_ids = torch.empty(5, 8, dtype=torch.int32, device=device)
    sgl_kernel.muse_spark_moe_routing_xe35(
        topk_weights, topk_ids, gating_output, bias, 0.0, 1.0, 1e-15, None, None
    )
    torch.testing.assert_close(
        topk_weights.cpu(), torch.full((5, 8), 0.125), rtol=0, atol=1e-6
    )
    expected = torch.topk(bias, k=8, dim=-1, sorted=True)[1].to(torch.int32)
    for token in range(5):
        assert torch.equal(topk_ids[token].cpu(), expected.cpu())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("num_experts", [32, 63, 64, 65, 127, 128, 129, 255, 256])
@pytest.mark.parametrize("topk", [7, 8])
def test_moe_routing_equal_scores_choose_lowest_xe35(dtype, num_experts, topk):
    """Equal scores select lowest IDs across vector tiles and masked tails."""
    logits = torch.zeros(3, num_experts, dtype=dtype, device=device)
    bias = torch.zeros(num_experts, dtype=torch.float32, device=device)
    valid = torch.tensor([2], dtype=torch.int32, device=device)
    weights = torch.empty(3, topk, dtype=torch.float32, device=device)
    ids = torch.empty(3, topk, dtype=torch.int32, device=device)
    sgl_kernel.muse_spark_moe_routing_xe35(
        weights, ids, logits, bias, 1.0, 1.0, 1e-15, None, valid
    )
    expected_ids = torch.arange(topk, dtype=torch.int32, device=device)
    for row in range(2):
        torch.testing.assert_close(ids[row], expected_ids, rtol=0, atol=0)
        torch.testing.assert_close(
            weights[row], torch.full_like(weights[row], 1.0 / topk), rtol=0, atol=1e-6
        )
    torch.testing.assert_close(ids[2], torch.full_like(ids[2], -1), rtol=0, atol=0)
    torch.testing.assert_close(weights[2], torch.zeros_like(weights[2]), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("num_experts", [63, 64, 65, 127, 128, 129, 255, 256])
@pytest.mark.parametrize("topk", [7, 8])
def test_moe_routing_full_tile_neighbors_xe35(dtype, num_experts, topk):
    torch.manual_seed(30000 + num_experts + topk)
    logits = torch.randn(17, num_experts, dtype=dtype, device=device)
    bias = torch.randn(num_experts, dtype=torch.float32, device=device)
    check_routing_xe35(logits, bias, topk, 1.0, 0.3535533905932738, 1e-15)


@pytest.mark.parametrize("num_experts", [128, 256])
def test_moe_routing_near_topk_boundary_xe35(num_experts):
    """A small biased-score gap must preserve the production expert ID."""
    multiplier = 0.3535533905932738
    logits_cpu = torch.zeros(1, num_experts, dtype=torch.float32)
    logits_cpu[0, 70], logits_cpu[0, 71] = 1.8000080585479736, 2.9113547801971436
    scores = torch.sigmoid(logits_cpu * multiplier)
    bias_cpu = torch.full((num_experts,), -2.0)
    bias_cpu[:7] = torch.linspace(2.0, 1.4, 7)
    bias_cpu[70] = 0.9 - scores[0, 70]
    bias_cpu[71] = 0.9 - 1e-7 - scores[0, 71]
    _, expected_ids, _, keys = routing_torch_native_xe35(
        logits_cpu, bias_cpu, 8, 1.0, multiplier, 1e-15
    )
    assert 0 < keys[0, 70] - keys[0, 71] < 1e-6
    assert expected_ids[0, -1] == 70
    check_routing_xe35(
        logits_cpu.to(device),
        bias_cpu.to(device),
        topk=8,
        route_scale=1.0,
        router_output_multiplier=multiplier,
        division_eps=1e-15,
    )


if __name__ == "__main__":
    pytest.main([__file__])
