import pytest
import sgl_kernel
import torch
import torch.nn.functional as F
import utils

device = utils.get_device()


def fused_topk_torch_native(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    assert (
        hidden_states.shape[0] == gating_output.shape[0]
    ), f"Number of tokens mismatch, {hidden_states.shape=} vs {gating_output.shape=}"
    M, _ = hidden_states.shape
    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
    topk_weights = F.softmax(gating_output.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(topk_weights, topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("n_token", [2, 32, 4096])
@pytest.mark.parametrize("n_expert", [8, 32])
@pytest.mark.parametrize("n_topk", [1, 2, 4])
@pytest.mark.parametrize("renormalize", [False, True])
def test_topk_softmax(dtype, n_token, n_topk, n_expert, renormalize):
    torch.manual_seed(1024)

    # expand gating_output by M, otherwise bfloat16 fall into same value aftering truncating
    hidden_states = torch.randn(n_token, 100, device=device, dtype=dtype)
    gating_output = (
        torch.randn(n_token, n_expert, device=device, dtype=dtype) * 2 * n_token
    )

    ref_token_weights, ref_topk_indices = fused_topk_torch_native(
        hidden_states.float(),
        gating_output.float(),
        n_topk,
        renormalize,
    )

    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    M, _ = hidden_states.shape

    topk_weights = torch.empty(
        M, n_topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_indices = torch.empty(
        M, n_topk, dtype=torch.int32, device=hidden_states.device
    )

    sgl_kernel.topk_softmax(
        topk_weights,
        topk_indices,
        gating_output,
        renormalize,
    )

    # Compare the results
    res = torch.zeros(n_token, n_expert, dtype=torch.float, device=hidden_states.device)
    ref = torch.zeros(n_token, n_expert, dtype=torch.float, device=hidden_states.device)
    res.scatter_(1, topk_indices.long(), topk_weights)
    ref.scatter_(1, ref_topk_indices.long(), ref_token_weights)
    
    atol = 3e-3
    rtol = 1e-3
    torch.testing.assert_close(res, ref, atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__])
