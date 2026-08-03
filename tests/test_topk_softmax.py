import sys

import pytest
import sgl_kernel
import torch
import torch.nn.functional as F
import utils

device = utils.get_device()


def assert_equal(
    score: torch.Tensor,
    indices_ref: torch.Tensor,
    indices_our: torch.Tensor,
    bs: int,
    max_permit_error: int = 0,
):
    indices_our_cpu = indices_our.cpu().tolist()
    indices_ref_cpu = indices_ref.cpu().tolist()

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
@pytest.mark.parametrize("n_expert", [8, 32, 256])
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

    assert_equal(
        F.softmax(gating_output.float(), dim=-1),
        ref_topk_indices,
        topk_indices,
        n_token,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
