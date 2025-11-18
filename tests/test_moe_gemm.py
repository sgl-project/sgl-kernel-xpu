import itertools

import pytest
import torch
import torch.nn.functional as F
from sgl_kernel import fused_experts


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def create_random_xpu_tensor(shape, dtype, mean=0, std=0.01):
    """Create a random xpu tensor

    Args:
        shape: Tensor shape
        dtype: Data type
        mean: Mean value
        std: Standard deviation

    Returns:
        torch.Tensor: Randomly initialized xpu tensor
    """
    return torch.empty(shape, dtype=dtype, device="xpu").normal_(mean, std)


def torch_naive_moe(
    a,
    w1,
    w2,
    topk_ids,
    topk_weight,
    topk,
):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)

    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            tmp = silu_and_mul(a[mask] @ w1[i].transpose(0, 1))
            # import pdb; pdb.set_trace()
            out[mask] = tmp @ w2[i].transpose(0, 1)

    return (
        out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)


@pytest.mark.parametrize(
    "num_tokens,topk,num_experts,hidden_size,intermediate_size",
    list(
        itertools.product(
            [1, 4, 33, 64, 222],  # num_tokens
            [1, 2, 6],  # topk
            [8, 64],  #  num_experts
            [128, 1024],  # hidden_size
            [128, 512, 1024],  # intermediate_size
        )
    ),
)
def test_moe_gemm(num_tokens, topk, num_experts, hidden_size, intermediate_size):
    rtol, atol = 1e-1, 1e-2
    a = create_random_xpu_tensor((num_tokens, hidden_size), torch.bfloat16)
    w1 = create_random_xpu_tensor(
        (num_experts, 2 * intermediate_size, hidden_size), torch.bfloat16
    )
    w2 = create_random_xpu_tensor(
        (num_experts, hidden_size, intermediate_size), torch.bfloat16
    )
    score = torch.randn([num_tokens, num_experts], dtype=torch.bfloat16).to("xpu")

    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    torch_output = torch_naive_moe(
        a,
        w1,
        w2,
        topk_ids,
        topk_weight,
        topk,
    )
    sglang_output = fused_experts(
        a,
        w1,
        w2,
        topk_weight,
        topk_ids,
    )
    # import pdb; pdb.set_trace()
    torch.testing.assert_close(torch_output, sglang_output, rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main([__file__])
