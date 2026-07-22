import sys
from typing import Tuple

import pytest
import torch
import utils
from sgl_kernel import hash_topk

device = utils.get_device()


def hash_topk_torch_native(
    router_logits: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor,
    num_fused_shared_experts: int,
    routed_scaling_factor: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scores = torch.nn.functional.softplus(router_logits).sqrt()

    num_token = scores.shape[0]
    num_routed_experts = scores.shape[1]
    topk_routed = tid2eid.shape[1]
    topk = topk_routed + num_fused_shared_experts

    topk_ids = torch.zeros((num_token, topk), dtype=torch.int32, device=scores.device)
    topk_weights = torch.zeros(
        (num_token, topk), dtype=scores.dtype, device=scores.device
    )

    if num_fused_shared_experts == 1:
        topk_ids[:, :-1] = tid2eid[input_ids]
        topk_weights[:, :-1] = scores.gather(1, topk_ids[:, :-1].long())
        topk_weights[:, :-1] /= topk_weights[:, :-1].sum(dim=-1, keepdim=True)

        topk_ids[:, -1] = torch.randint(
            low=num_routed_experts,
            high=num_routed_experts + num_fused_shared_experts,
            size=(num_token,),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )

        topk_weights[:, -1] = topk_weights[:, :-1].sum(dim=-1) / routed_scaling_factor
    else:
        # Torch native hash path currently routes non-shared case here.
        assert num_fused_shared_experts == 0
        topk_ids[:, :] = tid2eid[input_ids]
        topk_weights[:, :] = scores.gather(1, topk_ids[:, :].long())
        topk_weights[:, :] /= topk_weights[:, :].sum(dim=-1, keepdim=True)

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [1, 64, 1024, 4096])
@pytest.mark.parametrize("topk_routed", [6, 8])
@pytest.mark.parametrize("num_fused_shared_experts", [0, 1])
@pytest.mark.parametrize("num_routed_experts", [64, 256, 384])
def test_hash_topk(
    dtype, num_tokens, topk_routed, num_fused_shared_experts, num_routed_experts
):
    torch.manual_seed(1024)

    vocab_size = 1024
    routed_scaling_factor = 2.5

    router_logits = torch.randn(
        num_tokens, num_routed_experts, device=device, dtype=dtype
    )
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(num_tokens,),
        device=device,
        dtype=torch.int64,
    )
    tid2eid = torch.randint(
        low=0,
        high=num_routed_experts,
        size=(vocab_size, topk_routed),
        device=device,
        dtype=torch.int32,
    )

    ref_topk_weights, ref_topk_ids = hash_topk_torch_native(
        router_logits.float(),
        input_ids,
        tid2eid,
        num_fused_shared_experts,
        routed_scaling_factor,
    )

    out_topk_weights = torch.empty_like(ref_topk_weights)
    out_topk_ids = torch.empty_like(ref_topk_ids)

    hash_topk(
        router_logits,
        input_ids,
        tid2eid,
        out_topk_weights,
        out_topk_ids,
        routed_scaling_factor=routed_scaling_factor,
        scoring_func="sqrtsoftplus",
    )

    torch.testing.assert_close(out_topk_ids, ref_topk_ids)

    rtol = 1e-4 if dtype == torch.float32 else 1e-3
    atol = 1e-5 if dtype == torch.float32 else 1e-3

    torch.testing.assert_close(out_topk_weights, ref_topk_weights, rtol=rtol, atol=atol)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
