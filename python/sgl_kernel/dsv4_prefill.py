from typing import Optional

import torch


def dsv4_expand_prefill_causally_out(
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    extend_start_loc: Optional[torch.Tensor],
    seq_lens_causal: torch.Tensor,
    req_pool_indices_repeated: torch.Tensor,
    num_tokens: int,
    padded_num_tokens: int,
) -> None:
    torch.ops.sgl_kernel.dsv4_expand_prefill_causally_out(
        req_pool_indices,
        seq_lens,
        extend_seq_lens,
        extend_start_loc,
        seq_lens_causal,
        req_pool_indices_repeated,
        num_tokens,
        padded_num_tokens,
    )
