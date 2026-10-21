import torch


def mxfp4_small_permutation_xe35(
    topk_ids: torch.Tensor,
    padded_counts: torch.Tensor,
    perm_src: torch.Tensor,
    expanded_to_permuted: torch.Tensor,
    scale_map: torch.Tensor,
    alignment: int,
) -> None:
    """Build stable padded routing maps for at most 256 token/expert pairs.

    Expert IDs must be in [0, padded_counts.numel()). All tensors must be
    contiguous and on the same XPU device. Expert counts are powers of two
    up to 1024, alignment is in [1, 64], and outputs use int32. Output capacity
    must cover pairs + (alignment - 1) * min(experts, pairs), including slack.
    """
    torch.ops.sgl_kernel.mxfp4_small_permutation_xe35.default(
        topk_ids, padded_counts, perm_src, expanded_to_permuted, scale_map, alignment
    )
