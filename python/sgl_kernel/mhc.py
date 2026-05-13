from typing import Any, Dict, Optional

import torch


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
    pre: Optional[torch.Tensor] = None,
    post: Optional[torch.Tensor] = None,
    comb: Optional[torch.Tensor] = None,
):
    orig_shape = mixes.shape
    if mixes.dtype != torch.float32:
        raise TypeError(
            "hc_split_sinkhorn requires mixes to be float32, " f"got {mixes.dtype}"
        )
    if hc_mult <= 0:
        raise ValueError(f"hc_mult must be positive, got {hc_mult}")
    if hc_mult != 4:
        raise ValueError(
            "hc_split_sinkhorn currently supports only hc_mult=4 "
            "(kernel is specialized/unrolled for this value), "
            f"got hc_mult={hc_mult}"
        )
    if sinkhorn_iters != 20:
        raise ValueError(
            "hc_split_sinkhorn currently supports only sinkhorn_iters=20 "
            "(kernel is specialized/unrolled for this value), "
            f"got sinkhorn_iters={sinkhorn_iters}"
        )
    if mixes.ndim < 1:
        raise ValueError("mixes must have at least 1 dimension")

    col_size = (2 + hc_mult) * hc_mult
    if mixes.shape[-1] != col_size:
        raise ValueError(
            "Invalid mixes shape for hc_split_sinkhorn: "
            f"expected last dimension {col_size} for hc_mult={hc_mult}, "
            f"got {mixes.shape[-1]} (shape={tuple(mixes.shape)})"
        )
    if mixes.numel() % col_size != 0:
        raise ValueError(
            "mixes.numel() must be divisible by col_size in hc_split_sinkhorn: "
            f"numel={mixes.numel()}, col_size={col_size}"
        )

    T = mixes.numel() // col_size

    flat = mixes.view(T, col_size)
    hc_scale_c = hc_scale.to(device=mixes.device, dtype=torch.float32)
    hc_base_c = hc_base.to(device=mixes.device, dtype=torch.float32)

    pre_flat = (
        torch.empty((T, hc_mult), device=mixes.device, dtype=torch.float32)
        if pre is None
        else pre.to(device=mixes.device, dtype=torch.float32).view(T, hc_mult)
    )
    post_flat = (
        torch.empty((T, hc_mult), device=mixes.device, dtype=torch.float32)
        if post is None
        else post.to(device=mixes.device, dtype=torch.float32).view(T, hc_mult)
    )
    comb_flat = (
        torch.empty((T, hc_mult, hc_mult), device=mixes.device, dtype=torch.float32)
        if comb is None
        else comb.to(device=mixes.device, dtype=torch.float32).view(T, hc_mult, hc_mult)
    )

    torch.ops.sgl_kernel.hc_split_sinkhorn.default(
        flat,
        hc_scale_c,
        hc_base_c,
        pre_flat,
        post_flat,
        comb_flat,
        hc_mult,
        sinkhorn_iters,
        float(eps),
    )

    leading = orig_shape[:-1]
    return (
        pre_flat.view(*leading, hc_mult),
        post_flat.view(*leading, hc_mult),
        comb_flat.view(*leading, hc_mult, hc_mult),
    )
