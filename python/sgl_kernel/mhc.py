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


def hc_pre_big_fuse(
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    residual_flat: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    layer_input: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    n_splits: int = 1,
    rms_eps: float = 1e-5,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 2.0,
    norm_weight: Optional[torch.Tensor] = None,
    norm_eps: float = 1e-6,
):
    if hc_mult != 4:
        raise ValueError(
            f"hc_pre_big_fuse currently supports only hc_mult=4, got {hc_mult}"
        )
    if sinkhorn_iters != 20:
        raise ValueError(
            f"hc_pre_big_fuse currently supports only sinkhorn_iters=20, got {sinkhorn_iters}"
        )

    hc_scale_c = hc_scale.to(device=gemm_out_mul.device, dtype=torch.float32)
    hc_base_c = hc_base.to(device=gemm_out_mul.device, dtype=torch.float32)
    norm_weight_arg = (
        norm_weight.to(device=gemm_out_mul.device, dtype=torch.bfloat16)
        if norm_weight is not None
        else None
    )
    norm_eps_arg = float(norm_eps) if norm_weight is not None else None

    torch.ops.sgl_kernel.hc_pre_big_fuse.default(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale_c,
        hc_base_c,
        residual_flat,
        post_mix,
        comb_mix,
        layer_input,
        hc_mult,
        sinkhorn_iters,
        n_splits,
        float(rms_eps),
        float(hc_pre_eps),
        float(hc_sinkhorn_eps),
        float(hc_post_mult_value),
        norm_weight_arg,
        norm_eps_arg,
    )


def gemm_sqrsum(
    C: torch.Tensor,
    sqrsum: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
) -> None:
    """
    Compute C[m,n] = sum_k A[m,k]*B[n,k] and sqrsum[m] = sum_k A[m,k]^2.

    This is the mhc_pre GEMM+sqrsum stage: B is the weight matrix fn given as
    [N, K] (e.g. [24, 16384]), so C = A @ B^T. Leading singleton (n_splits=1)
    axes on any argument are squeezed away.

    Precision: when B is fp32 the kernel runs a tf32 x tf32 -> fp32 DPAS path
    (A widened to fp32, B taken as-is, both reinterpreted to tf32 at load). When
    A and B share a 16-bit dtype (half/bf16) the native DPAS path runs. C and
    sqrsum are always fp32.

    Args:
        C: Output tensor [M, N] fp32, filled with A @ B^T
        sqrsum: Output tensor [M] fp32, filled with row-wise squared sums of A
        A: Input tensor [M, K]   (bf16/fp16/fp32)
        B: Input tensor [N, K]   (fp32 for the production tf32 path)
    """
    torch.ops.sgl_kernel.gemm_sqrsum.default(C, sqrsum, A, B)
