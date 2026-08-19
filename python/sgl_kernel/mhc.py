from typing import Optional

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


def hc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(residual)
    torch.ops.sgl_kernel.hc_post.default(x, residual, post_layer_mix, comb_res_mix, out)
    return out


def hc_pre_gemm_sqr_sum(
    C: torch.Tensor,
    sqr_sum: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
) -> None:
    torch.ops.sgl_kernel.hc_pre_gemm_sqr_sum.default(C, sqr_sum, A, B)


def _mhc_pre_n_splits_pre(num_tokens: int) -> int:
    return 32 if num_tokens <= 2048 else 1


def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 2.0,
    sinkhorn_repeat: int = 20,
    n_splits: Optional[int] = None,
    n_splits_pre: Optional[int] = None,
    norm_weight: Optional[torch.Tensor] = None,
    norm_eps: float = 1e-6,
):
    if residual.dim() != 3:
        raise ValueError(
            f"residual must be [T, hc_mult, D] (3 dimensions), got {residual.dim()} "
            f"(shape={tuple(residual.shape)})"
        )
    num_tokens = residual.size(0)
    hc_mult = residual.size(1)
    hidden_size = residual.size(2)
    if hc_mult != 4:
        raise ValueError(f"mhc_pre currently supports only hc_mult=4, got {hc_mult}")
    hc_hidden = hc_mult * hidden_size
    hc_mult3 = (2 + hc_mult) * hc_mult

    if fn.size(0) != hc_mult3 or fn.size(1) != hc_hidden:
        raise ValueError(f"fn must be [{hc_mult3}, {hc_hidden}], got {tuple(fn.shape)}")

    if n_splits is not None and n_splits != 1:
        raise ValueError(f"mhc_pre requires n_splits==1, got {n_splits}")
    if n_splits_pre is None:
        n_splits_pre = _mhc_pre_n_splits_pre(num_tokens)

    device = residual.device

    A = residual.reshape(num_tokens, hc_hidden)

    gemm_out_mul = torch.empty(
        n_splits_pre, num_tokens, hc_mult3, dtype=torch.float32, device=device
    )
    gemm_out_sqrsum = torch.empty(
        n_splits_pre, num_tokens, dtype=torch.float32, device=device
    )
    hc_pre_gemm_sqr_sum(gemm_out_mul, gemm_out_sqrsum, A, fn)

    post_mix = torch.empty(num_tokens, hc_mult, dtype=torch.float32, device=device)
    comb_mix = torch.empty(
        num_tokens, hc_mult, hc_mult, dtype=torch.float32, device=device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=device
    )

    hc_pre_big_fuse(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual,
        post_mix,
        comb_mix,
        layer_input,
        hc_mult=hc_mult,
        sinkhorn_iters=sinkhorn_repeat,
        n_splits=n_splits_pre,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )

    return post_mix, comb_mix, layer_input
