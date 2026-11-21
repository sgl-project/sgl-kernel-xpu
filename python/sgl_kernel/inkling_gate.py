"""Inkling MoE gate epilogue: sigmoid + bias + top-k + logsigmoid renorm.

    sel = sigmoid(logits[:, :N]) + bias             # ranking score only
    idx = topk(sel, k)                              # lowest expert id wins ties
    w   = logsigmoid_norm(logits[idx] ++ logits[:, N:]) * route_scale * global_scale

`logits` is [M, N + S]: N routed experts followed by S shared-expert sink columns.
The sinks take no bias and never enter the top-k, but do join the normalizer, and
the normalized quantity is the RAW logit at the winner rather than its ranking
score -- so the kernel carries two quantities per column through the top-k.

Drop-in for sglang's `sigmoid_gate_topk_renorm` on XPU: same arguments, same
return tuple, same output dtypes. The top-k is k masked-max passes in registers
rather than `tl.topk` / `tl.bitonic_merge`, whose bitonic sort over every routed
column is what makes the sort-based kernel slow on Xe.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _inkling_gate_topk_renorm_kernel(
    logits_ptr,  # [M, N + S], column stride 1
    bias_ptr,  # [N]
    global_scale_ptr,  # [1]
    routed_w_ptr,  # [M, K] (unless RETURN_PACKED)
    indices_ptr,  # [M, K] int32 (unless RETURN_PACKED)
    packed_ptr,  # [M, K] int32 (RETURN_PACKED only)
    shared_w_ptr,  # [M, S]
    route_scale,
    M,
    stride_lm,
    N: tl.constexpr,
    K: tl.constexpr,
    S: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,  # >= N, power of 2
    BLOCK_A: tl.constexpr,  # >= K + S, power of 2
    RETURN_PACKED: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    row_ptr = logits_ptr + offs_m[:, None] * stride_lm

    raw = tl.load(
        row_ptr + offs_n[None, :], mask=mask_m[:, None] & mask_n[None, :], other=0.0
    ).to(tl.float32)
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    sel = tl.sigmoid(raw) + bias[None, :]
    sel = tl.where(sel == sel, sel, -1e30)  # NaN never outranks a finite score
    sel = tl.where(mask_n[None, :], sel, float("-inf"))

    # Slots 0 .. K hold the routed winners in descending rank order, slots
    # K .. K + S the sinks; all of them feed one normalizer.
    offs_a = tl.arange(0, BLOCK_A)
    active = tl.zeros([BLOCK_M, BLOCK_A], dtype=tl.float32)
    idx = tl.zeros([BLOCK_M, BLOCK_A], dtype=tl.int32)

    cur = sel
    remaining = tl.broadcast_to(mask_n[None, :], (BLOCK_M, BLOCK_N))
    for k in tl.static_range(K):
        max_val = tl.max(cur, axis=1)[:, None]
        lane = tl.where(remaining & (cur == max_val), offs_n[None, :], N + 1)
        win = tl.min(lane, axis=1)[:, None].to(tl.int32)
        is_win = offs_n[None, :] == win
        # Carry the RAW logit, not the score the top-k ranked on.
        win_raw = tl.sum(tl.where(is_win, raw, 0.0), axis=1)[:, None]
        slot = offs_a[None, :] == k
        active = tl.where(slot, win_raw, active)
        idx = tl.where(slot, win, idx)
        remaining = remaining & ~is_win
        cur = tl.where(remaining, cur, float("-inf"))

    offs_s = offs_a - K
    mask_s = (offs_s >= 0) & (offs_s < S)
    sink = tl.load(
        row_ptr + N + offs_s[None, :],
        mask=mask_m[:, None] & mask_s[None, :],
        other=0.0,
    ).to(tl.float32)
    active = tl.where(mask_s[None, :], sink, active)

    # exp(lp - logsumexp(lp)), lp = logsigmoid(x). Algebraically
    # sigmoid(x) / sum(sigmoid(x)), but fp32 sigmoid flushes to 0 below x ~ -104,
    # so the explicit form divides 0/0 once every active logit is that negative.
    # logsigmoid(x) = min(x, 0) - log1p(exp(-|x|)), exact for large |x|.
    mask_a = offs_a < K + S
    lp = tl.minimum(active, 0.0) - tl.log(1.0 + tl.exp(-tl.abs(active)))
    lp = tl.where(mask_a[None, :], lp, float("-inf"))
    e = tl.where(mask_a[None, :], tl.exp(lp - tl.max(lp, axis=1)[:, None]), 0.0)
    w = e / tl.sum(e, axis=1, keep_dims=True)
    w = w * (route_scale * tl.load(global_scale_ptr).to(tl.float32))

    mask_k = offs_a < K
    store_k = mask_m[:, None] & mask_k[None, :]
    offs_mk = offs_m[:, None] * K + offs_a[None, :]
    if RETURN_PACKED:
        # (id << 16) | bf16_bits(weight), bitwise identical to fused_pack_topk.
        w_bits = w.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32)
        tl.store(packed_ptr + offs_mk, (idx << 16) | (w_bits & 0xFFFF), mask=store_k)
    else:
        tl.store(
            routed_w_ptr + offs_mk,
            w.to(routed_w_ptr.dtype.element_ty),
            mask=store_k,
        )
        tl.store(indices_ptr + offs_mk, idx, mask=store_k)
    tl.store(
        shared_w_ptr + offs_m[:, None] * S + offs_s[None, :],
        w.to(shared_w_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_s[None, :],
    )


def inkling_gate_topk_renorm(
    logits: torch.Tensor,
    k: int,
    n_shared_experts: int,
    route_scale: float,
    global_scale: torch.Tensor,
    bias: torch.Tensor,
    *,
    return_packed_topk: bool = False,
) -> Tuple[
    Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]
]:
    """Fused Inkling gate top-k + logsigmoid renorm.

    `logits` is [M, N + n_shared_experts] and needs only column stride 1 (Inkling's
    gate logits are a [T, 258] slice of a padded [T, 264] GEMM output). Returns
    (routed_weights [M, k], topk_indices [M, k] int32, shared_weights [M, S],
    packed_topk [M, k] int32); weights are in `logits.dtype`. With
    `return_packed_topk` only the packed tensor is written and the routed
    weights / indices are None, otherwise packed_topk is None.
    """
    assert (
        logits.ndim == 2 and logits.stride(1) == 1
    ), f"{logits.shape=} {logits.stride()=}"
    assert (
        logits.shape[0] * logits.stride(0) <= 2**31
    ), f"assumes int32 indexing: {logits.stride()=}"
    assert 0 < k <= 32, f"{k=} must be in (0, 32]"
    assert n_shared_experts > 0, f"{n_shared_experts=} must be positive"
    M, G = logits.shape
    N = G - n_shared_experts
    assert k <= N, f"{k=} exceeds {N} routed experts"
    assert bias.numel() == N and bias.stride(-1) == 1, f"{bias.shape=} expected [{N}]"
    assert global_scale.numel() == 1, f"{global_scale.shape=} expected [1]"

    dev = logits.device
    shared_w = torch.empty((M, n_shared_experts), dtype=logits.dtype, device=dev)
    if return_packed_topk:
        packed = torch.empty((M, k), dtype=torch.int32, device=dev)
        routed_w = indices = None
        routed_w_arg = indices_arg = packed
    else:
        routed_w = torch.empty((M, k), dtype=logits.dtype, device=dev)
        indices = torch.empty((M, k), dtype=torch.int32, device=dev)
        packed = None
        routed_w_arg, indices_arg = routed_w, indices
    if M == 0:
        return routed_w, indices, shared_w, packed

    BLOCK_N = triton.next_power_of_2(N)
    # One warp per program keeps the k per-row reductions cheap; pack a few rows
    # per program only when rows are narrow, so small launches stay occupied.
    BLOCK_M = max(1, min(4, 256 // BLOCK_N))
    grid = (triton.cdiv(M, BLOCK_M),)
    _inkling_gate_topk_renorm_kernel[grid](
        logits,
        bias,
        global_scale,
        routed_w_arg,
        indices_arg,
        packed if packed is not None else indices_arg,
        shared_w,
        float(route_scale),
        M,
        logits.stride(0),
        N=N,
        K=k,
        S=n_shared_experts,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_A=triton.next_power_of_2(k + n_shared_experts),
        RETURN_PACKED=return_packed_topk,
        num_warps=1 if BLOCK_N <= 512 else 4,
    )
    return routed_w, indices, shared_w, packed
