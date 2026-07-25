import argparse
import json
import math
import os
import sys
import time

import torch

_dlopen_flags = sys.getdlopenflags()
if hasattr(os, "RTLD_LAZY") and hasattr(os, "RTLD_GLOBAL"):
    # This benchmark can be run from a prefill-only build that omits decode TUs.
    # Lazy binding avoids resolving unused decode runner symbols at import time.
    sys.setdlopenflags(os.RTLD_LAZY | os.RTLD_GLOBAL)
try:
    from sgl_kernel.flash_attn import flash_attn_with_kvcache
finally:
    sys.setdlopenflags(_dlopen_flags)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fixed paged FMHA prefill benchmark for the HD512 tile_o optimization."
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seqlen-q", type=int, default=512)
    parser.add_argument("--seqlen-k", type=int, default=4096)
    parser.add_argument("--heads-q", type=int, default=16)
    parser.add_argument("--heads-kv", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--head-dim-v", type=int, default=512)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify", type=int, default=1)
    parser.add_argument("--atol", type=float, default=0.08)
    parser.add_argument("--rtol", type=float, default=0.08)
    parser.add_argument("--label", default="current")
    return parser.parse_args()


def make_inputs(args):
    if args.heads_q % args.heads_kv != 0:
        raise ValueError("heads-q must be divisible by heads-kv")
    if args.seqlen_k % args.page_size != 0:
        raise ValueError("seqlen-k must be divisible by page-size for this fixed benchmark")

    torch.manual_seed(args.seed)
    device = torch.device("xpu")
    dtype = torch.bfloat16

    total_q = args.batch * args.seqlen_q
    pages_per_seq = args.seqlen_k // args.page_size
    num_pages = args.batch * pages_per_seq

    q = torch.randn(
        (total_q, args.heads_q, args.head_dim), device=device, dtype=dtype
    )
    k_cache = torch.randn(
        (num_pages, args.page_size, args.heads_kv, args.head_dim),
        device=device,
        dtype=dtype,
    )
    v_cache = torch.randn(
        (num_pages, args.page_size, args.heads_kv, args.head_dim_v),
        device=device,
        dtype=dtype,
    )
    page_table = torch.arange(num_pages, device=device, dtype=torch.int32).reshape(
        args.batch, pages_per_seq
    )
    cache_seqlens = torch.full(
        (args.batch,), args.seqlen_k, device=device, dtype=torch.int32
    )
    cu_seqlens_q = torch.arange(
        0,
        total_q + 1,
        step=args.seqlen_q,
        device=device,
        dtype=torch.int32,
    )
    out = torch.empty(
        (total_q, args.heads_q, args.head_dim_v), device=device, dtype=dtype
    )
    return q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, out


def run_kernel(args, tensors):
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, out = tensors
    return flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=args.seqlen_q,
        softmax_scale=1.0 / math.sqrt(args.head_dim),
        causal=True,
        window_size=(-1, -1),
        num_splits=1,
        out=out,
    )


def reference(args, tensors):
    q, k_cache, v_cache, page_table, _, _, _ = tensors
    group = args.heads_q // args.heads_kv
    refs = []
    for b in range(args.batch):
        q_b = q[b * args.seqlen_q : (b + 1) * args.seqlen_q].float()
        pages = page_table[b].to(torch.long)
        k_b = k_cache[pages].reshape(args.seqlen_k, args.heads_kv, args.head_dim)
        v_b = v_cache[pages].reshape(args.seqlen_k, args.heads_kv, args.head_dim_v)
        k_b = k_b.repeat_interleave(group, dim=1).float()
        v_b = v_b.repeat_interleave(group, dim=1).float()

        q_h = q_b.permute(1, 0, 2)
        k_h = k_b.permute(1, 0, 2)
        v_h = v_b.permute(1, 0, 2)
        scores = torch.matmul(q_h, k_h.transpose(-1, -2)) / math.sqrt(args.head_dim)

        rows = torch.arange(args.seqlen_q, device=q.device)[:, None]
        cols = torch.arange(args.seqlen_k, device=q.device)[None, :]
        mask = cols > (args.seqlen_k - args.seqlen_q + rows)
        scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))

        probs = torch.softmax(scores, dim=-1)
        refs.append(torch.matmul(probs, v_h).permute(1, 0, 2))
    return torch.cat(refs, dim=0).to(q.dtype)


def verify(args, tensors):
    out = run_kernel(args, tensors)
    torch.xpu.synchronize()
    ref = reference(args, tensors)
    torch.xpu.synchronize()

    diff = (out.float() - ref.float()).abs()
    tol = args.atol + args.rtol * ref.float().abs()
    bad = diff > tol
    max_abs = diff.max().item()
    max_rel = (diff / ref.float().abs().clamp_min(1e-12)).max().item()
    bad_count = bad.sum().item()
    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "bad_count": int(bad_count),
    }


def time_kernel(args, tensors):
    for _ in range(args.warmup):
        run_kernel(args, tensors)
    torch.xpu.synchronize()

    if hasattr(torch.xpu, "Event"):
        elapsed = []
        for _ in range(args.iters):
            start = torch.xpu.Event(enable_timing=True)
            end = torch.xpu.Event(enable_timing=True)
            start.record()
            run_kernel(args, tensors)
            end.record()
            end.synchronize()
            elapsed.append(start.elapsed_time(end))
        return sum(elapsed) / len(elapsed)

    start = time.perf_counter()
    for _ in range(args.iters):
        run_kernel(args, tensors)
    torch.xpu.synchronize()
    return (time.perf_counter() - start) * 1000.0 / args.iters


def main():
    args = parse_args()
    tensors = make_inputs(args)

    print(
        "shape: "
        f"batch={args.batch} sq={args.seqlen_q} sk={args.seqlen_k} "
        f"hq={args.heads_q} hkv={args.heads_kv} d={args.head_dim} "
        f"dv={args.head_dim_v} paged=1 page_size={args.page_size} causal=1"
    )

    result = {"label": args.label}
    if args.verify:
        verify_result = verify(args, tensors)
        result["verify"] = verify_result
        print(
            "verify: "
            f"max_abs={verify_result['max_abs']:.6g} "
            f"max_rel={verify_result['max_rel']:.6g} "
            f"bad={verify_result['bad_count']} "
            f"atol={args.atol} rtol={args.rtol}"
        )

    kernel_avg_ms = time_kernel(args, tensors)
    result["kernel_avg_ms"] = kernel_avg_ms
    result["iters"] = args.iters
    print(
        f"profile: label={args.label} kernel_avg_ms={kernel_avg_ms:.6f} "
        f"iters={args.iters}"
    )
    print("json:", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
