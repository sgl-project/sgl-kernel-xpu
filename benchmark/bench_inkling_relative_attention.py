import argparse
import math

import torch

from sgl_kernel import inkling_relative_attention


def _make_case(
    batch: int,
    seq_len: int,
    heads: int,
    kv_heads: int,
    d: int,
    dv: int,
    rel_len: int,
    *,
    decode: bool,
    dtype: torch.dtype,
):
    device = torch.device("xpu")
    q_len = 1 if decode else seq_len
    total_q = batch * q_len
    total_k = batch * seq_len
    q = (torch.randn((total_q, heads, d), device=device) * 0.25).to(dtype)
    k = (torch.randn((total_k, kv_heads, d), device=device) * 0.25).to(dtype)
    v = (torch.randn((total_k, kv_heads, dv), device=device) * 0.25).to(dtype)
    rel_bias = (
        torch.randn((total_q, heads, rel_len), device=device, dtype=torch.float32)
        * 0.05
        if rel_len > 0
        else None
    )
    q_to_seq = torch.arange(batch, device=device, dtype=torch.int32).repeat_interleave(q_len)
    q_base = seq_len - q_len
    q_pos = (
        torch.arange(q_len, device=device, dtype=torch.int32).repeat(batch) + q_base
    )
    cu_k = torch.arange(batch + 1, device=device, dtype=torch.int32) * seq_len
    return q, k, v, q_to_seq, q_pos, cu_k, rel_bias


def _time_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.xpu.synchronize()
    return start.elapsed_time(end) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--seq-len", type=int, default=640)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--dv", type=int, default=128)
    parser.add_argument("--rel-len", type=int, default=1024)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--window-left", type=int, default=-1)
    parser.add_argument("--window-right", type=int, default=-1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    q, k, v, q_to_seq, q_pos, cu_k, rel_bias = _make_case(
        args.batch,
        args.seq_len,
        args.heads,
        args.kv_heads,
        args.d,
        args.dv,
        args.rel_len,
        decode=args.decode,
        dtype=dtype,
    )
    out = torch.empty((q.shape[0], q.shape[1], v.shape[2]), dtype=dtype, device="xpu")
    scale = 1.0 / math.sqrt(args.d)

    def run():
        inkling_relative_attention(
            q,
            k,
            v,
            q_to_seq,
            q_pos,
            cu_k,
            rel_bias=rel_bias,
            softmax_scale=scale,
            causal=True,
            window_size=(args.window_left, args.window_right),
            out=out,
        )

    ms = _time_ms(run, args.warmup, args.iters)
    pairs_per_q = args.seq_len if args.window_left < 0 else min(args.seq_len, args.window_left + 1)
    valid_pairs = q.shape[0] * pairs_per_q * args.heads
    bytes_moved = valid_pairs * (2 * args.d + args.dv) * q.element_size()
    if rel_bias is not None:
        bytes_moved += valid_pairs * 4
    gbps = bytes_moved / (ms * 1.0e-3) / 1.0e9
    print(
        f"inkling_relative_attention dtype={args.dtype} decode={args.decode} "
        f"B={args.batch} Tq={q.shape[0]} Tk={k.shape[0]} H={args.heads}/{args.kv_heads} "
        f"D={args.d} Dv={args.dv} rel={args.rel_len} window=({args.window_left},{args.window_right}) "
        f"{ms * 1000.0:.3f} us {gbps:.2f} GB/s"
    )


if __name__ == "__main__":
    main()
