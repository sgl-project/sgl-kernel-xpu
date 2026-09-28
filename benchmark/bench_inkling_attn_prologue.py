import argparse

import torch
import triton

from sgl_kernel.inkling_attn_prologue import inkling_attn_prologue_verify


def _dtype(name: str) -> torch.dtype:
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[name]


def _make_case(
    batch_size: int,
    draft_tokens: int,
    dq: int,
    dkv: int,
    width: int,
    dtype: torch.dtype,
):
    token_count = batch_size * draft_tokens
    q_off = 0
    k_off = dq
    v_off = dq + dkv
    qkvr = torch.randn(
        (token_count, dq + 2 * dkv), device="xpu", dtype=dtype
    )
    k_cache = torch.randn(
        (batch_size, width - 1, dkv), device="xpu", dtype=dtype
    )
    v_cache = torch.randn_like(k_cache)
    cache_indices = torch.arange(
        batch_size, device="xpu", dtype=torch.int32
    )
    cache_mask = torch.ones(batch_size, device="xpu", dtype=torch.bool)
    k_weight = torch.randn((dkv, width), device="xpu", dtype=dtype)
    v_weight = torch.randn_like(k_weight)
    k_inter = torch.empty(
        (batch_size, draft_tokens, width - 1, dkv), device="xpu", dtype=dtype
    )
    v_inter = torch.empty_like(k_inter)
    q_gamma = torch.ones(128, device="xpu", dtype=dtype)
    k_gamma = torch.ones(128, device="xpu", dtype=dtype)
    loc = torch.arange(token_count, device="xpu", dtype=torch.int64)
    k_buf = torch.empty(
        (token_count, dkv // 128, 128), device="xpu", dtype=dtype
    )
    v_buf = torch.empty_like(k_buf)

    def run():
        inkling_attn_prologue_verify(
            qkvr[:, :dq],
            k_cache,
            v_cache,
            cache_indices,
            cache_mask,
            k_weight,
            v_weight,
            k_inter,
            v_inter,
            q_gamma,
            k_gamma,
            1e-5,
            loc,
            k_buf,
            v_buf,
            q_off,
            k_off,
            v_off,
            dq,
            dkv,
            draft_tokens,
            activation="silu",
        )

    bytes_moved = (
        qkvr.numel()
        + k_cache.numel()
        + v_cache.numel()
        + k_inter.numel()
        + v_inter.numel()
        + k_buf.numel()
        + v_buf.numel()
    ) * qkvr.element_size()
    return run, bytes_moved


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Device-time benchmark for Inkling attention prologue verify."
    )
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--draft-tokens", type=int, default=8)
    parser.add_argument("--dq", type=int, default=384)
    parser.add_argument("--dkv", type=int, default=256)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise RuntimeError("XPU device is required")
    if args.dq % 128 or args.dkv % 128:
        raise ValueError("--dq and --dkv must be multiples of 128")
    if args.width < 2 or args.batch_size <= 0 or args.draft_tokens <= 0:
        raise ValueError("invalid benchmark shape")

    run, bytes_moved = _make_case(
        args.batch_size,
        args.draft_tokens,
        args.dq,
        args.dkv,
        args.width,
        _dtype(args.dtype),
    )
    median_ms, p20_ms, p80_ms = triton.testing.do_bench(
        run,
        warmup=args.warmup,
        rep=args.iters,
        quantiles=[0.5, 0.2, 0.8],
    )
    gbps = bytes_moved / (median_ms * 1.0e6)
    print(
        f"verify dtype={args.dtype} B={args.batch_size} draft={args.draft_tokens} "
        f"dq={args.dq} dkv={args.dkv} W={args.width} "
        f"median={median_ms:.4f} ms p20={p20_ms:.4f} ms p80={p80_ms:.4f} ms "
        f"effective={gbps:.2f} GB/s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
