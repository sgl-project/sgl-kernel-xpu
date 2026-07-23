import argparse

import torch

from sgl_kernel import (
    inkling_moe_gate_gemv,
    inkling_moe_gate_gemv_fused,
    inkling_moe_gate_topk_renorm,
)

HIDDEN = 6144
N_ROUTED = 256
N_SHARED = 2
N_TOTAL = N_ROUTED + N_SHARED
N_PADDED = 264
TOPK = 6
ROUTE_SCALE = 8.0


def _make_inputs(tokens: int):
    device = torch.device("xpu")
    x = (torch.randn((tokens, HIDDEN), device=device) * 0.05).to(torch.bfloat16)
    weight = (torch.randn((N_PADDED, HIDDEN), device=device) * 0.02).to(
        torch.bfloat16
    )
    weight[N_TOTAL:].zero_()
    bias = torch.randn((N_ROUTED,), dtype=torch.float32, device=device) * 0.1
    global_scale = torch.tensor([1.25], dtype=torch.float32, device=device)
    logits = torch.empty((tokens, N_PADDED), dtype=torch.float32, device=device)
    if tokens > 0:
        logits[:, :N_TOTAL] = torch.mm(x.float(), weight.float().T)[:, :N_TOTAL]
    return x, weight, bias, global_scale, logits[:, :N_TOTAL]


def _bench(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--packed", action="store_true")
    args = parser.parse_args()

    for tokens in args.tokens:
        x, weight, bias, global_scale, logits = _make_inputs(tokens)
        topk_ms = _bench(
            lambda: inkling_moe_gate_topk_renorm(
                logits, bias, global_scale, ROUTE_SCALE, return_packed=args.packed
            ),
            args.warmup,
            args.iters,
        )
        gemv_ms = _bench(
            lambda: inkling_moe_gate_gemv(x, weight), args.warmup, args.iters
        )
        if tokens <= 64:
            fused_ms = _bench(
                lambda: inkling_moe_gate_gemv_fused(
                    x,
                    weight,
                    bias,
                    global_scale,
                    ROUTE_SCALE,
                    return_packed=args.packed,
                ),
                args.warmup,
                args.iters,
            )
            fused_text = f" fused_ms={fused_ms:.4f}"
        else:
            fused_text = ""
        gemv_flops = 2 * tokens * N_TOTAL * HIDDEN
        gemv_tflops = gemv_flops / (gemv_ms / 1e3) / 1e12 if gemv_ms > 0 else 0.0
        print(
            f"tokens={tokens} packed={args.packed} topk_ms={topk_ms:.4f} "
            f"gemv_ms={gemv_ms:.4f} gemv_tflops={gemv_tflops:.4f}{fused_text}"
        )


if __name__ == "__main__":
    main()
