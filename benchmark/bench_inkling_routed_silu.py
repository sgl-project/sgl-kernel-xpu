"""Benchmark the routed SiLU path with an installed sgl-kernel wheel.

Run on both the main and optimized wheels with the same GPU and arguments.
The default (24576, 768) bf16 case is the Inkling routed batch.
"""

import argparse
import hashlib

import sgl_kernel  # noqa: F401 - imports common_ops and registers torch.ops
import torch
import triton.testing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=24576)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=25, help="warmup milliseconds")
    parser.add_argument("--rep", type=int, default=100, help="measurement milliseconds")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch.manual_seed(42)
    inp = torch.randn((args.rows, 2 * args.width), dtype=dtype, device="xpu")
    out = torch.empty((args.rows, args.width), dtype=dtype, device="xpu")
    reference = torch.nn.functional.silu(inp[:, : args.width].float())
    reference *= inp[:, args.width :].float()
    reference = reference.to(dtype)

    def run():
        torch.ops.sgl_kernel.silu_and_mul(out, inp)

    run()
    # Matches the corresponding CUDA activation correctness ceiling for BF16.
    tolerance = 1e-2 if dtype == torch.bfloat16 else 1e-3
    torch.testing.assert_close(out, reference, rtol=tolerance, atol=tolerance)
    max_abs = (out.float() - reference.float()).abs().max().item()
    output_sha256 = hashlib.sha256(
        out.cpu().view(torch.int16).numpy().tobytes()
    ).hexdigest()

    p50_ms, p20_ms, p80_ms = triton.testing.do_bench(
        run, warmup=args.warmup, rep=args.rep, quantiles=[0.5, 0.2, 0.8]
    )
    bytes_moved = args.rows * args.width * 3 * inp.element_size()
    effective_gbps = bytes_moved / (p50_ms * 1e6)
    print(f"shape=({args.rows}, {args.width}) dtype={args.dtype} device=xpu")
    print(f"max_abs_vs_rounded_fp32={max_abs:.6f} output_sha256={output_sha256}")
    print(f"p50_ms={p50_ms:.6f} p20_ms={p20_ms:.6f} p80_ms={p80_ms:.6f}")
    print(f"effective_gbps={effective_gbps:.2f}")


if __name__ == "__main__":
    main()
