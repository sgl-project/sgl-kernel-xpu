"""Compare Xe2 BF16 QSA attention to the SGLang torch reference path."""

import argparse
import sys
import time

import sgl_kernel  # noqa: F401
import torch


def reference(q, k, v, slots, scale):
    result = []
    repeats = q.shape[1] // k.shape[1]
    for row in range(q.shape[0]):
        valid = slots[row] >= 0
        indices = slots[row, valid].long()
        if indices.numel() == 0:
            result.append(torch.zeros_like(q[row]))
            continue
        keys = k.index_select(0, indices).repeat_interleave(repeats, dim=1)
        values = v.index_select(0, indices).repeat_interleave(repeats, dim=1)
        scores = torch.einsum("hd,khd->hk", q[row].float(), keys.float()) * scale
        result.append(
            torch.einsum("hk,khd->hd", scores.softmax(dim=-1), values.float()).to(
                q.dtype
            )
        )
    return torch.stack(result)


def time_ms(fn, repetitions):
    for _ in range(3):
        fn()
    torch.xpu.synchronize()
    start = time.perf_counter()
    for _ in range(repetitions):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - start) * 1000 / repetitions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 16, 128, 1024, 4096])
    parser.add_argument("--topk", type=int, nargs="+", default=[64, 512, 2051])
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--skip-reference", action="store_true")
    args = parser.parse_args()
    if any(rows <= 0 for rows in args.rows):
        parser.error("--rows must contain positive integers")
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")
    if not torch.xpu.is_available():
        sys.exit("XPU required")
    torch.manual_seed(42)
    k = torch.randn(4096, 2, 256, device="xpu", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    print("rows,topk,reference_ms,qsa_ms,useful_tflops,logical_kv_gbps")
    for rows in args.rows:
        q = torch.randn(rows, 24, 256, device="xpu", dtype=torch.bfloat16)
        for topk in args.topk:
            slots = torch.randint(
                0, k.size(0), (rows, topk), device="xpu", dtype=torch.int32
            )
            ref = lambda: reference(q, k, v, slots, 0.0625)
            kernel = lambda: sgl_kernel.qsa_sparse_attention(q, k, v, slots, 0.0625)
            reference_ms = time_ms(ref, 5) if not args.skip_reference else float("nan")
            kernel_ms = time_ms(kernel, args.repetitions)
            flops = rows * q.size(1) * topk * q.size(2) * 4
            logical_kv_bytes = rows * q.size(1) * topk * q.size(2) * 2 * 2
            print(
                f"{rows},{topk},{reference_ms:.3f},{kernel_ms:.3f},"
                f"{flops / (kernel_ms * 1e9):.3f},"
                f"{logical_kv_bytes / (kernel_ms * 1e6):.1f}"
            )


if __name__ == "__main__":
    main()
