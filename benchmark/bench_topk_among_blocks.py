import argparse
import itertools

import torch
import triton
from sgl_kernel import topk_among_blocks

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _torch_topk_among_blocks(
    scores: torch.Tensor,
    blocks: torch.Tensor,
    topk: int,
    block_size: int,
) -> torch.Tensor:
    batch_size = scores.shape[0]
    offsets = torch.arange(block_size, device=scores.device)
    positions = (blocks.unsqueeze(-1) * block_size + offsets).reshape(batch_size, -1)
    candidate_scores = scores.float().gather(1, positions)
    selected = torch.topk(candidate_scores, topk, dim=-1, sorted=False).indices
    return positions.gather(1, selected)


def run_benchmarks(args: argparse.Namespace) -> None:
    if not torch.xpu.is_available():
        raise RuntimeError("XPU is required for bench_topk_among_blocks.py")

    print("batch,width,block_count,block_size,topk,dtype,kernel_us,torch_us,speedup")
    for batch_size, width, block_count, topk, dtype_name in itertools.product(
        args.batch_sizes, args.widths, args.block_counts, args.topks, args.dtypes
    ):
        if width % args.block_size:
            raise ValueError("width must be divisible by block_size")
        total_blocks = width // args.block_size
        if block_count > total_blocks:
            raise ValueError("block_count cannot exceed width // block_size")
        if topk > block_count * args.block_size:
            raise ValueError("topk cannot exceed the number of candidate positions")

        dtype = DTYPES[dtype_name]
        scores = torch.randn(batch_size, width, dtype=dtype, device="xpu")
        blocks = torch.stack(
            [torch.randperm(total_blocks)[:block_count] for _ in range(batch_size)]
        ).to(device="xpu", dtype=torch.int64)
        lengths = torch.full((batch_size,), width, dtype=torch.int64, device="xpu")

        def run_kernel() -> torch.Tensor:
            return topk_among_blocks(scores, lengths, blocks, topk, args.block_size)

        def run_torch() -> torch.Tensor:
            return _torch_topk_among_blocks(scores, blocks, topk, args.block_size)

        kernel_result = run_kernel()
        torch_result = run_torch()
        torch.testing.assert_close(
            torch.sort(kernel_result, dim=-1).values,
            torch.sort(torch_result, dim=-1).values,
        )
        torch.xpu.synchronize()

        for _ in range(args.warmup):
            run_kernel()
            run_torch()
        torch.xpu.synchronize()
        kernel_ms, _, _ = triton.testing.do_bench(
            run_kernel, quantiles=[0.5, 0.2, 0.8], warmup=args.warmup, rep=args.rep
        )
        torch_ms, _, _ = triton.testing.do_bench(
            run_torch, quantiles=[0.5, 0.2, 0.8], warmup=args.warmup, rep=args.rep
        )
        print(
            f"{batch_size},{width},{block_count},{args.block_size},{topk},"
            f"{dtype_name},{kernel_ms * 1000:.2f},{torch_ms * 1000:.2f},"
            f"{torch_ms / kernel_ms:.2f}x"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[32, 256])
    parser.add_argument("--widths", nargs="+", type=int, default=[4096, 16384])
    parser.add_argument("--block-counts", nargs="+", type=int, default=[8, 32])
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--topks", nargs="+", type=int, default=[32, 128])
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=["float32"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=100)
    run_benchmarks(parser.parse_args())
