"""Compare Xe2 QSA indexer scoring with the Torch reference."""

import argparse
import math
import time

import sgl_kernel  # noqa: F401
import torch


def time_ms(fn, repeats):
    for _ in range(3):
        fn()
    torch.xpu.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.xpu.synchronize()
    return 1000 * (time.perf_counter() - start) / repeats


def prefill_reference(q, k, starts, ends):
    scores = torch.einsum("rhd,cd->rch", q.float(), k[:, 0].float())
    logits = scores.relu().sum(-1) / math.sqrt(128)
    positions = torch.arange(k.shape[0], device=q.device)[None, :]
    return logits.masked_fill(
        (positions < starts[:, None]) | (positions >= ends[:, None]), -float("inf")
    )


def decode_reference(q, cache, pages, lengths, width):
    rows = q.shape[0]
    keys = cache[pages.long().reshape(-1), :, 0].reshape(rows, -1, 128)
    scores = torch.einsum("rhd,rcd->rch", q.float(), keys.float())
    logits = scores.relu().sum(-1)[:, :width] / math.sqrt(128)
    logits.masked_fill_(
        torch.arange(width, device=q.device)[None, :] >= lengths[:, None], -float("inf")
    )
    return logits


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill-rows", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--decode-rows", type=int, nargs="+", default=[1, 16, 64])
    parser.add_argument("--columns", type=int, nargs="+", default=[512, 1024, 4096])
    parser.add_argument("--repetitions", type=int, default=50)
    args = parser.parse_args(argv)
    if (
        any(x <= 0 for x in args.prefill_rows + args.decode_rows + args.columns)
        or args.repetitions <= 0
    ):
        parser.error(
            "prefill rows, decode rows, columns, and repetitions must be positive"
        )
    return args


def main():
    args = parse_args()
    if not torch.xpu.is_available():
        raise SystemExit("XPU required")

    torch.manual_seed(73)
    print("mode,rows,columns,torch_ms,xe2_ms,speedup")
    for rows in args.prefill_rows:
        q = torch.randn(rows, 4, 128, device="xpu", dtype=torch.bfloat16)
        for columns in args.columns:
            k = torch.randn(columns, 1, 128, device="xpu", dtype=torch.bfloat16)
            starts = torch.zeros(rows, device="xpu", dtype=torch.int32)
            ends = torch.full_like(starts, columns)

            def ref():
                return prefill_reference(q, k, starts, ends)

            def native():
                return torch.ops.sgl_kernel.qsa_mqa_prefill(q, k, starts, ends, None)

            ref_ms, xe2_ms = time_ms(ref, 3), time_ms(native, args.repetitions)
            print(
                f"prefill,{rows},{columns},{ref_ms:.3f},{xe2_ms:.3f},{ref_ms / xe2_ms:.2f}"
            )

    for rows in args.decode_rows:
        q = torch.randn(rows, 4, 128, device="xpu", dtype=torch.bfloat16)
        for columns in args.columns:
            page_size = 32
            pages_per_row = (columns + page_size - 1) // page_size
            cache = torch.randn(
                pages_per_row, page_size, 1, 128, device="xpu", dtype=torch.bfloat16
            )
            pages = (
                torch.arange(pages_per_row, device="xpu", dtype=torch.int32)
                .expand(rows, -1)
                .contiguous()
            )
            lengths = torch.full((rows,), columns, device="xpu", dtype=torch.int32)

            def ref():
                return decode_reference(q, cache, pages, lengths, columns)

            def native():
                return torch.ops.sgl_kernel.qsa_mqa_decode(
                    q, cache, pages, lengths, columns, None
                )

            ref_ms, xe2_ms = time_ms(ref, 3), time_ms(native, args.repetitions)
            print(
                f"decode,{rows},{columns},{ref_ms:.3f},{xe2_ms:.3f},{ref_ms / xe2_ms:.2f}"
            )


if __name__ == "__main__":
    main()
