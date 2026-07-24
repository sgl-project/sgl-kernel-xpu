import argparse
from dataclasses import dataclass

import torch

from sgl_kernel import rel_proj_small_t, row_compact_bf16, row_scale_bf16


@dataclass(frozen=True)
class RowCase:
    name: str
    rows: int
    inner: int
    stride: int


@dataclass(frozen=True)
class RelProjCase:
    name: str
    t: int
    h: int
    d: int
    e: int
    r_stride_t: int
    with_tau: bool


def _time_ms(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.xpu.synchronize()
    return start.elapsed_time(end) / iterations


def _row_cases(suite: str) -> list[RowCase]:
    if suite == "quick":
        return [
            RowCase("tail_inner_17", 37, 17, 25),
            RowCase("aligned_decode_r", 32, 256, 256),
            RowCase("strided_prefill_r", 1024, 256, 320),
        ]
    return [
        RowCase("prod_tp4_rop_t64", 64, 12 * 16, 12 * 128 + 2 * 128 + 12 * 16),
        RowCase("prod_tp8_rop_t2k", 2048, 6 * 16, 6 * 128 + 2 * 128 + 6 * 16),
        RowCase("prod_tp4_post_t2k", 2048, 12 * 1024, 12 * 1024),
    ]


def _rel_cases(suite: str) -> list[RelProjCase]:
    if suite == "quick":
        return [
            RelProjCase("tiny_t5", 5, 2, 16, 65, 48, True),
            RelProjCase("prod_t1_tau", 1, 16, 16, 1024, 256, True),
            RelProjCase("prod_t32_tau", 32, 16, 16, 1024, 256, True),
        ]
    return [
        RelProjCase(
            "prod_tp4_decode_t1", 1, 12, 16, 1024, 12 * 128 + 2 * 128 + 12 * 16, True
        ),
        RelProjCase(
            "prod_tp8_verify_t9", 9, 6, 16, 1024, 6 * 128 + 2 * 128 + 6 * 16, True
        ),
        RelProjCase(
            "prod_tp8_extend_t32", 32, 6, 16, 1024, 6 * 128 + 2 * 128 + 6 * 16, True
        ),
    ]


def bench_row(case: RowCase, op: str, warmup: int, iterations: int) -> None:
    x = torch.randn(case.rows, case.stride, device="xpu", dtype=torch.bfloat16)[
        :, : case.inner
    ]
    tau = 1.0 + 0.1 * torch.rand(case.rows, device="xpu", dtype=torch.float32)
    out = torch.empty((case.rows, case.inner), device="xpu", dtype=torch.bfloat16)

    if op == "scale":
        fn = lambda: row_scale_bf16(x, tau, out)
        bytes_moved = case.rows * case.inner * 2 * 2 + case.rows * 4
    else:
        fn = lambda: row_compact_bf16(x, out)
        bytes_moved = case.rows * case.inner * 2 * 2

    ms = _time_ms(fn, warmup, iterations)
    gbps = bytes_moved / (ms * 1.0e-3) / 1.0e9
    print(
        f"{op:7s} {case.name:24s} rows={case.rows:<5d} inner={case.inner:<6d} "
        f"stride={case.stride:<6d} time={ms * 1000.0:.3f} us bw={gbps:.2f} GB/s"
    )


def bench_rel(case: RelProjCase, warmup: int, iterations: int) -> None:
    packed = torch.randn(
        case.t, case.r_stride_t, device="xpu", dtype=torch.bfloat16
    )
    r = packed[:, : case.h * case.d].view(case.t, case.h, case.d)
    proj = torch.randn(case.d, case.e, device="xpu", dtype=torch.bfloat16)
    tau = (
        1.0 + 0.1 * torch.rand(case.t, device="xpu", dtype=torch.float32)
        if case.with_tau
        else None
    )
    out = torch.empty((case.t, case.h, case.e), device="xpu", dtype=torch.bfloat16)

    fn = lambda: rel_proj_small_t(r, proj, tau, out)
    ms = _time_ms(fn, warmup, iterations)
    flops = case.t * case.h * case.d * case.e * 2
    tops = flops / (ms * 1.0e-3) / 1.0e12
    print(
        f"relproj {case.name:24s} T={case.t:<3d} H={case.h:<3d} D={case.d:<3d} "
        f"E={case.e:<5d} tau={case.with_tau} time={ms * 1000.0:.3f} us "
        f"throughput={tops:.3f} TOPS"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["quick", "inkling"], default="quick")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    torch.xpu.set_device(0)
    for case in _row_cases(args.suite):
        bench_row(case, "scale", args.warmup, args.iterations)
        bench_row(case, "compact", args.warmup, args.iterations)
    for case in _rel_cases(args.suite):
        bench_rel(case, args.warmup, args.iterations)


if __name__ == "__main__":
    main()
