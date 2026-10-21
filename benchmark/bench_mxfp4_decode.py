#!/usr/bin/env python3
"""Benchmark Muse Spark B=8 MXFP4 MoE GEMMs through the production op."""

from __future__ import annotations

import argparse
import json
import statistics
import time



BATCH = 8
TOPK = 8
EXPERTS = 256
ALIGNMENT = 4
HIDDEN = 4096
INTERMEDIATE = 4096
UPPER_BOUND_ROWS = BATCH * TOPK + (ALIGNMENT - 1) * BATCH * TOPK


def make_padded_counts_xe35(seed: int, active_experts: int | None) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    if active_experts is not None:
        if not 1 <= active_experts <= BATCH * TOPK:
            raise ValueError(f"active_experts must be in [1, {BATCH * TOPK}]")
        experts = torch.randperm(EXPERTS, generator=generator)[:active_experts]
        routed = experts[torch.arange(BATCH * TOPK, dtype=torch.int64) % active_experts]
        counts = torch.bincount(routed, minlength=EXPERTS).to(torch.int32)
        return (
            torch.div(counts + ALIGNMENT - 1, ALIGNMENT, rounding_mode="floor")
            * ALIGNMENT
        )

    # top-k never selects the same expert twice for one token.
    routed = torch.stack(
        [torch.randperm(EXPERTS, generator=generator)[:TOPK] for _ in range(BATCH)]
    )
    counts = torch.bincount(routed.flatten(), minlength=EXPERTS).to(torch.int32)
    return (
        torch.div(counts + ALIGNMENT - 1, ALIGNMENT, rounding_mode="floor") * ALIGNMENT
    )


def make_sweep_counts_xe35(total_rows: int, experts: int = EXPERTS) -> torch.Tensor:
    if total_rows <= 0 or total_rows % ALIGNMENT:
        raise ValueError("sweep rows must be a positive multiple of 4")
    if experts <= 0:
        raise ValueError("experts must be positive")
    quads = total_rows // ALIGNMENT
    counts = torch.full((experts,), quads // experts, dtype=torch.int32)
    counts[: quads % experts] += 1
    return counts * ALIGNMENT


def event_samples_xe35(launch, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        launch()
    torch.xpu.synchronize()
    starts = [torch.xpu.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.xpu.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        start.record()
        launch()
        end.record()
    torch.xpu.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends)]


def logical_bytes_xe35(active_experts: int, padded_rows: int, n: int, k: int) -> int:
    a = padded_rows * (k // 2 + k // 32)
    weights = active_experts * n * (k // 2 + k // 32)
    output = padded_rows * n * 2
    return a + weights + output


def benchmark_projection_xe35(
    name: str,
    n: int,
    k: int,
    counts: torch.Tensor,
    warmup: int,
    iterations: int,
    allocated_rows: int,
) -> None:
    device = "xpu"
    experts = counts.numel()
    active_experts = int((counts != 0).sum())
    padded_rows = int(counts.sum())
    rows = counts.to(device)

    a = torch.randint(
        0, 256, (allocated_rows, k // 2), dtype=torch.uint8, device=device
    )
    a_scale = torch.randint(
        112, 143, (allocated_rows, k // 32), dtype=torch.uint8, device=device
    )
    weights = torch.randint(
        0, 256, (experts, n, k // 2), dtype=torch.uint8, device=device
    )
    weight_scales = torch.randint(
        112, 143, (experts, k // 32, n), dtype=torch.uint8, device=device
    )
    output = torch.empty((allocated_rows, n), dtype=torch.bfloat16, device=device)

    def launch():
        moe_grouped_mm_nt_mxfp4_w4a4_xe35(
            output,
            a,
            a_scale,
            weights,
            weight_scales,
            rows,
            n,
            k,
            experts,
        )

    launch()
    torch.xpu.synchronize()
    verify_grouped_output_xe35(
        output,
        a,
        a_scale,
        weights,
        weight_scales,
        counts,
        max_rows=None,
        atol=5e-3,
        rtol=5e-2,
    )
    bytes_per_call = logical_bytes_xe35(active_experts, padded_rows, n, k)
    samples = event_samples_xe35(launch, warmup, iterations)
    median_ms = statistics.median(samples)
    bandwidth = bytes_per_call / (median_ms * 1e6)
    print(
        json.dumps(
            {
                "projection": name,
                "policy": "auto",
                "active_experts": active_experts,
                "experts": experts,
                "padded_rows": padded_rows,
                "allocated_rows": allocated_rows,
                "n": n,
                "k": k,
                "median_ms": median_ms,
                "min_ms": min(samples),
                "max_ms": max(samples),
                "p20_ms": sorted(samples)[int(0.2 * (len(samples) - 1))],
                "p80_ms": sorted(samples)[int(0.8 * (len(samples) - 1))],
                "logical_GBs": bandwidth,
                "samples_ms": samples,
            }
        ),
        flush=True,
    )

    del a, a_scale, weights, weight_scales, output, rows
    torch.xpu.synchronize()


def run_comparison_case_xe35(
    n: int, allocated_rows: int, warmup: int, iterations: int, rounds: int
):
    k = 4096
    rows_cpu = (
        make_padded_counts_xe35(0, None)
        if allocated_rows == UPPER_BOUND_ROWS
        else make_sweep_counts_xe35(allocated_rows)
    )
    used = int(rows_cpu.sum())
    rows = rows_cpu.to("xpu")
    offsets = torch.cat(
        (torch.zeros(1, dtype=torch.int32), rows_cpu.cumsum(0, dtype=torch.int32)[:-1])
    ).to("xpu")
    problems = torch.stack(
        (rows_cpu, torch.full_like(rows_cpu, n), torch.full_like(rows_cpu, k)), 1
    ).to("xpu")
    a = torch.randint(0, 256, (allocated_rows, k // 2), dtype=torch.uint8, device="xpu")
    sa_rm = torch.randint(
        124, 129, (allocated_rows, k // 32), dtype=torch.uint8, device="xpu"
    )
    b = torch.randint(0, 256, (EXPERTS, n, k // 2), dtype=torch.uint8, device="xpu")
    sb_rm = torch.randint(
        124, 129, (EXPERTS, n, k // 32), dtype=torch.uint8, device="xpu"
    )
    sa_mn = torch.zeros_like(sa_rm)
    mxfp4_moe_scale_to_mn_major_xe35(sa_mn, sa_rm, rows)
    # The legacy runner checks for a contiguous 3D expert dimension; explicit
    # pointer arrays carry the ragged offsets and strides of this flat buffer.
    assert allocated_rows % EXPERTS == 0
    sa_legacy = sa_mn.view(EXPERTS, k // 32, allocated_rows // EXPERTS)
    sb_mn = sb_rm.transpose(1, 2).contiguous()
    out_branch = torch.empty((allocated_rows, n), dtype=torch.bfloat16, device="xpu")
    out_main = torch.empty((allocated_rows, n), dtype=torch.float32, device="xpu")
    workspace = torch.empty((64 * 1024 * 1024,), dtype=torch.uint8, device="xpu")
    empty_ptrs = torch.empty((0,), dtype=torch.int64, device="xpu")
    starts = offsets.cpu().tolist()
    groups = k // 32

    def pointers(base, strides):
        return torch.tensor(
            [base + offset for offset in strides], dtype=torch.uint64, device="xpu"
        )

    a_ptrs = pointers(a.data_ptr(), [m * (k // 2) for m in starts])
    b_ptrs = pointers(b.data_ptr(), [e * n * (k // 2) for e in range(EXPERTS)])
    out_ptrs = pointers(out_main.data_ptr(), [m * n * 4 for m in starts])
    sa_ptrs = pointers(sa_mn.data_ptr(), [m * groups for m in starts])
    sb_ptrs = pointers(sb_mn.data_ptr(), [e * n * groups for e in range(EXPERTS)])

    def branch():
        moe_grouped_mm_nt_mxfp4_w4a4_xe35(
            out_branch, a, sa_mn, b, sb_mn, rows, n, k, EXPERTS
        )

    def main_xe35():
        mxfp4_blockwise_scaled_grouped_mm(
            out_main,
            empty_ptrs,
            empty_ptrs,
            empty_ptrs,
            empty_ptrs,
            empty_ptrs,
            a,
            b,
            sa_rm,
            sb_rm,
            problems,
            offsets,
            workspace,
        )

    def main_bf16():
        main_xe35()
        return out_main.to(torch.bfloat16)

    def main_legacy_bf16():
        mxfp4_blockwise_scaled_grouped_mm(
            out_main,
            a_ptrs,
            b_ptrs,
            out_ptrs,
            sa_ptrs,
            sb_ptrs,
            a,
            b,
            sa_legacy,
            sb_mn,
            problems,
            offsets,
            workspace,
        )
        return out_main.to(torch.bfloat16)

    main_xe35()
    assert empty_ptrs.numel() == 0, "main_xe35 changed the empty-pointer sentinel"
    branch()
    torch.xpu.synchronize()
    verify_grouped_output_xe35(out_branch, a, sa_mn, b, sb_mn, rows_cpu)
    got = out_branch[:used].float()
    want = out_main[:used].clone()
    main_legacy_bf16()
    torch.xpu.synchronize()
    torch.testing.assert_close(out_main[:used], want, rtol=0, atol=0)
    torch.testing.assert_close(got, want, rtol=5e-3, atol=5e-3)
    abs_error = (got - want).abs()
    relative_error = abs_error / want.abs().clamp_min(1e-6)
    assert torch.isfinite(got).all() and torch.isfinite(want).all()
    accuracy = {
        "kind": "accuracy",
        "n": n,
        "k": k,
        "allocated_rows": allocated_rows,
        "used_rows": used,
        "active_experts": int((rows_cpu > 0).sum()),
        "elements": got.numel(),
        "max_abs_vs_main_fp32": abs_error.max().item(),
        "max_rel_vs_main_fp32": relative_error.max().item(),
        "main_fp32_abs_p50": want.abs().median().item(),
        "main_fp32_abs_max": want.abs().max().item(),
    }
    print(json.dumps(accuracy), flush=True)
    paths = (
        ("branch_gemm", branch),
        ("main_legacy_bf16", main_legacy_bf16),
        ("main_full_bf16", main_bf16),
    )
    for round_idx in range(rounds):
        order = paths if round_idx % 2 == 0 else tuple(reversed(paths))
        for label, call in order:
            samples = event_samples_xe35(call, warmup, iterations)
            ordered = sorted(samples)
            print(
                json.dumps(
                    {
                        "round": round_idx,
                        "path": label,
                        "n": n,
                        "k": k,
                        "experts": EXPERTS,
                        "allocated_rows": allocated_rows,
                        "used_rows": used,
                        "median_ms": statistics.median(samples),
                        "p20_ms": ordered[int(0.2 * (len(samples) - 1))],
                        "p80_ms": ordered[int(0.8 * (len(samples) - 1))],
                        "min_ms": min(samples),
                        "max_ms": max(samples),
                        "samples_ms": samples,
                    }
                ),
                flush=True,
            )
            torch.xpu.synchronize()


def main_xe35() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serving-batch", type=int, choices=(8,), default=8)
    parser.add_argument("--reference-threads", type=int, default=10)
    parser.add_argument(
        "--mode", choices=("decode", "compare", "both"), default="decode"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--payload-seed", type=int)
    parser.add_argument("--allocated-rows", type=int, default=UPPER_BOUND_ROWS)
    parser.add_argument("--sweep-rows", type=int)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--rows", type=int, nargs="+", default=[UPPER_BOUND_ROWS])
    parser.add_argument(
        "--active-experts",
        type=int,
        help="spread the 64 routed rows evenly over this many experts",
    )
    parser.add_argument(
        "--projection", choices=("gate_up", "down", "both"), default="both"
    )
    parser.add_argument("--isl", type=int, choices=(4096,), default=4096)
    parser.add_argument("--osl", type=int, choices=(16,), default=16)
    parser.add_argument("--tp", type=int, choices=(1,), default=1)
    parser.add_argument("--ep", type=int, choices=(1,), default=1)
    args = parser.parse_args()
    global torch, verify_grouped_output_xe35, is_xe3_arch
    global moe_grouped_mm_nt_mxfp4_w4a4_xe35
    global mxfp4_blockwise_scaled_grouped_mm, mxfp4_moe_scale_to_mn_major_xe35
    import torch
    from accuracy_reference import verify_grouped_output_xe35
    from sgl_kernel import (
        is_xe3_arch,
        moe_grouped_mm_nt_mxfp4_w4a4_xe35,
        mxfp4_blockwise_scaled_grouped_mm,
        mxfp4_moe_scale_to_mn_major_xe35,
    )
    if args.reference_threads <= 0:
        parser.error("--reference-threads must be positive")
    torch.set_num_threads(args.reference_threads)
    if args.payload_seed is not None:
        torch.manual_seed(args.payload_seed)
    if args.warmup < 0 or (args.iterations is not None and args.iterations <= 0):
        parser.error("--warmup must be nonnegative and --iterations must be positive")
    if args.rounds <= 0 or any(rows <= 0 or rows % EXPERTS for rows in args.rows):
        parser.error(
            "--rounds must be positive and comparison --rows must be positive multiples of 256"
        )

    if not torch.xpu.is_available() or not is_xe3_arch():
        raise RuntimeError("MXFP4 decode benchmark requires a CRI XPU")
    initial_device_rng = torch.xpu.get_rng_state() if args.mode == "both" else None
    if args.mode in ("compare", "both"):
        for allocated_rows in args.rows:
            for output_width in (4096, 8192):
                run_comparison_case_xe35(
                    output_width,
                    allocated_rows,
                    args.warmup,
                    args.iterations or 30,
                    args.rounds,
                )
    if args.mode == "compare":
        return
    if initial_device_rng is not None:
        torch.xpu.set_rng_state(initial_device_rng)
    counts = (
        make_sweep_counts_xe35(args.sweep_rows)
        if args.sweep_rows is not None
        else make_padded_counts_xe35(args.seed, args.active_experts)
    )
    if args.allocated_rows < int(counts.sum()):
        parser.error("allocated rows must cover all routed rows")
    if args.allocated_rows % ALIGNMENT:
        parser.error("allocated rows must be a multiple of 4")
    projections = []
    if args.projection in ("gate_up", "both"):
        projections.append(("gate_up", 2 * INTERMEDIATE, HIDDEN))
    if args.projection in ("down", "both"):
        projections.append(("down", HIDDEN, INTERMEDIATE))

    for projection in projections:
        torch.xpu.synchronize()
        time.sleep(2)
        benchmark_projection_xe35(
            *projection, counts, args.warmup, args.iterations or 50, args.allocated_rows
        )


if __name__ == "__main__":
    main_xe35()
