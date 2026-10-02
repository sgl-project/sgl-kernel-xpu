"""Device-time benchmark for generic BF16 MXFP4 K32 grouped-GEMM dispatch.

Run the same command on exact-main and candidate wheels, bound to the same
GPU. Cases cross every new row-policy boundary and include non-Inkling N/K
shapes. The installed sgl_kernel package registers the production operator.
"""

import argparse
import statistics

import sgl_kernel  # noqa: F401
import torch

EXPERTS = 16
SEED = 20260923
CASES = {
    "m8_tail": (8, 192, 256),
    "m16_narrow": (16, 192, 256),
    "m16_short_tail": (16, 320, 256),
    "m16_long_tail": (16, 320, 1056),
    "m17_boundary": (17, 320, 256),
    "m32_narrow": (32, 64, 256),
    "m48_prefill": (48, 128, 256),
}


def make_case(rows: int, gemm_n: int, gemm_k: int):
    torch.manual_seed(SEED)
    torch.xpu.manual_seed_all(SEED)
    total_m = EXPERTS * rows
    activations = torch.randn((total_m, gemm_k), device="xpu", dtype=torch.bfloat16)
    weights = torch.randint(
        0,
        256,
        (EXPERTS, gemm_n, gemm_k // 2),
        device="xpu",
        dtype=torch.uint8,
    )
    scales = torch.randint(
        121,
        126,
        (EXPERTS, gemm_n, gemm_k // 32),
        device="xpu",
        dtype=torch.uint8,
    )
    expert_rows = torch.full((EXPERTS,), rows, device="xpu", dtype=torch.int32)
    output = torch.empty((total_m, gemm_n), device="xpu", dtype=torch.bfloat16)

    def run() -> None:
        torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20_w4a16(
            output,
            activations,
            weights,
            scales,
            None,
            None,
            expert_rows,
            EXPERTS,
            False,
            32,
        )

    return run, output


def measure(run, warmup: int, repeat: int) -> list[float]:
    for _ in range(warmup):
        run()
    torch.xpu.synchronize()
    samples = []
    for _ in range(repeat):
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        torch.xpu.synchronize()
        samples.append(start.elapsed_time(end))
    return sorted(samples)


def percentile(samples: list[float], fraction: float) -> float:
    return samples[round(fraction * (len(samples) - 1))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=(*CASES, "all"), default="all")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat < 1:
        parser.error("--warmup must be non-negative and --repeat must be positive")

    cases = CASES.items() if args.case == "all" else [(args.case, CASES[args.case])]
    for name, (rows, gemm_n, gemm_k) in cases:
        run, output = make_case(rows, gemm_n, gemm_k)
        samples = measure(run, args.warmup, args.repeat)
        assert bool(torch.isfinite(output).all()), f"{name}: non-finite output"
        print(
            f"{name}: E={EXPERTS} rows/expert={rows} N={gemm_n} K={gemm_k} "
            f"dtype=bf16 quant=mxfp4_k32 "
            f"p20={percentile(samples, 0.2):.4f} "
            f"p50={statistics.median(samples):.4f} "
            f"p80={percentile(samples, 0.8):.4f} ms "
            f"warmup={args.warmup} repeat={args.repeat}"
        )


if __name__ == "__main__":
    main()
