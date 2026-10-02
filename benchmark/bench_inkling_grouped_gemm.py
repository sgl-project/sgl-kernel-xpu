"""Nonzero production-shape benchmark for Inkling TP4 routed MoE.

``--gemm routed`` invokes the production ``sgl_kernel.fused_experts`` wrapper
and device-times its complete route:

    prepare/scatter -> gate/up GEMM -> SiLU*up -> down GEMM -> combine

The gate and down choices retain isolated grouped-GEMM measurements. Every
choice uses the same deterministic, nonzero MXFP4 inputs so an exact-base
wheel and a candidate wheel can be compared with identical data.
"""

import argparse
import ctypes
import os
import statistics
from pathlib import Path
from typing import Callable

import torch

TOKENS = 4096
TOPK = 6
E = 256
HIDDEN = 6144
INTERMEDIATE = 768
ROUTES = TOKENS * TOPK
MXFP4_GROUP_SIZE = 32
PEAK_TFLOPS = 98.0
DEFAULT_SEED = 20260923


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark nonzero Inkling TP4 MXFP4 routed-MoE shapes."
    )
    parser.add_argument(
        "--gemm",
        choices=("gate", "down", "routed"),
        required=True,
        help="'routed' times fused_experts from scatter through combine.",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Input seed; use the same value for exact-base/candidate A/B.",
    )
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument(
        "--save-reference",
        type=Path,
        metavar="PATH",
        help="Save routed output and input metadata from an exact-base run.",
    )
    reference.add_argument(
        "--reference",
        type=Path,
        metavar="PATH",
        help="Compare routed output with a saved exact-base result.",
    )
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    if (args.save_reference or args.reference) and args.gemm != "routed":
        parser.error("--save-reference/--reference require --gemm routed")
    return args


def load_sgl_kernel():
    if instance_libraries := os.environ.get("SGL_W4A16_INSTANCE_LIBRARY"):
        for instance_library in instance_libraries.split(os.pathsep):
            ctypes.CDLL(instance_library, mode=ctypes.RTLD_GLOBAL)
    import sgl_kernel

    return sgl_kernel


def event_samples(fn: Callable[[], None], warmup: int, repeat: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()

    timings = []
    for _ in range(repeat):
        begin = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        begin.record()
        fn()
        end.record()
        torch.xpu.synchronize()
        timings.append(begin.elapsed_time(end))
    return sorted(timings)


def percentile(timings: list[float], fraction: float) -> float:
    return timings[round(fraction * (len(timings) - 1))]


def make_routing(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    scores = torch.randn((TOKENS, E), generator=generator, dtype=torch.float32)
    probabilities = torch.softmax(scores, dim=-1)
    topk_weights, topk_ids = torch.topk(probabilities, TOPK, dim=-1)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    return topk_ids.to(torch.int32), topk_weights.to(torch.bfloat16)


def make_xpu_inputs(seed: int) -> dict[str, torch.Tensor]:
    # Seed before the first XPU random allocation. Together with CPU-generated
    # routing this gives exact-base and candidate runs the same input bytes.
    torch.manual_seed(seed)
    torch.xpu.manual_seed_all(seed)
    topk_ids_cpu, topk_weights_cpu = make_routing(seed)

    hidden_states = (
        torch.randn((TOKENS, HIDDEN), dtype=torch.bfloat16, device="xpu") * 0.02
    )
    w1 = torch.randint(
        0,
        256,
        (E, 2 * INTERMEDIATE, HIDDEN // 2),
        dtype=torch.uint8,
        device="xpu",
    )
    w2 = torch.randint(
        0,
        256,
        (E, HIDDEN, INTERMEDIATE // 2),
        dtype=torch.uint8,
        device="xpu",
    )
    w1_scale = torch.randint(
        121,
        126,
        (E, 2 * INTERMEDIATE, HIDDEN // MXFP4_GROUP_SIZE),
        dtype=torch.uint8,
        device="xpu",
    )
    w2_scale = torch.randint(
        121,
        126,
        (E, HIDDEN, INTERMEDIATE // MXFP4_GROUP_SIZE),
        dtype=torch.uint8,
        device="xpu",
    )
    return {
        "hidden_states": hidden_states,
        "w1": w1,
        "w2": w2,
        "w1_scale": w1_scale,
        "w2_scale": w2_scale,
        "topk_ids": topk_ids_cpu.to("xpu"),
        "topk_weights": topk_weights_cpu.to("xpu"),
        "rows_cpu": torch.bincount(topk_ids_cpu.flatten(), minlength=E).to(torch.int32),
    }


def make_grouped_gemm(
    inputs: dict[str, torch.Tensor], gemm: str
) -> tuple[Callable[[], None], int, int]:
    if gemm == "gate":
        activations = inputs["hidden_states"].repeat_interleave(TOPK, dim=0)
        output = torch.empty(
            (ROUTES, 2 * INTERMEDIATE), dtype=torch.bfloat16, device="xpu"
        )
        weights, scales = inputs["w1"], inputs["w1_scale"]
        n, k = 2 * INTERMEDIATE, HIDDEN
    else:
        activations = (
            torch.randn((ROUTES, INTERMEDIATE), dtype=torch.bfloat16, device="xpu")
            * 0.02
        )
        output = torch.empty((ROUTES, HIDDEN), dtype=torch.bfloat16, device="xpu")
        weights, scales = inputs["w2"], inputs["w2_scale"]
        n, k = HIDDEN, INTERMEDIATE
    rows = inputs["rows_cpu"].to("xpu")

    def run() -> None:
        torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20_w4a16(
            output,
            activations,
            weights,
            scales,
            None,
            None,
            rows,
            E,
            False,
            MXFP4_GROUP_SIZE,
        )

    return run, n, k


def make_routed(
    inputs: dict[str, torch.Tensor], sgl_kernel
) -> tuple[Callable[[], None], list[torch.Tensor]]:
    # Keep the current output alive while every timed call follows the same
    # public production wrapper and its cached internal workspaces.
    outputs = []

    def run() -> None:
        outputs[:] = [
            sgl_kernel.fused_experts(
                inputs["hidden_states"],
                inputs["w1"],
                inputs["w2"],
                inputs["topk_weights"],
                inputs["topk_ids"],
                activation="silu",
                use_mxfp4_w4a16=True,
                w1_scale=inputs["w1_scale"],
                w2_scale=inputs["w2_scale"],
            )
        ]

    return run, outputs


def output_metadata(seed: int) -> dict[str, object]:
    return {
        "seed": seed,
        "tokens": TOKENS,
        "topk": TOPK,
        "experts": E,
        "hidden": HIDDEN,
        "intermediate": INTERMEDIATE,
        "dtype": "bfloat16",
        "quantization": "mxfp4_group32",
    }


def check_or_save_reference(
    output: torch.Tensor,
    seed: int,
    save_reference: Path | None,
    reference: Path | None,
) -> None:
    torch.xpu.synchronize()
    actual = output.cpu()
    if not bool(torch.isfinite(actual).all()):
        raise AssertionError("routed output contains non-finite values")

    metadata = output_metadata(seed)
    if save_reference:
        save_reference.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"metadata": metadata, "output": actual}, save_reference)
        print(f"correctness: finite=True saved_reference={save_reference}")
    elif reference:
        expected = torch.load(reference, map_location="cpu", weights_only=True)
        if expected.get("metadata") != metadata:
            raise AssertionError(
                f"reference metadata mismatch: {expected.get('metadata')} != {metadata}"
            )
        torch.testing.assert_close(actual, expected["output"], rtol=1e-1, atol=1e-2)
        error = (actual.float() - expected["output"].float()).abs()
        print(
            "correctness: finite=True reference_match=True rtol=0.1 atol=0.01 "
            f"max_abs_error={error.max().item():.6g} "
            f"mean_abs_error={error.mean().item():.6g}"
        )
    else:
        print("correctness: finite=True reference=not_requested")


def main() -> None:
    args = parse_args()
    sgl_kernel = load_sgl_kernel()
    inputs = make_xpu_inputs(args.seed)
    rows = inputs["rows_cpu"]
    print(
        f"config: gemm={args.gemm} seed={args.seed} tokens={TOKENS} topk={TOPK} "
        f"routes={ROUTES} experts={E} hidden={HIDDEN} intermediate={INTERMEDIATE} "
        f"rows={rows.min().item()}..{rows.max().item()} dtype=bf16 quant=mxfp4"
    )

    if args.gemm == "routed":
        run, outputs = make_routed(inputs, sgl_kernel)
        timings = event_samples(run, args.warmup, args.repeat)
        name = "routed_fused_experts"
        check_or_save_reference(
            outputs[0], args.seed, args.save_reference, args.reference
        )
    else:
        run, n, k = make_grouped_gemm(inputs, args.gemm)
        timings = event_samples(run, args.warmup, args.repeat)
        name = args.gemm
        p50 = statistics.median(timings)
        tflops = 2 * ROUTES * n * k / (p50 * 1e-3) / 1e12
        roofline_ms = 2 * ROUTES * n * k / (PEAK_TFLOPS * 1e12) * 1e3
        print(
            f"{name}_throughput: effective={tflops:.2f} TFLOP/s "
            f"compute_roofline={roofline_ms:.4f} ms "
            f"attainment={roofline_ms / p50 * 100:.1f}%"
        )

    print(
        f"{name}: p20={percentile(timings, 0.2):.4f} "
        f"median={statistics.median(timings):.4f} "
        f"p80={percentile(timings, 0.8):.4f} ms "
        f"warmup={args.warmup} repeat={args.repeat}"
    )


if __name__ == "__main__":
    main()
