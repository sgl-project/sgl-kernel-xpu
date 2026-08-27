"""Small opt-in benchmark for the FP8 split-activation MoE path."""

import argparse
import ctypes
import importlib.util
import statistics
from pathlib import Path

import torch

# Xe2 executes these FP8 E4M3 weights with BF16 activations through W8A16.


def _preload_fp8_instances():
    spec = importlib.util.find_spec("sgl_kernel")
    if spec is None or spec.submodule_search_locations is None:
        return
    for package_dir in spec.submodule_search_locations:
        for path in sorted(
            Path(package_dir).glob("libsgl-ops-sycl-GroupGemmFp8Xe20_inst*.so")
        ):
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)


_preload_fp8_instances()
from sgl_kernel import fused_experts

PROFILES = {
    "qwen-tp4": (3584, 1280, 8, 8),
    "deepseek-tp8": (7168, 512, 8, 8),
    "qwen35-tp4": (2048, 128, 16, 10),
}
TOKENS = [1, 32, 2048]
DEFAULT_WARMUP = 20
DEFAULT_REPETITIONS = 30
DEFAULT_INNER_REPETITIONS = 10


def _make_case(hidden, intermediate, experts, topk, tokens):
    torch.manual_seed(0)
    torch.xpu.manual_seed_all(0)
    hidden_states = torch.randn((tokens, hidden), dtype=torch.bfloat16, device="xpu")
    w1_bf16 = torch.randn(
        (experts, 2 * intermediate, hidden), dtype=torch.bfloat16, device="xpu"
    )
    w2_bf16 = torch.randn(
        (experts, hidden, intermediate), dtype=torch.bfloat16, device="xpu"
    )
    w1 = w1_bf16.to(torch.float8_e4m3fn)
    w2 = w2_bf16.to(torch.float8_e4m3fn)
    del w1_bf16, w2_bf16
    w1_scale = torch.ones(
        (experts, 2 * intermediate // 128, hidden // 128),
        dtype=torch.float32,
        device="xpu",
    )
    w2_scale = torch.ones(
        (experts, hidden // 128, intermediate // 128),
        dtype=torch.float32,
        device="xpu",
    )
    topk_ids = (
        torch.arange(tokens * topk, device="xpu", dtype=torch.int32).reshape(
            tokens, topk
        )
        % experts
    )
    topk_weights = torch.full(
        (tokens, topk), 1.0 / topk, dtype=torch.float32, device="xpu"
    )
    return hidden_states, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale


def _run_case(profile_name, tokens, warmup, repetitions, inner_repetitions):
    hidden, intermediate, experts, topk = PROFILES[profile_name]
    args = _make_case(hidden, intermediate, experts, topk, tokens)

    def run():
        fused_experts(
            *args[:5],
            activation="silu",
            use_fp8_w8a8=True,
            w1_scale=args[5],
            w2_scale=args[6],
        )

    for _ in range(warmup):
        run()
    torch.xpu.synchronize()
    samples = []
    for _ in range(repetitions):
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        for _ in range(inner_repetitions):
            run()
        end.record()
        torch.xpu.synchronize()
        samples.append(start.elapsed_time(end) / inner_repetitions)
    samples.sort()
    median = statistics.median(samples)
    del args
    torch.xpu.empty_cache()
    return hidden, intermediate, experts, topk, median


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=TOKENS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument(
        "--inner-repetitions", type=int, default=DEFAULT_INNER_REPETITIONS
    )
    args = parser.parse_args()
    print("profile tokens hidden intermediate experts topk median_ms", flush=True)
    for profile_name in PROFILES:
        for tokens in args.tokens:
            hidden, intermediate, experts, topk, median = _run_case(
                profile_name,
                tokens,
                args.warmup,
                args.repetitions,
                args.inner_repetitions,
            )
            print(
                profile_name,
                tokens,
                hidden,
                intermediate,
                experts,
                topk,
                f"{median:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
