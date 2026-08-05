"""Benchmark runner for sgl-kernel-xpu, mirroring tests/run_suite.py.

The per-commit CI benchmark step used to be one long ``&&``-chained shell
command inside .github/workflows/pr-test-xpu.yml. That made it easy to lose an
``&&`` (silently dropping the failure of the preceding benchmark), easy to run
the same benchmark twice, and impossible to see where the time went. This file
owns the list instead, so adding or skipping a benchmark is a one-line edit.

Usage:
    python3 run_suite.py --suite per-commit

Each entry names a script under benchmark/ plus the log file it should be
tee'd to. Two log names are load-bearing: ``flash.log`` and ``fused_moe.log``
are parsed by benchmark/update_baseline_from_log.py to compare against
baseline.json, so do not rename them.

``args`` narrows a benchmark's sweep via flags the script already exposes.
Prefer that over editing sweep constants: the full sweep stays available for
local runs and for the nightly suite, and CI just asks for less of it.
"""

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import List

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class BenchFile:
    name: str
    # Log filename, relative to benchmark/. Defaults to "<name>.log".
    tee_log: str = ""
    # Extra CLI args passed to the script, used to narrow sweeps in CI.
    args: List[str] = field(default_factory=list)
    estimated_time: float = 20

    @property
    def log_name(self) -> str:
        return self.tee_log or f"{self.name}.log"


suites = {
    "per-commit": [
        # --- baseline-locked: do not trim these sweeps, and do not rename
        # their logs. update_baseline_from_log.py parses flash.log and
        # fused_moe.log against baseline.json; dropping a config silently
        # removes the matching baseline keys instead of failing loudly. ---
        BenchFile("bench_flash_attn.py", tee_log="flash.log", estimated_time=75),
        BenchFile("bench_fused_moe.py", tee_log="fused_moe.log", estimated_time=90),
        # --- attention / MLA ---
        BenchFile(
            "bench_flash_mla_decode.py",
            tee_log="mla.log",
            # Two block sizes instead of [16, 32, 64, 128]: keeps the smallest
            # and largest, which bracket the paging behaviour.
            args=["--block-sizes", "16", "128"],
            estimated_time=30,
        ),
        BenchFile(
            "bench_flash_mla_prefill.py",
            tee_log="mla_prefill.log",
            args=["--block-sizes", "16", "128"],
            estimated_time=40,
        ),
        BenchFile(
            "bench_flash_mla_with_kvcache.py",
            tee_log="mla_with_kvcache.log",
            estimated_time=30,
        ),
        BenchFile(
            "bench_flash_mla_sparse_fwd.py",
            tee_log="mla_sparse_fwd.log",
            estimated_time=20,
        ),
        # --- MoE ---
        BenchFile("bench_moe_topk_sigmoid.py", tee_log="moe_topk_sigmoid.log"),
        BenchFile("bench_moe_topk_softmax.py", tee_log="moe.log"),
        BenchFile("bench_moe_sum_reduce.py", tee_log="moe_sum_reduce.log"),
        BenchFile("bench_moe_fused_gate.py", tee_log="moe_fused_gate.py.log"),
        BenchFile("bench_moe_w4a16_grouped_gemm.py"),
        BenchFile(
            "bench_fused_experts_w4a16.py",
            # Drop the mid token count and two of three route seeds: the seeds
            # only average routing noise, and 1/2048 bracket the shape range.
            args=["--tokens", "1", "2048", "--route-seeds", "0"],
            estimated_time=30,
        ),
        BenchFile("bench_scatter_tokens_to_experts.py"),
        # --- norm / rope / quant ---
        BenchFile("bench_merge_states_v2.py", tee_log="merge_states.py.log"),
        BenchFile("bench_mrope.py", tee_log="mrope.py.log"),
        BenchFile("bench_swiglu_alpha_limit.py", tee_log="swiglu_alpha_limit.py.log"),
        BenchFile("bench_fused_qk_rope_with_cache.py"),
        BenchFile("bench_per_token_group_quant_mxfp4.py"),
        BenchFile("bench_per_token_group_quant_8bit_v2.py"),
        BenchFile("bench_per_token_quant_fp8.py"),
        BenchFile("bench_per_token_group_quant_mxfp4_fusion.py"),
        BenchFile("bench_top_k_renorm_probs.py"),
        BenchFile("bench_min_p_sampling_from_probs.py"),
        BenchFile("bench_biased_topk.py"),
        BenchFile(
            "bench_silu_and_mul_clamp.py", tee_log="bench_silu_and_mul_clamp.log"
        ),
        # --- hc / mhc ---
        BenchFile("bench_hc_split_sinkhorn.py"),
        BenchFile("bench_hc_pre_fuse.py"),
        BenchFile("bench_hc_pre_gemm_sqr_sum.py"),
        BenchFile("bench_hc_post.py"),
        BenchFile("bench_mhc_pre.py"),
        # --- LoRA ---
        BenchFile("bench_embedding_lora_a_fwd.py", estimated_time=40),
        BenchFile("bench_sgemm_lora_a_fwd.py", estimated_time=25),
        # --- JIT kernels (need icpx from the oneAPI toolchain) ---
        BenchFile("bench_jit_rmsnorm.py"),
        BenchFile("bench_jit_qknorm.py"),
        BenchFile("bench_jit_rope.py"),
        BenchFile("bench_jit_timestep_embedding.py"),
        BenchFile("bench_jit_per_token_group_quant_8bit.py"),
        BenchFile("bench_jit_moe_topk_sigmoid.py"),
        BenchFile("bench_jit_moe_fused_gate.py"),
        BenchFile("bench_jit_per_tensor_quant_fp8.py"),
        BenchFile("bench_jit_moe_align_block_size.py"),
        BenchFile("bench_jit_per_token_group_quant_8bit_v2.py"),
        BenchFile("bench_jit_activation.py"),
    ],
}

# Benchmarks deliberately excluded from per-commit, with the reason. Kept here
# rather than as a deleted line so the next person knows they exist and why
# they are not running.
SKIPPED = {
    "bench_fused_qk_norm_rope.py": (
        "XPU OOM in the reference path wedges the device and times the step out; "
        "re-enable once fixed"
    ),
    "bench_per_token_group_quant_8bit.py": (
        "disabled to unblock other benchmarks; see PR #256"
    ),
}


def auto_partition(files, rank, size):
    """Balance ``files`` into ``size`` partitions by ``estimated_time`` and
    return the ``rank``-th partition. Mirrors tests/run_suite.py.
    """
    weights = [f.estimated_time for f in files]
    if not weights or size <= 0 or size > len(weights):
        return []

    indexed_weights = [(w, -i) for i, w in enumerate(weights)]
    indexed_weights = sorted(indexed_weights, reverse=True)
    indexed_weights = [(w, -i) for w, i in indexed_weights]

    partitions = [[] for _ in range(size)]
    sums = [0.0] * size
    for weight, idx in indexed_weights:
        min_sum_idx = sums.index(min(sums))
        partitions[min_sum_idx].append(idx)
        sums[min_sum_idx] += weight

    return [files[i] for i in partitions[rank]]


def run_benchmark_files(files: List[BenchFile], timeout_per_file: float) -> int:
    """Run each benchmark, tee-ing combined output to its log file.

    Every benchmark runs even if an earlier one fails, so one broken kernel
    does not hide the results of everything after it; the failures are
    reported together at the end.
    """
    tic = time.perf_counter()
    failed = []
    elapsed_by_name = {}

    for i, bench in enumerate(files):
        script = os.path.join(BENCH_DIR, bench.name)
        log_path = os.path.join(BENCH_DIR, bench.log_name)
        cmd = [sys.executable, script] + bench.args

        print(
            f".\n.\nBegin ({i}/{len(files) - 1}):\n{' '.join(cmd)}\n"
            f"log: {bench.log_name}\n.\n.\n",
            flush=True,
        )
        start = time.perf_counter()

        try:
            proc = subprocess.run(
                cmd,
                cwd=BENCH_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout_per_file,
            )
            with open(log_path, "wb") as log_f:
                log_f.write(proc.stdout)
            sys.stdout.buffer.write(proc.stdout)
            sys.stdout.flush()
            rc = proc.returncode
            if rc != 0:
                print(f"\nFAILED: {bench.name} exited {rc}\n", flush=True)
                failed.append(bench.name)
        except subprocess.TimeoutExpired as e:
            # Persist whatever the benchmark managed to print before the
            # timeout; the tail is usually where the hang is visible.
            with open(log_path, "wb") as log_f:
                log_f.write(e.output or b"")
            print(
                f"\nTIMEOUT: {bench.name} exceeded {timeout_per_file}s\n",
                flush=True,
            )
            failed.append(bench.name)
        except FileNotFoundError:
            print(f"\nMISSING: {bench.name} not found under {BENCH_DIR}\n", flush=True)
            failed.append(bench.name)

        elapsed = time.perf_counter() - start
        elapsed_by_name[bench.name] = elapsed
        print(
            f".\n.\nEnd ({i}/{len(files) - 1}):\n"
            f"{bench.name}, {elapsed=:.0f}s, estimated={bench.estimated_time}\n.\n.\n",
            flush=True,
        )

    total = time.perf_counter() - tic
    print("\n===== benchmark timing summary (slowest first) =====", flush=True)
    for name, elapsed in sorted(elapsed_by_name.items(), key=lambda kv: -kv[1]):
        print(f"{elapsed:7.1f}s  {name}", flush=True)
    print(f"Total: {total:.0f}s ({total / 60:.1f} min) over {len(files)} benchmarks")

    if failed:
        print(f"\nFail. {len(failed)} benchmark(s) failed: {failed}", flush=True)
        return 1

    print("\nSuccess. All benchmarks passed.", flush=True)
    return 0


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--timeout-per-file",
        type=int,
        default=900,
        help="The time limit for running one benchmark in seconds.",
    )
    arg_parser.add_argument(
        "--suite",
        type=str,
        default=list(suites.keys())[0],
        choices=list(suites.keys()),
        help="The suite to run",
    )
    arg_parser.add_argument(
        "--range-begin",
        type=int,
        default=0,
        help="The begin index of the range of the benchmarks to run.",
    )
    arg_parser.add_argument(
        "--range-end",
        type=int,
        default=None,
        help="The end index of the range of the benchmarks to run.",
    )
    arg_parser.add_argument(
        "--auto-partition-id",
        type=int,
        help="Use auto load balancing. The part id.",
    )
    arg_parser.add_argument(
        "--auto-partition-size",
        type=int,
        help="Use auto load balancing. The number of parts.",
    )
    arg_parser.add_argument(
        "--list",
        action="store_true",
        help="Print the selected benchmarks and exit without running them.",
    )
    args = arg_parser.parse_args()
    print(f"{args=}")

    files = suites[args.suite]

    if args.auto_partition_size:
        files = auto_partition(files, args.auto_partition_id, args.auto_partition_size)
    else:
        files = files[args.range_begin : args.range_end]

    print("The running benchmarks are ", [f.name for f in files])
    if SKIPPED:
        print("Skipped benchmarks:")
        for name, reason in SKIPPED.items():
            print(f"  {name}: {reason}")

    if args.list:
        sys.exit(0)

    exit_code = run_benchmark_files(files, args.timeout_per_file)

    sys.exit(exit_code)
