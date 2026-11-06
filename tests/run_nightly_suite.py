"""Nightly runner for sgl-kernel-xpu.

Runs the same per-commit lists as PR CI, taken from tests/run_suite.py and
benchmark/run_suite.py, so nightly coverage cannot drift from CI. Unlike PR
CI, every file runs even after a failure; all failures are reported at the
end and the exit code is non-zero if any file failed.

Two suites:
  * sgl-kernel-test       -- tests/run_suite.py per-commit list
  * sgl-kernel-benchmark  -- benchmark/run_suite.py per-commit list

--variant aot|jit appends the matching *_ONLY_* list below to the per-commit
list. Those extras run only in that nightly, never in PR CI.

Both are invoked as:
    python3 run_nightly_suite.py --suite <name> --variant <aot|jit>
"""

import argparse
import importlib.util
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.join(REPO_ROOT, "tests")
BENCH_DIR = os.path.join(REPO_ROOT, "benchmark")

sys.path.insert(0, TESTS_DIR)
import run_suite as test_suite  # noqa: E402
from test_utils import _recover_xpu, run_unittest_files  # noqa: E402


def _load_bench_suite():
    # Both runners are named run_suite.py; load the benchmark one under its own name.
    path = os.path.join(BENCH_DIR, "run_suite.py")
    spec = importlib.util.spec_from_file_location("bench_run_suite", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench_suite = _load_bench_suite()
TestFile = test_suite.TestFile
BenchFile = bench_suite.BenchFile

# ---------------------------------------------------------------------------
# Nightly-only extras, added on top of the per-commit lists.
# To run something in PR CI and both nightlies, add it to the per-commit list
# in tests/run_suite.py or benchmark/run_suite.py instead.
# ---------------------------------------------------------------------------

# Tests that run only in the JIT nightly (USE_SYCL_JIT=ON wheel). Example:
#   JIT_ONLY_TESTS = [
#       TestFile("test_my_jit_kernel.py"),
#       TestFile("test_my_slow_jit_case.py", estimated_time=300),
#   ]
JIT_ONLY_TESTS = []

# Tests that run only in the AOT nightly. Example:
#   AOT_ONLY_TESTS = [
#       TestFile("test_my_aot_kernel.py"),
#   ]
AOT_ONLY_TESTS = []

# Benchmarks that run only in the JIT nightly; tee_log/args are optional. Example:
#   JIT_ONLY_BENCHMARKS = [
#       BenchFile("bench_my_jit_kernel.py"),
#       BenchFile("bench_my_jit_sweep.py", tee_log="my_jit.log", args=["--quick"]),
#   ]
JIT_ONLY_BENCHMARKS = []

# Benchmarks that run only in the AOT nightly. Example:
#   AOT_ONLY_BENCHMARKS = [
#       BenchFile("bench_my_aot_kernel.py"),
#   ]
AOT_ONLY_BENCHMARKS = []

# Per-file time limits (seconds) that replace --timeout-per-file in the JIT nightly.
# :latest-jit starts with an empty JIT kernel cache, and each new FMHA config takes
# ~90-100 s to compile on first use. Measured on a B60 (2026-11-05):
# test_flash_attention.py takes ~70 min with a warm cache and ~3.75 h with a cold one,
# so the default 7200 s limit stops it at ~23%. bench_flash_attn.py hit its 1800 s
# limit while still compiling; its cold time has not been measured yet.
JIT_TIMEOUT_OVERRIDES = {
    "test_flash_attention.py": 16200,
    "bench_flash_attn.py": 7200,
}

TIMEOUT_OVERRIDES = {
    "jit": JIT_TIMEOUT_OVERRIDES,
}

SUITES = {
    "sgl-kernel-test": ("test", TESTS_DIR, test_suite.suites["per-commit"]),
    "sgl-kernel-benchmark": ("bench", BENCH_DIR, bench_suite.suites["per-commit"]),
}

EXTRAS = {
    ("sgl-kernel-test", "jit"): JIT_ONLY_TESTS,
    ("sgl-kernel-test", "aot"): AOT_ONLY_TESTS,
    ("sgl-kernel-benchmark", "jit"): JIT_ONLY_BENCHMARKS,
    ("sgl-kernel-benchmark", "aot"): AOT_ONLY_BENCHMARKS,
}


def select_entries(suite: str, variant):
    """Per-commit list plus the extras for ``variant``, skipping duplicates."""
    entries = list(SUITES[suite][2])
    names = {e.name for e in entries}
    for extra in EXTRAS.get((suite, variant), []):
        if extra.name in names:
            print(f"[skip] {extra.name} is already in the per-commit list")
            continue
        entries.append(extra)
        names.add(extra.name)
    return entries


def run_tests(files, workdir: str, timeout_per_file: int, overrides=None) -> int:
    # run_unittest_files resolves file names against the cwd.
    os.chdir(workdir)
    tic = time.perf_counter()
    failed = []
    for i, f in enumerate(files):
        timeout = (overrides or {}).get(f.name, timeout_per_file)
        print(
            f"\n===== [{i + 1}/{len(files)}] {f.name} (timeout {timeout}s) =====",
            flush=True,
        )
        # One file per call keeps run_unittest_files' timeout/retry handling
        # but stops its stop-at-first-failure from skipping the rest.
        if run_unittest_files([f], timeout) != 0:
            failed.append(f.name)
            # Give a crashed or wedged device time to reset before the next file.
            _recover_xpu()

    total = time.perf_counter() - tic
    print(f"\n===== nightly tests: {len(files)} files, {total / 60:.1f} min =====")
    if failed:
        print(f"Fail. {len(failed)} test file(s) failed:", flush=True)
        for name in failed:
            print(f"  {name}", flush=True)
        return 1
    print("Success. All test files passed.", flush=True)
    return 0


def run_benchmarks(files, workdir: str, timeout_per_file: int, overrides=None) -> int:
    # benchmark/run_suite.py already runs every benchmark and reports failures at the end.
    return bench_suite.run_benchmark_files(files, timeout_per_file, overrides)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        required=True,
        choices=list(SUITES.keys()),
        help="Which nightly suite to run.",
    )
    parser.add_argument(
        "--timeout-per-file",
        type=int,
        default=1800,
        help="Time limit per test/benchmark in seconds.",
    )
    parser.add_argument(
        "--variant",
        choices=["aot", "jit"],
        default=None,
        help="Append that variant's *_ONLY_* list (default: per-commit only).",
    )
    parser.add_argument(
        "--range-begin",
        type=int,
        default=0,
        help="Begin index (inclusive) of the slice to run.",
    )
    parser.add_argument(
        "--range-end",
        type=int,
        default=None,
        help="End index (exclusive) of the slice to run.",
    )
    parser.add_argument(
        "--auto-partition-id",
        type=int,
        help="Auto load-balancing: partition index (0..size-1).",
    )
    parser.add_argument(
        "--auto-partition-size",
        type=int,
        help="Auto load-balancing: total number of partitions.",
    )
    args = parser.parse_args()

    kind, workdir, _ = SUITES[args.suite]
    entries = select_entries(args.suite, args.variant)

    if args.auto_partition_size:
        entries = test_suite.auto_partition(
            entries, args.auto_partition_id, args.auto_partition_size
        )
    else:
        entries = entries[args.range_begin : args.range_end]

    print(
        f"Suite: {args.suite} ({kind}), variant: {args.variant}; "
        f"{len(entries)} entries; workdir: {workdir}"
    )
    print("Selected entries:", [e.name for e in entries])

    overrides = {
        name: t
        for name, t in TIMEOUT_OVERRIDES.get(args.variant, {}).items()
        if name in {e.name for e in entries}
    }
    if overrides:
        print("Per-file timeout overrides:", overrides)

    if kind == "test":
        rc = run_tests(entries, workdir, args.timeout_per_file, overrides)
    else:
        rc = run_benchmarks(entries, workdir, args.timeout_per_file, overrides)

    sys.exit(rc)
