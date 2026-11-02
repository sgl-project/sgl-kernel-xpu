"""Nightly runner for sgl-kernel-xpu.

Runs the same per-commit lists as PR CI, taken from tests/run_suite.py and
benchmark/run_suite.py, so nightly coverage cannot drift from CI. Unlike PR
CI, every file runs even after a failure; all failures are reported at the
end and the exit code is non-zero if any file failed.

Two suites:
  * sgl-kernel-test       -- tests/run_suite.py per-commit list
  * sgl-kernel-benchmark  -- benchmark/run_suite.py per-commit list

Both are invoked as:
    python3 run_nightly_suite.py --suite <name>
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

SUITES = {
    "sgl-kernel-test": ("test", TESTS_DIR, test_suite.suites["per-commit"]),
    "sgl-kernel-benchmark": ("bench", BENCH_DIR, bench_suite.suites["per-commit"]),
}


def run_tests(files, workdir: str, timeout_per_file: int) -> int:
    # run_unittest_files resolves file names against the cwd.
    os.chdir(workdir)
    tic = time.perf_counter()
    failed = []
    for i, f in enumerate(files):
        print(f"\n===== [{i + 1}/{len(files)}] {f.name} =====", flush=True)
        # One file per call keeps run_unittest_files' timeout/retry handling
        # but stops its stop-at-first-failure from skipping the rest.
        if run_unittest_files([f], timeout_per_file) != 0:
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


def run_benchmarks(files, workdir: str, timeout_per_file: int) -> int:
    # benchmark/run_suite.py already runs every benchmark and reports failures at the end.
    return bench_suite.run_benchmark_files(files, timeout_per_file)


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

    kind, workdir, entries = SUITES[args.suite]

    if args.auto_partition_size:
        entries = test_suite.auto_partition(
            entries, args.auto_partition_id, args.auto_partition_size
        )
    else:
        entries = entries[args.range_begin : args.range_end]

    print(f"Suite: {args.suite} ({kind}); {len(entries)} entries; workdir: {workdir}")
    print("Selected entries:", [e.name for e in entries])

    if kind == "test":
        rc = run_tests(entries, workdir, args.timeout_per_file)
    else:
        rc = run_benchmarks(entries, workdir, args.timeout_per_file)

    sys.exit(rc)
