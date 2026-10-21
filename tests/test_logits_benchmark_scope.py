import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("batch", [1, 7, 9])
def test_logits_benchmark_rejects_serving_batch_before_imports(batch):
    benchmark = (
        Path(__file__).resolve().parents[1]
        / "benchmark"
        / "bench_logits_vocab_sweep.py"
    )
    result = subprocess.run(
        [sys.executable, "-S", str(benchmark), "--batch", str(batch)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "B8/ISL4096/OSL1024/TP4/EP1" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr


def test_logits_benchmark_rejects_environment_batch_before_imports():
    benchmark = (
        Path(__file__).resolve().parents[1]
        / "benchmark"
        / "bench_logits_vocab_sweep.py"
    )
    result = subprocess.run(
        [sys.executable, "-S", str(benchmark)],
        env={**os.environ, "BATCHES": "7"},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "ModuleNotFoundError" not in result.stderr


@pytest.mark.parametrize("batch", [1, 7, 9])
def test_native_ab_rejects_serving_batch_before_imports(batch):
    benchmark = (
        Path(__file__).resolve().parents[1] / "benchmark" / "bench_logits_vocab_ab.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(benchmark),
            "--serving-batch",
            str(batch),
            "--baseline-library",
            "unused.so",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "ModuleNotFoundError" not in result.stderr


def load_native_benchmark():
    path = (
        Path(__file__).resolve().parents[1] / "benchmark" / "bench_logits_vocab_ab.py"
    )
    spec = importlib.util.spec_from_file_location("native_logits_vocab_benchmark", path)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    return benchmark


def test_native_ab_rejects_shared_sycl_kernel_ids():
    benchmark = load_native_benchmark()
    kernels = [{"name": "SoftcapLogitsFloat4Kernel"}] * 4
    with pytest.raises(RuntimeError, match="distinct SYCL kernel IDs"):
        benchmark.validate_native_kernel_order(kernels, ["baseline", "candidate"], 2)


def test_native_ab_balances_embedding_buffers_and_launch_order():
    benchmark = load_native_benchmark()
    assignments = [
        (index % 2, benchmark.embedding_buffer_order(index)) for index in range(8)
    ]
    for launch_order in (0, 1):
        for buffer_order in ((0, 1), (1, 0)):
            assert assignments.count((launch_order, buffer_order)) == 2


@pytest.mark.parametrize("candidate_us", [0.9, 1.0])
def test_native_ab_latency_gate_accepts_no_regression(candidate_us):
    load_native_benchmark().validate_latency_ratio(1.0, candidate_us)


def test_native_ab_latency_gate_rejects_regression():
    with pytest.raises(RuntimeError, match="exceeds baseline"):
        load_native_benchmark().validate_latency_ratio(1.0, 1.001)


@pytest.mark.parametrize("duration", [0.0, -1.0, float("nan"), float("inf")])
@pytest.mark.parametrize("baseline", [False, True])
def test_native_ab_latency_gate_rejects_invalid_duration(duration, baseline):
    values = (duration, 1.0) if baseline else (1.0, duration)
    with pytest.raises(RuntimeError, match="finite and positive"):
        load_native_benchmark().validate_latency_ratio(*values)


@pytest.mark.parametrize(
    "names", [["baseline", "candidate"], ["candidate", "baseline"]]
)
def test_native_ab_accepts_distinct_sycl_kernel_ids(names):
    benchmark = load_native_benchmark()
    kernel_names = {
        "baseline": "logits_vocab_baseline::SoftcapLogitsFloat4Kernel",
        "candidate": "SoftcapLogitsFloat4Kernel",
    }
    kernels = [{"name": kernel_names[name]} for name in names] * 2
    benchmark.validate_native_kernel_order(kernels, names, 2)
