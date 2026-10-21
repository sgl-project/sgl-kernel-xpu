import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("batch", [1, 7, 9])
def test_position_benchmark_rejects_serving_batch_before_imports(batch):
    benchmark = (
        Path(__file__).resolve().parents[1]
        / "benchmark"
        / "bench_position_occupancy.py"
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


@pytest.mark.parametrize("variable", ["BATCHES", "BATCH_SIZE", "SERVING_BATCH"])
def test_position_benchmark_rejects_environment_batch_before_imports(variable):
    benchmark = (
        Path(__file__).resolve().parents[1]
        / "benchmark"
        / "bench_position_occupancy.py"
    )
    result = subprocess.run(
        [sys.executable, "-S", str(benchmark)],
        env={**os.environ, variable: "7"},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "ModuleNotFoundError" not in result.stderr


@pytest.mark.parametrize(
    "argument,value", [("--isl", "2048"), ("--osl", "16"), ("--tp", "1"), ("--ep", "4")]
)
def test_position_benchmark_rejects_retired_scope_before_imports(argument, value):
    benchmark = (
        Path(__file__).resolve().parents[1]
        / "benchmark"
        / "bench_position_occupancy.py"
    )
    result = subprocess.run(
        [sys.executable, "-S", str(benchmark), argument, value],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "B8/ISL4096/OSL1024/TP4/EP1" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
