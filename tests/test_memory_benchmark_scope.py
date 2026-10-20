import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("batch", [1, 7, 9])
@pytest.mark.parametrize(
    "mode", ["profiler", "event", "tiles", "write-groups", "aux", "prefill"]
)
def test_memory_benchmark_rejects_serving_batch_override(batch, mode):
    benchmark = (
        Path(__file__).resolve().parents[1] / "benchmark" / "bench_memory_compare.py"
    )
    command = [
        sys.executable,
        "-S",
        str(benchmark),
        "--mode",
        mode,
        "--batch",
        str(batch),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert "Muse-Spark serving benchmarks require --batch 8" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr


@pytest.mark.parametrize(
    "mode", ["profiler", "event", "tiles", "write-groups", "aux", "prefill"]
)
def test_memory_benchmark_help_without_device_imports(mode):
    benchmark = (
        Path(__file__).resolve().parents[1] / "benchmark" / "bench_memory_compare.py"
    )
    result = subprocess.run(
        [sys.executable, "-S", str(benchmark), "--mode", mode, "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "--op" in result.stdout
    assert "ModuleNotFoundError" not in result.stderr


def test_memory_benchmark_rejects_library_override():
    benchmark = (
        Path(__file__).resolve().parents[1] / "benchmark" / "bench_memory_compare.py"
    )
    result = subprocess.run(
        [sys.executable, "-S", str(benchmark), "--test-library", "tuning.so"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "unrecognized arguments: --test-library" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr


@pytest.mark.parametrize(
    "arguments,mode,samples",
    [
        ([], "profiler", 31),
        (
            [
                "--mode",
                "event",
                "--op",
                "extend",
                "--prefix-len",
                "0",
                "--extend-len",
                "4096",
                "--production-only",
            ],
            "event",
            31,
        ),
        (["--mode", "tiles"], "tiles", 51),
        (["--mode", "write-groups"], "write-groups", 51),
        (["--mode", "aux"], "aux", 30),
        (["--mode", "prefill"], "prefill", 31),
    ],
)
def test_memory_benchmark_mode_defaults(arguments, mode, samples):
    benchmark = (
        Path(__file__).resolve().parents[1] / "benchmark" / "bench_memory_compare.py"
    )
    spec = importlib.util.spec_from_file_location("memory_benchmark_cli", benchmark)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parsed = module.parse_args(arguments)
    assert parsed.mode == mode
    assert parsed.samples == samples
    assert parsed.batch == 8
    assert parsed.decode_steps == (0, 1, 63, 64, 1023)
    if mode == "tiles":
        assert parsed.cases == ("extend", "fused")
        assert parsed.lengths == (
            64,
            128,
            256,
            384,
            511,
            512,
            513,
            768,
            1024,
            1536,
            2047,
            2048,
            2049,
            3072,
            4096,
        )


@pytest.mark.parametrize("step", [-1, 1024])
def test_memory_benchmark_rejects_decode_outside_production_output(step):
    benchmark = (
        Path(__file__).resolve().parents[1] / "benchmark" / "bench_memory_compare.py"
    )
    result = subprocess.run(
        [sys.executable, "-S", str(benchmark), "--decode-steps", str(step)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "ModuleNotFoundError" not in result.stderr


@pytest.mark.parametrize("arguments", [["--main-only"], ["--mode", "aux"]])
def test_memory_benchmark_rejects_unpaired_regression_gate(arguments):
    benchmark = (
        Path(__file__).resolve().parents[1] / "benchmark" / "bench_memory_compare.py"
    )
    result = subprocess.run(
        [sys.executable, "-S", str(benchmark), "--require-no-regression", *arguments],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "--require-no-regression requires paired profiler mode" in result.stderr
