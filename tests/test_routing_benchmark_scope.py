import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("batch", [1, 7, 9])
@pytest.mark.parametrize("mode", ["audit", "padded-gather"])
def test_routing_benchmark_rejects_serving_batch_override_xe35(batch, mode):
    benchmark = (
        Path(__file__).resolve().parents[1]
        / "benchmark"
        / "bench_routing_quant_audit.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(benchmark),
            "--mode",
            mode,
            "--serving-batch",
            str(batch),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "Muse-Spark serving benchmarks require --serving-batch 8" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr


@pytest.mark.parametrize("mode", ["audit", "padded-gather"])
def test_routing_benchmark_help_without_device_imports_xe35(mode):
    benchmark = (
        Path(__file__).resolve().parents[1]
        / "benchmark"
        / "bench_routing_quant_audit.py"
    )
    result = subprocess.run(
        [sys.executable, "-S", str(benchmark), "--mode", mode, "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "--op" in result.stdout


@pytest.mark.parametrize("mode,warmup", [("audit", 5), ("padded-gather", 3)])
def test_routing_benchmark_mode_defaults_xe35(mode, warmup):
    benchmark = (
        Path(__file__).resolve().parents[1]
        / "benchmark"
        / "bench_routing_quant_audit.py"
    )
    spec = importlib.util.spec_from_file_location("routing_benchmark_cli", benchmark)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parsed = module.parse_args_xe35(["--mode", mode])
    assert parsed.warmup == warmup
    assert parsed.serving_batch == 8
    assert parsed.tokens == (8, 32768)
    assert parsed.graph_replay == 1
