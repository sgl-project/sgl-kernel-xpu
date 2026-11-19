"""Paired cold-weight benchmark for the native dense Xe2 BF16 GEMM.

Rotate at least 512 MiB of physical weight addresses to approximate separate
transformer layers. Timings include graph replay device work only. No serving
speedup can be inferred from this microbenchmark alone.

Attention projection cases (M=256):
  python benchmark/bench_bf16_dense_gemm_xpu.py --m 256 --n 2560 --k 1536
  python benchmark/bench_bf16_dense_gemm_xpu.py --m 256 --n 1536 --k 2048
  python benchmark/bench_bf16_dense_gemm_xpu.py --m 256 --n 1536 --k 4096
"""

import argparse
import json
import math
import statistics

import torch
import torch.nn.functional as F
from sgl_kernel import bf16_dense_gemm_xe20


def capture(operation, count):
    for index in range(count):
        operation(index)
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        for _ in range(4):
            for index in range(count):
                output = operation(index)
    return graph, output


def measure(graph, count):
    for _ in range(2):
        graph.replay()
    torch.xpu.synchronize()
    start, end = torch.xpu.Event(enable_timing=True), torch.xpu.Event(
        enable_timing=True
    )
    start.record()
    for _ in range(10):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (10 * 4 * count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=1536)
    parser.add_argument("--k", type=int, default=12288)
    args = parser.parse_args()
    torch.manual_seed(42)
    a = torch.randn(args.m, args.k, device="xpu", dtype=torch.bfloat16)
    weight = torch.randn(args.n, args.k, device="xpu", dtype=torch.bfloat16) * 0.02
    count = max(4, math.ceil(512 * 1024**2 / (weight.numel() * 2)))
    banks = [weight.clone() for _ in range(count)]
    assert len({w.data_ptr() for w in banks}) == count
    actual = bf16_dense_gemm_xe20(a, banks[0])
    reference = F.linear(a.float(), banks[0].float())
    torch.testing.assert_close(actual.float(), reference, atol=0.01, rtol=0.01)
    operations = {
        "torch": lambda index: F.linear(a, banks[index]),
        "dense": lambda index: bf16_dense_gemm_xe20(a, banks[index]),
    }
    graphs = {name: capture(operation, count) for name, operation in operations.items()}
    for name, (graph, output) in graphs.items():
        output.fill_(float("nan"))
        graph.replay()
        torch.xpu.synchronize()
        assert (
            torch.isfinite(output).all().item()
        ), f"{name} graph did not write its output"
        torch.testing.assert_close(output.float(), reference, atol=0.01, rtol=0.01)
    rounds = [
        {"backend": name, "device_us": measure(graphs[name][0], count)}
        for name in ("torch", "dense", "dense", "torch")
    ]
    means = {
        name: statistics.mean(x["device_us"] for x in rounds if x["backend"] == name)
        for name in operations
    }
    print(
        json.dumps(
            {
                "shape": vars(args),
                "weight_dtype": "bfloat16",
                "bank_count": count,
                "rotating_bytes": count * weight.numel() * 2,
                "device": torch.xpu.get_device_name(),
                "torch_version": torch.__version__,
                "rounds": rounds,
                "means_us": means,
                "lower_percent": 100 * (1 - means["dense"] / means["torch"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
