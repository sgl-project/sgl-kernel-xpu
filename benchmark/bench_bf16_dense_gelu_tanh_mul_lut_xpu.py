"""Paired cold-weight benchmark for the native fused dense Xe2 BF16 gate/up GEMM.

Rotate at least 512 MiB of physical weight addresses to approximate separate
transformer layers. Timings include graph replay device work only. No serving
speedup can be inferred from this microbenchmark alone.

The default invocation measures M128. To measure the optimized prefill shape:
    python benchmark/bench_bf16_dense_gelu_tanh_mul_lut_xpu.py --m 4096 --n 24576 --k 1536
The control here uses the native GELU activation. Comparing against a standalone
lookup activation requires a separate benchmark with that implementation.

"""

import argparse
import hashlib
import json
import math
import statistics

import torch
import torch.nn.functional as F
from sgl_kernel import bf16_dense_gelu_tanh_mul_lut_xe20, gelu_tanh_and_mul


def capture(operation, count):
    for index in range(count):
        operation(index)
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    outputs = []
    with torch.xpu.graph(graph):
        for index in range(count):
            outputs.append(operation(index))
    assert len({output.data_ptr() for output in outputs}) == count
    return graph, outputs


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
    return start.elapsed_time(end) * 1000 / (10 * count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=24576)
    parser.add_argument("--k", type=int, default=1536)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(42)
    a = torch.randn(args.m, args.k, device="xpu", dtype=torch.bfloat16)
    weight = torch.randn(args.n, args.k, device="xpu", dtype=torch.bfloat16) * 0.02
    count = max(4, math.ceil(512 * 1024**2 / (weight.numel() * 2)))
    banks = [weight.clone() for _ in range(count)]
    assert len({w.data_ptr() for w in banks}) == count
    assert all(torch.equal(bank, weight) for bank in banks)
    gates = (
        torch.arange(65536, dtype=torch.int32)
        .to(torch.int16)
        .view(torch.bfloat16)
        .to("xpu")
    )
    table = gelu_tanh_and_mul(
        torch.cat((gates, torch.ones_like(gates))).view(1, -1)
    ).flatten()
    actual = bf16_dense_gelu_tanh_mul_lut_xe20(a, banks[0], table)
    reference = gelu_tanh_and_mul(F.linear(a, banks[0])).float()
    torch.testing.assert_close(actual.float(), reference, atol=0.01, rtol=0.01)
    operations = {
        "torch_native_activation": lambda index: gelu_tanh_and_mul(
            F.linear(a, banks[index])
        ),
        "dense_lut": lambda index: bf16_dense_gelu_tanh_mul_lut_xe20(
            a, banks[index], table
        ),
    }
    graphs = {name: capture(operation, count) for name, operation in operations.items()}
    replay_proof = []
    original = a.clone()
    for scale in (0.5, 1.0):
        a.copy_(original * scale)
        for name, (graph, outputs) in graphs.items():
            expected = operations[name](0)
            for output in outputs:
                output.fill_(float("nan"))
            graph.replay()
            torch.xpu.synchronize()
            hashes = []
            for output in outputs:
                assert torch.equal(output.view(torch.int16), expected.view(torch.int16))
                hashes.append(
                    hashlib.sha256(
                        output.view(torch.uint8).cpu().numpy().tobytes()
                    ).hexdigest()
                )
            replay_proof.append(
                {"backend": name, "scale": scale, "output_sha256": hashes}
            )
    rounds = [
        {"backend": name, "device_us": measure(graphs[name][0], count)}
        for name in (
            "torch_native_activation",
            "dense_lut",
            "dense_lut",
            "torch_native_activation",
        )
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
                "grad_enabled": torch.is_grad_enabled(),
                "inference_mode": torch.is_inference_mode_enabled(),
                "replay_proof": replay_proof,
                "rounds": rounds,
                "means_us": means,
                "lower_percent": 100
                * (1 - means["dense_lut"] / means["torch_native_activation"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
