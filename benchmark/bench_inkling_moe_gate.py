"""Production-shape device benchmark and roofline for Inkling's MoE gate."""

import argparse
import json
import statistics

import torch

HIDDEN = 6144
ROUTED = 256
SHARED = 2
ACTIVE = ROUTED + SHARED
PADDED = 264
TOPK = 6


def _reference_projection(x, weight):
    return torch.mm(x, weight.T, out_dtype=torch.float32)[:, :ACTIVE]


def _reference_topk(logits, bias, global_scale, route_scale, *, return_packed):
    scores = torch.sigmoid(logits.float())
    ranking = scores[:, :ROUTED] + bias
    _, indices = torch.topk(ranking, TOPK, dim=-1, sorted=True)
    routed = scores[:, :ROUTED].gather(1, indices)
    shared = scores[:, ROUTED:ACTIVE]
    scale = (
        route_scale
        * global_scale
        / (routed.sum(dim=1, keepdim=True) + shared.sum(dim=1, keepdim=True))
    )
    routed *= scale
    shared *= scale
    if return_packed:
        weight_bits = routed.to(torch.bfloat16).view(torch.int16).to(torch.int32)
        packed = (indices.to(torch.int32) << 16) | (weight_bits & 0xFFFF)
        return None, None, shared, packed
    return routed, indices.to(torch.int32), shared, None


def _device_bench(fn, warmup, rep, rounds):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    samples = []
    for _ in range(rounds):
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        for _ in range(rep):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / rep)
    return statistics.median(samples), samples


def _roofline(flops, bytes_moved, compute_tops, bandwidth_gbps):
    compute_ms = flops / (compute_tops * 1e12) * 1e3
    memory_ms = bytes_moved / (bandwidth_gbps * 1e9) * 1e3
    return compute_ms, memory_ms, max(compute_ms, memory_ms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--implementation", choices=("kernel", "reference"), default="kernel"
    )
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rep", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--compute-tops", type=float, default=50.0)
    parser.add_argument("--bandwidth-gbps", type=float, default=350.0)
    args = parser.parse_args()
    if args.implementation == "kernel":
        from sgl_kernel import inkling_moe_gate_projection, inkling_moe_gate_topk_renorm

        projection = inkling_moe_gate_projection
        topk = inkling_moe_gate_topk_renorm
    else:
        projection = _reference_projection
        topk = _reference_topk

    results = []
    for dtype in (torch.bfloat16, torch.float16):
        x = torch.randn(args.tokens, HIDDEN, device="xpu", dtype=dtype)
        weight = torch.randn(PADDED, HIDDEN, device="xpu", dtype=dtype)

        projection_flops = 2 * args.tokens * HIDDEN * PADDED
        projection_bytes = (
            args.tokens * HIDDEN + PADDED * HIDDEN
        ) * dtype.itemsize + args.tokens * PADDED * 4
        projection_roof = _roofline(
            projection_flops,
            projection_bytes,
            args.compute_tops,
            args.bandwidth_gbps,
        )
        median, samples = _device_bench(
            lambda: projection(x, weight),
            args.warmup,
            args.rep,
            args.rounds,
        )
        results.append(
            {
                "op": "projection",
                "implementation": args.implementation,
                "dtype": str(dtype),
                "tokens": args.tokens,
                "device_median_ms": median,
                "device_samples_ms": samples,
                "four_layer_ms": median * 4,
                "tflops": projection_flops / (median / 1e3) / 1e12,
                "bytes": projection_bytes,
                "arithmetic_intensity_flop_per_byte": projection_flops
                / projection_bytes,
                "compute_roof_ms": projection_roof[0],
                "memory_roof_ms": projection_roof[1],
                "binding_roof_ms": projection_roof[2],
                "roofline_efficiency": projection_roof[2] / median,
                "target_four_layer_ms": 0.905,
                "target_met": median * 4 <= 0.905,
            }
        )
        del x, weight

    logits = torch.randn(args.tokens, PADDED, device="xpu", dtype=torch.float32)[
        :, :ACTIVE
    ]
    bias = torch.randn(ROUTED, device="xpu", dtype=torch.float32)
    global_scale = torch.tensor([1.25], device="xpu", dtype=torch.float32)
    topk_bytes = (
        args.tokens * ACTIVE * 4
        + ROUTED * 4
        + 4
        + args.tokens * (TOPK * 4 + SHARED * 4)
    )
    topk_roof = _roofline(
        0,
        topk_bytes,
        args.compute_tops,
        args.bandwidth_gbps,
    )
    median, samples = _device_bench(
        lambda: topk(
            logits,
            bias,
            global_scale,
            8.0,
            return_packed=True,
        ),
        args.warmup,
        args.rep,
        args.rounds,
    )
    results.append(
        {
            "op": "topk_renorm",
            "implementation": args.implementation,
            "dtype": "torch.float32",
            "tokens": args.tokens,
            "device_median_ms": median,
            "device_samples_ms": samples,
            "four_layer_ms": median * 4,
            "effective_gbps": topk_bytes / (median / 1e3) / 1e9,
            "bytes": topk_bytes,
            "memory_roof_ms": topk_roof[1],
            "roofline_efficiency": topk_roof[2] / median,
            "target_four_layer_ms": 0.065,
            "target_met": median * 4 <= 0.065,
        }
    )

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
