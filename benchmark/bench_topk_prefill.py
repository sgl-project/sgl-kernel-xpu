import argparse
import json

import torch
import triton.testing
from sgl_kernel import fast_topk_v2
from torch.profiler import ProfilerActivity, profile


def get_effective_bytes(score, lengths, output, row_starts=None):
    """Count logical I/O, excluding scratch traffic and score rescans."""
    score_elements = int(lengths[lengths > output.shape[1]].sum())
    metadata_bytes = lengths.nbytes
    if row_starts is not None:
        metadata_bytes += row_starts.nbytes
    return score_elements * score.element_size() + metadata_bytes + output.nbytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="Optional JSON results path")
    parser.add_argument(
        "--warmup", type=float, default=25, help="Triton warmup budget in milliseconds"
    )
    parser.add_argument(
        "--rep", type=float, default=100, help="Triton timing budget in milliseconds"
    )
    parser.add_argument("--check-dispatch", action="store_true")
    parser.add_argument("--suite", choices=["default", "dispatch"], default="default")
    parser.add_argument(
        "--batches",
        nargs="+",
        type=int,
        default=[1, 4, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
    )
    parser.add_argument(
        "--widths", nargs="+", type=int, default=[2048, 4096, 4097, 8192, 10240]
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.rep <= 0:
        parser.error("--warmup must be nonnegative and --rep must be positive")
    print(torch.xpu.get_device_properties(0), flush=True)
    cases = [
        ("decode_8k", 4, 8192, 512, "normal", 0),
        ("small_batch", 64, 8192, 512, "normal", 0),
        ("medium_batch", 128, 8192, 512, "normal", 0),
        ("prefill_8k", 8192, 2048, 512, "causal", 0),
        ("narrow_decode", 4, 4096, 512, "normal", 0),
        ("narrow_batch128", 128, 4096, 512, "normal", 0),
        ("narrow_batch1024", 1024, 4096, 512, "normal", 0),
        ("narrow_causal", 4096, 4096, 512, "triangle", 0),
        ("narrow_k2048", 4096, 4096, 2048, "triangle", 0),
        ("narrow_normal", 4096, 4096, 512, "normal", 0),
        ("narrow_shifted", 4096, 4089, 512, "normal", 7),
        ("narrow_uniform", 4096, 4096, 512, "uniform", 0),
        ("narrow_equal", 4096, 4096, 512, "equal", 0),
        ("narrow_one_bin", 4096, 4096, 512, "one_bin", 0),
        ("prefill_32k_first", 4096, 8192, 512, "first", 0),
        ("prefill_32k_middle", 4096, 8192, 512, "middle", 0),
        ("prefill_32k_last", 4096, 8192, 512, "last", 0),
        ("prefill_32k_shifted", 4096, 8192, 512, "last", 7),
        ("ragged_8k", 1024, 8192, 512, "normal", 7),
        ("k2048", 4096, 8192, 2048, "last", 0),
        ("uniform", 4096, 8192, 512, "uniform", 0),
        ("equal", 4096, 8192, 512, "equal", 0),
        ("one_bin", 4096, 8192, 512, "one_bin", 0),
    ]
    if args.suite == "dispatch":
        cases = []
        for batch in args.batches:
            for width in args.widths:
                for k in [512, 2048]:
                    distributions = (
                        ["normal", "uniform", "ramp"] if width > k else ["normal"]
                    )
                    for distribution in distributions:
                        cases.append(
                            (
                                f"b{batch}_w{width}_k{k}_{distribution}",
                                batch,
                                width,
                                k,
                                distribution,
                                0,
                            )
                        )
    torch.manual_seed(7)
    results = []
    for name, batch, width, k, distribution, start in cases:
        score = torch.randn(batch, width + start, device="xpu")
        if distribution == "uniform":
            score.uniform_()
        elif distribution == "equal":
            score.zero_()
        elif distribution == "one_bin":
            score = 1 + torch.rand_like(score) * 0.001
        if distribution == "ramp":
            lengths = ((torch.arange(batch, device="xpu") + 1) * width // batch).clamp(
                min=1
            )
        elif distribution == "triangle":
            lengths = torch.arange(batch, device="xpu") + 1
        elif distribution == "causal":
            lengths = ((torch.arange(batch, device="xpu") + 1) // 4).clamp(min=1)
        elif distribution in ("first", "middle", "last"):
            offset = {"first": 1, "middle": 16385, "last": 28673}[distribution]
            lengths = ((torch.arange(batch, device="xpu") + offset) // 4).clamp(min=1)
        else:
            lengths = torch.full((batch,), width, device="xpu")
        lengths = lengths.int()
        starts = torch.full_like(lengths, start) if start else None

        def run():
            return fast_topk_v2(score, lengths, k, row_starts=starts)

        output = run().cpu()
        lengths_cpu = lengths.cpu()
        for row in torch.linspace(0, batch - 1, min(batch, 16)).long().tolist():
            n = int(lengths_cpu[row])
            indices = output[row, : min(n, k)].long()
            assert indices.unique().numel() == min(n, k)
            assert ((indices >= 0) & (indices < n)).all()
            values = score[row, start : start + n].cpu()
            torch.testing.assert_close(
                values[indices].sort().values,
                values.topk(min(n, k)).values.sort().values,
                rtol=0,
                atol=0,
            )
            if n < k:
                assert (output[row, n:] == -1).all()
        ms, p20_ms, p80_ms = triton.testing.do_bench(
            run, warmup=args.warmup, rep=args.rep, quantiles=[0.5, 0.2, 0.8]
        )
        assert ms > 0, (name, ms)
        total_bytes = get_effective_bytes(score, lengths_cpu, output, starts)
        bandwidth_gbs = total_bytes / (ms / 1e3) / 1e9
        events = []
        if args.check_dispatch:
            # Profile one untimed call for kernel identity, not latency.
            torch.xpu.synchronize()
            with profile(activities=[ProfilerActivity.XPU]) as prof:
                run()
                torch.xpu.synchronize()
            events = [e for e in prof.key_averages() if "FastTopK" in e.key]
            assert events, [e.key for e in prof.key_averages()]
            physical_width = score.shape[1]
            expected = (batch >= 2048 and 4096 < physical_width <= 6144) or (
                batch >= 4096 and 6144 < physical_width <= 10240
            )
            actual = any("FastTopKMediumPrefillKernel" in e.key for e in events)
            assert actual == expected, (name, [e.key for e in events])
            expected = (batch >= 256 and physical_width <= 2048) or (
                batch >= 512 and 2048 < physical_width <= 4096
            )
            actual = any("FastTopKNarrowPrefillKernel" in e.key for e in events)
            assert actual == expected, (name, [e.key for e in events])
        us = ms * 1e3
        results.append(
            {
                "case": name,
                "batch": batch,
                "width": score.shape[1],
                "k": k,
                "distribution": distribution,
                "us": us,
                "p20_us": p20_ms * 1e3,
                "p80_us": p80_ms * 1e3,
                "effective_bytes": total_bytes,
                "bandwidth_gbs": bandwidth_gbs,
                "kernels": [e.key for e in events],
            }
        )
        print(f"{name:24s} {us:9.2f} us {bandwidth_gbs:9.2f} GB/s", flush=True)
    if args.output is not None:
        with open(args.output, "w") as f:
            json.dump(
                {
                    "cache": "cold",
                    "timing": "triton.testing.do_bench",
                    "statistic": "median",
                    "warmup_ms": args.warmup,
                    "rep_ms": args.rep,
                    "bandwidth": "logical I/O; excludes scratch traffic and score rescans",
                    "results": results,
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
