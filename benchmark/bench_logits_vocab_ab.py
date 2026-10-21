"""Device-time A/B against the preserved muse_logits_vocab native operators."""

import argparse
import json
import math
import os
import time
from pathlib import Path
from statistics import median


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving-batch", type=int, choices=(8,), default=8)
    parser.add_argument("--baseline-library", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=8192)
    parser.add_argument("--vocab", type=int, default=202048)
    parser.add_argument("--softcap-widths", type=int, nargs="+")
    parser.add_argument(
        "--embedding-layout", choices=("production", "historical"), default="production"
    )
    parser.add_argument("--tokens", type=int, nargs="+", default=[8, 32, 32768])
    args = parser.parse_args()
    if any(
        os.environ.get(name, "8") != "8"
        for name in ("BATCHES", "BATCH_SIZE", "SERVING_BATCH")
    ):
        parser.error("Muse-Spark serving workload requires B=8")
    if (
        min(
            args.reps,
            args.rounds,
            args.hidden,
            args.vocab,
            *args.tokens,
            *(args.softcap_widths or []),
        )
        <= 0
    ):
        parser.error("dimensions, reps, and rounds must be positive")
    return args


def validate_native_kernel_order(kernels, names, reps):
    if len(kernels) != len(names) * reps:
        raise RuntimeError(
            f"Expected one kernel per native call, got {len(kernels)} for {len(names) * reps} calls"
        )
    for position, name in enumerate(names):
        expected_baseline = name == "baseline"
        for event in kernels[position :: len(names)]:
            if ("logits_vocab_baseline" in event["name"]) != expected_baseline:
                raise RuntimeError(
                    f"{name} launched unexpected kernel {event['name']}; baseline and candidate must have distinct SYCL kernel IDs"
                )


def embedding_buffer_order(round_index):
    return (0, 1) if (round_index // 2) % 2 == 0 else (1, 0)


def validate_latency_ratio(baseline_us, candidate_us):
    if not all(math.isfinite(value) and value > 0 for value in (baseline_us, candidate_us)):
        raise RuntimeError("native kernel durations must be finite and positive")
    if candidate_us > baseline_us:
        raise RuntimeError(
            f"candidate median {candidate_us} us exceeds baseline {baseline_us} us"
        )


def main():
    args = parse_args()
    import tempfile

    import sgl_kernel
    import torch

    if not torch.xpu.is_available() or not sgl_kernel.is_xe3_arch():
        raise RuntimeError("Xe35 is required")
    torch.ops.load_library(str(args.baseline_library.resolve()))
    candidate = torch.ops.sgl_kernel
    baseline = torch.ops.sgl_kernel_baseline
    regressions = []
    torch.manual_seed(20261021)
    print(
        json.dumps(
            {
                "device": torch.xpu.get_device_name(),
                "torch": torch.__version__,
                "serving_batch": 8,
                "isl": 4096,
                "osl": 1024,
                "tp": 4,
                "ep": 1,
                "baseline_revision": "448f0c0cf6cd13de7c2d6ec28de1d392039fdaca",
                "baseline_kernel_namespace": "at::native::xpu::logits_vocab_baseline",
                "timing": "interleaved torch.profiler XPU kernel duration, us",
                "reps": args.reps,
                "rounds": args.rounds,
                "embedding_layout": args.embedding_layout,
            }
        ),
        flush=True,
    )

    def report_result(result):
        try:
            validate_latency_ratio(result["baseline_us"], result["candidate_us"])
            result["performance_gate"] = "pass"
        except RuntimeError as error:
            result["performance_gate"] = "fail"
            regressions.append(f"{result['op']} {result['shape']}: {error}")
        print(json.dumps(result), flush=True)

    def measure_pair(old_call, new_call, round_setup=None):
        samples = {"baseline": [], "candidate": []}
        kernel_names = {"baseline": set(), "candidate": set()}
        round_ratios = []
        for round_index in range(args.rounds):
            if round_setup is not None:
                round_setup(round_index)
            calls = [("baseline", old_call), ("candidate", new_call)]
            if round_index % 2:
                calls.reverse()
            for _ in range(10):
                for _, call in calls:
                    call()
            torch.xpu.synchronize()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.XPU,
                ]
            ) as profiler:
                for _ in range(args.reps):
                    for _, call in calls:
                        call()
                torch.xpu.synchronize()
            with tempfile.TemporaryDirectory() as directory:
                trace = Path(directory) / "trace.json"
                profiler.export_chrome_trace(str(trace))
                events = json.loads(trace.read_text())["traceEvents"]
            kernels = sorted(
                (event for event in events if event.get("cat") == "kernel"),
                key=lambda event: event["ts"],
            )
            validate_native_kernel_order(
                kernels, [name for name, _ in calls], args.reps
            )
            round_samples = {}
            for position, (name, _) in enumerate(calls):
                values = [event["dur"] for event in kernels[position::2]]
                kernel_names[name].update(
                    event["name"] for event in kernels[position::2]
                )
                samples[name].extend(values)
                round_samples[name] = values
            round_ratios.append(
                median(round_samples["candidate"]) / median(round_samples["baseline"])
            )
        return {
            "baseline_us": median(samples["baseline"]),
            "candidate_us": median(samples["candidate"]),
            "candidate_over_baseline": median(samples["candidate"])
            / median(samples["baseline"]),
            "round_ratios": round_ratios,
            "kernel_names": {
                name: sorted(values) for name, values in kernel_names.items()
            },
            "samples_us": samples,
        }

    for width in args.softcap_widths or [args.vocab, args.vocab // 4]:
        torch.xpu.synchronize()
        time.sleep(2)
        source = torch.randn(8, width, device="xpu", dtype=torch.float32) * 10
        old_logits, new_logits = source.clone(), source.clone()
        expected = (20 * torch.tanh(source.cpu().double() / 20)).float()
        old_call = lambda: baseline.softcap_logits(old_logits, 20.0)
        new_call = lambda: candidate.softcap_logits_xe35(new_logits, 20.0)
        old_call()
        new_call()
        torch.testing.assert_close(old_logits.cpu(), expected, atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(new_logits.cpu(), expected, atol=3e-5, rtol=3e-5)
        result = measure_pair(old_call, new_call)
        result.update(
            op="softcap",
            shape=[8, width],
            dtype="float32",
            logical_bytes=2 * source.numel() * source.element_size(),
        )
        report_result(result)

    shard = args.vocab // 4
    bounds = (0, shard, 0, args.vocab, args.vocab)
    local_rows = shard
    if args.embedding_layout == "historical":
        bounds = (0, 512, 0, 1010, 1026)
        local_rows = 528
    weight = torch.randn(local_rows, args.hidden, device="xpu", dtype=torch.bfloat16)
    for count in args.tokens:
        torch.xpu.synchronize()
        time.sleep(2)
        ids = torch.randint(0, args.vocab, (count,), device="xpu", dtype=torch.int64)
        ids[: min(count, 4)] = torch.tensor(
            [0, shard - 1, shard, -1][: min(count, 4)], device="xpu"
        )
        if args.embedding_layout == "historical":
            values = [0, 1, 511, 512, 1009, 1010, 1025, 1026, -1]
            ids = torch.tensor(
                (values * ((count + len(values) - 1) // len(values)))[:count],
                device="xpu",
                dtype=torch.int64,
            )
        ids_cpu = ids.cpu()
        original = (ids_cpu >= bounds[0]) & (ids_cpu < bounds[1])
        added = (ids_cpu >= bounds[3]) & (ids_cpu < bounds[4])
        expected_invalid = ~(original | added)
        added_offset = bounds[3] - (bounds[1] - bounds[0]) - bounds[2]
        expected_mapped = torch.where(
            original, ids_cpu - bounds[0], torch.where(added, ids_cpu - added_offset, 0)
        )
        old_call = lambda: baseline.vocab_mask(ids, *bounds)
        new_call = lambda: candidate.vocab_mask_xe35(ids, *bounds)
        for call in (old_call, new_call):
            mapped, invalid = call()
            torch.testing.assert_close(mapped.cpu(), expected_mapped, atol=0, rtol=0)
            torch.testing.assert_close(invalid.cpu(), expected_invalid, atol=0, rtol=0)
        result = measure_pair(old_call, new_call)
        result.update(
            op="vocab_mask", shape=[count], dtype="int64", logical_bytes=count * 17
        )
        report_result(result)
        torch.xpu.synchronize()
        time.sleep(2)
        old_output = torch.empty(count, args.hidden, device="xpu", dtype=weight.dtype)
        new_output = torch.empty_like(old_output)
        outputs = (old_output, new_output)
        buffer_slots = [0, 1]
        old_call = lambda: baseline.muse_vocab_parallel_embedding(
            ids, weight, outputs[buffer_slots[0]], *bounds
        )
        new_call = lambda: candidate.vocab_parallel_embedding_xe35(
            ids, weight, outputs[buffer_slots[1]], *bounds
        )
        old_call()
        new_call()
        expected = weight[expected_mapped.to("xpu")].masked_fill(
            expected_invalid.to("xpu").unsqueeze(-1), 0
        )
        torch.testing.assert_close(old_output, expected, atol=0, rtol=0)
        torch.testing.assert_close(new_output, expected, atol=0, rtol=0)

        def select_buffers(round_index):
            buffer_slots[:] = embedding_buffer_order(round_index)

        result = measure_pair(old_call, new_call, select_buffers)
        result.update(
            op="embedding",
            shape=[count, args.hidden],
            weight_shape=list(weight.shape),
            dtype="bfloat16",
            buffer_orders=[embedding_buffer_order(index) for index in range(args.rounds)],
            logical_bytes=(count + int((~expected_invalid).sum())) * args.hidden * 2,
        )
        report_result(result)

    if regressions:
        raise RuntimeError("Native latency regression(s): " + "; ".join(regressions))


if __name__ == "__main__":
    main()
