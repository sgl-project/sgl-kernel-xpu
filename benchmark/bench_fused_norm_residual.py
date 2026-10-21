"""Muse-Spark B8 residual normalization: CPU oracle and XPU graph/event timing."""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import os
import statistics
import sys
import time
from pathlib import Path

POST_EPS = 1e-8
PRE_EPS = 1e-5


def parse_args_xe35():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, choices=[8], default=8)
    parser.add_argument("--isl", type=int, choices=[4096], default=4096)
    parser.add_argument("--osl", type=int, choices=[16], default=16)
    parser.add_argument("--tp", type=int, choices=[1], default=1)
    parser.add_argument("--ep", type=int, choices=[1], default=1)
    parser.add_argument("--provider", choices=["source", "port", "port-native"], default="port")
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--accuracy-only", action="store_true")
    parser.add_argument("--decode-focus", action="store_true")
    parser.add_argument("--cpu-submit-trace", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    for name in ("BATCHES", "BATCH", "SERVING_BATCH", "BATCH_SIZE", "SGLANG_BATCH_SIZE", "SGLANG_MUSE_BATCH_SIZE",
                 "MUSE_BATCH_SIZE", "SERVING_BATCH_SIZE", "MUSE_BS", "BS"):
        value = os.environ.get(name)
        if value is not None and value != "8":
            parser.error(f"{name} must be 8 for this B8-only entry point")
    for name, expected in (("ISL", "4096"), ("OSL", "16"), ("TP", "1"), ("EP", "1")):
        if os.environ.get(name, expected) != expected:
            parser.error(f"{name} must be {expected} for this production entry point")
    if min(args.iterations, args.samples) <= 0 or args.warmup < 0:
        parser.error("positive iterations/samples and nonnegative warmup are required")
    if args.accuracy_only:
        args.check = True
    return args


def round_bfloat16_xe35(values):
    """Round an independent FP64 oracle directly to BF16, without an FP32 double-round."""
    high = values.float()
    midpoint = ((high.view(torch.int32) & 0xFFFF) == 0x8000) & (values != high.double())
    if midpoint.any():
        direction = torch.where(
            values[midpoint] > high[midpoint].double(), float("inf"), float("-inf")
        ).float()
        high[midpoint] = torch.nextafter(high[midpoint], direction)
    return high.bfloat16()


def reference_xe35(
    branch: torch.Tensor,
    stream: torch.Tensor,
    post_weight: torch.Tensor,
    pre_weight: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    post_norm_input_scale: bool,
    round_pre_norm_input: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent FP64 mathematics with the model's BF16 boundaries and FP32 stream store."""
    post_input = branch.double()
    if post_weight.numel() and post_norm_input_scale:
        post_input = round_bfloat16_xe35(post_input * post_weight.double()).double()
    post_eps = torch.tensor(POST_EPS, dtype=torch.float32, device="cpu").double()
    pre_eps = torch.tensor(PRE_EPS, dtype=torch.float32, device="cpu").double()
    post_rstd = torch.rsqrt(post_input.square().mean(dim=-1, keepdim=True) + post_eps)
    normed_branch = post_input * post_rstd
    if post_weight.numel() and not post_norm_input_scale:
        normed_branch = normed_branch * post_weight.double()
    normed_branch = round_bfloat16_xe35(normed_branch)
    rounded_post_norm = normed_branch.float()
    new_stream = (
        alpha.double() * stream.double() + beta.double() * rounded_post_norm.double()
    ).float()
    pre_norm_input = (
        new_stream.bfloat16().double() if round_pre_norm_input else new_stream.double()
    )
    pre_rstd = torch.rsqrt(pre_norm_input.square().mean(dim=-1, keepdim=True) + pre_eps)
    output = (pre_norm_input * pre_rstd * pre_weight.double()).float()
    return output, new_stream, normed_branch



def check_accuracy_xe35(inputs, output, updated, widened):
    branch, stream, post, pre, alpha, beta, post_scale, round_pre = inputs
    cpu_weights = [value.cpu() for value in (post, pre, alpha, beta)]
    actual_output = torch.empty_like(output, device="cpu")
    actual_stream = torch.empty_like(updated, device="cpu")
    max_norm_error = 0.0
    max_stream_error = 0.0
    for start in range(0, branch.shape[0], 512):
        expected_output, expected_stream, _ = reference_xe35(
            branch[start:start + 512].cpu(), stream[start:start + 512].cpu(),
            *cpu_weights, post_scale, round_pre,
        )
        norm_bits = output[start:start + 512].cpu()
        norm_slice = norm_bits.float()
        stream_slice = updated[start:start + 512].cpu()
        torch.testing.assert_close(norm_slice, expected_output, rtol=5e-3, atol=5e-3)
        torch.testing.assert_close(stream_slice, expected_stream, rtol=1e-5, atol=1e-5)
        max_norm_error = max(max_norm_error, (norm_slice - expected_output).abs().max().item())
        max_stream_error = max(max_stream_error, (stream_slice - expected_stream).abs().max().item())
        if widened is not None:
            torch.testing.assert_close(widened[start:start + 512].cpu(), norm_slice, rtol=0, atol=0)
        actual_output[start:start + 512].copy_(norm_bits)
        actual_stream[start:start + 512].copy_(stream_slice)
    accuracy = {
        "norm_max_abs": max_norm_error,
        "stream_max_abs": max_stream_error,
        "norm_tolerance": [5e-3, 5e-3],
        "stream_tolerance": [1e-5, 1e-5],
        "norm_sha256": hashlib.sha256(actual_output.view(torch.int16).numpy().tobytes()).hexdigest(),
        "stream_sha256": hashlib.sha256(actual_stream.numpy().tobytes()).hexdigest(),
    }
    return accuracy, {"norm": actual_output, "stream": actual_stream}


def run_case_xe35(args, invoke_op, rows, post_mode, round_pre, output_fp32):
    torch.manual_seed(42)
    hidden_size = 8192
    branch = torch.randn(rows, hidden_size, device="xpu", dtype=torch.bfloat16)
    stream = torch.randn(rows, hidden_size, device="xpu", dtype=torch.float32)
    post = (torch.empty(0, device="xpu", dtype=torch.bfloat16) if post_mode == "none"
            else torch.randn(hidden_size, device="xpu", dtype=torch.bfloat16))
    pre = torch.randn(hidden_size, device="xpu", dtype=torch.bfloat16)
    alpha = torch.rand(hidden_size, device="xpu", dtype=torch.float32)
    beta = torch.rand(hidden_size, device="xpu", dtype=torch.float32)
    output = torch.empty_like(branch)
    updated = torch.empty_like(stream)
    widened = torch.empty_like(stream) if output_fp32 else None

    def invoke_xe35():
        invoke_op(branch, stream, post, pre, alpha, beta, output, updated,
                  POST_EPS, PRE_EPS, post_mode == "input_scale", round_pre, widened)

    invoke_xe35()
    torch.xpu.synchronize()
    print(f"case_start rows={rows} post_mode={post_mode} round_pre={round_pre} output_fp32={output_fp32} "
          f"allocated_bytes={torch.xpu.memory_allocated()} reserved_bytes={torch.xpu.memory_reserved()}", flush=True)
    result = {
        "provider": args.provider, "rows": rows, "hidden_size": hidden_size,
        "acceptance": ("native ABI forwarding diagnostic; not official full package"
                       if args.provider == "port-native" else "normal complete package path"),
        "post_mode": post_mode, "round_pre_norm_input": round_pre,
        "output_fp32": output_fp32,
        "scope": "B8/ISL4096/OSL16/TP1/EP1",
        "device": torch.xpu.get_device_name(0),
        "dtype": "BF16 branch/post/pre/output; FP32 stream/alpha/beta/new_stream",
        "layout": "contiguous row-major", "eps": [POST_EPS, PRE_EPS],
        "cpu_environment": {
            "KMP_BLOCKTIME": os.environ.get("KMP_BLOCKTIME"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "torch_num_threads": torch.get_num_threads(),
            "reference_before_timing": args.check,
        },
    }
    if args.check:
        result["accuracy"], cpu_outputs = check_accuracy_xe35(
            (branch, stream, post, pre, alpha, beta, post_mode == "input_scale", round_pre),
            output, updated, widened,
        )
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            key = f"{rows}_{post_mode}_{int(round_pre)}_{int(output_fp32)}"
            torch.save(cpu_outputs, args.output_dir / f"{key}.pt")
    if not args.accuracy_only:
        for _ in range(args.warmup):
            invoke_xe35()
        torch.xpu.synchronize()
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            for _ in range(args.iterations):
                invoke_xe35()
        torch.xpu.synchronize()
        for _ in range(3):
            graph.replay()
        torch.xpu.synchronize()
        samples = []
        cpu_samples = []
        for _ in range(args.samples):
            start = torch.xpu.Event(enable_timing=True)
            end = torch.xpu.Event(enable_timing=True)
            enqueue_begin = time.perf_counter_ns() if args.cpu_submit_trace else 0
            start.record()
            start_done = time.perf_counter_ns() if args.cpu_submit_trace else 0
            replay_begin = time.perf_counter_ns() if args.cpu_submit_trace else 0
            graph.replay()
            replay_done = time.perf_counter_ns() if args.cpu_submit_trace else 0
            end_begin = time.perf_counter_ns() if args.cpu_submit_trace else 0
            end.record()
            enqueue_done = time.perf_counter_ns() if args.cpu_submit_trace else 0
            end.synchronize()
            event_interval = start.elapsed_time(end) * 1000
            samples.append(event_interval / args.iterations)
            if args.cpu_submit_trace:
                cpu_samples.append({
                    "start_record_us": (start_done - enqueue_begin) / 1000,
                    "start_to_replay_cpu_gap_us": (replay_begin - start_done) / 1000,
                    "graph_submit_us": (replay_done - replay_begin) / 1000,
                    "replay_to_end_cpu_gap_us": (end_begin - replay_done) / 1000,
                    "end_record_us": (enqueue_done - end_begin) / 1000,
                    "cpu_enqueue_span_us": (enqueue_done - enqueue_begin) / 1000,
                    "device_event_interval_us": event_interval,
                })
        result.update(samples_us=samples, p50_us=statistics.median(samples),
                      iterations=args.iterations, warmup=args.warmup,
                      timing="XPU event interval across graph replay; may include CPU enqueue gaps; not exclusive kernel time")
        if args.cpu_submit_trace:
            result["cpu_enqueue_samples"] = cpu_samples
    print(json.dumps(result), flush=True)


def main_xe35():
    args = parse_args_xe35()
    global torch
    import torch
    if not torch.xpu.is_available():
        raise RuntimeError("Xe35 residual normalization requires XPU availability")
    import sgl_kernel
    from sgl_kernel.utils import is_xe3_arch
    if not is_xe3_arch():
        raise RuntimeError("Xe35 residual normalization requires compute capability 3.5")
    if args.provider == "port":
        invoke_op = sgl_kernel.fused_post_norm_residual_pre_norm_xe35
    else:
        invoke_op = torch.ops.sgl_kernel.fused_post_norm_residual_pre_norm
    print(f"package={sgl_kernel.__file__} op={invoke_op}", flush=True)
    assert Path(sgl_kernel.__file__).is_relative_to(Path(sys.prefix))
    print(f"torch={torch.__version__} gpu_mask={os.environ.get('ZE_AFFINITY_MASK')}", flush=True)
    cases = itertools.product(
        [8, 32768], ["none", "weight", "input_scale"], [False, True], [False, True]
    )
    if args.decode_focus:
        cases = [(8, "none", False, False), (8, "weight", False, True),
                 (8, "weight", True, True), (8, "input_scale", False, False)]
    for rows, post_mode, round_pre, output_fp32 in cases:
        run_case_xe35(args, invoke_op, rows, post_mode, round_pre, output_fp32)
        torch.xpu.synchronize()
        gc.collect()
        torch.xpu.empty_cache()
        time.sleep(2)
    libraries = {
        line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
        if any(name in line for name in ("FusedNormResidual", "source_ops_xe35", "common_ops", "libgomp", "libiomp"))
    }
    for library in sorted(libraries):
        print(f"loaded_lib={library} sha256={hashlib.sha256(Path(library).read_bytes()).hexdigest()}", flush=True)
    assert not any("source_ops_xe35" in library for library in libraries)
    assert any("/sgl_kernel/libsgl-ops-sycl-FusedNormResidual.so" in library for library in libraries)
    assert any("common_ops" in library for library in libraries)


if __name__ == "__main__":
    main_xe35()
