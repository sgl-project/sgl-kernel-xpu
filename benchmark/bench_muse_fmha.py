"""Muse-Spark B8-only source/port device-event and independent-FP32 validation."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Case_xe35:
    name: str
    requests: int
    query_length: int
    cache_length: int
    window: int


def production_cases_xe35():
    for requests in (1, 7, 8):
        for window in (-1, 2047):
            yield Case_xe35(f"prefill-wave{requests}-w{window}", requests, 4096, 4096, window)
    for step in range(1, 17):
        for window in (-1, 2047):
            yield Case_xe35(f"decode-osl{step}-w{window}", 8, 1, 4096 + step, window)


def arguments_xe35(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving-batch", type=int, default=8)
    parser.add_argument("--isl", type=int, default=4096)
    parser.add_argument("--osl", type=int, default=16)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--guard-only", action="store_true")
    parser.add_argument("--mode", choices=("bench", "accuracy"), default="bench")
    parser.add_argument("--part", choices=("all", "decode", "prefill"), default="all")
    parser.add_argument("--cases", default="")
    parser.add_argument("--label", default="port")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--paired-dir", type=Path)
    parser.add_argument("--write-paired", action="store_true")
    parser.add_argument("--reference", choices=("full", "sampled"), default="full")
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=21)
    parser.add_argument("--replays", type=int, default=8)
    args = parser.parse_args(argv)
    fixed = {"serving_batch": 8, "isl": 4096, "osl": 16, "tp": 1, "ep": 1}
    for key, expected in fixed.items():
        if getattr(args, key) != expected:
            parser.error(f"{key} must be {expected}; rejected before GPU initialization")
    for key, expected in {"BATCHES": "8", "SERVING_BATCH": "8", "BATCH_SIZE": "8", "ISL": "4096", "OSL": "16", "TP": "1", "EP": "1"}.items():
        if os.environ.get(key, expected).strip() != expected:
            parser.error(f"{key} must be {expected}; rejected before GPU initialization")
    if min(args.chunk, args.warmup, args.iterations, args.replays) <= 0:
        parser.error("chunk and timing counts must be positive")
    if not args.guard_only and args.output is None:
        parser.error("--output is required")
    selected = [case for case in production_cases_xe35() if args.part == "all" or case.name.startswith(args.part)]
    if args.cases:
        requested = set(args.cases.split(","))
        if requested - {case.name for case in selected}:
            parser.error("--cases contains a non-production case")
        selected = [case for case in selected if case.name in requested]
    return args, selected


def tensors_xe35(case, torch):
    torch.manual_seed(42)
    pages_per_request = (case.cache_length + 63) // 64
    pages = case.requests * pages_per_request
    query = torch.randn(case.requests * case.query_length, 128, 64, device="xpu", dtype=torch.bfloat16)
    key = torch.randn(pages, 64, 16, 64, device="xpu", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    table = torch.randperm(pages, device="xpu", dtype=torch.int32).view(case.requests, pages_per_request)
    lengths = torch.full((case.requests,), case.cache_length, device="xpu", dtype=torch.int32)
    offsets = torch.arange(case.requests + 1, device="xpu", dtype=torch.int32) * case.query_length
    output = torch.empty_like(query)
    return query, key, value, table, lengths, offsets, output


def launch_xe35(case, tensors, attention):
    query, key, value, table, lengths, offsets, output = tensors
    return attention(query, key, value, cache_seqlens=lengths, page_table=table,
                     cu_seqlens_q=offsets, max_seqlen_q=case.query_length,
                     max_seqlen_k=case.cache_length, softmax_scale=64**-0.5,
                     causal=True, window_size=(case.window, 0),
                     rotary_interleaved=False, num_splits=0, out=output)


def measure_xe35(launch, args, torch):
    for _ in range(args.warmup):
        launch()
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        for _ in range(args.replays):
            launch()
    torch.xpu.synchronize()
    for _ in range(args.warmup):
        graph.replay()
    torch.xpu.synchronize()
    samples = []
    for _ in range(args.iterations):
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.xpu.synchronize()
        samples.append(start.elapsed_time(end) / args.replays)
    return samples


def reference_xe35(case, tensors, args, torch):
    query, key, value, table, _, _, actual = tensors
    max_abs, checked = 0.0, 0
    sampled = [0, 63, 127, 2046, 2047, 2048, 4095]
    positions = list(range(case.query_length)) if args.reference == "full" or case.query_length == 1 else sampled
    key_positions = torch.arange(case.cache_length, device="xpu")
    for request in range(case.requests):
        keys = key[table[request].long()].reshape(-1, 16, 64)[:case.cache_length].float().transpose(0, 1)
        values = value[table[request].long()].reshape(-1, 16, 64)[:case.cache_length].float().transpose(0, 1)
        for offset in range(0, len(positions), args.chunk):
            local_positions = positions[offset:offset + args.chunk]
            indices = torch.tensor(local_positions, device="xpu")
            queries = query[request * case.query_length + indices].float().reshape(-1, 16, 8, 64).permute(1, 2, 0, 3)
            scores = torch.matmul(queries, keys[:, None].transpose(-1, -2)) * (64**-0.5)
            absolute_positions = indices + case.cache_length - case.query_length
            mask = key_positions[None, :] > absolute_positions[:, None]
            if case.window >= 0:
                mask |= key_positions[None, :] < absolute_positions[:, None] - case.window
            probabilities = scores.masked_fill(mask, -float("inf")).softmax(dim=-1)
            expected = torch.matmul(probabilities, values[:, None]).permute(2, 0, 1, 3).reshape(-1, 128, 64)
            observed = actual[request * case.query_length + indices].float()
            torch.testing.assert_close(observed, expected, atol=5e-3, rtol=5e-3)
            max_abs = max(max_abs, (observed - expected).abs().max().item())
            checked += observed.numel()
    return {"reference": "independent FP32 masked matmul/softmax/matmul on actual BF16 inputs",
            "coverage": "full" if len(positions) == case.query_length else "sampled-query-rows-all-heads-all-requests",
            "query_positions": "all" if len(positions) == case.query_length else positions,
            "checked_elements": checked, "max_abs": max_abs, "atol": 5e-3, "rtol": 5e-3}


def provenance_xe35(package, torch):
    with open("/proc/self/maps") as maps:
        paths = sorted({line.split()[-1] for line in maps if "/sgl_kernel/" in line and ".so" in line})
    return {"package": str(package.__file__), "torch": torch.__version__,
            "device": str(torch.xpu.get_device_properties(0)),
            "reference_fp32_matmul_precision": torch.get_float32_matmul_precision(),
            "gpu_affinity": os.environ.get("ZE_AFFINITY_MASK"),
            "libraries": [{"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()} for path in paths]}


def is_fmha_arch_xe35(torch_module, is_xe3_arch):
    return hasattr(torch_module, "xpu") and torch_module.xpu.is_available() and is_xe3_arch()


def main_xe35(argv=None):
    args, cases = arguments_xe35(argv)
    if args.guard_only:
        return
    if os.environ.get("ZE_AFFINITY_MASK") != "5":
        raise SystemExit("requires the CRI wrapper with --gpu-lock --gpu 5 before GPU initialization")
    import torch
    import sgl_kernel
    from sgl_kernel.utils import is_xe3_arch

    if not is_fmha_arch_xe35(torch, is_xe3_arch):
        result = {"label": args.label, "status": "SKIP", "cases": [],
                  "reason": "Xe35-only FMHA cases require is_xe3_arch(): major == 3 and minor == 5"}
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        return
    from sgl_kernel.flash_attn import flash_attn_with_kvcache

    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(8)
    results = {"label": args.label, "workload": "B8 ISL4096 OSL16 TP1 EP1 BF16 Hq128 Hkv16 D64 page64",
               "architecture_guard": "is_xe3_arch(): major == 3 and minor == 5",
               "timing": "XPU device events around 8-launch graph replay; no host wall-time", "cases": []}
    with torch.inference_mode():
        for case in cases:
            torch.xpu.synchronize()
            time.sleep(2)
            tensors = tensors_xe35(case, torch)
            launch = lambda: launch_xe35(case, tensors, flash_attn_with_kvcache)
            record = {"name": case.name, "requests": case.requests, "query_length": case.query_length,
                      "cache_length": case.cache_length, "window": case.window}
            if args.mode == "bench":
                samples = measure_xe35(launch, args, torch)
                record.update(samples_ms=samples, median_ms=statistics.median(samples), warmup=args.warmup,
                              iterations=args.iterations, graph_replays=args.replays)
            else:
                launch()
                torch.xpu.synchronize()
                record.update(reference_xe35(case, tensors, args, torch))
                if args.paired_dir:
                    path = args.paired_dir / (case.name + ".pt")
                    if args.write_paired:
                        args.paired_dir.mkdir(parents=True, exist_ok=True)
                        torch.save(tensors[-1].cpu(), path)
                        record["paired"] = "source-output-saved"
                    else:
                        source = torch.load(path, map_location="cpu", weights_only=True)
                        actual = tensors[-1].cpu()
                        torch.testing.assert_close(actual, source, atol=5e-3, rtol=5e-3)
                        record["paired_max_abs"] = (actual.float() - source.float()).abs().max().item()
                        record["paired"] = "PASS"
                record["accuracy"] = "PASS"
            results["cases"].append(record)
            print(json.dumps(record), flush=True)
            torch.xpu.synchronize()
            del tensors, launch
            args.output.write_text(json.dumps(results, indent=2) + "\n")
    results["provenance"] = provenance_xe35(sgl_kernel, torch)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results["provenance"]), flush=True)


if __name__ == "__main__":
    main_xe35()
