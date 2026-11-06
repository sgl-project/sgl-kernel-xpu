"""Standalone two-stage ConvRot correctness, latency, memory and trace benchmark."""

import argparse
import gc
import json
import os
import statistics
import time
from pathlib import Path

import torch
from sgl_kernel.convrot import convrot_int8_linear_prequant, convrot_rotate_quantize


def reference_quantize(x, h):
    rotated = (x.reshape(-1, 256) @ h).reshape_as(x)
    scale = (rotated.abs().amax(-1).float() / 127).clamp_min(1e-30)
    math_scale = scale.to(x.dtype)
    math_scale = torch.where(math_scale == 0, torch.finfo(x.dtype).tiny, math_scale)
    q = (rotated / math_scale[:, None]).round().clamp(-128, 127).to(torch.int8)
    return q, scale


def reference_linear(q, weight, scale, weight_scale):
    acc = torch._int_mm(q, weight.T.contiguous())
    rows = max(1, min(q.shape[0], 256 * 1024 * 1024 // (weight.shape[0] * 4)))
    parts = []
    for start in range(0, q.shape[0], rows):
        scales = scale[start : start + rows, None] * weight_scale[None, :]
        parts.append((acc[start : start + rows].float() * scales).bfloat16())
    return torch.cat(parts)


def measure(fn, iterations):
    torch.xpu.synchronize()
    gc.collect()
    torch.xpu.reset_peak_memory_stats()
    allocated = torch.xpu.memory_allocated()
    durations = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = fn()
        torch.xpu.synchronize()
        durations.append((time.perf_counter() - start) * 1000)
        del result
    return {
        "median_ms": statistics.median(durations),
        "samples_ms": durations,
        "extra_peak_allocated_bytes": torch.xpu.max_memory_allocated() - allocated,
    }


def run_case(m, n, k, device, iterations, profile_path):
    with torch.xpu.device(device), torch.inference_mode():
        torch.manual_seed(79)
        x = torch.randn(m, k, device=f"xpu:{device}", dtype=torch.bfloat16)
        weight = torch.randint(-127, 128, (n, k), device=x.device, dtype=torch.int8)
        weight_scale = torch.rand(n, device=x.device) / 128
        h4 = torch.tensor(
            [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
            device=x.device,
            dtype=x.dtype,
        )
        h = h4
        for _ in range(3):
            h = torch.kron(h, h4)
        h = h / 16

        q, scale = convrot_rotate_quantize(x)
        q_ref, scale_ref = reference_quantize(x, h)
        q_mismatches = (q != q_ref).sum().item()
        scale_mismatches = (scale != scale_ref).sum().item()
        q_max_error = (q.to(torch.int16) - q_ref.to(torch.int16)).abs().max().item()
        # A BF16 midpoint crossing can move round-to-even INT8 by two (e.g. 106 -> 108).
        if (
            q_mismatches > max(1, x.numel() * 1e-6)
            or q_max_error > 2
            or scale_mismatches
        ):
            raise AssertionError(
                f"{m,n,k}: quantized data mismatches={q_mismatches}, "
                f"scale mismatches={scale_mismatches}"
            )
        out = convrot_int8_linear_prequant(q, weight, scale, weight_scale)
        output_mismatches = 0
        max_abs_error = 0.0
        eager_output_mismatches = 0
        eager_max_abs_error = 0.0
        error_squared = 0.0
        reference_squared = 0.0
        for start in range(0, m, 512):
            end = min(start + 512, m)
            ref = reference_linear(q[start:end], weight, scale[start:end], weight_scale)
            output_mismatches += (out[start:end] != ref).sum().item()
            max_abs_error = max(
                max_abs_error, (out[start:end] - ref).abs().max().item()
            )
            del ref
            eager_ref = reference_linear(
                q_ref[start:end], weight, scale_ref[start:end], weight_scale
            )
            diff = out[start:end].float() - eager_ref.float()
            eager_output_mismatches += (out[start:end] != eager_ref).sum().item()
            eager_max_abs_error = max(eager_max_abs_error, diff.abs().max().item())
            error_squared += diff.square().sum().item()
            reference_squared += eager_ref.float().square().sum().item()
            del eager_ref, diff
        if output_mismatches:
            raise AssertionError(
                f"{m,n,k}: output mismatches={output_mismatches}, "
                f"max_abs={max_abs_error}"
            )
        relative_l2 = (error_squared / max(reference_squared, 1e-30)) ** 0.5
        if relative_l2 > 1e-4:
            raise AssertionError(f"{m,n,k}: eager output relative L2={relative_l2}")
        del out, q_ref, scale_ref

        def two_stage():
            aq, asc = convrot_rotate_quantize(x)
            return convrot_int8_linear_prequant(aq, weight, asc, weight_scale)

        def eager():
            aq, asc = reference_quantize(x, h)
            return reference_linear(aq, weight, asc, weight_scale)

        for fn in (two_stage, eager):
            warmup = fn()
            torch.xpu.synchronize()
            del warmup
        measurements = {
            "rotate_quantize": measure(lambda: convrot_rotate_quantize(x), iterations),
            "prequant_gemm": measure(
                lambda: convrot_int8_linear_prequant(q, weight, scale, weight_scale),
                iterations,
            ),
            "two_stage": measure(two_stage, iterations),
            "eager_reference": measure(eager, iterations),
        }
        if profile_path:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.XPU,
                ],
                record_shapes=False,
                with_stack=False,
            ) as prof:
                result = two_stage()
                torch.xpu.synchronize()
            prof.export_chrome_trace(str(profile_path))
            del result
        return {
            "shape": {"M": m, "N": n, "K": k},
            "logical_device": device,
            "device_uuid": str(torch.xpu.get_device_properties(device).uuid),
            "correctness": {
                "quantized_data_mismatches": q_mismatches,
                "scale_mismatches": scale_mismatches,
                "quantized_data_max_abs_error": q_max_error,
                "output_mismatches": output_mismatches,
                "max_abs_error": max_abs_error,
                "eager_output_mismatches": eager_output_mismatches,
                "eager_output_max_abs_error": eager_max_abs_error,
                "eager_output_relative_l2": relative_l2,
            },
            "measurements": measurements,
            "speedup_vs_eager": measurements["eager_reference"]["median_ms"]
            / measurements["two_stage"]["median_ms"],
            "trace": str(profile_path) if profile_path else None,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[63744, 66048, 53376])
    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    if min(args.rows) <= 0 or args.iterations <= 0:
        parser.error("rows and iterations must be positive")
    widths = [(5376, 5376), (1792, 5376), (5376, 7168), (3584, 5376)]
    results = {
        "torch_version": torch.__version__,
        "affinity": os.environ.get("ZE_AFFINITY_MASK"),
        "timing": "synchronized host latency; sequential devices, not TP4 or model inference",
        "accuracy_contract": {
            "prequant_gemm": "bit-exact for identical quantized inputs",
            "vs_eager_matmul_rotation": "FP32 summation reorder is not bit-exact",
            "allowed_quantized_mismatch_fraction": 1e-6,
            "allowed_quantized_max_abs_error": 2,
            "allowed_output_relative_l2": 1e-4,
        },
        "cases": [],
    }
    for m in args.rows:
        for k, n in widths:
            index = len(results["cases"])
            device = args.devices[index % len(args.devices)]
            trace = args.trace if index == 0 else None
            result = run_case(m, n, k, device, args.iterations, trace)
            results["cases"].append(result)
            args.output.write_text(json.dumps(results, indent=2))
            print(json.dumps(result), flush=True)
            gc.collect()
            with torch.xpu.device(device):
                torch.xpu.empty_cache()


if __name__ == "__main__":
    main()
