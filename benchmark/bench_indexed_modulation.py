"""Benchmark: indexed_scale_shift_bf16_ / indexed_gate_bf16_  —  SYCL kernel vs
pure-PyTorch reference vs Triton kernel.

Kernel signatures (match unit tests):
    indexed_scale_shift_bf16_(x: Tensor[rows, hidden], shift: Tensor[num_indices, hidden],
                               scale: Tensor[num_indices, hidden], indices: Tensor[rows]) -> Tensor
    indexed_gate_bf16_(x: Tensor[rows, hidden], gate: Tensor[num_indices, hidden],
                       other: Tensor[rows, hidden], indices: Tensor[rows]) -> Tensor

Run:
    python benchmark/bench_indexed_modulation.py
"""

import itertools

import pandas as pd
import torch
import triton
import triton.language as tl
import triton.testing
from sgl_kernel import indexed_gate_bf16_, indexed_scale_shift_bf16_


# ---------------------------------------------------------------------------
# Pure-PyTorch reference  (same logic as tests/test_indexed_modulation.py)
# ---------------------------------------------------------------------------
def _round_bf16_to_fp32(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.bfloat16).to(torch.float32)


def indexed_scale_shift_bf16_torch(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    selected_shift = shift.index_select(0, indices.long()).float()
    selected_scale = scale.index_select(0, indices.long()).float()
    one_plus_scale = _round_bf16_to_fp32(1.0 + selected_scale)
    scaled = _round_bf16_to_fp32(x.float() * one_plus_scale)
    x.copy_((scaled + selected_shift).to(torch.bfloat16))
    return x


def indexed_gate_bf16_torch(
    x: torch.Tensor, gate: torch.Tensor, other: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    selected_gate = gate.index_select(0, indices.long()).float()
    gated = _round_bf16_to_fp32(selected_gate * other.float())
    x.copy_((x.float() + gated).to(torch.bfloat16))
    return x


# ---------------------------------------------------------------------------
# Triton reference implementation
# ---------------------------------------------------------------------------
@triton.jit
def _round_bf16_to_fp32_tl(value):
    bits = value.to(tl.int32, bitcast=True)
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    rounded_bits = (bits + rounding_bias) & -65536
    return rounded_bits.to(tl.float32, bitcast=True)


@triton.jit
def _indexed_scale_shift_bf16_kernel(
    x_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    hidden_size,
    stride_x_row,
    stride_shift_row,
    stride_scale_row,
    stride_indices,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_N)
    mask = columns < hidden_size
    index = tl.load(indices_ptr + row * stride_indices)

    x = tl.load(x_ptr + row * stride_x_row + columns, mask=mask, other=0.0).to(
        tl.float32
    )
    shift = tl.load(
        shift_ptr + index * stride_shift_row + columns, mask=mask, other=0.0
    ).to(tl.float32)
    scale = tl.load(
        scale_ptr + index * stride_scale_row + columns, mask=mask, other=0.0
    ).to(tl.float32)

    one_plus_scale = _round_bf16_to_fp32_tl(1.0 + scale)
    scaled = _round_bf16_to_fp32_tl(x * one_plus_scale)
    tl.store(x_ptr + row * stride_x_row + columns, scaled + shift, mask=mask)


@triton.jit
def _indexed_gate_bf16_kernel(
    x_ptr,
    gate_ptr,
    other_ptr,
    indices_ptr,
    hidden_size,
    stride_x_row,
    stride_gate_row,
    stride_other_row,
    stride_indices,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_N)
    mask = columns < hidden_size
    index = tl.load(indices_ptr + row * stride_indices)

    x = tl.load(x_ptr + row * stride_x_row + columns, mask=mask, other=0.0).to(
        tl.float32
    )
    gate = tl.load(
        gate_ptr + index * stride_gate_row + columns, mask=mask, other=0.0
    ).to(tl.float32)
    other = tl.load(
        other_ptr + row * stride_other_row + columns, mask=mask, other=0.0
    ).to(tl.float32)

    gated = _round_bf16_to_fp32_tl(gate * other)
    tl.store(x_ptr + row * stride_x_row + columns, x + gated, mask=mask)


def indexed_scale_shift_bf16_triton(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    rows, hidden_size = x.shape
    if rows == 0:
        return x
    block_n = triton.next_power_of_2(hidden_size)
    _indexed_scale_shift_bf16_kernel[(rows,)](
        x,
        shift,
        scale,
        indices,
        hidden_size,
        x.stride(0),
        shift.stride(0),
        scale.stride(0),
        indices.stride(0),
        BLOCK_N=block_n,
        num_warps=8,
    )
    return x


def indexed_gate_bf16_triton(
    x: torch.Tensor, gate: torch.Tensor, other: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    rows, hidden_size = x.shape
    if rows == 0:
        return x
    block_n = triton.next_power_of_2(hidden_size)
    _indexed_gate_bf16_kernel[(rows,)](
        x,
        gate,
        other,
        indices,
        hidden_size,
        x.stride(0),
        gate.stride(0),
        other.stride(0),
        indices.stride(0),
        BLOCK_N=block_n,
        num_warps=8,
    )
    return x


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------
def _make_indices(rows: int, num_indices: int, device) -> torch.Tensor:
    return torch.arange(rows, device=device) % num_indices


def calculate_diff_scale_shift(rows: int, hidden: int, num_indices: int) -> None:
    device = torch.device("xpu")
    x = torch.randn(rows, hidden, dtype=torch.bfloat16, device=device)
    shift = torch.randn(num_indices, hidden, dtype=torch.bfloat16, device=device)
    scale = torch.randn(num_indices, hidden, dtype=torch.bfloat16, device=device)
    indices = _make_indices(rows, num_indices, device)

    out_torch = indexed_scale_shift_bf16_torch(x.clone(), shift, scale, indices)
    out_triton = indexed_scale_shift_bf16_triton(x.clone(), shift, scale, indices)
    out_sycl = indexed_scale_shift_bf16_(x.clone(), shift, scale, indices)
    torch.xpu.synchronize()

    tag = f"rows={rows}, hidden={hidden}, num_indices={num_indices}"
    print(f"  [scale_shift] {tag}")
    for name, out in [("triton", out_triton), ("sycl", out_sycl)]:
        cos_sim = torch.nn.functional.cosine_similarity(
            out.reshape(-1).float(), out_torch.reshape(-1).float(), dim=0
        ).item()
        status = "✅" if cos_sim > 0.99 else "❌"
        print(f"    {name} vs torch: cos_sim={cos_sim:.6f}  {status}")


def calculate_diff_gate(rows: int, hidden: int, num_indices: int) -> None:
    device = torch.device("xpu")
    x = torch.randn(rows, hidden, dtype=torch.bfloat16, device=device)
    gate = torch.randn(num_indices, hidden, dtype=torch.bfloat16, device=device)
    other = torch.randn(rows, hidden, dtype=torch.bfloat16, device=device)
    indices = _make_indices(rows, num_indices, device)

    out_torch = indexed_gate_bf16_torch(x.clone(), gate, other, indices)
    out_triton = indexed_gate_bf16_triton(x.clone(), gate, other, indices)
    out_sycl = indexed_gate_bf16_(x.clone(), gate, other, indices)
    torch.xpu.synchronize()

    tag = f"rows={rows}, hidden={hidden}, num_indices={num_indices}"
    print(f"  [gate] {tag}")
    for name, out in [("triton", out_triton), ("sycl", out_sycl)]:
        cos_sim = torch.nn.functional.cosine_similarity(
            out.reshape(-1).float(), out_torch.reshape(-1).float(), dim=0
        ).item()
        status = "✅" if cos_sim > 0.99 else "❌"
        print(f"    {name} vs torch: cos_sim={cos_sim:.6f}  {status}")


# ---------------------------------------------------------------------------
# Bandwidth helper
# ---------------------------------------------------------------------------
def _bw_metrics(rows: int, hidden: int, num_tables: int, time_ms: float) -> dict:
    elem_bytes = 2  # bf16
    # read x + num_tables indexed rows, write x back (in-place)
    total_bytes = rows * hidden * (1 + num_tables) * elem_bytes
    time_s = time_ms / 1000.0
    bw_gbs = (total_bytes / 1e9) / time_s if time_s > 0 else 0.0
    return {"total_bytes_mb": total_bytes / 1e6, "bandwidth_gbs": bw_gbs}


# ---------------------------------------------------------------------------
# Benchmark configs
# ---------------------------------------------------------------------------
rows_range = [0, 1, 16, 64, 128, 256, 512, 1024, 2048, 4096]
hidden_range = [96, 128, 512, 1024, 2048, 4096, 7168]
num_indices_range = [2, 3, 4, 5, 8, 16]

all_results = []


def _make_benchmark(kernel_name: str):
    configs = list(itertools.product(rows_range, hidden_range, num_indices_range))

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["rows", "hidden", "num_indices"],
            x_vals=configs,
            line_arg="provider",
            line_vals=["torch", "triton", "sycl"],
            line_names=["PyTorch Reference", "Triton Kernel", "SYCL Kernel"],
            styles=[("blue", "-"), ("orange", "-"), ("green", "-")],
            ylabel="us",
            plot_name=f"{kernel_name}",
            args={},
        )
    )
    def bench(rows, hidden, num_indices, provider):
        device = torch.device("xpu")
        x = torch.randn(rows, hidden, dtype=torch.bfloat16, device=device)
        indices = _make_indices(rows, num_indices, device)
        quantiles = [0.5, 0.2, 0.8]

        if kernel_name == "indexed_scale_shift_bf16":
            shift = torch.randn(
                num_indices, hidden, dtype=torch.bfloat16, device=device
            )
            scale = torch.randn(
                num_indices, hidden, dtype=torch.bfloat16, device=device
            )
            num_tables = 2
            fns = {
                "torch": lambda: indexed_scale_shift_bf16_torch(
                    x.clone(), shift, scale, indices
                ),
                "triton": lambda: indexed_scale_shift_bf16_triton(
                    x.clone(), shift, scale, indices
                ),
                "sycl": lambda: indexed_scale_shift_bf16_(
                    x.clone(), shift, scale, indices
                ),
            }
        else:
            gate = torch.randn(num_indices, hidden, dtype=torch.bfloat16, device=device)
            other = torch.randn(rows, hidden, dtype=torch.bfloat16, device=device)
            num_tables = 1
            fns = {
                "torch": lambda: indexed_gate_bf16_torch(
                    x.clone(), gate, other, indices
                ),
                "triton": lambda: indexed_gate_bf16_triton(
                    x.clone(), gate, other, indices
                ),
                "sycl": lambda: indexed_gate_bf16_(x.clone(), gate, other, indices),
            }

        ms, min_ms, max_ms = triton.testing.do_bench(fns[provider], quantiles=quantiles)
        bw = _bw_metrics(rows, hidden, num_tables, ms)
        all_results.append(
            {
                "kernel": kernel_name,
                "rows": rows,
                "hidden": hidden,
                "num_indices": num_indices,
                "provider": provider,
                "time_us": round(1000 * ms, 3),
                "bandwidth_gbs": round(bw["bandwidth_gbs"], 2),
                "total_bytes_mb": round(bw["total_bytes_mb"], 2),
            }
        )
        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return bench


if __name__ == "__main__":
    # --- correctness spot checks ---
    print("=" * 70)
    print("Correctness check (SYCL / Triton vs PyTorch reference)")
    print("=" * 70)
    for rows, hidden, num_indices in [(16, 1024, 3), (128, 4096, 3), (1024, 7168, 3)]:
        calculate_diff_scale_shift(rows, hidden, num_indices)
        calculate_diff_gate(rows, hidden, num_indices)
    print()

    # --- performance benchmark ---
    for kernel_name in ["indexed_scale_shift_bf16", "indexed_gate_bf16"]:
        print("=" * 70)
        print(f"Benchmarking {kernel_name}")
        print("=" * 70)
        bench_fn = _make_benchmark(kernel_name)
        bench_fn.run(print_data=True)

    # --- summary table ---
    if all_results:
        print("\n" + "=" * 70)
        print("Full results (all shapes & kernels)")
        print("=" * 70)
        df = pd.DataFrame(all_results)
        print(df.to_markdown(index=False))

        print("\n" + "=" * 70)
        print("Summary by kernel & provider")
        print("=" * 70)
        summary = (
            df.groupby(["kernel", "provider"])
            .agg(
                time_us_mean=("time_us", "mean"),
                time_us_min=("time_us", "min"),
                time_us_max=("time_us", "max"),
                bw_gbs_mean=("bandwidth_gbs", "mean"),
            )
            .round(2)
        )
        print(summary.to_markdown())
