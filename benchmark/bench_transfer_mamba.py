"""Benchmark the Mamba HiCache transfer kernels: PyTorch vs AOT SYCL.

Both providers move exactly the same bytes -- the kernels are pure gather /
scatter copies with no arithmetic -- so the interesting number is achieved
bandwidth:

  torch : PyTorch advanced-indexing copy (what runs without a kernel)
  sycl  : sgl_kernel.transfer_kv_mamba_*  (AOT SYCL, compiled into the wheel)

  load   : dst[dst_idx[i]]        <- src[src_idx[i], layer_id]
  backup : dst[dst_idx[i], layer] <- src[layer, src_idx[i]]   (all layers)
"""

import itertools

import pandas as pd
import torch
import triton

try:
    from sgl_kernel import transfer_kv_mamba_lf_pf, transfer_kv_mamba_pf_lf

    HAS_SYCL = True
except ImportError:
    HAS_SYCL = False
    print("Warning: sgl_kernel Mamba transfer kernels not available")

all_results = []

# Layers copied per backup call (a real Mamba HiCache backup is all-layer).
NUM_LAYERS = 4
DTYPE = torch.bfloat16


def _make_indices(pool, num_items, device):
    src_idx = torch.randperm(pool, device=device)[:num_items].to(torch.int64)
    dst_idx = torch.randperm(pool, device=device)[:num_items].to(torch.int64)
    return src_idx, dst_idx


def torch_load_pf_lf(src, dst, src_idx, dst_idx, layer_id, item_elems, num_layers):
    """PyTorch reference gather: page_first -> single-layer."""
    dst.view(-1, item_elems)[dst_idx] = src.view(-1, num_layers, item_elems)[
        src_idx, layer_id
    ]


def torch_backup_lf_pf(src_layers, dst, src_idx, dst_idx, item_elems, num_layers):
    """PyTorch reference scatter: layer_first -> page_first (all layers)."""
    s = src_layers.view(num_layers, -1, item_elems)
    dst.view(-1, num_layers, item_elems)[dst_idx] = s[:, src_idx, :].permute(1, 0, 2)


def bytes_moved(op, num_items, item_elems, num_layers, elem_size):
    """Bytes read + written by one call (copy => 2x the payload)."""
    items = num_items if op == "load" else num_items * num_layers
    return 2 * items * item_elems * elem_size


# (num_items, item_elems, op). item_elems spans a small per-page Mamba state up
# to a large fused conv+ssm state; num_items spans a short prefix-cache hit up to
# a full-batch backup.
configs = list(
    itertools.product(
        [16, 128, 512],  # num_items (pages transferred)
        [1024, 8192, 65536],  # item_elems (state elements per page per layer)
        ["load", "backup"],
    )
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_items", "item_elems", "op"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["torch", "sycl"],
        line_names=["PyTorch (index copy)", "SYCL"],
        styles=[("blue", "-"), ("red", "-")],
        ylabel="us",
        plot_name="torch-vs-sycl-transfer-mamba-performance",
        args={},
    )
)
def benchmark(num_items, item_elems, op, provider):
    device = torch.device("xpu")
    num_layers = NUM_LAYERS
    pool = num_items  # every page in the pool participates
    elem_size = torch.tensor([], dtype=DTYPE).element_size()
    item_size = item_elems * elem_size
    layout_dim = item_size * num_layers
    layer_id = min(2, num_layers - 1)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "sycl" and not HAS_SYCL:
        print("Warning: sgl_kernel Mamba transfer not available, skipping")
        return 0, 0, 0

    src_idx, dst_idx = _make_indices(pool, num_items, device)

    if op == "load":
        src = torch.randn(pool, num_layers, item_elems, device=device, dtype=DTYPE)
        dst = torch.zeros(pool, item_elems, device=device, dtype=DTYPE)
        if provider == "torch":
            fn = lambda: torch_load_pf_lf(
                src, dst, src_idx, dst_idx, layer_id, item_elems, num_layers
            )
        else:
            fn = lambda: transfer_kv_mamba_pf_lf(
                src, dst, src_idx, dst_idx, layer_id, item_size, layout_dim
            )
    else:
        src = torch.randn(num_layers, pool, item_elems, device=device, dtype=DTYPE)
        dst = torch.zeros(pool, num_layers, item_elems, device=device, dtype=DTYPE)
        if provider == "torch":
            fn = lambda: torch_backup_lf_pf(
                src, dst, src_idx, dst_idx, item_elems, num_layers
            )
        else:
            fn = lambda: transfer_kv_mamba_lf_pf(
                src, dst, src_idx, dst_idx, item_size, layout_dim, num_layers
            )

    # Warm up (first launch pays kernel setup) before the measurement window.
    fn()
    torch.xpu.synchronize()

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    total_bytes = bytes_moved(op, num_items, item_elems, num_layers, elem_size)
    all_results.append(
        {
            "op": op,
            "num_items": num_items,
            "item_elems": item_elems,
            "num_layers": num_layers,
            "provider": provider,
            "time_us": 1000 * ms,
            "bandwidth_gbs": (total_bytes / 1e9) / (ms / 1e3),
        }
    )

    return 1000 * ms, 1000 * min_ms, 1000 * max_ms


if __name__ == "__main__":
    if not torch.xpu.is_available():
        print("ERROR: Intel XPU not available.")
        exit(1)
    if not HAS_SYCL:
        print(
            "ERROR: sgl_kernel Mamba transfer kernels not available. Install "
            "sgl-kernel-xpu on an XPU host."
        )
        exit(1)

    print("Running PyTorch vs SYCL Mamba HiCache transfer benchmarks...")
    print("torch: advanced-indexing gather/scatter copy")
    print("sycl : sgl_kernel.transfer_kv_mamba_{pf_lf,lf_pf}")
    print(f"dtype={DTYPE}, num_layers={NUM_LAYERS}")
    print("\n" + "=" * 80 + "\n")

    benchmark.run(print_data=True)

    print("Benchmark finished!")
    df = pd.DataFrame(all_results)
    df["time_us"] = df["time_us"].round(2)
    df["bandwidth_gbs"] = df["bandwidth_gbs"].round(1)
    print(df.to_markdown(index=False))
