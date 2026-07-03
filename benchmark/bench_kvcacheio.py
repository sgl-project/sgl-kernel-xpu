"""Benchmark for KV-cache scatter/gather transfer kernels (XPU).

Covers:
  - transfer_kv_per_layer      (lf→lf, K+V, device↔device)
  - transfer_kv_per_layer_mla  (lf→lf, K only, device↔device)
  - transfer_kv_all_layer      (lf_tbl→lf_tbl, K+V, all layers fused)
  - transfer_kv_all_layer_lf_ph / transfer_kv_per_layer_ph_lf
  - transfer_kv_direct         (Python copy_ fallback, host↔device)
  - transfer_kv_all_layer_direct_lf_pf / transfer_kv_per_layer_direct_pf_lf

Usage:
    python benchmark/bench_kvcacheio.py
    python benchmark/bench_kvcacheio.py --num-tokens 128 512 1024 4096 8192
    python benchmark/bench_kvcacheio.py --suite per_layer all_layer host_device page_head
"""

import argparse

import sgl_kernel  # noqa: F401 – registers XPU ops
import torch
from sgl_kernel.kvcacheio import (
    transfer_kv_all_layer,
    transfer_kv_all_layer_direct_lf_pf,
    transfer_kv_all_layer_lf_ph,
    transfer_kv_all_layer_mla,
    transfer_kv_direct,
    transfer_kv_per_layer,
    transfer_kv_per_layer_direct_pf_lf,
    transfer_kv_per_layer_mla,
    transfer_kv_per_layer_ph_lf,
)

DEVICE = "xpu"
DTYPE = torch.bfloat16
ELEM = DTYPE.itemsize  # 2 bytes


def _sync():
    torch.xpu.synchronize()


def _bench(fn, warmup=10, rep=100):
    """Return median latency in ms using XPU events (no host-sync overhead)."""
    for _ in range(warmup):
        fn()
    _sync()
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    _sync()
    return start.elapsed_time(end) / rep


def _header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(f"  {'N':>6}  {'item_size':>10}  {'BW GB/s':>9}  {'us':>8}  {'% peak':>8}")
    print(f"  {'-'*55}")


def _row(N, item_size, nbytes, ms):
    bw = nbytes / (ms * 1e-3) / 1e9
    pct = bw / 192 * 100
    print(f"  {N:>6}  {item_size:>10}  {bw:>9.1f}  {ms*1000:>8.1f}  {pct:>7.0f}%")
    return bw


# ---------------------------------------------------------------------------
# Suite 1: per_layer  (device→device)
# ---------------------------------------------------------------------------


def bench_per_layer(token_counts, item_sizes):
    _header("transfer_kv_per_layer  [lf→lf, K+V, device→device]")
    for N in token_counts:
        for item_size in item_sizes:
            total = max(N * 4, 16384)
            src_k = torch.randn(total, item_size, dtype=DTYPE, device=DEVICE)
            src_v = torch.randn(total, item_size, dtype=DTYPE, device=DEVICE)
            dst_k = torch.zeros_like(src_k)
            dst_v = torch.zeros_like(src_v)
            perm = torch.randperm(total, device=DEVICE)
            si, di = perm[:N], perm[N : 2 * N]
            ib = item_size * ELEM
            nbytes = N * ib * 4  # read K+V, write K+V

            ms = _bench(
                lambda: transfer_kv_per_layer(src_k, dst_k, src_v, dst_v, si, di, ib)
            )
            _row(N, item_size, nbytes, ms)


def bench_per_layer_mla(token_counts, item_sizes):
    _header("transfer_kv_per_layer_mla  [lf→lf, K only / MLA, device→device]")
    for N in token_counts:
        for item_size in item_sizes:
            total = max(N * 4, 16384)
            src = torch.randn(total, item_size, dtype=DTYPE, device=DEVICE)
            dst = torch.zeros_like(src)
            perm = torch.randperm(total, device=DEVICE)
            si, di = perm[:N], perm[N : 2 * N]
            ib = item_size * ELEM
            nbytes = N * ib * 2  # read K, write K

            ms = _bench(lambda: transfer_kv_per_layer_mla(src, dst, si, di, ib))
            _row(N, item_size, nbytes, ms)


# ---------------------------------------------------------------------------
# Suite 2: all_layer  (fused across layers, device→device)
# ---------------------------------------------------------------------------


def bench_all_layer(token_counts, item_sizes, num_layers=32):
    _header(
        f"transfer_kv_all_layer  [lf_tbl→lf_tbl, K+V, {num_layers} layers fused, device→device]"
    )
    for N in token_counts:
        for item_size in item_sizes:
            total = max(N * 4, 16384)
            # Skip configs that would exhaust B580 VRAM (6 GB).
            # 4 (src_k+src_v+dst_k+dst_v) × num_layers × total × item_size × 2B
            if 4 * num_layers * total * item_size * ELEM > 4 * 1024**3:
                print(f"  {N:>6}  {item_size:>10}  (skipped — would exceed VRAM)")
                continue
            src_k = [
                torch.randn(total, item_size, dtype=DTYPE, device=DEVICE)
                for _ in range(num_layers)
            ]
            src_v = [
                torch.randn(total, item_size, dtype=DTYPE, device=DEVICE)
                for _ in range(num_layers)
            ]
            dst_k = [torch.zeros_like(x) for x in src_k]
            dst_v = [torch.zeros_like(x) for x in src_v]
            perm = torch.randperm(total, device=DEVICE)
            si, di = perm[:N], perm[N : 2 * N]

            def _ptrs(lst):
                return torch.tensor(
                    [x.data_ptr() for x in lst], dtype=torch.uint64, device=DEVICE
                )

            sk_t, sv_t = _ptrs(src_k), _ptrs(src_v)
            dk_t, dv_t = _ptrs(dst_k), _ptrs(dst_v)
            ib = item_size * ELEM
            nbytes = N * ib * 4 * num_layers

            ms = _bench(
                lambda: transfer_kv_all_layer(
                    sk_t, dk_t, sv_t, dv_t, si, di, ib, num_layers
                )
            )
            _row(N, item_size, nbytes, ms)


def bench_all_layer_mla(token_counts, item_sizes, num_layers=32):
    _header(
        f"transfer_kv_all_layer_mla  [lf_tbl→lf_tbl, K only, {num_layers} layers, device→device]"
    )
    for N in token_counts:
        for item_size in item_sizes:
            total = max(N * 4, 16384)
            # 2 (src+dst) × num_layers × total × item_size × 2B
            if 2 * num_layers * total * item_size * ELEM > 4 * 1024**3:
                print(f"  {N:>6}  {item_size:>10}  (skipped — would exceed VRAM)")
                continue
            src = [
                torch.randn(total, item_size, dtype=DTYPE, device=DEVICE)
                for _ in range(num_layers)
            ]
            dst = [torch.zeros_like(x) for x in src]
            perm = torch.randperm(total, device=DEVICE)
            si, di = perm[:N], perm[N : 2 * N]

            def _ptrs(lst):
                return torch.tensor(
                    [x.data_ptr() for x in lst], dtype=torch.uint64, device=DEVICE
                )

            s_t, d_t = _ptrs(src), _ptrs(dst)
            ib = item_size * ELEM
            nbytes = N * ib * 2 * num_layers

            ms = _bench(
                lambda: transfer_kv_all_layer_mla(s_t, d_t, si, di, ib, num_layers)
            )
            _row(N, item_size, nbytes, ms)


# ---------------------------------------------------------------------------
# Suite 3: host↔device  (pinned memory, PCIe path)
# ---------------------------------------------------------------------------


def bench_host_device(token_counts, item_sizes):
    _header("transfer_kv_direct  [pinned host→device, copy_ fallback]")
    for N in token_counts:
        for item_size in item_sizes:
            total = max(N * 4, 16384)
            src_k = torch.randn(total, item_size, dtype=DTYPE).pin_memory()
            src_v = torch.randn(total, item_size, dtype=DTYPE).pin_memory()
            dst_k = torch.zeros(total, item_size, dtype=DTYPE, device=DEVICE)
            dst_v = torch.zeros_like(dst_k)
            perm = torch.randperm(total)
            si = perm[:N]
            di = perm[N : 2 * N].to(DEVICE)
            page_size = 16
            nbytes = N * item_size * ELEM * 2  # read from host only counts once

            ms = _bench(
                lambda: transfer_kv_direct(
                    [src_k, src_v], [dst_k, dst_v], si, di, page_size
                ),
                warmup=5,
                rep=20,
            )
            _row(N, item_size, nbytes, ms)

    _header("transfer_kv_all_layer_direct_lf_pf  [device lf→pinned host pf]")
    for N in token_counts:
        for item_size in item_sizes:
            page_size = 16
            if N % page_size != 0:
                continue
            total = max(N * 4, 16384)
            total_pages = total // page_size
            num_layers = 4
            src_k = [
                torch.randn(total, item_size, dtype=DTYPE, device=DEVICE)
                for _ in range(num_layers)
            ]
            src_v = [
                torch.randn(total, item_size, dtype=DTYPE, device=DEVICE)
                for _ in range(num_layers)
            ]
            dst_k = torch.zeros(
                total_pages, num_layers, page_size, item_size, dtype=DTYPE
            ).pin_memory()
            dst_v = torch.zeros_like(dst_k).pin_memory()
            perm = torch.randperm(total_pages)
            num_pages = N // page_size
            si = torch.cat(
                [
                    torch.arange(p * page_size, (p + 1) * page_size)
                    for p in perm[:num_pages]
                ]
            )
            di = torch.cat(
                [
                    torch.arange(p * page_size, (p + 1) * page_size)
                    for p in perm[num_pages : 2 * num_pages]
                ]
            )
            nbytes = N * item_size * ELEM * 2 * num_layers

            ms = _bench(
                lambda: transfer_kv_all_layer_direct_lf_pf(
                    src_k + src_v, [dst_k, dst_v], si, di, page_size
                ),
                warmup=5,
                rep=20,
            )
            _row(N, item_size, nbytes, ms)

    _header("transfer_kv_per_layer_direct_pf_lf  [pinned host pf→device lf, 1 layer]")
    for N in token_counts:
        for item_size in item_sizes:
            page_size = 16
            if N % page_size != 0:
                continue
            total = max(N * 4, 16384)
            total_pages = total // page_size
            num_layers = 4
            layer_id = 0
            src_k = torch.randn(
                total_pages, num_layers, page_size, item_size, dtype=DTYPE
            ).pin_memory()
            src_v = torch.randn(
                total_pages, num_layers, page_size, item_size, dtype=DTYPE
            ).pin_memory()
            dst_k = torch.zeros(total, item_size, dtype=DTYPE, device=DEVICE)
            dst_v = torch.zeros_like(dst_k)
            perm = torch.randperm(total_pages)
            num_pages = N // page_size
            si = torch.cat(
                [
                    torch.arange(p * page_size, (p + 1) * page_size)
                    for p in perm[:num_pages]
                ]
            )
            di = torch.cat(
                [
                    torch.arange(p * page_size, (p + 1) * page_size)
                    for p in perm[num_pages : 2 * num_pages]
                ]
            )
            # Single layer transferred (K+V): read from host counts once.
            nbytes = N * item_size * ELEM * 2

            ms = _bench(
                lambda: transfer_kv_per_layer_direct_pf_lf(
                    [src_k, src_v], [dst_k, dst_v], si, di, layer_id, page_size
                ),
                warmup=5,
                rep=20,
            )
            _row(N, item_size, nbytes, ms)


# ---------------------------------------------------------------------------
# Suite 4: page-head layout
# ---------------------------------------------------------------------------


def bench_page_head(token_counts, item_sizes, head_num=16, num_layers=4):
    page_size = 16

    _header(
        f"transfer_kv_all_layer_lf_ph  [lf_tbl→ph, {num_layers} layers, head_num={head_num}]"
    )
    for N in token_counts:
        for item_size in item_sizes:
            if item_size % head_num != 0:
                continue
            total = max(N * 4, 16384)
            total_pages = total // page_size
            src_k = [
                torch.randn(total, item_size, dtype=DTYPE, device=DEVICE)
                for _ in range(num_layers)
            ]
            src_v = [
                torch.randn(total, item_size, dtype=DTYPE, device=DEVICE)
                for _ in range(num_layers)
            ]
            dst_k = torch.zeros(
                total_pages,
                head_num,
                page_size,
                num_layers,
                item_size // head_num,
                dtype=DTYPE,
            ).pin_memory()
            dst_v = torch.zeros_like(dst_k).pin_memory()

            sk_t = torch.tensor(
                [x.data_ptr() for x in src_k], dtype=torch.uint64, device=DEVICE
            )
            sv_t = torch.tensor(
                [x.data_ptr() for x in src_v], dtype=torch.uint64, device=DEVICE
            )
            perm = torch.randperm(total, device=DEVICE)
            si, di = perm[:N], perm[N : 2 * N]
            ib = item_size * ELEM
            dst_layout_dim = item_size * num_layers * ELEM
            nbytes = N * ib * 4 * num_layers

            ms = _bench(
                lambda: transfer_kv_all_layer_lf_ph(
                    sk_t,
                    dst_k,
                    sv_t,
                    dst_v,
                    si,
                    di,
                    ib,
                    dst_layout_dim,
                    num_layers,
                    page_size,
                    head_num,
                ),
                warmup=5,
                rep=20,
            )
            _row(N, item_size, nbytes, ms)

    _header(f"transfer_kv_per_layer_ph_lf  [ph→lf, layer 0, head_num={head_num}]")
    for N in token_counts:
        for item_size in item_sizes:
            if item_size % head_num != 0:
                continue
            total = max(N * 4, 16384)
            total_pages = total // page_size
            src_k = torch.randn(
                total_pages,
                head_num,
                page_size,
                num_layers,
                item_size // head_num,
                dtype=DTYPE,
            ).pin_memory()
            src_v = torch.zeros_like(src_k).pin_memory()
            dst_k = torch.zeros(total, item_size, dtype=DTYPE, device=DEVICE)
            dst_v = torch.zeros_like(dst_k)
            perm = torch.randperm(total, device=DEVICE)
            si, di = perm[:N], perm[N : 2 * N]
            ib = item_size * ELEM
            src_layout_dim = item_size * num_layers * ELEM
            nbytes = N * ib * 4

            ms = _bench(
                lambda: transfer_kv_per_layer_ph_lf(
                    src_k,
                    dst_k,
                    src_v,
                    dst_v,
                    si,
                    di,
                    0,
                    ib,
                    src_layout_dim,
                    page_size,
                    head_num,
                ),
                warmup=5,
                rep=20,
            )
            _row(N, item_size, nbytes, ms)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SUITES = {
    "per_layer": [bench_per_layer, bench_per_layer_mla],
    "all_layer": [bench_all_layer, bench_all_layer_mla],
    "host_device": [bench_host_device],
    "page_head": [bench_page_head],
}


def parse_args():
    p = argparse.ArgumentParser(description="KV cache IO benchmark for XPU")
    p.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=[128, 512, 1024, 4096, 8192],
        metavar="N",
        help="Token counts to sweep (default: 128 512 1024 4096 8192)",
    )
    p.add_argument(
        "--item-sizes",
        type=int,
        nargs="+",
        default=[256, 512, 1024],
        metavar="S",
        help="item_size values (number of elements, default: 256 512 1024)",
    )
    p.add_argument(
        "--suite",
        nargs="+",
        choices=list(SUITES.keys()),
        default=list(SUITES.keys()),
        help="Which benchmark suites to run (default: all)",
    )
    p.add_argument(
        "--num-layers",
        type=int,
        default=32,
        help="Number of layers for all_layer benchmarks (default: 32)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Device : {torch.xpu.get_device_name(0)}")
    print(f"dtype  : {DTYPE}")
    print(f"tokens : {args.num_tokens}")
    print(f"items  : {args.item_sizes}")
    print(f"suites : {args.suite}")

    fns_to_run = []
    for s in args.suite:
        fns_to_run.extend(SUITES[s])

    for fn in fns_to_run:
        import inspect

        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        kwargs = {"token_counts": args.num_tokens, "item_sizes": args.item_sizes}
        if "num_layers" in params:
            kwargs["num_layers"] = args.num_layers
        fn(**kwargs)

    print()


if __name__ == "__main__":
    main()
