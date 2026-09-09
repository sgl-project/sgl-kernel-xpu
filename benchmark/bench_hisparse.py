import argparse
import os
from itertools import product

import pandas as pd
import torch
import triton
from sgl_kernel import (
    load_cache_to_device_buffer_dsv4_mla,
    load_cache_to_device_buffer_mla,
    transfer_cache_dsv4_mla,
)

DEVICE = "xpu"
SEED = 42

# Linear-layout MLA item: 512 kv-lora + 64 rope, bf16 (matches the DSA MLA cache).
DTYPE = torch.bfloat16
ELEM = DTYPE.itemsize
KV_DIM = 576
LINEAR_ITEM_BYTES = KV_DIM * ELEM

# Page-padded C4 layout constants (must match c4_layout.hpp).
DSV4_PAGE_SIZE = 64
DSV4_VALUE_BYTES = 576
DSV4_SCALE_BYTES = 8
DSV4_ITEM_BYTES = DSV4_VALUE_BYTES + DSV4_SCALE_BYTES
DSV4_PAGE_BYTES = ((DSV4_ITEM_BYTES * DSV4_PAGE_SIZE + 575) // 576) * 576
DSV4_SCALE_OFFSET = DSV4_VALUE_BYTES * DSV4_PAGE_SIZE

VRAM_BUDGET_FRACTION = 0.5

QUANTILES = [0.5, 0.25, 0.75]  # median, fastest, slowest
RESULT_DIR = "bench_bmg_hisparse_res"

all_results = []


def _div_up(a, b):
    return (a + b - 1) // b


def _release(*objs):
    """Free device buffers; holding several configs live at once halves bandwidth."""
    for o in objs:
        if isinstance(o, (list, dict)):
            o.clear()
    torch.xpu.synchronize()
    torch.xpu.empty_cache()


def _gbps(nbytes, ms):
    if not nbytes or ms <= 0:
        return float("nan")
    return nbytes * 1e-9 / (ms * 1e-3)


def _dsv4_views(cache):
    """Expose a page-padded C4 cache as (value, scale) views, no copy.

    Slicing + ``view`` would fail (the slice is not contiguous, row stride is
    DSV4_PAGE_BYTES), so build genuine views with the layout's own strides.
    """
    pages = cache.shape[0]
    value = cache.as_strided(
        (pages, DSV4_PAGE_SIZE, DSV4_VALUE_BYTES),
        (DSV4_PAGE_BYTES, DSV4_VALUE_BYTES, 1),
    )
    scale = cache.as_strided(
        (pages, DSV4_PAGE_SIZE, DSV4_SCALE_BYTES),
        (DSV4_PAGE_BYTES, DSV4_SCALE_BYTES, 1),
        storage_offset=DSV4_SCALE_OFFSET,
    )
    return value, scale


def _page_split(index):
    return index // DSV4_PAGE_SIZE, index % DSV4_PAGE_SIZE


def _bench(fn, warmup=10):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    return triton.testing.do_bench(fn, quantiles=QUANTILES)


def _bench_cold(make_state, call, reps=20):
    warm = make_state()
    call(warm)  # warm the kernel and the allocator
    warm.clear()
    torch.xpu.synchronize()

    samples = []
    for _ in range(reps):
        state = make_state()
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        call(state)
        end.record()
        torch.xpu.synchronize()
        samples.append(start.elapsed_time(end))
        state.clear()

    samples.sort()
    last = len(samples) - 1
    return tuple(samples[min(int(q * len(samples)), last)] for q in QUANTILES)


# ---------------------------------------------------------------------------
# swap-in state
# ---------------------------------------------------------------------------

# Every miss rep reuses the same host cache; only the *device* state has to be
# rebuilt. torch's caching host allocator never returns pinned blocks to the OS,
# so allocating one per rep starved the transfer suite of its 2.3 GB.
_HOST_CACHES = {}


def _host_cache(batch_size, num_top_k, hot_buffer_size, is_dsv4):
    key = (batch_size, num_top_k, hot_buffer_size, is_dsv4)
    cached = _HOST_CACHES.get(key)
    if cached is not None:
        return cached

    num_host_items = batch_size * (num_top_k + hot_buffer_size) + 1
    if is_dsv4:
        shape = (_div_up(num_host_items, DSV4_PAGE_SIZE), DSV4_PAGE_BYTES)
        cache = torch.empty(shape, dtype=torch.uint8, device="cpu").pin_memory()
        cache.fill_(7)
    else:
        shape = (num_host_items, 1, KV_DIM)
        cache = torch.empty(shape, dtype=DTYPE, device="cpu").pin_memory()
        cache.fill_(1.0)
    _HOST_CACHES[key] = cache
    return cache


def release_host_caches():
    _release(_HOST_CACHES)


def _make_swapin_state(batch_size, num_top_k, hot_buffer_size, is_dsv4, regime, block):
    slots_per_req = hot_buffer_size + 1  # +1 reserved newest slot
    num_device_items = batch_size * slots_per_req
    num_host_items = batch_size * (num_top_k + hot_buffer_size) + 1
    host_cache = _host_cache(batch_size, num_top_k, hot_buffer_size, is_dsv4)

    if is_dsv4:
        device_buffer = torch.zeros(
            (_div_up(num_device_items, DSV4_PAGE_SIZE), DSV4_PAGE_BYTES),
            dtype=torch.uint8,
            device=DEVICE,
        )
        item_size_bytes = DSV4_ITEM_BYTES
    else:
        device_buffer = torch.zeros(
            (num_device_items, 1, KV_DIM), dtype=DTYPE, device=DEVICE
        )
        item_size_bytes = LINEAR_ITEM_BYTES

    # Each request owns a contiguous, disjoint span of device slots.
    device_buffer_locs = (
        torch.arange(num_device_items, dtype=torch.int32, device=DEVICE)
        .view(batch_size, slots_per_req)
        .contiguous()
    )
    host_cache_locs = (
        torch.arange(num_host_items, dtype=torch.int64, device=DEVICE)
        .view(1, -1)
        .repeat(batch_size, 1)
        .contiguous()
    )

    # Request r asks for tokens [base_r, base_r + num_top_k).
    base = torch.arange(batch_size, dtype=torch.int32, device=DEVICE).view(-1, 1) * (
        num_top_k + hot_buffer_size
    )
    top_k_tokens = (
        base + torch.arange(num_top_k, dtype=torch.int32, device=DEVICE).view(1, -1)
    ).contiguous()

    device_buffer_tokens = torch.full(
        (batch_size, slots_per_req), -1, dtype=torch.int32, device=DEVICE
    )
    if regime == "hit":
        device_buffer_tokens[:, :num_top_k] = top_k_tokens
    else:
        device_buffer_tokens[:, :hot_buffer_size] = (
            base
            + num_top_k
            + torch.arange(hot_buffer_size, dtype=torch.int32, device=DEVICE).view(
                1, -1
            )
        )
    device_buffer_tokens[:, hot_buffer_size] = top_k_tokens[:, -1]

    lru_slots = (
        torch.arange(hot_buffer_size, dtype=torch.int16, device=DEVICE)
        .view(1, -1)
        .repeat(batch_size, 1)
        .contiguous()
    )
    torch.xpu.synchronize()

    return {
        "top_k_tokens": top_k_tokens,
        "device_buffer_tokens": device_buffer_tokens,
        "host_cache_locs": host_cache_locs,
        "device_buffer_locs": device_buffer_locs,
        "host_cache": host_cache,
        "device_buffer": device_buffer,
        "top_k_device_locs": torch.full_like(top_k_tokens, -1),
        "req_pool_indices": torch.arange(batch_size, dtype=torch.int64, device=DEVICE),
        # seq_len must exceed hot_buffer_size or the kernel takes the fast path.
        "seq_lens": torch.full(
            (batch_size,), num_top_k + hot_buffer_size, dtype=torch.int32, device=DEVICE
        ),
        "lru_slots": lru_slots,
        "item_size_bytes": item_size_bytes,
        "num_top_k": num_top_k,
        "hot_buffer_size": hot_buffer_size,
        "page_size": DSV4_PAGE_SIZE if is_dsv4 else 1,
        "block_size": block,
        "num_real_reqs": torch.tensor([batch_size], dtype=torch.int32, device=DEVICE),
    }


def _swapin(state, is_dsv4):
    fn = (
        load_cache_to_device_buffer_dsv4_mla
        if is_dsv4
        else load_cache_to_device_buffer_mla
    )
    fn(**state)


def _torch_swapin(state, is_dsv4):
    """Pure-PyTorch equivalent of ``load_cache_to_device_buffer_*_mla``.

    A *timing* reference only -- the kernel's exact miss-to-slot assignment is
    internal, so slot ids may differ; accuracy is covered by tests/. The dominant
    cost is structural: indexing a pinned CPU tensor needs the gather indices on
    the host, forcing a device->host sync per call. The kernel stays on device.
    """
    top_k = state["top_k_tokens"]  # [B, K] int32
    dbt = state["device_buffer_tokens"]  # [B, S] int32
    dbl = state["device_buffer_locs"]  # [B, S] int32
    hcl = state["host_cache_locs"]  # [B, H] int64
    host_cache = state["host_cache"]
    device_buffer = state["device_buffer"]
    lru = state["lru_slots"]  # [B, hot] int16

    # Classify each requested token as hit or miss.
    eq = top_k.unsqueeze(2) == dbt.unsqueeze(1)  # [B, K, S]
    is_hit = eq.any(dim=2)  # [B, K]
    hit_slot = eq.to(torch.uint8).argmax(dim=2)  # [B, K]
    slot_is_hit = eq.any(dim=1)  # [B, S]

    # Pick evict slots in LRU order; the stable sort brings evictable slots
    # first while preserving that order.
    lru_long = lru.to(torch.int64)
    evictable = ~slot_is_hit.gather(1, lru_long)  # [B, hot] in LRU order
    order = torch.argsort(~evictable, dim=1, stable=True)
    evict_slots = lru_long.gather(1, order)  # [B, hot]

    # Rank each miss, and match it to the evict slot of the same rank.
    miss = ~is_hit
    miss_rank = miss.cumsum(dim=1) - 1  # [B, K]
    evict_pick = evict_slots.gather(1, miss_rank.clamp_(min=0))
    assigned = torch.where(miss, evict_pick, hit_slot)  # [B, K]

    # Stream the misses in.
    b_idx, k_idx = miss.nonzero(as_tuple=True)
    if b_idx.numel():
        token = top_k[b_idx, k_idx].to(torch.int64)
        host_loc = hcl[b_idx, token]
        dev_loc = dbl[b_idx, assigned[b_idx, k_idx]].to(torch.int64)
        # Pinned host cache must be indexed with CPU indices -> D2H sync.
        host_loc_cpu = host_loc.cpu()
        if is_dsv4:
            h_val, h_scale = _dsv4_views(host_cache)
            d_val, d_scale = _dsv4_views(device_buffer)
            hp, ho = _page_split(host_loc_cpu)
            dp, do = _page_split(dev_loc)
            d_val[dp, do] = h_val[hp, ho].to(DEVICE, non_blocking=True)
            d_scale[dp, do] = h_scale[hp, ho].to(DEVICE, non_blocking=True)
        else:
            staged = host_cache[host_loc_cpu].to(DEVICE, non_blocking=True)
            device_buffer[dev_loc] = staged

        dbt[b_idx, assigned[b_idx, k_idx]] = top_k[b_idx, k_idx]

    state["top_k_device_locs"].copy_(dbl.gather(1, assigned))
    # Touched slots become most-recently-used; untouched keep LRU order first.
    touched = torch.zeros_like(slot_is_hit)
    touched.scatter_(1, assigned, True)
    key = touched.gather(1, lru_long).to(torch.uint8)
    lru.copy_(lru_long.gather(1, torch.argsort(key, dim=1, stable=True)).to(lru.dtype))


# ---------------------------------------------------------------------------
# transfer state
# ---------------------------------------------------------------------------

# One live state at a time. triton iterates providers innermost, so a shape's
# block sizes arrive back to back and reuse cuts allocation work 3x.
_TRANSFER_STATE = {"key": None, "value": None}


def _transfer_state(num_items, num_layers):
    key = (num_items, num_layers)
    if _TRANSFER_STATE["key"] == key:
        return _TRANSFER_STATE["value"]

    release_transfer_state()
    pages = _div_up(num_items, DSV4_PAGE_SIZE)
    srcs = [
        torch.full((pages, DSV4_PAGE_BYTES), 3, dtype=torch.uint8, device=DEVICE)
        for _ in range(num_layers)
    ]
    dsts = [
        torch.zeros((pages, DSV4_PAGE_BYTES), dtype=torch.uint8, device=DEVICE)
        for _ in range(num_layers)
    ]
    value = (
        srcs,
        dsts,
        # Raw addresses only -- srcs/dsts must stay alive alongside them.
        torch.tensor([t.data_ptr() for t in srcs], dtype=torch.uint64, device=DEVICE),
        torch.tensor([t.data_ptr() for t in dsts], dtype=torch.uint64, device=DEVICE),
        torch.arange(num_items, dtype=torch.int64, device=DEVICE),
    )
    torch.xpu.synchronize()
    _TRANSFER_STATE.update(key=key, value=value)
    return value


def _torch_transfer(srcs, dsts, src_indices, dst_indices):
    """Pure-PyTorch equivalent of ``transfer_cache_dsv4_mla``.

    The kernel walks all layers in one launch; eager needs an indexed copy per
    layer, which is the cost this comparison isolates.
    """
    sp, so = _page_split(src_indices)
    dp, do = _page_split(dst_indices)
    for src, dst in zip(srcs, dsts):
        s_val, s_scale = _dsv4_views(src)
        d_val, d_scale = _dsv4_views(dst)
        d_val[dp, do] = s_val[sp, so]
        d_scale[dp, do] = s_scale[sp, so]


def release_transfer_state():
    value = _TRANSFER_STATE["value"]
    if value is not None:
        _release(value[0], value[1])
    _TRANSFER_STATE.update(key=None, value=None)


def _vram_budget():
    return int(torch.xpu.get_device_properties(0).total_memory * VRAM_BUDGET_FRACTION)


def _transfer_footprint(num_items, num_layers):
    return 2 * num_layers * _div_up(num_items, DSV4_PAGE_SIZE) * DSV4_PAGE_BYTES


# ---------------------------------------------------------------------------
# swap-in benchmark
# ---------------------------------------------------------------------------


def swapin_providers(with_torch):
    impls = ("sglang", "torch") if with_torch else ("sglang",)
    return [
        f"{impl}-{layout}-{regime}"
        for layout in ("linear", "dsv4")
        for regime in ("hit", "miss")
        for impl in impls
    ]


_STYLES = [
    ("blue", "-"),
    ("blue", "--"),
    ("green", "-"),
    ("green", "--"),
    ("red", "-"),
    ("red", "--"),
    ("orange", "-"),
    ("orange", "--"),
]


def benchmark_swapin(
    batch_size, num_top_k, hot_buffer_size, provider, block_size, reps
):
    impl, layout, regime = provider.split("-")
    print(
        f"benchmark load_cache_to_device_buffer {provider} batch_size={batch_size} "
        f"num_top_k={num_top_k} hot_buffer_size={hot_buffer_size} "
        f"block_size={block_size}"
    )
    torch.xpu.manual_seed_all(SEED)

    is_dsv4 = layout == "dsv4"
    item_bytes = DSV4_ITEM_BYTES if is_dsv4 else LINEAR_ITEM_BYTES
    run = _torch_swapin if impl == "torch" else _swapin

    if regime == "miss":
        # A miss reads one item from the host and writes one to the device buffer.
        nbytes = batch_size * num_top_k * item_bytes * 2
        ms, fast_ms, slow_ms = _bench_cold(
            lambda: _make_swapin_state(
                batch_size, num_top_k, hot_buffer_size, is_dsv4, "miss", block_size
            ),
            lambda st: run(st, is_dsv4),
            reps=reps,
        )
    else:
        # Hits move no bytes: index resolution plus LRU bookkeeping only.
        nbytes = 0
        state = _make_swapin_state(
            batch_size, num_top_k, hot_buffer_size, is_dsv4, "hit", block_size
        )
        ms, fast_ms, slow_ms = _bench(lambda: run(state, is_dsv4))
        state.clear()
    _release()

    all_results.append(
        {
            "op": f"load_cache_to_device_buffer [{regime}]",
            "impl": impl,
            "layout": layout,
            "batch_size": batch_size,
            "num_top_k": num_top_k,
            "hot_buffer_size": hot_buffer_size,
            "block_size": block_size,
            "us (median)": round(ms * 1000, 2),
            "GB/s (median)": round(_gbps(nbytes, ms), 2),
            "GB/s (min)": round(_gbps(nbytes, slow_ms), 2),
            "GB/s (max)": round(_gbps(nbytes, fast_ms), 2),
        }
    )
    return ms * 1000, fast_ms * 1000, slow_ms * 1000


def swapin_mark(configs, with_torch=False):
    providers = swapin_providers(with_torch)
    return triton.testing.Mark(
        benchmark_swapin,
        triton.testing.Benchmark(
            x_names=["batch_size", "num_top_k", "hot_buffer_size"],
            x_vals=configs,
            line_arg="provider",
            line_vals=providers,
            line_names=[p.replace("-", " ") for p in providers],
            styles=_STYLES[: len(providers)],
            ylabel="us",
            plot_name="hisparse-load-cache-to-device-buffer",
            args={},
        ),
    )


# ---------------------------------------------------------------------------
# transfer benchmark
# ---------------------------------------------------------------------------


def benchmark_transfer(num_items, num_layers, provider):
    is_torch = provider == "torch"
    block = 0 if is_torch else int(provider)
    print(
        f"benchmark transfer_cache_dsv4_mla {provider} num_items={num_items} "
        f"num_layers={num_layers}"
    )
    torch.xpu.manual_seed_all(SEED)

    footprint = _transfer_footprint(num_items, num_layers)
    budget = _vram_budget()
    if footprint > budget:
        # Say what was skipped: a silent drop reads as "covered" in the table.
        print(
            f"  SKIPPED: needs {footprint / 1024**3:.1f} GiB of VRAM, "
            f"budget is {budget / 1024**3:.1f} GiB"
        )
        nan = float("nan")
        return nan, nan, nan

    srcs, dsts, src_ptrs, dst_ptrs, idx = _transfer_state(num_items, num_layers)
    nbytes = num_items * num_layers * DSV4_ITEM_BYTES * 2  # read + write

    if is_torch:
        fn = lambda: _torch_transfer(srcs, dsts, idx, idx)  # noqa: E731
    else:
        fn = lambda: transfer_cache_dsv4_mla(  # noqa: E731
            src_ptrs=src_ptrs,
            dst_ptrs=dst_ptrs,
            src_indices=idx,
            dst_indices=idx,
            block_size=block,
        )
    ms, fast_ms, slow_ms = _bench(fn)

    all_results.append(
        {
            "op": "transfer_cache_dsv4_mla",
            "impl": "torch" if is_torch else "sglang",
            "num_items": num_items,
            "num_layers": num_layers,
            "block_size": block,
            "us (median)": round(ms * 1000, 2),
            "GB/s (median)": round(_gbps(nbytes, ms), 2),
            "GB/s (min)": round(_gbps(nbytes, slow_ms), 2),
            "GB/s (max)": round(_gbps(nbytes, fast_ms), 2),
        }
    )
    return _gbps(nbytes, ms), _gbps(nbytes, slow_ms), _gbps(nbytes, fast_ms)


def transfer_mark(configs, block_sizes, with_torch=False):
    providers = [str(b) for b in block_sizes] + (["torch"] if with_torch else [])
    return triton.testing.Mark(
        benchmark_transfer,
        triton.testing.Benchmark(
            x_names=["num_items", "num_layers"],
            x_vals=configs,
            line_arg="provider",
            line_vals=providers,
            line_names=[
                "torch eager" if p == "torch" else f"block={p}" for p in providers
            ],
            styles=_STYLES[: len(providers)],
            ylabel="GB/s",
            plot_name="hisparse-transfer-cache-dsv4-mla",
            args={},
        ),
    )


# ---------------------------------------------------------------------------
# speedup analysis
# ---------------------------------------------------------------------------


def speedup_analysis(df, transfer_block_size):
    if "impl" not in df.columns or "torch" not in set(df["impl"]):
        return

    print("\n" + "=" * 80)
    print("SPEEDUP ANALYSIS  (torch eager / sglang, higher = kernel is faster)")
    print("=" * 80)

    for op, sub in df.groupby("op", sort=False):
        sgl = sub[sub["impl"] == "sglang"]
        ref = sub[sub["impl"] == "torch"]
        if sgl.empty or ref.empty:
            continue
        if op == "transfer_cache_dsv4_mla":
            sgl = sgl[sgl["block_size"] == transfer_block_size]
            group, label = "num_layers", f"block={transfer_block_size}"
        else:
            group, label = "layout", "all layouts"

        keys = [
            c
            for c in ("layout", "batch_size", "num_top_k", "num_items", "num_layers")
            if c in sub.columns
        ]
        merged = sgl.merge(ref, on=keys, how="inner", suffixes=("_sgl", "_torch"))
        if merged.empty:
            continue
        merged["speedup"] = merged["us (median)_torch"] / merged["us (median)_sgl"]

        print(f"\n### {op}  ({label})\n")
        print(
            f"  overall: avg={merged['speedup'].mean():.2f}x  "
            f"max={merged['speedup'].max():.2f}x  "
            f"min={merged['speedup'].min():.2f}x"
        )
        print(f"\n  by {group}:")
        for value, rows in merged.groupby(group, sort=True):
            print(
                f"    {value!s:>8}: avg={rows['speedup'].mean():6.2f}x  "
                f"max={rows['speedup'].max():6.2f}x  "
                f"min={rows['speedup'].min():6.2f}x"
            )


def check_nonzero():
    probe = torch.tensor([[0, 1, 0], [1, 0, 1]], dtype=torch.bool, device=DEVICE)
    rows, cols = probe.nonzero(as_tuple=True)
    ok = rows.tolist() == [0, 1, 1] and cols.tolist() == [1, 0, 2]
    if not ok:
        print(
            "WARNING: torch.nonzero is wrong on this stack "
            f"(got rows={rows.tolist()} cols={cols.tolist()}, "
            "expected rows=[0, 1, 1] cols=[1, 0, 2]). "
            "The torch-eager baseline is unreliable; speedups will be overstated."
        )
    return ok


# ---------------------------------------------------------------------------
# regression tracking
# ---------------------------------------------------------------------------

# Columns identifying one benchmarked configuration, for joining two runs.
_KEY_COLUMNS = [
    "op",
    "impl",
    "layout",
    "batch_size",
    "num_top_k",
    "hot_buffer_size",
    "num_items",
    "num_layers",
    "block_size",
]


def compare_results(df):
    os.makedirs(RESULT_DIR, exist_ok=True)
    previous_csv = os.path.join(RESULT_DIR, "previous.csv")
    current_csv = os.path.join(RESULT_DIR, "current.csv")

    df.to_csv(current_csv, index=False)
    print(f"\nCurrent results saved to: {current_csv}")

    if not os.path.exists(previous_csv):
        print(f"No {previous_csv} found, nothing to compare against.")
        print("Tip: copy current.csv to previous.csv to set a baseline.")
        return

    try:
        prev = pd.read_csv(previous_csv)
    except Exception as e:  # noqa: BLE001 -- a stale CSV must not kill the run
        print(f"Error loading {previous_csv}: {e}")
        return
    print(f"Loaded previous results from: {previous_csv}")

    keys = [c for c in _KEY_COLUMNS if c in df.columns and c in prev.columns]
    merged = df.merge(prev, on=keys, how="inner", suffixes=("", "_prev"))
    if merged.empty:
        print("No configurations in common with the previous run.")
        return

    merged["delta %"] = (
        (merged["us (median)"] - merged["us (median)_prev"])
        / merged["us (median)_prev"]
        * 100
    ).round(1)
    report = merged[keys + ["us (median)_prev", "us (median)", "delta %"]].rename(
        columns={"us (median)_prev": "previous us", "us (median)": "current us"}
    )

    print("\n" + "=" * 80)
    print("REGRESSION vs previous.csv  (positive delta = slower than before)")
    print("=" * 80 + "\n")
    print(report.dropna(axis=1, how="all").to_markdown(index=False))
    print(
        f"\nWorst: {report['delta %'].max():+.1f}%    "
        f"Best: {report['delta %'].min():+.1f}%    "
        f"Mean: {report['delta %'].mean():+.1f}%"
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="HiSparse KV-cache benchmark for XPU")
    p.add_argument(
        "--suite",
        nargs="+",
        choices=["swapin", "transfer"],
        default=["swapin", "transfer"],
        help="Which ops to benchmark (default: both)",
    )
    p.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 8, 32, 128],
        metavar="B",
        help="Swap-in batch sizes (default: 1 8 32 128)",
    )
    p.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[64, 256],
        metavar="K",
        help="Swap-in num_top_k values; hot_buffer_size is set equal to each "
        "(default: 64 256)",
    )
    p.add_argument(
        "--swapin-block-size",
        type=int,
        default=256,
        help="block_size for the swap-in op (default: 256)",
    )
    p.add_argument(
        "--reps",
        type=int,
        default=20,
        help="Reps for the cold (miss) swap-in regime (default: 20)",
    )
    p.add_argument(
        "--num-items",
        type=int,
        nargs="+",
        default=[64, 512, 4096, 32768],
        metavar="N",
        help="Transfer item counts (default: 64 512 4096 32768)",
    )
    p.add_argument(
        "--num-layers",
        type=int,
        nargs="+",
        default=[1, 8, 61],
        metavar="L",
        help="Transfer layer counts (default: 1 8 61)",
    )
    p.add_argument(
        "--block-sizes",
        type=int,
        nargs="+",
        default=[256, 512, 1024],
        choices=[256, 512, 1024],
        metavar="B",
        help="block_size values for the transfer op (default: 256 512 1024)",
    )
    p.add_argument(
        "--with-torch",
        action="store_true",
        help="Add a torch-eager provider and print a speedup analysis. Off by "
        "default: the eager path allocates heavily and its allocator traffic "
        "widens the kernels' own measured spread",
    )
    p.add_argument(
        "--save-csv",
        action="store_true",
        help="Save results to CSV and diff against the previous run "
        "(default: only print results)",
    )
    return p.parse_args()


def main():
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        print("ERROR: no XPU device available.")
        raise SystemExit(1)

    args = parse_args()
    props = torch.xpu.get_device_properties(0)
    print(f"Device : {torch.xpu.get_device_name(0)}")
    print(f"VRAM   : {props.total_memory / 1024**3:.1f} GiB")
    print(f"dtype  : {DTYPE}")
    print(f"suites : {args.suite}")
    print(f"torch  : {'yes' if args.with_torch else 'no (kernels only)'}\n")

    if args.with_torch:
        check_nonzero()

    all_results.clear()

    if "swapin" in args.suite:
        configs = [(b, k, k) for b, k in product(args.batch_sizes, args.top_k)]
        swapin_mark(configs, args.with_torch).run(
            print_data=True,
            show_plots=False,
            save_path=None,
            block_size=args.swapin_block_size,
            reps=args.reps,
        )
        release_host_caches()

    if "transfer" in args.suite:
        configs = list(product(args.num_items, args.num_layers))
        transfer_mark(configs, args.block_sizes, args.with_torch).run(
            print_data=True, show_plots=False, save_path=None
        )
        release_transfer_state()

    if not all_results:
        print("No results collected.")
        return

    print("\nBenchmark finished!")
    df = pd.DataFrame(all_results)
    for op, sub in df.groupby("op", sort=False):
        sub = sub.dropna(axis=1, how="all").reset_index(drop=True)
        print(f"\n### {op}\n")
        print(sub.drop(columns=["op"]).to_markdown(index=False))

    speedup_analysis(df, max(args.block_sizes))

    if args.save_csv:
        compare_results(df)


if __name__ == "__main__":
    main()
