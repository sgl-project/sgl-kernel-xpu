"""Benchmark the XPU/SYCL HiSparse KV-offload swap-in kernels vs torch eager.

Covers the three JIT entry points added alongside ``tests/test_hisparse_jit.py``:

- ``load_cache_to_device_buffer_mla``      — linear host/device layout
- ``load_cache_to_device_buffer_dsv4_mla`` — page-padded C4 layout
- ``transfer_cache_dsv4_mla``              — bulk evict / backup copy

Each is compared against a pure-PyTorch (``torch`` provider) implementation of
the same operation, so the reported speedup is kernel vs eager on the same
device and the same inputs. The eager versions are written the way a user
naturally would: vectorized advanced indexing, one ``index`` / ``index_copy_``
per layer or per request batch, no fusion across layers.

Two swap-in regimes are reported separately, because both the kernel and the
eager reference mutate ``device_buffer_tokens`` / ``lru_slots`` in place:

- **hit**  — every top-k token is already resident. Idempotent across reps, so
  ``triton.testing.do_bench`` measures it directly. This is the steady state.
- **miss** — every top-k token must stream in from the host cache. Only the
  *first* call on a given state actually misses, so reps are timed one at a
  time with XPU events and the state is rebuilt outside the timed window.

The kernels are templated on ``(block_size, num_top_k, hot_buffer_size,
is_mla, is_dsv4_layout)`` and each distinct tuple triggers its own ``icpx``
compile, so the sweep varies ``batch_size`` (a runtime argument) and keeps the
template configuration list short. Expect a one-off JIT compile pause on the
first run; later runs hit the ``~/.cache/sgl_kernel/jit_sycl`` ``.so`` cache.

Usage (with oneAPI on PATH so JIT compilation can find ``icpx``)::

    source /opt/intel/oneapi/2025.3/oneapi-vars.sh
    ZE_AFFINITY_MASK=0 python benchmark/bench_jit_hisparse.py

Pin the run to a *single* device. These kernels are per-rank, and exposing
several devices to one process halves the achieved memory bandwidth on the
device actually used (measured on Arc Pro B60: 826 GB/s with one device
visible vs 377 GB/s with four). That is a runtime/driver effect, not a
property of these kernels -- a bare ``torch.Tensor.copy_`` shows the same 2x
drop -- but it makes multi-device numbers understate the kernel by ~2x.
"""

import itertools

import pandas as pd
import torch
import triton

try:
    from sgl_kernel.jit import (
        load_cache_to_device_buffer_dsv4_mla,
        load_cache_to_device_buffer_mla,
        transfer_cache_dsv4_mla,
    )

    HAS_SGL_JIT = True
except ImportError:
    HAS_SGL_JIT = False
    print("Warning: sgl_kernel JIT HiSparse not available")

DEVICE = "xpu"

# Linear-layout MLA item: 512 kv-lora + 64 rope, bf16 (matches DSA MLA cache).
KV_DIM = 576
DTYPE = torch.bfloat16
LINEAR_ITEM_BYTES = KV_DIM * torch.empty((), dtype=DTYPE).element_size()

# Page-padded C4 layout constants (must match c4_layout.hpp).
DSV4_PAGE_SIZE = 64
DSV4_VALUE_BYTES = 576
DSV4_SCALE_BYTES = 8
DSV4_ITEM_BYTES = DSV4_VALUE_BYTES + DSV4_SCALE_BYTES
DSV4_PAGE_BYTES = ((DSV4_ITEM_BYTES * DSV4_PAGE_SIZE + 575) // 576) * 576
DSV4_SCALE_OFFSET = DSV4_VALUE_BYTES * DSV4_PAGE_SIZE

BLOCK_SIZE = 256

all_results = []


def _pinned(shape, dtype):
    return torch.empty(shape, dtype=dtype, device="cpu").pin_memory()


def _release(*objs):
    """Drop device buffers and return the memory to the driver.

    The largest transfer config allocates ``num_layers`` src+dst pairs
    (~2.3 GB at 32768 items x 61 layers). Holding several configs' worth live
    at once pushes a 24 GB card into allocator thrash and depresses the
    measured bandwidth by ~2x, so each state is freed once it has been timed.
    """
    for o in objs:
        if isinstance(o, list):
            o.clear()
    torch.xpu.synchronize()
    torch.xpu.empty_cache()


def _dsv4_views(cache):
    """Expose a page-padded C4 cache as (value, scale) views, no copy.

    A page is [VALUE 0..63][SCALE 0..63][pad to 576B]. Plain slicing + ``view``
    would fail (the slice is not contiguous, row stride is kPageBytes), so use
    ``as_strided`` to build genuine views with the layout's own strides.
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
    """Logical token slot -> (page number, offset in page)."""
    return index // DSV4_PAGE_SIZE, index % DSV4_PAGE_SIZE


# ---------------------------------------------------------------------------
# swap-in state
# ---------------------------------------------------------------------------


def _make_swapin_state(batch_size, num_top_k, hot_buffer_size, is_dsv4, regime):
    """Build inputs for one swap-in configuration.

    ``regime="hit"`` seeds ``device_buffer_tokens`` with exactly the requested
    top-k tokens (no host traffic). ``regime="miss"`` seeds it with a disjoint
    token range so every top-k lookup misses and must be streamed in.
    """
    slots_per_req = hot_buffer_size + 1  # +1 reserved newest slot
    num_device_items = batch_size * slots_per_req
    # Host cache holds resident + non-resident tokens for every request.
    num_host_items = batch_size * (num_top_k + hot_buffer_size) + 1

    if is_dsv4:
        host_pages = (num_host_items + DSV4_PAGE_SIZE - 1) // DSV4_PAGE_SIZE
        dev_pages = (num_device_items + DSV4_PAGE_SIZE - 1) // DSV4_PAGE_SIZE
        host_cache = _pinned((host_pages, DSV4_PAGE_BYTES), torch.uint8)
        host_cache.fill_(7)
        device_buffer = torch.zeros(
            (dev_pages, DSV4_PAGE_BYTES), dtype=torch.uint8, device=DEVICE
        )
        item_size_bytes = DSV4_ITEM_BYTES
    else:
        host_cache = _pinned((num_host_items, 1, KV_DIM), DTYPE)
        host_cache.fill_(1.0)
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
        # Resident set == requested set, so no host traffic is needed.
        device_buffer_tokens[:, :num_top_k] = top_k_tokens
    else:
        # Resident set is disjoint from the requested set -> every slot misses.
        device_buffer_tokens[:, :hot_buffer_size] = (
            base
            + num_top_k
            + torch.arange(hot_buffer_size, dtype=torch.int32, device=DEVICE).view(1, -1)
        )
    device_buffer_tokens[:, hot_buffer_size] = top_k_tokens[:, -1]

    lru_slots = (
        torch.arange(hot_buffer_size, dtype=torch.int16, device=DEVICE)
        .view(1, -1)
        .repeat(batch_size, 1)
        .contiguous()
    )
    req_pool_indices = torch.arange(batch_size, dtype=torch.int64, device=DEVICE)
    # seq_len must exceed hot_buffer_size or the kernel takes the fast path.
    seq_lens = torch.full(
        (batch_size,), num_top_k + hot_buffer_size, dtype=torch.int32, device=DEVICE
    )
    top_k_device_locs = torch.full_like(top_k_tokens, -1)
    num_real_reqs = torch.tensor([batch_size], dtype=torch.int32, device=DEVICE)
    torch.xpu.synchronize()

    return {
        "top_k_tokens": top_k_tokens,
        "device_buffer_tokens": device_buffer_tokens,
        "host_cache_locs": host_cache_locs,
        "device_buffer_locs": device_buffer_locs,
        "host_cache": host_cache,
        "device_buffer": device_buffer,
        "top_k_device_locs": top_k_device_locs,
        "req_pool_indices": req_pool_indices,
        "seq_lens": seq_lens,
        "lru_slots": lru_slots,
        "item_size_bytes": item_size_bytes,
        "num_top_k": num_top_k,
        "hot_buffer_size": hot_buffer_size,
        "page_size": DSV4_PAGE_SIZE if is_dsv4 else 1,
        "block_size": BLOCK_SIZE,
        "num_real_reqs": num_real_reqs,
    }


def _sglang_swapin(state, is_dsv4):
    fn = (
        load_cache_to_device_buffer_dsv4_mla
        if is_dsv4
        else load_cache_to_device_buffer_mla
    )
    fn(**state)


# ---------------------------------------------------------------------------
# torch eager reference: swap-in
# ---------------------------------------------------------------------------


def _torch_swapin(state, is_dsv4):
    """Pure-PyTorch equivalent of ``load_cache_to_device_buffer_*_mla``.

    Same observable effect as the kernel: resolve which top-k tokens are already
    resident, assign the misses to the least-recently-used evictable slots,
    stream those items host->device, and refresh the LRU order.

    This is a *timing* reference. It reproduces the resident-token set and the
    LRU ordering, but the kernel's exact miss-to-slot assignment is an internal
    detail, so slot ids may differ; accuracy is covered by tests/.

    The dominant cost here is structural, not a missing optimization: indexing a
    pinned CPU tensor needs the gather indices on the host, which forces a
    device->host sync per call. The kernel does the whole thing on device.
    """
    top_k = state["top_k_tokens"]  # [B, K] int32
    dbt = state["device_buffer_tokens"]  # [B, S] int32
    dbl = state["device_buffer_locs"]  # [B, S] int32
    hcl = state["host_cache_locs"]  # [B, H] int64
    host_cache = state["host_cache"]
    device_buffer = state["device_buffer"]
    lru = state["lru_slots"]  # [B, hot] int16
    hot = state["hot_buffer_size"]

    # ---- classify each requested token as hit or miss -------------------
    eq = top_k.unsqueeze(2) == dbt.unsqueeze(1)  # [B, K, S]
    is_hit = eq.any(dim=2)  # [B, K]
    hit_slot = eq.to(torch.uint8).argmax(dim=2)  # [B, K]
    slot_is_hit = eq.any(dim=1)  # [B, S]

    # ---- pick evict slots in LRU order ----------------------------------
    lru_long = lru.to(torch.int64)
    evictable = ~slot_is_hit.gather(1, lru_long)  # [B, hot] in LRU order
    # Stable sort brings evictable slots first while preserving LRU order.
    order = torch.argsort(~evictable, dim=1, stable=True)
    evict_slots = lru_long.gather(1, order)  # [B, hot]

    # Rank each miss, and match it to the evict slot of the same rank.
    miss = ~is_hit
    miss_rank = miss.cumsum(dim=1) - 1  # [B, K]
    evict_pick = evict_slots.gather(1, miss_rank.clamp_(min=0))
    assigned = torch.where(miss, evict_pick, hit_slot)  # [B, K]

    # ---- stream the misses in -------------------------------------------
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

        # ---- refresh residency + LRU order ------------------------------
        dbt[b_idx, assigned[b_idx, k_idx]] = top_k[b_idx, k_idx]

    state["top_k_device_locs"].copy_(dbl.gather(1, assigned))
    # Touched slots become most-recently-used: untouched keep LRU order first.
    touched = torch.zeros_like(slot_is_hit)
    touched.scatter_(1, assigned, True)
    key = touched.gather(1, lru_long).to(torch.uint8)
    lru.copy_(lru_long.gather(1, torch.argsort(key, dim=1, stable=True)).to(lru.dtype))


def _swapin_call(provider, state, is_dsv4):
    if provider == "sglang":
        _sglang_swapin(state, is_dsv4)
    else:
        _torch_swapin(state, is_dsv4)


def _time_miss_regime(provider, batch_size, num_top_k, hot_buffer_size, is_dsv4, reps=20):
    """Time the cold (all-miss) path, rebuilding state outside the timed window.

    ``do_bench`` cannot be used here: the first call makes every token
    resident, so reps 2..n would measure the hit path instead.
    """
    # Warm up JIT compilation and the module cache before timing anything.
    _swapin_call(
        provider,
        _make_swapin_state(batch_size, num_top_k, hot_buffer_size, is_dsv4, "miss"),
        is_dsv4,
    )
    torch.xpu.synchronize()

    samples = []
    for _ in range(reps):
        state = _make_swapin_state(
            batch_size, num_top_k, hot_buffer_size, is_dsv4, "miss"
        )
        start, end = torch.xpu.Event(enable_timing=True), torch.xpu.Event(
            enable_timing=True
        )
        start.record()
        _swapin_call(provider, state, is_dsv4)
        end.record()
        torch.xpu.synchronize()
        samples.append(start.elapsed_time(end))
        state.clear()  # free before building the next rep's state
    _release()
    samples.sort()
    return samples[len(samples) // 2]  # median ms


# (num_top_k, hot_buffer_size) — each pair is one extra JIT compile per layout.
TEMPLATE_CONFIGS = [(64, 64), (256, 256)]
BATCH_SIZES = [1, 8, 32, 128]

swapin_configs = [
    (b, k, h) for b, (k, h) in itertools.product(BATCH_SIZES, TEMPLATE_CONFIGS)
]

SWAPIN_PROVIDERS = [
    f"{p}-{layout}-{regime}"
    for layout in ("linear", "dsv4")
    for regime in ("hit", "miss")
    for p in ("sglang", "torch")
]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "num_top_k", "hot_buffer_size"],
        x_vals=swapin_configs,
        line_arg="provider",
        line_vals=SWAPIN_PROVIDERS,
        line_names=[p.replace("-", " ") for p in SWAPIN_PROVIDERS],
        styles=[
            ("blue", "-"),
            ("blue", "--"),
            ("cyan", "-"),
            ("cyan", "--"),
            ("green", "-"),
            ("green", "--"),
            ("orange", "-"),
            ("orange", "--"),
        ],
        ylabel="us",
        plot_name="hisparse-load-cache-to-device-buffer-performance",
        args={},
    )
)
def benchmark_swapin(batch_size, num_top_k, hot_buffer_size, provider):
    impl, layout, regime = provider.split("-")
    is_dsv4 = layout == "dsv4"

    if regime == "miss":
        ms = _time_miss_regime(
            impl, batch_size, num_top_k, hot_buffer_size, is_dsv4
        )
        min_ms = max_ms = ms
    else:
        state = _make_swapin_state(
            batch_size, num_top_k, hot_buffer_size, is_dsv4, "hit"
        )
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: _swapin_call(impl, state, is_dsv4), quantiles=[0.5, 0.2, 0.8]
        )
        state.clear()
        _release()

    item_bytes = DSV4_ITEM_BYTES if is_dsv4 else LINEAR_ITEM_BYTES
    # A miss reads one item from host and writes one to the device buffer.
    moved = 0 if regime == "hit" else batch_size * num_top_k * item_bytes * 2
    all_results.append(
        {
            "kernel": "load_cache_to_device_buffer",
            "provider": impl,
            "case": f"{layout}-{regime}",
            "batch_size": batch_size,
            "num_top_k": num_top_k,
            "hot_buffer_size": hot_buffer_size,
            "time_us": 1000 * ms,
            "GB_s": (moved / (ms * 1e-3) / 1e9) if moved and ms > 0 else float("nan"),
        }
    )
    return 1000 * ms, 1000 * min_ms, 1000 * max_ms


# ---------------------------------------------------------------------------
# transfer_cache_dsv4_mla
# ---------------------------------------------------------------------------


def _make_transfer_state(num_items, num_layers, block_size):
    pages = (num_items + DSV4_PAGE_SIZE - 1) // DSV4_PAGE_SIZE
    srcs = [
        torch.full((pages, DSV4_PAGE_BYTES), 3, dtype=torch.uint8, device=DEVICE)
        for _ in range(num_layers)
    ]
    dsts = [
        torch.zeros((pages, DSV4_PAGE_BYTES), dtype=torch.uint8, device=DEVICE)
        for _ in range(num_layers)
    ]
    src_ptrs = torch.tensor(
        [t.data_ptr() for t in srcs], dtype=torch.uint64, device=DEVICE
    )
    dst_ptrs = torch.tensor(
        [t.data_ptr() for t in dsts], dtype=torch.uint64, device=DEVICE
    )
    idx = torch.arange(num_items, dtype=torch.int64, device=DEVICE)
    torch.xpu.synchronize()
    # Keep srcs/dsts alive: src_ptrs only holds raw addresses.
    return srcs, dsts, src_ptrs, dst_ptrs, idx, block_size


def _torch_transfer(srcs, dsts, src_indices, dst_indices):
    """Pure-PyTorch equivalent of ``transfer_cache_dsv4_mla``.

    The kernel walks all layers inside one launch; eager has to issue an
    indexed copy per layer, which is the cost this comparison isolates.
    """
    sp, so = _page_split(src_indices)
    dp, do = _page_split(dst_indices)
    for src, dst in zip(srcs, dsts):
        s_val, s_scale = _dsv4_views(src)
        d_val, d_scale = _dsv4_views(dst)
        d_val[dp, do] = s_val[sp, so]
        d_scale[dp, do] = s_scale[sp, so]


transfer_configs = list(itertools.product([64, 512, 4096, 32768], [1, 8, 61]))

TRANSFER_PROVIDERS = ["sglang-bs256", "sglang-bs512", "sglang-bs1024", "torch"]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_items", "num_layers"],
        x_vals=transfer_configs,
        line_arg="provider",
        line_vals=TRANSFER_PROVIDERS,
        line_names=["sglang block=256", "sglang block=512", "sglang block=1024", "torch"],
        styles=[("blue", "-"), ("green", "-"), ("red", "-"), ("orange", "--")],
        ylabel="us",
        plot_name="hisparse-transfer-cache-dsv4-mla-performance",
        args={},
    )
)
def benchmark_transfer(num_items, num_layers, provider):
    is_torch = provider == "torch"
    block_size = 1024 if is_torch else int(provider.removeprefix("sglang-bs"))
    srcs, dsts, src_ptrs, dst_ptrs, idx, bs = _make_transfer_state(
        num_items, num_layers, block_size
    )

    if is_torch:
        fn = lambda: _torch_transfer(srcs, dsts, idx, idx)
    else:
        fn = lambda: transfer_cache_dsv4_mla(
            src_ptrs=src_ptrs,
            dst_ptrs=dst_ptrs,
            src_indices=idx,
            dst_indices=idx,
            block_size=bs,
        )
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    _release(srcs, dsts)

    moved = num_items * num_layers * DSV4_ITEM_BYTES * 2  # read + write
    all_results.append(
        {
            "kernel": "transfer_cache_dsv4_mla",
            "provider": "torch" if is_torch else "sglang",
            "case": "transfer" if is_torch else f"transfer-bs{bs}",
            "num_items": num_items,
            "num_layers": num_layers,
            "time_us": 1000 * ms,
            "GB_s": moved / (ms * 1e-3) / 1e9 if ms > 0 else float("nan"),
        }
    )
    return 1000 * ms, 1000 * min_ms, 1000 * max_ms


# ---------------------------------------------------------------------------
# speedup analysis
# ---------------------------------------------------------------------------


def _report_speedup(df, index_cols, case_label, title):
    """Print the torch-vs-sglang speedup summary for one kernel."""
    pivot = df.pivot_table(index=index_cols, columns="provider", values="time_us")
    if "torch" not in pivot.columns or "sglang" not in pivot.columns:
        return
    pivot = pivot.dropna(subset=["torch", "sglang"])
    if pivot.empty:
        return

    pivot["speedup"] = pivot["torch"] / pivot["sglang"]
    print("\n" + "=" * 80)
    print(f"Speedup Analysis (torch vs sglang) — {title}")
    print("=" * 80)
    print(f"\nOverall average speedup: {pivot['speedup'].mean():.2f}x")
    print(f"Overall max speedup:     {pivot['speedup'].max():.2f}x")
    print(f"Overall min speedup:     {pivot['speedup'].min():.2f}x")

    print(f"\nSpeedup by {case_label}:")
    levels = pivot.index.get_level_values(case_label)
    for value in dict.fromkeys(levels):
        sp = pivot.loc[levels == value, "speedup"]
        if not sp.empty:
            print(
                f"  {str(value):>14s}: avg={sp.mean():.2f}x  "
                f"max={sp.max():.2f}x  min={sp.min():.2f}x"
            )


if __name__ == "__main__":
    if not HAS_SGL_JIT:
        print("ERROR: sgl_kernel JIT HiSparse kernels unavailable.")
        raise SystemExit(1)
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        print("ERROR: no XPU device available.")
        raise SystemExit(1)

    print("HiSparse swap-in kernels (JIT SYCL) vs torch eager")
    print("First run compiles each template configuration with icpx; be patient.")
    print("=" * 80)
    benchmark_swapin.run(print_data=True)

    print("\n" + "=" * 80)
    print("transfer_cache_dsv4_mla (evict / backup path)")
    print("=" * 80)
    benchmark_transfer.run(print_data=True)

    df = pd.DataFrame(all_results)
    df["time_us"] = df["time_us"].round(2)
    df["GB_s"] = df["GB_s"].round(2)

    print("\n" + "=" * 80)
    print("Raw Results")
    print("=" * 80)
    print(df.to_markdown(index=False))

    swapin = df[df["kernel"] == "load_cache_to_device_buffer"]
    _report_speedup(
        swapin,
        ["batch_size", "num_top_k", "hot_buffer_size", "case"],
        "case",
        "load_cache_to_device_buffer",
    )

    # Compare eager against the default block size only (block is within noise).
    transfer = df[
        (df["kernel"] == "transfer_cache_dsv4_mla")
        & (df["case"].isin(["transfer", "transfer-bs1024"]))
    ]
    _report_speedup(
        transfer,
        ["num_items", "num_layers"],
        "num_layers",
        "transfer_cache_dsv4_mla (block=1024)",
    )

    print("\nBenchmark finished!")
