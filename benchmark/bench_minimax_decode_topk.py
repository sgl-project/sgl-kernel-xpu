import itertools
import os

import pandas as pd
import torch
import triton
from sgl_kernel import minimax_decode_topk, minimax_decode_topk_page_table

DEVICE = "xpu"


SMALL_THRESHOLD = 128  # kSmallThreshold = 8 * kNumWarps
CTA_SIZE = 512
MAX_NUM_BLOCKS = 4096

# MiniMax lightning-attention decode shape defaults.
BLOCK_SIZE = 64
PAGE_SIZE = 16
TOPK = 16
NUM_QO_HEADS = 32  # block-id kernel is per query head
NUM_KV_HEADS = 8  # page-table kernel is per kv head

SCORE_BYTES = 4
IDX_BYTES = 4

# Both kernels are selection kernels: no multiply-accumulate work to count, so
# achieved bandwidth against the device spec sheet is the only meaningful
# efficiency figure. The driver cannot supply peak bandwidth -- it reports
# memory_bus_width=64 per controller rather than the 192-bit aggregate, and
# memory_clock_rate as the base clock rather than the GDDR6 data rate, so a
# derived number would be several times too low.
PEAK_BW_GB_S = {
    "Intel(R) Arc(TM) Pro B60 Graphics": 456.0,
}

all_results = []


def _peak_bw_gb_s():
    """Device peak memory bandwidth in GB/s, or None if unknown."""
    override = os.environ.get("SGL_PEAK_BW_GB_S")
    if override:
        return float(override)
    return PEAK_BW_GB_S.get(torch.xpu.get_device_properties(0).name)


def _release(*objs):
    for o in objs:
        if isinstance(o, dict):
            o.clear()
    torch.xpu.synchronize()
    torch.xpu.empty_cache()


def _regime(num_blocks, topk):
    """Label the code path topk_forward will take for this shape."""
    if num_blocks <= topk:
        return "trivial"
    if num_blocks <= SMALL_THRESHOLD:
        return "small"
    if num_blocks <= CTA_SIZE:
        return "register-1"
    return "register-M"


def _bytes_block_id(num_blocks, batch, num_heads, topk):
    """Bytes the block-id kernel actually touches for this shape.

    The trivial regime (num_blocks <= topk) emits block ids straight from the
    loop counter and never reads score, so counting the score row there would
    inflate the reported bandwidth.
    """
    per_row = topk * IDX_BYTES
    if num_blocks > topk:
        per_row += num_blocks * SCORE_BYTES
    return num_heads * batch * per_row + batch * SCORE_BYTES


def _bytes_page_table(num_blocks, batch, num_heads, topk, block_size, page_size):
    """Bytes the page-table kernel actually touches for this shape.

    Every emitted page costs one req_to_token read plus one page_table write.
    The trivial regime emits num_blocks * ppb pages instead of topk * ppb, and
    skips the score row entirely.
    """
    ppb = block_size // page_size
    pages = (num_blocks if num_blocks <= topk else topk) * ppb
    per_row = 2 * pages * IDX_BYTES + IDX_BYTES
    if num_blocks > topk:
        per_row += num_blocks * SCORE_BYTES
    return num_heads * batch * per_row + batch * SCORE_BYTES


def _make_topk_state(num_blocks, batch, num_heads, topk, block_size, seed=0):
    torch.manual_seed(seed)
    # At least topk wide so the trivial regime still has a well-formed score
    # row; the kernel clamps num_blocks to max_seqblock, and seq_lens alone
    # selects the regime.
    max_seqblock = max(num_blocks, topk)
    score = torch.randn(
        (num_heads, batch, max_seqblock), dtype=torch.float32, device=DEVICE
    )
    seq_lens = torch.full(
        (batch,), num_blocks * block_size, dtype=torch.int32, device=DEVICE
    )
    out = torch.empty((num_heads, batch, topk), dtype=torch.int32, device=DEVICE)
    torch.xpu.synchronize()
    return {
        "score": score,
        "seq_lens": seq_lens,
        "out": out,
        "block_size": block_size,
        "topk": topk,
        "max_seqblock": max_seqblock,
    }


def _sglang_topk_block_ids(state):
    return minimax_decode_topk(
        score=state["score"],
        seq_lens=state["seq_lens"],
        block_size=state["block_size"],
        topk=state["topk"],
        out=state["out"],
    )


def _make_page_table_state(
    num_blocks, batch, num_heads, topk, block_size, page_size, seed=0
):
    torch.manual_seed(seed)
    max_seqblock = max(num_blocks, topk)
    score = torch.randn(
        (num_heads, batch, max_seqblock), dtype=torch.float32, device=DEVICE
    )
    seq_lens = torch.full(
        (batch,), num_blocks * block_size, dtype=torch.int32, device=DEVICE
    )
    max_kv_len = max_seqblock * block_size
    req_to_token = (
        torch.arange(batch * max_kv_len, dtype=torch.int32, device=DEVICE)
        .view(batch, max_kv_len)
        .contiguous()
    )
    slot_ids = torch.arange(batch, dtype=torch.int64, device=DEVICE)
    torch.xpu.synchronize()
    return {
        "score": score,
        "seq_lens": seq_lens,
        "req_to_token": req_to_token,
        "slot_ids": slot_ids,
        "block_size": block_size,
        "topk": topk,
        "page_size": page_size,
        "max_seqblock": max_seqblock,
        "max_kv_len": max_kv_len,
    }


def _sglang_page_table(state):
    return minimax_decode_topk_page_table(
        score=state["score"],
        seq_lens=state["seq_lens"],
        req_to_token=state["req_to_token"],
        slot_ids=state["slot_ids"],
        block_size=state["block_size"],
        topk=state["topk"],
        page_size=state["page_size"],
    )


def _record(kernel, num_blocks, batch, num_heads, topk, ms, moved):
    gb_s = moved / (ms * 1e-3) / 1e9 if ms > 0 else float("nan")
    peak = _peak_bw_gb_s()
    all_results.append(
        {
            "kernel": kernel,
            "case": _regime(num_blocks, topk),
            "num_blocks": num_blocks,
            "batch": batch,
            "num_heads": num_heads,
            "topk": topk,
            "time_us": 1000 * ms,
            "GB_s": gb_s,
            "pct_peak": 100 * gb_s / peak if peak else float("nan"),
        }
    )


NUM_BLOCKS = [8, SMALL_THRESHOLD, CTA_SIZE, MAX_NUM_BLOCKS]
BATCH_SIZES = [1, 16, 64]

blockid_configs = list(itertools.product(NUM_BLOCKS, BATCH_SIZES))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_blocks", "batch"],
        x_vals=blockid_configs,
        line_arg="provider",
        line_vals=["sglang"],
        line_names=["sglang (SYCL)"],
        styles=[("blue", "-")],
        ylabel="us",
        plot_name="minimax-decode-topk-block-id-performance",
        args={},
    )
)
def benchmark_block_id(num_blocks, batch, provider):
    state = _make_topk_state(num_blocks, batch, NUM_QO_HEADS, TOPK, BLOCK_SIZE)
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: _sglang_topk_block_ids(state), quantiles=[0.5, 0.2, 0.8]
    )
    _record(
        "minimax_decode_topk",
        num_blocks,
        batch,
        NUM_QO_HEADS,
        TOPK,
        ms,
        _bytes_block_id(num_blocks, batch, NUM_QO_HEADS, TOPK),
    )
    _release(state)
    return 1000 * ms, 1000 * min_ms, 1000 * max_ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_blocks", "batch"],
        x_vals=blockid_configs,
        line_arg="provider",
        line_vals=["sglang"],
        line_names=["sglang (SYCL)"],
        styles=[("green", "-")],
        ylabel="us",
        plot_name="minimax-decode-topk-page-table-performance",
        args={},
    )
)
def benchmark_page_table(num_blocks, batch, provider):
    state = _make_page_table_state(
        num_blocks, batch, NUM_KV_HEADS, TOPK, BLOCK_SIZE, PAGE_SIZE
    )
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: _sglang_page_table(state), quantiles=[0.5, 0.2, 0.8]
    )
    _record(
        "minimax_decode_topk_page_table",
        num_blocks,
        batch,
        NUM_KV_HEADS,
        TOPK,
        ms,
        _bytes_page_table(num_blocks, batch, NUM_KV_HEADS, TOPK, BLOCK_SIZE, PAGE_SIZE),
    )
    _release(state)
    return 1000 * ms, 1000 * min_ms, 1000 * max_ms


TOPK_SWEEP_NUM_BLOCKS = 1024
TOPK_SWEEP_BATCH = 16
TOPK_VALUES = [1, 4, 16, 32]  # 32 == kMaxTopK


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["topk"],
        x_vals=TOPK_VALUES,
        line_arg="provider",
        line_vals=["sglang"],
        line_names=["sglang (SYCL)"],
        styles=[("blue", "-")],
        ylabel="us",
        plot_name="minimax-decode-topk-vs-topk-performance",
        args={},
    )
)
def benchmark_topk_sweep(topk, provider):
    state = _make_topk_state(
        TOPK_SWEEP_NUM_BLOCKS, TOPK_SWEEP_BATCH, NUM_QO_HEADS, topk, BLOCK_SIZE
    )
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: _sglang_topk_block_ids(state), quantiles=[0.5, 0.2, 0.8]
    )
    _record(
        "minimax_decode_topk (topk sweep)",
        TOPK_SWEEP_NUM_BLOCKS,
        TOPK_SWEEP_BATCH,
        NUM_QO_HEADS,
        topk,
        ms,
        _bytes_block_id(TOPK_SWEEP_NUM_BLOCKS, TOPK_SWEEP_BATCH, NUM_QO_HEADS, topk),
    )
    _release(state)
    return 1000 * ms, 1000 * min_ms, 1000 * max_ms


def _report_bandwidth(df, case_label, title):
    """Print achieved bandwidth against device peak for one kernel."""
    if df.empty:
        return
    peak = _peak_bw_gb_s()
    print("\n" + "=" * 80)
    print(f"Achieved Bandwidth — {title}")
    print("=" * 80)
    if peak:
        print(f"\nDevice peak: {peak:.0f} GB/s")
        print(
            f"Best:    {df['GB_s'].max():.2f} GB/s ({df['pct_peak'].max():.1f}% peak)"
        )
        print(
            f"Median:  {df['GB_s'].median():.2f} GB/s "
            f"({df['pct_peak'].median():.1f}% peak)"
        )
    else:
        print("\nDevice peak unknown; set SGL_PEAK_BW_GB_S to report % of peak.")
        print(f"Best:    {df['GB_s'].max():.2f} GB/s")
        print(f"Median:  {df['GB_s'].median():.2f} GB/s")

    print(f"\nBy {case_label}:")
    for value in dict.fromkeys(df[case_label]):
        rows = df[df[case_label] == value]
        line = f"  {str(value):>14s}: {rows['GB_s'].median():8.2f} GB/s"
        if peak:
            line += f"  ({rows['pct_peak'].median():5.1f}% peak)"
        print(line)


if __name__ == "__main__":
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        print("ERROR: no XPU device available.")
        raise SystemExit(1)

    peak = _peak_bw_gb_s()
    print("MiniMax decode block-top-k kernels (SYCL)")
    print("Kernels are AOT-built into sgl_kernel; no compile step.")
    print(f"device={torch.xpu.get_device_properties(0).name}")
    print(f"peak bandwidth={f'{peak:.0f} GB/s' if peak else 'unknown'}")
    print(
        f"block_size={BLOCK_SIZE} page_size={PAGE_SIZE} topk={TOPK} "
        f"qo_heads={NUM_QO_HEADS} kv_heads={NUM_KV_HEADS}"
    )
    print("=" * 80)
    print("minimax_decode_topk (block-id output)")
    print("=" * 80)
    benchmark_block_id.run(print_data=True)

    print("\n" + "=" * 80)
    print("minimax_decode_topk_page_table (fused top-k + page table)")
    print("=" * 80)
    benchmark_page_table.run(print_data=True)

    print("\n" + "=" * 80)
    print(
        f"minimax_decode_topk vs topk "
        f"(num_blocks={TOPK_SWEEP_NUM_BLOCKS}, batch={TOPK_SWEEP_BATCH})"
    )
    print("=" * 80)
    benchmark_topk_sweep.run(print_data=True)

    df = pd.DataFrame(all_results)
    df["time_us"] = df["time_us"].round(2)
    df["GB_s"] = df["GB_s"].round(2)
    df["pct_peak"] = df["pct_peak"].round(1)

    print("\n" + "=" * 80)
    print("Raw Results")
    print("=" * 80)
    print(df.to_markdown(index=False))

    _report_bandwidth(
        df[df["kernel"] == "minimax_decode_topk"],
        "case",
        "minimax_decode_topk",
    )
    _report_bandwidth(
        df[df["kernel"] == "minimax_decode_topk_page_table"],
        "case",
        "minimax_decode_topk_page_table",
    )
    _report_bandwidth(
        df[df["kernel"] == "minimax_decode_topk (topk sweep)"],
        "topk",
        "minimax_decode_topk (topk sweep)",
    )

    print("\nBenchmark finished!")
