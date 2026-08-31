import itertools

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

all_results = []


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


def _num_blocks_from(seq_lens, block_size, max_seqblock):
    return ((seq_lens.to(torch.int64) + block_size - 1) // block_size).clamp(
        max=max_seqblock
    )


def _make_topk_state(num_blocks, batch, num_heads, topk, block_size, seed=0):
    torch.manual_seed(seed)
    # Widened past num_blocks so the trivial regime still works: torch.topk in
    # the eager reference needs k <= S, and seq_lens alone selects the regime.
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


def _torch_topk_block_ids(state):
    score = state["score"]
    topk = state["topk"]
    num_heads, batch, max_seqblock = score.shape

    num_blocks = _num_blocks_from(state["seq_lens"], state["block_size"], max_seqblock)
    block_ids = torch.arange(max_seqblock, device=score.device)
    valid = block_ids[None, :] < num_blocks[:, None]

    masked = score.masked_fill(~valid[None], float("-inf"))
    idx = masked.topk(topk, dim=-1).indices.to(torch.int32)

    k_eff = num_blocks.clamp(max=topk)
    keep = torch.arange(topk, device=score.device)[None, :] < k_eff[:, None]
    state["out"].copy_(torch.where(keep[None], idx, torch.full_like(idx, -1)))
    return state["out"]


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


def _torch_page_table(state):
    score = state["score"]
    req_to_token = state["req_to_token"]
    topk, block_size, page_size = (
        state["topk"],
        state["block_size"],
        state["page_size"],
    )
    num_heads, batch, max_seqblock = score.shape
    max_reqs, max_kv_len = req_to_token.shape
    ppb = block_size // page_size

    seq_lens = state["seq_lens"].to(torch.int64)
    num_blocks = _num_blocks_from(seq_lens, block_size, max_seqblock)
    block_ids = torch.arange(max_seqblock, device=score.device)
    valid = block_ids[None, :] < num_blocks[:, None]

    masked = score.masked_fill(~valid[None], float("-inf"))
    idx = masked.topk(topk, dim=-1).indices

    k_eff = num_blocks.clamp(max=topk)
    keep = torch.arange(topk, device=score.device)[None, :] < k_eff[:, None]
    # Fill invalid slots with an out-of-range id on purpose, so the sort below
    # pushes them to the tail and the first k_eff entries come out ascending.
    sel = torch.where(keep[None], idx, torch.full_like(idx, max_seqblock))
    sel, _ = sel.sort(dim=-1)

    bid = sel.clamp(max=max_seqblock - 1)
    rem = (seq_lens[None, :, None] - bid * block_size).clamp(min=0, max=block_size)
    real_seq_lens = (rem * keep[None]).sum(dim=-1).to(torch.int32)

    offsets = torch.arange(ppb, device=score.device) * page_size
    tok = (bid[..., None] * block_size + offsets).clamp(max=max_kv_len - 1)

    rows = req_to_token[state["slot_ids"] % max_reqs]
    pages = torch.gather(
        rows[None].expand(num_heads, -1, -1), 2, tok.reshape(num_heads, batch, -1)
    )
    head_ids = torch.arange(num_heads, device=score.device).view(num_heads, 1, 1)
    pages = pages // page_size * num_heads + head_ids

    page_table = (
        pages.reshape(num_heads, batch, topk * ppb)
        .permute(1, 0, 2)
        .reshape(batch * num_heads, topk * ppb)
        .to(torch.int32)
    )
    return page_table, real_seq_lens.permute(1, 0).reshape(-1)


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


NUM_BLOCKS = [8, SMALL_THRESHOLD, CTA_SIZE, MAX_NUM_BLOCKS]
BATCH_SIZES = [1, 16, 64]

blockid_configs = list(itertools.product(NUM_BLOCKS, BATCH_SIZES))
PROVIDERS = ["sglang", "torch"]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_blocks", "batch"],
        x_vals=blockid_configs,
        line_arg="provider",
        line_vals=PROVIDERS,
        line_names=["sglang (SYCL)", "torch eager"],
        styles=[("blue", "-"), ("orange", "--")],
        ylabel="us",
        plot_name="minimax-decode-topk-block-id-performance",
        args={},
    )
)
def benchmark_block_id(num_blocks, batch, provider):
    state = _make_topk_state(num_blocks, batch, NUM_QO_HEADS, TOPK, BLOCK_SIZE)
    fn = _sglang_topk_block_ids if provider == "sglang" else _torch_topk_block_ids
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: fn(state), quantiles=[0.5, 0.2, 0.8]
    )

    moved = NUM_QO_HEADS * batch * (num_blocks * SCORE_BYTES + TOPK * IDX_BYTES)
    all_results.append(
        {
            "kernel": "minimax_decode_topk",
            "provider": provider,
            "case": _regime(num_blocks, TOPK),
            "num_blocks": num_blocks,
            "batch": batch,
            "num_heads": NUM_QO_HEADS,
            "topk": TOPK,
            "time_us": 1000 * ms,
            "GB_s": moved / (ms * 1e-3) / 1e9 if ms > 0 else float("nan"),
        }
    )
    _release(state)
    return 1000 * ms, 1000 * min_ms, 1000 * max_ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_blocks", "batch"],
        x_vals=blockid_configs,
        line_arg="provider",
        line_vals=PROVIDERS,
        line_names=["sglang (SYCL)", "torch eager"],
        styles=[("green", "-"), ("orange", "--")],
        ylabel="us",
        plot_name="minimax-decode-topk-page-table-performance",
        args={},
    )
)
def benchmark_page_table(num_blocks, batch, provider):
    state = _make_page_table_state(
        num_blocks, batch, NUM_KV_HEADS, TOPK, BLOCK_SIZE, PAGE_SIZE
    )
    fn = _sglang_page_table if provider == "sglang" else _torch_page_table
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: fn(state), quantiles=[0.5, 0.2, 0.8]
    )

    ppb = BLOCK_SIZE // PAGE_SIZE
    moved = (
        NUM_KV_HEADS
        * batch
        * (num_blocks * SCORE_BYTES + 2 * TOPK * ppb * IDX_BYTES + IDX_BYTES)
    )
    all_results.append(
        {
            "kernel": "minimax_decode_topk_page_table",
            "provider": provider,
            "case": _regime(num_blocks, TOPK),
            "num_blocks": num_blocks,
            "batch": batch,
            "num_heads": NUM_KV_HEADS,
            "topk": TOPK,
            "time_us": 1000 * ms,
            "GB_s": moved / (ms * 1e-3) / 1e9 if ms > 0 else float("nan"),
        }
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
        line_vals=PROVIDERS,
        line_names=["sglang (SYCL)", "torch eager"],
        styles=[("blue", "-"), ("orange", "--")],
        ylabel="us",
        plot_name="minimax-decode-topk-vs-topk-performance",
        args={},
    )
)
def benchmark_topk_sweep(topk, provider):
    state = _make_topk_state(
        TOPK_SWEEP_NUM_BLOCKS, TOPK_SWEEP_BATCH, NUM_QO_HEADS, topk, BLOCK_SIZE
    )
    fn = _sglang_topk_block_ids if provider == "sglang" else _torch_topk_block_ids
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: fn(state), quantiles=[0.5, 0.2, 0.8]
    )

    moved = (
        NUM_QO_HEADS
        * TOPK_SWEEP_BATCH
        * (TOPK_SWEEP_NUM_BLOCKS * SCORE_BYTES + topk * IDX_BYTES)
    )
    all_results.append(
        {
            "kernel": "minimax_decode_topk (topk sweep)",
            "provider": provider,
            "case": _regime(TOPK_SWEEP_NUM_BLOCKS, topk),
            "num_blocks": TOPK_SWEEP_NUM_BLOCKS,
            "batch": TOPK_SWEEP_BATCH,
            "num_heads": NUM_QO_HEADS,
            "topk": topk,
            "time_us": 1000 * ms,
            "GB_s": moved / (ms * 1e-3) / 1e9 if ms > 0 else float("nan"),
        }
    )
    _release(state)
    return 1000 * ms, 1000 * min_ms, 1000 * max_ms


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
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        print("ERROR: no XPU device available.")
        raise SystemExit(1)

    print("MiniMax decode block-top-k kernels (SYCL) vs torch eager")
    print("Kernels are AOT-built into sgl_kernel; no compile step.")
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

    print("\n" + "=" * 80)
    print("Raw Results")
    print("=" * 80)
    print(df.to_markdown(index=False))

    _report_speedup(
        df[df["kernel"] == "minimax_decode_topk"],
        ["num_blocks", "batch", "case"],
        "case",
        "minimax_decode_topk",
    )
    _report_speedup(
        df[df["kernel"] == "minimax_decode_topk_page_table"],
        ["num_blocks", "batch", "case"],
        "case",
        "minimax_decode_topk_page_table",
    )
    _report_speedup(
        df[df["kernel"] == "minimax_decode_topk (topk sweep)"],
        ["topk", "case"],
        "topk",
        "minimax_decode_topk (topk sweep)",
    )

    print("\nBenchmark finished!")
