"""Benchmark build_tree_kernel_efficient (EAGLE draft-tree metadata) on XPU.

Compares the SYCL kernel against the upstream Triton kernel (sgl_kernel.eagle_utils,
ported from sglang/kernels/ops/speculative/spec_tree.py) so the baseline is
exactly what the XPU path runs today (eagle_utils dispatches _is_xpu ->
sgl_build_tree_kernel_triton).

The Triton baseline times its cumsum too: the Triton kernel takes
seq_len_prefix_sum as an input, so that launch is part of its cost. The SYCL
kernel folds the prefix sum into the kernel via a group reduction.

Both providers are timed on identical inputs and their outputs are checked
against each other once per config before timing, so a regression shows up as a
correctness failure rather than a misleading speedup.
"""

import itertools

import pandas as pd
import torch
import triton
from sgl_kernel import TreeMaskMode, build_tree_kernel_efficient
from sgl_kernel.eagle_utils import sgl_build_tree_kernel_triton


def run_triton(
    parent_list,
    selected_index,
    seq_lens,
    bufs,
    topk,
    depth,
    draft_token_num,
    mode,
):
    tree_mask, positions, r_index, r_next_token, r_next_sibling = bufs
    sgl_build_tree_kernel_triton(
        parent_list,
        selected_index,
        seq_lens,
        tree_mask,
        positions,
        r_index,
        r_next_token,
        r_next_sibling,
        topk,
        depth,
        draft_token_num,
        mode,
    )


def run_sycl(
    parent_list,
    selected_index,
    seq_lens,
    bufs,
    topk,
    depth,
    draft_token_num,
    mode,
):
    tree_mask, positions, r_index, r_next_token, r_next_sibling = bufs
    build_tree_kernel_efficient(
        parent_list,
        selected_index,
        seq_lens,
        tree_mask,
        positions,
        r_index,
        r_next_token,
        r_next_sibling,
        topk,
        depth,
        draft_token_num,
        mode,
    )


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def gen_draft_tree(bs, topk, num_steps, draft_token_num, device):
    """Simulate the EAGLE draft loop to get a valid (parent_list, selected_index)."""
    scores = torch.rand(bs, topk, dtype=torch.float32, device=device)
    score_chunks = [scores]
    parents_chunks = [
        torch.arange(-1, topk, dtype=torch.int64, device=device).expand(bs, -1)
    ]
    cum_scores = scores
    for i in range(1, num_steps):
        step_p = torch.rand(bs, topk, topk, dtype=torch.float32, device=device)
        expand_scores = cum_scores.unsqueeze(2) * step_p
        cum_scores, topk_cs_index = torch.topk(
            expand_scores.flatten(start_dim=1), topk, dim=-1
        )
        score_chunks.append(expand_scores.flatten(start_dim=1))
        parents_chunks.append(topk_cs_index + (topk * topk * (i - 1) + topk))
    score_flat = torch.cat(score_chunks, dim=1)
    selected_index = torch.sort(
        torch.topk(score_flat, draft_token_num - 1, dim=-1).indices, dim=-1
    ).values
    parent_list = torch.cat(parents_chunks[:-1], dim=1).contiguous()
    return parent_list, selected_index.contiguous()


def alloc_bufs(bs, draft_token_num, seq_lens_sum, mode, device):
    if mode == TreeMaskMode.QLEN_ONLY:
        numel = bs * draft_token_num * draft_token_num
    else:
        numel = seq_lens_sum * draft_token_num + bs * draft_token_num * draft_token_num
    tree_mask = torch.full((numel,), True, dtype=torch.bool, device=device)
    positions = torch.zeros(bs * draft_token_num, dtype=torch.int64, device=device)
    retrieve_buf = torch.full(
        (3, bs, draft_token_num), -1, dtype=torch.int64, device=device
    )
    return (tree_mask, positions, *retrieve_buf)


# Realistic EAGLE / MTP serving shapes: (topk, spec_steps, draft_token_num).
TREE_SHAPES = [
    (1, 3, 4),  # MTP chain
    (4, 3, 8),
    (4, 3, 16),
    (8, 4, 32),
    (8, 5, 64),
]
BATCH_SIZES = [1, 8, 32, 64, 128, 256]
SEQ_LEN = 2048  # committed context per request

configs = list(itertools.product(BATCH_SIZES, TREE_SHAPES))
all_results = []


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["bs", "tree_shape"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["sycl", "triton"],
        line_names=["SYCL", "Triton (upstream)"],
        styles=[("green", "-"), ("blue", "--")],
        ylabel="Time (us)",
        plot_name="build-eagle-tree-performance",
        args={},
    )
)
def benchmark(bs, tree_shape, provider):
    topk, spec_steps, draft_token_num = tree_shape
    device = "xpu"
    torch.manual_seed(42)

    mode = TreeMaskMode.FULL_MASK
    seq_lens = torch.full((bs,), SEQ_LEN, dtype=torch.int64, device=device)
    seq_lens_sum = int(seq_lens.sum())
    parent_list, selected_index = gen_draft_tree(
        bs, topk, spec_steps, draft_token_num, device
    )

    runner = run_sycl if provider == "sycl" else run_triton
    bufs = alloc_bufs(bs, draft_token_num, seq_lens_sum, mode, device)

    args = (
        parent_list,
        selected_index,
        seq_lens,
        bufs,
        topk,
        spec_steps,
        draft_token_num,
        mode,
    )

    # Correctness gate: both providers must agree before we trust the timings.
    ref_bufs = alloc_bufs(bs, draft_token_num, seq_lens_sum, mode, device)
    run_sycl(
        parent_list,
        selected_index,
        seq_lens,
        ref_bufs,
        topk,
        spec_steps,
        draft_token_num,
        mode,
    )
    runner(*args)
    torch.xpu.synchronize()
    for name, got, want in zip(
        (
            "tree_mask",
            "positions",
            "retrieve_index",
            "retrieve_next_token",
            "retrieve_next_sibling",
        ),
        bufs,
        ref_bufs,
    ):
        if not torch.equal(got, want):
            raise AssertionError(
                f"{provider} disagrees with the SYCL reference on {name} "
                f"(bs={bs}, topk={topk}, steps={spec_steps}, N={draft_token_num})"
            )

    for _ in range(10):
        runner(*args)
    torch.xpu.synchronize()

    ms = triton.testing.do_bench(
        lambda: runner(*args), quantiles=[0.5, 0.25, 0.75], return_mode="median"
    )
    if isinstance(ms, (tuple, list)):
        ms = ms[0]
    us = ms * 1e3

    torch.xpu.empty_cache()

    # Mask bytes dominate the traffic: one bool per (draft token, tree column),
    # plus the prefix columns the FULL_MASK layout skips over.
    mask_bytes = bs * draft_token_num * draft_token_num
    all_results.append(
        {
            "provider": provider,
            "bs": bs,
            "topk": topk,
            "steps": spec_steps,
            "draft_tokens": draft_token_num,
            "us": us,
            "nodes_per_sec_M": bs * draft_token_num / (ms / 1e3) / 1e6,
            "mask_cells_per_sec_M": mask_bytes / (ms / 1e3) / 1e6,
        }
    )
    return us


if __name__ == "__main__":
    benchmark.run(print_data=False)
    print("Benchmark finished!")

    df = pd.DataFrame(all_results)
    print("\n" + "=" * 88)
    print("BUILD_EAGLE_TREE BENCHMARK RESULTS")
    print("=" * 88)
    print(df.to_markdown(index=False, floatfmt=".2f"))

    pivot = df.pivot_table(
        index=["bs", "topk", "steps", "draft_tokens"],
        columns="provider",
        values="us",
    )
    if {"sycl", "triton"}.issubset(pivot.columns):
        pivot["speedup_x"] = pivot["triton"] / pivot["sycl"]
        print("\n" + "=" * 88)
        print("SYCL vs Triton (speedup = triton_us / sycl_us; >1 means SYCL wins)")
        print("=" * 88)
        print(pivot.to_markdown(floatfmt=".2f"))
        print(
            f"\n  geomean speedup: "
            f"{pivot['speedup_x'].prod() ** (1 / len(pivot)):.2f}x"
            f"\n  min: {pivot['speedup_x'].min():.2f}x"
            f"   max: {pivot['speedup_x'].max():.2f}x"
        )
    print("\n")
