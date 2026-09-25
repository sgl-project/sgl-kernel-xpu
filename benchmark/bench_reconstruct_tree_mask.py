from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import triton
import triton.language as tl
from sgl_kernel import reconstruct_indices_from_tree_mask

configs = [
    # (b_s, num_branch_token)
    (1, 8),
    (8, 8),
    (32, 8),
    (64, 8),
    (128, 8),
    (256, 8),
    (64, 16),
    (256, 16),
    (64, 32),
]

all_results = []


@triton.jit
def sgl_reconstruct_indices_from_tree_mask_triton(
    tree_mask_ptr,
    verified_seq_len_ptr,
    positions_ptr,
    retrive_index_ptr,
    retrive_next_token_ptr,
    retrive_next_sibling_ptr,
    draft_token_num: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Triton kernel inverting an NGRAM tree mask into verify metadata.

    One program per request. The mask is the transitive-closure ancestor matrix
    with the diagonal set, so the whole contract is a set of reductions over the
    [n, n] block:
      ancestor[i, j]  = mask[i, j] && j < i        (strict ancestors)
      positions[i]    = |ancestor[i, :]| + verified_seq_len
      parent[i]       = max{ j : ancestor[i, j] }, else -1
      next_token[i]   = min{ k > i : ancestor[k, i] }, else -1
      next_sibling[i] = min{ k > i : parent[k] == parent[i] }, -1 if parent[i] < 0
    """
    bid = tl.program_id(0)
    n = draft_token_num

    idx = tl.arange(0, BLOCK)
    rows = idx[:, None]
    cols = idx[None, :]
    valid = (rows < n) & (cols < n)

    # The whole mask block for this request: mask_block[i, j] == "j is an
    # ancestor of i". Out-of-range lanes read 0 so they never win a reduction.
    mask_block = tl.load(
        tree_mask_ptr + bid * n * n + rows * n + cols, mask=valid, other=0
    ).to(tl.int32)
    ancestor = (mask_block != 0) & (cols < rows) & valid

    out_base = bid * n
    seq_len = tl.load(verified_seq_len_ptr + bid).to(tl.int64)

    # Row-local outputs: depth is the ancestor count, parent the highest one.
    depth = tl.sum(ancestor.to(tl.int32), axis=1)
    tl.store(positions_ptr + out_base + idx, depth.to(tl.int64) + seq_len, mask=idx < n)
    tl.store(
        retrive_index_ptr + out_base + idx, (out_base + idx).to(tl.int64), mask=idx < n
    )
    parent = tl.max(tl.where(ancestor, cols, -1), axis=1)

    # Column scans. Reducing over axis 0 turns "smallest row k that satisfies P"
    # into one min per column; n stands in for "none found" so it loses to any
    # real hit and maps back to -1.
    first_child = tl.min(tl.where(ancestor, rows, n), axis=0)
    tl.store(
        retrive_next_token_ptr + out_base + idx,
        tl.where(first_child < n, first_child, -1).to(tl.int64),
        mask=idx < n,
    )

    # Roots never link to each other: parent[i] < 0 rules the column out, and a
    # root k would fail parent[k] == parent[i] anyway.
    is_sibling = (
        (rows > cols)
        & (parent[:, None] == parent[None, :])
        & (parent[None, :] >= 0)
        & valid
    )
    first_sibling = tl.min(tl.where(is_sibling, rows, n), axis=0)
    tl.store(
        retrive_next_sibling_ptr + out_base + idx,
        tl.where(first_sibling < n, first_sibling, -1).to(tl.int64),
        mask=idx < n,
    )


def reconstruct_indices_from_tree_mask_triton(
    tree_mask: torch.Tensor,
    verified_seq_len: torch.Tensor,
    positions: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    batch_size: int,
    draft_token_num: int,
) -> None:
    """Triton-based implementation, same signature as the SYCL op."""
    # One program per request, lanes padded to the next power of two so the
    # [n, n] mask block is a legal Triton tile at non-power-of-two widths.
    grid = (batch_size,)
    sgl_reconstruct_indices_from_tree_mask_triton[grid](
        tree_mask,
        verified_seq_len,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        draft_token_num=draft_token_num,
        BLOCK=triton.next_power_of_2(draft_token_num),
    )


def make_valid_tree_mask(bs: int, n: int, seed: int) -> np.ndarray:
    """Random *valid* tree mask [bs, n, n]: mask[b, i, j] == node j is an ancestor
    of node i (transitive closure, diagonal set). Each node either roots (~30%) or
    attaches under a uniformly random earlier node, so multi-root batches -- the
    case where roots must not link to each other as siblings -- are covered."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((bs, n, n), dtype=bool)
    for b in range(bs):
        ancestors = [set() for _ in range(n)]
        for i in range(n):
            ancestors[i].add(i)
            if i > 0 and rng.random() >= 0.3:
                parent = int(rng.integers(0, i))
                ancestors[i] |= ancestors[parent]
            for j in ancestors[i]:
                mask[b, i, j] = True
    return mask


def generate_test_inputs(
    batch_size: int,
    num_branch_token: int,
    verified_seq_len: List[int],
    device: str = "xpu",
    seed: int = 0,
) -> Tuple[torch.Tensor, ...]:
    """Generate test inputs for the tree-mask reconstruction kernels.

    Deterministic for a given (batch_size, num_branch_token, seed), so each
    provider can be handed identical inputs and its own output buffers.
    """
    mask = make_valid_tree_mask(batch_size, num_branch_token, seed)
    tree_mask = torch.from_numpy(mask).reshape(-1).contiguous().to(device=device)

    verified_seq_len_tensor = torch.tensor(
        verified_seq_len, device=device, dtype=torch.long
    )

    positions = torch.empty(
        batch_size * num_branch_token, device=device, dtype=torch.long
    )

    # The retrieve buffers are 2D: (batch_size, num_branch_token); pack the three
    # of them into one 3D tensor so they are allocated in one shot.
    retrieve_buf = torch.full(
        (3, batch_size, num_branch_token), -1, device=device, dtype=torch.long
    )
    retrive_index, retrive_next_token, retrive_next_sibling = retrieve_buf

    return (
        tree_mask,
        verified_seq_len_tensor,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
    )


def verify_correctness(
    batch_size: int,
    num_branch_token: int,
    verified_seq_len: List[int],
    device: str = "xpu",
) -> Tuple[bool, str]:
    """Run SYCL and Triton on identical inputs and compare their outputs."""
    outputs = {}
    for provider in ("sycl", "triton"):
        (
            tree_mask,
            verified_seq_len_tensor,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
        ) = generate_test_inputs(batch_size, num_branch_token, verified_seq_len, device)

        impl = (
            reconstruct_indices_from_tree_mask
            if provider == "sycl"
            else reconstruct_indices_from_tree_mask_triton
        )
        impl(
            tree_mask,
            verified_seq_len_tensor,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            batch_size,
            num_branch_token,
        )
        torch.xpu.synchronize()

        outputs[provider] = {
            "positions": positions,
            "retrive_index": retrive_index,
            "retrive_next_token": retrive_next_token,
            "retrive_next_sibling": retrive_next_sibling,
        }

    mismatches = []
    for field in (
        "positions",
        "retrive_index",
        "retrive_next_token",
        "retrive_next_sibling",
    ):
        ref = outputs["sycl"][field]
        got = outputs["triton"][field]
        if not torch.equal(ref, got):
            diff = int((ref != got).sum().item())
            mismatches.append(f"{field} {diff}/{ref.numel()} differ")

    return not mismatches, ", ".join(mismatches)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["b_s", "num_branch_token"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["sycl", "triton"],
        line_names=["SYCL", "Triton"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="Time (ms)",
        plot_name="reconstruct-tree-mask-performance",
        args={},
    )
)
def benchmark(b_s, num_branch_token, provider):
    print(f"benchmark {provider} with b_s={b_s} num_branch_token={num_branch_token}")
    torch.set_default_device("xpu")
    torch.xpu.manual_seed_all(42)

    verified_seq_len = [10] * b_s

    (
        tree_mask,
        verified_seq_len_tensor,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
    ) = generate_test_inputs(b_s, num_branch_token, verified_seq_len, "xpu")

    impl = (
        reconstruct_indices_from_tree_mask
        if provider == "sycl"
        else reconstruct_indices_from_tree_mask_triton
    )
    bench_lambda = lambda: impl(
        tree_mask,
        verified_seq_len_tensor,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        b_s,
        num_branch_token,
    )

    # Warmup
    for _ in range(10):
        bench_lambda()

    torch.xpu.synchronize()

    quantiles = [0.5, 0.25, 0.75]
    ms, _, _ = triton.testing.do_bench(
        bench_lambda, quantiles=quantiles, return_mode="median"
    )

    torch.xpu.empty_cache()

    all_results.append(
        {
            "provider": provider,
            "b_s": b_s,
            "num_branch_token": num_branch_token,
            "us": ms * 1e3,
            "draft_Mtok_per_sec": (b_s * num_branch_token) / (ms / 1e3) / 1e6,
            "req_per_sec": b_s / (ms / 1e3),
        }
    )
    return ms


if __name__ == "__main__":
    torch.set_default_device("xpu")

    print("=" * 80)
    print("CORRECTNESS CHECK: SYCL vs Triton")
    print("=" * 80)
    correctness = {}
    for b_s, num_branch_token in configs:
        ok, detail = verify_correctness(b_s, num_branch_token, [10] * b_s)
        correctness[(b_s, num_branch_token)] = ok
        status = "PASS" if ok else f"FAIL ({detail})"
        print(f"  b_s={b_s:<4} num_branch_token={num_branch_token:<3} {status}")

    num_ok = sum(correctness.values())
    if num_ok == len(configs):
        print(
            f"\nCORRECTNESS VERIFIED: SYCL matches Triton on all "
            f"{len(configs)} configs\n"
        )
    else:
        print(
            f"\nCORRECTNESS FAILED: {len(configs) - num_ok}/{len(configs)} "
            f"configs mismatch\n"
        )

    benchmark.run(print_data=False)
    print("Benchmark finished!")

    df = pd.DataFrame(all_results)
    print("\n" + "=" * 80)
    print("RECONSTRUCT_INDICES_FROM_TREE_MASK BENCHMARK RESULTS")
    print("=" * 80)
    print(df.to_markdown(index=False))
    print("\n")

    keys = ["b_s", "num_branch_token"]
    cmp = df.pivot_table(index=keys, columns="provider", values="us").reset_index()
    cmp = cmp.rename(columns={"sycl": "sycl_us", "triton": "triton_us"})
    cmp["speedup"] = cmp["triton_us"] / cmp["sycl_us"]
    cmp["faster"] = cmp["speedup"].map(lambda s: "SYCL" if s > 1 else "Triton")
    # pivot_table sorts its index; restore the declared `configs` order so this
    # table lines up row-for-row with the one above.
    order = {cfg: i for i, cfg in enumerate(configs)}
    cmp = (
        cmp.assign(
            _order=[order[(r.b_s, r.num_branch_token)] for r in cmp.itertuples()]
        )
        .sort_values("_order")
        .drop(columns="_order")
    )
    print("SYCL vs Triton:")
    print(cmp.to_markdown(index=False))
    print("\n")

    sycl = df[df["provider"] == "sycl"]
    print("Summary Statistics (SYCL):")
    print(f"  Min latency: {sycl['us'].min():.2f} us")
    print(f"  Median latency: {sycl['us'].median():.2f} us")
    print(f"  Max latency: {sycl['us'].max():.2f} us")
    print(f"  Best throughput: {sycl['draft_Mtok_per_sec'].max():.2f} draft Mtok/s")
    print(f"  Best speedup vs Triton: {cmp['speedup'].max():.2f}x")
    print(f"  Worst speedup vs Triton: {cmp['speedup'].min():.2f}x")
