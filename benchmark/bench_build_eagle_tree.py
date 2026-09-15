from typing import List, Tuple

import pandas as pd
import torch
import triton
from sgl_kernel import TreeMaskMode, build_tree_kernel_efficient
from sgl_kernel.eagle_utils import sgl_build_tree_kernel_triton

configs = [
    # (b_s, topk, depth)
    (1, 4, 3),
    (4, 4, 3),
    (8, 4, 3),
    (1, 8, 3),
    (1, 4, 4),
    (16, 4, 3),
]

all_results = []


def generate_test_inputs(
    batch_size: int,
    topk: int,
    depth: int,
    verified_seq_len: List[int],
    device: str = "xpu",
) -> Tuple[torch.Tensor, ...]:
    """Generate test inputs for tree building kernels.

    Uses a simple root-only topology (all tokens at root) for simplicity and determinism.
    """
    draft_token_num = sum(topk**i for i in range(depth))

    # For a root-only topology (simplest valid case):
    # - selected_index: all zeros (select first child at each position)
    # - parent_list: just [-1] for root
    selected_index = torch.zeros(
        (batch_size, draft_token_num), device=device, dtype=torch.long
    )
    parent_list = torch.full((batch_size, 1), -1, device=device, dtype=torch.long)

    verified_seq_len_tensor = torch.tensor(
        verified_seq_len, device=device, dtype=torch.long
    )

    # Pre-allocate output buffers.
    # tree_mask size matches the FULL_MASK layout used by the kernels.
    seq_lens_sum = sum(verified_seq_len)
    tree_mask_size = (
        seq_lens_sum * draft_token_num + batch_size * draft_token_num * draft_token_num
    )
    tree_mask = torch.full((tree_mask_size,), True, device=device, dtype=torch.bool)

    positions = torch.empty(
        batch_size * draft_token_num, device=device, dtype=torch.long
    )

    # Retrieve buffers are 2D: (batch_size, draft_token_num); pack into one 3D tensor.
    retrieve_buf = torch.full(
        (3, batch_size, draft_token_num),
        -1,
        device=device,
        dtype=torch.long,
    )
    retrive_index, retrive_next_token, retrive_next_sibling = retrieve_buf

    return (
        parent_list,
        selected_index,
        verified_seq_len_tensor,
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
    )


def verify_correctness(
    batch_size: int,
    topk: int,
    depth: int,
    verified_seq_len: List[int],
    device: str = "xpu",
    tree_mask_mode: TreeMaskMode = TreeMaskMode.FULL_MASK,
) -> Tuple[bool, str]:
    """Run SYCL and Triton on identical inputs and compare their outputs."""
    draft_token_num = sum(topk**i for i in range(depth))

    outputs = {}
    for provider in ("sycl", "triton"):
        # generate_test_inputs() is deterministic, so calling it per provider
        # gives each one identical inputs and its own output buffers.
        (
            parent_list,
            selected_index,
            verified_seq_len_tensor,
            tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
        ) = generate_test_inputs(batch_size, topk, depth, verified_seq_len, device)

        if provider == "sycl":
            build_tree_kernel_efficient(
                parent_list,
                selected_index,
                verified_seq_len_tensor,
                tree_mask,
                positions,
                retrive_index,
                retrive_next_token,
                retrive_next_sibling,
                topk,
                depth,
                draft_token_num,
                int(tree_mask_mode),
            )
        else:
            sgl_build_tree_kernel_triton(
                parent_list,
                selected_index,
                verified_seq_len_tensor,
                tree_mask,
                positions,
                retrive_index,
                retrive_next_token,
                retrive_next_sibling,
                topk,
                depth,
                draft_token_num,
                tree_mask_mode,
            )
        torch.xpu.synchronize()

        outputs[provider] = {
            "positions": positions,
            "retrive_index": retrive_index,
            "retrive_next_token": retrive_next_token,
            "retrive_next_sibling": retrive_next_sibling,
            "tree_mask": tree_mask,
        }

    mismatches = []
    for field in (
        "positions",
        "retrive_index",
        "retrive_next_token",
        "retrive_next_sibling",
        "tree_mask",
    ):
        ref = outputs["sycl"][field]
        got = outputs["triton"][field]
        if not torch.equal(ref, got):
            diff = int((ref != got).sum().item())
            mismatches.append(f"{field} {diff}/{ref.numel()} differ")

    return not mismatches, ", ".join(mismatches)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["b_s", "topk", "depth"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["sycl", "triton"],
        line_names=["SYCL", "Triton"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="Time (ms)",
        plot_name="build-tree-performance",
        args={},
    )
)
def benchmark(b_s, topk, depth, provider):
    print(f"benchmark {provider} with b_s={b_s} topk={topk} depth={depth}")
    torch.set_default_device("xpu")
    torch.xpu.manual_seed_all(42)

    verified_seq_len = [10] * b_s

    inputs = generate_test_inputs(b_s, topk, depth, verified_seq_len, "xpu")

    (
        parent_list,
        selected_index,
        verified_seq_len_tensor,
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
    ) = inputs

    draft_token_num = sum(topk**i for i in range(depth))
    tree_mask_mode = TreeMaskMode.FULL_MASK

    if provider == "sycl":
        bench_lambda = lambda: build_tree_kernel_efficient(
            parent_list,
            selected_index,
            verified_seq_len_tensor,
            tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            topk,
            depth,
            draft_token_num,
            int(tree_mask_mode),
        )
    else:
        bench_lambda = lambda: sgl_build_tree_kernel_triton(
            parent_list,
            selected_index,
            verified_seq_len_tensor,
            tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            topk,
            depth,
            draft_token_num,
            tree_mask_mode,
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
            "topk": topk,
            "depth": depth,
            "draft_token_num": draft_token_num,
            "us": ms * 1e3,
            "draft_Mtok_per_sec": (b_s * draft_token_num) / (ms / 1e3) / 1e6,
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
    for b_s, topk, depth in configs:
        ok, detail = verify_correctness(b_s, topk, depth, [10] * b_s)
        correctness[(b_s, topk, depth)] = ok
        status = "PASS" if ok else f"FAIL ({detail})"
        print(f"  b_s={b_s:<4} topk={topk:<3} depth={depth:<3} {status}")

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
    print("BUILD_TREE_KERNEL_EFFICIENT BENCHMARK RESULTS")
    print("=" * 80)
    print(df.to_markdown(index=False))
    print("\n")

    keys = ["b_s", "topk", "depth", "draft_token_num"]
    cmp = df.pivot_table(index=keys, columns="provider", values="us").reset_index()
    cmp = cmp.rename(columns={"sycl": "sycl_us", "triton": "triton_us"})
    cmp["speedup"] = cmp["triton_us"] / cmp["sycl_us"]
    cmp["faster"] = cmp["speedup"].map(lambda s: "SYCL" if s > 1 else "Triton")
    # pivot_table sorts its index; restore the declared `configs` order so this
    # table lines up row-for-row with the one above.
    order = {cfg: i for i, cfg in enumerate(configs)}
    cmp = (
        cmp.assign(_order=[order[(r.b_s, r.topk, r.depth)] for r in cmp.itertuples()])
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
