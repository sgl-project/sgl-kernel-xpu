from typing import List, Tuple

import pandas as pd
import torch
import triton
import triton.language as tl
from sgl_kernel import TreeMaskMode, build_tree_kernel_efficient

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


@triton.jit
def sgl_build_tree_kernel_efficient_triton(
    parent_list_ptr,
    selected_index_ptr,
    verified_seq_len_ptr,
    seq_len_prefix_sum_ptr,
    tree_mask_ptr,
    positions_ptr,
    retrieve_index_ptr,
    retrieve_next_token_ptr,
    retrieve_next_sibling_ptr,
    topk: tl.constexpr,
    depth: tl.constexpr,
    draft_token_num: tl.constexpr,
    tree_mask_mode: tl.constexpr,
    batch_size: tl.constexpr,
    parent_list_stride: tl.constexpr,
    selected_index_stride: tl.constexpr,
):
    """
    Triton kernel for building EAGLE tree structure.
    Each program handles one batch item (batch_idx).
    """
    batch_idx = tl.program_id(0)

    # Calculate seq_tree_idx
    seq_len = tl.load(verified_seq_len_ptr + batch_idx)
    seq_len_prefix_sum = tl.load(seq_len_prefix_sum_ptr + batch_idx)

    # Cast initial value to match the dtype of loaded tensors to avoid type inconsistency
    seq_tree_idx = (
        tl.cast(draft_token_num * draft_token_num * batch_idx, seq_len.dtype)
        + seq_len_prefix_sum * draft_token_num
    )

    positions_offset = batch_idx * draft_token_num
    tl.store(positions_ptr + positions_offset, seq_len)

    retrieve_index_offset = batch_idx * draft_token_num

    # Build retrieval index structure (reverse loop from draft_token_num-1 to 1)
    for i in range(draft_token_num - 1, 0, -1):
        current_token_idx = retrieve_index_offset + i
        tl.store(
            retrieve_index_ptr + batch_idx * draft_token_num + i,
            current_token_idx,
        )

        parent_tb_idx = (
            tl.load(selected_index_ptr + batch_idx * selected_index_stride + (i - 1))
            // topk
        )
        parent_position = 0
        found = 0

        if parent_tb_idx == 0:
            found = 1
        else:
            parent_token_idx = tl.load(
                parent_list_ptr + batch_idx * parent_list_stride + parent_tb_idx
            )

            # Find parent position
            for pp in range(draft_token_num - 1):
                if found == 0:
                    sel_idx = tl.load(
                        selected_index_ptr + batch_idx * selected_index_stride + pp
                    )
                    if sel_idx == parent_token_idx:
                        parent_position = pp + 1
                        found = 1

        if found == 1:
            # Update next token links
            next_tok_addr = (
                retrieve_next_token_ptr + batch_idx * draft_token_num + parent_position
            )
            next_tok = tl.load(next_tok_addr)

            if next_tok == -1:
                tl.store(next_tok_addr, i)
            else:
                tl.store(next_tok_addr, i)
                tl.store(
                    retrieve_next_sibling_ptr + batch_idx * draft_token_num + i,
                    next_tok,
                )

    tl.store(retrieve_index_ptr + batch_idx * draft_token_num, retrieve_index_offset)

    # Process all draft token indices for tree mask
    for draft_tokenx in range(draft_token_num):
        if tree_mask_mode == 0:  # FULL_MASK
            token_tree_idx = (
                seq_tree_idx + (seq_len + draft_token_num) * draft_tokenx + seq_len + 1
            )
        else:
            token_tree_idx = (
                draft_token_num * draft_token_num * batch_idx
                + draft_token_num * draft_tokenx
                + 1
            )

        tl.store(tree_mask_ptr + token_tree_idx - 1, 1)
        for i in range(draft_token_num - 1):
            tl.store(tree_mask_ptr + token_tree_idx + i, 0)

        if draft_tokenx > 0:
            # Build tree path for draft_tokenx > 0
            cur_position = draft_tokenx - 1
            position = 0
            should_continue = 1

            for _ in range(depth):
                if should_continue:
                    position += 1
                    tl.store(tree_mask_ptr + token_tree_idx + cur_position, 1)

                    parent_tb_idx = (
                        tl.load(
                            selected_index_ptr
                            + batch_idx * selected_index_stride
                            + cur_position
                        )
                        // topk
                    )
                    if parent_tb_idx == 0:
                        should_continue = 0
                    else:
                        parent_token_idx = tl.load(
                            parent_list_ptr
                            + batch_idx * parent_list_stride
                            + parent_tb_idx
                        )

                        # Find cur_position for next iteration
                        found = 0
                        for cp in range(draft_token_num - 1):
                            if found == 0:
                                if (
                                    tl.load(
                                        selected_index_ptr
                                        + batch_idx * selected_index_stride
                                        + cp
                                    )
                                    == parent_token_idx
                                ):
                                    cur_position = cp
                                    found = 1
                        if found == 0:
                            should_continue = 0

            tl.store(
                positions_ptr + batch_idx * draft_token_num + draft_tokenx,
                position + seq_len,
            )


def sgl_build_tree_kernel_triton(
    parent_list: torch.Tensor,
    selected_index: torch.Tensor,
    verified_seq_len: torch.Tensor,
    tree_mask: torch.Tensor,
    positions: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    topk: int,
    depth: int,
    draft_token_num: int,
    tree_mask_mode: TreeMaskMode = TreeMaskMode.FULL_MASK,
):
    """Triton-based implementation."""
    # TODO: Add support for QLEN_ONLY_BITPACKING mode
    if tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
        raise NotImplementedError(
            "QLEN_ONLY_BITPACKING is not supported in Triton implementation"
        )

    batch_size = verified_seq_len.shape[0]
    seq_len_prefix_sum = torch.cumsum(verified_seq_len, dim=0) - verified_seq_len

    # Launch kernel with one program per batch item
    grid = (batch_size,)

    sgl_build_tree_kernel_efficient_triton[grid](
        parent_list,
        selected_index,
        verified_seq_len,
        seq_len_prefix_sum,
        tree_mask,
        positions,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        topk=topk,
        depth=depth,
        draft_token_num=draft_token_num,
        tree_mask_mode=int(tree_mask_mode),
        batch_size=batch_size,
        # A 1-D parent_list is one flat list shared by every request, so the
        # per-request offset must vanish: batch_idx * 0 keeps all programs
        # reading the same entries instead of walking past the end.
        parent_list_stride=(parent_list.stride(0) if parent_list.dim() > 1 else 0),
        selected_index_stride=selected_index.stride(0),
    )


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
