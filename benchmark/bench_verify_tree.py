from typing import Tuple

import pandas as pd
import torch
import triton
import triton.language as tl
from sgl_kernel import verify_tree_greedy

configs = [
    # (b_s,  num_draft_tokens)
    (1, 21),
    (4, 21),
    (8, 21),
    (16, 21),
    (1, 73),
    (1, 85),
    (32, 21),
    (1, 341),
]

all_results = []


@triton.jit
def verify_tree_greedy_kernel_triton(
    predicts_ptr,
    accept_index_ptr,
    accept_token_num_ptr,
    candidates_ptr,
    retrieve_index_ptr,
    retrieve_next_token_ptr,
    retrieve_next_sibling_ptr,
    target_predict_ptr,
    batch_size: tl.constexpr,
    num_speculative_tokens: tl.constexpr,
    num_draft_tokens: tl.constexpr,
):
    """
    Triton kernel for verifying EAGLE tree in greedy mode.
    Each program handles one batch item.
    """
    bx = tl.program_id(0)

    # Initialize
    last_accept_retrieve_idx = tl.load(retrieve_index_ptr + bx * num_draft_tokens)
    tl.store(accept_index_ptr + bx * num_speculative_tokens, last_accept_retrieve_idx)
    # Cast to match dtype of loaded tensors to avoid type inconsistency
    num_accept_tokens = tl.cast(0, last_accept_retrieve_idx.dtype)
    cur_index = tl.cast(0, last_accept_retrieve_idx.dtype)

    # Tree traversal loop
    should_continue = 1
    for j in range(1, num_speculative_tokens):
        if should_continue:  # Early exit guard
            cur_index = tl.load(
                retrieve_next_token_ptr + bx * num_draft_tokens + cur_index
            )

            # Load target token once per level (before sibling search)
            # last_accept_retrieve_idx is constant during sibling traversal
            target_row = last_accept_retrieve_idx // num_draft_tokens
            target_col = last_accept_retrieve_idx % num_draft_tokens
            target_token = tl.load(
                target_predict_ptr + target_row * num_draft_tokens + target_col
            )

            # Traverse siblings
            found_match = 0
            for _ in range(num_draft_tokens):  # Max iterations = num_draft_tokens
                if found_match == 0:  # Early exit guard
                    # Check if we've reached end of sibling list
                    is_valid = cur_index != -1

                    # Use masked loads with safe address (0 when invalid)
                    safe_cur_index = (
                        cur_index * is_valid
                    )  # 0 if invalid, cur_index if valid
                    safe_index = bx * num_draft_tokens + safe_cur_index

                    # Load draft token info (loads from index 0 when invalid, but we won't use it)
                    draft_index = tl.load(retrieve_index_ptr + safe_index)
                    draft_token = tl.load(candidates_ptr + safe_index)

                    # Check for token match (only valid when is_valid is True)
                    token_match = is_valid & (draft_token == target_token)

                    # Accept token using predicated stores (only write if matched)
                    tl.store(
                        predicts_ptr + last_accept_retrieve_idx,
                        target_token,
                        mask=token_match,
                    )
                    next_num_accept_tokens = num_accept_tokens + 1
                    tl.store(
                        accept_index_ptr
                        + bx * num_speculative_tokens
                        + next_num_accept_tokens,
                        draft_index,
                        mask=token_match,
                    )

                    num_accept_tokens = num_accept_tokens + token_match
                    last_accept_retrieve_idx = (
                        token_match * draft_index
                        + (~token_match) * last_accept_retrieve_idx
                    )
                    found_match = token_match * 1 + (~is_valid) * (-1)

                    # Masked load: only load next sibling when no match (hardware predication)
                    # When matched: returns cur_index (other); when not matched: loads sibling
                    cur_index = tl.load(
                        retrieve_next_sibling_ptr + safe_index,
                        mask=~token_match
                        & is_valid,  # Only load when valid and NOT matched
                        other=cur_index,  # Keep cur_index when matched or invalid
                    )

            if found_match != 1:
                should_continue = 0

    # Store final results
    tl.store(accept_token_num_ptr + bx, num_accept_tokens)

    target_row = last_accept_retrieve_idx // num_draft_tokens
    target_col = last_accept_retrieve_idx % num_draft_tokens
    final_target = tl.load(
        target_predict_ptr + target_row * num_draft_tokens + target_col
    )
    tl.store(predicts_ptr + last_accept_retrieve_idx, final_target)


def verify_tree_greedy_triton(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
):
    """Triton-based implementation."""
    batch_size = candidates.shape[0]
    num_speculative_tokens = accept_index.shape[1]
    num_draft_tokens = candidates.shape[1]

    # Launch kernel with one program per batch item
    grid = (batch_size,)

    verify_tree_greedy_kernel_triton[grid](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        target_predict,
        batch_size=batch_size,
        num_speculative_tokens=num_speculative_tokens,
        num_draft_tokens=num_draft_tokens,
    )


def generate_test_inputs(
    batch_size: int,
    num_draft_tokens: int,
    num_speculative_tokens: int,
    device: str = "xpu",
) -> Tuple[torch.Tensor, ...]:
    """Generate test inputs for verify tree greedy kernels.

    Args:
        batch_size: Number of requests in batch
        num_draft_tokens: Total number of draft tokens (e.g., 21 for topk=4, depth=3)
        num_speculative_tokens: Max speculative tokens to verify (typically num_draft_tokens)
        device: Device to create tensors on
    """
    # Generate random but valid inputs
    # Note: CUDA kernel has mixed dtype requirements:
    # - Output tensors (predicts, accept_index, accept_token_num): int32
    # - Token ID tensors (candidates, target_predict): int64
    # - Tree structure tensors (retrive_*): int64

    # predicts: flattened predictions for all batches (int32 for CUDA)
    predicts = torch.full(
        (batch_size * num_draft_tokens,), -1, device=device, dtype=torch.int32
    )

    # accept_index: tracks which draft tokens were accepted (int32 for CUDA)
    accept_index = torch.full(
        (batch_size, num_speculative_tokens), -1, device=device, dtype=torch.int32
    )

    # accept_token_num: number of accepted tokens per batch (int32 for CUDA)
    accept_token_num = torch.zeros((batch_size,), device=device, dtype=torch.int32)

    # candidates: draft token IDs (int64 for CUDA)
    candidates = torch.randint(
        0, 32000, (batch_size, num_draft_tokens), device=device, dtype=torch.int64
    )

    # retrive_index: indices in the tree structure (int64 for CUDA)
    retrive_index = (
        torch.arange(num_draft_tokens, device=device, dtype=torch.int64)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .contiguous()
    )

    # retrive_next_token: next token in traversal (int64 for CUDA)
    # Use a simple linear chain structure to avoid potential infinite loops
    retrive_next_token = torch.full(
        (batch_size, num_draft_tokens), -1, device=device, dtype=torch.int64
    )
    # Create a simple chain: 0 -> 1 -> 2 -> ... -> n-1 -> -1
    for i in range(num_draft_tokens - 1):
        retrive_next_token[:, i] = i + 1

    # retrive_next_sibling: sibling node in tree (int64 for CUDA)
    # Set all to -1 (no siblings in a simple chain)
    retrive_next_sibling = torch.full(
        (batch_size, num_draft_tokens), -1, device=device, dtype=torch.int64
    )

    # target_predict: target model predictions (int64 for CUDA)
    target_predict = torch.randint(
        0, 32000, (batch_size, num_draft_tokens), device=device, dtype=torch.int64
    )

    # Make some candidates match target to simulate acceptance
    match_ratio = 0.3  # 30% match rate
    num_matches = int(num_draft_tokens * match_ratio)
    for b in range(batch_size):
        match_indices = torch.randperm(num_draft_tokens)[:num_matches]
        candidates[b, match_indices] = target_predict[b, match_indices]

    return (
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        target_predict,
    )


def verify_correctness(
    batch_size: int,
    num_draft_tokens: int,
    device: str = "xpu",
) -> Tuple[bool, str]:
    """Run SYCL and Triton on identical inputs and compare their outputs."""
    torch.xpu.manual_seed_all(42)
    inputs = generate_test_inputs(
        batch_size, num_draft_tokens, num_draft_tokens, device
    )
    (
        predicts_template,
        accept_index_template,
        accept_token_num_template,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        target_predict,
    ) = inputs

    outputs = {}
    for provider in ("sycl", "triton"):
        # Each provider gets its own output buffers so they cannot alias.
        predicts = predicts_template.clone()
        accept_index = accept_index_template.clone()
        accept_token_num = accept_token_num_template.clone()

        if provider == "sycl":
            verify_tree_greedy(
                predicts=predicts,
                accept_index=accept_index,
                accept_token_num=accept_token_num,
                candidates=candidates,
                retrive_index=retrive_index,
                retrive_next_token=retrive_next_token,
                retrive_next_sibling=retrive_next_sibling,
                target_predict=target_predict,
            )
        else:
            verify_tree_greedy_triton(
                predicts=predicts,
                accept_index=accept_index,
                accept_token_num=accept_token_num,
                candidates=candidates,
                retrieve_index=retrive_index,
                retrieve_next_token=retrive_next_token,
                retrieve_next_sibling=retrive_next_sibling,
                target_predict=target_predict,
            )
        torch.xpu.synchronize()

        outputs[provider] = {
            "predicts": predicts,
            "accept_index": accept_index,
            "accept_token_num": accept_token_num,
        }

    mismatches = []
    for field in ("predicts", "accept_index", "accept_token_num"):
        ref = outputs["sycl"][field]
        got = outputs["triton"][field]
        if not torch.equal(ref, got):
            diff = int((ref != got).sum().item())
            mismatches.append(f"{field} {diff}/{ref.numel()} differ")

    return not mismatches, ", ".join(mismatches)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["b_s", "num_draft_tokens"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["sycl", "triton"],
        line_names=["SYCL", "Triton"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="Time (ms)",
        plot_name="verify-tree-performance",
        args={},
    )
)
def benchmark(b_s, num_draft_tokens, provider):
    print(f"benchmark {provider} with b_s={b_s} num_draft_tokens={num_draft_tokens} ")
    torch.set_default_device("xpu")
    torch.xpu.manual_seed_all(42)

    num_speculative_tokens = num_draft_tokens

    inputs = generate_test_inputs(b_s, num_draft_tokens, num_speculative_tokens, "xpu")

    (
        predicts_template,
        accept_index_template,
        accept_token_num_template,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        target_predict,
    ) = inputs

    predicts_sycl = predicts_template.clone()
    accept_index_sycl = accept_index_template.clone()
    accept_token_num_sycl = accept_token_num_template.clone()

    if provider == "sycl":
        bench_lambda = lambda: verify_tree_greedy(
            predicts=predicts_sycl,
            accept_index=accept_index_sycl,
            accept_token_num=accept_token_num_sycl,
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            target_predict=target_predict,
        )
    else:
        bench_lambda = lambda: verify_tree_greedy_triton(
            predicts=predicts_sycl,
            accept_index=accept_index_sycl,
            accept_token_num=accept_token_num_sycl,
            candidates=candidates,
            retrieve_index=retrive_index,
            retrieve_next_token=retrive_next_token,
            retrieve_next_sibling=retrive_next_sibling,
            target_predict=target_predict,
        )

    # Warmup
    for _ in range(10):
        bench_lambda()

    torch.xpu.synchronize()

    total_accepted = int(accept_token_num_sycl.sum().item())

    quantiles = [0.5, 0.25, 0.75]
    ms, _, _ = triton.testing.do_bench(
        bench_lambda, quantiles=quantiles, return_mode="median"
    )

    torch.xpu.empty_cache()

    all_results.append(
        {
            "provider": provider,
            "b_s": b_s,
            "num_draft_tokens": num_draft_tokens,
            "num_spec_steps": num_speculative_tokens,
            "accepted_per_req": total_accepted / b_s,
            "us": ms * 1e3,
            "draft_Mtok_per_sec": (b_s * num_draft_tokens) / (ms / 1e3) / 1e6,
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
    for b_s, num_draft_tokens in configs:
        ok, detail = verify_correctness(b_s, num_draft_tokens)
        correctness[(b_s, num_draft_tokens)] = ok
        status = "PASS" if ok else f"FAIL ({detail})"
        print(f"  b_s={b_s:<4} num_draft_tokens={num_draft_tokens:<5} {status}")

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
    print("VERIFY_TREE_GREEDY BENCHMARK RESULTS")
    print("=" * 80)
    print(df.to_markdown(index=False))
    print("\n")

    keys = ["b_s", "num_draft_tokens"]
    cmp = df.pivot_table(index=keys, columns="provider", values="us").reset_index()
    cmp = cmp.rename(columns={"sycl": "sycl_us", "triton": "triton_us"})
    cmp["speedup"] = cmp["triton_us"] / cmp["sycl_us"]
    cmp["faster"] = cmp["speedup"].map(lambda s: "SYCL" if s > 1 else "Triton")
    # pivot_table sorts its index; restore the declared `configs` order so this
    # table lines up row-for-row with the one above.
    order = {cfg: i for i, cfg in enumerate(configs)}
    cmp = (
        cmp.assign(
            _order=[order[(r.b_s, r.num_draft_tokens)] for r in cmp.itertuples()]
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
