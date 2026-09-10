import pandas as pd
import torch
import triton
from sgl_kernel import verify_tree_greedy
from sgl_kernel.eagle_utils import verify_tree_greedy_triton
from typing import List, Tuple

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

def generate_test_inputs(
    batch_size: int,
    num_draft_tokens: int,
    num_speculative_tokens: int,
    device: str = "cuda",
) -> Tuple[torch.Tensor, ...]:
    """Generate test inputs for verify tree greedy kernels.

    Args:
        batch_size: Number of requests in batch
        num_draft_tokens: Total number of draft tokens (e.g., 21 for topk=4, depth=3)
        num_speculative_tokens: Max speculative tokens to verify (typically num_draft_tokens)
        device: Device to create tensors on
    """
    print(f"   Generating inputs: batch_size={batch_size}, num_draft_tokens={num_draft_tokens}")

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
    retrive_index = torch.arange(
        num_draft_tokens, device=device, dtype=torch.int64
    ).unsqueeze(0).expand(batch_size, -1).contiguous()

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
    print(
        f"benchmark {provider} with b_s={b_s} num_draft_tokens={num_draft_tokens} "
    )
    torch.set_default_device("xpu")
    torch.xpu.manual_seed_all(42)

    num_speculative_tokens = num_draft_tokens

    inputs = generate_test_inputs(
        b_s, num_draft_tokens, num_speculative_tokens, "xpu"
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
            "ms": ms,
            "us": ms * 1e3,
            "draft_Mtok_per_sec": (b_s * num_draft_tokens) / (ms / 1e3) / 1e6,
            "req_per_sec": b_s / (ms / 1e3),
        }
    )
    return ms


if __name__ == "__main__":
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
