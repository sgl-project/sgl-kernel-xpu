"""
Performance comparison test for EAGLE verify tree greedy kernels.

Compares Triton vs SYCL (sgl_kernel) implementations across different configurations.

Usage:
    python -m pytest benchmark/bench_verify_eagle_tree.py -v -s

    # Run comprehensive suite directly:
    python benchmark/bench_verify_eagle_tree.py
"""

import time
from typing import Callable, List, Tuple

import pytest
import torch

try:
    from sgl_kernel import verify_tree_greedy as verify_tree_greedy_sycl

    SYCL_KERNEL_AVAILABLE = True
except ImportError:
    SYCL_KERNEL_AVAILABLE = False
    print("Warning: sgl_kernel SYCL implementation not available")

from sgl_kernel.eagle_utils import verify_tree_greedy_triton


class PerfMetrics:
    """Store and compute performance metrics."""

    def __init__(self, name: str):
        self.name = name
        self.latencies = []

    def add(self, latency: float):
        self.latencies.append(latency)

    def summary(self) -> dict:
        if not self.latencies:
            return {"name": self.name, "count": 0}

        latencies_ms = [l * 1000 for l in self.latencies]
        return {
            "name": self.name,
            "count": len(latencies_ms),
            "mean_ms": sum(latencies_ms) / len(latencies_ms),
            "median_ms": sorted(latencies_ms)[len(latencies_ms) // 2],
            "min_ms": min(latencies_ms),
            "max_ms": max(latencies_ms),
            "p95_ms": sorted(latencies_ms)[int(len(latencies_ms) * 0.95)],
            "p99_ms": sorted(latencies_ms)[int(len(latencies_ms) * 0.99)],
        }


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
    print(
        f"   Generating inputs: batch_size={batch_size}, num_draft_tokens={num_draft_tokens}"
    )

    # Generate random but valid inputs
    # Note: SYCL kernel has mixed dtype requirements:
    # - Output tensors (predicts, accept_index, accept_token_num): int32
    # - Token ID tensors (candidates, target_predict): int64
    # - Tree structure tensors (retrive_*): int64

    # predicts: flattened predictions for all batches (int32 for SYCL)
    predicts = torch.full(
        (batch_size * num_draft_tokens,), -1, device=device, dtype=torch.int32
    )

    # accept_index: tracks which draft tokens were accepted (int32 for SYCL)
    accept_index = torch.full(
        (batch_size, num_speculative_tokens), -1, device=device, dtype=torch.int32
    )

    # accept_token_num: number of accepted tokens per batch (int32 for SYCL)
    accept_token_num = torch.zeros((batch_size,), device=device, dtype=torch.int32)

    # candidates: draft token IDs (int64 for SYCL)
    candidates = torch.randint(
        0, 32000, (batch_size, num_draft_tokens), device=device, dtype=torch.int64
    )

    # retrive_index: indices in the tree structure (int64 for SYCL)
    retrive_index = (
        torch.arange(num_draft_tokens, device=device, dtype=torch.int64)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .contiguous()
    )

    # retrive_next_token: next token in traversal (int64 for SYCL)
    # Use a simple linear chain structure to avoid potential infinite loops
    retrive_next_token = torch.full(
        (batch_size, num_draft_tokens), -1, device=device, dtype=torch.int64
    )
    # Create a simple chain: 0 -> 1 -> 2 -> ... -> n-1 -> -1
    for i in range(num_draft_tokens - 1):
        retrive_next_token[:, i] = i + 1

    # retrive_next_sibling: sibling node in tree (int64 for SYCL)
    # Set all to -1 (no siblings in a simple chain)
    retrive_next_sibling = torch.full(
        (batch_size, num_draft_tokens), -1, device=device, dtype=torch.int64
    )

    # target_predict: target model predictions (int64 for SYCL)
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


def _run_sycl(outputs, inputs):
    """Call the SYCL kernel with pre-cloned output buffers."""
    predicts, accept_index, accept_token_num = outputs
    (
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        target_predict,
    ) = inputs
    verify_tree_greedy_sycl(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        target_predict=target_predict,
    )


def _run_triton(outputs, inputs):
    """Call the Triton kernel with pre-cloned output buffers (it spells the tree tensors retrieve_*)."""
    predicts, accept_index, accept_token_num = outputs
    (
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        target_predict,
    ) = inputs
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


def get_implementations() -> List[Tuple[str, Callable]]:
    """Return (name, runner) pairs to compare."""
    return [("SYCL", _run_sycl), ("Triton", _run_triton)]


def benchmark_impl(
    name: str,
    runner: Callable,
    inputs: Tuple[torch.Tensor, ...],
    warmup: int = 10,
    iterations: int = 100,
) -> PerfMetrics:
    """Benchmark one implementation. Same timed sequence as the per-kernel loops:
    clone the three output buffers, start the timer, call, synchronize, stop."""
    metrics = PerfMetrics(name)

    templates = inputs[:3]
    kernel_inputs = inputs[3:]

    def fresh_outputs():
        return tuple(t.clone() for t in templates)

    # Warmup
    for _ in range(warmup):
        runner(fresh_outputs(), kernel_inputs)
    torch.xpu.synchronize()

    # Actual benchmarking
    for _ in range(iterations):
        # Clone outputs for each iteration
        outputs = fresh_outputs()

        start = time.perf_counter()
        runner(outputs, kernel_inputs)
        torch.xpu.synchronize()
        end = time.perf_counter()

        metrics.add(end - start)

    return metrics


def benchmark_triton_kernel(
    inputs: Tuple[torch.Tensor, ...],
    warmup: int = 10,
    iterations: int = 100,
) -> PerfMetrics:
    """Benchmark Triton kernel."""
    metrics = PerfMetrics("Triton")

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

    # Warmup
    for _ in range(warmup):
        predicts = predicts_template.clone()
        accept_index = accept_index_template.clone()
        accept_token_num = accept_token_num_template.clone()

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

    # Actual benchmarking
    for _ in range(iterations):
        # Clone outputs for each iteration
        predicts = predicts_template.clone()
        accept_index = accept_index_template.clone()
        accept_token_num = accept_token_num_template.clone()

        start = time.perf_counter()
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
        end = time.perf_counter()

        metrics.add(end - start)

    return metrics


def benchmark_sycl_kernel(
    inputs: Tuple[torch.Tensor, ...],
    warmup: int = 10,
    iterations: int = 100,
) -> PerfMetrics:
    """Benchmark SYCL kernel from sgl_kernel."""
    metrics = PerfMetrics("SYCL")

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

    # Warmup
    for _ in range(warmup):
        predicts = predicts_template.clone()
        accept_index = accept_index_template.clone()
        accept_token_num = accept_token_num_template.clone()

        verify_tree_greedy_sycl(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            target_predict=target_predict,
        )
    torch.xpu.synchronize()

    # Actual benchmarking
    for _ in range(iterations):
        # Clone outputs for each iteration
        predicts = predicts_template.clone()
        accept_index = accept_index_template.clone()
        accept_token_num = accept_token_num_template.clone()

        start = time.perf_counter()
        verify_tree_greedy_sycl(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            target_predict=target_predict,
        )
        torch.xpu.synchronize()
        end = time.perf_counter()

        metrics.add(end - start)

    return metrics


def verify_correctness(
    inputs: Tuple[torch.Tensor, ...],
) -> bool:
    """Verify that Triton and SYCL implementations produce the same results."""
    if not SYCL_KERNEL_AVAILABLE:
        print("SYCL kernel not available, skipping correctness check")
        return False

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

    # All implementations use int32 dtype
    # SYCL outputs
    predicts_sycl = predicts_template.clone()
    accept_index_sycl = accept_index_template.clone()
    accept_token_num_sycl = accept_token_num_template.clone()

    # Triton outputs
    predicts_triton = predicts_template.clone()
    accept_index_triton = accept_index_template.clone()
    accept_token_num_triton = accept_token_num_template.clone()

    # Run SYCL implementation
    try:
        verify_tree_greedy_sycl(
            predicts=predicts_sycl,
            accept_index=accept_index_sycl,
            accept_token_num=accept_token_num_sycl,
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            target_predict=target_predict,
        )
        torch.xpu.synchronize()
    except Exception as e:
        print(f"   SYCL implementation failed: {e}", flush=True)
        import traceback

        traceback.print_exc()
        return False

    # Run Triton implementation
    try:
        verify_tree_greedy_triton(
            predicts=predicts_triton,
            accept_index=accept_index_triton,
            accept_token_num=accept_token_num_triton,
            candidates=candidates,
            retrieve_index=retrive_index,
            retrieve_next_token=retrive_next_token,
            retrieve_next_sibling=retrive_next_sibling,
            target_predict=target_predict,
        )
        torch.xpu.synchronize()
    except (AttributeError, RuntimeError) as e:
        print(f"   Triton implementation failed: {e}", flush=True)
        import traceback

        traceback.print_exc()
        return False

    # Compare outputs
    checks = [
        ("predicts", torch.equal(predicts_sycl, predicts_triton)),
        ("accept_index", torch.equal(accept_index_sycl, accept_index_triton)),
        (
            "accept_token_num",
            torch.equal(accept_token_num_sycl, accept_token_num_triton),
        ),
    ]

    all_correct = all(check[1] for check in checks)

    if not all_correct:
        print("Correctness check FAILED:")
        for name, result in checks:
            if not result:
                print(f"  ✗ {name} mismatch")
                sycl_val = locals()[f"{name}_sycl"]
                triton_val = locals()[f"{name}_triton"]
                if name == "accept_token_num":
                    print(f"    SYCL:   {sycl_val}")
                    print(f"    Triton: {triton_val}")
                else:
                    diff = (sycl_val != triton_val).sum()
                    print(f"    Diff count: {diff} / {sycl_val.numel()}")
                    if diff < 20:  # Show details for small diffs
                        print(f"    SYCL:   {sycl_val.flatten()[:20]}")
                        print(f"    Triton: {triton_val.flatten()[:20]}")
        return False

    return True


def print_comparison_table(results: List[dict]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 110)
    print("PERFORMANCE COMPARISON: Triton vs SYCL (sgl_kernel) - Verify Tree Greedy")
    print("=" * 110)

    # Group by configuration
    configs = {}
    for r in results:
        key = (r["batch_size"], r["num_draft_tokens"])
        if key not in configs:
            configs[key] = {"sycl": None, "triton": None}
        if "sycl" in r["name"].lower():
            configs[key]["sycl"] = r
        else:
            configs[key]["triton"] = r

    for (bs, num_tokens), impls in sorted(configs.items()):
        print(f"\nConfig: batch_size={bs}, num_draft_tokens={num_tokens}")
        print("-" * 110)
        print(
            f"{'Implementation':<15} {'Mean (ms)':<12} {'Median (ms)':<12} {'P95 (ms)':<12} {'P99 (ms)':<12} {'Speedup':<10}"
        )
        print("-" * 110)

        sycl_metrics = impls["sycl"]
        triton_metrics = impls["triton"]

        if sycl_metrics and triton_metrics:
            speedup = sycl_metrics["mean_ms"] / triton_metrics["mean_ms"]

            print(
                f"{'SYCL':<15} {sycl_metrics['mean_ms']:>11.4f} {sycl_metrics['median_ms']:>11.4f} "
                f"{sycl_metrics['p95_ms']:>11.4f} {sycl_metrics['p99_ms']:>11.4f} {'1.00x':<10}"
            )
            print(
                f"{'Triton':<15} {triton_metrics['mean_ms']:>11.4f} {triton_metrics['median_ms']:>11.4f} "
                f"{triton_metrics['p95_ms']:>11.4f} {triton_metrics['p99_ms']:>11.4f} {f'{speedup:.2f}x':<10}"
            )

            if speedup > 1.05:
                print(f"  → Triton is {speedup:.2f}x FASTER than SYCL")
            elif speedup < 0.95:
                print(f"  → SYCL is {1/speedup:.2f}x FASTER than Triton")
            else:
                print(f"  → Performance is equivalent (within 5%)")

    print("=" * 110)


def print_markdown_table(results: List[dict]):
    """Print a markdown table (one row per config) suitable for pasting into a PR description."""
    # Group by configuration (same keying as print_comparison_table).
    configs = {}
    for r in results:
        key = (r["batch_size"], r["num_draft_tokens"])
        if key not in configs:
            configs[key] = {"sycl": None, "triton": None}
        if "sycl" in r["name"].lower():
            configs[key]["sycl"] = r
        else:
            configs[key]["triton"] = r

    header = ["Config", "SYCL (wall mean, ms)", "Triton (wall mean, ms)", "Result"]
    print("\n| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")

    for (bs, num_tokens), impls in sorted(configs.items()):
        sycl_metrics = impls["sycl"]
        triton_metrics = impls["triton"]
        if not (sycl_metrics and triton_metrics):
            continue

        # Same ratio as print_comparison_table's speedup.
        speedup = sycl_metrics["mean_ms"] / triton_metrics["mean_ms"]
        if speedup > 1.05:
            result = f"**Triton {speedup:.2f}x faster**"
        elif speedup < 0.95:
            result = f"**SYCL {1/speedup:.2f}x faster**"
        else:
            result = "equivalent (within 5%)"

        row = [
            f"bs={bs}, num_draft_tokens={num_tokens}",
            f"{sycl_metrics['mean_ms']:.4f}",
            f"{triton_metrics['mean_ms']:.4f}",
            result,
        ]
        print("| " + " | ".join(row) + " |")


@pytest.mark.parametrize(
    "batch_size,num_draft_tokens",
    [
        (1, 21),  # Small: topk=4, depth=3
        (4, 21),  # Medium batch
        (8, 21),  # Large batch
        (16, 21),  # Very large batch
        (1, 73),  # High topk: topk=8, depth=3
        (1, 85),  # Deep: topk=4, depth=4
        (32, 21),  # Huge batch
        (1, 341),  # Very deep: topk=4, depth=5
    ],
)
def test_verify_kernel_performance(batch_size: int, num_draft_tokens: int):
    """Test and compare performance of verify tree greedy kernels."""
    if not torch.xpu.is_available():
        pytest.skip("XPU not available")

    if not SYCL_KERNEL_AVAILABLE:
        pytest.skip("sgl_kernel SYCL implementation not available")

    device = "xpu"
    num_speculative_tokens = num_draft_tokens

    print(f"\n{'='*80}")
    print(f"Testing: batch_size={batch_size}, num_draft_tokens={num_draft_tokens}")
    print(f"{'='*80}")

    # Generate inputs
    inputs = generate_test_inputs(
        batch_size, num_draft_tokens, num_speculative_tokens, device
    )

    # Verify correctness first
    print("\n1. Verifying correctness...")
    correct = verify_correctness(inputs)
    if correct:
        print("   ✓ Correctness check PASSED - Triton matches SYCL output")
    else:
        pytest.fail("Correctness check failed! Triton output differs from SYCL")

    # Benchmark SYCL implementation
    print("\n2. Benchmarking SYCL (sgl_kernel) implementation...")
    sycl_metrics = benchmark_sycl_kernel(
        inputs,
        warmup=10,
        iterations=100,
    )
    sycl_summary = sycl_metrics.summary()
    print(f"   Mean:   {sycl_summary['mean_ms']:.4f} ms")
    print(f"   Median: {sycl_summary['median_ms']:.4f} ms")
    print(f"   P95:    {sycl_summary['p95_ms']:.4f} ms")

    # Benchmark Triton implementation
    print("\n3. Benchmarking Triton implementation...")
    try:
        triton_metrics = benchmark_triton_kernel(
            inputs,
            warmup=10,
            iterations=100,
        )
        triton_summary = triton_metrics.summary()
        print(f"   Mean:   {triton_summary['mean_ms']:.4f} ms")
        print(f"   Median: {triton_summary['median_ms']:.4f} ms")
        print(f"   P95:    {triton_summary['p95_ms']:.4f} ms")

        # Compute speedup
        speedup = sycl_summary["mean_ms"] / triton_summary["mean_ms"]
        print(f"\n4. Performance Comparison:")
        print(f"   Speedup: {speedup:.2f}x")
        if speedup > 1.05:
            print(f"   → Triton is {speedup:.2f}x FASTER than SYCL")
        elif speedup < 0.95:
            print(f"   → SYCL is {1/speedup:.2f}x FASTER than Triton")
        else:
            print(f"   → Performance is roughly equivalent (within 5%)")

    except (AttributeError, RuntimeError) as e:
        pytest.fail(f"Triton implementation failed: {e}")


def test_comprehensive_performance_suite():
    """Run a comprehensive performance test suite and generate report."""
    if not torch.xpu.is_available():
        pytest.skip("XPU not available")

    if not SYCL_KERNEL_AVAILABLE:
        pytest.skip("sgl_kernel SYCL implementation not available")

    device = "xpu"
    results = []

    configs = [
        # (batch_size, num_draft_tokens)
        (1, 21),  # Baseline: topk=4, depth=3
        (4, 21),  # Batch scaling
        (8, 21),
        (16, 21),
        (32, 21),
        (64, 21),  # Very large batch
        (1, 7),  # Small tree: topk=2, depth=3
        (1, 73),  # Large tree: topk=8, depth=3
        (1, 85),  # Deep tree: topk=4, depth=4
        (1, 341),  # Very deep: topk=4, depth=5
        (16, 73),  # Large batch + large tree
    ]

    print("\n" + "=" * 110)
    print("COMPREHENSIVE PERFORMANCE SUITE - Verify Tree Greedy")
    print("Comparing Triton vs SYCL (sgl_kernel) implementations")
    print("=" * 110)

    for batch_size, num_draft_tokens in configs:
        num_speculative_tokens = num_draft_tokens
        inputs = generate_test_inputs(
            batch_size, num_draft_tokens, num_speculative_tokens, device
        )

        # Verify correctness
        if not verify_correctness(inputs):
            print(
                f"✗ FAILED correctness: batch_size={batch_size}, num_draft_tokens={num_draft_tokens}"
            )
            continue

        # SYCL
        sycl_metrics = benchmark_sycl_kernel(
            inputs,
            warmup=10,
            iterations=100,
        )
        sycl_summary = sycl_metrics.summary()
        sycl_summary.update(
            {"batch_size": batch_size, "num_draft_tokens": num_draft_tokens}
        )
        results.append(sycl_summary)

        # Triton
        try:
            triton_metrics = benchmark_triton_kernel(
                inputs,
                warmup=10,
                iterations=100,
            )
            triton_summary = triton_metrics.summary()
            triton_summary.update(
                {"batch_size": batch_size, "num_draft_tokens": num_draft_tokens}
            )
            results.append(triton_summary)
        except (AttributeError, RuntimeError) as e:
            print(f"✗ Triton failed: {e}")

        print(
            f"✓ Completed: batch_size={batch_size}, num_draft_tokens={num_draft_tokens}"
        )

    print_comparison_table(results)
    print_markdown_table(results)


if __name__ == "__main__":
    # Run comprehensive suite when executed directly
    test_comprehensive_performance_suite()
