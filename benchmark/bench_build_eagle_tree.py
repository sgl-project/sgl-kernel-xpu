"""
Performance comparison test for EAGLE tree building kernels.

Compares the native kernel (sgl_kernel: CUDA/HIP/MUSA, or the CPU op) against
the Triton implementation across different configurations. The set of
implementations exercised is chosen from whatever ships for the current device:

    - CUDA / HIP / MUSA: sgl_kernel native op + Triton
    - Intel XPU:          Triton only
    - CPU:                sgl_kernel CPU op only

The pure-PyTorch reference (sgl_build_tree_kernel_efficient_pytorch) was removed
upstream, so it is no longer part of the comparison.

Usage:
    python -m pytest test/manual/spec/eagle/test_tree_kernel_perf_v2.py -v -s

    # Run comprehensive suite directly:
    python test/manual/spec/eagle/test_tree_kernel_perf_v2.py
"""

import time
from typing import Callable, List, Tuple

import pytest
import torch
from sgl_kernel import TreeMaskMode
from sgl_kernel.eagle_utils import sgl_build_tree_kernel_triton


# sglang is not a dependency of sgl-kernel-xpu, so its device probes are inlined
# here (same semantics) to keep this benchmark runnable standalone.
def is_xpu() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def is_cuda() -> bool:
    return (
        getattr(torch.version, "cuda", None) is not None and torch.cuda.is_available()
    )


def is_hip() -> bool:
    return getattr(torch.version, "hip", None) is not None


def is_musa() -> bool:
    return hasattr(torch, "musa") and torch.musa.is_available()


def is_cpu() -> bool:
    return not (is_cuda() or is_hip() or is_musa() or is_xpu())


# The native tree-build op is registered per device family. Mirror the same
# conditional import eagle_utils uses so this test picks up whatever shipped.
_native_build_tree = None
_NATIVE_NAME = None
try:
    if is_cuda() or is_hip() or is_musa():
        from sgl_kernel import build_tree_kernel_efficient as _native_build_tree

        _NATIVE_NAME = "sgl_kernel"
    elif is_xpu():
        # AOT SYCL op from this repo (src/sycl/SpecBuildTree.cpp); same in-place
        # signature as the CUDA op, so _run_native drives it unchanged.
        from sgl_kernel import build_tree_kernel_efficient as _native_build_tree

        _NATIVE_NAME = "sgl_kernel_sycl"
    elif is_cpu():
        from sgl_kernel import build_tree_kernel_efficient_cpu as _native_build_tree

        _NATIVE_NAME = "sgl_kernel_cpu"
except ImportError:
    _native_build_tree = None


def _detect_device() -> str:
    if is_xpu() and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = _detect_device()

# Triton ships for the GPU backends; there is no Triton path for the CPU op.
_TRITON_AVAILABLE = DEVICE in ("cuda", "xpu")


def _sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "xpu":
        torch.xpu.synchronize()


# Device timing events isolate on-device kernel time (excludes host/launch
# overhead). Only the GPU backends expose timing events; the CPU op has none.
_DEVICE_EVENTS = {"cuda": torch.cuda, "xpu": torch.xpu}.get(DEVICE)


def _make_timing_events():
    """Return a (start, end) event pair for device timing, or None if unsupported."""
    if _DEVICE_EVENTS is None:
        return None
    return (
        _DEVICE_EVENTS.Event(enable_timing=True),
        _DEVICE_EVENTS.Event(enable_timing=True),
    )


def _run_native(inputs, topk, depth, draft_token_num, tree_mask_mode):
    (
        parent_list,
        selected_index,
        verified_seq_len,
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
    ) = inputs
    _native_build_tree(
        parent_list,
        selected_index,
        verified_seq_len,
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        topk,
        depth,
        draft_token_num,
        int(tree_mask_mode),  # native op expects an int mask mode
    )


def _run_triton(inputs, topk, depth, draft_token_num, tree_mask_mode):
    (
        parent_list,
        selected_index,
        verified_seq_len,
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
    ) = inputs
    sgl_build_tree_kernel_triton(
        parent_list,
        selected_index,
        verified_seq_len,
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


def get_implementations() -> List[Tuple[str, Callable]]:
    """Return (name, runner) pairs available on the current device."""
    impls: List[Tuple[str, Callable]] = []
    if _native_build_tree is not None:
        impls.append((_NATIVE_NAME, _run_native))
    if _TRITON_AVAILABLE:
        impls.append(("triton", _run_triton))
    return impls


def _latency_stats(latencies_s: List[float]) -> dict:
    """Compute mean/median/min/max/p95/p99 in ms from a list of second-latencies."""
    latencies_ms = sorted(l * 1000 for l in latencies_s)
    n = len(latencies_ms)
    return {
        "mean_ms": sum(latencies_ms) / n,
        "median_ms": latencies_ms[n // 2],
        "min_ms": latencies_ms[0],
        "max_ms": latencies_ms[-1],
        "p95_ms": latencies_ms[min(int(n * 0.95), n - 1)],
        "p99_ms": latencies_ms[min(int(n * 0.99), n - 1)],
    }


class PerfMetrics:
    """Store and compute performance metrics (wall-clock and device time)."""

    def __init__(self, name: str):
        self.name = name
        self.wall_latencies = []
        self.device_latencies = []

    def add(self, wall_s: float, device_s: float = None):
        self.wall_latencies.append(wall_s)
        if device_s is not None:
            self.device_latencies.append(device_s)

    def summary(self) -> dict:
        if not self.wall_latencies:
            return {"name": self.name, "count": 0}

        # Wall-clock stats stay at the top level (mean_ms, p95_ms, ...); device
        # stats are mirrored under dev_* keys when timing events were available.
        result = {"name": self.name, "count": len(self.wall_latencies)}
        result.update(_latency_stats(self.wall_latencies))
        if self.device_latencies:
            for k, v in _latency_stats(self.device_latencies).items():
                result[f"dev_{k}"] = v
        return result


def generate_test_inputs(
    batch_size: int,
    topk: int,
    depth: int,
    verified_seq_len: List[int],
    device: str = DEVICE,
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


def benchmark_impl(
    name: str,
    runner: Callable,
    inputs: Tuple[torch.Tensor, ...],
    topk: int,
    depth: int,
    draft_token_num: int,
    tree_mask_mode: TreeMaskMode,
    warmup: int = 10,
    iterations: int = 100,
) -> PerfMetrics:
    """Benchmark a single tree-build implementation."""
    metrics = PerfMetrics(name)

    (
        _,
        _,
        _,
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
    ) = inputs

    def reset_outputs():
        tree_mask.fill_(False)
        positions.fill_(0)
        retrive_index.fill_(0)
        retrive_next_token.fill_(0)
        retrive_next_sibling.fill_(0)

    for _ in range(warmup):
        runner(inputs, topk, depth, draft_token_num, tree_mask_mode)
    _sync()

    events = _make_timing_events()
    for _ in range(iterations):
        reset_outputs()
        start = time.perf_counter()
        if events is not None:
            start_ev, end_ev = events
            start_ev.record()
            runner(inputs, topk, depth, draft_token_num, tree_mask_mode)
            end_ev.record()
            _sync()
            end = time.perf_counter()
            # elapsed_time() is in ms; store seconds to match wall-clock samples.
            metrics.add(end - start, start_ev.elapsed_time(end_ev) / 1000.0)
        else:
            runner(inputs, topk, depth, draft_token_num, tree_mask_mode)
            _sync()
            end = time.perf_counter()
            metrics.add(end - start)

    return metrics


def _fresh_output_buffers(inputs, draft_token_num):
    """Allocate a private set of output buffers matching `inputs`' shapes/device."""
    parent_list, selected_index, verified_seq_len = inputs[0], inputs[1], inputs[2]
    device = parent_list.device
    batch_size = verified_seq_len.shape[0]
    seq_lens_sum = int(verified_seq_len.sum().item())
    total_mask_size = (
        seq_lens_sum * draft_token_num + draft_token_num * batch_size * draft_token_num
    )

    tree_mask = torch.full((total_mask_size,), True, device=device, dtype=torch.bool)
    positions = torch.empty(
        batch_size * draft_token_num, device=device, dtype=torch.long
    )
    retrieve_buf = torch.full(
        (3, batch_size, draft_token_num), -1, device=device, dtype=torch.long
    )
    retrive_index, retrive_next_token, retrive_next_sibling = retrieve_buf
    return (
        parent_list,
        selected_index,
        verified_seq_len,
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
    )


def verify_correctness(
    inputs: Tuple[torch.Tensor, ...],
    topk: int,
    depth: int,
    draft_token_num: int,
    tree_mask_mode: TreeMaskMode,
) -> bool:
    """Verify that all available implementations produce identical results.

    The first available implementation is used as the reference; the rest are
    compared against it. With a single implementation there is nothing to
    cross-check, so it is treated as trivially correct.
    """
    impls = get_implementations()
    if len(impls) < 2:
        print("   Only one implementation available; skipping cross-check")
        return True

    outputs = {}
    for name, runner in impls:
        buf = _fresh_output_buffers(inputs, draft_token_num)
        try:
            runner(buf, topk, depth, draft_token_num, tree_mask_mode)
            _sync()
        except NotImplementedError as e:
            print(f"   {name} not available for this mode: {e}")
            continue
        # buf = (parent_list, selected_index, verified_seq_len, tree_mask,
        #        positions, retrive_index, retrive_next_token, retrive_next_sibling)
        outputs[name] = {
            "tree_mask": buf[3],
            "positions": buf[4],
            "retrive_index": buf[5],
            "retrive_next_token": buf[6],
            "retrive_next_sibling": buf[7],
        }

    if len(outputs) < 2:
        print("   Fewer than two implementations ran; skipping cross-check")
        return True

    ref_name = impls[0][0]
    ref = outputs[ref_name]
    all_correct = True
    for name, out in outputs.items():
        if name == ref_name:
            continue
        for field in (
            "positions",
            "retrive_index",
            "retrive_next_token",
            "retrive_next_sibling",
            "tree_mask",
        ):
            if not torch.equal(ref[field], out[field]):
                print(f"   {ref_name} vs {name}: {field} MISMATCH")
                if field == "tree_mask":
                    diff = (ref[field] != out[field]).sum()
                    print(f"     Diff count: {diff} / {ref[field].numel()}")
                else:
                    print(f"     {ref_name}: {ref[field].flatten()[:10]}...")
                    print(f"     {name}: {out[field].flatten()[:10]}...")
                all_correct = False

    return all_correct


def print_comparison_table(results: List[dict]):
    """Print a formatted comparison table keyed by (batch_size, topk, depth)."""
    print("\n" + "=" * 110)
    print(f"PERFORMANCE COMPARISON (device={DEVICE})")
    print("=" * 110)

    configs = {}
    for r in results:
        key = (r["batch_size"], r["topk"], r["depth"])
        configs.setdefault(key, {})[r["name"]] = r

    for (bs, topk, depth), impls in sorted(configs.items()):
        draft_tokens = next(iter(impls.values()))["draft_tokens"]
        print(
            f"\nConfig: batch_size={bs}, topk={topk}, depth={depth}, "
            f"draft_tokens={draft_tokens}"
        )
        print("-" * 110)
        print(
            f"{'Implementation':<18} {'wall mean':<12} {'wall med':<12} "
            f"{'wall p95':<12} {'dev mean':<12} {'dev med':<12} {'dev p95':<12}  (ms)"
        )
        print("-" * 110)

        for name, m in sorted(impls.items()):
            dev = (
                f"{m['dev_mean_ms']:>11.4f} {m['dev_median_ms']:>11.4f} "
                f"{m['dev_p95_ms']:>11.4f}"
                if "dev_mean_ms" in m
                else f"{'n/a':>11} {'n/a':>11} {'n/a':>11}"
            )
            print(
                f"{name:<18} {m['mean_ms']:>11.4f} {m['median_ms']:>11.4f} "
                f"{m['p95_ms']:>11.4f} {dev}"
            )

        # Pairwise speedup summary when at least two impls ran.
        names = sorted(impls.keys())
        if len(names) >= 2:
            base = names[0]
            for other in names[1:]:
                speedup = impls[base]["mean_ms"] / impls[other]["mean_ms"]
                if speedup > 1.05:
                    print(f"  -> {other} is {speedup:.2f}x FASTER than {base}")
                elif speedup < 0.95:
                    print(f"  -> {base} is {1 / speedup:.2f}x FASTER than {other}")
                else:
                    print(f"  -> {other} vs {base} equivalent (within 5%)")

    print("=" * 110)


@pytest.mark.parametrize(
    "batch_size,topk,depth",
    [
        (1, 4, 3),  # Small: single batch, 4 topk, depth 3 -> 21 draft tokens
        (4, 4, 3),  # Medium: 4 batches
        (8, 4, 3),  # Large: 8 batches
        (1, 8, 3),  # High topk: 8 topk -> 73 draft tokens
        (1, 4, 4),  # Deep: depth 4 -> 85 draft tokens
        (16, 4, 3),  # Very large batch
    ],
)
@pytest.mark.parametrize(
    "tree_mask_mode", [TreeMaskMode.FULL_MASK, TreeMaskMode.QLEN_ONLY]
)
def test_tree_kernel_performance(
    batch_size: int, topk: int, depth: int, tree_mask_mode: TreeMaskMode
):
    """Test and compare performance of the available tree building kernels."""
    impls = get_implementations()
    if not impls:
        pytest.skip(f"No tree-build implementation available on device={DEVICE}")

    draft_token_num = sum(topk**i for i in range(depth))
    verified_seq_len = [10] * batch_size  # Each request has 10 tokens verified

    print(f"\n{'='*80}")
    print(
        f"Testing (device={DEVICE}): batch_size={batch_size}, topk={topk}, "
        f"depth={depth}, draft_tokens={draft_token_num}"
    )
    print(f"Tree mask mode: {tree_mask_mode}")
    print(f"Implementations: {[n for n, _ in impls]}")
    print(f"{'='*80}")

    inputs = generate_test_inputs(batch_size, topk, depth, verified_seq_len, DEVICE)

    print("\n1. Verifying correctness...")
    correct = verify_correctness(inputs, topk, depth, draft_token_num, tree_mask_mode)
    if correct:
        print("   Correctness check PASSED")
    else:
        pytest.fail("Correctness check failed! Output differs between implementations")

    print("\n2. Benchmarking...")
    for name, runner in impls:
        try:
            metrics = benchmark_impl(
                name,
                runner,
                inputs,
                topk,
                depth,
                draft_token_num,
                tree_mask_mode,
                warmup=10,
                iterations=100,
            )
        except NotImplementedError as e:
            print(f"   {name} not available for this mode: {e}")
            continue
        summary = metrics.summary()
        print(
            f"   {name:<18} wall: mean={summary['mean_ms']:.4f} ms  "
            f"median={summary['median_ms']:.4f} ms  p95={summary['p95_ms']:.4f} ms"
        )
        if "dev_mean_ms" in summary:
            print(
                f"   {'':<18} dev:  mean={summary['dev_mean_ms']:.4f} ms  "
                f"median={summary['dev_median_ms']:.4f} ms  "
                f"p95={summary['dev_p95_ms']:.4f} ms"
            )


def test_comprehensive_performance_suite():
    """Run a comprehensive performance test suite and generate a report."""
    impls = get_implementations()
    if not impls:
        pytest.skip(f"No tree-build implementation available on device={DEVICE}")

    tree_mask_mode = TreeMaskMode.FULL_MASK
    results = []

    configs = [
        (1, 4, 3),  # Baseline
        (4, 4, 3),  # Batch scaling
        (8, 4, 3),
        (16, 4, 3),
        (32, 4, 3),  # Large batch
        (1, 2, 3),  # Topk scaling
        (1, 8, 3),
        (1, 4, 2),  # Depth scaling
        (1, 4, 4),
        (1, 4, 5),  # Very deep
    ]

    print("\n" + "=" * 110)
    print(f"COMPREHENSIVE PERFORMANCE SUITE (device={DEVICE})")
    print(f"Implementations: {[n for n, _ in impls]}")
    print("=" * 110)

    for batch_size, topk, depth in configs:
        draft_token_num = sum(topk**i for i in range(depth))
        verified_seq_len = [10] * batch_size

        inputs = generate_test_inputs(batch_size, topk, depth, verified_seq_len, DEVICE)

        for name, runner in impls:
            try:
                metrics = benchmark_impl(
                    name,
                    runner,
                    inputs,
                    topk,
                    depth,
                    draft_token_num,
                    tree_mask_mode,
                    warmup=10,
                    iterations=100,
                )
            except NotImplementedError:
                continue
            summary = metrics.summary()
            summary.update(
                {
                    "batch_size": batch_size,
                    "topk": topk,
                    "depth": depth,
                    "draft_tokens": draft_token_num,
                }
            )
            results.append(summary)

        print(f"Completed: batch_size={batch_size}, topk={topk}, depth={depth}")

    print_comparison_table(results)


if __name__ == "__main__":
    # Run comprehensive suite when executed directly
    test_comprehensive_performance_suite()
