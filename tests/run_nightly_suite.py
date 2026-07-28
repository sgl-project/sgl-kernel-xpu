"""Nightly runner for sgl-kernel-xpu.

Independent of tests/run_suite.py (which owns per-commit CI). Edit the two
suites below freely to add or drop nightly coverage without touching CI.

Two suites:
  * sgl-kernel-test       -- pytest-style tests under tests/
  * sgl-kernel-benchmark  -- __main__-style benchmark scripts under benchmark/

Both are invoked as:
    python3 run_nightly_suite.py --suite <name>
"""

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.join(REPO_ROOT, "tests")
BENCH_DIR = os.path.join(REPO_ROOT, "benchmark")


@dataclass
class Entry:
    name: str
    estimated_time: float = 60
    tee_log: str = ""


NIGHTLY_TESTS: List[Entry] = [
    Entry("test_awq_dequant.py"),
    Entry("test_topk_sigmoid.py"),
    Entry("test_topk_softmax.py"),
    Entry("test_flash_attention.py"),
    Entry("test_flash_attn_sparse.py"),
    Entry("test_flash_mla_decode.py"),
    Entry("test_flash_mla_prefill.py"),
    Entry("test_flash_mla_with_kvcache.py"),
    Entry("test_moe_align.py"),
    Entry("test_moe_gemm.py"),
    Entry("test_moe_sum_reduce.py"),
    Entry("test_moe_prepare_input.py"),
    Entry("test_swiglu_with_alpha_limit.py"),
    Entry("test_per_token_group_quant_8bit.py"),
    Entry("test_per_token_group_quant_mxfp4.py"),
    Entry("test_moe_fused_gate.py"),
    Entry("test_mrope.py"),
    Entry("test_per_tensor_quant_fp8.py"),
    Entry("test_per_token_quant_fp8.py"),
    Entry("test_fused_qk_norm_rope.py"),
    Entry("test_fused_qk_rope_with_cache.py"),
    Entry("test_merge_state.py"),
    Entry("test_merge_state_v2.py"),
    Entry("test_norm.py"),
    Entry("test_per_token_group_quant_8bit_v2.py"),
    Entry("test_activation.py"),
    Entry("test_scatter_tokens_to_experts.py"),
    Entry("test_sampling.py"),
    Entry("test_hc_split_sinkhorn.py"),
    Entry("test_hc_pre_fuse.py"),
    Entry("test_fused_experts_mxfp4_dsv4_shapes.py"),
    Entry("test_store_cache_xpu.py"),
    Entry("test_hc_pre_gemm_sqr_sum.py"),
    Entry("test_mhc_pre.py"),
    Entry("test_per_token_group_quant_mxfp4_fused.py"),
    Entry("test_silu_and_mul_clamp.py"),
    Entry("test_hadamard.py"),
    Entry("test_fp8_paged_mqa_logits.py"),
    Entry("test_c128_v2.py"),
    Entry("test_c4_v2.py"),
    Entry("test_fused_q_indexer_rope_hadamard_quant.py"),
    Entry("test_fused_norm_rope_v2.py"),
    Entry("test_hc_post.py"),
    Entry("test_jit_kernels.py"),
    Entry("test_embedding_lora_a_fwd.py"),
    Entry("test_sgemm_lora_a_fwd.py"),
    Entry("test_fused_q_norm_rope.py"),
]


NIGHTLY_BENCHMARKS: List[Entry] = [
    Entry("bench_flash_attn.py", tee_log="flash.log"),
    Entry("bench_flash_mla_decode.py", tee_log="mla.log"),
    Entry("bench_flash_mla_with_kvcache.py", tee_log="mla_with_kvcache.log"),
    Entry("bench_moe_topk_sigmoid.py", tee_log="moe_topk_sigmoid.log"),
    Entry("bench_moe_topk_softmax.py", tee_log="moe.log"),
    Entry("bench_fused_moe.py", tee_log="fused_moe.log"),
    Entry("bench_moe_sum_reduce.py", tee_log="moe_sum_reduce.log"),
    Entry("bench_moe_fused_gate.py", tee_log="moe_fused_gate.py.log"),
    Entry("bench_merge_states_v2.py", tee_log="merge_states.py.log"),
    Entry("bench_mrope.py", tee_log="mrope.py.log"),
    Entry("bench_swiglu_alpha_limit.py", tee_log="swiglu_alpha_limit.py.log"),
    Entry("bench_fused_qk_norm_rope.py", tee_log="fused_qk_norm_rope.py.log"),
    Entry(
        "bench_fused_qk_rope_with_cache.py",
        tee_log="bench_fused_qk_rope_with_cache.py.log",
    ),
    Entry(
        "bench_per_token_group_quant_mxfp4.py",
        tee_log="per_token_group_quant_mxfp4.py.log",
    ),
    Entry(
        "bench_per_token_group_quant_8bit_v2.py",
        tee_log="bench_per_token_group_quant_8bit_v2.py.log",
    ),
    Entry("bench_per_token_quant_fp8.py", tee_log="bench_per_token_quant_fp8.py.log"),
    Entry(
        "bench_per_token_group_quant_mxfp4_fusion.py",
        tee_log="bench_per_token_group_quant_mxfp4_fusion.py.log",
    ),
    Entry(
        "bench_scatter_tokens_to_experts.py",
        tee_log="bench_scatter_tokens_to_experts.py.log",
    ),
    Entry("bench_top_k_renorm_probs.py", tee_log="bench_top_k_renorm_probs.py.log"),
    Entry("bench_hc_split_sinkhorn.py", tee_log="bench_hc_split_sinkhorn.py.log"),
    Entry("bench_hc_pre_fuse.py", tee_log="bench_hc_pre_fuse.py.log"),
    Entry("bench_embedding_lora_a_fwd.py", tee_log="bench_embedding_lora_a_fwd.py.log"),
    Entry("bench_moe_mxfp4_w4a16_gemm.py", tee_log="bench_moe_mxfp4_w4a16_gemm.py.log"),
    Entry("bench_fused_experts_mxfp4.py", tee_log="bench_fused_experts_mxfp4.py.log"),
    Entry("bench_hc_pre_gemm_sqr_sum.py", tee_log="bench_hc_pre_gemm_sqr_sum.py.log"),
    Entry("bench_mhc_pre.py", tee_log="bench_mhc_pre.py.log"),
    Entry("bench_silu_and_mul_clamp.py", tee_log="bench_silu_and_mul_clamp.log"),
    Entry("bench_jit_rmsnorm.py", tee_log="bench_jit_rmsnorm.py.log"),
    Entry("bench_jit_qknorm.py", tee_log="bench_jit_qknorm.py.log"),
    Entry("bench_jit_rope.py", tee_log="bench_jit_rope.py.log"),
    Entry(
        "bench_jit_timestep_embedding.py", tee_log="bench_jit_timestep_embedding.py.log"
    ),
    Entry(
        "bench_jit_per_tensor_quant_fp8.py",
        tee_log="bench_jit_per_tensor_quant_fp8.py.log",
    ),
    Entry(
        "bench_jit_moe_align_block_size.py",
        tee_log="bench_jit_moe_align_block_size.py.log",
    ),
    Entry(
        "bench_jit_per_token_group_quant_8bit_v2.py",
        tee_log="bench_jit_per_token_group_quant_8bit_v2.py.log",
    ),
    Entry("bench_jit_activation.py", tee_log="bench_jit_activation.py.log"),
    Entry("bench_hc_post.py", tee_log="bench_hc_post.py.log"),
    Entry("bench_sgemm_lora_a_fwd.py", tee_log="bench_sgemm_lora_a_fwd.py.log"),
]


SUITES = {
    "sgl-kernel-test": ("test", TESTS_DIR, NIGHTLY_TESTS),
    "sgl-kernel-benchmark": ("bench", BENCH_DIR, NIGHTLY_BENCHMARKS),
}


def run_tests(files: List[Entry], workdir: str, timeout_per_file: int) -> int:
    # Reuse tests/test_utils.run_unittest_files so nightly test execution matches
    # the harness per-commit CI uses.
    sys.path.insert(0, workdir)
    from run_suite import TestFile  # noqa: E402
    from test_utils import run_unittest_files  # noqa: E402

    return run_unittest_files(
        [TestFile(f.name, f.estimated_time) for f in files],
        timeout_per_file,
    )


def run_benchmarks(files: List[Entry], workdir: str, timeout_per_file: int) -> int:
    failed = []
    for entry in files:
        script = os.path.join(workdir, entry.name)
        log_path = os.path.join(workdir, entry.tee_log or f"{entry.name}.log")
        print(
            f"\n=== Running {entry.name} (log: {entry.tee_log or entry.name + '.log'}) ==="
        )
        try:
            with open(log_path, "wb") as log_f:
                proc = subprocess.run(
                    [sys.executable, script],
                    cwd=workdir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_per_file,
                )
                log_f.write(proc.stdout)
                sys.stdout.buffer.write(proc.stdout)
            if proc.returncode != 0:
                print(f"FAILED: {entry.name} (exit {proc.returncode})")
                failed.append(entry.name)
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT: {entry.name} exceeded {timeout_per_file}s")
            failed.append(entry.name)
        except FileNotFoundError:
            print(f"MISSING: {entry.name} not found under {workdir}")
            failed.append(entry.name)

    if failed:
        print(f"\nBenchmark failures: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        required=True,
        choices=list(SUITES.keys()),
        help="Which nightly suite to run.",
    )
    parser.add_argument(
        "--timeout-per-file",
        type=int,
        default=1800,
        help="Time limit per test/benchmark in seconds.",
    )
    args = parser.parse_args()

    kind, workdir, entries = SUITES[args.suite]
    print(f"Suite: {args.suite} ({kind}); {len(entries)} entries; workdir: {workdir}")

    if kind == "test":
        rc = run_tests(entries, workdir, args.timeout_per_file)
    else:
        rc = run_benchmarks(entries, workdir, args.timeout_per_file)

    sys.exit(rc)
