"""
Test MoE GEMM with typical real-model shapes for prefill and decode.

Run one test at a time with xpu.synchronize() between tests to avoid
overloading the simulator kernel queue.

Shapes derived from:
  - Mixtral 8x7B:   E=8,  topk=2, H=4096, I=14336
  - DeepSeek-V2:    E=64, topk=6, H=5120, I=1536
  - DeepSeek-V3:    E=256,topk=8, H=7168, I=2048
  - Qwen2-MoE-57B: E=64, topk=4, H=2048, I=1408

Dimensions scaled for simulator (H=512, I=256) while preserving structural
properties: expert counts, topk ratios, and alignment requirements.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

from test_moe_gemm import create_random_cpu_tensor, torch_naive_moe
from sgl_kernel import fused_experts, is_xe2_arch


# -- Decode shapes: small token counts (1 token per request in a batch) --
DECODE_SHAPES = [
    # (num_tokens, topk, num_experts, hidden_size, intermediate_size, label)
    # Mixtral-like: E=8, topk=2
    (1, 2, 8, 512, 256, "mixtral decode bs=1"),
    (4, 2, 8, 512, 256, "mixtral decode bs=4"),
    (16, 2, 8, 512, 256, "mixtral decode bs=16"),
    # DeepSeek-V2-like: E=64, topk=6
    (1, 6, 64, 512, 256, "dsv2 decode bs=1"),
    (4, 6, 64, 512, 256, "dsv2 decode bs=4"),
    (8, 6, 64, 512, 256, "dsv2 decode bs=8"),
    # DeepSeek-V3-like: E=64, topk=8
    (1, 8, 64, 512, 256, "dsv3 decode bs=1"),
    (8, 8, 64, 512, 256, "dsv3 decode bs=8"),
    (32, 8, 64, 512, 256, "dsv3 decode bs=32"),
    # Qwen2-MoE-like: E=64, topk=4
    (1, 4, 64, 512, 256, "qwen2-moe decode bs=1"),
    (4, 4, 64, 512, 256, "qwen2-moe decode bs=4"),
]

# -- Prefill shapes: larger token counts (sequence processing) --
PREFILL_SHAPES = [
    # Mixtral-like prefill
    (64, 2, 8, 512, 256, "mixtral prefill seq=64"),
    (128, 2, 8, 512, 256, "mixtral prefill seq=128"),
    (256, 2, 8, 512, 256, "mixtral prefill seq=256"),
    # DeepSeek-V2-like prefill
    (64, 6, 64, 512, 256, "dsv2 prefill seq=64"),
    (128, 6, 64, 512, 256, "dsv2 prefill seq=128"),
    # DeepSeek-V3-like prefill
    (64, 8, 64, 512, 256, "dsv3 prefill seq=64"),
    (128, 8, 64, 512, 256, "dsv3 prefill seq=128"),
    (256, 8, 64, 512, 256, "dsv3 prefill seq=256"),
    # Qwen2-MoE-like prefill
    (64, 4, 64, 512, 256, "qwen2-moe prefill seq=64"),
    (128, 4, 64, 512, 256, "qwen2-moe prefill seq=128"),
]


def run_single_test(num_tokens, topk, num_experts, hidden_size, intermediate_size,
                    act_type="silu", label=""):
    torch.xpu.manual_seed_all(0)

    gating_factor = 1 if act_type == "relu2" else 2
    rtol, atol = 1e-4, 1e-3

    a = create_random_cpu_tensor((num_tokens, hidden_size), torch.bfloat16)
    w1 = create_random_cpu_tensor(
        (num_experts, gating_factor * intermediate_size, hidden_size), torch.bfloat16
    )
    w2 = create_random_cpu_tensor(
        (num_experts, hidden_size, intermediate_size), torch.bfloat16
    )

    score = torch.randn([num_tokens, num_experts], dtype=torch.bfloat16)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)

    torch_output = torch_naive_moe(
        a, w1, w2, topk_ids, topk_weight, topk, None, None, activations=act_type,
    )

    device = "xpu"
    t0 = time.time()
    sglang_output = fused_experts(
        a.to(device), w1.to(device), w2.to(device),
        topk_weight.to(device), topk_ids.to(device),
        None, None, activation=act_type,
    )
    torch.xpu.synchronize()
    elapsed = time.time() - t0

    try:
        torch.testing.assert_close(
            torch_output, sglang_output.to("cpu"), rtol=rtol, atol=atol
        )
        return True, elapsed, None
    except AssertionError as e:
        return False, elapsed, str(e)


def main():
    passed = 0
    failed = 0
    errors = []

    print("=" * 70)
    print("DECODE shapes (small token counts)")
    print("=" * 70)

    for tokens, topk, experts, hidden, inter, label in DECODE_SHAPES:
        print(f"[tokens={tokens:3d} topk={topk} E={experts:2d} H={hidden} I={inter}]")
        ok, elapsed, err = run_single_test(tokens, topk, experts, hidden, inter, label=label)
        if ok:
            print(f"  PASS ({elapsed:.1f}s) - {label}")
            passed += 1
        else:
            print(f"  FAIL ({elapsed:.1f}s) - {label}: {err[:150]}")
            failed += 1
            errors.append((label, err))
        torch.xpu.synchronize()

    print()
    print("=" * 70)
    print("PREFILL shapes (larger token counts)")
    print("=" * 70)

    for tokens, topk, experts, hidden, inter, label in PREFILL_SHAPES:
        print(f"[tokens={tokens:3d} topk={topk} E={experts:2d} H={hidden} I={inter}]")
        ok, elapsed, err = run_single_test(tokens, topk, experts, hidden, inter, label=label)
        if ok:
            print(f"  PASS ({elapsed:.1f}s) - {label}")
            passed += 1
        else:
            print(f"  FAIL ({elapsed:.1f}s) - {label}: {err[:150]}")
            failed += 1
            errors.append((label, err))
        torch.xpu.synchronize()

    print()
    print("=" * 70)
    print(f"SUMMARY: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 70)

    if errors:
        print("\nFailed tests:")
        for label, err in errors:
            print(f"  - {label}: {err[:150]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
