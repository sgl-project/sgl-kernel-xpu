"""Benchmark: sparse MLA prefill (sgl_kernel).

Times sgl_kernel.flash_mla_sparse_prefill across prefill shapes and
reports average latency + effective memory bandwidth.

Usage:
  python benchmark/bench_flash_mla_sparse_prefill.py
"""

import torch
from sgl_kernel import flash_mla_sparse_prefill

# ── constants ──
D_QK = 576  # qk_nope(512) + qk_rope(64)
D_V = 512
H_KV = 1
S_KV = 16384

# (label, s_q, h_q, topk)
CONFIGS = [
    ("TP=8  (16 heads) topk=2048", 512, 16, 2048),
    ("TP=4  (32 heads) topk=2048", 512, 32, 2048),
    ("TP=1 (128 heads) topk=2048", 512, 128, 2048),
    ("TP=8  (16 heads) topk=512", 2048, 16, 512),
    ("TP=1 (128 heads) topk=512", 2048, 128, 512),
]


def build_inputs(s_q, h_q, topk, device="xpu", dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    q = torch.randn((s_q, h_q, D_QK), device=device, dtype=dtype)
    kv = torch.randn((S_KV, H_KV, D_QK), device=device, dtype=dtype)
    indices = torch.full((s_q, H_KV, topk), S_KV, dtype=torch.int32, device=device)
    for t in range(s_q):
        n = min(topk, max(1, S_KV))
        i_i = torch.randperm(S_KV, device=device)[:n].to(torch.int32)
        indices[t, 0, : len(i_i)] = i_i
    return q, kv, indices


def effective_bytes(s_q, h_q, topk):
    # q read + gathered kv read + out write (bf16 = 2 bytes; fp32 lse/max negligible)
    q_bytes = s_q * h_q * D_QK * 2
    kv_bytes = s_q * topk * D_QK * 2
    out_bytes = s_q * h_q * D_V * 2
    return q_bytes + kv_bytes + out_bytes


def bench(fn, warmup=10, iters=30):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.xpu.synchronize()
    return start.elapsed_time(end) / iters  # ms


def main():
    if not torch.xpu.is_available():
        print("XPU not available")
        return

    hdr = f"{'config':30s} {'ms':>9s} {'GB/s':>9s}"
    print(hdr)
    print("-" * len(hdr))

    for label, s_q, h_q, topk in CONFIGS:
        q, kv, indices = build_inputs(s_q, h_q, topk)
        sm_scale = D_QK**-0.5

        def run_sgl():
            return flash_mla_sparse_prefill(
                q, kv, indices, sm_scale=sm_scale, d_v=D_V, return_softmax_lse=False
            )

        run_sgl()  # warmup / build
        avg_ms = bench(run_sgl)
        gbs = effective_bytes(s_q, h_q, topk) / (avg_ms * 1e-3) / 1e9

        print(f"{label:30s} {avg_ms:9.3f} {gbs:9.2f}")


if __name__ == "__main__":
    main()
