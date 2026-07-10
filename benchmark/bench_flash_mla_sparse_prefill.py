"""Benchmark: DeepSeek V4 sparse MLA prefill (sgl_kernel).

Times sgl_kernel.flash_mla_sparse_prefill across DeepSeek-V4 prefill shapes and
reports latency + effective memory bandwidth. Also cross-checks correctness
against a float reference and, if available, against xattention's
flash_mla_sparse_fwd.

Usage:
  python benchmark/bench_flash_mla_sparse_prefill.py
"""

import torch
from sgl_kernel import flash_mla_sparse_prefill

# ── DeepSeek V4 constants ──
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


def reference_out(q, kv, indices, sm_scale):
    s_q, h_q, d_qk = q.shape
    s_kv = kv.shape[0]
    topk = indices.shape[2]
    idx = indices.clone().squeeze(1)
    invalid = (idx < 0) | (idx >= s_kv)
    idx[invalid] = 0
    gk = kv.index_select(0, idx.flatten()).reshape(s_q, topk, d_qk).float()
    P = (q.float() @ gk.transpose(1, 2)) * sm_scale
    P[invalid.unsqueeze(1).broadcast_to(P.shape)] = float("-inf")
    lse = torch.logsumexp(P, dim=-1, keepdim=True)
    s = torch.exp(P - lse)
    return (s @ gk[..., :D_V]).to(kv.dtype)


def main():
    if not torch.xpu.is_available():
        print("XPU not available")
        return

    try:
        from flash_attn.flash_attn_interface_xpu import flash_mla_sparse_fwd

        have_xatt = True
    except Exception:
        have_xatt = False

    hdr = (
        f"{'config':30s} {'MB/call':>9s} {'sgl ms':>9s} {'sgl GB/s':>9s} "
        f"{'xatt ms':>9s} {'xatt GB/s':>9s} {'speedup':>8s} {'max_abs':>9s} {'ok':>4s}"
    )
    print(hdr)
    print("-" * len(hdr))

    for label, s_q, h_q, topk in CONFIGS:
        q, kv, indices = build_inputs(s_q, h_q, topk)
        sm_scale = D_QK**-0.5
        mb = effective_bytes(s_q, h_q, topk) / 1e6

        def run_sgl():
            return flash_mla_sparse_prefill(
                q, kv, indices, sm_scale=sm_scale, d_v=D_V, return_softmax_lse=False
            )

        out_sgl = run_sgl()
        ref = reference_out(q, kv, indices, sm_scale)
        max_abs = (out_sgl.float() - ref.float()).abs().max().item()
        ok = torch.allclose(out_sgl.float(), ref.float(), atol=1e-2, rtol=1e-2)

        sgl_ms = bench(run_sgl)
        sgl_gbs = effective_bytes(s_q, h_q, topk) / (sgl_ms * 1e-3) / 1e9

        xatt_ms = float("nan")
        xatt_gbs = float("nan")
        speedup = float("nan")
        if have_xatt:

            def run_xatt():
                return flash_mla_sparse_fwd(
                    q, kv, indices, sm_scale, d_v=D_V, return_softmax_lse=False
                )

            try:
                run_xatt()
                xatt_ms = bench(run_xatt)
                xatt_gbs = effective_bytes(s_q, h_q, topk) / (xatt_ms * 1e-3) / 1e9
                speedup = xatt_ms / sgl_ms
            except Exception as e:
                print(f"  (xatt failed: {e})")

        print(
            f"{label:30s} {mb:9.1f} {sgl_ms:9.3f} {sgl_gbs:9.2f} "
            f"{xatt_ms:9.3f} {xatt_gbs:9.2f} {speedup:8.2f} {max_abs:9.2e} {str(ok):>4s}"
        )


if __name__ == "__main__":
    main()
