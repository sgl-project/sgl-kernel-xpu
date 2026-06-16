"""Benchmark FP8 MQA logits kernels (fp8_mqa_logits and fp8_paged_mqa_logits).

Typical GLM5.1 NSA parameters:
  H=64 (index_n_heads), D=128 (index_head_dim), page_size=128, index_topk=2048

Reported metrics:
  us    — median kernel latency in microseconds
  GB/s  — effective memory bandwidth (read inputs + write output)
  TFLOPS — effective throughput (dominant GEMM FLOPs: 2*M*K*N)
"""

import argparse
import struct

import sgl_kernel  # noqa: F401
import torch
import triton
import triton.testing
from sgl_kernel.nsa import _fp8_mqa_logits_impl


def make_fp8_tensor(shape, device="xpu"):
    t = torch.randn(shape, dtype=torch.float32, device=device) * 0.5
    return t.to(torch.float8_e4m3fn)


def make_kv_cache(num_pages, page_size, D, device="xpu"):
    """Build a KV cache with random FP8 keys and per-token float32 scales.
    KV cache is uint8 because each token packs 128 bytes of FP8 key data + 4 bytes of float32 scale (132 bytes total).
    Since it mixes two dtypes, uint8 is the only representation — this matches the real NSA KV cache format.
    """
    head_dim_with_sf = D + 4
    kv = torch.zeros(
        num_pages, page_size, 1, head_dim_with_sf, dtype=torch.uint8, device="cpu"
    )
    for p in range(num_pages):
        k_fp8 = torch.randn(page_size, D, dtype=torch.float32) * 0.5
        k_fp8 = k_fp8.to(torch.float8_e4m3fn).view(torch.uint8)
        kv[p, :, 0, :D] = k_fp8
        for t in range(page_size):
            scale = 0.5 + torch.rand(1).item()
            scale_bytes = struct.pack("<f", scale)
            for i, b in enumerate(scale_bytes):
                kv[p, t, 0, D + i] = b
    return kv.to(device)


def _print_table(title, col_names, rows):
    print(title)
    widths = [
        max(len(c), max(len(str(r[i])) for r in rows)) for i, c in enumerate(col_names)
    ]
    header = "  ".join(c.rjust(w) for c, w in zip(col_names, widths))
    sep = "  ".join("-" * w for w in widths)
    print(header)
    print(sep)
    for row in rows:
        print("  ".join(str(v).rjust(w) for v, w in zip(row, widths)))
    print()


# ---------------------------------------------------------------------------
# Benchmark: fp8_mqa_logits (prefill/extend)
#
# All memory traffic through device memory (including intermediates):
#   Stage 1 (_scaled_mm GEMM):
#     Read:  q (Nq*H*D fp8)  +  k (Nk*D fp8)
#     Write: dots (Nq*H*Nk f32)
#   Stage 2 (relu + weighted head-sum + masking):
#     Read:  dots (Nq*H*Nk f32)  +  weights (Nq*H f32)  +  k_scale (Nk f32)
#            + ks, ke (Nq i32 each)
#     Write: output (Nq*Nk f32)
#
# FLOPs: 2 * Nq * H * D * Nk  (dominant GEMM term)
# ---------------------------------------------------------------------------
def run_bench_fp8_mqa_logits(Nq, H, D, x_vals=None):
    device = "xpu"
    if x_vals is None:
        x_vals = [128, 256, 512, 1024, 2048, 4096]

    rows = []
    for Nk in x_vals:
        q = make_fp8_tensor((Nq, H, D), device)
        k = make_fp8_tensor((Nk, D), device)
        k_scale = torch.rand(Nk, dtype=torch.float32, device=device) + 0.5
        weights = torch.rand(Nq, H, dtype=torch.float32, device=device)
        ks = torch.zeros(Nq, dtype=torch.int32, device=device)
        ke = torch.full((Nq,), Nk, dtype=torch.int32, device=device)
        q_u8 = q.view(torch.uint8)
        k_u8 = k.view(torch.uint8)

        ms = triton.testing.do_bench(
            lambda: _fp8_mqa_logits_impl(q_u8, k_u8, k_scale, weights, ks, ke),
        )
        us = ms * 1e3

        bytes_io = (
            Nq * H * D  # q fp8 (read by GEMM)
            + Nk * D  # k fp8 (read by GEMM)
            + Nq * H * Nk * 4  # dots f32 (written by GEMM)
            + Nq * H * Nk * 4  # dots f32 (read by reduction)
            + Nq * H * 4  # weights f32 (read by reduction)
            + Nk * 4  # k_scale f32 (read by reduction)
            + Nq * 4 * 2  # ks, ke i32 (read by reduction)
            + Nq * Nk * 4  # output f32 (written)
        )
        flops = 2 * Nq * H * D * Nk  # dominant GEMM

        bw = bytes_io / (ms * 1e-3) / 1e9
        tflops = flops / (ms * 1e-3) / 1e12

        rows.append((Nk, f"{us:.3f}", f"{bw:.1f}", f"{tflops:.4f}"))

    _print_table(
        f"fp8_mqa_logits (Nq={Nq}, H={H}, D={D})",
        ["Nk", "us", "GB/s", "TFLOPS"],
        rows,
    )


# ---------------------------------------------------------------------------
# Benchmark: fp8_paged_mqa_logits (decode)
#
# All memory traffic through device memory (including intermediates):
#   Stage 1 (PagedKGatherKernel):
#     Read:  kv_cache (B*msl*(D+4) uint8)
#     Write: k_gathered (B*msl*D uint8)  +  k_scale_gathered (B*msl f32)
#   Stage 2 (batched FP8 GEMM):
#     Read:  q (B*H*D fp8)  +  k_gathered (B*msl*D uint8)
#     Write: dots (B*H*msl f32)
#   Stage 3 (Fp8PagedMqaLogitsReduceKernel):
#     Read:  dots (B*H*msl f32)  +  weights (B*H f32)  +  k_scale_gathered (B*msl f32)
#     Write: logits (B*msl f32)
#
# FLOPs: 2 * B * H * D * seq_len  (dominant GEMM term)
# ---------------------------------------------------------------------------
def run_bench_fp8_paged_mqa_logits(B, H, D, page_size, x_vals=None):
    device = "xpu"
    if x_vals is None:
        x_vals = [128, 256, 512, 1024, 2048, 4096, 8192]

    rows = []
    for seq_len in x_vals:
        num_blocks = (seq_len + page_size - 1) // page_size
        num_pages = num_blocks * B
        msl = num_blocks * page_size  # max_seq_len (rounded up to page boundary)

        kv_cache = make_kv_cache(num_pages, page_size, D, device)
        q = make_fp8_tensor((B, 1, H, D), device)
        weights = torch.rand(B, H, dtype=torch.float32, device=device)
        seq_lens = torch.full((B,), seq_len, dtype=torch.int32, device=device)
        block_tables = (
            torch.arange(num_blocks, dtype=torch.int32, device=device)
            .unsqueeze(0)
            .expand(B, -1)
            .contiguous()
        )
        q_u8 = q.view(torch.uint8)

        ms = triton.testing.do_bench(
            lambda: torch.ops.sgl_kernel.fp8_paged_mqa_logits.default(
                q_u8, kv_cache, weights, seq_lens, block_tables, None, msl, False
            ),
        )
        us = ms * 1e3

        bytes_io = (
            B * msl * (D + 4)  # kv_cache (read by gather)
            + B * msl * D  # k_gathered (written by gather)
            + B * msl * 4  # k_scale_gathered (written by gather)
            + B * H * D  # q fp8 (read by GEMM)
            + B * msl * D  # k_gathered (read by GEMM)
            + B * H * msl * 4  # dots (written by GEMM)
            + B * H * msl * 4  # dots (read by reduction)
            + B * H * 4  # weights (read by reduction)
            + B * msl * 4  # k_scale_gathered (read by reduction)
            + B * msl * 4  # logits output (written)
        )
        flops = 2 * B * H * D * seq_len  # dominant GEMM

        bw = bytes_io / (ms * 1e-3) / 1e9
        tflops = flops / (ms * 1e-3) / 1e12

        rows.append((seq_len, f"{us:.3f}", f"{bw:.1f}", f"{tflops:.4f}"))

    _print_table(
        f"fp8_paged_mqa_logits (B={B}, H={H}, D={D}, page_size={page_size})",
        ["seq_len", "us", "GB/s", "TFLOPS"],
        rows,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark FP8 MQA logits kernels")
    parser.add_argument("--H", type=int, default=64, help="Number of heads")
    parser.add_argument("--D", type=int, default=128, help="Head dimension")
    parser.add_argument("--Nq", type=int, default=1, help="Number of queries (ragged)")
    parser.add_argument("--B", type=int, default=1, help="Batch size (paged)")
    parser.add_argument("--page-size", type=int, default=128, help="Page size (paged)")
    args = parser.parse_args()

    print(f"Config: H={args.H}, D={args.D}")
    print()

    run_bench_fp8_mqa_logits(Nq=args.Nq, H=args.H, D=args.D)
    run_bench_fp8_paged_mqa_logits(
        B=args.B, H=args.H, D=args.D, page_size=args.page_size
    )

    print("Benchmark finished!")
