"""Benchmark FP8 MQA logits kernels (fp8_mqa_logits and fp8_paged_mqa_logits).

Typical GLM5.1 NSA parameters:
  H=64 (index_n_heads), D=128 (index_head_dim), page_size=64, index_topk=2048
"""

import argparse
import os
import struct

import sgl_kernel  # noqa: F401
import torch
import triton


def make_fp8_tensor(shape, device="xpu"):
    t = torch.randn(shape, dtype=torch.float32, device=device) * 0.5
    return t.to(torch.float8_e4m3fn)


def make_kv_cache(num_pages, page_size, D, device="xpu"):
    """Build a KV cache with random FP8 keys and float32 scales.
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
        scale = 0.5 + torch.rand(1).item()
        scale_bytes = struct.pack("<f", scale)
        for i, b in enumerate(scale_bytes):
            kv[p, :, 0, D + i] = b
    return kv.to(device)


# ---------------------------------------------------------------------------
# Benchmark: fp8_mqa_logits (prefill/extend)
# ---------------------------------------------------------------------------
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["Nk"],
        x_vals=[128, 256, 512, 1024, 2048, 4096],
        x_log=False,
        line_arg="provider",
        line_vals=["sgl-kernel"],
        line_names=["sgl-kernel"],
        styles=[("blue", "-")],
        ylabel="us",
        plot_name="fp8_mqa_logits",
        args={},
    )
)
def bench_fp8_mqa_logits(Nk, provider, Nq, H, D):
    device = "xpu"
    q = make_fp8_tensor((Nq, H, D), device)
    k = make_fp8_tensor((Nk, D), device)
    k_scale = torch.rand(Nk, dtype=torch.float32, device=device) + 0.5
    weights = torch.rand(Nq, H, dtype=torch.float32, device=device)
    ks = torch.zeros(Nq, dtype=torch.int32, device=device)
    ke = torch.full((Nq,), Nk, dtype=torch.int32, device=device)

    q_u8 = q.view(torch.uint8)
    k_u8 = k.view(torch.uint8)

    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: torch.ops.sgl_kernel.fp8_mqa_logits.default(
            q_u8, k_u8, k_scale, weights, ks, ke
        ),
        quantiles=[0.5, 0.2, 0.8],
    )
    return ms * 1000, max_ms * 1000, min_ms * 1000


# ---------------------------------------------------------------------------
# Benchmark: fp8_paged_mqa_logits (decode)
# ---------------------------------------------------------------------------
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=[128, 256, 512, 1024, 2048, 4096, 8192],
        x_log=False,
        line_arg="provider",
        line_vals=["sgl-kernel"],
        line_names=["sgl-kernel"],
        styles=[("blue", "-")],
        ylabel="us",
        plot_name="fp8_paged_mqa_logits",
        args={},
    )
)
def bench_fp8_paged_mqa_logits(seq_len, provider, B, H, D, page_size):
    device = "xpu"
    num_blocks = (seq_len + page_size - 1) // page_size
    num_pages = num_blocks * B
    max_seq_len = num_blocks * page_size

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

    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: torch.ops.sgl_kernel.fp8_paged_mqa_logits.default(
            q_u8, kv_cache, weights, seq_lens, block_tables, None, max_seq_len, False
        ),
        quantiles=[0.5, 0.2, 0.8],
    )
    return ms * 1000, max_ms * 1000, min_ms * 1000


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

    save_path = "bench_fp8_mqa_logits_res"
    os.makedirs(save_path, exist_ok=True)

    print(f"=== fp8_mqa_logits (Nq={args.Nq}) ===")
    bench_fp8_mqa_logits.run(
        print_data=True,
        save_path=save_path,
        Nq=args.Nq,
        H=args.H,
        D=args.D,
    )

    print()
    print(f"=== fp8_paged_mqa_logits (B={args.B}, page_size={args.page_size}) ===")
    bench_fp8_paged_mqa_logits.run(
        print_data=True,
        save_path=save_path,
        B=args.B,
        H=args.H,
        D=args.D,
        page_size=args.page_size,
    )

    print("\nBenchmark finished!")
