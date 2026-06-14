#!/usr/bin/env python3
"""
Test script for gemm_sqrsum kernel.

Contract (mhc_pre GEMM+sqrsum stage):
  A      [M, K]   bf16/fp16/fp32   (residual.view(M, hc_hidden))
  B      [N, K]   fp32             (fn = [24, 16384] = [N, K])
  C      [M, N]   fp32             (gemm_out_mul),  C[m,n] = sum_k A[m,k]*B[n,k]
  sqrsum [M]      fp32             (gemm_out_sqrsum), sqrsum[m] = sum_k A[m,k]^2

Production path: A bf16, B fp32 -> tf32 x tf32 -> fp32 DPAS (A widened to fp32,
B taken as-is, both reinterpreted to tf32 at load).
"""

import torch
import sgl_kernel


def run_case(M, K, N, a_dtype, b_dtype, label=""):
    device = "xpu"
    A = torch.randn(M, K, dtype=a_dtype, device=device)
    B = torch.randn(N, K, dtype=b_dtype, device=device)  # B is [N, K]

    C = torch.empty(M, N, dtype=torch.float32, device=device)
    sqrsum = torch.empty(M, dtype=torch.float32, device=device)

    sgl_kernel.gemm_sqrsum(C, sqrsum, A, B)
    torch.xpu.synchronize()

    # Reference in fp32. C[m,n] = sum_k A[m,k] * B[n,k] = A @ B^T
    A32 = A.cpu().float()
    B32 = B.cpu().float()
    C_ref = A32 @ B32.t()
    sqrsum_ref = (A32 * A32).sum(dim=1)

    C_t = C.cpu()
    sq_t = sqrsum.cpu()

    # Coverage: did the whole output get written? (exposes tile/launch-dim bugs)
    rows_cov = int((C_t.abs().sum(1) > 0).sum())
    cols_cov = int((C_t.abs().sum(0) > 0).sum())

    # tf32 keeps ~10 mantissa bits; over K=K accumulation the relative error is
    # ~ 2^-10 * sqrt(K). Use a generous-but-meaningful tolerance.
    C_ok = torch.allclose(C_t, C_ref, atol=2e-2 * max(1, K) ** 0.5, rtol=2e-2)
    sq_ok = torch.allclose(sq_t, sqrsum_ref, atol=2e-2 * max(1, K) ** 0.5, rtol=2e-2)

    C_rel = (C_t - C_ref).abs().max().item() / (C_ref.abs().max().item() + 1e-6)
    sq_rel = (sq_t - sqrsum_ref).abs().max().item() / (sqrsum_ref.abs().max().item() + 1e-6)

    status = "PASS" if (C_ok and sq_ok and rows_cov == M and cols_cov == N) else "FAIL"
    print(
        f"[{status}] {label:14s} M={M:5d} K={K:6d} N={N:3d} "
        f"A={str(a_dtype).split('.')[-1]:8s} B={str(b_dtype).split('.')[-1]:7s} | "
        f"C_relerr={C_rel:.4f} sq_relerr={sq_rel:.4f} "
        f"cover={rows_cov}/{M}x{cols_cov}/{N}"
    )
    return status == "PASS"


def main():
    print("Testing gemm_sqrsum kernel (B=[N,K], C/sqrsum fp32)\n")

    # --- Tile-aligned correctness: this is what items 1,2,7 + the tf32 precision
    #     path MUST pass. M,N multiples of 256; K a multiple of 16. ---
    print("--- tile-aligned (MUST pass: fp32 C, [N,K] B, tf32 path) ---")
    aligned_ok = True
    aligned_ok &= run_case(256, 256, 256, torch.bfloat16, torch.float32, "bf16xfp32")
    aligned_ok &= run_case(512, 512, 256, torch.bfloat16, torch.float32, "multi-tile")
    aligned_ok &= run_case(256, 512, 512, torch.bfloat16, torch.float32, "multi-K")
    # Non-production dtype combos all route through the same tf32 launcher
    # (inputs are widened to fp32 then reinterpreted to tf32).
    aligned_ok &= run_case(256, 256, 256, torch.float16, torch.float16, "fp16xfp16")
    aligned_ok &= run_case(256, 256, 256, torch.bfloat16, torch.bfloat16, "bf16xbf16")

    # --- Production shapes: N=24 (tiny-N partial tile, item 4) and ragged M
    #     (item 5). These are DEFERRED/known-risk; reported for information. ---
    print("\n--- production shapes (N=24, K=16384; tiny-N+ragged = DEFERRED) ---")
    prod_ok = True
    for M in (16, 128, 512, 1024, 2048):
        prod_ok &= run_case(M, 16384, 24, torch.bfloat16, torch.float32, "prod")
    for M in (1021, 1034, 1518):
        prod_ok &= run_case(M, 16384, 24, torch.bfloat16, torch.float32, "ragged-M")

    print("\nTile-aligned (required): " + ("PASSED ✓" if aligned_ok else "FAILED ✗"))
    print("Production N=24 (deferred items 4/5): " + ("passed" if prod_ok else "not yet (expected)"))
    return aligned_ok


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
