#!/usr/bin/env python3
"""
Test script for gemm_sqrsum kernel

Computes:
  C = A @ B
  sqrsum[i] = sum(A[i,:]^2) for each row i
"""

import torch
import sgl_kernel

def test_gemm_sqrsum():
    print("Testing gemm_sqrsum kernel...")

    # Test dimensions
    M, K, N = 256, 128, 256
    device = "xpu"
    dtype = torch.float16

    print(f"  M={M}, K={K}, N={N}")
    print(f"  dtype={dtype}, device={device}")

    # Create input tensors
    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)

    # Allocate outputs
    C = torch.empty(M, N, dtype=dtype, device=device)
    sqrsum = torch.empty(M, dtype=torch.float32, device=device)  # sqrsum must be float32

    # Run kernel
    print("  Running kernel...")
    sgl_kernel.gemm_sqrsum(C, sqrsum, A, B)

    # Compute reference on CPU
    print("  Computing reference...")
    A_cpu = A.cpu().float()
    B_cpu = B.cpu().float()

    C_ref = torch.matmul(A_cpu, B_cpu).to(dtype).to(device)
    sqrsum_ref = (A_cpu * A_cpu).sum(dim=1).to(dtype).to(device)

    # Verify results
    C_error = (C - C_ref).abs().max().item()
    sqrsum_error = (sqrsum - sqrsum_ref).abs().max().item()

    C_rel_error = (C_error / (C_ref.abs().max().item() + 1e-6))
    sqrsum_rel_error = (sqrsum_error / (sqrsum_ref.abs().max().item() + 1e-6))

    print(f"\nResults:")
    print(f"  C max abs error: {C_error:.6f}")
    print(f"  C max rel error: {C_rel_error:.6f}")
    print(f"  sqrsum max abs error: {sqrsum_error:.6f}")
    print(f"  sqrsum max rel error: {sqrsum_rel_error:.6f}")

    # Check against tolerance
    tol = 1e-2  # fp16 tolerance
    if C_rel_error < tol and sqrsum_rel_error < tol:
        print("\n✓ Test PASSED!")
        return True
    else:
        print("\n✗ Test FAILED!")
        print(f"  First few C values (kernel):    {C[0, :5]}")
        print(f"  First few C values (reference): {C_ref[0, :5]}")
        print(f"  First few sqrsum (kernel):      {sqrsum[:5]}")
        print(f"  First few sqrsum (reference):   {sqrsum_ref[:5]}")
        return False

if __name__ == "__main__":
    try:
        test_gemm_sqrsum()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
