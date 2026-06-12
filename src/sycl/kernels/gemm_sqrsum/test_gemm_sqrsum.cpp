/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
/*! \file
    \brief Simple test for GEMM + Square Sum kernel
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <sycl/sycl.hpp>

// Simple CPU reference implementation
void gemm_sqrsum_reference(
    int M, int K, int N,
    const float* A,  // [M, K] row-major
    const float* B,  // [K, N] row-major
    float* C,        // [M, N] row-major
    float* sqrsum) { // [M]

  // Initialize outputs
  for (int i = 0; i < M * N; i++) C[i] = 0.0f;
  for (int i = 0; i < M; i++) sqrsum[i] = 0.0f;

  // Compute C = A @ B
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) {
        sum += A[m * K + k] * B[k * N + n];
      }
      C[m * N + n] = sum;
    }
  }

  // Compute sqrsum[i] = sum(A[i,:]^2)
  for (int m = 0; m < M; m++) {
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
      float val = A[m * K + k];
      sum += val * val;
    }
    sqrsum[m] = sum;
  }
}

bool verify_results(
    int M, int N,
    const float* C_ref, const float* C_test,
    const float* sqrsum_ref, const float* sqrsum_test,
    float tolerance = 1e-3f) {

  bool passed = true;

  // Verify C
  for (int i = 0; i < M * N; i++) {
    float diff = std::abs(C_ref[i] - C_test[i]);
    float rel_err = diff / (std::abs(C_ref[i]) + 1e-6f);
    if (rel_err > tolerance) {
      std::cout << "C mismatch at " << i << ": ref=" << C_ref[i]
                << " test=" << C_test[i] << " rel_err=" << rel_err << std::endl;
      passed = false;
      if (i > 10) break;  // Don't spam too many errors
    }
  }

  // Verify sqrsum
  for (int i = 0; i < M; i++) {
    float diff = std::abs(sqrsum_ref[i] - sqrsum_test[i]);
    float rel_err = diff / (std::abs(sqrsum_ref[i]) + 1e-6f);
    if (rel_err > tolerance) {
      std::cout << "sqrsum mismatch at " << i << ": ref=" << sqrsum_ref[i]
                << " test=" << sqrsum_test[i] << " rel_err=" << rel_err << std::endl;
      passed = false;
      if (i > 10) break;
    }
  }

  return passed;
}

int main(int argc, char** argv) {
  // Test dimensions
  int M = 256;
  int K = 128;
  int N = 256;

  if (argc > 1) M = std::atoi(argv[1]);
  if (argc > 2) K = std::atoi(argv[2]);
  if (argc > 3) N = std::atoi(argv[3]);

  std::cout << "Testing GEMM + Square Sum kernel with M=" << M << " K=" << K << " N=" << N << std::endl;

  // Allocate host memory
  std::vector<float> h_A(M * K);
  std::vector<float> h_B(K * N);
  std::vector<float> h_C_ref(M * N);
  std::vector<float> h_C_test(M * N);
  std::vector<float> h_sqrsum_ref(M);
  std::vector<float> h_sqrsum_test(M);

  // Initialize with random values
  for (int i = 0; i < M * K; i++) {
    h_A[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
  }
  for (int i = 0; i < K * N; i++) {
    h_B[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
  }

  // Compute reference on CPU
  std::cout << "Computing reference on CPU..." << std::endl;
  gemm_sqrsum_reference(M, K, N, h_A.data(), h_B.data(), h_C_ref.data(), h_sqrsum_ref.data());

  try {
    // Create SYCL queue
    sycl::queue q{sycl::gpu_selector_v};
    std::cout << "Running on device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    // Allocate device memory
    float* d_A = sycl::malloc_device<float>(M * K, q);
    float* d_B = sycl::malloc_device<float>(K * N, q);
    float* d_C = sycl::malloc_device<float>(M * N, q);
    float* d_sqrsum = sycl::malloc_device<float>(M, q);

    // Copy data to device
    q.memcpy(d_A, h_A.data(), M * K * sizeof(float)).wait();
    q.memcpy(d_B, h_B.data(), K * N * sizeof(float)).wait();

    // TODO: Launch kernel here
    // cutlass::gemm_sqrsum::launch_gemm_sqrsum(q, M, K, N, d_A, d_B, d_C, d_sqrsum);

    std::cout << "NOTE: Kernel launch not implemented yet - this is a skeleton test" << std::endl;
    std::cout << "To complete:" << std::endl;
    std::cout << "  1. Fix template instantiation issues" << std::endl;
    std::cout << "  2. Add proper epilogue for writing outputs" << std::endl;
    std::cout << "  3. Integrate with CMake build system" << std::endl;

    // Copy results back (would be done after kernel launch)
    // q.memcpy(h_C_test.data(), d_C, M * N * sizeof(float)).wait();
    // q.memcpy(h_sqrsum_test.data(), d_sqrsum, M * sizeof(float)).wait();

    // Verify results
    // if (verify_results(M, N, h_C_ref.data(), h_C_test.data(),
    //                    h_sqrsum_ref.data(), h_sqrsum_test.data())) {
    //   std::cout << "Test PASSED!" << std::endl;
    // } else {
    //   std::cout << "Test FAILED!" << std::endl;
    // }

    // Cleanup
    sycl::free(d_A, q);
    sycl::free(d_B, q);
    sycl::free(d_C, q);
    sycl::free(d_sqrsum, q);

  } catch (sycl::exception const& e) {
    std::cout << "SYCL exception: " << e.what() << std::endl;
    return 1;
  }

  std::cout << "\nReference results computed successfully!" << std::endl;
  std::cout << "First few sqrsum values: ";
  for (int i = 0; i < std::min(5, M); i++) {
    std::cout << h_sqrsum_ref[i] << " ";
  }
  std::cout << std::endl;

  return 0;
}
