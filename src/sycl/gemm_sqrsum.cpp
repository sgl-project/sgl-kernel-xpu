
#define SYCL_INTEL_TARGET 20
#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>

#include "Utils.h"
#include "sycl/kernels/gemm_sqrsum/device/gemm_sqrsum_types.hpp"

void gemm_with_sqrsum(at::Tensor& C, at::Tensor& sqrsum, const at::Tensor& A, const at::Tensor& B) {
  c10::DeviceGuard guard(A.device());

  CHECK_DEVICE(A);
  CHECK_DEVICE(B);
  CHECK_INPUT(C);
  CHECK_INPUT(sqrsum);

  auto a_dtype = A.scalar_type();
  auto b_dtype = B.scalar_type();

  TORCH_CHECK(C.scalar_type() == at::ScalarType::Float, "C must be float32");
  TORCH_CHECK(sqrsum.scalar_type() == at::ScalarType::Float, "sqrsum must be float32");

  auto is_supported = [](at::ScalarType t) {
    return t == at::ScalarType::Half || t == at::ScalarType::BFloat16 || t == at::ScalarType::Float;
  };
  TORCH_CHECK(is_supported(a_dtype), "A must be Half, BFloat16, or Float, got ", a_dtype);
  TORCH_CHECK(is_supported(b_dtype), "B must be Half, BFloat16, or Float, got ", b_dtype);

  TORCH_CHECK(A.dim() == 2, "A must be 2D [M, K]");
  TORCH_CHECK(B.dim() == 2, "B must be 2D [N, K]");
  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(0);

  TORCH_CHECK(B.size(1) == K, "K mismatch for GEMM: A.K=", K, " but B.K=", B.size(1));
  TORCH_CHECK(C.dim() == 3 && C.size(1) == M && C.size(2) == N, "Output C must be [n_splits, ", M, ", ", N, "]");
  TORCH_CHECK(
      sqrsum.dim() == 2 && sqrsum.size(0) == C.size(0) && sqrsum.size(1) == M,
      "Output sqrsum must be [n_splits, ",
      M,
      "] matching C's leading dim");

  runGemmSqrSum(C, sqrsum, A, B);
}

#undef SYCL_INTEL_TARGET
