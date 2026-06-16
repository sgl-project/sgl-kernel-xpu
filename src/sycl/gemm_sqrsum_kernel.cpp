#define SYCL_INTEL_TARGET 20

#include "sycl/kernels/gemm_sqrsum/device/gemm_sqrsum_dispatch.hpp"
#include "sycl/kernels/gemm_sqrsum/device/gemm_sqrsum_types.hpp"

namespace gemm_sqrsum {

void launch_gemm_sqrsum(at::Tensor& C, at::Tensor& sqrsum, const at::Tensor& A, const at::Tensor& B) {
  runGemmSqrSum(C, sqrsum, A, B);
}

}  // namespace gemm_sqrsum
