#pragma once

#include <ATen/ATen.h>
#include <sycl/sycl.hpp>

namespace gemm_sqrsum {

void launch_gemm_sqrsum(
    at::Tensor& C,
    at::Tensor& sqrsum,
    const at::Tensor& A,
    const at::Tensor& B);

}  // namespace gemm_sqrsum
