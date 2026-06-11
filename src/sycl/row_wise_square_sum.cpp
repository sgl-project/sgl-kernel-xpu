/***************************************************************************************************
 * Copyright 2025 SGLang Team. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 **************************************************************************************************/
/*! \file
    \brief Row-wise square sum dispatch interface for PyTorch
*/

#define SYCL_INTEL_TARGET 20

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>
#include <sycl/sycl.hpp>

#include "Utils.h"
#include "sycl/kernels/reduction/collective/xe_row_square_sum_epilogue.hpp"
#include "sycl/kernels/reduction/collective/xe_row_square_sum_mainloop.hpp"
#include "sycl/kernels/reduction/device/row_square_sum_runner.hpp"
#include "sycl/kernels/reduction/kernel/xe_row_square_sum_kernel.hpp"

using namespace cute;

namespace {

///////////////////////////////////////////////////////////////////////////////
/// Type Configuration
///////////////////////////////////////////////////////////////////////////////
template <typename Element>
struct RowSquareSumConfig {
  // Tile shape: 1 row x 256 columns per iteration
  using TileShape = Shape<_1, _256>;

  // Strides
  using StrideA = cute::Stride<int, cute::Int<1>>;  // Row-major: (N, 1)
  using StrideD = cute::Int<1>;                     // Output vector: stride 1

  // Collectives
  using CollectiveMainloop =
      cutlass::reduction::collective::XeRowSquareSumMainloop<TileShape, Element, StrideA>;

  using CollectiveEpilogue =
      cutlass::reduction::collective::XeRowSquareSumEpilogue<Element, float, StrideD>;

  // Kernel
  using Kernel = cutlass::reduction::kernel::XeRowSquareSumKernel<CollectiveMainloop, CollectiveEpilogue>;

  // Device adapter
  using DeviceOp = cutlass::reduction::device::RowSquareSum<Kernel>;
};

///////////////////////////////////////////////////////////////////////////////
/// Templated launcher
///////////////////////////////////////////////////////////////////////////////
template <typename Element>
void launch_row_wise_sum_impl(const at::Tensor& A, at::Tensor& D) {
  using Config = RowSquareSumConfig<Element>;
  using DeviceOp = typename Config::DeviceOp;
  using Kernel = typename Config::Kernel;
  using StrideA = typename Config::StrideA;
  using StrideD = typename Config::StrideD;

  const int M = static_cast<int>(A.size(0));
  const int N = static_cast<int>(A.size(1));

  // Prepare arguments
  typename Kernel::Arguments args{
      {M, N},                                   // shape
      A.data_ptr<Element>(),                    // ptr_A
      StrideA{N, cute::Int<1>{}},               // stride_A
      D.data_ptr<Element>(),                    // ptr_D
      cute::Int<1>{}                            // stride_D (contiguous vector)
  };

  // Create device op and run
  DeviceOp device_op;

  cutlass::Status status = device_op.can_implement(args);
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "RowSquareSum kernel cannot implement this problem: M=",
      M,
      ", N=",
      N);

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  status = device_op.initialize(args, nullptr, queue);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "RowSquareSum initialization failed");

  status = device_op.run(queue);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "RowSquareSum execution failed");

  c10::xpu::syncStreamsOnDevice(A.device().index());
}

}  // namespace

///////////////////////////////////////////////////////////////////////////////
/// Public interface
///////////////////////////////////////////////////////////////////////////////
void row_wise_sum_cutlass(const at::Tensor& A, at::Tensor& D) {
  CHECK_INPUT(A);
  CHECK_INPUT(D);

  TORCH_CHECK(A.dim() == 2, "Input must be 2D, got ", A.dim(), "D");
  TORCH_CHECK(D.dim() == 1, "Output must be 1D, got ", D.dim(), "D");

  const int64_t M = A.size(0);
  const int64_t N = A.size(1);

  TORCH_CHECK(D.size(0) == M, "Output size (", D.size(0), ") must equal number of rows (", M, ")");

  auto dtype = A.scalar_type();
  TORCH_CHECK(
      dtype == at::ScalarType::Float || dtype == at::ScalarType::Half || dtype == at::ScalarType::BFloat16,
      "Unsupported data type: ",
      dtype);

  TORCH_CHECK(A.scalar_type() == D.scalar_type(), "Input and output must have same dtype");

  c10::DeviceGuard guard(A.device());

  // Dispatch by data type
  if (dtype == at::ScalarType::Float) {
    launch_row_wise_sum_impl<float>(A, D);
  } else if (dtype == at::ScalarType::Half) {
    launch_row_wise_sum_impl<at::Half>(A, D);
  } else if (dtype == at::ScalarType::BFloat16) {
    launch_row_wise_sum_impl<at::BFloat16>(A, D);
  } else {
    TORCH_CHECK(false, "Unsupported data type");
  }
}
