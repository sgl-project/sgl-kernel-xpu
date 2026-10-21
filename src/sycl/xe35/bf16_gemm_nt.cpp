/***************************************************************************************************
 * Copyright 2026 SGLang Team
 * SPDX-License-Identifier: Apache-2.0
 **************************************************************************************************/

#define SYCL_INTEL_TARGET 35

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cute/tensor.hpp>
#include <type_traits>

#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "sgl_kernel/xe35/bf16_gemm_ops.h"

namespace {

using namespace cute;

template <int TileM, int TileN, int SgM, int SgN, int TileK = 32, bool PackedB = false, int Stages = 2>
struct Bf16GemmNtConfig_xe35 {
  using ElementAccumulator = float;
  using Element = cutlass::bfloat16_t;
  using LayoutB = std::conditional_t<PackedB, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>;
  using TileShape = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;
  using SubgroupLayout = Layout<Shape<Int<SgM>, Int<SgN>, _1>, Stride<Int<SgN>, _1, _0>>;

  using TiledMma =
      typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>, Layout<TileShape>, SubgroupLayout>::
          TiledMMA;

  using EpiloguePolicy = cutlass::epilogue::IntelXeXMX16;
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      Element,
      float,
      ElementAccumulator,
      ElementAccumulator,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using FusionCallbacks = cutlass::epilogue::fusion::
      FusionCallbacks<EpiloguePolicy, EpilogueOp, TileShape, decltype(tile_shape(TiledMma()))>;
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
      EpiloguePolicy,
      TileShape,
      ElementAccumulator,
      cutlass::gemm::TagToStrideC_t<cutlass::layout::RowMajor>,
      Element,
      cutlass::gemm::TagToStrideC_t<cutlass::layout::RowMajor>,
      FusionCallbacks,
      XE_2D_U32x8x16_LD_N,
      void,
      void,
      XE_2D_U16x8x16_ST_N,
      void,
      void>;

  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      cutlass::gemm::MainloopXeL1Staged<Stages>,
      TileShape,
      Element,
      cutlass::gemm::TagToStrideA_t<cutlass::layout::RowMajor>,
      Element,
      cutlass::gemm::TagToStrideB_t<LayoutB>,
      TiledMma,
      void,
      void,
      void,
      cute::identity,
      void,
      void,
      void,
      cute::identity>;
  using Kernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

template <typename Gemm>
cutlass::Status launch_xe35(
    const at::Tensor& a,
    const at::Tensor& b,
    at::Tensor& out,
    const cutlass::KernelHardwareInfo& hw_info,
    sycl::queue& queue,
    int n,
    int k,
    int packed_b_width) {
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementD = typename Gemm::CollectiveEpilogue::ElementOutput;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  const int m = static_cast<int>(a.size(0));
  auto problem = make_shape(m, n, k, 1);
  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, make_shape(m, k, 1));
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, make_shape(packed_b_width, k, 1));
  StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, make_shape(m, n, 1));
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, make_shape(m, n, 1));

  typename Gemm::GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem,
      {static_cast<ElementA*>(a.data_ptr()), stride_a, static_cast<ElementB*>(b.data_ptr()), stride_b},
      {{1.0f, 0.0f}, nullptr, stride_c, static_cast<ElementD*>(out.data_ptr()), stride_d},
      hw_info};

  Gemm op;
  auto status = op.can_implement(args);
  if (status != cutlass::Status::kSuccess) return status;
  status = op.initialize(args, nullptr, &queue);
  if (status != cutlass::Status::kSuccess) return status;
  return op.run(&queue);
}

template <typename Config>
cutlass::Status launch_config_xe35(
    const at::Tensor& a,
    const at::Tensor& b,
    at::Tensor& out,
    const cutlass::KernelHardwareInfo& hw_info,
    sycl::queue& queue,
    int n,
    int k,
    int packed_b_width) {
  return launch_xe35<typename Config::Gemm>(a, b, out, hw_info, queue, n, k, packed_b_width);
}

}  // namespace

void bf16_gemm_nt_xpu_xe35(at::Tensor& output, const at::Tensor& mat_a, const at::Tensor& mat_b, int64_t config) {
  TORCH_CHECK(mat_a.dim() == 2 && mat_b.dim() == 2 && output.dim() == 2, "bf16_gemm_nt expects 2D tensors");
  TORCH_CHECK(mat_a.is_xpu() && mat_b.is_xpu() && output.is_xpu(), "bf16_gemm_nt expects XPU tensors");
  TORCH_CHECK(
      mat_a.device() == mat_b.device() && mat_a.device() == output.device(),
      "bf16_gemm_nt tensors must be on the same device");
  TORCH_CHECK(
      mat_a.scalar_type() == at::kBFloat16 && mat_b.scalar_type() == at::kBFloat16 &&
          output.scalar_type() == at::kBFloat16,
      "bf16_gemm_nt currently supports BF16 inputs and output");
  TORCH_CHECK(mat_a.is_contiguous() && mat_b.is_contiguous() && output.is_contiguous(), "tensors must be contiguous");
  const bool packed_b = config == 2;
  TORCH_CHECK(mat_a.size(1) == (packed_b ? mat_b.size(0) : mat_b.size(1)), "K mismatch");
  TORCH_CHECK(
      output.size(0) == mat_a.size(0) && output.size(1) > 0 &&
          (packed_b ? output.size(1) <= mat_b.size(1) : output.size(1) == mat_b.size(0)),
      "output shape mismatch");
  TORCH_CHECK(mat_a.size(0) > 0 && mat_b.size(0) > 0 && mat_a.size(1) > 0, "dimensions must be positive");
  TORCH_CHECK(
      mat_a.size(0) <= INT_MAX && mat_b.size(0) <= INT_MAX && mat_a.size(1) <= INT_MAX && mat_b.size(1) <= INT_MAX &&
          output.size(1) <= INT_MAX,
      "dimension too large");

  c10::DeviceGuard guard(mat_a.device());
  sycl::queue& queue = at::xpu::getCurrentXPUStream(mat_a.device().index()).queue();
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = mat_a.device().index();
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  const int m = static_cast<int>(mat_a.size(0));
  const int n = static_cast<int>(output.size(1));
  const int k = static_cast<int>(mat_a.size(1));
  const int packed_b_width = packed_b ? static_cast<int>(mat_b.size(1)) : n;
  TORCH_CHECK(
      n % 16 == 0 && k % 16 == 0 && packed_b_width % 16 == 0, "N, K, and packed B width must be multiples of 16");
  cutlass::Status status;
  if (config == 0 || config == 1) {
    status = launch_config_xe35<Bf16GemmNtConfig_xe35<64, 256, 8, 4, 64>>(
        mat_a, mat_b, output, hw_info, queue, n, k, packed_b_width);
  } else if (config == 2) {
    const bool tp4_projection =
        (n == 4608 && k == 8192) || (n == 4096 && k == 8192) || ((n == 8192 || n == 8448) && k == 4096);
    if (tp4_projection && m <= 16) {
      status = m <= 8 ? launch_config_xe35<Bf16GemmNtConfig_xe35<8, 256, 1, 16, 32, true, 4>>(
                            mat_a, mat_b, output, hw_info, queue, n, k, packed_b_width)
                      : launch_config_xe35<Bf16GemmNtConfig_xe35<16, 256, 2, 16, 32, true, 4>>(
                            mat_a, mat_b, output, hw_info, queue, n, k, packed_b_width);
    } else if (m <= 8 && n == 18432 && k == 8192) {
      status = launch_config_xe35<Bf16GemmNtConfig_xe35<8, 256, 1, 16, 32, true, 2>>(
          mat_a, mat_b, output, hw_info, queue, n, k, packed_b_width);
    } else if (m <= 16) {
      status = launch_config_xe35<Bf16GemmNtConfig_xe35<16, 256, 2, 16, 32, true, 2>>(
          mat_a, mat_b, output, hw_info, queue, n, k, packed_b_width);
    } else {
      status = launch_config_xe35<Bf16GemmNtConfig_xe35<32, 256, 4, 8, 64, true, 2>>(
          mat_a, mat_b, output, hw_info, queue, n, k, packed_b_width);
    }
  } else {
    status = cutlass::Status::kErrorInvalidProblem;
  }
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "bf16_gemm_nt failed: ",
      cutlassGetStatusString(status),
      " for M=",
      mat_a.size(0),
      " N=",
      n,
      " K=",
      mat_a.size(1));
}

#undef SYCL_INTEL_TARGET
