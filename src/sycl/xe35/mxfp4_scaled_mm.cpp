/***************************************************************************************************
 * Copyright 2026 Intel corporation. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Dense MXFP4 (E2M1 data, UE8M0 scales, block=32) scaled GEMM on Intel XPU (xe35).

    D = alpha * (A @ B^T), with A/B dequantized per 32-element K block by their
    UE8M0 scales. HW-accelerated via XE_BDPAS. NVFP4 (block 16, E4M3 scales) has
    no Xe equivalent, so this op implements the MX flavour of the format.
*/

#undef SYCL_INTEL_TARGET
#define SYCL_INTEL_TARGET 35

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cute/tensor.hpp>

#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/float_subbyte.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/packed_stride.hpp"
#include "sgl_kernel_export.h"

#define CUTLASS_CHECK(status)              \
  {                                        \
    auto s_ = (status);                    \
    if (s_ != cutlass::Status::kSuccess) { \
      return s_;                           \
    }                                      \
  }

using namespace cute;

namespace {

// A is (M, K) RowMajor and B is (N, K) ColumnMajor: both K-contiguous, which is
// what torch hands us for packed (M, K/2) and (N, K/2) uint8 buffers, so neither
// operand needs a transpose. Only the scales do (the mainloop wants MN-major).
template <class TileShape_, class WarpLayout_, typename OutputT>
struct MXFP4ScaledGemmConfigT {
  using MxType = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
  using ElementInput = typename MxType::DataType;         // float_e2m1_t
  using ElementScale = typename MxType::ScaleFactorType;  // float_ue8m0_t

  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ElementOutput = OutputT;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  using StrideScale = cute::Stride<_1, int64_t, int64_t>;

  using TileShape = TileShape_;

  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_BDPAS_TT<8, ElementAccumulator, ElementInput>>,
      cute::Layout<TileShape>,
      WarpLayout_>::TiledMMA;

  // Block-scaled mainloop: XE_BDPAS applies one UE8M0 scale per 32-element K
  // block, the only scale block size it accepts for E2M1.
  using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16BlockScaled<2>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGeneric;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementOutput,
      ElementComputeEpilogue,
      ElementAccumulator,
      ElementAccumulator,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using FusionCallbacks = cutlass::epilogue::fusion::
      FusionCallbacks<EpilogueDispatchPolicy, EpilogueOp, TileShape, decltype(tile_shape(TiledMma()))>;

  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
      EpilogueDispatchPolicy,
      TileShape,
      void,  // EpilogueTile: derived from the WG tile
      void,  // ElementC: no source operand
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      ElementOutput,
      cutlass::gemm::TagToStrideC_t<LayoutD>,
      FusionCallbacks,
      void,
      void>;

  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      GEMMDispatchPolicy,
      TileShape,
      cute::tuple<ElementInput, ElementScale>,
      cute::tuple<cutlass::gemm::TagToStrideA_t<LayoutA>, StrideScale>,
      cute::tuple<ElementInput, ElementScale>,
      cute::tuple<cutlass::gemm::TagToStrideB_t<LayoutB>, StrideScale>,
      TiledMma,
      // void copy atoms select block-2D auto-detection. They are wrapped in
      // cute::type_list rather than a tuple because DPCPP's SYCL kernel-name
      // registrar instantiates this argument and no tuple can hold void members.
      cute::type_list<void, void>,
      void,
      void,
      cute::identity,
      cute::type_list<void, void>,
      void,
      void,
      cute::identity>;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename Gemm>
struct MXFP4ScaledGemmRunner {
  using CollectiveMainloop = typename Gemm::CollectiveMainloop;
  using CollectiveEpilogue = typename Gemm::CollectiveEpilogue;

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementOutput = typename CollectiveEpilogue::ElementOutput;
  using ElementScaleA = typename CollectiveMainloop::NonVoidElementScaleA;
  using ElementScaleB = typename CollectiveMainloop::NonVoidElementScaleB;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using StrideScaleA = typename CollectiveMainloop::StrideScaleA;
  using StrideScaleB = typename CollectiveMainloop::StrideScaleB;

  cutlass::Status
  run(const at::Tensor& mat_a,    // (M, K/2) uint8, packed E2M1
      const at::Tensor& mat_b,    // (N, K/2) uint8, packed E2M1
      const at::Tensor& scale_a,  // (K/32, padded_M) uint8, UE8M0, MN-major
      const at::Tensor& scale_b,  // (K/32, padded_N) uint8, UE8M0, MN-major
      const float* alpha_ptr,     // device scalar, or nullptr for alpha = 1
      at::Tensor& out,
      const cutlass::KernelHardwareInfo& hw_info,
      sycl::queue* queue) {
    const int M = static_cast<int>(mat_a.size(0));
    const int N = static_cast<int>(mat_b.size(0));
    const int K = static_cast<int>(mat_a.size(1)) * 2;
    const int L = 1;

    auto problem_shape = cute::make_shape(M, N, K, L);
    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

    // Scale strides come from the padded buffers, not from M/N, so the block-2D
    // pitch stays DWord-aligned even when M or N is not a multiple of 4.
    StrideScaleA stride_SA{_1{}, scale_a.stride(0), scale_a.numel()};
    StrideScaleB stride_SB{_1{}, scale_b.stride(0), scale_b.numel()};

    using EpilogueArgs = typename Gemm::GemmKernel::EpilogueArguments;
    EpilogueArgs epilogue_args{
        {/*alpha*/ 1.0f, /*beta*/ 0.0f, alpha_ptr, /*beta_ptr*/ nullptr},
        /*C*/ nullptr,
        stride_C,
        static_cast<ElementOutput*>(out.data_ptr()),
        stride_D};

    typename Gemm::GemmKernel::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_shape,
        {reinterpret_cast<ElementA const*>(mat_a.data_ptr()),
         stride_A,
         reinterpret_cast<ElementB const*>(mat_b.data_ptr()),
         stride_B,
         reinterpret_cast<ElementScaleA const*>(scale_a.data_ptr()),
         stride_SA,
         reinterpret_cast<ElementScaleB const*>(scale_b.data_ptr()),
         stride_SB},
        epilogue_args,
        hw_info};

    Gemm gemm_op;

    // Torch's allocator is stream-ordered; cutlass::device_memory frees without
    // waiting, which would race the kernel if the workspace were ever non-empty.
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    at::Tensor workspace = at::empty({static_cast<int64_t>(workspace_size)}, mat_a.options().dtype(at::kByte));

    // The mainloop rejects M or N % 4 != 0 because a scale surface narrower than
    // a DWord is illegal, but CUTLASS rounds that width up itself and the
    // epilogue drops the extra rows/columns, so the padded scale buffers make
    // the load safe. Only consult can_implement where it can pass.
    if (M % 4 == 0 && N % 4 == 0) {
      CUTLASS_CHECK(gemm_op.can_implement(arguments));
    }
    CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.data_ptr(), queue));
    CUTLASS_CHECK(gemm_op.run(queue));

    return cutlass::Status::kSuccess;
  }
};

using WarpLayoutLarge = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;
using WarpLayoutSmall = Layout<Shape<_4, _4, _1>, Stride<_4, _1, _0>>;

// K_TILE=64 throughout: SG_K must be >= 32 and the mainloop wants two 32-wide
// scale groups per K step.
template <typename OutputT>
cutlass::Status run_mxfp4_scaled_gemm_dispatch(
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& SA,
    const at::Tensor& SB,
    const float* alpha_ptr,
    at::Tensor& out,
    const cutlass::KernelHardwareInfo& hw_info,
    sycl::queue* queue) {
  const int64_t M = A.size(0);
  const int64_t N = B.size(0);

  if (M <= 128 && N <= 128) {
    using Config = MXFP4ScaledGemmConfigT<Shape<_128, _128, _64>, WarpLayoutSmall, OutputT>;
    return MXFP4ScaledGemmRunner<typename Config::Gemm>{}.run(A, B, SA, SB, alpha_ptr, out, hw_info, queue);
  } else if (N <= 128) {
    using Config = MXFP4ScaledGemmConfigT<Shape<_256, _128, _64>, WarpLayoutLarge, OutputT>;
    return MXFP4ScaledGemmRunner<typename Config::Gemm>{}.run(A, B, SA, SB, alpha_ptr, out, hw_info, queue);
  } else {
    using Config = MXFP4ScaledGemmConfigT<Shape<_256, _256, _64>, WarpLayoutLarge, OutputT>;
    return MXFP4ScaledGemmRunner<typename Config::Gemm>{}.run(A, B, SA, SB, alpha_ptr, out, hw_info, queue);
  }
}

// Scales arrive row-major (MN, K/32) straight from the quantizer; the mainloop
// needs them MN-major. The block-2D scale load rounds its surface width (MN
// bytes) up to a DWord, so pad the transpose target and zero the pad: the MMA
// reads those lanes before the epilogue discards them.
at::Tensor to_mn_major_scales(const at::Tensor& sf, int64_t mn, int64_t scale_k) {
  constexpr int64_t kScaleAlign = 4;
  const int64_t padded_mn = (mn + kScaleAlign - 1) / kScaleAlign * kScaleAlign;
  at::Tensor out =
      (padded_mn == mn) ? at::empty({scale_k, padded_mn}, sf.options()) : at::zeros({scale_k, padded_mn}, sf.options());
  out.narrow(1, 0, mn).copy_(sf.t());
  return out;
}

}  // namespace

// D = alpha * (A @ B^T) in MXFP4.
//   A:     (M, K/2) uint8, two E2M1 nibbles per byte (low nibble first)
//   B:     (N, K/2) uint8, same packing
//   A_sf:  (M, K/32) uint8 UE8M0, row-major (stored_byte = exponent + 127)
//   B_sf:  (N, K/32) uint8 UE8M0, row-major
//   alpha: optional 1-element fp32 device tensor; read on device, so passing it
//          never forces a sync
//   D:     (M, N) in float32, bfloat16, or float16
SGL_KERNEL_EXPORT void cutlass_scaled_fp4_mm(
    torch::Tensor& D,
    torch::Tensor const& A,
    torch::Tensor const& B,
    torch::Tensor const& A_sf,
    torch::Tensor const& B_sf,
    c10::optional<torch::Tensor> const& alpha) {
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A/B must be 2D");
  TORCH_CHECK(A.size(1) == B.size(1), "A and B must agree on the packed K extent");
  TORCH_CHECK(D.dim() == 2 && D.size(0) == A.size(0) && D.size(1) == B.size(0), "D must be (M, N)");

  TORCH_CHECK(A.is_xpu() && B.is_xpu() && A_sf.is_xpu() && B_sf.is_xpu() && D.is_xpu(), "all tensors must be on XPU");
  TORCH_CHECK(B.device() == A.device() && D.device() == A.device(), "all tensors must share one device");

  TORCH_CHECK(
      A.scalar_type() == at::ScalarType::Byte && B.scalar_type() == at::ScalarType::Byte,
      "A/B must be uint8 (packed MXFP4)");
  TORCH_CHECK(
      A_sf.scalar_type() == at::ScalarType::Byte && B_sf.scalar_type() == at::ScalarType::Byte,
      "A_sf/B_sf must be uint8 (UE8M0)");
  TORCH_CHECK(
      D.scalar_type() == at::kFloat || D.scalar_type() == at::kBFloat16 || D.scalar_type() == at::kHalf,
      "D must be float32, bfloat16, or float16");
  TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && D.is_contiguous(), "A/B/D must be contiguous");

  constexpr int64_t kBlockSize = 32;
  const int64_t M = A.size(0);
  const int64_t N = B.size(0);
  const int64_t K = A.size(1) * 2;
  const int64_t scale_k = K / kBlockSize;

  // A 128-bit block-2D load of E2M1 covers 32 elements, which is also the scale
  // block, so one check covers both.
  TORCH_CHECK(K % kBlockSize == 0, "K (", K, ") must be a multiple of ", kBlockSize);
  TORCH_CHECK(A_sf.dim() == 2 && A_sf.size(0) == M && A_sf.size(1) == scale_k, "A_sf must be (M, K/32) row-major");
  TORCH_CHECK(B_sf.dim() == 2 && B_sf.size(0) == N && B_sf.size(1) == scale_k, "B_sf must be (N, K/32) row-major");

  const float* alpha_ptr = nullptr;
  if (alpha.has_value()) {
    const at::Tensor& a = alpha.value();
    TORCH_CHECK(
        a.is_xpu() && a.scalar_type() == at::kFloat && a.numel() == 1, "alpha must be a 1-element float32 XPU tensor");
    alpha_ptr = a.data_ptr<float>();
  }

  c10::DeviceGuard guard(A.device());
  auto stream = at::xpu::getCurrentXPUStream(A.device().index());
  sycl::queue& queue = stream.queue();

  at::Tensor SA = to_mn_major_scales(A_sf, M, scale_k);
  at::Tensor SB = to_mn_major_scales(B_sf, N, scale_k);

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = A.device().index();
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  cutlass::Status status;
  if (D.scalar_type() == at::kFloat) {
    status = run_mxfp4_scaled_gemm_dispatch<float>(A, B, SA, SB, alpha_ptr, D, hw_info, &queue);
  } else if (D.scalar_type() == at::kBFloat16) {
    status = run_mxfp4_scaled_gemm_dispatch<cutlass::bfloat16_t>(A, B, SA, SB, alpha_ptr, D, hw_info, &queue);
  } else {
    status = run_mxfp4_scaled_gemm_dispatch<cutlass::half_t>(A, B, SA, SB, alpha_ptr, D, hw_info, &queue);
  }

  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "cutlass_scaled_fp4_mm failed with status: " + std::string(cutlassGetStatusString(status)));
}

#undef SYCL_INTEL_TARGET
