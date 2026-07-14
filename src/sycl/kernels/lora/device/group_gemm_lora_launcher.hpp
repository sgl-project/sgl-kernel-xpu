/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
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
    \brief Shared CUTLASS pointer-array grouped GEMM launcher core for LoRA forward. 
*/

#pragma once

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cute/tensor.hpp>
#include <sycl/sycl.hpp>
#include <type_traits>
#include <utility>
#include <vector>

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/xe_array_epilogue.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/half.h"
#include "cutlass/layout/matrix.h"
#include "sycl/SYCLHelpers.h"

namespace at::native::xpu {

//----------------- Shared device-side grouped-GEMM metadata ------------------//
//
// Per-segment problem sizes, element byte-offsets, and strides that the CUTLASS
// pointer-array grouped GEMM consumes. Built on device by a single SYCL kernel
// (one thread per segment) 
//
// For each segment s in [0, num_segments):
//   M_s = seg_indptr[s+1] - seg_indptr[s]
//
//   a_off[s] = seg_indptr[s]        * K * elem_bytes    (into input_x)
//   b_off[s] = weight_indices[s] * N * K * elem_bytes   (into weights)
//   d_off[s] = seg_indptr[s]        * N * elem_bytes    (into output)

struct GroupedGemmMeta {
  torch::Tensor problem_sizes;  // int32 [num_segments, 3]  (M_s, N, K), on device
  torch::Tensor stride_A;       // int64 [num_segments]     leading dim of A = K
  torch::Tensor stride_B;       // int64 [num_segments]     leading dim of B = K
  torch::Tensor stride_D;       // int64 [num_segments]     leading dim of D = N
  torch::Tensor a_off;          // int64 [num_segments]     byte offset into A per segment (device)
  torch::Tensor b_off;          // int64 [num_segments]     byte offset into B per segment (device)
  torch::Tensor d_off;          // int64 [num_segments]     byte offset into D per segment (device)
};

// One thread per segment: derive M_s / lora_id from the index tensors and write
// problem sizes, constant strides, and byte offsets straight into device memory.
struct BuildGroupedGemmMetaKernel {
  const int32_t* seg_indptr;      // [num_segments + 1]
  const int32_t* weight_indices;  // [num_segments]
  int32_t* problem_sizes;         // [num_segments * 3]
  int64_t* stride_A;              // [num_segments]
  int64_t* stride_B;              // [num_segments]
  int64_t* stride_D;              // [num_segments]
  int64_t* a_off;                 // [num_segments]
  int64_t* b_off;                 // [num_segments]
  int64_t* d_off;                 // [num_segments]
  int N;
  int K;
  int64_t elem_bytes;
  int num_segments;

  void operator()(sycl::nd_item<1> item) const {
    const int s = static_cast<int>(item.get_global_linear_id());
    if (s >= num_segments) {
      return;
    }
    const int32_t row_start = seg_indptr[s];
    const int32_t M_s = seg_indptr[s + 1] - row_start;
    const int32_t lora_id = weight_indices[s];

    problem_sizes[3 * s + 0] = M_s;
    problem_sizes[3 * s + 1] = N;
    problem_sizes[3 * s + 2] = K;

    // Strides in elements (leading dim of A/B = K, D = N).
    stride_A[s] = static_cast<int64_t>(K);
    stride_B[s] = static_cast<int64_t>(K);
    stride_D[s] = static_cast<int64_t>(N);

    a_off[s] = static_cast<int64_t>(row_start) * K * elem_bytes;
    b_off[s] = static_cast<int64_t>(lora_id) * static_cast<int64_t>(N) * K * elem_bytes;
    d_off[s] = static_cast<int64_t>(row_start) * N * elem_bytes;
  }
};

// One thread per segment: turn a base address + per-segment byte offset into an
// absolute device pointer for the pointer-array grouped GEMM.
struct MakeDevicePtrsKernel {
  int64_t base_addr;
  const int64_t* off_bytes;  // [num_segments]
  int64_t* ptrs;             // [num_segments]
  int num_segments;

  void operator()(sycl::nd_item<1> item) const {
    const int s = static_cast<int>(item.get_global_linear_id());
    if (s >= num_segments) {
      return;
    }
    ptrs[s] = base_addr + off_bytes[s];
  }
};

// Round num_segments up to a whole number of work-groups of `wg` threads.
template <typename Kernel>
inline void submit_per_segment(sycl::queue& queue, int num_segments, Kernel kernel) {
  constexpr int wg = 256;
  const int64_t global = (static_cast<int64_t>(num_segments) + wg - 1) / wg * wg;
  sycl_kernel_submit(sycl::range<1>(global), sycl::range<1>(wg), queue, kernel);
}

inline GroupedGemmMeta build_grouped_gemm_meta(
    const torch::Tensor& seg_indptr_i32,      // int32 [num_segments + 1]
    const torch::Tensor& weight_indices_i32,  // int32 [num_segments]
    const int N,
    const int K,
    const int num_segments,
    const int64_t elem_bytes,
    const at::Device device,
    sycl::queue& queue) {
  auto opt_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
  auto opt_i64 = torch::TensorOptions().dtype(torch::kInt64).device(device);

  GroupedGemmMeta meta;
  meta.problem_sizes = torch::empty({num_segments, 3}, opt_i32);
  meta.stride_A = torch::empty({num_segments}, opt_i64);
  meta.stride_B = torch::empty({num_segments}, opt_i64);
  meta.stride_D = torch::empty({num_segments}, opt_i64);
  meta.a_off = torch::empty({num_segments}, opt_i64);
  meta.b_off = torch::empty({num_segments}, opt_i64);
  meta.d_off = torch::empty({num_segments}, opt_i64);

  BuildGroupedGemmMetaKernel kernel{
      seg_indptr_i32.data_ptr<int32_t>(),
      weight_indices_i32.data_ptr<int32_t>(),
      meta.problem_sizes.data_ptr<int32_t>(),
      meta.stride_A.data_ptr<int64_t>(),
      meta.stride_B.data_ptr<int64_t>(),
      meta.stride_D.data_ptr<int64_t>(),
      meta.a_off.data_ptr<int64_t>(),
      meta.b_off.data_ptr<int64_t>(),
      meta.d_off.data_ptr<int64_t>(),
      N,
      K,
      elem_bytes,
      num_segments};
  submit_per_segment(queue, num_segments, kernel);
  return meta;
}

// Turn a base tensor + device byte-offsets into a device int64 pointer array
// (one absolute device address per segment) for the pointer-array grouped GEMM.
inline torch::Tensor make_device_ptrs(const torch::Tensor& base, const torch::Tensor& off_bytes, sycl::queue& queue) {
  const int64_t base_addr = reinterpret_cast<int64_t>(base.data_ptr());
  const int num_segments = static_cast<int>(off_bytes.numel());
  auto ptrs = torch::empty({num_segments}, off_bytes.options());

  MakeDevicePtrsKernel kernel{base_addr, off_bytes.data_ptr<int64_t>(), ptrs.data_ptr<int64_t>(), num_segments};
  submit_per_segment(queue, num_segments, kernel);
  return ptrs;
}

template <typename T>
struct ToCutlassElementType {
  using type = T;  // float -> float (identity)
};

template <>
struct ToCutlassElementType<at::Half> {
  using type = cutlass::half_t;
};

template <>
struct ToCutlassElementType<at::BFloat16> {
  using type = cutlass::bfloat16_t;
};

//----------------- Group problem shape (shared across all instantiations) ----//
using GroupedProblemShape = cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;
using UnderlyingProblemShapeType = typename GroupedProblemShape::UnderlyingProblemShape;

template <typename TensorDType, typename TileShape_, typename ThreadLayout_, typename LayoutB_, int PipelineStages_>
void launch_group_gemm_lora_fwd(
    sycl::queue& queue,
    const torch::Tensor& problem_sizes,  // int32  [num_segments, 3]  (M_s, N, K)
    const torch::Tensor& a_ptrs,         // int64  [num_segments]     pointers into input_x
    const torch::Tensor& b_ptrs,         // int64  [num_segments]     pointers into weights[weight_indices[s]]
    const torch::Tensor& c_ptrs,         // int64  [num_segments]     pointers into C (source for residual add).
                                  //                          For A-fwd (beta=0) the caller passes d_ptrs here -- the
                                  //                          loads are inert. For B-fwd (beta!=0) the caller passes the
                                  //                          real residual source (typically = d_ptrs for in-place).
    const torch::Tensor& d_ptrs,    // int64  [num_segments]     pointers into output
    const torch::Tensor& stride_A,  // int64  [num_segments]     leading dim of A = K
    const torch::Tensor& stride_B,  // int64  [num_segments]     leading dim of B = K
    const torch::Tensor&
        stride_C,  // int64  [num_segments]     leading dim of C (must equal stride_D when LayoutC == LayoutD)
    const torch::Tensor& stride_D,  // int64  [num_segments]     leading dim of D = N
    int num_segments,
    float alpha,   // epilogue scalar: D = alpha * (A @ B) + beta * C
    float beta) {  //   A-fwd uses (1.0, 0.0); B-fwd / QKV-B-fwd use (1.0, 1.0) for in-place residual.
  using ElementA = typename ToCutlassElementType<TensorDType>::type;
  using ElementB = ElementA;             // same storage dtype as A (LoRA weights match input dtype)
  using ElementOutput = ElementA;        // output matches input dtype (bf16/fp16/fp32)
  using ElementAccumulator = float;      // XMX accumulates in fp32 -- required by every atom we support
  using ElementComputeEpilogue = float;  // alpha/beta arithmetic in fp32

  using ElementMma = ElementA;
  using ElementMmaB = ElementMma;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = LayoutB_;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  //--------- Tile / thread layout (host-supplied perf knobs) ---------//
  using TileShape = TileShape_;
  using ThreadLayout = ThreadLayout_;

  //--------- TiledMma (which XMX MMA atom + how many subgroups tile it) ---//
  using MmaAtom = cute::XE_DPAS_TT<8, ElementAccumulator, ElementMma>;
  using TiledMma =
      typename cute::TiledMMAHelper<cute::MMA_Atom<MmaAtom>, cute::Layout<TileShape>, ThreadLayout>::TiledMMA;

  //--------- Dispatch policies ---------//
  // MainloopXeL1StagedGroup builds
  // prefetch through make_block_2d_prefetch()/XE_PREFETCH_2D
  constexpr int PipelineStages = PipelineStages_;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopXeL1StagedGroup<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGenericGroup;

  //--------- Gmem copy atoms ---------//
  // void => CUTLASS auto-selects the correct 2D load/store/prefetch ops per
  // dtype and layout (see get_block_2d_copy_A/B and make_block_2d_prefetch).
  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;
  using GmemTiledCopyC = void;
  using GmemTiledCopyD = void;

  //--------- Epilogue: D = alpha * acc + beta * C ---------//
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementAccumulator,  // <- must be ElementAccumulator for Grouped GEMM
      ElementComputeEpilogue,
      ElementAccumulator,
      ElementAccumulator,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using FusionCallbacks = cutlass::epilogue::fusion::
      FusionCallbacks<EpilogueDispatchPolicy, EpilogueOp, TileShape, decltype(cute::tile_shape(TiledMma()))>;

  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
      EpilogueDispatchPolicy,
      TileShape,
      void,  // EpilogueTile (void = automatic)
      ElementAccumulator,
      cutlass::gemm::TagToStrideC_t<LayoutC*>,
      ElementOutput,  // bf16/fp16 narrow here; fp32 stores as-is
      cutlass::gemm::TagToStrideC_t<LayoutD*>,
      FusionCallbacks,
      GmemTiledCopyC,   // load C (void = automatic)
      GmemTiledCopyD>;  // store D (void = automatic)

  //--------- Mainloop ---------//
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      GEMMDispatchPolicy,
      TileShape,
      ElementMma,
      cutlass::gemm::TagToStrideA_t<LayoutA*>,
      ElementMmaB,
      cutlass::gemm::TagToStrideB_t<LayoutB*>,
      TiledMma,
      GmemTiledCopyA,
      void,
      void,
      cute::identity,
      GmemTiledCopyB,
      void,
      void,
      cute::identity>;

  //--------- GemmKernel + adapter ---------//

  using GemmKernel = cutlass::gemm::kernel::
      GemmUniversal<GroupedProblemShape, CollectiveMainloop, CollectiveEpilogue, cutlass::gemm::GroupScheduler>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;

  //--------- Step 3: build Arguments + workspace, then run ----------------//
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = static_cast<int>(d_ptrs.device().index());
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  auto* problem_sizes_ptr = reinterpret_cast<UnderlyingProblemShapeType*>(problem_sizes.data_ptr<int32_t>());
  auto* a_ptrs_dev = reinterpret_cast<ElementMma const**>(a_ptrs.data_ptr());
  auto* b_ptrs_dev = reinterpret_cast<ElementMmaB const**>(b_ptrs.data_ptr());
  auto* d_ptrs_dev = reinterpret_cast<ElementOutput**>(d_ptrs.data_ptr());
  auto* stride_A_ptr = reinterpret_cast<StrideA*>(stride_A.data_ptr<int64_t>());
  auto* stride_B_ptr = reinterpret_cast<StrideB*>(stride_B.data_ptr<int64_t>());
  auto* stride_C_ptr = reinterpret_cast<StrideC*>(stride_C.data_ptr<int64_t>());
  auto* stride_D_ptr = reinterpret_cast<StrideD*>(stride_D.data_ptr<int64_t>());

  // Epilogue thread arguments: D = alpha * acc + beta * C.
  typename Gemm::Arguments arguments;
  decltype(arguments.epilogue.thread) fusion_args;
  fusion_args.alpha = alpha;
  fusion_args.beta = beta;
  fusion_args.alpha_ptr = nullptr;
  fusion_args.beta_ptr = nullptr;
  fusion_args.alpha_ptr_array = nullptr;
  fusion_args.beta_ptr_array = nullptr;
  fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
  fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};

  using RasterOrderOptions =
      typename cutlass::gemm::kernel::detail::PersistentTileSchedulerXeGroup<GroupedProblemShape>::RasterOrderOptions;

  using PtrCType = decltype(std::declval<typename GemmKernel::EpilogueArguments>().ptr_C);
  auto c_ptrs_dev = reinterpret_cast<PtrCType>(c_ptrs.data_ptr());
  typename GemmKernel::Arguments gemm_args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      typename GemmKernel::ProblemShape{num_segments, problem_sizes_ptr, nullptr},
      typename GemmKernel::MainloopArguments{a_ptrs_dev, stride_A_ptr, b_ptrs_dev, stride_B_ptr},
      typename GemmKernel::EpilogueArguments{
          fusion_args,
          c_ptrs_dev,  // ptr_C (caller-supplied; aliased to d_ptrs for beta=0 paths)
          stride_C_ptr,
          d_ptrs_dev,  // ptr_D
          stride_D_ptr},
      hw_info,
      typename GemmKernel::TileSchedulerArguments{1, RasterOrderOptions::AlongN}};

  Gemm gemm_op;

  cutlass::Status status = gemm_op.can_implement(gemm_args);
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "launch_group_gemm_lora_fwd: can_implement failed: status=",
      cutlassGetStatusString(status));

  size_t workspace_size = Gemm::get_workspace_size(gemm_args);
  auto workspace_opts = torch::TensorOptions().dtype(torch::kUInt8).device(d_ptrs.device());
  auto workspace = torch::empty({static_cast<int64_t>(workspace_size)}, workspace_opts);

  status = gemm_op.initialize(gemm_args, workspace.data_ptr());
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "launch_group_gemm_lora_fwd: initialize failed: status=",
      cutlassGetStatusString(status));

  status = gemm_op.run(&queue);
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "launch_group_gemm_lora_fwd: run failed: status=",
      cutlassGetStatusString(status));
}

}  // namespace at::native::xpu
