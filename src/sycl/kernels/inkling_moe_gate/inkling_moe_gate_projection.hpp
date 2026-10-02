/* Copyright 2026 SGLang Team. All Rights Reserved.
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#pragma once

#include <ATen/ATen.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_epilogue.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace inkling_moe_gate_impl {

using namespace cute;

constexpr int kHidden = 6144;
constexpr int kPaddedExperts = 264;
constexpr int kActiveExperts = 258;

template <typename Element>
struct MmaElement;

template <>
struct MmaElement<cutlass::bfloat16_t> {
  using type = cute::bfloat16_t;
};

template <>
struct MmaElement<cutlass::half_t> {
  using type = cute::half_t;
};

template <typename Element>
struct ProjectionTypes {
  using Accumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  // The model stores weight as row-major [N,K]. For GEMM B's logical [N,K]
  // view that is ColumnMajor in CUTLASS's A @ B^T convention.
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutD = cutlass::layout::RowMajor;
  using TileShape = Shape<_256, _64, _32>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _0>>;
  using AtomElement = typename MmaElement<Element>::type;
  using MmaAtom = MMA_Atom<XE_DPAS_TT<8, float, AtomElement>>;
  using TiledMma = typename TiledMMAHelper<MmaAtom, Layout<TileShape>, SubgroupLayout>::TiledMMA;
  using KernelSchedule = cutlass::gemm::KernelXeCooperative;
  using MainloopPolicy = cutlass::gemm::MainloopXeL1Staged<4, KernelSchedule>;
  using TileScheduler = cutlass::gemm::StreamKScheduler;
  using EpiloguePolicy = cutlass::epilogue::IntelXeGeneric;
  using EpilogueOp = cutlass::epilogue::fusion::
      LinearCombination<float, float, Accumulator, Accumulator, cutlass::FloatRoundStyle::round_to_nearest>;
  using FusionCallbacks = cutlass::epilogue::fusion::
      FusionCallbacks<EpiloguePolicy, EpilogueOp, TileShape, decltype(tile_shape(TiledMma()))>;
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
      EpiloguePolicy,
      TileShape,
      void,
      Accumulator,
      cutlass::gemm::TagToStrideC_t<LayoutD>,
      float,
      cutlass::gemm::TagToStrideC_t<LayoutD>,
      FusionCallbacks,
      void,
      void>;
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      MainloopPolicy,
      TileShape,
      Element,
      cutlass::gemm::TagToStrideA_t<LayoutA>,
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
  using Kernel = cutlass::gemm::kernel::
      GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, TileScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

template <typename Element>
inline void run_projection(const at::Tensor& input, const at::Tensor& weight, at::Tensor& output, sycl::queue& queue) {
  using Bundle = ProjectionTypes<Element>;
  using Kernel = typename Bundle::Kernel;
  using Gemm = typename Bundle::Gemm;
  using Problem = typename Kernel::ProblemShape;

  const int m = static_cast<int>(input.size(0));
  const int n = kPaddedExperts;
  const int k = kHidden;
  Problem problem{m, n, k, 1};
  auto stride_a = cutlass::make_cute_packed_stride(typename Kernel::StrideA{}, make_shape(m, k, 1));
  auto stride_b = cutlass::make_cute_packed_stride(typename Kernel::StrideB{}, make_shape(n, k, 1));
  auto stride_d = cutlass::make_cute_packed_stride(typename Kernel::StrideD{}, make_shape(m, n, 1));

  typename Kernel::MainloopArguments mainloop{
      reinterpret_cast<Element*>(input.data_ptr()), stride_a, reinterpret_cast<Element*>(weight.data_ptr()), stride_b};
  typename Kernel::EpilogueArguments epilogue{{1.0f, 0.0f}, nullptr, stride_d, output.data_ptr<float>(), stride_d};

  cutlass::KernelHardwareInfo hardware;
  hardware.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hardware.device_id);
  using Params = cutlass::gemm::kernel::detail::PersistentTileSchedulerXeStreamKParams;
  typename Kernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem,
      mainloop,
      epilogue,
      hardware,
      {1, Params::DecompositionMode::DataParallel}};

  Gemm gemm;
  auto status = gemm.can_implement(args);
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "Inkling gate projection can_implement failed: ",
      cutlassGetStatusString(status));
  auto workspace = at::empty({static_cast<int64_t>(Gemm::get_workspace_size(args))}, input.options().dtype(at::kByte));
  status = gemm.run(args, workspace.data_ptr(), &queue);
  TORCH_CHECK(
      status == cutlass::Status::kSuccess, "Inkling gate projection launch failed: ", cutlassGetStatusString(status));
}

void launch_projection_bf16(const at::Tensor&, const at::Tensor&, at::Tensor&, sycl::queue&);
void launch_projection_fp16(const at::Tensor&, const at::Tensor&, at::Tensor&, sycl::queue&);

}  // namespace inkling_moe_gate_impl
