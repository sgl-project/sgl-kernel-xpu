/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Host-side launcher for the unified int4/mxfp4 W4A16 MoE grouped GEMM.
 * Ported from vllm-xpu-kernels
 * (csrc/xpu/grouped_gemm/xe_2/grouped_gemm_xe2_interface.hpp MoEGEMMLauncher),
 * adapted to sgl-kernel-xpu's AOT instantiation scheme. The device kernel is
 * moe_w4a16::MoEGEMM in grouped_gemm_xe2.hpp.
 **************************************************************************************************/
#pragma once

#include <torch/all.h>

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include "gemm_xe2_policy.hpp"
#include "grouped_gemm_xe2.hpp"

namespace moe_w4a16 {
using namespace cute;

// Unique type tag for naming the SYCL kernel per instantiation.
template <typename, typename, typename, typename, bool, char, char, class>
class GemmCuteName;

template <
    char layoutA,
    char layoutB,
    bool HasZero,
    class policy,
    typename ElementA,
    typename ElementB,
    typename ElementS,
    typename ElementBI,
    typename ElementD>
void MoEGEMMLauncher(
    sycl::queue& stream,
    const ElementA* activations,
    const ElementB* weights,
    const ElementS* scales,
    const ElementS* zeros,
    const ElementBI* bias,
    ElementD* outputs,
    const int gemm_n,
    const int gemm_k,
    const int* rows_per_expert,
    const int num_experts,
    const int group_size,
    int32_t* atomic_buffer) {
  using ElementA_non_CV = cutlass::platform::remove_cv_t<ElementA>;
  auto op = XE_DPAS_TT<8, float, ElementA_non_CV>{};

  using WGTile = typename policy::WGTile;
  using SGLayout = typename policy::SGLayout;
  using MMA = typename TiledMMAHelper<MMA_Atom<decltype(op)>, Layout<WGTile>, SGLayout>::TiledMMA;
  auto mma = MMA{};

  int sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  auto MaxThreadsPerWorkgroup = size(mma);

  static constexpr int MaxThreadsPerSM = 512;
  TORCH_CHECK(
      MaxThreadsPerSM % MaxThreadsPerWorkgroup == 0, "MaxThreadsPerSM must be divisible by MaxThreadsPerWorkgroup");

  sycl::range<3> local(1, 1, MaxThreadsPerWorkgroup);
  sycl::range<3> global(1, sm_count * MaxThreadsPerSM / MaxThreadsPerWorkgroup, 1);

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  syclex::properties kernel_props{syclex::sub_group_size<16>, intelex::grf_size<256>};

  using GmemTiledCopyA = typename policy::GmemTiledCopyA;
  using GmemTiledCopyB = typename policy::GmemTiledCopyB;
  using GmemTiledCopyD = typename policy::GmemTiledCopyD;

  stream.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<int32_t, 1> local_mem(sycl::range<1>(1), cgh);
    cgh.parallel_for<GemmCuteName<ElementA, ElementB, ElementS, ElementD, HasZero, layoutA, layoutB, policy>>(
        sycl::nd_range<3>{global * local, local}, kernel_props, [=](auto) {
      moe_w4a16::MoEGEMM<GmemTiledCopyA, GmemTiledCopyB, GmemTiledCopyD, layoutA, layoutB, 'R', HasZero>(
              activations,
              weights,
              scales,
              zeros,
              bias,
              outputs,
              mma,
              rows_per_expert,
              num_experts,
              group_size,
              gemm_n,
              gemm_k,
              atomic_buffer,
              local_mem);
        });
  });
}

// Type-erased entry point AOT-instantiated per (policy, ElementS, ElementA).
// ElementS selects the quantization flavour: bfloat16_t => int4 (scale is a
// direct multiplier), uint8_t => mxfp4 (scale is an E8M0 exponent decoded
// in-kernel). ElementA selects the activation/output dtype: bfloat16_t or
// half_t (fp16), mirroring vllm-xpu-kernels' A_dtype runtime dispatch.
// Packed weights are uint8 (two 4-bit values per byte, unsigned/raw codes
// for int4 -- no folded zero-point), bias is optional float32 (nullptr when
// absent). `zeros` is only meaningful for int4 (same [E, N, K/group_size]
// layout/dtype as `scales`, holding the raw per-group zero-point in code
// units); pass nullptr for mxfp4 or for int4 checkpoints with no separate
// zero-point (e.g. symmetric quantization).
template <typename Policy, typename ElementS, typename ElementA>
void w4a16_launch(
    sycl::queue stream,
    const void* activations,
    const void* packed_weights,
    const void* scales,
    const void* zeros,
    const void* bias,
    void* outputs,
    const int gemm_n,
    const int gemm_k,
    const int* rows_per_expert,
    const int num_experts,
    const int group_size,
    int32_t* atomic_buffer) {
  auto launch = [&](auto has_zero) {
    constexpr bool HasZero = decltype(has_zero)::value;
    MoEGEMMLauncher<'R', 'C', HasZero, Policy>(
        stream,
        static_cast<const ElementA*>(activations),
        static_cast<const uint8_t*>(packed_weights),
        static_cast<const ElementS*>(scales),
        static_cast<const ElementS*>(zeros),
        static_cast<const float*>(bias),
        static_cast<ElementA*>(outputs),
        gemm_n,
        gemm_k,
        rows_per_expert,
        num_experts,
        group_size,
        atomic_buffer);
  };

  if constexpr (std::is_same_v<ElementS, uint8_t>) {
    launch(std::false_type{});
  } else if (zeros != nullptr) {
    launch(std::true_type{});
  } else {
    launch(std::false_type{});
  }
}

}  // namespace moe_w4a16
