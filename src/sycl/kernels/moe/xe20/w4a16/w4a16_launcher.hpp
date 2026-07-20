/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
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

// Type-erased entry point AOT-instantiated per policy, scale type, and
// activation type. ElementS is uint8_t for MXFP4 E8M0 scales; otherwise it
// matches ElementA for INT4 direct scales. ElementA is BF16 or FP16.
// Packed weights hold two 4-bit values per byte. For INT4, `zeros` selects
// unsigned codes with raw per-group zero-points; without it, codes are signed
// (for example, zero-point-folded). MXFP4 ignores `zeros`; bias is optional FP32.
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
