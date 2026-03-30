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
/*!
  \file
  \brief Shared type definitions for MLA decode kernel instantiations
*/

#pragma once

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cute/tensor.hpp>
#include <sycl/sycl.hpp>

#include "../../Utils.h"
#include "mla_runner.hpp"
#include "mla_tile_scheduler.hpp"
#include "xe_mla_epilogue.hpp"
#include "xe_mla_kernel.hpp"
#include "xe_mla_mainloop.hpp"

using namespace cute;

//----------------- set persistent options --------------------//
template <bool v>
struct IsPersistent {
  static const bool value = v;
};

//----------------- set page size options --------------------//
template <int PageSize>
struct PageSizeOption {
  static constexpr int value = PageSize;
};

//----------------- set element type options --------------------//
template <typename T>
struct ToCutlassElementType {
  using type = T;
};

template <>
struct ToCutlassElementType<sycl::half> {
  using type = cutlass::half_t;
};

template <>
struct ToCutlassElementType<sycl::ext::oneapi::bfloat16> {
  using type = cutlass::bfloat16_t;
};

//----------------- define problem shape --------------------//
struct FMlAProblemShape {
  int batch = 0;
  int num_heads_q = 0;
  int num_heads_kv = 0;
  int seq_len_qo = 0;
  int seq_len_kv = 0;
  int head_size_q_nope = 0;
  int head_size_q_pe = 0;
  int head_size_kv = 0;
  int head_size_k_pe = 0;
  int head_size_o = 0;
  int page_size = 0;
  int total_page = 0;

  FMlAProblemShape() = default;
};

//----------------- define MLA Xe configuration --------------------//
template <typename T, typename PageSizeOpt = PageSizeOption<64>, typename PersistenceOption = IsPersistent<false>>
struct MlaXe {
  // TODO: add persistence option support in tile scheduler
  using TileScheduler = typename cutlass::flash_attention::kernel::XeMlaIndividualTileScheduler;

  static constexpr int PAGE_SIZE = PageSizeOpt::value;
  using KvTileSizeType = cute::Int<PAGE_SIZE>;

  static constexpr int NumSubgroupsN = PAGE_SIZE / 16;

  using TileShapeQK = Shape<_1, KvTileSizeType, _64>;
  using TileShapePV = Shape<_1, _64, KvTileSizeType>;
  using TileShapeOutput = Shape<_1, _512>;

  using SubgroupLayoutQK = Layout<Shape<_1, cute::Int<NumSubgroupsN>, _1>>;
  using SubgroupLayoutPV = decltype(cutlass::flash_attention::collective::get_sg_layout_pv(SubgroupLayoutQK{}));

  using ElementType = typename ToCutlassElementType<T>::type;
  using ElementQ = ElementType;
  using ElementK = ElementType;
  using ElementV = ElementType;
  using ElementO = ElementType;
  static constexpr int SGTileQ = get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})))();
  // TODO: handle special float8 types float_e5m2_t, float_e4m3_t for MMA operation
  using MMAOperation = XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, ElementQ>;

  using StrideQ = Stride<int, _1, int, int>;
  using StrideK = Stride<int, _1, int, int>;
  using StrideV = Stride<_1, int, int, int>;
  using StrideO = Stride<int, _1, int, int>;
  using GmemTiledCopyQ = void;
  using GmemTiledCopyK = void;
  using GmemTiledCopyV = void;
  using GmemTiledCopyO = void;

  using ProblemShape = FMlAProblemShape;

  using TiledMMAQK = typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<TileShapeQK>, SubgroupLayoutQK>::TiledMMA;
  using TiledMMAPV = typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<TileShapePV>, SubgroupLayoutPV>::TiledMMA;

  static_assert(
      get<0>(TileShapeOutput{}) == get<0>(TileShapePV{}),
      "Output tile and P*V tile have different sizes in Q dimension");
  static constexpr int VTiles = get<1>(TileShapeOutput{}) / get<1>(TileShapePV{});

  // Helper template function to create dummy tensor types
  template <typename Element, typename Stride>
  static auto make_dummy_tensor_type(Element val, Stride stride) {
    return make_tensor(make_gmem_ptr(&val), make_layout(repeat<rank_v<Stride>>(1), stride));
  }

  using TensorQ = decltype(make_dummy_tensor_type(ElementQ{}, StrideQ{}));
  using TensorK = decltype(make_dummy_tensor_type(ElementK{}, StrideK{}));
  using TensorV = decltype(make_dummy_tensor_type(ElementV{}, StrideV{}));
  using TensorO = decltype(make_dummy_tensor_type(ElementO{}, StrideO{}));

  // Collective Mainloop
  static constexpr int PipelineStages = 1;
  using MainloopDispatchPolicy = cutlass::flash_attention::XeDefault<PipelineStages>;
  using CollectiveMainloop = cutlass::flash_attention::collective::XeMlaMainloop<
      MainloopDispatchPolicy,
      true,
      TiledMMAQK,
      TiledMMAPV,
      VTiles,
      TensorQ,
      TensorK,
      TensorV,
      GmemTiledCopyQ,
      GmemTiledCopyK,
      GmemTiledCopyV>;

  // Collective Epilogue
  using CollectiveEpilogue =
      cutlass::flash_attention::collective::XeMlaEpilogue<CollectiveMainloop, TileShapeOutput, TensorO, GmemTiledCopyO>;

  // Kernel instantiation
  using FmlaKernel = cutlass::flash_attention::kernel::
      XeMlaFwdKernel<ProblemShape, CollectiveMainloop, CollectiveEpilogue, TileScheduler>;

  using Fmla = cutlass::flash_attention::device::MLA<FmlaKernel>;
};

template <typename T>
inline typename T::Fmla::Arguments args_from_options(
    at::Tensor const& out,
    at::Tensor const& q_nope,
    at::Tensor const& q_pe,
    at::Tensor const& kv_c_and_k_pe_cache,
    at::Tensor const& seq_lens,
    at::Tensor const& page_table,
    double sm_scale,
    int64_t num_kv_splits) {
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = q_nope.device().index();
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  // Extract dimensions from tensors
  // q_nope: (bs, num_heads, v_head_dim) where v_head_dim = 512 (d_latent)
  // q_pe:   (bs, num_heads, q_pe_dim)   where q_pe_dim = 64 (d_rope)
  // kv_cache: (num_blocks, block_size, head_dim) where head_dim = 576 (d_latent + d_rope)
  // out:    (bs, num_heads, v_head_dim)
  int batch = q_nope.size(0);
  int num_heads = q_nope.size(1);
  int v_head_dim = q_nope.size(2);
  int q_pe_dim = q_pe.size(2);
  int head_dim = kv_c_and_k_pe_cache.size(2);
  int page_size = kv_c_and_k_pe_cache.size(1);
  int page_count_per_seq = page_table.size(1);
  int max_seq_len = page_size * page_count_per_seq;
  int total_page = kv_c_and_k_pe_cache.size(0);

  FMlAProblemShape problem_shape;
  problem_shape.batch = batch;
  problem_shape.num_heads_q = num_heads;
  problem_shape.num_heads_kv = 1;
  problem_shape.seq_len_qo = 1;
  problem_shape.seq_len_kv = max_seq_len;
  problem_shape.head_size_q_nope = v_head_dim;
  problem_shape.head_size_q_pe = q_pe_dim;
  problem_shape.head_size_kv = v_head_dim;
  problem_shape.head_size_k_pe = q_pe_dim;
  problem_shape.head_size_o = v_head_dim;
  problem_shape.page_size = page_size;
  problem_shape.total_page = total_page;

  using StrideQ = typename T::StrideQ;
  using StrideK = typename T::StrideK;
  using StrideV = typename T::StrideV;
  using StrideO = typename T::StrideO;
  using ElementQ = typename T::ElementQ;
  using ElementK = typename T::ElementK;
  using ElementO = typename T::ElementO;

  StrideQ stride_Q_nope = cute::make_stride(
      static_cast<int>(batch * num_heads * v_head_dim),
      cute::_1{},
      static_cast<int>(q_nope.stride(1)),
      static_cast<int>(q_nope.stride(0)));

  StrideQ stride_Q_pe = cute::make_stride(
      static_cast<int>(batch * num_heads * q_pe_dim),
      cute::_1{},
      static_cast<int>(q_pe.stride(1)),
      static_cast<int>(q_pe.stride(0)));

  StrideK stride_K = cute::make_stride(
      static_cast<int>(kv_c_and_k_pe_cache.stride(1)),
      cute::_1{},
      static_cast<int>(kv_c_and_k_pe_cache.stride(0)),
      static_cast<int>(1));

  StrideV stride_V = cute::make_stride(
      cute::_1{},
      static_cast<int>(kv_c_and_k_pe_cache.stride(1)),
      static_cast<int>(kv_c_and_k_pe_cache.stride(0)),
      static_cast<int>(1));

  StrideO stride_O = cute::make_stride(
      static_cast<int>(batch * num_heads * v_head_dim),
      cute::_1{},
      static_cast<int>(out.stride(1)),
      static_cast<int>(out.stride(0)));

  typename T::Fmla::KernelArguments kernel_args{};
  kernel_args.shape = problem_shape;
  kernel_args.Q_nope = static_cast<const ElementQ*>(q_nope.data_ptr());
  kernel_args.dQ_nope = stride_Q_nope;
  kernel_args.Q_pe = static_cast<const ElementQ*>(q_pe.data_ptr());
  kernel_args.dQ_pe = stride_Q_pe;
  kernel_args.K = static_cast<const ElementK*>(kv_c_and_k_pe_cache.data_ptr());
  kernel_args.dK = stride_K;
  kernel_args.K_pe = static_cast<const ElementK*>(kv_c_and_k_pe_cache.data_ptr()) + v_head_dim;
  kernel_args.dK_pe = stride_K;
  kernel_args.dV = stride_V;
  kernel_args.O = static_cast<ElementO*>(out.data_ptr());
  kernel_args.dO = stride_O;
  kernel_args.seq_lens = static_cast<const int*>(seq_lens.data_ptr());

  typename T::CollectiveMainloop::Arguments mainloop_args{
      static_cast<float>(sm_scale),
      static_cast<const int*>(page_table.data_ptr()),
      page_size,
      total_page,
      page_count_per_seq};

  typename T::Fmla::Arguments arguments{kernel_args, mainloop_args, {}, hw_info};

  return arguments;
}

template <typename Element, typename PageSizeOpt>
inline void runMla(
    at::Tensor const& out,
    at::Tensor const& q_nope,
    at::Tensor const& q_pe,
    at::Tensor const& kv_c_and_k_pe_cache,
    at::Tensor const& seq_lens,
    at::Tensor const& page_table,
    at::Tensor const& workspace,
    double sm_scale,
    int64_t num_kv_splits) {
  using MlaXeType = MlaXe<Element, PageSizeOpt, IsPersistent<false>>;
  typename MlaXeType::Fmla fmla;
  auto arguments = args_from_options<MlaXeType>(
      out, q_nope, q_pe, kv_c_and_k_pe_cache, seq_lens, page_table, sm_scale, num_kv_splits);

  CUTLASS_CHECK(fmla.can_implement(arguments));

  CUTLASS_CHECK(fmla.run(arguments, workspace.data_ptr()));
}
