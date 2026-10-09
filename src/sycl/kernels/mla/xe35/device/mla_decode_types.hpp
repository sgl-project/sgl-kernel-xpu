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

#include <cmath>
#include <cute/tensor.hpp>
#include <optional>
#include <sycl/sycl.hpp>

#include "../../../../Utils.h"
#include "sycl/kernels/mla/xe35/collective/xe_mla_epilogue.hpp"
#include "sycl/kernels/mla/xe35/collective/xe_mla_mainloop.hpp"
#include "sycl/kernels/mla/xe35/device/mla_runner.hpp"
#include "sycl/kernels/mla/xe35/kernel/mla_tile_scheduler.hpp"
#include "sycl/kernels/mla/xe35/kernel/xe_mla_kernel.hpp"

using namespace cute;

//----------------- set splitkv options --------------------//
template <bool v>
struct EnabledSplitKV {
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

//----------------- decode Q-tile (bucket) selection --------------------//
// Largest instantiated Q tile: MlaXe::kMaxSubgroups (8) subgroups x 2 query
// rows each. Must agree with MLA_DECODE_Q_TILES in MlaDecodeXe{20,35}.cmake.
inline constexpr int kMlaDecodeMaxQTile = 16;

// Q-tile instantiation for a request with seq_len_q query tokens. seq_len_q is
// rounded up to the next instantiated tile (1, 2, 4, 8, 16); the padding rows
// read zeros, are masked out of the softmax and are never stored (the Xe 2D
// block copies clamp to the tensor's real row count). Above kMlaDecodeMaxQTile
// the 16-row kernel is launched with ceil(seq_len_q / 16) Q tiles, the last of
// which may be partial in the same way. Returns the tile for any seq_len_q >= 1.
inline int mla_decode_q_tile(int64_t seq_len_q) {
  int tile = 1;
  while (tile < seq_len_q && tile < kMlaDecodeMaxQTile)
    tile *= 2;
  return tile;
}

//----------------- define MLA Xe configuration --------------------//
// QTileM is the Q-tile height, mla_decode_q_tile(seq_len_q). It is >= the
// request's seq_len_q for seq_len_q <= 16 (padding rows are masked) and 16 with
// several Q tiles per request beyond that.
//   1:  single-token decode. One Q row per subgroup (DPAS M=1); the PAGE_SIZE/16
//       subgroups split the KV tile along N; 128-GRF launch. Unchanged path.
//   >1: multi-token (MTP / speculative) decode. Two Q rows per subgroup (DPAS
//       M=2) so each K/V fragment is shared by both rows. Subgroups are laid out
//       (Q_TILE_M/2) x NumSubgroupsN with the product capped at kMaxSubgroups;
//       when the cap shrinks NumSubgroupsN below PAGE_SIZE/16 the KV tile is a
//       sub-page (16 * NumSubgroupsN) and the mainloop walks several tiles per
//       page. Each subgroup thus keeps the same 16-column K/V footprint as the
//       single-token kernel and only the O accumulator doubles -> 256-GRF
//       launch (device::MLA::kGrfSize). The s_q tokens are the last s_q
//       positions of the sequence and are causally masked among themselves.
template <
    typename T,
    typename PageSizeOpt = PageSizeOption<64>,
    typename SplitKVOption = EnabledSplitKV<false>,
    int QTileM = 1>
struct MlaXe {
  // TODO: add persistence option support in tile scheduler
  using TileScheduler = typename cutlass::flash_attention::kernel::XeMlaIndividualTileScheduler;

  static constexpr int PAGE_SIZE = PageSizeOpt::value;

  static constexpr int Q_TILE_M = QTileM;
  static_assert(Q_TILE_M >= 1 && Q_TILE_M <= kMlaDecodeMaxQTile, "decode Q tile out of range");

  static constexpr int RowsPerSubgroup = (Q_TILE_M == 1) ? 1 : 2;
  static_assert(Q_TILE_M % RowsPerSubgroup == 0, "Q tile must be a whole number of subgroup rows");
  static constexpr int NumSubgroupsM = Q_TILE_M / RowsPerSubgroup;
  // Work-group cap: 8 subgroups (128 work-items), the single-token kernel's
  // size at PAGE_SIZE=128. Keeps the epilogue's cross-subgroup SLM reduction
  // buffer at <= 8 * (2 x 512) fp32 = 32 KB.
  static constexpr int kMaxSubgroups = 8;
  static constexpr int NumSubgroupsNCap = kMaxSubgroups / NumSubgroupsM;
  static constexpr int NumSubgroupsN = (PAGE_SIZE / 16 < NumSubgroupsNCap) ? PAGE_SIZE / 16 : NumSubgroupsNCap;
  static_assert(NumSubgroupsN >= 1, "at least one subgroup along KV");
  // Each subgroup owns 16 KV columns; the KV tile is a whole page for the
  // single-token kernel and a page divisor for the multi-token variants.
  static constexpr int KV_TILE = NumSubgroupsN * 16;
  static_assert(PAGE_SIZE % KV_TILE == 0, "KV tile must evenly divide the page");

  using QTileSizeType = cute::Int<Q_TILE_M>;
  using KvTileSizeType = cute::Int<KV_TILE>;

  using TileShapeQK = Shape<QTileSizeType, KvTileSizeType, _64>;
  using TileShapePV = Shape<QTileSizeType, _64, KvTileSizeType>;
  using TileShapeOutput = Shape<QTileSizeType, _512>;

  using SubgroupLayoutQK = Layout<Shape<cute::Int<NumSubgroupsM>, cute::Int<NumSubgroupsN>, _1>>;
  using SubgroupLayoutPV = decltype(cutlass::flash_attention::collective::get_sg_layout_pv(SubgroupLayoutQK{}));

  using ElementType = typename ToCutlassElementType<T>::type;
  using ElementQ = ElementType;
  using ElementK = ElementType;
  using ElementV = ElementType;
  using ElementO = ElementType;
  static constexpr int SGTileQ = get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})))();
  // TODO: handle special float8 types float_e5m2_t, float_e4m3_t for MMA operation
  using MMAOperation = XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, ElementQ>;

  using ElementLSE = float;  // Softmax LSE, log2 domain

  using StrideQ = Stride<int, _1, int, int>;
  using StrideK = Stride<int, _1, int, int>;
  using StrideV = Stride<_1, int, int, int>;
  using StrideO = Stride<int, _1, int, int>;
  using StrideLSE = Stride<int, _1, int>;
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
  using TensorLSE = decltype(make_dummy_tensor_type(ElementLSE{}, StrideLSE{}));

  // Single-token decode attends to all past KV (no mask). Multi-token decode
  // masks the s_q new tokens among themselves: row q sees k <= seq_len_kv - s_q + q.
  static constexpr bool CausalMask = (Q_TILE_M > 1);

  // Collective Mainloop
  static constexpr int PipelineStages = 1;
  using MainloopDispatchPolicy = cutlass::flash_attention::XeDefault<PipelineStages>;
  using CollectiveMainloop = cutlass::flash_attention::collective::XeMlaMainloop<
      MainloopDispatchPolicy,
      CausalMask,
      TiledMMAQK,
      TiledMMAPV,
      VTiles,
      TensorQ,
      TensorK,
      TensorV,
      GmemTiledCopyQ,
      GmemTiledCopyK,
      GmemTiledCopyV,
      false>;  // IsPrefill: decode launches at 128 GRF for Q_TILE_M=1 (small
               // footprint; doubles thread/EU occupancy for memory-bound
               // decode) and 256 GRF for Q_TILE_M>1. Prefill defaults to
               // true → 256 GRF. See device::MLA::kGrfSize.

  // Collective Epilogue
  using CollectiveEpilogue = cutlass::flash_attention::collective::
      XeMlaEpilogue<CollectiveMainloop, TileShapeOutput, TensorO, GmemTiledCopyO, TensorLSE>;

  // Kernel instantiation
  static constexpr bool is_split_kv = SplitKVOption::value;
  using FmlaKernel = typename cute::conditional_t<
      SplitKVOption::value,
      cutlass::flash_attention::kernel::
          XeMlaSplitKVKernel<ProblemShape, CollectiveMainloop, CollectiveEpilogue, TileScheduler>,
      cutlass::flash_attention::kernel::
          XeMlaFwdKernel<ProblemShape, CollectiveMainloop, CollectiveEpilogue, TileScheduler>>;

  using Fmla = cutlass::flash_attention::device::MLA<FmlaKernel>;
};

template <typename T>
inline typename T::Fmla::Arguments args_from_options(
    at::Tensor& out,
    std::optional<at::Tensor> const& lse,
    at::Tensor const& q_nope,
    at::Tensor const& q_pe,
    at::Tensor const& kv_c_and_k_pe_cache,
    at::Tensor const& seq_lens,
    at::Tensor const& page_table,
    at::Tensor& workspace,
    double sm_scale,
    int64_t num_kv_splits) {
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = q_nope.device().index();
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  // Extract dimensions from tensors
  // q_nope: (bs, [s_q,] num_heads, v_head_dim) where v_head_dim = 512 (d_latent)
  // q_pe:   (bs, [s_q,] num_heads, q_pe_dim)   where q_pe_dim = 64 (d_rope)
  // kv_cache: (num_blocks, block_size, head_dim) where head_dim = 576 (d_latent + d_rope)
  // out:    (bs, [s_q,] num_heads, v_head_dim)
  // lse:    (bs, [s_q,] num_heads)
  // 3D tensors are single-token decode (s_q == 1); 4D carry any s_q >= 1, with
  // runMla having checked that T::Q_TILE_M is the tile mla_decode_q_tile picks
  // for it. Dims and strides are indexed from the back, which is valid for both
  // ranks.
  int batch = q_nope.size(0);
  int seq_len_q = q_nope.dim() == 4 ? q_nope.size(1) : 1;
  int num_heads = q_nope.size(-2);
  int v_head_dim = q_nope.size(-1);
  int q_pe_dim = q_pe.size(-1);
  int head_dim = kv_c_and_k_pe_cache.size(2);
  int page_size = kv_c_and_k_pe_cache.size(1);
  int page_count_per_seq = page_table.size(1);
  int max_seq_len = page_size * page_count_per_seq;
  int total_page = kv_c_and_k_pe_cache.size(0);

  FMlAProblemShape problem_shape;
  problem_shape.batch = batch;
  problem_shape.num_heads_q = num_heads;
  problem_shape.num_heads_kv = 1;
  // The real token count, not the tile: it sizes the Q/O/LSE tensors (so the 2D
  // block copies clamp padding rows), the causal offset and the Q-tile grid.
  problem_shape.seq_len_qo = seq_len_q;
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
  using StrideLSE = typename T::StrideLSE;
  using ElementQ = typename T::ElementQ;
  using ElementK = typename T::ElementK;
  using ElementO = typename T::ElementO;
  using ElementLSE = typename T::ElementLSE;

  // Kernel strides are (seq_q, dim, head, batch). stride(-3) is the s_q stride
  // of a 4D tensor; on a 3D tensor it is the batch stride, which is harmless
  // because seq_q == 1 there and the stride is never dereferenced.
  StrideQ stride_Q_nope = cute::make_stride(
      static_cast<int>(q_nope.stride(-3)),
      cute::_1{},
      static_cast<int>(q_nope.stride(-2)),
      static_cast<int>(q_nope.stride(0)));

  StrideQ stride_Q_pe = cute::make_stride(
      static_cast<int>(q_pe.stride(-3)),
      cute::_1{},
      static_cast<int>(q_pe.stride(-2)),
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
      static_cast<int>(out.stride(-3)), cute::_1{}, static_cast<int>(out.stride(-2)), static_cast<int>(out.stride(0)));

  // lse: (batch, [s_q,] num_heads) -> kernel layout (seq_len_qo, num_heads, batch)
  StrideLSE stride_LSE = cute::make_stride(
      static_cast<int>(lse.has_value() ? lse->stride(-2) : 0),
      cute::_1{},
      static_cast<int>(lse.has_value() ? lse->stride(0) : 0));

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
  if (lse.has_value()) {
    kernel_args.LSE = static_cast<ElementLSE*>(lse->data_ptr());
  }
  kernel_args.dLSE_out = stride_LSE;
  kernel_args.seq_lens = static_cast<const int*>(seq_lens.data_ptr());

  if constexpr (T::is_split_kv) {
    using cutlass::flash_attention::kernel::SplitKVWorkspaceLayout;
    SplitKVWorkspaceLayout ws(
        batch, problem_shape.num_heads_q, num_kv_splits, problem_shape.head_size_o, sizeof(ElementO), T::Q_TILE_M);

    auto* ws_ptr = static_cast<char*>(workspace.data_ptr());
    auto* o_accum_ptr = reinterpret_cast<ElementO*>(ws_ptr + ws.o_accum_offset);
    auto* exp_sums_ptr = reinterpret_cast<float*>(ws_ptr + ws.exp_sums_offset);
    auto* max_logits_ptr = reinterpret_cast<float*>(ws_ptr + ws.max_logits_offset);

    // Workspace layouts are [s_q][batch][head][split][...] (see SplitKVWorkspaceLayout).
    StrideO stride_O_accum = cute::make_stride(
        static_cast<int>(batch * problem_shape.num_heads_q * num_kv_splits * problem_shape.head_size_o),
        cute::_1{},
        static_cast<int>(problem_shape.head_size_o),
        static_cast<int>(problem_shape.num_heads_q * num_kv_splits * problem_shape.head_size_o));
    StrideO stride_lse = cute::make_stride(
        static_cast<int>(problem_shape.batch * problem_shape.num_heads_q * num_kv_splits),
        cute::_1{},
        static_cast<int>(num_kv_splits),
        static_cast<int>(problem_shape.num_heads_q * num_kv_splits));
    kernel_args.O_accum = o_accum_ptr;
    kernel_args.dO_accum = stride_O_accum;
    kernel_args.exp_sums = exp_sums_ptr;
    kernel_args.max_logits = max_logits_ptr;
    kernel_args.dLSE = stride_lse;
  }
  typename T::CollectiveMainloop::Arguments mainloop_args{
      static_cast<float>(sm_scale),
      static_cast<const int*>(page_table.data_ptr()),
      page_size,
      total_page,
      page_count_per_seq};

  typename T::Fmla::Arguments arguments{kernel_args, mainloop_args, {}, hw_info, static_cast<int>(num_kv_splits)};

  return arguments;
}

template <typename Element, typename PageSizeOpt, typename SplitKVOpt, int QTileM>
inline void runMlaImpl(
    at::Tensor& out,
    std::optional<at::Tensor> const& lse,
    at::Tensor const& q_nope,
    at::Tensor const& q_pe,
    at::Tensor const& kv_c_and_k_pe_cache,
    at::Tensor const& seq_lens,
    at::Tensor const& page_table,
    at::Tensor& workspace,
    double sm_scale,
    int64_t num_kv_splits) {
  using MlaXeType = MlaXe<Element, PageSizeOpt, SplitKVOpt, QTileM>;
  typename MlaXeType::Fmla fmla;
  auto arguments = args_from_options<MlaXeType>(
      out, lse, q_nope, q_pe, kv_c_and_k_pe_cache, seq_lens, page_table, workspace, sm_scale, num_kv_splits);

  CUTLASS_CHECK(fmla.can_implement(arguments));

  CUTLASS_CHECK(fmla.run(arguments, workspace.data_ptr()));
}

// QTileM must be mla_decode_q_tile(seq_len_q) for the input's seq_len_q (1 for
// 3D q, q.size(1) for 4D q); the host dispatcher (mla_decode.cpp) selects it.
template <typename Element, typename PageSizeOpt, int QTileM = 1>
inline void runMla(
    at::Tensor& out,
    std::optional<at::Tensor> const& lse,
    at::Tensor const& q_nope,
    at::Tensor const& q_pe,
    at::Tensor const& kv_c_and_k_pe_cache,
    at::Tensor const& seq_lens,
    at::Tensor const& page_table,
    at::Tensor& workspace,
    double sm_scale,
    int64_t num_kv_splits) {
  TORCH_CHECK(num_kv_splits >= 1, "num_kv_splits must be resolved before calling runMla, got ", num_kv_splits);

  const bool is_4d = q_nope.dim() == 4;
  const int seq_len_q = is_4d ? q_nope.size(1) : 1;
  TORCH_CHECK(
      mla_decode_q_tile(seq_len_q) == QTileM,
      "MLA decode kernel instantiated for Q tile ",
      QTileM,
      " but q has seq_len_q=",
      seq_len_q,
      " (wants Q tile ",
      mla_decode_q_tile(seq_len_q),
      ")");

  if (num_kv_splits > 1) {
    using cutlass::flash_attention::kernel::SplitKVWorkspaceLayout;
    int batch = q_nope.size(0);
    int num_heads = q_nope.size(-2);
    int head_size_o = q_nope.size(-1);
    SplitKVWorkspaceLayout ws(batch, num_heads, num_kv_splits, head_size_o, sizeof(Element), seq_len_q);
    TORCH_CHECK(
        static_cast<size_t>(workspace.numel()) >= ws.total_bytes,
        "MLA workspace too small: need ",
        ws.total_bytes,
        " bytes for num_kv_splits=",
        num_kv_splits,
        ", but got ",
        workspace.numel(),
        " bytes. Reallocate workspace with flash_mla_get_workspace_size() using "
        "matching (batch, num_heads, max_seq_len, page_size, seq_len_q) parameters.");
  }

  if (num_kv_splits == 1) {
    runMlaImpl<Element, PageSizeOpt, EnabledSplitKV<false>, QTileM>(
        out, lse, q_nope, q_pe, kv_c_and_k_pe_cache, seq_lens, page_table, workspace, sm_scale, num_kv_splits);
  } else {
    runMlaImpl<Element, PageSizeOpt, EnabledSplitKV<true>, QTileM>(
        out, lse, q_nope, q_pe, kv_c_and_k_pe_cache, seq_lens, page_table, workspace, sm_scale, num_kv_splits);
  }
}
