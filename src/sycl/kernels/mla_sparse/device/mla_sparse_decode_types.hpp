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
  \brief Shared type definitions for Sparse MLA decode kernel instantiations
         (DeepSeek V4: two KV cache pools, scattered gather, attn_sink merge)
*/

#pragma once

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cmath>
#include <cute/tensor.hpp>
#include <limits>
#include <optional>
#include <sycl/sycl.hpp>

#include "../../../Utils.h"
#include "sycl/kernels/mla_sparse/collective/xe_mla_sparse_epilogue.hpp"
#include "sycl/kernels/mla_sparse/collective/xe_mla_sparse_mainloop.hpp"
#include "sycl/kernels/mla_sparse/device/mla_sparse_runner.hpp"
#include "sycl/kernels/mla_sparse/kernel/mla_sparse_tile_scheduler.hpp"
#include "sycl/kernels/mla_sparse/kernel/xe_mla_sparse_kernel.hpp"

using namespace cute;

//----------------- set HasExtra options --------------------//
template <bool v>
struct MlaSparseHasExtra {
  static constexpr bool value = v;
};

//----------------- set page size options --------------------//
template <int PageSize>
struct SparseMlaPageSizeOption {
  static constexpr int value = PageSize;
};

//----------------- set element type options --------------------//
template <typename T>
struct SparseMlaToCutlassElementType {
  using type = T;
};

template <>
struct SparseMlaToCutlassElementType<sycl::half> {
  using type = cutlass::half_t;
};

template <>
struct SparseMlaToCutlassElementType<sycl::ext::oneapi::bfloat16> {
  using type = cutlass::bfloat16_t;
};

//----------------- define problem shape --------------------//
// DeepSeek V4 Sparse MLA: topk + extra_topk instead of seq_len_kv
struct FSparseMlAProblemShape {
  int batch = 0;
  int num_heads_q = 0;
  int num_heads_kv = 0;
  int seq_len_qo = 0;
  int head_size_q_nope = 0;  // D_NOPE = 448
  int head_size_q_pe = 0;    // D_ROPE = 64 (mainloop computes offset from Q_nope)
  int head_size_k = 0;       // D_NOPE = 448 (FP8 nope elements)
  int head_size_v = 0;       // D_V = 512 (dequanted: 448 nope + 64 rope)
  int head_size_o = 0;       // D_V = 512
  int page_size = 0;         // PAGE_SIZE = 128
  int total_page = 0;        // Total pages in primary KV cache
  int total_extra_page = 0;  // Total pages in extra KV cache
  int swa_topk = 0;          // SWA window token count (e.g. 128)
  int extra_topk = 0;        // C4 routing top-k token count (e.g. 512)
  int extra_page_size = 0;   // Page size of extra KV cache (may differ from primary)
  int heads_per_wg = 4;      // Multi-head fusion: heads processed per workgroup

  FSparseMlAProblemShape() = default;
};

//----------------- define Sparse MLA Xe configuration --------------------//
template <
    typename T,
    typename PageSizeOpt = SparseMlaPageSizeOption<128>,
    typename HasExtraOpt = MlaSparseHasExtra<true>>
struct MlaSparseXe {
  static constexpr bool HasExtra = HasExtraOpt::value;
  using TileScheduler = typename cutlass::flash_attention::kernel::XeMlaSparseIndividualTileScheduler;

  static constexpr int PAGE_SIZE = PageSizeOpt::value;
  using KvTileSizeType = cute::Int<PAGE_SIZE>;

  static constexpr int NumSubgroupsN = PAGE_SIZE / 16;

  using TileShapeQK = Shape<_1, KvTileSizeType, _128>;
  using TileShapePV = Shape<_1, _128, KvTileSizeType>;
  using TileShapeOutput = Shape<_1, _512>;

  using SubgroupLayoutQK = Layout<Shape<_1, cute::Int<NumSubgroupsN>, _1>>;
  using SubgroupLayoutPV = decltype(cutlass::flash_attention::collective::get_sg_layout_pv(SubgroupLayoutQK{}));

  using ElementType = typename SparseMlaToCutlassElementType<T>::type;
  using ElementQ = ElementType;
  using ElementK = cutlass::float_e4m3_t;
  using ElementV = cutlass::float_e4m3_t;
  using ElementO = ElementType;
  static constexpr int SGTileQ = get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})))();
  using MMAOperation = XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, ElementQ>;

  using StrideQ = Stride<int, _1, int, int>;
  using StrideK = Stride<int, _1, int, int>;
  using StrideV = Stride<_1, int, int, int>;
  using StrideO = Stride<int, _1, int, int>;
  // 4-rank stride for paged K components: (token_stride, _1, page_stride, 1)
  // Matches normal MLA's StrideK layout with dummy 4th dimension
  using StrideKV = Stride<int, _1, int, int>;
  // 4-rank stride for paged V components: (_1, token_stride, page_stride, 1)
  // Transposed view of same data — matches normal MLA's StrideV
  using StrideKV_V = Stride<_1, int, int, int>;
  using GmemTiledCopyQ = void;
  using GmemTiledCopyK = void;
  using GmemTiledCopyV = void;
  using GmemTiledCopyO = void;

  using ProblemShape = FSparseMlAProblemShape;

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
  using TensorK = decltype(make_dummy_tensor_type(ElementK{}, StrideK{}));       // K_nope (FP8)
  using TensorV = decltype(make_dummy_tensor_type(ElementV{}, StrideV{}));       // V_nope (FP8, transposed)
  using TensorKPe = decltype(make_dummy_tensor_type(ElementQ{}, StrideKV{}));    // K_pe (bf16)
  using TensorVPe = decltype(make_dummy_tensor_type(ElementQ{}, StrideKV_V{}));  // V_pe (bf16, transposed)
  using TensorScale = decltype(make_dummy_tensor_type(uint8_t{}, StrideKV{}));   // KV_scale (uint8)
  using TensorO = decltype(make_dummy_tensor_type(ElementO{}, StrideO{}));

  // Collective Mainloop (Sparse MLA with two-phase scattered gather)
  static constexpr int PipelineStages = 1;
  using MainloopDispatchPolicy = cutlass::flash_attention::XeDefault<PipelineStages>;
  using CollectiveMainloop = cutlass::flash_attention::collective::XeMlaSparseMainloop<
      MainloopDispatchPolicy,
      true,
      HasExtra,
      TiledMMAQK,
      TiledMMAPV,
      VTiles,
      TensorQ,
      TensorK,
      TensorV,
      TensorKPe,
      TensorVPe,
      TensorScale,
      GmemTiledCopyQ,
      GmemTiledCopyK,
      GmemTiledCopyV>;

  // Collective Epilogue (with LSE output + attn_sink merge)
  using CollectiveEpilogue = cutlass::flash_attention::collective::
      XeMlaSparseEpilogue<CollectiveMainloop, TileShapeOutput, TensorO, GmemTiledCopyO>;

  // Kernel instantiation
  using FmlaKernel = cutlass::flash_attention::kernel::
      XeMlaSparseFwdKernel<ProblemShape, CollectiveMainloop, CollectiveEpilogue, TileScheduler>;

  using Fmla = cutlass::flash_attention::device::MLASparse<FmlaKernel>;
};

template <typename T>
inline typename T::Fmla::Arguments args_from_options_sparse(
    at::Tensor const& out,
    at::Tensor const& lse_out,
    at::Tensor const& q,                          // [B, 1, H, D_qk=576]
    at::Tensor const& kv_c_and_k_pe_cache,        // [num_pages, page_size, 1, D]
    at::Tensor const& extra_kv_c_and_k_pe_cache,  // [num_extra_pages, page_size, 1, D]
    at::Tensor const& swa_indices,                // [B, 1, swa_topk]
    at::Tensor const& extra_indices,              // [B, 1, extra_topk]
    at::Tensor const& attn_sink,                  // [H]
    at::Tensor const& topk_length,                // [B]
    at::Tensor const& extra_topk_length,          // [B]
    double sm_scale,
    int64_t head_dim_v,
    bool is_fp8_kvcache = false) {
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = q.device().index();
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  // q: [B, s_q=1, H, D_qk=576]
  // Q_nope = q[:, :, :, :head_dim_v]
  // Q_pe   = q[:, :, :, head_dim_v:]
  // Both share the same stride layout — just offset the base pointer (like K_pe = K + v_head_dim)
  int batch = q.size(0);
  int num_heads_q = q.size(2);
  int D_qk = q.size(3);
  int v_head_dim_int = static_cast<int>(head_dim_v);  // 512
  int q_nope_dim = 448;                               // fixed for DSV4
  int q_pe_dim = 64;                                  // fixed for DSV4

  int page_size = kv_c_and_k_pe_cache.size(1);
  int extra_page_size = extra_kv_c_and_k_pe_cache.size(1);
  int total_page = kv_c_and_k_pe_cache.size(0);
  int total_extra_page = extra_kv_c_and_k_pe_cache.size(0);
  int swa_topk = swa_indices.size(2);
  int extra_topk = extra_indices.size(2);

  FSparseMlAProblemShape problem_shape;
  problem_shape.batch = batch;
  problem_shape.num_heads_q = num_heads_q;
  problem_shape.num_heads_kv = 1;
  problem_shape.seq_len_qo = 1;
  problem_shape.head_size_q_nope = q_nope_dim;
  problem_shape.head_size_q_pe = q_pe_dim;
  problem_shape.head_size_k = q_nope_dim;
  problem_shape.head_size_v = v_head_dim_int;
  problem_shape.head_size_o = v_head_dim_int;
  problem_shape.page_size = page_size;
  problem_shape.total_page = total_page;
  problem_shape.total_extra_page = total_extra_page;
  problem_shape.swa_topk = swa_topk;
  problem_shape.extra_topk = extra_topk;
  problem_shape.extra_page_size = extra_page_size;
  problem_shape.heads_per_wg = T::CollectiveMainloop::HEADS_PER_WG;

  using StrideQ = typename T::StrideQ;
  using StrideK = typename T::StrideK;
  using StrideO = typename T::StrideO;
  using StrideKV = typename T::StrideKV;
  using StrideKV_V = typename T::StrideKV_V;
  using ElementQ = typename T::ElementQ;
  using ElementK = typename T::ElementK;
  using ElementO = typename T::ElementO;

  // q is [B, 1, H, D_qk]. After squeeze(1) logically [B, H, D_qk].
  // q_nope and q_pe share the same memory — just different base pointers.
  StrideQ stride_Q = cute::make_stride(
      static_cast<int>(batch * num_heads_q * D_qk),
      cute::_1{},
      static_cast<int>(q.stride(2)),
      static_cast<int>(q.stride(0)));

  // Output stride: out is [B, 1, H, D_V]
  StrideO stride_O = cute::make_stride(
      static_cast<int>(batch * num_heads_q * v_head_dim_int),
      cute::_1{},
      static_cast<int>(out.stride(2)),
      static_cast<int>(out.stride(0)));

  // KV cache strides per component (3-rank: token_stride, _1, page_stride)
  // Page layout: data section at stride (nope_dim + rope_dim*2 = 576) per token,
  // then scale section at stride 8 per token. NOT interleaved!
  // The as_strided view has stride 584 (576+8), but actual data stride is 576.
  int data_stride_bytes = q_nope_dim + q_pe_dim * 2;                        // 576 = data section token stride
  int token_stride_bytes = data_stride_bytes;                               // use data section stride for K/V access
  int page_stride_bytes = static_cast<int>(kv_c_and_k_pe_cache.stride(0));  // total_page_bytes

  // K_nope (FP8, 1 byte/elem): token_stride = token_stride_bytes, page_stride = page_stride_bytes
  StrideKV stride_K_nope = cute::make_stride(token_stride_bytes, cute::_1{}, page_stride_bytes, static_cast<int>(1));

  // K_pe (bf16, 2 bytes/elem): strides in bf16 element units
  StrideKV stride_K_pe =
      cute::make_stride(token_stride_bytes / 2, cute::_1{}, page_stride_bytes / 2, static_cast<int>(1));

  // KV_scale (uint8, 1 byte/elem): scales are packed contiguously in their section
  // Scale section has its own token stride = 8 bytes, page stride = page_stride_bytes
  StrideKV stride_KV_scale = cute::make_stride(8, cute::_1{}, page_stride_bytes, static_cast<int>(1));

  // Extra cache strides (same data stride, different page_size may give different page_stride)
  int extra_token_stride_bytes = data_stride_bytes;  // same data section stride = 576
  int extra_page_stride_bytes = static_cast<int>(extra_kv_c_and_k_pe_cache.stride(0));

  StrideKV stride_K_nope_extra =
      cute::make_stride(extra_token_stride_bytes, cute::_1{}, extra_page_stride_bytes, static_cast<int>(1));
  StrideKV stride_K_pe_extra =
      cute::make_stride(extra_token_stride_bytes / 2, cute::_1{}, extra_page_stride_bytes / 2, static_cast<int>(1));
  StrideKV stride_KV_scale_extra = cute::make_stride(8, cute::_1{}, extra_page_stride_bytes, static_cast<int>(1));

  // V_nope = K_nope (shared latent in MLA) — transposed view for P*V GEMM
  // V stride: (_1, token_stride, page_stride, 1) — first two dims swapped vs K
  StrideKV_V stride_V_nope = cute::make_stride(cute::_1{}, token_stride_bytes, page_stride_bytes, static_cast<int>(1));
  StrideKV_V stride_V_pe =
      cute::make_stride(cute::_1{}, token_stride_bytes / 2, page_stride_bytes / 2, static_cast<int>(1));

  // Extra V strides — transposed view, same as primary
  StrideKV_V stride_V_nope_extra =
      cute::make_stride(cute::_1{}, extra_token_stride_bytes, extra_page_stride_bytes, static_cast<int>(1));
  StrideKV_V stride_V_pe_extra =
      cute::make_stride(cute::_1{}, extra_token_stride_bytes / 2, extra_page_stride_bytes / 2, static_cast<int>(1));

  typename T::Fmla::KernelArguments kernel_args{};
  kernel_args.shape = problem_shape;
  kernel_args.Q = static_cast<const ElementQ*>(q.data_ptr());
  kernel_args.dQ = stride_Q;
  kernel_args.Q_pe = static_cast<const ElementQ*>(q.data_ptr()) + q_nope_dim;
  kernel_args.dQ_pe = stride_Q;
  auto* kv_base = static_cast<const char*>(kv_c_and_k_pe_cache.data_ptr());
  kernel_args.K_nope = reinterpret_cast<const ElementK*>(kv_base);
  kernel_args.dK_nope = stride_K_nope;
  kernel_args.K_pe = reinterpret_cast<const ElementQ*>(kv_base + q_nope_dim);
  kernel_args.dK_pe = stride_K_pe;
  kernel_args.KV_scale = reinterpret_cast<const uint8_t*>(kv_base + static_cast<int64_t>(page_size) * 576);
  kernel_args.dKV_scale = stride_KV_scale;
  kernel_args.V_nope = reinterpret_cast<const ElementK*>(kv_base);
  kernel_args.dV_nope = stride_V_nope;
  kernel_args.V_pe = reinterpret_cast<const ElementQ*>(kv_base + q_nope_dim);
  kernel_args.dV_pe = stride_V_pe;

  // Extra KV cache: same layout, different buffer
  auto* extra_kv_base = static_cast<const char*>(extra_kv_c_and_k_pe_cache.data_ptr());
  kernel_args.K_nope_extra = reinterpret_cast<const ElementK*>(extra_kv_base);
  kernel_args.dK_nope_extra = stride_K_nope_extra;
  kernel_args.K_pe_extra = reinterpret_cast<const ElementQ*>(extra_kv_base + q_nope_dim);
  kernel_args.dK_pe_extra = stride_K_pe_extra;
  kernel_args.KV_scale_extra =
      reinterpret_cast<const uint8_t*>(extra_kv_base + static_cast<int64_t>(extra_page_size) * 576);
  kernel_args.dKV_scale_extra = stride_KV_scale_extra;
  kernel_args.V_nope_extra = reinterpret_cast<const ElementK*>(extra_kv_base);
  kernel_args.dV_nope_extra = stride_V_nope_extra;
  kernel_args.V_pe_extra = reinterpret_cast<const ElementQ*>(extra_kv_base + q_nope_dim);
  kernel_args.dV_pe_extra = stride_V_pe_extra;
  kernel_args.O = static_cast<ElementO*>(out.data_ptr());
  kernel_args.dO = stride_O;
  kernel_args.lse_out = static_cast<float*>(lse_out.data_ptr());

  typename T::CollectiveMainloop::Arguments mainloop_args{
      static_cast<float>(sm_scale),
      page_size,
      extra_page_size,
      total_page,
      static_cast<const int*>(swa_indices.data_ptr()),
      swa_topk,
      static_cast<const int*>(extra_indices.data_ptr()),
      extra_topk,
      static_cast<const int*>(topk_length.data_ptr()),
      static_cast<const int*>(extra_topk_length.data_ptr()),
      is_fp8_kvcache};

  // Epilogue arguments: attn_sink and LSE output
  typename T::CollectiveEpilogue::Arguments epilogue_args{
      static_cast<const float*>(attn_sink.data_ptr()), static_cast<float*>(lse_out.data_ptr()), num_heads_q};

  typename T::Fmla::Arguments arguments{kernel_args, mainloop_args, epilogue_args, hw_info};

  return arguments;
}

template <
    typename Element,
    typename PageSizeOpt = SparseMlaPageSizeOption<128>,
    typename HasExtraOpt = MlaSparseHasExtra<true>>
inline void runMlaSparseImpl(
    at::Tensor& out,
    at::Tensor& lse_out,
    const at::Tensor& q,
    const at::Tensor& kv_c_and_k_pe_cache,
    const at::Tensor& extra_kv_c_and_k_pe_cache,
    const at::Tensor& swa_indices,
    const at::Tensor& extra_indices,
    const at::Tensor& attn_sink,
    const at::Tensor& topk_length,
    const at::Tensor& extra_topk_length,
    double sm_scale,
    int64_t head_dim_v,
    bool is_fp8_kvcache = false) {
  using MlaSparseXeType = MlaSparseXe<Element, PageSizeOpt, HasExtraOpt>;
  typename MlaSparseXeType::Fmla fmla;
  auto arguments = args_from_options_sparse<MlaSparseXeType>(
      out,
      lse_out,
      q,
      kv_c_and_k_pe_cache,
      extra_kv_c_and_k_pe_cache,
      swa_indices,
      extra_indices,
      attn_sink,
      topk_length,
      extra_topk_length,
      sm_scale,
      head_dim_v,
      is_fp8_kvcache);

  auto workspace = at::empty({0}, at::TensorOptions().dtype(at::kByte).device(q.device()));
  CUTLASS_CHECK(fmla.can_implement(arguments));
  CUTLASS_CHECK(fmla.run(arguments, workspace.data_ptr()));
}

template <typename Element, typename PageSizeOpt = SparseMlaPageSizeOption<128>>
inline void runMlaSparse(
    at::Tensor& out,
    at::Tensor& lse_out,
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& indices,
    const std::optional<at::Tensor>& topk_length,
    const std::optional<at::Tensor>& extra_k_cache,
    const std::optional<at::Tensor>& extra_indices,
    const std::optional<at::Tensor>& extra_topk_length,
    const std::optional<at::Tensor>& attn_sink,
    double sm_scale,
    int64_t head_dim_v,
    bool is_fp8_kvcache) {
  int B = q.size(0);
  int H = q.size(2);

  // Handle optional attn_sink: use very-negative if None (no dampening)
  at::Tensor attn_sink_resolved;
  if (attn_sink.has_value()) {
    attn_sink_resolved = attn_sink.value();
  } else {
    attn_sink_resolved = at::full({H}, -1e9f, at::TensorOptions().dtype(at::kFloat).device(q.device()));
  }

  at::Tensor topk_length_resolved, extra_topk_length_resolved;
  if (topk_length.has_value()) {
    topk_length_resolved = topk_length.value();
  } else {
    int swa_topk = indices.size(2);
    topk_length_resolved = at::full({B}, swa_topk, at::TensorOptions().dtype(at::kInt).device(q.device()));
  }
  if (extra_topk_length.has_value()) {
    extra_topk_length_resolved = extra_topk_length.value();
  } else {
    int extra_topk_val = (extra_indices.has_value()) ? static_cast<int>(extra_indices.value().size(2)) : 0;
    extra_topk_length_resolved = at::full({B}, extra_topk_val, at::TensorOptions().dtype(at::kInt).device(q.device()));
  }

  if (extra_k_cache.has_value() && extra_indices.has_value()) {
    runMlaSparseImpl<Element, PageSizeOpt, MlaSparseHasExtra<true>>(
        out,
        lse_out,
        q,
        k_cache,
        extra_k_cache.value(),
        indices,
        extra_indices.value(),
        attn_sink_resolved,
        topk_length_resolved,
        extra_topk_length_resolved,
        sm_scale,
        head_dim_v,
        is_fp8_kvcache);
  } else {
    int page_size = k_cache.size(1);
    int D = k_cache.size(3);
    auto dummy_extra_cache = at::zeros({1, page_size, 1, D}, k_cache.options());
    auto dummy_extra_indices = at::full({B, 1, 1}, -1, indices.options());

    runMlaSparseImpl<Element, PageSizeOpt, MlaSparseHasExtra<false>>(
        out,
        lse_out,
        q,
        k_cache,
        dummy_extra_cache,
        indices,
        dummy_extra_indices,
        attn_sink_resolved,
        topk_length_resolved,
        extra_topk_length_resolved,
        sm_scale,
        head_dim_v,
        is_fp8_kvcache);
  }
}
