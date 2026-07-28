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

#include <cstdint>
#include <type_traits>

#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/algorithm/subgroup_algorithms.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/sycl_vector_types.h"
#include "fmha_fusion.hpp"

namespace cutlass::fmha {

template <int Stages>
class XeDefault {};  // Default FMHA mainloop, P in registers.

};  // namespace cutlass::fmha

namespace cutlass::fmha::collective {

using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <bool AppendKV_, class ElementK_, class ElementV_>
struct KVParams {};

template <class ElementK_, class ElementV_>
struct KVParams<true, ElementK_, ElementV_> {
  ElementK_ const* ptr_K = nullptr;
  ElementV_ const* ptr_V = nullptr;
  int const* ptr_cu_seqlens_k = nullptr;
  int const* ptr_cache_seqlens = nullptr;
  int seq_len_kv = 0;
  int total_kv = 0;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    class DispatchPolicy_,
    bool CausalMask_,
    bool CachedKV_,
    bool PagedKV_,
    class TiledMMAQK_,  // Tiling for Q*K GEMM
    class TiledMMAPV_,  // Tiling for P*V GEMM
    int VTiles_,        // # of tiles in V dimension
    class TensorQ_,     // Global Q/K/V tensors
    class TensorK_,
    class TensorV_,
    class TensorK_cache_,
    class TensorV_cache_,
    class TiledCopyQ_ = void,        // Optional TiledCopy for loading Q
    class TiledCopyK_ = void,        // Optional TiledCopy for loading K
    class TiledCopyV_ = void,        // Optional TiledCopy for loading V
    class TiledCopyK_cache_ = void,  // Optional TiledCopy for loading K_cache
    class TiledCopyV_cache_ = void,  // Optional TiledCopy for loading V_cache
    bool LocalMask_ = false,
    // PackGQA: the M tile holds the head_group_q query heads of one GQA group
    // (decode only, seq_len_qo == 1). All packed rows share the single decode
    // KV position, so per-row masking must use a fixed decode row. Default
    // false keeps prefill (and non-packed decode) unaffected.
    bool PackGQA_ = false,
    bool AppendKV_ = false,
    bool DirectAppendKV_ = false,
    bool WideAppendKV_ = false>
struct FMHAFwdMainloop {
  static_assert(cutlass::detail::dependent_false<DispatchPolicy_>, "Could not find a mainloop specialization.");
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    int Stages,
    bool CausalMask_,
    bool CachedKV_,
    bool PagedKV_,
    class TiledMMAQK_,
    class TiledMMAPV_,
    int VTiles_,
    class TensorQ_,
    class TensorK_,
    class TensorV_,
    class TensorK_cache_,
    class TensorV_cache_,
    class TiledCopyQ_,
    class TiledCopyK_,
    class TiledCopyV_,
    class TiledCopyK_cache_,
    class TiledCopyV_cache_,
    bool LocalMask_,
    bool PackGQA_,
    bool AppendKV_,
    bool DirectAppendKV_,
    bool WideAppendKV_>
struct FMHAFwdMainloop<
    XeDefault<Stages>,
    CausalMask_,
    CachedKV_,
    PagedKV_,
    TiledMMAQK_,
    TiledMMAPV_,
    VTiles_,
    TensorQ_,
    TensorK_,
    TensorV_,
    TensorK_cache_,
    TensorV_cache_,
    TiledCopyQ_,
    TiledCopyK_,
    TiledCopyV_,
    TiledCopyK_cache_,
    TiledCopyV_cache_,
    LocalMask_,
    PackGQA_,
    AppendKV_,
    DirectAppendKV_,
    WideAppendKV_> {
  //
  // Type Aliases
  //
  using TiledMMAQK = TiledMMAQK_;
  using TiledMMAPV = TiledMMAPV_;
  using TileShapeQK = decltype(TiledMMAQK{}.tile_mnk());
  using TileShapePV = decltype(TiledMMAPV{}.tile_mnk());
  static constexpr int VTiles = VTiles_;
  using SubgroupLayoutQK = decltype(TiledMMAQK{}.get_atom_layout_mnk());
  using SGPerWG = decltype(product(take<1, 4>(shape(typename TiledMMAQK::ThrLayoutVMNK{}))));

  using TensorQ = TensorQ_;
  using TensorK = TensorK_;
  using TensorV = TensorV_;

  using ElementQ = typename TensorQ::engine_type::value_type;
  using ElementK = typename TensorK::engine_type::value_type;

  using TensorQ2D = decltype(TensorQ_{}(append<rank_v<TensorQ_>>(make_coord(_, _), 0)));
  using TensorK2D = decltype(TensorK_{}(append<rank_v<TensorK_>>(make_coord(_, _), 0)));
  using TensorV2D = decltype(TensorV_{}(append<rank_v<TensorV_>>(make_coord(_, _), 0)));

  using TiledCopyQ =
      conditional_t<is_void_v<TiledCopyQ_>, decltype(make_block_2d_copy_A(TiledMMAQK{}, TensorQ2D{})), TiledCopyQ_>;
  using TiledCopyK =
      conditional_t<is_void_v<TiledCopyK_>, decltype(make_block_2d_copy_B(TiledMMAQK{}, TensorK2D{})), TiledCopyK_>;
  using TiledCopyV =
      conditional_t<is_void_v<TiledCopyV_>, decltype(make_block_2d_copy_B(TiledMMAPV{}, TensorV2D{})), TiledCopyV_>;
  using TensorK_cache = TensorK_cache_;
  using TensorV_cache = TensorV_cache_;
  using TensorK_cache2D = decltype(TensorK_cache_{}(append<rank_v<TensorK_cache_>>(make_coord(_, _), 0)));
  using TensorV_cache2D = decltype(TensorV_cache_{}(append<rank_v<TensorV_cache_>>(make_coord(_, _), 0)));
  using TiledCopyK_cache = conditional_t<
      is_void_v<TiledCopyK_cache_>,
      decltype(make_block_2d_copy_B(TiledMMAQK{}, TensorK_cache2D{})),
      TiledCopyK_cache_>;
  using TiledCopyV_cache = conditional_t<
      is_void_v<TiledCopyV_cache_>,
      decltype(make_block_2d_copy_B(TiledMMAPV{}, TensorV_cache2D{})),
      TiledCopyV_cache_>;

  // TODO: static_asserts on TiledMMAPV here...

  //
  // Accumulator types
  //
  // FragS:    accumulator for Q*K MMA
  // FragO:    accumulator for P*V MMAs.
  //           Note: v mode may be split into multiple pieces
  //             to reduce register pressure.
  // Frag*Row types are reductions of the corresponding Frag* types
  //   over rows.
  //
  template <typename TiledMMA>
  using FragC = decltype(TiledMMA{}.get_slice(0).partition_sg_fragment_C(
      make_identity_tensor(select<0, 1>(TiledMMA{}.tile_mnk()))));

  using FragS = FragC<TiledMMAQK>;
  using FragSRow = decltype(reduce<1>(FragS{}, sycl::plus<void>{}));
  using FragSCol = decltype(reduce<0>(FragS{}, sycl::plus<void>{}));
  using ElementS = typename TiledMMAQK::ValTypeD;

  using SingleFragA = FragC<TiledMMAPV>;                       // (atom val,q',v')
  using FragA = expand_sg_fragment_t<SingleFragA, 1, VTiles>;  // (atom val,q',v',VV)
  using FragARow = decltype(reduce<1>(FragA{}, sycl::plus<void>{}));
  using ElementA = typename TiledMMAPV::ValTypeD;

  static constexpr bool CausalMask = CausalMask_;
  static constexpr bool CachedKV = CachedKV_;
  static constexpr bool PagedKV = PagedKV_;
  static constexpr bool LocalMask = LocalMask_;
  static constexpr bool PackGQA = PackGQA_;

  // FP8 KV cache: enabled when the K element type is an 8-bit float. The fp8
  // K/V are dequantized (cast to ElementQ and multiplied by the per-tensor
  // scale) inside the mainloop after the block-2D load.
  static constexpr bool Fp8KV = is_any_of_v<ElementK, float_e5m2_t, float_e4m3_t>;
  static constexpr bool AppendKV = AppendKV_;
  static constexpr bool DirectAppendKV = DirectAppendKV_;
  static constexpr bool WideAppendKV = WideAppendKV_;
  static_assert(!DirectAppendKV || (AppendKV && PagedKV), "Direct AppendKV requires paged AppendKV");
  static_assert(
      !WideAppendKV || (AppendKV && PagedKV && !DirectAppendKV), "Wide AppendKV requires fused paged AppendKV");
  using KVStorage = KVParams<AppendKV, typename TensorK_cache::element_type, typename TensorV_cache::element_type>;

  // User-facing arguments
  struct Arguments {
    ElementS const scale;
    int const* ptr_page_table = nullptr;
    int page_size = 0;
    int max_num_pages_per_seq = 0;
    int window_size_left = -1;
    int window_size_right = -1;
    KVStorage kv{};
  };

  // Kernel-facing parameters
  using Params = Arguments;

  // SLM data
  struct SharedStorage {};

  Params params;

  //
  // Methods
  //

  FMHAFwdMainloop(Params const& params_, SharedStorage&) : params(params_) {}

  static constexpr Params to_underlying_arguments(Arguments const& args, void* /* workspace */) {
    constexpr double kLog2e = 1.4426950408889634074;  // log_2(e)
    ElementS val = args.scale * static_cast<ElementS>(kLog2e);
    return Params{
        val,
        args.ptr_page_table,
        args.page_size,
        args.max_num_pages_per_seq,
        args.window_size_left,
        args.window_size_right,
        args.kv};
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) {
    return true;
  }

  CUTLASS_DEVICE
  int get_physical_k_tile(int K, int l_coord, int seq_len_kv_cache) {
    int next_page_logical_idx = K * get<1>(TileShapeQK{}) / params.page_size;
    // get<1>(TileShapeQK{}) usually smaller than page_size.
    // assuming page_size is multiple of get<1>(TileShapeQK{})
    int tiles_per_page = params.page_size / get<1>(TileShapeQK{});
    // int batch_offset =
    //     params.num_pages_per_seq ? params.num_pages_per_seq[l_coord] : l_coord * (seq_len_kv_cache /
    //     params.page_size);
    int batch_offset = l_coord * params.max_num_pages_per_seq;

    return params.ptr_page_table[batch_offset + next_page_logical_idx] * tiles_per_page + K % tiles_per_page;
  }

  CUTLASS_DEVICE
  void store_kv(
      TensorK_cache2D const& K_cache_2D,
      TensorV_cache2D const& V_cache_2D,
      int batch,
      int kv_head,
      int num_heads_kv,
      int thr_id,
      int kv_len_total,
      int append_store_len = -1,
      int append_store_begin = 0) const {
    if constexpr (AppendKV) {
      int const store_begin = cute::min(cute::max(append_store_begin, 0), kv_len_total);
      int const available_len = kv_len_total - store_begin;
      int const kv_len =
          append_store_len < 0 ? available_len : (append_store_len < available_len ? append_store_len : available_len);
      if (kv_len <= 0) {
        return;
      }

      auto& K_dst = const_cast<TensorK_cache2D&>(K_cache_2D);
      auto& V_dst = const_cast<TensorV_cache2D&>(V_cache_2D);
      int const lane_idx = thr_id % intel::sg_size;
      int const sub_group_id = thr_id / intel::sg_size;
      int const kv_begin = (params.kv.ptr_cu_seqlens_k != nullptr ? params.kv.ptr_cu_seqlens_k[batch]
                                                                  : batch * params.kv.seq_len_kv) +
                            store_begin;
      int const cache_len_old = params.kv.ptr_cache_seqlens[batch] + store_begin;
      int const head_size_qk = size<1>(K_cache_2D);
      int const head_size_vo = size<0>(V_cache_2D);
      int const max_hd = head_size_qk > head_size_vo ? head_size_qk : head_size_vo;

      // Every WG that can read this appended KV range scatters the same source
      // k_new/v_new values before loading K/V. Multiple WGs, including GQA
      // query heads that map to one kv_head, therefore overwrite identical bit
      // patterns at the same cache address; the append is intentionally
      // idempotent and does not rely on a grid-wide producer.
      if constexpr (
          sizeof(typename TensorK_cache::element_type) == 2 && sizeof(typename TensorV_cache::element_type) == 2) {
        // The q256 HD64 specialization uses 32B vectors to reduce flattened
        // store index overhead; other paths retain the 16B transaction.
        constexpr int kVecElems = WideAppendKV ? 16 : 8;
        using StoreVec = cute::conditional_t<WideAppendKV, cutlass::ulonglong4, cutlass::ulonglong2>;
        if (head_size_qk == head_size_vo && (head_size_qk % kVecElems) == 0) {
          // Large appends are bandwidth-bound, so split tokens across SGs;
          // small appends keep the old single-SG path to avoid control overhead.
          constexpr int kMinMultiSgTokens = 64;
          int const active_sg_count = kv_len >= kMinMultiSgTokens ? int(SGPerWG::value) : 1;
          if (sub_group_id >= active_sg_count) {
            return;
          }
          int const head_size = head_size_qk;
          bool const flatten_token_vectors = kv_len >= kMinMultiSgTokens && head_size == 64;
          int const vecs_per_token = head_size / kVecElems;
          int const worker = sub_group_id * intel::sg_size + lane_idx;
          int const worker_count = active_sg_count * intel::sg_size;
          bool single_dst_page = true;
          int single_row_base = cache_len_old;
          if constexpr (PagedKV) {
            int const first_dst_page = cache_len_old / params.page_size;
            int const last_dst_page = (cache_len_old + kv_len - 1) / params.page_size;
            single_dst_page = first_dst_page == last_dst_page;
            if (single_dst_page) {
              int const first_page_token = first_dst_page * params.page_size;
              int const logical_page = batch * params.max_num_pages_per_seq + first_dst_page;
              int const phys_page = params.ptr_page_table[logical_page];
              single_row_base = phys_page * params.page_size + (cache_len_old - first_page_token);
            }
          }

          if constexpr (PagedKV) {
            if (!single_dst_page) {
              // Cross-page append is common for large k_new; resolve the page
              // table once per destination page instead of once per token.
              for (int kv_tok0 = 0; kv_tok0 < kv_len;) {
                int const dst_tok0 = cache_len_old + kv_tok0;
                int const page = dst_tok0 / params.page_size;
                int const tok_in_page0 = dst_tok0 - page * params.page_size;
                int const tokens_in_page = params.page_size - tok_in_page0;
                int const page_len = tokens_in_page < (kv_len - kv_tok0) ? tokens_in_page : (kv_len - kv_tok0);
                int const logical_page = batch * params.max_num_pages_per_seq + page;
                int const phys_page = params.ptr_page_table[logical_page];
                int const row_base = phys_page * params.page_size + tok_in_page0;
                size_t src_base =
                    ((size_t)(kv_begin + kv_tok0) * (size_t)num_heads_kv + (size_t)kv_head) * (size_t)head_size;
                size_t const token_stride = (size_t)num_heads_kv * (size_t)head_size;

                if (flatten_token_vectors) {
                  for (int page_vec = worker; page_vec < page_len * vecs_per_token; page_vec += worker_count) {
                    int const page_tok = page_vec / vecs_per_token;
                    int const d_vec = page_vec - page_tok * vecs_per_token;
                    int const dst_row = row_base + page_tok;
                    size_t const token_src_base = src_base + (size_t)page_tok * token_stride;
                    int const d = d_vec * kVecElems;
                    size_t const src = token_src_base + (size_t)d;
                    StoreVec const k_value = *reinterpret_cast<StoreVec const*>(params.kv.ptr_K + src);
                    StoreVec const v_value = *reinterpret_cast<StoreVec const*>(params.kv.ptr_V + src);
                    *reinterpret_cast<StoreVec*>(&K_dst(dst_row, d)) = k_value;
                    *reinterpret_cast<StoreVec*>(&V_dst(d, dst_row)) = v_value;
                  }
                } else {
                  for (int page_tok = sub_group_id; page_tok < page_len; page_tok += active_sg_count) {
                    int const dst_row = row_base + page_tok;
                    size_t const token_src_base = src_base + (size_t)page_tok * token_stride;
                    for (int d_vec = lane_idx; d_vec < vecs_per_token; d_vec += intel::sg_size) {
                      int const d = d_vec * kVecElems;
                      size_t const src = token_src_base + (size_t)d;
                      StoreVec const k_value = *reinterpret_cast<StoreVec const*>(params.kv.ptr_K + src);
                      StoreVec const v_value = *reinterpret_cast<StoreVec const*>(params.kv.ptr_V + src);
                      *reinterpret_cast<StoreVec*>(&K_dst(dst_row, d)) = k_value;
                      *reinterpret_cast<StoreVec*>(&V_dst(d, dst_row)) = v_value;
                    }
                  }
                }
                kv_tok0 += page_len;
              }
              return;
            }
          }

          if (flatten_token_vectors) {
            for (int kv_vec = worker; kv_vec < kv_len * vecs_per_token; kv_vec += worker_count) {
              int const kv_tok = kv_vec / vecs_per_token;
              int const d_vec = kv_vec - kv_tok * vecs_per_token;
              int const kv_abs_tok = kv_begin + kv_tok;
              int const dst_row = single_row_base + kv_tok;
              size_t const src_base =
                  ((size_t)kv_abs_tok * (size_t)num_heads_kv + (size_t)kv_head) * (size_t)head_size;
              int const d = d_vec * kVecElems;
              size_t const src = src_base + (size_t)d;
              StoreVec const k_value = *reinterpret_cast<StoreVec const*>(params.kv.ptr_K + src);
              StoreVec const v_value = *reinterpret_cast<StoreVec const*>(params.kv.ptr_V + src);
              *reinterpret_cast<StoreVec*>(&K_dst(dst_row, d)) = k_value;
              *reinterpret_cast<StoreVec*>(&V_dst(d, dst_row)) = v_value;
            }
          } else {
            for (int kv_tok = sub_group_id; kv_tok < kv_len; kv_tok += active_sg_count) {
              int const kv_abs_tok = kv_begin + kv_tok;
              int const dst_row = single_row_base + kv_tok;
              size_t const src_base =
                  ((size_t)kv_abs_tok * (size_t)num_heads_kv + (size_t)kv_head) * (size_t)head_size;
              for (int d_vec = lane_idx; d_vec < vecs_per_token; d_vec += intel::sg_size) {
                int const d = d_vec * kVecElems;
                size_t const src = src_base + (size_t)d;
                StoreVec const k_value = *reinterpret_cast<StoreVec const*>(params.kv.ptr_K + src);
                StoreVec const v_value = *reinterpret_cast<StoreVec const*>(params.kv.ptr_V + src);
                *reinterpret_cast<StoreVec*>(&K_dst(dst_row, d)) = k_value;
                *reinterpret_cast<StoreVec*>(&V_dst(d, dst_row)) = v_value;
              }
            }
          }
          return;
        }
      }

      for (int linear = lane_idx; linear < kv_len * max_hd; linear += intel::sg_size) {
        int const d = linear % max_hd;
        int const kv_tok = linear / max_hd;
        int const kv_abs_tok = kv_begin + kv_tok;
        int const dst_tok = cache_len_old + kv_tok;
        int dst_row = dst_tok;
        if constexpr (PagedKV) {
          int const page = dst_tok / params.page_size;
          int const tok_in_page = dst_tok - page * params.page_size;
          int const logical_page = batch * params.max_num_pages_per_seq + page;
          int const phys_page = params.ptr_page_table[logical_page];
          dst_row = phys_page * params.page_size + tok_in_page;
        }

        if (d < head_size_qk) {
          size_t const src =
              ((size_t)kv_abs_tok * (size_t)num_heads_kv + (size_t)kv_head) * (size_t)head_size_qk + (size_t)d;
          K_dst(dst_row, d) = params.kv.ptr_K[src];
        }
        if (d < head_size_vo) {
          size_t const src =
              ((size_t)kv_abs_tok * (size_t)num_heads_kv + (size_t)kv_head) * (size_t)head_size_vo + (size_t)d;
          V_dst(d, dst_row) = params.kv.ptr_V[src];
        }
      }
    } else {
      (void)K_cache_2D;
      (void)V_cache_2D;
      (void)batch;
      (void)kv_head;
      (void)num_heads_kv;
      (void)thr_id;
      (void)kv_len_total;
      (void)append_store_len;
      (void)append_store_begin;
    }
  }

  template <bool PackedGQA = PackGQA_, typename QVCoord>
  CUTLASS_DEVICE void operator()(
      TensorQ2D const& Q_2D,  // (q,d)
      TensorK2D const& K_2D,  // (k,d)
      TensorV2D const& V_2D,  // (d,k)
      FragA& tArA,            // Output accumulator (q,v)
      FragARow& tA_max,       // Softmax row-wise max accumulator
      FragARow& tA_sum,       // Softmax row-wise sum accumulator
      QVCoord blk_qv,         // WG tile indices: (Q,V)
      int blk_k0,             // K block range: [K0,K1)
      int blk_k1,
      int total_blk,  // Total # of K blocks
      int blk_k1_causal,
      int thr_id,
      int seq_len,
      int seq_len_kv_cache,
      int l_coord,
      int kv_head,
      int num_heads_kv,
      int full_tile_offset,
      int discard_seq_coord,
      std::bool_constant<PackedGQA> = {},
      int append_total_len = 0,
      int append_store_len = -1,
      int append_store_begin = 0,
      TensorK_cache2D const& K_cache_2D = TensorK_cache2D{},
      TensorV_cache2D const& V_cache_2D = TensorV_cache2D{},
      float scale_k = 1.0f) {  // FP8 K per-tensor dequant scale
    using namespace sycl::ext::oneapi::this_work_item;

    // Short dimension names:
    //    q = sequence len dimension for Q
    //    k = sequence len dimension for K
    //    d = head size dimension for K/Q
    //    v = head size dimension for V
    //   VV = MMA tile indices for V
    // Capital letters (Q, K, ...) refer to WG block indices.
    // Primed letters (q', k', ...) refer to atom block indices.

    auto tile_shape_v = make_shape(get<1>(TileShapePV{}) * C<VTiles>{}, get<2>(TileShapePV{}));

    /* Create proxy coordinate tensors for Q/K/P/V */
    Tensor cQ = make_identity_tensor(Q_2D.shape());               // (q,d)
    Tensor cK = make_identity_tensor(K_2D.shape());               // (k,d)
    Tensor cV = make_identity_tensor(V_2D.shape());               // (v,k)
    Tensor cK_cache = make_identity_tensor(K_cache_2D.shape());   // (k,d)
    Tensor cV_cache = make_identity_tensor(V_cache_2D.shape());   // (v,k)
    Tensor cP = make_identity_tensor(take<0, 2>(TileShapeQK{}));  // (q,k)

    /* Partition global tensors into workgroup tiles */
    Tensor gQ = local_tile(cQ, TileShapeQK{}, append(blk_qv, _), Step<_1, X, _1>{});          // (q,d,D)
    Tensor gK = local_tile(cK, TileShapeQK{}, make_coord(_, _, _), Step<X, _1, _1>{});        // (k,d,K,D)
    Tensor gV = local_tile(cV, tile_shape_v, make_coord(get<1>(blk_qv), _));                  // (v,k,K)
    Tensor gV_split = local_tile(gV, TileShapePV{}, make_coord(_, _, 0), Step<X, _1, _1>{});  // (v,k,VV,K)

    Tensor gK_cache = local_tile(cK_cache, TileShapeQK{}, make_coord(_, _, _), Step<X, _1, _1>{});        // (k,d,K,D)
    Tensor gV_cache = local_tile(cV_cache, tile_shape_v, make_coord(get<1>(blk_qv), _));                  // (v,k,K)
    Tensor gV_cache_split = local_tile(gV_cache, TileShapePV{}, make_coord(_, _, 0), Step<X, _1, _1>{});  // (v,k,VV,K)

    /* Create global -> register copies */
    TiledCopyQ copy_q{Q_2D};
    TiledCopyK copy_k{K_2D};
    TiledCopyV copy_v{V_2D};
    TiledCopyK_cache copy_k_cache{K_cache_2D};
    TiledCopyV_cache copy_v_cache{V_cache_2D};

    /* Create MMAs */
    TiledMMAQK mma_qk{};
    TiledMMAPV mma_pv{};

    /* Slice TiledCopy/TiledMMA operations down to to work-item level */
    auto thr_copy_q = copy_q.get_slice(thr_id);
    auto thr_copy_k = copy_k.get_slice(thr_id);
    auto thr_copy_v = copy_v.get_slice(thr_id);
    auto thr_copy_k_cache = copy_k_cache.get_slice(thr_id);
    auto thr_copy_v_cache = copy_v_cache.get_slice(thr_id);
    auto thr_mma_qk = mma_qk.get_slice(thr_id);
    auto thr_mma_pv = mma_pv.get_slice(thr_id);

    /* Partition coordinate tensors for copy */
    auto tQgQ = thr_copy_q.partition_S(gQ);        // (atom_val,q',d',D)
    auto tKgK = thr_copy_k.partition_S(gK);        // (atom_val,k',d',K,D)
    auto tVgV = thr_copy_v.partition_S(gV_split);  // (atom_val,v',k',VV,K)
    auto tKgK_cache = thr_copy_k_cache.partition_S(gK_cache);
    auto tVgV_cache = thr_copy_v_cache.partition_S(gV_cache_split);

    /* Create register fragments for MMA and copies */
    auto tQrQ = thr_copy_q.partition_sg_fragment_D(gQ(_, _, 0));
    auto tSrQ = thr_mma_qk.partition_sg_fragment_A(gQ(_, _, 0));

    auto tKrK = thr_copy_k.partition_sg_fragment_D(gK(_, _, 0, 0));
    auto tSrK = thr_mma_qk.partition_sg_fragment_B(gK(_, _, 0, 0));

    auto tSrS = thr_mma_qk.partition_sg_fragment_C(cP);
    auto tArP = thr_mma_pv.partition_sg_fragment_A(cP);

    auto tVrV = thr_copy_v.partition_sg_fragment_D(gV_split(_, _, 0, 0));
    auto tArV = thr_mma_pv.partition_sg_fragment_B(gV_split(_, _, 0, 0));

    /* Create TiledCopy objects for prefetches */
    auto prefetch_q = make_block_2d_prefetch(copy_q);
    auto prefetch_k = make_block_2d_prefetch(copy_k);
    auto prefetch_v = make_block_2d_prefetch(copy_v);
    auto prefetch_k_cache = make_block_2d_prefetch(copy_k_cache);
    auto prefetch_v_cache = make_block_2d_prefetch(copy_v_cache);

    /* Partition global tensors for prefetch */
    auto pQgQ = prefetch_q.get_slice(thr_id).partition_S(gQ);
    auto pKgK = prefetch_k.get_slice(thr_id).partition_S(gK);
    auto pVgV = prefetch_v.get_slice(thr_id).partition_S(gV_split);
    auto pKgK_cache = prefetch_k_cache.get_slice(thr_id).partition_S(gK_cache);
    auto pVgV_cache = prefetch_v_cache.get_slice(thr_id).partition_S(gV_cache_split);

    // ------
    // Kernel
    // ------

    if constexpr (AppendKV) {
      store_kv(
          K_cache_2D,
          V_cache_2D,
          l_coord,
          kv_head,
          num_heads_kv,
          thr_id,
          append_total_len,
          append_store_len,
          append_store_begin);
      if constexpr (DirectAppendKV) {
        constexpr int kTileKV = get<1>(TileShapeQK{});
        int const cache_len_old = params.kv.ptr_cache_seqlens[l_coord];
        int const kv_begin =
            params.kv.ptr_cu_seqlens_k != nullptr ? params.kv.ptr_cu_seqlens_k[l_coord]
                                                  : l_coord * params.kv.seq_len_kv;
        if ((cache_len_old % kTileKV) != 0 || (kv_begin % kTileKV) != 0) {
          barrier();
        }
      } else {
        barrier();
      }
    }

    /* Initialization steps for first block: Q prefetch, O init */
    for (int D = 0; D < size<3>(pQgQ); D++) {
      prefetch(prefetch_q, pQgQ(_, _, _, D));
    }
    // Always initialize the per-WG accumulators: the caller (kernel) may pass
    // blk_k0 > 0 when sliding-window pruning skips leading K blocks, so we can
    // no longer key initialization off of (blk_k0 == 0).
    clear(tArA);
    fill(tA_max, cutlass::platform::numeric_limits<ElementA>::lowest());
    clear(tA_sum);

    /* Check if */
    bool check_remainder_k = (seq_len % get<1>(TileShapeQK{}) != 0);

    // FP8 K dequant: S = Q*K is linear in K, so the per-tensor scale_k is folded
    // into the softmax Q*K scale (qk_scale = params.scale * scale_k) instead of
    // rescaling every K register element in GEMM1. The V dequant scale (scale_v)
    // is likewise folded into the epilogue normalization.
    ElementS qk_scale = params.scale;
    if constexpr (Fp8KV) {
      qk_scale = params.scale * static_cast<ElementS>(scale_k);
    }

    constexpr int kTileKV = get<1>(TileShapeQK{});
    int const kblocks_total = ceil_div(seq_len_kv_cache, kTileKV);
    bool direct_append = false;
    int direct_block0 = kblocks_total;
    int direct_source_block0 = 0;
    if constexpr (DirectAppendKV) {
      int const cache_len_old = params.kv.ptr_cache_seqlens[l_coord];
      int const kv_begin =
          params.kv.ptr_cu_seqlens_k != nullptr ? params.kv.ptr_cu_seqlens_k[l_coord]
                                                : l_coord * params.kv.seq_len_kv;
      direct_append = (cache_len_old % kTileKV) == 0 && (kv_begin % kTileKV) == 0;
      direct_block0 = cache_len_old / kTileKV;
      direct_source_block0 = kv_begin / kTileKV;
    }

    auto run_k_blocks = [&](auto source_tag, int loop_k0, int loop_k1) {
      constexpr bool LoadDirect = decltype(source_tag)::value;
      if (loop_k0 >= loop_k1) {
        return;
      }

      int next_source_idx = loop_k0;
      if constexpr (LoadDirect) {
        next_source_idx = direct_source_block0 + loop_k0 - direct_block0;
        for (int D = 0; D < size<4>(pKgK); D++) {
          prefetch(prefetch_k, pKgK(_, _, _, next_source_idx, D));
        }
      } else {
        if constexpr (PagedKV) {
          next_source_idx = get_physical_k_tile(loop_k0, l_coord, seq_len_kv_cache);
        }
        for (int D = 0; D < size<4>(pKgK); D++) {
          prefetch(prefetch_k_cache, pKgK_cache(_, _, _, next_source_idx, D));
        }
      }

      for (int K = loop_k0; K < loop_k1 && K < kblocks_total; K++) {
        barrier_arrive(ScopeWorkgroup);

        bool need_causal = false;
        if constexpr (CausalMask) {
          need_causal = K >= blk_k1_causal;
        }

        int const source_idx = next_source_idx;
        int const next_k = K + 1;
        bool const has_next_k = next_k < loop_k1 && next_k < kblocks_total;
        if constexpr (LoadDirect) {
          next_source_idx = direct_source_block0 + next_k - direct_block0;
        } else {
          next_source_idx = next_k;
          if constexpr (PagedKV) {
            if (has_next_k) {
              next_source_idx = get_physical_k_tile(next_k, l_coord, seq_len_kv_cache);
            }
          }
        }

        clear(tSrS);
        CUTLASS_PRAGMA_UNROLL
        for (int D = 0; D < size<4>(tKgK); D++) {
          copy(copy_q, tQgQ(_, _, _, D), tQrQ);
          if constexpr (LoadDirect) {
            copy(copy_k, tKgK(_, _, _, source_idx, D), tKrK);
          } else {
            copy(copy_k_cache, tKgK_cache(_, _, _, source_idx, D), tKrK);
          }
          reorder(tQrQ, tSrQ);
          reorder(tKrK, tSrK);
          cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
        }

        CUTLASS_PRAGMA_UNROLL
        for (int VV = 0; VV < VTiles; VV++) {
          if constexpr (LoadDirect) {
            prefetch(prefetch_v, pVgV(_, _, _, VV, source_idx));
          } else {
            prefetch(prefetch_v_cache, pVgV_cache(_, _, _, VV, source_idx));
          }
        }

        if constexpr (CausalMask) {
          if (need_causal) {
            if constexpr (PackedGQA) {
              Tensor cPgP = make_identity_tensor(make_shape(seq_len, seq_len));
              Tensor gP = local_tile(cPgP, take<0, 2>(TileShapeQK{}), make_coord(get<0>(blk_qv), K));
              auto cS_thread = thr_mma_qk.partition_C(gP);
              CUTLASS_PRAGMA_UNROLL
              for (int i = 0; i < tSrS.size(); ++i) {
                int col_idx = get<1>(cS_thread(i));
                if (full_tile_offset < col_idx) {
                  tSrS(i) = ElementS(-INFINITY);
                }
              }
            } else {
              int lane_id = thr_id % intel::sg_size;
              constexpr int sg_tile_q = get<0>(TileShapeQK{}) / SGPerWG::value;
              int row_base = get<0>(blk_qv) * get<0>(TileShapeQK{}) + (thr_id / intel::sg_size) * sg_tile_q;
              constexpr int kTileK = get<1>(TileShapeQK{});
              constexpr int n_reps = kTileK / intel::sg_size;
              constexpr int elems_per_n = cute::size_v<typename decltype(tSrS)::layout_type> / n_reps;
              int k_base = K * kTileK;
              CUTLASS_PRAGMA_UNROLL
              for (int n = 0; n < n_reps; n++) {
                int col = k_base + n * intel::sg_size + lane_id;
                int causal_bound = col - full_tile_offset - row_base;
                CUTLASS_PRAGMA_UNROLL
                for (int j = 0; j < elems_per_n; j++) {
                  if (j < causal_bound) {
                    tSrS(n * elems_per_n + j) = ElementS(-INFINITY);
                  }
                }
              }
            }
          }
        }

        if constexpr (LocalMask) {
          Tensor cPgP = make_identity_tensor(make_shape(seq_len, seq_len));
          Tensor gP = local_tile(cPgP, take<0, 2>(TileShapeQK{}), make_coord(get<0>(blk_qv), K));
          auto cS_thread = thr_mma_qk.partition_C(gP);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tSrS.size(); ++i) {
            int row_idx = get<0>(cS_thread(i));
            int col_idx = get<1>(cS_thread(i));
            int row_kv_idx = (PackedGQA ? 0 : row_idx) + full_tile_offset;
            bool left_mask = col_idx < row_kv_idx - params.window_size_left;
            bool right_mask = col_idx > row_kv_idx + params.window_size_right;
            if (left_mask || right_mask) {
              tSrS(i) = ElementS(-INFINITY);
            }
          }
        }

        if (check_remainder_k && K == total_blk - 1) {
          FragSCol k_rem_mask;
          int k_val = get<0>(tKgK_cache(0, 0, 0, K, 0));
          int k = k_val + get_sub_group().get_local_id()[0];
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < k_rem_mask.size(); i++, k += intel::sg_size) {
            k_rem_mask(i) = (k < seq_len) ? ElementS(sycl::nan(0u)) : ElementS(-INFINITY);
          }
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tSrS.size(); i++) {
            tSrS(i) = sycl::fmin(tSrS(i), broadcast<1>(k_rem_mask, tSrS, i));
          }
        }

        auto rescale = softmax(K == blk_k0, tSrS, tA_max, tA_sum, qk_scale);
        reorder(tSrS, tArP);

        CUTLASS_PRAGMA_UNROLL
        for (int VV = 0; VV < VTiles; VV++) {
          if constexpr (LoadDirect) {
            copy(copy_v, tVgV(_, _, _, VV, source_idx), tVrV);
          } else {
            copy(copy_v_cache, tVgV_cache(_, _, _, VV, source_idx), tVrV);
          }
          reorder(tVrV, tArV);
          if (K != blk_k0) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < tArA.size() / VTiles; i++) {
              tArA(_, _, _, VV)(i) *= broadcast<0>(rescale, tArA, i);
            }
          }
          cute::gemm(mma_pv, tArP, tArV, tArA(_, _, _, VV));
        }

        if (has_next_k) {
          if constexpr (LoadDirect) {
            for (int D = 0; D < size<4>(pKgK); D++) {
              prefetch(prefetch_k, pKgK(_, _, _, next_source_idx, D));
            }
          } else {
            for (int D = 0; D < size<4>(pKgK); D++) {
              prefetch(prefetch_k_cache, pKgK_cache(_, _, _, next_source_idx, D));
            }
          }
        }

        barrier_wait(ScopeWorkgroup);
      }
    };

    int const cache_loop_k1 = direct_append ? cute::min(blk_k1, direct_block0) : blk_k1;
    run_k_blocks(cute::false_type{}, blk_k0, cache_loop_k1);
    if constexpr (DirectAppendKV) {
      if (direct_append) {
        run_k_blocks(cute::true_type{}, cute::max(blk_k0, direct_block0), blk_k1);
      }
    }
  }

  // Single step of blocked softmax.
  CUTLASS_DEVICE
  FragSRow softmax(
      bool first_block,     // First softmax block?
      FragS& tS,            // Softmax src/dst block
      FragSRow& tS_max,     // Softmax row-wise max accumulator
      FragSRow& tS_sum,     // Softmax row-wise sum accumulator
      ElementS qk_scale) {  // Q*K scale (folds in fp8 K per-tensor scale_k)

    /* Compute row-wise maxima for this block */
    auto tS_bmax = reduce<1>(tS, sycl::maximum{});

    /* Update (scaled) maxima and compute rescale factor */
    FragSRow rescale;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_max.size(); i++) {
      ElementS new_max = sycl::max(tS_max(i), qk_scale * tS_bmax(i));
      rescale(i) = sycl::native::exp2(tS_max(i) - new_max);
      tS_max(i) = new_max;
    }

    /* Scale S and subtract maxima, then exponentiate */
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS.size(); i++)
      tS(i) = sycl::native::exp2(qk_scale * tS(i) - broadcast<0>(tS_max, tS, i));

    /* Rescale existing S sums */
    if (!first_block) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tS_sum.size(); i++) {
        tS_sum(i) *= rescale(i);
      }
    }

    /* Update sums */
    auto tS_bsum = reduce<1>(tS, sycl::plus<void>{});
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_sum.size(); i++)
      tS_sum(i) += tS_bsum(i);

    return rescale;
  }
};

template <
    class DispatchPolicy_,
    bool PagedKV_,
    bool CausalMask_,
    class TiledMMAQK_,  // Tiling for Q*K GEMM
    class TiledMMAPV_,  // Tiling for P*V GEMM
    int VTiles_,        // # of tiles in V dimension
    class TensorQ_,     // Global Q/K/V tensors
    class TensorK_,
    class TensorV_,
    class TiledCopyQ_ = void,  // Optional TiledCopy for loading Q
    class TiledCopyK_ = void,  // Optional TiledCopy for loading K
    class TiledCopyV_ = void,  // Optional TiledCopy for loading V
    bool LocalMask_ = false>
struct DecodeFwdMainloop {
  static_assert(cutlass::detail::dependent_false<DispatchPolicy_>, "Could not find a mainloop specialization.");
};

template <
    int Stages,
    bool PagedKV_,
    bool CausalMask_,
    class TiledMMAQK_,
    class TiledMMAPV_,
    int VTiles_,
    class TensorQ_,
    class TensorK_,
    class TensorV_,
    class TiledCopyQ_,
    class TiledCopyK_,
    class TiledCopyV_,
    bool LocalMask_>
struct DecodeFwdMainloop<
    XeDefault<Stages>,
    PagedKV_,
    CausalMask_,
    TiledMMAQK_,
    TiledMMAPV_,
    VTiles_,
    TensorQ_,
    TensorK_,
    TensorV_,
    TiledCopyQ_,
    TiledCopyK_,
    TiledCopyV_,
    LocalMask_> {
  //
  // Type Aliases
  //
  using TiledMMAQK = TiledMMAQK_;
  using TiledMMAPV = TiledMMAPV_;
  using TileShapeQK = decltype(TiledMMAQK{}.tile_mnk());
  using TileShapePV = decltype(TiledMMAPV{}.tile_mnk());
  static constexpr int VTiles = VTiles_;
  using SubgroupLayoutQK = decltype(TiledMMAQK{}.get_atom_layout_mnk());
  using SGPerWG = decltype(product(take<1, 4>(shape(typename TiledMMAQK::ThrLayoutVMNK{}))));

  using TensorQ = TensorQ_;
  using TensorK = TensorK_;
  using TensorV = TensorV_;

  using ElementQ = typename TensorQ::engine_type::value_type;
  using ElementK = typename TensorK::engine_type::value_type;

  using TensorQ2D = decltype(TensorQ_{}(append<rank_v<TensorQ_>>(make_coord(_, _), 0)));
  using TensorK2D = decltype(TensorK_{}(append<rank_v<TensorK_>>(make_coord(_, _), 0)));
  using TensorV2D = decltype(TensorV_{}(append<rank_v<TensorV_>>(make_coord(_, _), 0)));

  using TiledCopyQ =
      conditional_t<is_void_v<TiledCopyQ_>, decltype(make_block_2d_copy_A(TiledMMAQK{}, TensorQ2D{})), TiledCopyQ_>;
  using TiledCopyK =
      conditional_t<is_void_v<TiledCopyK_>, decltype(make_block_2d_copy_B(TiledMMAQK{}, TensorK2D{})), TiledCopyK_>;
  using TiledCopyV =
      conditional_t<is_void_v<TiledCopyV_>, decltype(make_block_2d_copy_B(TiledMMAPV{}, TensorV2D{})), TiledCopyV_>;

  // TODO: static_asserts on TiledMMAPV here...

  //
  // Accumulator types
  //
  // FragS:    accumulator for Q*K MMA
  // FragO:    accumulator for P*V MMAs.
  //           Note: v mode may be split into multiple pieces
  //             to reduce register pressure.
  // Frag*Row types are reductions of the corresponding Frag* types
  //   over rows.
  //
  template <typename TiledMMA>
  using FragC = decltype(TiledMMA{}.get_slice(0).partition_sg_fragment_C(
      make_identity_tensor(select<0, 1>(TiledMMA{}.tile_mnk()))));

  using FragS = FragC<TiledMMAQK>;
  using FragSRow = decltype(reduce<1>(FragS{}, sycl::plus<void>{}));
  using FragSCol = decltype(reduce<0>(FragS{}, sycl::plus<void>{}));
  using ElementS = typename TiledMMAQK::ValTypeD;

  using SingleFragA = FragC<TiledMMAPV>;                       // (atom val,q',v')
  using FragA = expand_sg_fragment_t<SingleFragA, 1, VTiles>;  // (atom val,q',v',VV)
  using FragARow = decltype(reduce<1>(FragA{}, sycl::plus<void>{}));
  // static_assert(is_same_v<decltype(FragSRow{}.shape()), float>, "dtype
  // mismatched");
  using ElementA = typename TiledMMAPV::ValTypeD;

  static constexpr bool PagedKV = PagedKV_;
  static constexpr bool CausalMask = CausalMask_;
  static constexpr bool Fp8KV = is_any_of_v<ElementK, float_e5m2_t, float_e4m3_t>;
  static constexpr bool LocalMask = LocalMask_;

  // User-facing arguments
  struct Arguments {
    ElementS const scale;
    // Paged KV Cache
    int const* ptr_page_table;
    int page_size;
    int max_pages_per_seq;
    int total_seqlen_kv;
    // Local Mask
    int window_size_left;
    int window_size_right;
  };

  // Kernel-facing parameters
  using Params = Arguments;

  // SLM data
  struct SharedStorage {};

  Params params;

  //
  // Methods
  //

  DecodeFwdMainloop(Params const& params_, SharedStorage&) : params(params_) {}

  static constexpr Params to_underlying_arguments(Arguments const& args, void* /* workspace */) {
    constexpr double kLog2e = 1.4426950408889634074;  // log_2(e)
    ElementS val = args.scale * static_cast<ElementS>(kLog2e);
    return Params{
        val,
        args.ptr_page_table,
        args.page_size,
        args.max_pages_per_seq,
        args.total_seqlen_kv,
        args.window_size_left,
        args.window_size_right};
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) {
    return true;
  }

  template <typename QVCoord>
  CUTLASS_DEVICE void operator()(
      TensorQ2D const& Q_2D,  // (q,d)
      TensorK2D const& K_2D,  // (k,d)
      TensorV2D const& V_2D,  // (d,k)
      FragA& tArA,            // Output accumulator (q,v)
      FragARow& tA_max,       // Softmax row-wise max accumulator
      FragARow& tA_sum,       // Softmax row-wise sum accumulator
      QVCoord blk_qv,         // WG tile indices: (Q,V)
      int const& idx_b,       // WG tile indices: (B)
      int blk_k0,             // K block range: [K0,K1)
      int blk_k1,
      int total_blk,  // Total # of K blocks
      int thr_id,
      int seq_len,
      int full_tile_offset,
      int discard_seq_coord,
      float scale_k = 1.0f) {  // FP8 K per-tensor dequant scale (applied in GEMM1)
    using namespace sycl::ext::oneapi::this_work_item;

    // Short dimension names:
    //    q = sequence len dimension for Q
    //    k = sequence len dimension for K
    //    d = head size dimension for K/Q
    //    v = head size dimension for V
    //   VV = MMA tile indices for V
    // Capital letters (Q, K, ...) refer to WG block indices.
    // Primed letters (q', k', ...) refer to atom block indices.

    auto tile_shape_v = make_shape(get<1>(TileShapePV{}) * C<VTiles>{}, get<2>(TileShapePV{}));

    /* Create proxy coordinate tensors for Q/K/P/V */
    Tensor cQ = make_identity_tensor(Q_2D.shape());               // (q,d)
    Tensor cK = make_identity_tensor(K_2D.shape());               // (k,d)
    Tensor cV = make_identity_tensor(V_2D.shape());               // (v,k)
    Tensor cP = make_identity_tensor(take<0, 2>(TileShapeQK{}));  // (q,k)

    /* Partition global tensors into workgroup tiles */
    Tensor gQ = local_tile(cQ, TileShapeQK{}, append(blk_qv, _), Step<_1, X, _1>{});          // (q,d,D)
    Tensor gK = local_tile(cK, TileShapeQK{}, make_coord(_, _, _), Step<X, _1, _1>{});        // (k,d,K,D)
    Tensor gV = local_tile(cV, tile_shape_v, make_coord(get<1>(blk_qv), _));                  // (v,k,K)
    Tensor gV_split = local_tile(gV, TileShapePV{}, make_coord(_, _, 0), Step<X, _1, _1>{});  // (v,k,VV,K)

    /* Create global -> register copies */
    TiledCopyQ copy_q{Q_2D};
    TiledCopyK copy_k{K_2D};
    TiledCopyV copy_v{V_2D};

    /* Create MMAs */
    TiledMMAQK mma_qk{};
    TiledMMAPV mma_pv{};

    auto copyQ = make_block_2d_copy_A(TiledMMAQK{}, TensorQ2D{});

    /* Slice TiledCopy/TiledMMA operations down to to work-item level */
    auto thr_copy_q = copy_q.get_slice(thr_id);
    auto thr_copy_k = copy_k.get_slice(thr_id);
    auto thr_copy_v = copy_v.get_slice(thr_id);
    auto thr_mma_qk = mma_qk.get_slice(thr_id);
    auto thr_mma_pv = mma_pv.get_slice(thr_id);

    /* Partition coordinate tensors for copy */
    auto tQgQ = thr_copy_q.partition_S(gQ);        // (atom_val,q',d',D)
    auto tKgK = thr_copy_k.partition_S(gK);        // (atom_val,k',d',K,D)
    auto tVgV = thr_copy_v.partition_S(gV_split);  // (atom_val,v',k',VV,K)

    /* Create register fragments for MMA and copies */
    auto tQrQ = thr_copy_q.partition_sg_fragment_D(gQ(_, _, 0));
    auto tSrQ = thr_mma_qk.partition_sg_fragment_A(gQ(_, _, 0));

    auto tKrK = thr_copy_k.partition_sg_fragment_D(gK(_, _, 0, 0));
    auto tSrK = thr_mma_qk.partition_sg_fragment_B(gK(_, _, 0, 0));

    auto tSrS = thr_mma_qk.partition_sg_fragment_C(cP);
    auto tArP = thr_mma_pv.partition_sg_fragment_A(cP);

    auto tVrV = thr_copy_v.partition_sg_fragment_D(gV_split(_, _, 0, 0));
    auto tArV = thr_mma_pv.partition_sg_fragment_B(gV_split(_, _, 0, 0));

    /* Create TiledCopy objects for prefetches */
    auto prefetch_q = make_block_2d_prefetch(copy_q);
    auto prefetch_k = make_block_2d_prefetch(copy_k);
    auto prefetch_v = make_block_2d_prefetch<SGPerWG::value>(tile_shape_v, V_2D);

    /* Partition global tensors for prefetch */
    auto pQgQ = prefetch_q.get_slice(thr_id).partition_S(gQ);
    auto pKgK = prefetch_k.get_slice(thr_id).partition_S(gK);
    auto pVgV = prefetch_v.get_slice(thr_id).partition_S(gV);

    // ------
    // Kernel
    // ------

    // PagedKV
    int tiles_per_page = params.page_size / get<1>(TileShapeQK{});
    int tile_idx = blk_k0;
    int b_offset = idx_b * params.max_pages_per_seq;
    if constexpr (PagedKV) {
      int page_local_idx = tile_idx * get<1>(TileShapeQK{}) / params.page_size;
      tile_idx = params.ptr_page_table[b_offset + page_local_idx] * tiles_per_page + tile_idx % tiles_per_page;
    }

    /* Initialization steps for first block: Q/K prefetch, O init */
    /* TODO: limit D prefetch for large head size, and reorder K prefetches */
    for (int D = 0; D < size<3>(pQgQ); D++) {
      prefetch(prefetch_q, pQgQ(_, _, _, D));
    }

    for (int D = 0; D < size<4>(pKgK); D++) {
      prefetch(prefetch_k, pKgK(_, _, _, tile_idx, D));
    }

    clear(tArA);
    fill(tA_max, cutlass::platform::numeric_limits<ElementA>::lowest());
    clear(tA_sum);

    /* Check if */
    bool check_remainder_k = (seq_len % get<1>(TileShapeQK{}) != 0);

    // FP8 K dequant: S = Q*K is linear in K, so the per-tensor scale_k is folded
    // into the softmax Q*K scale (qk_scale = params.scale * scale_k) instead of
    // rescaling every K register element in GEMM1. The V dequant scale (scale_v)
    // is folded into the softmax normalization in the epilogue, not here.
    ElementS qk_scale = params.scale;
    if constexpr (Fp8KV) {
      qk_scale = params.scale * static_cast<ElementS>(scale_k);
    }

    /* Main loop, blocked in k. */
    int next_tile_idx;
    for (int K = blk_k0; K < blk_k1; K++) {
      /* Split barrier to keep threads together */
      // barrier_arrive(ScopeWorkgroup);

      auto tKgK_cache = PagedKV ? tKgK(_, _, _, tile_idx, _) : tKgK(_, _, _, K, _);
      auto tVgV_cache = PagedKV ? tVgV(_, _, _, _, tile_idx) : tVgV(_, _, _, _, K);

      /* GEMM 1: S = K * Q */
      clear(tSrS); /* TODO: fuse w/ initial gemm call */
      for (int D = 0; D < size<4>(tKgK); D++) {
        copy(copy_q, tQgQ(_, _, _, D), tQrQ);
        copy(copy_k, tKgK_cache(_, _, _, D), tKrK);

        reorder(tQrQ, tSrQ);
        reorder(tKrK, tSrK);

        cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
      }
      /* V prefetch for GEMM 2 */
      prefetch(prefetch_v, pVgV(_, _, _, tile_idx));

      /* Causal masking */
      // No Causal masking in decoding
      // if constexpr (CausalMask) {
      //   if (K == blk_k1 - 1) {
      //     // Need to get global col and row indices to mask the elements
      //     Tensor cPgP = make_identity_tensor(make_shape(seq_len, seq_len));
      //     Tensor gP = local_tile(cPgP, take<0,2>(TileShapeQK{}),
      //     make_coord(get<0>(blk_qv), K)); auto cS_thread =
      //     thr_mma_qk.partition_C(gP); CUTLASS_PRAGMA_UNROLL for (int i = 0; i
      //     < tSrS.size(); ++i) {
      //       int row_idx = get<0>(cS_thread(i));
      //       int col_idx = get<1>(cS_thread(i));
      //       if (col_idx - full_tile_offset > row_idx - discard_seq_coord) {
      //         tSrS(i) = ElementS(-INFINITY);
      //       }
      //     }
      //   }
      // }

      /* Local/sliding window masking */
      if constexpr (LocalMask) {
        // For decode, all packed GQA heads share the same KV position
        // (seq_len_kv - 1). Use a fixed decode row for all elements.
        int decode_row = seq_len - 1 - full_tile_offset;
        Tensor cPgP = make_identity_tensor(make_shape(seq_len, seq_len));
        Tensor gP = local_tile(cPgP, take<0, 2>(TileShapeQK{}), make_coord(get<0>(blk_qv), K));
        auto cS_thread = thr_mma_qk.partition_C(gP);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); ++i) {
          int col_idx = get<1>(cS_thread(i)) - full_tile_offset;
          bool left_mask = col_idx < decode_row - params.window_size_left;
          bool right_mask = col_idx > decode_row + params.window_size_right;
          if (left_mask || right_mask) {
            tSrS(i) = ElementS(-INFINITY);
          }
        }
      }

      /* k masking for remainder tiles */
      if (check_remainder_k && K == blk_k1 - 1) {
        FragSCol k_rem_mask;
        int k = get<0>(tKgK(0, 0, 0, K, 0)) + get_sub_group().get_local_id()[0];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < k_rem_mask.size(); i++, k += intel::sg_size) {
          k_rem_mask(i) = (k < seq_len) ? ElementS(sycl::nan(0u)) : ElementS(-INFINITY);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); i++) {
          tSrS(i) = sycl::fmin(tSrS(i), broadcast<1>(k_rem_mask, tSrS, i));
        }
      }

      /* Apply softmax and scaling */
      softmax(K == 0, tSrS, tA_max, tA_sum, tArA, qk_scale);
      reorder(tSrS, tArP);

      /* GEMM 2: A += P * V, split in v dimension */
      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        copy(copy_v, tVgV_cache(_, _, _, VV), tVrV);
        reorder(tVrV, tArV);
        // FP8 V dequant (scale_v) is deferred to the epilogue.
        cute::gemm(mma_pv, tArP, tArV, tArA(_, _, _, VV));
      }

      barrier();

      // next tile_idx
      next_tile_idx = K + 1;
      if constexpr (PagedKV) {
        int next_page_local_idx = next_tile_idx * get<1>(TileShapeQK{}) / params.page_size;
        if (next_page_local_idx < params.max_pages_per_seq) {
          next_tile_idx =
              params.ptr_page_table[b_offset + next_page_local_idx] * tiles_per_page + next_tile_idx % tiles_per_page;
        } else {
          // set to last page
          next_tile_idx = params.max_pages_per_seq * tiles_per_page - 1;
        }
      }
      tile_idx = next_tile_idx;

      /* K prefetch */
      for (int D = 0; D < size<4>(pKgK); D++) {
        prefetch(prefetch_k, pKgK(_, _, _, tile_idx, D));
      }

      // barrier_wait(ScopeWorkgroup);
    }
  }

  // Single step of blocked softmax.
  CUTLASS_DEVICE
  void softmax(
      bool first_block,     // First softmax block?
      FragS& tS,            // Softmax src/dst block
      FragSRow& tS_max,     // Softmax row-wise max accumulator
      FragSRow& tS_sum,     // Softmax row-wise sum accumulator
      FragA& tA,            // O accumulator (for rescaling)
      ElementS qk_scale) {  // Q*K scale (folds in fp8 K per-tensor scale_k)

    /* Compute row-wise maxima for this block */
    auto tS_bmax = reduce<1>(tS, sycl::maximum{});

    /* Update (scaled) maxima */
    auto tS_prev_max = tS_max;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_max.size(); i++) {
      tS_max(i) = sycl::max(tS_max(i), qk_scale * tS_bmax(i));
    }

    /* Scale S and subtract maxima, then exponentiate */
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS.size(); i++)
      tS(i) = sycl::native::exp2(qk_scale * tS(i) - broadcast<0>(tS_max, tS, i));

    /* Rescale existing S sums and O accumulator */
    if (!first_block) {
      FragSRow rescale;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tS_max.size(); i++) {
        rescale(i) = sycl::native::exp2(tS_prev_max(i) - tS_max(i));
        tS_sum(i) *= rescale(i);
      }

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tA.size(); i++)
        tA(i) *= broadcast<0>(rescale, tA, i);
    }

    /* Update sums */
    auto tS_bsum = reduce<1>(tS, sycl::plus<void>{});
    for (int i = 0; i < tS_sum.size(); i++)
      tS_sum(i) += tS_bsum(i);
  }
};

template <typename SGLayoutQK>
CUTLASS_HOST_DEVICE constexpr auto get_sg_layout_pv(SGLayoutQK const&) {
  return make_layout(get<0>(SGLayoutQK{}), Layout<_1, _0>{}, get<1>(SGLayoutQK{}));
}

// Get a P*V TiledMMA given K*Q tile size and SG configuration, for mainloops
//   not supporting S data interchange among subgroups (e.g. XeDefault).
template <typename MMAOp, typename WGTileQK, typename SGLayoutQK, typename TileV>
CUTLASS_HOST_DEVICE constexpr auto
get_tiled_mma_pv(MMAOp const&, WGTileQK const& wg_tile_qk, SGLayoutQK const& sg_layout_qk, TileV const&) {
  using TileQ = decltype(get<0>(wg_tile_qk));
  using TileK = decltype(get<1>(wg_tile_qk));

  using WGTilePV = Shape<TileQ, TileV, TileK>;
  using SGLayoutPV = decltype(get_sg_layout_pv(sg_layout_qk));

  static_assert(size(SGLayoutPV{}) == size(SGLayoutQK{}), "Q*K cannot be parallelized in the head size dimension");

  return TiledMMAHelper<MMAOp, WGTilePV, SGLayoutPV>{};
}

}  // namespace cutlass::fmha::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
