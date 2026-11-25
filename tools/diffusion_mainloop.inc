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

// Dense D128 mainloop and dtype-specific softmax schedules.
    static_assert(!CausalMask && !CachedKV && !PagedKV && !BlockScale &&
                  !F8kvF16mma && !PerTensorScale && !FP4Input,
                  "Only dense noncausal unscaled FP16/BF16 is supported");
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
    Tensor cQ = make_identity_tensor(Q_2D.shape());             // (q,d)
    Tensor cK = make_identity_tensor(K_2D.shape());             // (k,d)
    Tensor cV = make_identity_tensor(V_2D.shape());             // (v,k)
    Tensor cK_cache = make_identity_tensor(K_cache_2D.shape()); // (k,d)
    Tensor cV_cache = make_identity_tensor(V_cache_2D.shape()); // (v,k)
    Tensor cP = make_identity_tensor(take<0,2>(TileShapeQK{})); // (q,k)

    /* Partition global tensors into workgroup tiles */
    Tensor gQ       = local_tile(cQ, TileShapeQK{}, append(blk_qv,_),             Step<_1,X,_1>{});   // (q,d,D)
    Tensor gK       = local_tile(cK, TileShapeQK{}, make_coord(_,_,_),            Step<X,_1,_1>{});   // (k,d,K,D)
    Tensor gV       = local_tile(cV, tile_shape_v,  make_coord(get<1>(blk_qv),_));                    // (v,k,K)
    Tensor gV_split = local_tile(gV, TileShapePV{}, make_coord(_,_,0),            Step<X,_1,_1>{});   // (v,k,VV,K)

    Tensor gK_cache       = local_tile(cK_cache, TileShapeQK{}, make_coord(_,_,_),            Step<X,_1,_1>{});   // (k,d,K,D)
    Tensor gV_cache       = local_tile(cV_cache, tile_shape_v,  make_coord(get<1>(blk_qv),_));                    // (v,k,K)
    Tensor gV_cache_split = local_tile(gV_cache, TileShapePV{}, make_coord(_,_,0),            Step<X,_1,_1>{});   // (v,k,VV,K)

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
    auto tQgQ = thr_copy_q.partition_S(gQ);                // (atom_val,q',d',D)
    auto tKgK = thr_copy_k.partition_S(gK);                // (atom_val,k',d',K,D)
    auto tVgV = thr_copy_v.partition_S(gV_split);          // (atom_val,v',k',VV,K)
    auto tKgK_cache = thr_copy_k_cache.partition_S(gK_cache);
    auto tVgV_cache = thr_copy_v_cache.partition_S(gV_cache_split);

    /* Create register fragments for MMA and copies */
    auto tQrQ = thr_copy_q.partition_sg_fragment_D(gQ(_,_,0));
    auto tSrQ = thr_mma_qk.partition_sg_fragment_A(gQ(_,_,0));

    auto tKrK = thr_copy_k.partition_sg_fragment_D(gK(_,_,0,0));
    auto tSrK = thr_mma_qk.partition_sg_fragment_B(gK(_,_,0,0));

    auto tSrS = thr_mma_qk.partition_sg_fragment_C(cP);
    auto tArP = thr_mma_pv.partition_sg_fragment_A(cP);

    auto tVrV = thr_copy_v.partition_sg_fragment_D(gV_split(_,_,0,0));
    auto tArV = thr_mma_pv.partition_sg_fragment_B(gV_split(_,_,0,0));

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

    /* Initialization steps for first block: Q/K prefetch, O init */
    /* TODO: limit D prefetch for large head size, and reorder K prefetches */
    int kblocks_cache = ceil_div(seq_len_kv_cache, get<1>(TileShapeQK{}));
    for (int D = 0; D < size<3>(pQgQ); D++) {
      prefetch(prefetch_q, pQgQ(_,_,_,D));
    }
    for (int D = 0; D < size<4>(pKgK); D++) {
      CUTLASS_PRAGMA_UNROLL
      for (int K = 0; K < Stages; K++) {
        if (K < kblocks_cache) {
          if constexpr (PagedKV) {
            int physical_K_tile = get_physical_k_tile(K, l_coord, seq_len_kv_cache);
            prefetch(prefetch_k_cache, pKgK_cache(_,_,_,physical_K_tile,D));
          } else {
            prefetch(prefetch_k_cache, pKgK_cache(_,_,_,K,D));
          }
        } else {
          prefetch(prefetch_k, pKgK(_,_,_,K - kblocks_cache,D));
        }
      }
    }
    if (blk_k0 == 0) {
      clear(tArA);
      fill(tA_max, cutlass::platform::numeric_limits<ElementA>::lowest());
      clear(tA_sum);
    }

    /* Check if */
    bool check_remainder_k = (seq_len % get<1>(TileShapeQK{}) != 0);

    /* Main loop body */
    auto mainloop_body = [&](auto cached_k, int K,
                            auto& copy_k_cur, auto& copy_v_cur,
                            auto& prefetch_v_cur, auto& tKgK_cur,
                            auto& tVgV_cur, auto& pVgV_cur) {
      /* Split barrier to keep threads together */
      barrier_arrive(ScopeWorkgroup);
      constexpr bool is_cache = decltype(cached_k)::value;

      int k_idx;
      if constexpr (is_cache) {
        k_idx = K;
        if constexpr (PagedKV) {
          k_idx = get_physical_k_tile(K, l_coord, seq_len_kv_cache);
        }
      } else {
        k_idx = K - kblocks_cache;
      }

      /* GEMM 1: S = K * Q */
      clear(tSrS);
      CUTLASS_PRAGMA_UNROLL
#if SGLD_ATTENTION_TWO_EDIT
      for (int D = 0; D < DTiles; D++) {
#else
      for (int D = 0; D < size<4>(tKgK); D++) {
#endif
        copy(copy_q, tQgQ(_,_,_,D), tQrQ);
        copy(copy_k_cur, tKgK_cur(_,_,_,k_idx,D), tKrK);
        reorder(tQrQ, tSrQ);
        reorder(tKrK, tSrK);

        cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
      }

      /* V prefetch for GEMM 2 */
      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        prefetch(prefetch_v_cur, pVgV_cur(_,_,_,VV,k_idx));
      }
      /* Causal masking - only in non-cache mode */
      if constexpr (!is_cache && CausalMask) {
        if (K == total_blk - 1) {
          // Need to get global col and row indices to mask the elements
          Tensor cPgP = make_identity_tensor(make_shape(seq_len, seq_len));
          Tensor gP = local_tile(cPgP, take<0,2>(TileShapeQK{}), make_coord(get<0>(blk_qv), K));
          auto cS_thread = thr_mma_qk.partition_C(gP);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tSrS.size(); ++i) {
            int row_idx = get<0>(cS_thread(i));
            int col_idx = get<1>(cS_thread(i));
            if (col_idx - seq_len_kv_cache - full_tile_offset > row_idx - discard_seq_coord) {
              tSrS(i) = ElementS(-INFINITY);
            }
          }
        }
      }
      /* k masking for remainder tiles */
      if constexpr (!is_cache) {
        if (check_remainder_k && K == total_blk - 1) {
          // Use CuTe's MMA coordinates for every key in a KV32 tail.
          Tensor cPgP = make_identity_tensor(make_shape(seq_len, seq_len));
          Tensor gP = local_tile(cPgP, take<0,2>(TileShapeQK{}),
                                 make_coord(get<0>(blk_qv), K));
          auto cS_thread = thr_mma_qk.partition_C(gP);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tSrS.size(); ++i) {
            if (get<1>(cS_thread(i)) >= seq_len) tSrS(i) = ElementS(-INFINITY);
          }
        }
      }

      /* Apply softmax and scaling (tA rescaling fused into GEMM2 VTile loop) */
#if SGLD_ATTENTION_TWO_EDIT
      auto [rescale, tS_partial_sum] = softmax(tSrS, tA_max, tA_sum);
      auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
      constexpr int kSumSize = decltype(tA_sum.size())::value;
      constexpr bool kSumDivVT = (kSumSize % VTiles == 0);
      constexpr int kSumPerVT = kSumDivVT ? (kSumSize / VTiles) : 0;
#else
      auto rescale = softmax(K == blk_k0, tSrS, tA_max, tA_sum);
#endif
      reorder(tSrS, tArP);

      /* GEMM 2: A += P * V, split in v dimension.
        tArA rescaling is fused to per-VTile */
      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        copy(copy_v_cur, tVgV_cur(_,_,_,VV,k_idx), tVrV);
        reorder(tVrV, tArV);
        if (K != blk_k0) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tArA.size() / VTiles; i++)
            tArA(_,_,_,VV)(i) *= broadcast<0>(rescale, tArA, i);
        }

#if SGLD_ATTENTION_TWO_EDIT
        if constexpr (kSumDivVT) {
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < kSumPerVT; j++) {
            int const i = VV * kSumPerVT + j;
            tA_sum(i) = tA_sum(i) * group_broadcast(sg, rescale(0), i) + tS_partial_sum(i);
          }
        }
#endif
        cute::gemm(mma_pv, tArP, tArV, tArA(_,_,_,VV));
      }

      /* K prefetch */
      int K_next = K + Stages;
      for (int D = 0; D < size<4>(pKgK); D++) {
        if constexpr (is_cache) {
          bool is_cache_next = K_next < kblocks_cache;
          int physical_K_next = K_next;
          if constexpr (PagedKV) {
            if (is_cache_next) {
              physical_K_next = get_physical_k_tile(K_next, l_coord, seq_len_kv_cache);
            }
          }
          if (is_cache_next) {
            prefetch(prefetch_k_cache, pKgK_cache(_,_,_,physical_K_next,D));
          } else {
            prefetch(prefetch_k, pKgK(_,_,_,K_next-kblocks_cache,D));
          }
        } else {
          prefetch(prefetch_k, pKgK(_,_,_,K_next-kblocks_cache,D));
        }
      }
      barrier_wait(ScopeWorkgroup);
    };

    /* Main loop, blocked in k. */
    if constexpr (CachedKV) {
      for (int K = blk_k0; K < kblocks_cache; K++) {
        mainloop_body(std::bool_constant<true>{}, K,
                      copy_k_cache, copy_v_cache,
                      prefetch_v_cache, tKgK_cache,
                      tVgV_cache, pVgV_cache);
      }
    }

    for (int K = (blk_k0 > kblocks_cache ? blk_k0 : kblocks_cache); K < blk_k1; K++) {
      mainloop_body(std::bool_constant<false>{}, K,
                    copy_k, copy_v,
                    prefetch_v, tKgK,
                    tVgV, pVgV);
    }
  }

#if !SGLD_ATTENTION_TWO_EDIT
  // Single step of blocked softmax.
  CUTLASS_DEVICE
  FragSRow
  softmax(bool       first_block, // First softmax block?
          FragS    & tS,          // Softmax src/dst block
          FragSRow & tS_max,      // Softmax row-wise max accumulator
          FragSRow & tS_sum) {    // Softmax row-wise sum accumulator
    /* Compute row-wise maxima for this block */
    auto tS_bmax = reduce<1>(tS, sycl::maximum{});

    FragSRow rescale;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_max.size(); i++) {
      ElementS new_max = sycl::max(tS_max(i), params.scale * tS_bmax(i));
      rescale(i) = sycl::native::exp2(tS_max(i) - new_max);
      tS_max(i) = new_max;
    }

    /* Scale S and subtract maxima, then exponentiate */
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS.size(); i++)
      tS(i) = sycl::native::exp2(params.scale * tS(i) - broadcast<0>(tS_max, tS, i));

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
#else
  // Keep lane-local partial sums until the output epilogue.
  CUTLASS_DEVICE
  auto
  softmax(FragS &tS, FragSRow &tS_max, FragSPartialRow &tS_sum) {
    auto tS_bmax = reduce<1>(tS, sycl::maximum{});
    FragSRow rescale;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_max.size(); i++) {
      ElementS new_max = sycl::max(tS_max(i), params.scale * tS_bmax(i));
      rescale(i) = sycl::native::exp2(tS_max(i) - new_max);
      tS_max(i) = new_max;
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS.size(); i++)
      tS(i) = sycl::native::exp2(params.scale * tS(i) - broadcast<0>(tS_max, tS, i));

    auto tS_partial_sum = reduce<1, ReduceMode::Vertical>(tS, sycl::plus<void>{});
    if constexpr (decltype(tS_sum.size())::value % VTiles != 0) {
      auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tS_sum.size(); i++)
        tS_sum(i) = tS_sum(i) * group_broadcast(sg, rescale(0), i) + tS_partial_sum(i);
    }
    return cute::make_tuple(rescale, tS_partial_sum);
  }
#endif
