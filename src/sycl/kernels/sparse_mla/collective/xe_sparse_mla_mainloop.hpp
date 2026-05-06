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
    \brief XPU Sparse MLA Mainloop - Two-phase scattered gather with SWA + Extra indices
*/

#pragma once

#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/algorithm/subgroup_algorithms.hpp"
#include "cute/atom/copy_traits_xe_2d.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/float8.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "sycl/comm/copy_block_slm.hpp"

namespace cutlass::flash_attention {
template <int Stages>
class XeDefault {};

}  // namespace cutlass::flash_attention

namespace cutlass::flash_attention::collective {
using namespace cute;
////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    class DispatchPolicy_,
    bool CausalMask_,
    bool HasExtra_,            // Whether extra KV cache phase is enabled
    class TiledMMAQK_,         // Tiling for Q*K GEMM
    class TiledMMAPV_,         // Tiling for P*V GEMM
    int VTiles_,               // # of tiles in V dimension
    class TensorQ_,            // Global Q tensor (bf16/half)
    class TensorK_,            // Global K_nope tensor (FP8, StrideKV)
    class TensorV_,            // Global V_nope tensor (FP8, StrideKV_V)
    class TensorKPe_,          // Global K_pe tensor (bf16, StrideKV)
    class TensorVPe_,          // Global V_pe tensor (bf16, StrideKV_V)
    class TensorScale_,        // Global KV_scale tensor (uint8, StrideKV)
    class TiledCopyQ_ = void,  // Optional TiledCopy for loading Q
    class TiledCopyK_ = void,  // Optional TiledCopy for loading KV
    class TiledCopyV_ = void>  // Optional TiledCopy for loading V
struct XeMlaSparseMainloop {
  static_assert(cutlass::detail::dependent_false<DispatchPolicy_>, "Could not find a mainloop specialization.");
};

/////////////////////////////////////////////////////////////////////////////////////////////////
template <
    int Stages,
    bool CausalMask_,
    bool HasExtra_,
    class TiledMMAQK_,
    class TiledMMAPV_,
    int VTiles_,
    class TensorQ_,
    class TensorK_,
    class TensorV_,
    class TensorKPe_,
    class TensorVPe_,
    class TensorScale_,
    class TiledCopyQ_,
    class TiledCopyK_,
    class TiledCopyV_>
struct XeMlaSparseMainloop<
    cutlass::flash_attention::XeDefault<Stages>,
    CausalMask_,
    HasExtra_,
    TiledMMAQK_,
    TiledMMAPV_,
    VTiles_,
    TensorQ_,
    TensorK_,
    TensorV_,
    TensorKPe_,
    TensorVPe_,
    TensorScale_,
    TiledCopyQ_,
    TiledCopyK_,
    TiledCopyV_> {
  //
  // Type Aliases
  //
  static constexpr bool HasExtra = HasExtra_;

  using TiledMMAQK = TiledMMAQK_;
  using TiledMMAPV = TiledMMAPV_;
  using TileShapeQK = decltype(TiledMMAQK{}.tile_mnk());
  using TileShapePV = decltype(TiledMMAPV{}.tile_mnk());

  static constexpr auto QK_BLK_M = get<0>(TileShapeQK{});
  static constexpr auto QK_BLK_N = get<1>(TileShapeQK{});
  static constexpr auto QK_BLK_K = get<2>(TileShapeQK{});

  static constexpr int VTiles = VTiles_;
  using SubgroupLayoutQK = decltype(TiledMMAQK{}.get_atom_layout_mnk());
  using SGPerWG = decltype(product(take<1, 4>(shape(typename TiledMMAQK::ThrLayoutVMNK{}))));

  // Number of subgroups for cross-SG reduction
  static constexpr int NumSubgroups = SGPerWG::value;

  // TILE_N = 256 tokens per tile (same DPAS geometry as standard MLA)
  static constexpr int TILE_N = QK_BLK_N;

  using TensorQ = TensorQ_;
  using TensorK = TensorK_;          // K_nope (FP8)
  using TensorV = TensorV_;          // V_nope (FP8, transposed)
  using TensorKPe = TensorKPe_;      // K_pe (bf16)
  using TensorVPe = TensorVPe_;      // V_pe (bf16, transposed)
  using TensorScale = TensorScale_;  // KV_scale (uint8)

  using ElementQ = typename TensorQ_::value_type;
  using ElementK = typename TensorK_::value_type;

  // 2D/3D tensor types: derived by slicing away trailing dimensions
  // (same pattern as normal MLA: TensorK3D = TensorK(_, _, _, 0))
  using TensorQ2D = decltype(TensorQ_{}(append<rank_v<TensorQ_>>(make_coord(_, _), 0)));

  using TensorK2D = decltype(TensorK_{}(append<rank_v<TensorK_>>(make_coord(_, _), 0)));
  using TensorK3D = decltype(TensorK_{}(append<rank_v<TensorK_>>(make_coord(_, _, _), 0)));

  using TensorV2D = decltype(TensorV_{}(append<rank_v<TensorV_>>(make_coord(_, _), 0)));
  using TensorV3D = decltype(TensorV_{}(append<rank_v<TensorV_>>(make_coord(_, _, _), 0)));

  using TensorKPe3D = decltype(TensorKPe_{}(append<rank_v<TensorKPe_>>(make_coord(_, _, _), 0)));
  using TensorVPe3D = decltype(TensorVPe_{}(append<rank_v<TensorVPe_>>(make_coord(_, _, _), 0)));
  using TensorScale3D = decltype(TensorScale_{}(append<rank_v<TensorScale_>>(make_coord(_, _, _), 0)));

  using TiledCopyQ =
      conditional_t<is_void_v<TiledCopyQ_>, decltype(make_block_2d_copy_A(TiledMMAQK{}, TensorQ2D{})), TiledCopyQ_>;
  using TiledCopyK =
      conditional_t<is_void_v<TiledCopyK_>, decltype(make_block_2d_copy_B(TiledMMAQK{}, TensorK2D{})), TiledCopyK_>;
  using TiledCopyV =
      conditional_t<is_void_v<TiledCopyV_>, decltype(make_block_2d_copy_B(TiledMMAPV{}, TensorV2D{})), TiledCopyV_>;

  //
  // Accumulator types
  //
  template <typename TiledMMA>
  using FragC = decltype(TiledMMA{}.get_slice(0).partition_sg_fragment_C(
      make_identity_tensor(select<0, 1>(TiledMMA{}.tile_mnk()))));

  using FragS = FragC<TiledMMAQK>;
  using FragSRow = decltype(reduce<1>(FragS{}, sycl::plus<void>{}));
  using ElementS = typename TiledMMAQK::ValTypeD;

  using SingleFragA = FragC<TiledMMAPV>;                       // (atom val,q',v')
  using FragA = expand_sg_fragment_t<SingleFragA, 1, VTiles>;  // (atom val,q',v',VV)
  using FragARow = decltype(reduce<1>(FragA{}, sycl::plus<void>{}));
  using ElementA = typename TiledMMAPV::ValTypeD;

  static constexpr bool CausalMask = CausalMask_;

  //
  // Arguments
  //
  // User-facing arguments for sparse MLA with two KV pools and token-level indices
  struct Arguments {
    ElementS scale;

    // Primary KV cache pool (used for SWA window tokens)
    int page_size = 0;
    // Physical page size of extra KV cache (may differ from primary)
    int extra_page_size = 0;
    int total_page = 0;

    // SWA (sliding window attention)
    int const* ptr_swa_indices = nullptr;
    int swa_topk = 0;

    // Extra KV cache pool
    int const* ptr_extra_indices = nullptr;
    int extra_topk = 0;

    int const* ptr_topk = nullptr;
    int const* ptr_extra_topk = nullptr;

    // FP8 flag: true = KV cache is FP8 packed (584 bytes/token with dequant)
    //           false = KV cache is bf16 (576 elements/token, direct copy)
    bool is_fp8_cache = true;
  };

  //
  // Params
  //
  // Kernel-facing parameters
  using Params = Arguments;

  // D_NOPE = 448 (7 tiles × 64, production V4 value)
  static constexpr int D_NOPE = 448;
  static constexpr int D_ROPE = 64;
  static constexpr int D_SLICE = QK_BLK_K;  // One d-slice width = DPAS K-tile = 64

  //
  // Shared memory storage:
  //   - sg_max_data: cross-subgroup softmax reduction
  //   - tile_slice: ONE d-slice [TILE_N, D_SLICE=64] gathered at a time
  //     Reused for each GEMM1 d-iteration AND each GEMM2 v-iteration.
  //     Total: TILE_N * 64 * sizeof(ElementQ) = 256 * 64 * 2 = 32 KB
  //
  struct SharedStorage {
    cute::array<ElementS, NumSubgroups> sg_max_data;
    ElementQ tile_slice[TILE_N * D_SLICE];  // [TILE_N, 64] — one d-slice at a time
  };

  Params params;
  SharedStorage& shared;

  //
  // Methods
  //
  XeMlaSparseMainloop(Params const& params_, SharedStorage& shared_) : params(params_), shared(shared_) {}

  static constexpr Params to_underlying_arguments(Arguments const& args, void* /* workspace */) {
    // exp(x) = exp2(x * log2(e))
    constexpr double kLog2e = 1.4426950408889634074;  // log_2(e)
    ElementS val = args.scale * static_cast<ElementS>(kLog2e);
    return Params{
        val,
        args.page_size,
        args.extra_page_size,
        args.total_page,
        args.ptr_swa_indices,
        args.swa_topk,
        args.ptr_extra_indices,
        args.extra_topk,
        args.ptr_topk,
        args.ptr_extra_topk,
        args.is_fp8_cache};
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) {
    return true;
  }

  //
  // Tile address structure: computed once per TILE_N tile, reused for K_c, K_pe, KV_scale, and V loads.
  // Analogous to normal MLA's (physical_block_idx, intra_page_tile_idx) but for scattered tokens.
  //
  struct TileAddresses {
    int page_ids[TILE_N];       // Physical page index for each token in the tile
    int token_offsets[TILE_N];  // Intra-page token offset for each token
    bool valid[TILE_N];         // Whether this slot has a real token (vs padding)
    int num_valid = 0;          // Count of valid tokens in this tile
  };

  //
  // Get physical KV tile addresses from index array — called ONCE per tile.
  // Mirrors normal MLA's get_physical_k_tile but for scattered (non-contiguous) token indices.
  //
  // In normal MLA: sequential KV cache → one (page_id, tile_offset) per TILE_N block.
  // In sparse MLA: scattered indices → per-token (page_id, token_offset) for TILE_N tokens.
  //
  // The resulting addresses are reused for all gather calls:
  //   K_nope[token_offset, d, page_id], K_pe[token_offset, d, page_id],
  //   KV_scale[token_offset, tile, page_id],
  //   V_nope[d, token_offset, page_id], V_pe[d, token_offset, page_id]
  //
  CUTLASS_DEVICE
  TileAddresses get_physical_kv_tile_addr(
      int const* indices_ptr,  // Batch-local index array (swa_indices or extra_indices for this batch)
      int tile_start,          // Starting position in the index array (tile_idx * TILE_N)
      int num_indices,         // Total valid indices for this batch (from topk_length)
      int page_size) {         // Page size of this KV cache pool
    TileAddresses addrs;
    addrs.num_valid = 0;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < TILE_N; i++) {
      int pos = tile_start + i;
      if (pos < num_indices) {
        int idx = indices_ptr[pos];
        addrs.valid[i] = (idx >= 0);
        // Clamp invalid indices to 0 (will be masked out in scores anyway)
        int safe_idx = (idx >= 0) ? idx : 0;
        addrs.page_ids[i] = safe_idx / page_size;
        addrs.token_offsets[i] = safe_idx % page_size;
        if (idx >= 0) addrs.num_valid++;
      } else {
        // Beyond the index array — mark invalid
        addrs.valid[i] = false;
        addrs.page_ids[i] = -1;
        addrs.token_offsets[i] = -1;
      }
    }
    return addrs;
  }

  //
  // Gather ONE d-slice [TILE_N, D_SLICE=64] from scattered page positions into SLM.
  //
  // This is called once per DPAS iteration (8 times for K_c, 1 for K_pe, 8 for V).
  // Each call fills shared.tile_slice with 64 bf16 elements per token.
  //
  // Two modes:
  //   dequant=false: KV cache is bf16. Direct element copy.
  //   dequant=true:  KV cache is FP8 packed (DeepSeek V4 production format).
  //                  Reads FP8_E4M3 bytes + UE8M0 scale, dequants to bf16.
  //
  // FP8 packed page layout (per page of page_size tokens):
  //   Bytes [0 .. page_size * 576):
  //     Per token: [448B FP8 nope | 128B BF16 rope]  (stride = 576 bytes/token)
  //   Bytes [page_size * 576 .. page_size * 576 + page_size * 8):
  //     Per token: [7B UE8M0 scales + 1B pad]        (stride = 8 bytes/token)
  //
  // Parameters:
  //   kv_cache_ptr: base pointer (raw bytes cast as ElementSrc for typing)
  //   addrs: pre-computed (page_id, token_offset, valid) for TILE_N tokens
  //   stride_page: byte stride between pages (= page_size * 576 + ceil_576(page_size * 8))\n  //   d_offset: element
  //   offset within each token's data (0..447 for nope, 0..63 for rope) dequant: true = FP8 path, false = bf16 direct
  //   copy is_rope: true = reading rope section (bf16 always, no dequant even if dequant=true) thr_id: work-item ID
  //   within work-group
  //
  // FP8 dequantization formula:
  //   result[col] = fp8_e4m3_to_float(nope_byte[col]) * exp2(scale_byte[tile_idx] - 127)
  //   where tile_idx = col / 64 (7 tiles of 64 for 448 nope dims)
  //

  //
  // E4M3FN dequantization: fast conversion from raw FP8 byte to float.
  // Directly reconstructs IEEE 754 float32 bits — eliminates ldexp calls.
  //
  // E4M3FN layout: sign(1) exponent(4) mantissa(3), bias=7, no NaN/Inf.
  // Normal (exp!=0): (-1)^s * 2^(e-7) * (1 + m/8)
  // Subnormal (exp==0): (-1)^s * 2^(-6) * (m/8) = ±m * 2^(-9)
  //
  CUTLASS_DEVICE static float fp8_e4m3fn_to_float(uint8_t raw) {
    uint32_t sign_bit = static_cast<uint32_t>(raw & 0x80u) << 24;
    uint32_t exp_bits = (raw >> 3) & 0xFu;
    uint32_t mant_bits = raw & 0x7u;

    // Normal path (fast): directly construct float32 bit pattern.
    // float32 exponent = fp8_exp - 7 + 127 = fp8_exp + 120
    // float32 mantissa = fp8_mant << 20 (3-bit → 23-bit field)
    uint32_t f32_bits = sign_bit | ((exp_bits + 120u) << 23) | (mant_bits << 20);

    // Subnormal path (exp==0, rare: 7/128 magnitudes):
    // value = ±mant * 2^(-9). For mant==0 → 0.0 (handled by f32_bits = sign_bit | 0).
    // Use a simple multiply for the rare case.
    if (exp_bits == 0u) {
      float sub_val = static_cast<float>(mant_bits) * 0.001953125f;
      __builtin_memcpy(&f32_bits, &sub_val, 4);
      f32_bits |= sign_bit;
    }

    float result;
    __builtin_memcpy(&result, &f32_bits, 4);
    return result;
  }

  //
  // Process a tile of TILE_N tokens using SLICE-BY-SLICE scattered gather.
  // Only ONE d-slice [TILE_N, D_SLICE=64] is in SLM at a time.
  // Addresses computed ONCE via get_physical_kv_tile_addr(), passed in here.
  //
  // Flow per tile:
  //   GEMM1: S = Σ Q_nope[d] @ K_nope[d]^T + Q_pe @ K_pe^T
  //     7 nope d-slices + 1 pe d-slice = 8 GEMM1 accumulations
  //   Softmax: P = softmax(S)
  //   GEMM2: O += P @ V
  //     7 nope d-slices + 1 pe d-slice = 8 GEMM2 accumulations (one per VTile)
  //
  template <typename QVCoord, typename S2rCopyK, typename SLMTensorK, typename S2rCopyV, typename SLMTensorV>
  CUTLASS_DEVICE void process_token_tile(
      TensorQ2D const& Q_2D,
      TensorQ2D const& Q_pe_2D,
      TensorK3D const& K_nope,        // (page_size, D_NOPE, num_pages)
      TensorKPe3D const& K_pe,        // (page_size, D_ROPE, num_pages)
      TensorScale3D const& KV_scale,  // (page_size, 7, num_pages)
      TensorV3D const& V_nope,        // (D_NOPE, page_size, num_pages) — transposed
      TensorVPe3D const& V_pe,        // (D_ROPE, page_size, num_pages) — transposed
      FragA& tArA,
      FragARow& tA_max,
      FragARow& tA_sum,
      QVCoord blk_qv,
      TileAddresses const& addrs,  // Pre-computed from get_physical_kv_tile_addr()
      bool is_first_tile,
      int thr_id,
      int cache_page_size,
      // SLM copy infrastructure (pre-constructed)
      S2rCopyK const& s2r_k,
      SLMTensorK& sK,
      S2rCopyV const& s2r_v,
      SLMTensorV& sV) {
    /* Create MMAs */
    TiledMMAQK mma_qk{};
    TiledMMAPV mma_pv{};
    auto thr_mma_qk = mma_qk.get_slice(thr_id);
    auto thr_mma_pv = mma_pv.get_slice(thr_id);

    /* Create proxy coordinate tensors */
    Tensor cQnope = make_identity_tensor(Q_2D.shape());                // (q, d_nope)
    Tensor cQpe = make_identity_tensor(Q_pe_2D.shape());               // (q, d_rope)
    Tensor cP = make_identity_tensor(take<0, 2>(TileShapeQK{}));       // (q, k)
    Tensor cB_qk = make_identity_tensor(select<1, 2>(TileShapeQK{}));  // (TILE_N, D_SLICE)
    Tensor cB_pv = make_identity_tensor(select<1, 2>(TileShapePV{}));  // (D_SLICE, TILE_N)

    /* Partition global tensors into workgroup tiles */
    Tensor gQnope = local_tile(cQnope, TileShapeQK{}, append(blk_qv, _), Step<_1, X, _1>{});
    Tensor gQpe = local_tile(cQpe, TileShapeQK{}, append(blk_qv, _), Step<_1, X, _1>{});

    /* Create global -> register copies for Q */
    TiledCopyQ copy_qnope{Q_2D};
    TiledCopyQ copy_qpe{Q_pe_2D};
    auto thr_copy_qnope = copy_qnope.get_slice(thr_id);
    auto thr_copy_qpe = copy_qpe.get_slice(thr_id);

    /* Partition Q for copy */
    auto tQnopegQ = thr_copy_qnope.partition_S(gQnope);
    auto tQpegQ = thr_copy_qpe.partition_S(gQpe);

    /* Q register fragments */
    auto tQnoperQ = thr_copy_qnope.partition_sg_fragment_D(gQnope(_, _, 0));
    auto tSrQnope = thr_mma_qk.partition_sg_fragment_A(gQnope(_, _, 0));
    auto tQperQ = thr_copy_qpe.partition_sg_fragment_D(gQpe(_, _, 0));
    auto tSrQpe = thr_mma_qk.partition_sg_fragment_A(gQpe(_, _, 0));

    /* Score/P fragments */
    auto tSrS = thr_mma_qk.partition_sg_fragment_C(cP);
    auto tArP = thr_mma_pv.partition_sg_fragment_A(cP);

    /* K and V register fragments (from SLM via s2r) */
    auto thr_s2r_k = s2r_k.get_slice(thr_id);
    auto tKsK = thr_s2r_k.partition_S(sK);
    auto tSrK = thr_mma_qk.partition_sg_fragment_B(cB_qk);
    auto tKrK_mma = thr_s2r_k.retile_D(tSrK);

    auto thr_s2r_v = s2r_v.get_slice(thr_id);
    auto tVsV = thr_s2r_v.partition_S(sV);
    auto tArV = thr_mma_pv.partition_sg_fragment_B(cB_pv);
    auto tVrV_mma = thr_s2r_v.retile_D(tArV);

    /* ================================================================
     * Pre-compute per-token pointers and scales (reused across all 16 d-slice iterations).
     * Eliminates repeated CuTe stride arithmetic in gather functions.
     * Each work-item caches its token's base pointers + 7 scale factors.
     * ================================================================ */
    static constexpr int NUM_NOPE_SLICES = D_NOPE / D_SLICE;  // 448/64 = 7
    float cached_scales[NUM_NOPE_SLICES];
    const uint8_t* my_k_nope_ptr = nullptr;  // base of K_nope for this token
    const ElementQ* my_k_pe_ptr = nullptr;   // base of K_pe for this token
    const uint8_t* my_v_nope_ptr = nullptr;  // base of V_nope for this token
    const ElementQ* my_v_pe_ptr = nullptr;   // base of V_pe for this token
    {
      int token_idx = thr_id;
      if (token_idx < TILE_N && addrs.valid[token_idx]) {
        int page_id = addrs.page_ids[token_idx];
        int tok_off = addrs.token_offsets[token_idx];
        // Precompute base pointers (d-dim is contiguous from here)
        my_k_nope_ptr = reinterpret_cast<const uint8_t*>(&K_nope(tok_off, 0, page_id));
        my_k_pe_ptr = &K_pe(tok_off, 0, page_id);
        my_v_nope_ptr = reinterpret_cast<const uint8_t*>(&V_nope(0, tok_off, page_id));
        my_v_pe_ptr = &V_pe(0, tok_off, page_id);
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < NUM_NOPE_SLICES; s++) {
          uint8_t scale_byte = KV_scale(tok_off, s, page_id);
          cached_scales[s] = sycl::native::exp2(static_cast<float>(scale_byte) - 127.0f);
        }
      } else {
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < NUM_NOPE_SLICES; s++) {
          cached_scales[s] = 0.0f;
        }
      }
    }

    /* ================================================================
     * GEMM 1: S = Q_nope @ K_nope^T + Q_pe @ K_pe^T
     * ================================================================ */
    clear(tSrS);

    // Part 1: Accumulate Q_nope @ K_nope^T (7 d-slices)
    CUTLASS_PRAGMA_UNROLL
    for (int d_slice = 0; d_slice < NUM_NOPE_SLICES; d_slice++) {
      // Inline gather using precomputed pointer
      {
        int token_idx = thr_id;
        if (token_idx < TILE_N) {
          if (my_k_nope_ptr) {
            const uint8_t* slice_ptr = my_k_nope_ptr + d_slice * D_SLICE;
            float sf = cached_scales[d_slice];
            CUTLASS_PRAGMA_UNROLL
            for (int chunk = 0; chunk < D_SLICE; chunk += 8) {
              uint64_t packed = *reinterpret_cast<const uint64_t*>(slice_ptr + chunk);
              CUTLASS_PRAGMA_UNROLL
              for (int b = 0; b < 8; b++) {
                uint8_t raw = static_cast<uint8_t>(packed >> (b * 8));
                float f_val = fp8_e4m3fn_to_float(raw) * sf;
                shared.tile_slice[token_idx + (chunk + b) * TILE_N] = static_cast<ElementQ>(f_val);
              }
            }
          } else {
            CUTLASS_PRAGMA_UNROLL
            for (int col = 0; col < D_SLICE; col++) {
              shared.tile_slice[token_idx + col * TILE_N] = ElementQ(0);
            }
          }
        }
      }
      sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_work_group<3>());
      copy(s2r_k, tKsK, tKrK_mma);
      copy(copy_qnope, tQnopegQ(_, _, _, d_slice), tQnoperQ);
      reorder(tQnoperQ, tSrQnope);
      cute::gemm(mma_qk, tSrQnope, tSrK, tSrS);
      sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_work_group<3>());
    }

    // Part 2: Accumulate Q_pe @ K_pe^T (1 d-slice)
    {
      int token_idx = thr_id;
      if (token_idx < TILE_N) {
        if (my_k_pe_ptr) {
          CUTLASS_PRAGMA_UNROLL
          for (int chunk = 0; chunk < D_SLICE; chunk += 4) {
            uint64_t packed = *reinterpret_cast<const uint64_t*>(my_k_pe_ptr + chunk);
            const ElementQ* vals = reinterpret_cast<const ElementQ*>(&packed);
            CUTLASS_PRAGMA_UNROLL
            for (int b = 0; b < 4; b++) {
              shared.tile_slice[token_idx + (chunk + b) * TILE_N] = vals[b];
            }
          }
        } else {
          CUTLASS_PRAGMA_UNROLL
          for (int col = 0; col < D_SLICE; col++) {
            shared.tile_slice[token_idx + col * TILE_N] = ElementQ(0);
          }
        }
      }
      sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_work_group<3>());
      copy(s2r_k, tKsK, tKrK_mma);
      copy(copy_qpe, tQpegQ(_, _, _, 0), tQperQ);
      reorder(tQperQ, tSrQpe);
      cute::gemm(mma_qk, tSrQpe, tSrK, tSrS);
      sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_work_group<3>());
    }

    /* ================================================================
     * Mask invalid tokens: set scores to -INFINITY
     * ================================================================ */
    {
      constexpr int SG_SIZE = intel::sg_size;
      int sg_id = thr_id / SG_SIZE;
      int lane_id = thr_id % SG_SIZE;
      int token_idx = sg_id * SG_SIZE + lane_id;
      bool token_valid = (token_idx < TILE_N && addrs.valid[token_idx]);
      if (!token_valid) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); i++) {
          tSrS(i) = -cutlass::platform::numeric_limits<ElementS>::infinity();
        }
      }
    }

    /* ================================================================
     * Softmax
     * ================================================================ */
    softmax(is_first_tile, tSrS, tA_max, tA_sum, tArA);
    reorder(tSrS, tArP);

    /* ================================================================
     * GEMM 2: O += P @ V (reuses cached pointers and scales)
     * ================================================================ */
    CUTLASS_PRAGMA_UNROLL
    for (int vv = 0; vv < VTiles; vv++) {
      {
        int token_idx = thr_id;
        if (token_idx < TILE_N) {
          if (vv < NUM_NOPE_SLICES) {
            if (my_v_nope_ptr) {
              const uint8_t* slice_ptr = my_v_nope_ptr + vv * D_SLICE;
              float sf = cached_scales[vv];
              CUTLASS_PRAGMA_UNROLL
              for (int chunk = 0; chunk < D_SLICE; chunk += 8) {
                uint64_t packed = *reinterpret_cast<const uint64_t*>(slice_ptr + chunk);
                CUTLASS_PRAGMA_UNROLL
                for (int b = 0; b < 8; b++) {
                  uint8_t raw = static_cast<uint8_t>(packed >> (b * 8));
                  float f_val = fp8_e4m3fn_to_float(raw) * sf;
                  shared.tile_slice[(chunk + b) + token_idx * D_SLICE] = static_cast<ElementQ>(f_val);
                }
              }
            } else {
              CUTLASS_PRAGMA_UNROLL
              for (int v = 0; v < D_SLICE; v++) {
                shared.tile_slice[v + token_idx * D_SLICE] = ElementQ(0);
              }
            }
          } else {
            if (my_v_pe_ptr) {
              CUTLASS_PRAGMA_UNROLL
              for (int chunk = 0; chunk < D_SLICE; chunk += 4) {
                uint64_t packed = *reinterpret_cast<const uint64_t*>(my_v_pe_ptr + chunk);
                const ElementQ* vals = reinterpret_cast<const ElementQ*>(&packed);
                CUTLASS_PRAGMA_UNROLL
                for (int b = 0; b < 4; b++) {
                  shared.tile_slice[(chunk + b) + token_idx * D_SLICE] = vals[b];
                }
              }
            } else {
              CUTLASS_PRAGMA_UNROLL
              for (int v = 0; v < D_SLICE; v++) {
                shared.tile_slice[v + token_idx * D_SLICE] = ElementQ(0);
              }
            }
          }
        }
      }
      sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_work_group<3>());

      copy(s2r_v, tVsV, tVrV_mma);

      cute::gemm(mma_pv, tArP, tArV, tArA(_, _, _, vv));

      sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_work_group<3>());
    }
  }

  //
  // Sparse MLA Mainloop Operator - Two-phase: SWA indices then Extra indices
  //
  // Phase 1: Process tokens from SWA (sliding window) indices using primary K cache
  // Phase 2: Process tokens from Extra indices using extra K cache
  //
  template <typename QVCoord>
  CUTLASS_DEVICE void operator()(
      TensorQ2D const& Q_2D,                // Q nope (q, d_nope)
      TensorQ2D const& Q_pe_2D,             // Q rope (q, d_rope)
      TensorK3D const& K_nope,              // Primary K nope (page_size, D_NOPE, num_pages)
      TensorKPe3D const& K_pe,              // Primary K rope (page_size, D_ROPE, num_pages)
      TensorScale3D const& KV_scale,        // Primary scales (page_size, 7, num_pages)
      TensorV3D const& V_nope,              // Primary V nope (D_NOPE, page_size, num_pages) — transposed
      TensorVPe3D const& V_pe,              // Primary V rope (D_ROPE, page_size, num_pages) — transposed
      TensorK3D const& K_nope_extra,        // Extra K nope (extra_page_size, D_NOPE, num_extra_pages)
      TensorKPe3D const& K_pe_extra,        // Extra K rope (extra_page_size, D_ROPE, num_extra_pages)
      TensorScale3D const& KV_scale_extra,  // Extra scales (extra_page_size, 7, num_extra_pages)
      TensorV3D const& V_nope_extra,        // Extra V nope (D_NOPE, extra_page_size, num_extra_pages) — transposed
      TensorVPe3D const& V_pe_extra,        // Extra V rope (D_ROPE, extra_page_size, num_extra_pages) — transposed
      FragA& tArA,                          // Output accumulator (q,v)
      FragARow& tA_max,                     // Softmax row-wise max accumulator
      FragARow& tA_sum,                     // Softmax row-wise sum accumulator
      QVCoord blk_qv,                       // WG tile indices: (Q,V)
      int thr_id,
      int batch_coord) {
    /* Initialize accumulators */
    clear(tArA);
    fill(tA_max, cutlass::platform::numeric_limits<ElementA>::lowest());
    clear(tA_sum);

    /* Create MMAs for SLM copy setup */
    TiledMMAQK mma_qk{};
    TiledMMAPV mma_pv{};

    /* -----------------------------------------------------------
     * Set up SLM s2r copies for QK and PV B operands
     * Using make_B_slm_copies with dummy global copies for type deduction.
     * SLM layout is compact N-major:
     *   QK: SLM[token + d_col * TILE_N]
     *   PV: SLM[v_dim + token * D_SLICE]
     * ----------------------------------------------------------- */
    // Dummy bf16 tensors for creating block 2D copies (shape matches B operand)
    auto dummy_k = make_tensor(
        make_gmem_ptr(static_cast<ElementQ const*>(nullptr)), make_layout(make_shape(Int<TILE_N>{}, Int<D_SLICE>{})));
    auto dummy_v = make_tensor(
        make_gmem_ptr(static_cast<ElementQ const*>(nullptr)), make_layout(make_shape(Int<D_SLICE>{}, Int<TILE_N>{})));

    auto global_copy_k = make_coop_block_2d_copy_B(mma_qk, dummy_k);
    auto global_copy_v = make_coop_block_2d_copy_B(mma_pv, dummy_v);

    auto [r2s_k, s2r_k] = make_B_slm_copies(mma_qk, global_copy_k);
    auto [r2s_v, s2r_v] = make_B_slm_copies(mma_pv, global_copy_v);

    // SLM tensors: use Tiler_MN from the r2s copy as the shape (matches xe_gemm_slm pattern)
    auto sK = make_tensor(make_smem_ptr(shared.tile_slice), make_layout(typename decltype(r2s_k)::Tiler_MN{}));
    auto sV = make_tensor(make_smem_ptr(shared.tile_slice), make_layout(typename decltype(r2s_v)::Tiler_MN{}));

    // Track whether this is the first tile (for softmax initialization)
    bool is_first_tile = true;

    // ========================================================================
    // Phase 1: SWA (sliding window) indices — primary KV cache
    // ========================================================================
    int const* swa_indices_batch = params.ptr_swa_indices + batch_coord * params.swa_topk;
    int swa_num_valid = params.ptr_topk[batch_coord];
    int swa_num_tiles = (swa_num_valid + TILE_N - 1) / TILE_N;

    for (int tile_idx = 0; tile_idx < swa_num_tiles; tile_idx++) {
      int tile_start = tile_idx * TILE_N;

      TileAddresses addrs = get_physical_kv_tile_addr(swa_indices_batch, tile_start, swa_num_valid, params.page_size);
      if (addrs.num_valid == 0) continue;

      process_token_tile(
          Q_2D,
          Q_pe_2D,
          K_nope,
          K_pe,
          KV_scale,
          V_nope,
          V_pe,
          tArA,
          tA_max,
          tA_sum,
          blk_qv,
          addrs,
          is_first_tile,
          thr_id,
          params.page_size,
          s2r_k,
          sK,
          s2r_v,
          sV);

      is_first_tile = false;
    }

    // ========================================================================
    // Phase 2: Extra indices — extra KV cache
    // ========================================================================
    if constexpr (HasExtra) {
      int const* extra_indices_batch = params.ptr_extra_indices + batch_coord * params.extra_topk;
      int extra_num_valid = params.ptr_extra_topk[batch_coord];
      int extra_num_tiles = (extra_num_valid + TILE_N - 1) / TILE_N;

      for (int tile_idx = 0; tile_idx < extra_num_tiles; tile_idx++) {
        int tile_start = tile_idx * TILE_N;

        TileAddresses addrs =
            get_physical_kv_tile_addr(extra_indices_batch, tile_start, extra_num_valid, params.extra_page_size);
        if (addrs.num_valid == 0) continue;

        process_token_tile(
            Q_2D,
            Q_pe_2D,
            K_nope_extra,
            K_pe_extra,
            KV_scale_extra,
            V_nope_extra,
            V_pe_extra,
            tArA,
            tA_max,
            tA_sum,
            blk_qv,
            addrs,
            is_first_tile,
            thr_id,
            params.extra_page_size,
            s2r_k,
            sK,
            s2r_v,
            sV);

        is_first_tile = false;
      }
    }
  }

  //
  // Online Softmax Algorithm (identical to standard MLA):
  //   1. Apply scale: tS_scaled = tS * scale
  //   2. Compute current tile max: m_curr = max(tS_scaled)
  //   3. Compute new global max: m_new = max(m_prev, m_curr)
  //   4. Compute correction factor: correction = exp2(m_prev - m_new)
  //   5. Compute P_tile (unnormalized): P_tile = exp2(tS_scaled - m_new)
  //   6. Compute sum for current tile: l_curr = sum(P_tile)
  //   7. Update running sum: l_new = l_prev * correction + l_curr
  //   8. Update output accumulator: O_new = O_prev * correction
  //   9. Update state: m_prev <- m_new, l_prev <- l_new
  //   10. Final output after all tiles: O_final = O_new / l_new [epilogue step]
  CUTLASS_DEVICE
  void softmax(
      bool first_block,  // First softmax block?
      FragS& tS,         // Softmax src/dst block
      FragSRow& tS_max,  // Softmax row-wise max accumulator
      FragSRow& tS_sum,  // Softmax row-wise sum accumulator
      FragA& tA) {       // O accumulator (for rescaling)

    /* Compute row-wise maxima for this block */
    auto tS_bmax = reduce<1>(tS, sycl::maximum{});

    /* Update (scaled) maxima */
    auto tS_prev_max = tS_max;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_max.size(); i++) {
      tS_max(i) = sycl::max(tS_max(i), params.scale * tS_bmax(i));
    }

    /* Scale S and subtract maxima, then exponentiate */
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS.size(); i++)
      tS(i) = sycl::native::exp2(params.scale * tS(i) - broadcast<0>(tS_max, tS, i));

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

/////////////////////////////////////////////////////////////////////////////////////////////////
// Helper to derive PV subgroup layout from QK subgroup layout
// (redeclared here for self-containedness; identical to standard MLA)
template <typename SGLayoutQK>
CUTLASS_HOST_DEVICE constexpr auto get_sg_layout_pv(SGLayoutQK const&) {
  return make_layout(get<0>(SGLayoutQK{}), Layout<_1, _0>{}, get<1>(SGLayoutQK{}));
}

}  // namespace cutlass::flash_attention::collective
