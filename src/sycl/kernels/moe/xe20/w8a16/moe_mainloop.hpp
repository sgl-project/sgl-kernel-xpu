/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

// 8-bit weight (FP8 E4M3 or INT8/UINT8), BF16 activation grouped-GEMM mainloop
// for Intel Xe2 (BMG).
//
// Supports all W8A16 weight scale layouts in a single unified mainloop:
//   - ScalarSingle ([E, 1]) / ScalarGateUp ([E, 2]) / PerChannelN ([E, N]):
//     unscaled K-loop + FP32 epilogue scaling.
//   - BlockFP32 ([E, ceil(N/128), K/128]), GroupNK ([E, N, K/G]),
//     BlockMXFP8 ([E, N, K/32] UE8M0), and asymmetric UINT8 with zero-point:
//     scales (and optional zero-points) folded directly into the 8-bit -> BF16
//     VISA subgroup reorder using a single FP32 accumulator `tCrC`.

#pragma once

#include <cute/tensor.hpp>
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include "cutlass/bfloat16.h"
#include "cutlass/float8.h"
#include "cutlass/half.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_ref.h"
#include "scale_mode.h"
#include "sycl/SYCLHelpers.h"
#include "w8a16_dequant.hpp"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace moe_w8a16 {

using namespace cute;

static constexpr int FP8_GROUP_SIZE_K = 128;

// Number of work-items per subgroup on Xe (SIMD lane count).
static constexpr int SUBGROUP_SIZE = 16;

template <int Stages>
class W8A16MainloopPolicy {};

template <
    class DispatchPolicy_,
    class TiledCopyA_,
    class TiledCopyBPacked_,
    class TiledCopyD_,
    class ATensor_,
    class BPackedTensor_,
    class DTensor_,
    class TiledMMA_,
    bool IsInt8 = false>
struct W8A16Mainloop {
  static_assert(cutlass::detail::dependent_false<DispatchPolicy_>, "Could not find a mainloop specialization.");
};

template <
    int Stages,
    class TiledCopyA_,
    class TiledCopyBPacked_,
    class TiledCopyD_,
    class ATensor_,
    class BPackedTensor_,
    class DTensor_,
    class TiledMMA_,
    bool IsInt8>
struct W8A16Mainloop<
    W8A16MainloopPolicy<Stages>,
    TiledCopyA_,
    TiledCopyBPacked_,
    TiledCopyD_,
    ATensor_,
    BPackedTensor_,
    DTensor_,
    TiledMMA_,
    IsInt8> {
  using TiledMMA = TiledMMA_;
  using TiledCopyA = TiledCopyA_;
  using TiledCopyBPacked = TiledCopyBPacked_;
  using TiledCopyD = TiledCopyD_;
  using ATensor = ATensor_;
  using BPackedTensor = BPackedTensor_;
  using DTensor = DTensor_;

  W8A16Mainloop() {}

  CUTE_DEVICE static float read_fp_scale(const uint8_t* scale_bytes, int64_t idx, bool scale_is_bf16) {
    if (scale_is_bf16) {
      return static_cast<float>(reinterpret_cast<const cutlass::bfloat16_t*>(scale_bytes)[idx]);
    }
    return reinterpret_cast<const float*>(scale_bytes)[idx];
  }

  template <typename Coord>
  CUTLASS_DEVICE void operator()(
      ATensor& A,
      BPackedTensor& Bp,
      const uint8_t* w_scale_bytes,
      const uint8_t* w_zero_gmem,
      int w_scale_n_stride,
      int w_scale_k_stride,
      int w_zero_n_stride,
      int w_zero_k_stride,
      DTensor& D,
      Coord blk_coord,
      TiledMMA mma,
      int thr_id,
      const float* Bias,
      int gemm_n,
      int scale_mode,
      int group_size,
      bool scale_is_bf16,
      bool is_signed_int8) {
    auto wg_m = get<0>(blk_coord);
    auto wg_n = get<1>(blk_coord);
    auto wg_tile = mma.tile_mnk();
    auto wg_coord = make_coord(wg_m, wg_n, 0);
    constexpr int BLK_M = get<0>(decltype(wg_tile){});
    constexpr int BLK_N = get<1>(decltype(wg_tile){});
    constexpr int BLK_K = get<2>(decltype(wg_tile){});
    constexpr int ATOM_M_V = get<1>(typename TiledMMA::ThrLayoutVMNK{}.shape());
    constexpr int ATOM_N_V = get<2>(typename TiledMMA::ThrLayoutVMNK{}.shape());
    constexpr int SG_M = BLK_M / ATOM_M_V;
    constexpr int SG_N = BLK_N / ATOM_N_V;

    Tensor cA = make_identity_tensor(A.shape());
    Tensor cBp = make_identity_tensor(Bp.shape());
    Tensor cD = make_identity_tensor(D.shape());
    Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(wg_m, _));
    Tensor gBp = local_tile(cBp, select<1, 2>(wg_tile), make_coord(wg_n, _));
    Tensor gD = local_tile(cD, wg_tile, wg_coord, Step<_1, _1, X>{});

    TiledCopyA tiled_copy_a{A};
    TiledCopyBPacked tiled_copy_b{Bp};
    TiledCopyD tiled_copy_d{D};
    auto thr_copy_a = tiled_copy_a.get_slice(thr_id);
    auto thr_copy_b = tiled_copy_b.get_slice(thr_id);
    auto thr_copy_d = tiled_copy_d.get_slice(thr_id);
    auto thr_mma = mma.get_slice(thr_id);
    auto tAgA = thr_copy_a.partition_S(gA);
    auto tBgBp = thr_copy_b.partition_S(gBp);
    auto tArA_packed = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
    auto tBrB_packed = thr_copy_b.partition_sg_fragment_D(gBp(_, _, 0));
    auto tSrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
    auto tSrB = thr_mma.partition_sg_fragment_B(gBp(_, _, 0));
    SubgroupTensor tCrC = thr_mma.partition_sg_fragment_C(gD);
    cute::clear(tCrC);

    auto prefetch_a = make_block_2d_prefetch(tiled_copy_a);
    auto prefetch_b = make_block_2d_prefetch(tiled_copy_b);
    auto pAgA = prefetch_a.get_slice(thr_id).partition_S(gA);
    auto pBgBp = prefetch_b.get_slice(thr_id).partition_S(gBp);
    constexpr SPIRVScope barrier_scope = ScopeWorkgroup;
    const int k_tile_count = ceil_div(shape<1>(A), BLK_K);

    CUTE_UNROLL
    for (int prefetch_k = 0; prefetch_k < Stages; ++prefetch_k) {
      if (prefetch_k < k_tile_count) {
        prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
        prefetch(prefetch_b, pBgBp(_, _, _, prefetch_k));
      }
    }

    const int sg_local_n_coord = cutlass::get_sub_group_id() % ATOM_N_V;
    const int sg_local_id = cutlass::get_sub_group_local_id();
    const int n_tile_start = wg_n * BLK_N;
    const int n_sg_start = sg_local_n_coord * SG_N;
    const bool use_epilogue_scale = is_epilogue_scale_mode(scale_mode) && (!IsInt8 || w_zero_gmem == nullptr);

    if (use_epilogue_scale) {
      for (int k_tile = 0, k_tile_prefetch = Stages; k_tile < k_tile_count; ++k_tile, ++k_tile_prefetch) {
        barrier_arrive(barrier_scope);
        copy(tiled_copy_a, tAgA(_, _, _, k_tile), tArA_packed);
        copy(tiled_copy_b, tBgBp(_, _, _, k_tile), tBrB_packed);
        if (k_tile_prefetch < k_tile_count) {
          prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
          prefetch(prefetch_b, pBgBp(_, _, _, k_tile_prefetch));
        }
        reorder(tArA_packed, tSrA);
        if constexpr (IsInt8) {
          if (is_signed_int8) {
            reorder(tBrB_packed, tSrB);
          } else {
            uint8b128_reorder(tBrB_packed, tSrB);
          }
        } else {
          reorder(tBrB_packed, tSrB);
        }
        cute::gemm(mma, tSrA, tSrB, tCrC);
        barrier_wait(barrier_scope);
      }

      CUTLASS_PRAGMA_UNROLL
      for (int sn = 0; sn < SG_N / SUBGROUP_SIZE; ++sn) {
        const int global_n = n_tile_start + n_sg_start + sn * SUBGROUP_SIZE + sg_local_id;
        const int clamped_n = cute::min(global_n, gemm_n - 1);
        const int scale_idx =
            (scale_mode == static_cast<int>(ScaleMode::PerChannelN))
                ? clamped_n
                : ((scale_mode == static_cast<int>(ScaleMode::ScalarGateUp) && clamped_n >= gemm_n / 2) ? 1 : 0);
        const float weight_scale = read_fp_scale(w_scale_bytes, scale_idx, scale_is_bf16);
        CUTLASS_PRAGMA_UNROLL
        for (int sm = 0; sm < SG_M; ++sm) {
          tCrC(sn * SG_M + sm) *= weight_scale;
        }
      }
    } else {
      using BFragLayout = decltype(tSrB.layout());
      static constexpr int channel_num = get<0>(get<0>(tSrB.shape()));
      static constexpr int thr_N = get<1>(tSrB.shape());
      static constexpr int frag_mode0 = size<0>(tSrB.shape());
      static_assert(channel_num == 2, "W8A16 folded multiply expects 2 VNNI channels per lane");
      static_assert(
          stride<0, 0>(BFragLayout{}) == 1 && stride<0, 1>(BFragLayout{}) == channel_num &&
              (thr_N == 1 || stride<1>(BFragLayout{}) == frag_mode0),
          "W8A16 folded dequant needs channel-innermost B fragment with n above mode 0");
      static_assert(frag_mode0 % 4 == 0, "a 4-value 8-bit reorder chunk must stay inside one n-block");

      intel::vector_t<float, 2> mul_pairs[thr_N];
      intel::vector_t<float, 2> zp_pairs[IsInt8 ? thr_N : 1];
      auto mul_of = [&](auto dv) -> intel::vector_t<float, 2> const& {
        constexpr int n = (decltype(dv)::value / frag_mode0) % thr_N;
        return mul_pairs[n];
      };
      auto zp_of = [&](auto dv) -> intel::vector_t<float, 2> const& {
        constexpr int n = (decltype(dv)::value / frag_mode0) % (IsInt8 ? thr_N : 1);
        return zp_pairs[n];
      };

      const int x_idx = sg_local_id / channel_num;
      const bool is_n_blocked_128 = (scale_mode == static_cast<int>(ScaleMode::BlockFP32));
      const bool is_mxfp8 = !IsInt8 && (scale_mode == static_cast<int>(ScaleMode::BlockMXFP8));
      const int scale_elem_bytes = is_mxfp8 ? 1 : (scale_is_bf16 ? 2 : 4);
      const int64_t scale_k_step_bytes = static_cast<int64_t>(w_scale_k_stride) * scale_elem_bytes;
      const int reload_mask =
          (scale_mode == static_cast<int>(ScaleMode::PerChannelN)) ? 0x7FFFFFFF : (group_size / BLK_K - 1);

      auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
      const bool has_zp = IsInt8 && (w_zero_gmem != nullptr);

      const uint8_t* scale_ptr[thr_N * channel_num];
      const uint8_t* zero_ptr[IsInt8 ? (thr_N * channel_num) : 1];
      int slot_shift[IsInt8 ? 1 : (thr_N * channel_num)];
      if constexpr (!IsInt8) {
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < thr_N * channel_num; ++s) {
          slot_shift[s] = kShiftUnset;
        }
      }

      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < thr_N; ++n) {
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < channel_num; ++c) {
          const int col = n_tile_start + n_sg_start + n * SUBGROUP_SIZE + x_idx + c * (SUBGROUP_SIZE / channel_num);
          const int clamped_col = cute::min(col, gemm_n - 1);
          const int scale_row = is_n_blocked_128 ? (clamped_col >> 7) : clamped_col;
          scale_ptr[n * channel_num + c] =
              w_scale_bytes + static_cast<int64_t>(scale_row) * w_scale_n_stride * scale_elem_bytes;
          if constexpr (IsInt8) {
            zero_ptr[n * channel_num + c] = (w_zero_gmem == nullptr)
                                                ? nullptr
                                                : (w_zero_gmem + static_cast<int64_t>(clamped_col) * w_zero_n_stride);
          }
        }
      }

      const uint8_t* init_scale_ptr[thr_N * channel_num];
      CUTLASS_PRAGMA_UNROLL
      for (int s = 0; s < thr_N * channel_num; ++s) {
        init_scale_ptr[s] = scale_ptr[s];
      }

      uint32_t out_of_fast_range = 0u;
      if constexpr (!IsInt8) {
        if (is_mxfp8 && BLK_K == 32) {
          // Single-basic-block MXFP8 K-loop: 1 scale per K-tile (BLK_K == 32),
          // zero intra-loop branches so IGC pipelines loads, reorder, and DPAS.
          for (int k_tile = 0, k_tile_prefetch = Stages; k_tile < k_tile_count; ++k_tile, ++k_tile_prefetch) {
            barrier_arrive(barrier_scope);
            copy(tiled_copy_a, tAgA(_, _, _, k_tile), tArA_packed);
            copy(tiled_copy_b, tBgBp(_, _, _, k_tile), tBrB_packed);

            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < thr_N; ++n) {
              CUTLASS_PRAGMA_UNROLL
              for (int c = 0; c < channel_num; ++c) {
                const int slot = n * channel_num + c;
                const uint8_t* sp = scale_ptr[slot];
                scale_ptr[slot] = sp + scale_k_step_bytes;
                const uint32_t e8m0 = static_cast<uint32_t>(sp[0]);
                out_of_fast_range |= static_cast<uint32_t>(e8m0 != 0xFFu && (e8m0 - 74u) > 124u);
                const uint32_t bits = (e8m0 == 0xFFu) ? 0x7FC00000u : ((e8m0 + 56u) << 23);
                mul_pairs[n][c] = sycl::bit_cast<float>(bits);
              }
            }

            if (k_tile_prefetch < k_tile_count) {
              prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
              prefetch(prefetch_b, pBgBp(_, _, _, k_tile_prefetch));
            }

            reorder(tArA_packed, tSrA);
            e4m3_reorder_dequant(tBrB_packed, tSrB, mul_of);
            cute::gemm(mma, tSrA, tSrB, tCrC);
            barrier_wait(barrier_scope);
          }
        } else {
          for (int k_tile = 0, k_tile_prefetch = Stages; k_tile < k_tile_count; ++k_tile, ++k_tile_prefetch) {
            barrier_arrive(barrier_scope);
            copy(tiled_copy_a, tAgA(_, _, _, k_tile), tArA_packed);
            copy(tiled_copy_b, tBgBp(_, _, _, k_tile), tBrB_packed);

            if ((k_tile & reload_mask) == 0) {
              CUTLASS_PRAGMA_UNROLL
              for (int n = 0; n < thr_N; ++n) {
                CUTLASS_PRAGMA_UNROLL
                for (int c = 0; c < channel_num; ++c) {
                  const int slot = n * channel_num + c;
                  const uint8_t* sp = scale_ptr[slot];
                  scale_ptr[slot] = sp + scale_k_step_bytes;
                  if (is_mxfp8) {
                    const uint32_t e8m0 = static_cast<uint32_t>(sp[0]);
                    out_of_fast_range |= static_cast<uint32_t>(e8m0 != 0xFFu && (e8m0 - 74u) > 124u);
                    const uint32_t bits = (e8m0 == 0xFFu) ? 0x7FC00000u : ((e8m0 + 56u) << 23);
                    mul_pairs[n][c] = sycl::bit_cast<float>(bits);
                  } else {
                    const float s = read_fp_scale(sp, 0, scale_is_bf16);
                    const uint32_t s_bits = sycl::bit_cast<uint32_t>(s);
                    const uint32_t exp_b = (s_bits >> 23) & 0xFFu;
                    out_of_fast_range |=
                        static_cast<uint32_t>((s_bits & 0x7FFFFFFFu) != 0u && exp_b != 0xFFu && (exp_b - 74u) > 124u);
                    mul_pairs[n][c] = s * 0x1p56f;
                  }
                }
              }
            }

            if (k_tile_prefetch < k_tile_count) {
              prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
              prefetch(prefetch_b, pBgBp(_, _, _, k_tile_prefetch));
            }

            reorder(tArA_packed, tSrA);
            e4m3_reorder_dequant(tBrB_packed, tSrB, mul_of);
            cute::gemm(mma, tSrA, tSrB, tCrC);
            barrier_wait(barrier_scope);
          }
        }
      } else if (!has_zp) {
        for (int k_tile = 0, k_tile_prefetch = Stages; k_tile < k_tile_count; ++k_tile, ++k_tile_prefetch) {
          barrier_arrive(barrier_scope);
          copy(tiled_copy_a, tAgA(_, _, _, k_tile), tArA_packed);
          copy(tiled_copy_b, tBgBp(_, _, _, k_tile), tBrB_packed);

          if ((k_tile & reload_mask) == 0) {
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < thr_N; ++n) {
              CUTLASS_PRAGMA_UNROLL
              for (int c = 0; c < channel_num; ++c) {
                const int slot = n * channel_num + c;
                const uint8_t* sp = scale_ptr[slot];
                scale_ptr[slot] = sp + scale_k_step_bytes;
                mul_pairs[n][c] = read_fp_scale(sp, 0, scale_is_bf16) * 256.0f;
              }
            }
          }

          if (k_tile_prefetch < k_tile_count) {
            prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
            prefetch(prefetch_b, pBgBp(_, _, _, k_tile_prefetch));
          }

          reorder(tArA_packed, tSrA);
          int8_sym_reorder_dequant(tBrB_packed, tSrB, mul_of, is_signed_int8);
          cute::gemm(mma, tSrA, tSrB, tCrC);
          barrier_wait(barrier_scope);
        }
      } else {
        for (int k_tile = 0, k_tile_prefetch = Stages; k_tile < k_tile_count; ++k_tile, ++k_tile_prefetch) {
          barrier_arrive(barrier_scope);
          copy(tiled_copy_a, tAgA(_, _, _, k_tile), tArA_packed);
          copy(tiled_copy_b, tBgBp(_, _, _, k_tile), tBrB_packed);

          if ((k_tile & reload_mask) == 0) {
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < thr_N; ++n) {
              CUTLASS_PRAGMA_UNROLL
              for (int c = 0; c < channel_num; ++c) {
                const int slot = n * channel_num + c;
                const uint8_t* sp = scale_ptr[slot];
                scale_ptr[slot] = sp + scale_k_step_bytes;
                mul_pairs[n][c] = read_fp_scale(sp, 0, scale_is_bf16) * 256.0f;
                const uint8_t* zp = zero_ptr[slot];
                zero_ptr[slot] = zp + w_zero_k_stride;
                zp_pairs[n][c] = -static_cast<float>(zp[0]) * (1.0f / 256.0f);
              }
            }
          }

          if (k_tile_prefetch < k_tile_count) {
            prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
            prefetch(prefetch_b, pBgBp(_, _, _, k_tile_prefetch));
          }

          reorder(tArA_packed, tSrA);
          uint8_zp_reorder_dequant(tBrB_packed, tSrB, zp_of, mul_of);
          cute::gemm(mma, tSrA, tSrB, tCrC);
          barrier_wait(barrier_scope);
        }
      }

      if constexpr (!IsInt8) {
        if (!sycl::any_of_group(sg, out_of_fast_range != 0u)) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tCrC.size(); ++i) {
            tCrC(i) *= 0x1p64f;
          }
        } else {
          // Exact multi-bucket fallback for extreme scales (< 2^-53 or > 2^71).
          cute::clear(tCrC);
          CUTLASS_PRAGMA_UNROLL
          for (int s = 0; s < thr_N * channel_num; ++s) {
            scale_ptr[s] = init_scale_ptr[s];
            slot_shift[s] = kShiftUnset;
          }

          for (int k_tile = 0, k_tile_prefetch = Stages; k_tile < k_tile_count; ++k_tile, ++k_tile_prefetch) {
            barrier_arrive(barrier_scope);
            copy(tiled_copy_a, tAgA(_, _, _, k_tile), tArA_packed);
            copy(tiled_copy_b, tBgBp(_, _, _, k_tile), tBrB_packed);

            if ((k_tile & reload_mask) == 0) {
              uint32_t raw_scale[thr_N * channel_num];
              int new_shift[thr_N * channel_num];
              bool need_rescale = false;

              CUTLASS_PRAGMA_UNROLL
              for (int n = 0; n < thr_N; ++n) {
                CUTLASS_PRAGMA_UNROLL
                for (int c = 0; c < channel_num; ++c) {
                  const int slot = n * channel_num + c;
                  const uint8_t* sp = scale_ptr[slot];
                  scale_ptr[slot] = sp + scale_k_step_bytes;
                  int req = kShiftUnset;
                  if (is_mxfp8) {
                    const uint32_t e8m0 = static_cast<uint32_t>(sp[0]);
                    raw_scale[slot] = e8m0;
                    if (e8m0 != 0xFFu) {
                      req = fp8_required_shift_from_biased_exp(static_cast<int>(e8m0));
                    }
                  } else {
                    const float s = read_fp_scale(sp, 0, scale_is_bf16);
                    const uint32_t s_bits = sycl::bit_cast<uint32_t>(s);
                    raw_scale[slot] = s_bits;
                    const int exp_b = static_cast<int>((s_bits >> 23) & 0xFFu);
                    if ((s_bits & 0x7FFFFFFFu) != 0u && exp_b != 0xFF) {
                      req = fp8_required_shift_from_biased_exp(exp_b);
                    }
                  }
                  const int cur = slot_shift[slot];
                  if (cur == kShiftUnset) {
                    new_shift[slot] = req;
                  } else if (req != kShiftUnset && req > cur) {
                    new_shift[slot] = req;
                    need_rescale = true;
                  } else {
                    new_shift[slot] = cur;
                  }
                }
              }

              if (k_tile > 0 && sycl::any_of_group(sg, need_rescale)) {
                const int src_lane = (sg_local_id & 7) * 2;
                CUTLASS_PRAGMA_UNROLL
                for (int n = 0; n < thr_N; ++n) {
                  const int old0 = sycl::select_from_group(sg, slot_shift[n * 2 + 0], src_lane);
                  const int old1 = sycl::select_from_group(sg, slot_shift[n * 2 + 1], src_lane);
                  const int new0 = sycl::select_from_group(sg, new_shift[n * 2 + 0], src_lane);
                  const int new1 = sycl::select_from_group(sg, new_shift[n * 2 + 1], src_lane);
                  const int my_old = (sg_local_id < 8) ? old0 : old1;
                  const int my_new = (sg_local_id < 8) ? new0 : new1;
                  if (my_old != kShiftUnset && my_new > my_old) {
                    const float rescale = fp8_shift_scale_factor(my_old - my_new);
                    CUTLASS_PRAGMA_UNROLL
                    for (int sm = 0; sm < SG_M; ++sm) {
                      tCrC(n * SG_M + sm) *= rescale;
                    }
                  }
                }
              }

              CUTLASS_PRAGMA_UNROLL
              for (int n = 0; n < thr_N; ++n) {
                CUTLASS_PRAGMA_UNROLL
                for (int c = 0; c < channel_num; ++c) {
                  const int slot = n * channel_num + c;
                  slot_shift[slot] = new_shift[slot];
                  const int eff_shift = (new_shift[slot] == kShiftUnset) ? 0 : new_shift[slot];
                  mul_pairs[n][c] = is_mxfp8
                                        ? mxfp8_fold_multiplier(raw_scale[slot], eff_shift)
                                        : fp32_e4m3_fold_multiplier(sycl::bit_cast<float>(raw_scale[slot]), eff_shift);
                }
              }
            }

            if (k_tile_prefetch < k_tile_count) {
              prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
              prefetch(prefetch_b, pBgBp(_, _, _, k_tile_prefetch));
            }

            reorder(tArA_packed, tSrA);
            e4m3_reorder_dequant(tBrB_packed, tSrB, mul_of);
            cute::gemm(mma, tSrA, tSrB, tCrC);
            barrier_wait(barrier_scope);
          }

          bool any_shifted = false;
          CUTLASS_PRAGMA_UNROLL
          for (int s = 0; s < thr_N * channel_num; ++s) {
            any_shifted |= (slot_shift[s] != 0 && slot_shift[s] != kShiftUnset);
          }
          if (sycl::any_of_group(sg, any_shifted)) {
            const int src_lane = (sg_local_id & 7) * 2;
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < thr_N; ++n) {
              const int s0 = sycl::select_from_group(sg, slot_shift[n * 2 + 0], src_lane);
              const int s1 = sycl::select_from_group(sg, slot_shift[n * 2 + 1], src_lane);
              const int my_shift = (sg_local_id < 8) ? s0 : s1;
              const int clean_shift = (my_shift == kShiftUnset) ? 0 : my_shift;
              const float unfold = fp8_shift_scale_factor(clean_shift);
              CUTLASS_PRAGMA_UNROLL
              for (int sm = 0; sm < SG_M; ++sm) {
                tCrC(n * SG_M + sm) *= unfold;
              }
            }
          }
        }
      }
    }

    if (Bias != nullptr) {
      add_bias<SG_M, SG_N, BLK_N>(Bias, tCrC, wg_n, thr_id, gemm_n);
    }
    SubgroupTensor tCrD = thr_copy_d.partition_sg_fragment_S(gD);
    Tensor tCgD = thr_copy_d.partition_D(gD);
    reorder(tCrC, tCrD);
    copy(tiled_copy_d, tCrD, tCgD);
  }

  template <int SG_M, int SG_N, int BLK_N, typename tCrC_t>
  void add_bias(const float* Bias, tCrC_t& tCrC, int wg_n, int thr_id, int gemm_n) {
    static constexpr auto ATOM_N = get<2>(typename TiledMMA::ThrLayoutVMNK{}.shape());
    constexpr int N_ATOMS = SG_N / SUBGROUP_SIZE;
    int sg_local_n_coord = (thr_id / SUBGROUP_SIZE) % ATOM_N;
    int lane = thr_id % SUBGROUP_SIZE;

    CUTLASS_PRAGMA_UNROLL
    for (int na = 0; na < N_ATOMS; ++na) {
      int n = wg_n * BLK_N + sg_local_n_coord * SG_N + na * SUBGROUP_SIZE + lane;
      float bias = (n < gemm_n) ? Bias[n] : 0.0f;
      CUTLASS_PRAGMA_UNROLL
      for (int sm = 0; sm < SG_M; ++sm) {
        tCrC(na * SG_M + sm) += bias;
      }
    }
  }
};

}  // namespace moe_w8a16
