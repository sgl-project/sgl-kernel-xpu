/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
/*!
  \file
  \brief Two-stage sparse MLA decode Stage 1 kernel for DeepSeek V4.

  SparseDecodeGatherDequantKernel: subgroup-coalesced gather + FP8->bf16 dequant
  into a dense [b, s_q, gathered_topk, 512] bf16 tile + int valid mask. Shared
  declarations (params, constants) come from
  xe_mla_sparse_decode_2stage_common.hpp. Stage 1 owns its own tile constants and
  does not use the Stage-2 config struct (MlaSparseDecode2StageXe).

  Correctness reference: tests/test_flash_mla_with_kvcache.py _gather_and_dequant.
*/

#pragma once

#include "sycl/kernels/mla_sparse/kernel/xe_mla_sparse_decode_2stage_common.hpp"

namespace cutlass::flash_attention::kernel {

template <int D_QK>
class SparseDecodeGatherDequantKernel {
 public:
  using Arguments = SparseAttnDecodeParams;
  using KernelArguments = SparseAttnDecodeParams;
  using Params = SparseAttnDecodeParams;

  static constexpr int NUM_THREADS = 128;
  static constexpr int SUBGROUP_SIZE = intel::sg_size;
  static constexpr int NUM_SUBGROUPS = NUM_THREADS / SUBGROUP_SIZE;
  static constexpr int B_TOPK = 64;

  // Gather uses no SLM.
  static constexpr int SharedStorageSize = 0;

  // launch<> contract (matches the manual policy the 2-stage launcher used before
  // the gather became a device::MLASparse companion): one work-group per
  // (batch*seq, topk-block); B_TOPK topk columns per work-group.
  static dim3 get_grid_shape(Params const& params) {
    return dim3(params.shape.b * params.shape.s_q, ceil_div(params.shape.gathered_topk, B_TOPK), 1);
  }

  static dim3 get_block_shape() {
    return dim3(NUM_THREADS, 1, 1);
  }
  static constexpr int FP8_VALUES_PER_PACK = 8;
  static constexpr int BF16_VALUES_PER_PACK = 4;
  using PackedElement = uint64_t;
  static_assert(D_QK == 512, "packed fp8 sparse decode currently supports logical D_QK=512");
  static_assert(D_QK % SUBGROUP_SIZE == 0, "D_QK must be divisible by SUBGROUP_SIZE");
  static_assert(
      D_QK == SPARSE_MLA_FP8_NOPE_BYTES + SPARSE_MLA_FP8_ROPE_DIM,
      "logical D_QK must match packed fp8 NoPE + RoPE dimensions");
  static_assert(
      SPARSE_MLA_FP8_NOPE_BYTES % FP8_VALUES_PER_PACK == 0, "NoPE fp8 bytes must be divisible by the packed fp8 width");
  static_assert(
      SPARSE_MLA_FP8_ROPE_DIM % BF16_VALUES_PER_PACK == 0,
      "RoPE bf16 values must be divisible by the packed bf16 width");
  static_assert(
      SPARSE_MLA_FP8_NOPE_BYTES / 64 == SPARSE_MLA_FP8_SCALE_BYTES_PER_TOKEN - 1,
      "only the first seven scale bytes are valid for 448 NoPE values");
  static_assert(sizeof(PackedElement) == FP8_VALUES_PER_PACK, "PackedElement must cover one fp8 lane chunk");
  static_assert(
      sizeof(PackedElement) == sizeof(cutlass::bfloat16_t) * BF16_VALUES_PER_PACK,
      "PackedElement must cover one bf16 lane chunk");
  static constexpr int NUM_VALS_PER_THREAD = D_QK / SUBGROUP_SIZE;

  CUTLASS_DEVICE
  static float e8m0_to_float(uint8_t scale_byte) {
    return sycl::native::exp2(static_cast<float>(static_cast<int>(scale_byte) - 127));
  }

  CUTLASS_DEVICE
  static uint16_t fp8_e4m3_scaled_to_bf16_bits(uint8_t fp8_byte, uint8_t scale_byte) {
    const float scale = e8m0_to_float(scale_byte);
    const auto fp8_val = cutlass::float_e4m3_t::bitcast(fp8_byte);
    return cutlass::bfloat16_t(static_cast<float>(fp8_val) * scale).storage;
  }

  CUTLASS_DEVICE
  static void store_dequantized_token_scalar(
      cutlass::bfloat16_t* gathered_row,
      const uint8_t* token_data,
      const uint8_t* token_scales,
      bool valid_token,
      int lane_id) {
    CUTE_UNROLL
    for (int n = 0; n < NUM_VALS_PER_THREAD; ++n) {
      int dim_idx = n * SUBGROUP_SIZE + lane_id;
      cutlass::bfloat16_t kv_val = cutlass::bfloat16_t(0.0f);
      if (valid_token && dim_idx < SPARSE_MLA_FP8_NOPE_BYTES) {
        int scale_idx = dim_idx / 64;
        float scale = e8m0_to_float(token_scales[scale_idx]);
        auto fp8_val = *reinterpret_cast<const cutlass::float_e4m3_t*>(token_data + dim_idx);
        kv_val = cutlass::bfloat16_t(static_cast<float>(fp8_val) * scale);
      } else if (valid_token) {
        const auto* rope_ptr = reinterpret_cast<const cutlass::bfloat16_t*>(token_data + SPARSE_MLA_FP8_NOPE_BYTES);
        kv_val = rope_ptr[dim_idx - SPARSE_MLA_FP8_NOPE_BYTES];
      }
      gathered_row[dim_idx] = kv_val;
    }
  }

  CUTLASS_DEVICE
  static void store_dequantized_token_packed(
      cutlass::bfloat16_t* gathered_row,
      const uint8_t* token_data,
      const uint8_t* token_scales,
      bool valid_token,
      int lane_id) {
    for (int d_base = lane_id * FP8_VALUES_PER_PACK; d_base < SPARSE_MLA_FP8_NOPE_BYTES;
         d_base += SUBGROUP_SIZE * FP8_VALUES_PER_PACK) {
      PackedElement packed_lo = 0;
      PackedElement packed_hi = 0;
      if (valid_token) {
        const PackedElement packed_fp8 = *reinterpret_cast<const PackedElement*>(token_data + d_base);
        const uint8_t scale_byte = token_scales[d_base / 64];
        CUTE_UNROLL
        for (int vec_offset = 0; vec_offset < FP8_VALUES_PER_PACK; ++vec_offset) {
          const uint8_t fp8_byte = static_cast<uint8_t>(packed_fp8 >> (8 * vec_offset));
          const uint16_t bf16_bits = fp8_e4m3_scaled_to_bf16_bits(fp8_byte, scale_byte);
          if (vec_offset < BF16_VALUES_PER_PACK) {
            packed_lo |= PackedElement(bf16_bits) << (16 * vec_offset);
          } else {
            packed_hi |= PackedElement(bf16_bits) << (16 * (vec_offset - BF16_VALUES_PER_PACK));
          }
        }
      }
      *reinterpret_cast<PackedElement*>(gathered_row + d_base) = packed_lo;
      *reinterpret_cast<PackedElement*>(gathered_row + d_base + BF16_VALUES_PER_PACK) = packed_hi;
    }

    for (int rope_base = lane_id * BF16_VALUES_PER_PACK; rope_base < SPARSE_MLA_FP8_ROPE_DIM;
         rope_base += SUBGROUP_SIZE * BF16_VALUES_PER_PACK) {
      const int dim_idx = SPARSE_MLA_FP8_NOPE_BYTES + rope_base;
      const PackedElement value =
          valid_token ? *reinterpret_cast<const PackedElement*>(
                            token_data + SPARSE_MLA_FP8_NOPE_BYTES + rope_base * sizeof(cutlass::bfloat16_t))
                      : PackedElement(0);
      *reinterpret_cast<PackedElement*>(gathered_row + dim_idx) = value;
    }
  }

  CUTLASS_DEVICE
  static void store_dequantized_token(
      cutlass::bfloat16_t* gathered_row,
      const uint8_t* active_kv,
      int token_idx,
      int active_page_block_size,
      int active_stride_kv_block,
      bool valid_token,
      bool can_pack,
      int lane_id) {
    const uint8_t* token_data = nullptr;
    const uint8_t* token_scales = nullptr;
    if (valid_token) {
      int block_idx = token_idx / active_page_block_size;
      int rel_idx = token_idx - block_idx * active_page_block_size;
      token_data = active_kv + block_idx * active_stride_kv_block + rel_idx * SPARSE_MLA_FP8_DATA_BYTES_PER_TOKEN;
      token_scales = active_kv + block_idx * active_stride_kv_block +
                     active_page_block_size * SPARSE_MLA_FP8_DATA_BYTES_PER_TOKEN +
                     rel_idx * SPARSE_MLA_FP8_SCALE_BYTES_PER_TOKEN;
    }

    if (can_pack) {
      store_dequantized_token_packed(gathered_row, token_data, token_scales, valid_token, lane_id);
    } else {
      store_dequantized_token_scalar(gathered_row, token_data, token_scales, valid_token, lane_id);
    }
  }

  CUTLASS_DEVICE
  void operator()(const Params& params, char* smem_buf) const {
    const int thr_id = int(ThreadIdxX());
    const int sg_id = thr_id / SUBGROUP_SIZE;
    const int lane_id = thr_id % SUBGROUP_SIZE;
    const int seq_linear_idx = int(BlockIdxX());
    const int batch_idx = seq_linear_idx / params.shape.s_q;
    const int seq_idx = seq_linear_idx - batch_idx * params.shape.s_q;
    const int topk_block_idx = int(BlockIdxY());
    const int topk_base = topk_block_idx * B_TOPK;

    auto* gathered_k =
        params.gathered_k + batch_idx * params.stride_gathered_k_b + seq_idx * params.stride_gathered_k_s_q;
    auto* gathered_valid_mask = params.gathered_valid_mask + batch_idx * params.stride_gathered_mask_b +
                                seq_idx * params.stride_gathered_mask_s_q;
    const int* main_indices =
        params.indices + batch_idx * params.stride_indices_b + seq_idx * params.stride_indices_s_q;
    const int* extra_indices = params.extra_indices == nullptr
                                   ? nullptr
                                   : params.extra_indices + batch_idx * params.stride_extra_indices_b +
                                         seq_idx * params.stride_extra_indices_s_q;

    auto resolve_topk_length = [&](const int* topk_length_ptr, int topk, int stride_b) {
      if (topk_length_ptr != nullptr) {
        return *(topk_length_ptr + batch_idx * stride_b);
      }
      return topk;
    };

    const int main_topk_length =
        resolve_topk_length(params.topk_length, params.shape.topk, params.stride_topk_length_b);
    const int extra_topk_length =
        resolve_topk_length(params.extra_topk_length, params.shape.extra_topk, params.stride_extra_topk_length_b);

    for (int local_topk_idx = sg_id; local_topk_idx < B_TOPK; local_topk_idx += NUM_SUBGROUPS) {
      int topk_idx = topk_base + local_topk_idx;
      if (topk_idx >= params.shape.gathered_topk) {
        continue;
      }

      bool is_extra = topk_idx >= params.shape.topk;
      int range_topk_idx = is_extra ? topk_idx - params.shape.topk : topk_idx;
      int active_topk = is_extra ? params.shape.extra_topk : params.shape.topk;
      int active_topk_length = is_extra ? extra_topk_length : main_topk_length;
      const int* active_indices = is_extra ? extra_indices : main_indices;
      const uint8_t* active_kv = is_extra ? params.extra_kv : params.kv;
      int active_num_blocks = is_extra ? params.shape.extra_num_blocks : params.shape.num_blocks;
      int active_page_block_size = is_extra ? params.shape.extra_page_block_size : params.shape.page_block_size;
      int active_stride_kv_block = is_extra ? params.stride_extra_kv_block : params.stride_kv_block;
      const bool can_pack = params.stride_gathered_k_topk % BF16_VALUES_PER_PACK == 0 &&
                            active_stride_kv_block % sizeof(PackedElement) == 0 &&
                            (reinterpret_cast<uintptr_t>(active_kv) & (sizeof(PackedElement) - 1)) == 0 &&
                            (reinterpret_cast<uintptr_t>(gathered_k) & (sizeof(PackedElement) - 1)) == 0;

      bool valid_token = false;
      int token_idx = -1;
      if (active_indices != nullptr && active_kv != nullptr && range_topk_idx < active_topk &&
          range_topk_idx < active_topk_length) {
        token_idx = active_indices[range_topk_idx];
        valid_token = token_idx >= 0 && token_idx < active_num_blocks * active_page_block_size;
      }

      cutlass::bfloat16_t* gathered_row = gathered_k + topk_idx * params.stride_gathered_k_topk;
      store_dequantized_token(
          gathered_row,
          active_kv,
          token_idx,
          active_page_block_size,
          active_stride_kv_block,
          valid_token,
          can_pack,
          lane_id);

      if (lane_id == 0) {
        gathered_valid_mask[topk_idx] = static_cast<int>(valid_token);
      }
    }
  }
};

}  // namespace cutlass::flash_attention::kernel
