/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
/*!
  \file
  \brief Two-stage sparse MLA Stage 1 gather kernel for DeepSeek V4 (decode + prefill).

  SparseGatherKernel<D_QK, SourcePolicy>: subgroup-coalesced gather of the indexed
  KV rows into a dense [b, s_q, gathered_topk, D_QK] bf16 tile + an int valid mask.
  The work-group grid, per-(batch, seq) base pointers, the per-subgroup topk-column
  loop, and the valid-mask write are shared; the *source* of each token -- how a KV
  row is located and materialized into bf16 -- is a policy so decode and prefill
  reuse one skeleton:

    - DecodeFp8PagedSource  : reads a *packed fp8 paged* KV cache and dequantizes
        (per-64 e8m0 scales; nope fp8 + rope bf16 -> 512-dim bf16), concatenating a
        primary + extra pool. Reference: tests/test_flash_mla_with_kvcache.py
        _gather_and_dequant.
    - PrefillDenseBf16Source: reads a dense *bf16 unpaged* KV source and does a plain
        D_QK-wide copy (D_QK is 512 or 576): no fp8 decode, no scale section, no extra
        pool, no paging. Reference: tests/test_flash_mla_sparse_fwd.py
        reference_mla_sparse_prefill (Stage 1).

  Shared declarations (the Gather2StageParams blocks, constants) come from
  xe_mla_sparse_decode_2stage_common.hpp. Stage 1 owns its own tile constants and
  does not use the Stage-2 config struct (MlaSparseDecode2StageXe).

  Stage 1 is a fully standalone kernel: its Params ARE the gather params (the source
  policy's decode / prefill child), with no reference to the Stage-2 dense params. It is
  the Stage-2 runner's companion kernel, wired in exactly like the dense MLA path's
  split-KV reduction companion (kernel/xe_mla_reduce_split_kv.hpp): it exposes the same
  host-side contract (Arguments/Params, to_underlying_arguments, can_implement,
  get_workspace_size, get_grid_shape, get_block_shape) and device::MLASparse launches it
  before the dense kernel on the in-order XPU queue. The two stages communicate only
  through the gathered_k / gathered_valid_mask HBM buffers, which each stage's params
  name independently.

  The two aliases below give each path a `template <int> class` entry point keyed on
  D_QK, which is how the Stage-2 config struct (MlaSparseDecode2StageXe's
  GatherKernelTmpl parameter) selects the decode vs prefill gather:
    - SparseDecodeGatherDequantKernel<D_QK> == SparseGatherKernel<D_QK, DecodeFp8PagedSource>
    - SparsePrefillGatherKernel<D_QK>       == SparseGatherKernel<D_QK, PrefillDenseBf16Source>
*/

#pragma once

#include "sycl/kernels/mla_sparse/device/xe_mla_sparse_decode_2stage_common.hpp"

namespace cutlass::flash_attention::kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////
// Source policy: decode (packed fp8 paged cache -> bf16 dequant, dual pool).
/////////////////////////////////////////////////////////////////////////////////////////////////
template <int D_QK>
struct DecodeFp8PagedSource {
  // The Stage-1 params this source reads. Stage 1 is a standalone kernel: these are
  // its complete Params, independent of the Stage-2 dense params (the two stages
  // share only the gathered-KV HBM buffers named in both).
  using GatherParams = DecodeGather2StageParams;

  static constexpr int SUBGROUP_SIZE = intel::sg_size;
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

  // Per-(batch, seq) invariants hoisted out of the topk-column loop: the two pools'
  // index pointers and (optional) topk-length caps.
  struct RowContext {
    const int* main_indices;
    const int* extra_indices;
    int main_topk_length;
    int extra_topk_length;
  };

  CUTLASS_DEVICE
  static RowContext begin(const GatherParams& params, int batch_idx, int seq_idx) {
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

    RowContext ctx;
    ctx.main_indices = main_indices;
    ctx.extra_indices = extra_indices;
    ctx.main_topk_length = resolve_topk_length(params.topk_length, params.topk, params.stride_topk_length_b);
    ctx.extra_topk_length =
        resolve_topk_length(params.extra_topk_length, params.extra_topk, params.stride_extra_topk_length_b);
    return ctx;
  }

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

  // Resolve the token for this topk column (from the primary or extra pool),
  // dequantize it into gathered_row, and return whether it was a valid token.
  CUTLASS_DEVICE
  static bool emit(
      const GatherParams& params,
      const RowContext& ctx,
      cutlass::bfloat16_t* gathered_k,
      cutlass::bfloat16_t* gathered_row,
      int topk_idx,
      int lane_id) {
    bool is_extra = topk_idx >= params.topk;
    int range_topk_idx = is_extra ? topk_idx - params.topk : topk_idx;
    int active_topk = is_extra ? params.extra_topk : params.topk;
    int active_topk_length = is_extra ? ctx.extra_topk_length : ctx.main_topk_length;
    const int* active_indices = is_extra ? ctx.extra_indices : ctx.main_indices;
    const uint8_t* active_kv = is_extra ? params.extra_kv : params.kv;
    int active_num_blocks = is_extra ? params.extra_num_blocks : params.num_blocks;
    int active_page_block_size = is_extra ? params.extra_page_block_size : params.page_block_size;
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

    store_dequantized_token(
        gathered_row,
        active_kv,
        token_idx,
        active_page_block_size,
        active_stride_kv_block,
        valid_token,
        can_pack,
        lane_id);
    return valid_token;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
// Source policy: prefill (dense bf16 unpaged source -> plain D_QK-wide copy).
/////////////////////////////////////////////////////////////////////////////////////////////////
template <int D_QK>
struct PrefillDenseBf16Source {
  // The Stage-1 params this source reads (prefill variant). See the decode source:
  // these are the standalone gather kernel's complete Params.
  using GatherParams = PrefillGather2StageParams;

  static constexpr int SUBGROUP_SIZE = intel::sg_size;
  // Coalesced packed copy: 4 bf16 (== 8 bytes) per lane per step.
  static constexpr int BF16_VALUES_PER_PACK = 4;
  using PackedElement = uint64_t;
  // Dense bf16 KV: D_QK is 512 (latent) or 576 (nope-512 + rope-64). Both tile
  // evenly over one subgroup's packed span (16 lanes x 4 bf16 = 64): 512/64=8,
  // 576/64=9. The whole D_QK row is copied verbatim (V uses its first-512 sub-view
  // downstream), so no nope/rope split is needed here.
  static_assert(D_QK == 512 || D_QK == 576, "sparse prefill supports dense bf16 D_QK in {512, 576}");
  static_assert(D_QK % (SUBGROUP_SIZE * BF16_VALUES_PER_PACK) == 0, "D_QK must tile evenly over packed lanes");
  static_assert(
      sizeof(PackedElement) == sizeof(cutlass::bfloat16_t) * BF16_VALUES_PER_PACK,
      "PackedElement must cover one bf16 lane chunk");

  // Per-(batch, seq) invariants hoisted out of the topk-column loop: the index
  // pointer and (optional) topk-length cap for the single dense pool.
  struct RowContext {
    const int* indices;
    int topk_length;
  };

  CUTLASS_DEVICE
  static RowContext begin(const GatherParams& params, int batch_idx, int seq_idx) {
    RowContext ctx;
    ctx.indices = params.indices + batch_idx * params.stride_indices_b + seq_idx * params.stride_indices_s_q;
    ctx.topk_length =
        params.topk_length != nullptr ? *(params.topk_length + batch_idx * params.stride_topk_length_b) : params.topk;
    return ctx;
  }

  // Copy one dense bf16 KV row (d_qk wide) into the gathered tile, packing 4 bf16 per
  // lane. Zero-fills invalid tokens so 0 * NaN can never pollute Stage 2.
  CUTLASS_DEVICE
  static void copy_token_packed(
      cutlass::bfloat16_t* gathered_row, const cutlass::bfloat16_t* token_row, bool valid_token, int lane_id) {
    for (int d_base = lane_id * BF16_VALUES_PER_PACK; d_base < D_QK; d_base += SUBGROUP_SIZE * BF16_VALUES_PER_PACK) {
      const PackedElement value =
          valid_token ? *reinterpret_cast<const PackedElement*>(token_row + d_base) : PackedElement(0);
      *reinterpret_cast<PackedElement*>(gathered_row + d_base) = value;
    }
  }

  // Scalar fallback when the packed path's alignment preconditions do not hold.
  CUTLASS_DEVICE
  static void copy_token_scalar(
      cutlass::bfloat16_t* gathered_row, const cutlass::bfloat16_t* token_row, bool valid_token, int lane_id) {
    for (int d = lane_id; d < D_QK; d += SUBGROUP_SIZE) {
      gathered_row[d] = valid_token ? token_row[d] : cutlass::bfloat16_t(0.0f);
    }
  }

  // Resolve the token for this topk column, copy its dense bf16 row into
  // gathered_row, and return whether it was a valid token.
  CUTLASS_DEVICE
  static bool emit(
      const GatherParams& params,
      const RowContext& ctx,
      cutlass::bfloat16_t* gathered_k,
      cutlass::bfloat16_t* gathered_row,
      int topk_idx,
      int lane_id) {
    // Packed copy needs both source and destination rows 8-byte aligned and the
    // gathered stride packable; dense bf16 rows normally satisfy this.
    const bool can_pack = params.stride_gathered_k_topk % BF16_VALUES_PER_PACK == 0 &&
                          params.stride_kv_dense_s % BF16_VALUES_PER_PACK == 0 &&
                          (reinterpret_cast<uintptr_t>(params.kv_dense) & (sizeof(PackedElement) - 1)) == 0 &&
                          (reinterpret_cast<uintptr_t>(gathered_k) & (sizeof(PackedElement) - 1)) == 0;

    bool valid_token = false;
    int token_idx = -1;
    if (params.indices != nullptr && params.kv_dense != nullptr && topk_idx < params.topk &&
        topk_idx < ctx.topk_length) {
      token_idx = ctx.indices[topk_idx];
      valid_token = token_idx >= 0 && token_idx < params.s_kv;
    }

    const cutlass::bfloat16_t* token_row =
        valid_token ? params.kv_dense + static_cast<long>(token_idx) * params.stride_kv_dense_s : nullptr;

    if (can_pack) {
      copy_token_packed(gathered_row, token_row, valid_token, lane_id);
    } else {
      copy_token_scalar(gathered_row, token_row, valid_token, lane_id);
    }
    return valid_token;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
// Shared Stage-1 gather skeleton. The SourcePolicy supplies the per-token resolve +
// materialize (emit); everything else -- grid, base pointers, the coalesced
// topk-column loop, and the valid-mask write -- is identical for decode and prefill.
/////////////////////////////////////////////////////////////////////////////////////////////////
template <int D_QK, template <int> class SourcePolicyTmpl>
class SparseGatherKernel {
 public:
  using Source = SourcePolicyTmpl<D_QK>;

  // Stage 1 is a standalone kernel with its own Params: the source policy's gather
  // params (decode or prefill child). It has no dependency on the Stage-2 dense
  // kernel's params -- the two stages meet only at the gathered-KV HBM buffers.
  using GatherParams = typename Source::GatherParams;
  using Arguments = GatherParams;
  using KernelArguments = GatherParams;
  using Params = GatherParams;

  static constexpr int NUM_THREADS = 128;
  static constexpr int SUBGROUP_SIZE = intel::sg_size;
  static constexpr int NUM_SUBGROUPS = NUM_THREADS / SUBGROUP_SIZE;
  static constexpr int B_TOPK = 64;

  // Gather uses no SLM.
  static constexpr int SharedStorageSize = 0;

  // Host-side contract for the device::MLASparse runner, mirroring what the split-KV
  // reduction companion provides to device::MLA (kernel/xe_mla_reduce_split_kv.hpp):
  // Arguments == Params (nothing to transform), no workspace of its own.
  static Params to_underlying_arguments(Arguments const& args, void* /* workspace */) {
    return args;
  }

  static bool can_implement(Arguments const& args) {
    return args.b > 0 && args.s_q > 0 && args.gathered_topk > 0 && args.gathered_k != nullptr &&
           args.gathered_valid_mask != nullptr;
  }

  static int get_workspace_size(Arguments const& /* args */) {
    return 0;
  }

  static cutlass::Status initialize_workspace(Arguments const& /* args */, void* /* workspace */ = nullptr) {
    return cutlass::Status::kSuccess;
  }

  // launch<> contract: one work-group per (batch*seq, topk-block); B_TOPK topk
  // columns per work-group.
  static dim3 get_grid_shape(Params const& params) {
    return dim3(params.b * params.s_q, ceil_div(params.gathered_topk, B_TOPK), 1);
  }

  static dim3 get_block_shape() {
    return dim3(NUM_THREADS, 1, 1);
  }

  CUTLASS_DEVICE
  void operator()(const Params& params, char* smem_buf) const {
    const GatherParams& gather = params;
    const int thr_id = int(ThreadIdxX());
    const int sg_id = thr_id / SUBGROUP_SIZE;
    const int lane_id = thr_id % SUBGROUP_SIZE;
    const int seq_linear_idx = int(BlockIdxX());
    const int batch_idx = seq_linear_idx / gather.s_q;
    const int seq_idx = seq_linear_idx - batch_idx * gather.s_q;
    const int topk_block_idx = int(BlockIdxY());
    const int topk_base = topk_block_idx * B_TOPK;

    auto* gathered_k =
        gather.gathered_k + batch_idx * gather.stride_gathered_k_b + seq_idx * gather.stride_gathered_k_s_q;
    auto* gathered_valid_mask = gather.gathered_valid_mask + batch_idx * gather.stride_gathered_mask_b +
                                seq_idx * gather.stride_gathered_mask_s_q;

    auto ctx = Source::begin(gather, batch_idx, seq_idx);

    for (int local_topk_idx = sg_id; local_topk_idx < B_TOPK; local_topk_idx += NUM_SUBGROUPS) {
      int topk_idx = topk_base + local_topk_idx;
      if (topk_idx >= gather.gathered_topk) {
        continue;
      }

      cutlass::bfloat16_t* gathered_row = gathered_k + topk_idx * gather.stride_gathered_k_topk;
      const bool valid_token = Source::emit(gather, ctx, gathered_k, gathered_row, topk_idx, lane_id);

      if (lane_id == 0) {
        gathered_valid_mask[topk_idx] = static_cast<int>(valid_token);
      }
    }
  }
};

// Per-path entry points with the `template <int> class` (D_QK-keyed) interface the
// Stage-2 config struct's GatherKernelTmpl param selects on. The config resolves one of
// these as MlaSparseDecode2StageXe::GatherKernel and passes it to device::MLASparse as
// the companion kernel.
template <int D_QK>
using SparseDecodeGatherDequantKernel = SparseGatherKernel<D_QK, DecodeFp8PagedSource>;

template <int D_QK>
using SparsePrefillGatherKernel = SparseGatherKernel<D_QK, PrefillDenseBf16Source>;

}  // namespace cutlass::flash_attention::kernel
