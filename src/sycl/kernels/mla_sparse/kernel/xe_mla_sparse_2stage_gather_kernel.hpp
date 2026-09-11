/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
/*!
  \file
  \brief Two-stage sparse MLA Stage 1 gather kernel for DeepSeek V4 (decode + prefill).

  SparseGatherKernel<D_QK, SourcePolicy>: subgroup-coalesced gather of the indexed
  KV rows into a dense [b, s_q, gathered_topk, D_QK] bf16 tile + an int valid mask.
  The work-group grid, the gathered-tile / valid-mask tensor views, the per-subgroup
  topk-column loop, and the valid-mask write are shared; the *source* of each token --
  how a KV row is located and materialized into bf16 -- is a policy so decode and
  prefill reuse one skeleton:

    - DecodeFp8PagedSource  : reads a *packed fp8 paged* KV cache and dequantizes
        (per-64 e8m0 scales; nope fp8 + rope bf16 -> 512-dim bf16), concatenating a
        primary + extra pool.
    - PrefillDenseBf16Source: reads a dense *bf16 unpaged* KV source and does a plain
        D_QK-wide copy (D_QK is 512 or 576): no fp8 decode, no scale section, no extra
        pool, no paging.
*/

#pragma once

#include "sycl/kernels/mla_sparse/device/xe_mla_sparse_2stage_common.hpp"
#include "sycl/kernels/mla_sparse/kernel/xe_mla_sparse_2stage_tile_scheduler.hpp"

namespace cutlass::flash_attention::kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////
// Both source policies vectorize their row copy by recasting the bf16 row tensor to a
// wider element; that is only legal if the recast base addresses are aligned to it.
// Because the tensors handed to the policies are already offset to *this* token's row,
// checking the two row pointers subsumes every base / stride / index contribution --
// no separate per-stride divisibility test is needed.
/////////////////////////////////////////////////////////////////////////////////////////////////
template <class PackedElement>
CUTLASS_DEVICE bool is_aligned_for(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & (sizeof(PackedElement) - 1)) == 0;
}

template <class PackedElement>
CUTLASS_DEVICE bool is_packed_aligned(const void* src, const void* dst) {
  constexpr uintptr_t kMask = sizeof(PackedElement) - 1;
  return ((reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst)) & kMask) == 0;
}

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
  using PackedOut = intel::ushort8;

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
  static_assert(
      sizeof(PackedOut) == sizeof(cutlass::bfloat16_t) * FP8_VALUES_PER_PACK,
      "PackedOut must hold every bf16 one fp8 pack dequantizes to");
  static_assert(
      SPARSE_MLA_FP8_SCALE_BYTES_PER_TOKEN == sizeof(PackedElement),
      "the e8m0 scale section must be exactly one packed load");
  static_assert(
      SPARSE_MLA_FP8_DATA_BYTES_PER_TOKEN % sizeof(PackedElement) == 0 &&
          SPARSE_MLA_FP8_NOPE_BYTES % sizeof(PackedElement) == 0,
      "scales inherit nope's alignment only if both section offsets are packed multiples");
  static constexpr int NUM_VALS_PER_THREAD = D_QK / SUBGROUP_SIZE;
  // Chunk counts once each section is viewed as PackedElements, and where the RoPE
  // section starts in the destination row's PackedElement view.
  static constexpr int NOPE_PACKS = SPARSE_MLA_FP8_NOPE_BYTES / FP8_VALUES_PER_PACK;
  static constexpr int ROPE_PACKS = SPARSE_MLA_FP8_ROPE_DIM / BF16_VALUES_PER_PACK;
  static constexpr int ROPE_PACK_BASE = SPARSE_MLA_FP8_NOPE_BYTES / BF16_VALUES_PER_PACK;
  static constexpr int NOPE_ITERS = cute::ceil_div(NOPE_PACKS, SUBGROUP_SIZE);
  static constexpr int ROPE_ITERS = cute::ceil_div(ROPE_PACKS, SUBGROUP_SIZE);
  static_assert(
      NOPE_PACKS * sizeof(PackedOut) == SPARSE_MLA_FP8_NOPE_BYTES * sizeof(cutlass::bfloat16_t),
      "the NoPE PackedOut chunks must tile the destination row's NoPE region exactly");

  // Per-(batch, seq) invariants hoisted out of the topk-column loop: the two pools'
  // index pointers and (optional) topk-length caps. The index arrays are 1-D and
  // contiguous, so they stay raw pointers -- a tensor view would buy nothing.
  struct RowContext {
    const int* main_indices;
    const int* extra_indices;
    int main_topk_length;
    int extra_topk_length;
  };

  // One token's packed record, resolved to typed tensor views over its three
  // sections. Null (rather than offset-from-null) for an invalid token, so the
  // pointer arithmetic below never runs off a null base.
  struct TokenRecord {
    const uint8_t* nope;              // [SPARSE_MLA_FP8_NOPE_BYTES] fp8 e4m3 bytes
    const cutlass::bfloat16_t* rope;  // [SPARSE_MLA_FP8_ROPE_DIM] bf16
    const uint8_t* scales;            // [SPARSE_MLA_FP8_SCALE_BYTES_PER_TOKEN] e8m0
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
  static auto make_nope_view(const uint8_t* nope) {
    return make_tensor(make_gmem_ptr(nope), make_layout(Shape<Int<SPARSE_MLA_FP8_NOPE_BYTES>>{}, Stride<_1>{}));
  }

  CUTLASS_DEVICE
  static auto make_rope_view(const cutlass::bfloat16_t* rope) {
    return make_tensor(make_gmem_ptr(rope), make_layout(Shape<Int<SPARSE_MLA_FP8_ROPE_DIM>>{}, Stride<_1>{}));
  }

  CUTLASS_DEVICE
  static float e8m0_to_float(uint8_t scale_byte) {
    return sycl::native::exp2(static_cast<float>(static_cast<int>(scale_byte) - 127));
  }

  CUTLASS_DEVICE
  static uint16_t fp8_e4m3_scaled_to_bf16_bits(uint8_t fp8_byte, float scale) {
    const auto fp8_val = cutlass::float_e4m3_t::bitcast(fp8_byte);
    return cutlass::bfloat16_t(static_cast<float>(fp8_val) * scale).storage;
  }

  // Scalar fallback: one bf16 per lane per step, NoPE dequantized and RoPE copied.
  template <class TensorGRow>
  CUTLASS_DEVICE static void
  store_dequantized_token_scalar(TensorGRow&& gRow, TokenRecord const& token, bool valid_token, int lane_id) {
    auto sNope = make_nope_view(token.nope);
    auto sRope = make_rope_view(token.rope);

    CUTE_UNROLL
    for (int n = 0; n < NUM_VALS_PER_THREAD; ++n) {
      const int dim_idx = n * SUBGROUP_SIZE + lane_id;
      cutlass::bfloat16_t kv_val = cutlass::bfloat16_t(0.0f);
      if (valid_token && dim_idx < SPARSE_MLA_FP8_NOPE_BYTES) {
        const float scale = e8m0_to_float(token.scales[dim_idx / 64]);
        const auto fp8_val = cutlass::float_e4m3_t::bitcast(sNope(dim_idx));
        kv_val = cutlass::bfloat16_t(static_cast<float>(fp8_val) * scale);
      } else if (valid_token) {
        kv_val = sRope(dim_idx - SPARSE_MLA_FP8_NOPE_BYTES);
      }
      gRow(dim_idx) = kv_val;
    }
  }

  // NoPE conversion and store, from already-loaded registers.
  template <class TensorGOut>
  CUTLASS_DEVICE static void dequantize_and_store_nope(
      TensorGOut&& gOut, PackedElement const (&packed_fp8)[NOPE_ITERS], PackedElement scale_word, int lane_id) {
    CUTE_UNROLL
    for (int u = 0; u < NOPE_ITERS; ++u) {
      const int i = lane_id + u * SUBGROUP_SIZE;
      if (i >= NOPE_PACKS) {
        continue;
      }
      const float scale = e8m0_to_float(uint8_t(scale_word >> (8 * ((i * FP8_VALUES_PER_PACK) / 64))));
      PackedOut out;
      CUTE_UNROLL
      for (int vec_offset = 0; vec_offset < FP8_VALUES_PER_PACK; ++vec_offset) {
        const uint8_t fp8_byte = static_cast<uint8_t>(packed_fp8[u] >> (8 * vec_offset));
        out[vec_offset] = fp8_e4m3_scaled_to_bf16_bits(fp8_byte, scale);
      }
      gOut(i) = out;
    }
  }

  // Vectorized path: each lane converts 8 fp8 into one 16-byte bf16 store (NoPE) or
  // moves 4 bf16 (RoPE). An invalid token arrives with zeroed inputs and so writes zeros.
  template <class TensorGRow>
  CUTLASS_DEVICE static void
  store_dequantized_token_packed(TensorGRow&& gRow, TokenRecord const& token, bool valid_token, int lane_id) {
    auto gPacked = recast<PackedElement>(gRow);                      // (D_QK / BF16_VALUES_PER_PACK)
    auto gOut = recast<PackedOut>(gRow);                             // (D_QK / FP8_VALUES_PER_PACK)
    auto sNope = recast<PackedElement>(make_nope_view(token.nope));  // (NOPE_BYTES / FP8_VALUES_PER_PACK)
    auto sRope = recast<PackedElement>(make_rope_view(token.rope));  // (ROPE_DIM / BF16_VALUES_PER_PACK)

    const PackedElement scale_word =
        valid_token ? *reinterpret_cast<const PackedElement*>(token.scales) : PackedElement(0);

    PackedElement packed_nope[NOPE_ITERS];
    CUTE_UNROLL
    for (int u = 0; u < NOPE_ITERS; ++u) {
      const int i = lane_id + u * SUBGROUP_SIZE;
      packed_nope[u] = (valid_token && i < NOPE_PACKS) ? sNope(i) : PackedElement(0);
    }
    PackedElement packed_rope[ROPE_ITERS];
    CUTE_UNROLL
    for (int u = 0; u < ROPE_ITERS; ++u) {
      const int j = lane_id + u * SUBGROUP_SIZE;
      packed_rope[u] = (valid_token && j < ROPE_PACKS) ? sRope(j) : PackedElement(0);
    }

    // Dequantize and store the NoPE portion of the token.
    dequantize_and_store_nope(gOut, packed_nope, scale_word, lane_id);

    // RoPE: already bf16 -- a straight packed copy into the tail of the row, from the
    // registers staged above.
    CUTE_UNROLL
    for (int u = 0; u < ROPE_ITERS; ++u) {
      const int j = lane_id + u * SUBGROUP_SIZE;
      if (j < ROPE_PACKS) {
        gPacked(ROPE_PACK_BASE + j) = packed_rope[u];
      }
    }
  }

  CUTLASS_DEVICE
  static TokenRecord locate_token(
      const uint8_t* active_kv,
      int token_idx,
      cutlass::FastDivmod const& active_page_block_divmod,
      int active_stride_kv_block) {
    const int active_page_block_size = active_page_block_divmod.divisor;
    int block_idx, rel_idx;
    active_page_block_divmod(block_idx, rel_idx, token_idx);
    const uint8_t* block = active_kv + block_idx * active_stride_kv_block;
    const uint8_t* record = block + rel_idx * SPARSE_MLA_FP8_DATA_BYTES_PER_TOKEN;

    TokenRecord token;
    token.nope = record;
    token.rope = reinterpret_cast<const cutlass::bfloat16_t*>(record + SPARSE_MLA_FP8_NOPE_BYTES);
    token.scales = block + active_page_block_size * SPARSE_MLA_FP8_DATA_BYTES_PER_TOKEN +
                   rel_idx * SPARSE_MLA_FP8_SCALE_BYTES_PER_TOKEN;
    return token;
  }

  // Resolve the token for this topk column (from the primary or extra pool),
  // dequantize it into gRow, and return whether it was a valid token.
  template <class TensorGRow>
  CUTLASS_DEVICE static bool
  emit(const GatherParams& params, const RowContext& ctx, TensorGRow&& gRow, int topk_idx, int lane_id) {
    const bool is_extra = topk_idx >= params.topk;
    const int range_topk_idx = is_extra ? topk_idx - params.topk : topk_idx;
    const int active_topk = is_extra ? params.extra_topk : params.topk;
    const int active_topk_length = is_extra ? ctx.extra_topk_length : ctx.main_topk_length;
    const int* active_indices = is_extra ? ctx.extra_indices : ctx.main_indices;
    const uint8_t* active_kv = is_extra ? params.extra_kv : params.kv;
    const int active_num_blocks = is_extra ? params.extra_num_blocks : params.num_blocks;
    const int active_page_block_size = is_extra ? params.extra_page_block_size : params.page_block_size;
    const int active_stride_kv_block = is_extra ? params.stride_extra_kv_block : params.stride_kv_block;
    cutlass::FastDivmod const& active_page_block_divmod =
        is_extra ? params.extra_page_block_divmod : params.page_block_divmod;

    bool valid_token = false;
    int token_idx = -1;
    if (active_indices != nullptr && active_kv != nullptr && range_topk_idx < active_topk &&
        range_topk_idx < active_topk_length) {
      token_idx = active_indices[range_topk_idx];
      valid_token = token_idx >= 0 && token_idx < active_num_blocks * active_page_block_size;
    }

    TokenRecord token{nullptr, nullptr, nullptr};
    if (valid_token) {
      token = locate_token(active_kv, token_idx, active_page_block_divmod, active_stride_kv_block);
    }

    if (is_packed_aligned<PackedElement>(token.nope, raw_pointer_cast(gRow.data()))) {
      store_dequantized_token_packed(gRow, token, valid_token, lane_id);
    } else {
      store_dequantized_token_scalar(gRow, token, valid_token, lane_id);
    }
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
  static constexpr int BF16_VALUES_PER_PACK = 8;
  using PackedElement = intel::ushort8;
  static_assert(D_QK == 512 || D_QK == 576, "sparse prefill supports dense bf16 D_QK in {512, 576}");
  static_assert(D_QK % BF16_VALUES_PER_PACK == 0, "D_QK must tile evenly over one lane's packed chunk");
  static_assert(
      sizeof(PackedElement) == sizeof(cutlass::bfloat16_t) * BF16_VALUES_PER_PACK,
      "PackedElement must cover one bf16 lane chunk");

  static constexpr int ROW_PACKS = D_QK / BF16_VALUES_PER_PACK;
  static constexpr int ROW_ITERS = cute::ceil_div(ROW_PACKS, SUBGROUP_SIZE);

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

  // Source row view: static shape, unit stride, so it recasts cleanly to PackedElement.
  CUTLASS_DEVICE
  static auto make_row_view(const cutlass::bfloat16_t* row) {
    return make_tensor(make_gmem_ptr(row), make_layout(Shape<Int<D_QK>>{}, Stride<_1>{}));
  }

  template <class TensorGRow, class TensorSRow>
  CUTLASS_DEVICE static void copy_token_packed(TensorGRow&& gRow, TensorSRow&& sRow, bool valid_token, int lane_id) {
    auto gPacked = recast<PackedElement>(gRow);
    auto sPacked = recast<PackedElement>(sRow);
    PackedElement staged[ROW_ITERS];
    CUTE_UNROLL
    for (int u = 0; u < ROW_ITERS; ++u) {
      const int i = lane_id + u * SUBGROUP_SIZE;
      staged[u] = (valid_token && i < ROW_PACKS) ? sPacked(i) : PackedElement(0);
    }
    CUTE_UNROLL
    for (int u = 0; u < ROW_ITERS; ++u) {
      const int i = lane_id + u * SUBGROUP_SIZE;
      if (i < ROW_PACKS) {
        gPacked(i) = staged[u];
      }
    }
  }

  // Scalar fallback when the packed path's alignment preconditions do not hold.
  template <class TensorGRow, class TensorSRow>
  CUTLASS_DEVICE static void copy_token_scalar(TensorGRow&& gRow, TensorSRow&& sRow, bool valid_token, int lane_id) {
    CUTE_NO_UNROLL
    for (int d = lane_id; d < D_QK; d += SUBGROUP_SIZE) {
      gRow(d) = valid_token ? sRow(d) : cutlass::bfloat16_t(0.0f);
    }
  }

  // Resolve the token for this topk column, copy its dense bf16 row into gRow, and
  // return whether it was a valid token.
  template <class TensorGRow>
  CUTLASS_DEVICE static bool
  emit(const GatherParams& params, const RowContext& ctx, TensorGRow&& gRow, int topk_idx, int lane_id) {
    bool valid_token = false;
    int token_idx = -1;
    if (params.indices != nullptr && params.kv_dense != nullptr && topk_idx < params.topk &&
        topk_idx < ctx.topk_length) {
      token_idx = ctx.indices[topk_idx];
      valid_token = token_idx >= 0 && token_idx < params.s_kv;
    }

    // Null source row for an invalid token: the copies below zero-fill without reading it.
    const cutlass::bfloat16_t* token_row =
        valid_token ? params.kv_dense + static_cast<long>(token_idx) * params.stride_kv_dense_s : nullptr;
    auto sRow = make_row_view(token_row);

    if (is_packed_aligned<PackedElement>(token_row, raw_pointer_cast(gRow.data()))) {
      copy_token_packed(gRow, sRow, valid_token, lane_id);
    } else {
      copy_token_scalar(gRow, sRow, valid_token, lane_id);
    }
    return valid_token;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
// Shared Stage-1 gather skeleton. The SourcePolicy supplies the per-token resolve +
// materialize (emit); everything else -- grid, tensor views of the gathered tile and
// valid mask, the coalesced topk-column loop, and the valid-mask write -- is identical
// for decode and prefill.
/////////////////////////////////////////////////////////////////////////////////////////////////
template <int D_QK, template <int> class SourcePolicyTmpl>
class SparseGatherKernel {
 public:
  using Source = SourcePolicyTmpl<D_QK>;

  using GatherParams = typename Source::GatherParams;
  using Arguments = GatherParams;
  using KernelArguments = GatherParams;
  using Params = GatherParams;

  static constexpr int NUM_THREADS = 128;
  static constexpr int SUBGROUP_SIZE = intel::sg_size;
  static constexpr int NUM_SUBGROUPS = NUM_THREADS / SUBGROUP_SIZE;
  static constexpr int B_TOPK = 64;

  using TileScheduler = XeMlaSparseGather2StageTileScheduler<B_TOPK>;

  struct SharedStorage {};
  static constexpr int SharedStorageSize = 0;

  static Params to_underlying_arguments(Arguments const& args, void* /* workspace */) {
    return args;
  }

  static bool can_implement(Arguments const& args) {
    return args.b > 0 && args.s_q > 0 && args.gathered_topk > 0 && args.gathered_k != nullptr &&
           args.gathered_valid_mask != nullptr;
  }

  // Stage 1 is NOT workspace-free: its outputs are the dense gathered-KV tile
  // ([b, s_q, gathered_topk, D_QK] bf16) plus the int32 valid mask
  // ([b, s_q, gathered_topk]), and those buffers are exactly this kernel's
  // workspace.
  static size_t get_workspace_size(Arguments const& args) {
    const size_t rows = size_t(args.b) * size_t(args.s_q) * size_t(args.gathered_topk);
    return rows * (size_t(D_QK) * sizeof(cutlass::bfloat16_t) + sizeof(int));
  }

  static cutlass::Status initialize_workspace(Arguments const& /* args */, void* /* workspace */ = nullptr) {
    return cutlass::Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::get_grid_shape(params);
  }

  static dim3 get_block_shape() {
    return dim3(NUM_THREADS, 1, 1);
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* /* smem_buf */) const {
    const int thr_id = int(ThreadIdxX());
    const int sg_id = thr_id / SUBGROUP_SIZE;
    const int lane_id = thr_id % SUBGROUP_SIZE;

    // The gathered tile is laid out d-major (mode 0 is the static D_QK with unit stride)
    // so one topk column is a contiguous D_QK span -- that is what the policies'
    // packed copies walk, and what lets them recast the row to a wider element.
    Tensor mK = make_tensor(
        make_gmem_ptr(params.gathered_k),
        make_layout(
            make_shape(Int<D_QK>{}, params.gathered_topk, params.s_q, params.b),
            make_stride(
                _1{}, params.stride_gathered_k_topk, params.stride_gathered_k_s_q, params.stride_gathered_k_b)));
    Tensor mMask = make_tensor(
        make_gmem_ptr(params.gathered_valid_mask),
        make_layout(
            make_shape(params.gathered_topk, params.s_q, params.b),
            make_stride(_1{}, params.stride_gathered_mask_s_q, params.stride_gathered_mask_b)));

    CUTLASS_PRAGMA_NO_UNROLL
    for (TileScheduler tile_scheduler{params}; tile_scheduler.is_valid(); ++tile_scheduler) {
      auto [batch_idx, seq_idx, topk_block_idx] = tile_scheduler.get_block_coord();

      Tensor gK = mK(_, _, seq_idx, batch_idx);
      const int topk_base = topk_block_idx * B_TOPK;

      auto ctx = Source::begin(params, batch_idx, seq_idx);

      for (int local_topk_idx = sg_id; local_topk_idx < B_TOPK; local_topk_idx += NUM_SUBGROUPS) {
        const int topk_idx = topk_base + local_topk_idx;
        if (topk_idx >= params.gathered_topk) {
          continue;
        }

        const bool valid_token = Source::emit(params, ctx, gK(_, topk_idx), topk_idx, lane_id);

        if (lane_id == 0) {
          mMask(topk_idx, seq_idx, batch_idx) = static_cast<int>(valid_token);
        }
      }
    }
  }
};

template <int D_QK>
using SparseDecodeGatherDequantKernel = SparseGatherKernel<D_QK, DecodeFp8PagedSource>;

template <int D_QK>
using SparsePrefillGatherKernel = SparseGatherKernel<D_QK, PrefillDenseBf16Source>;

}  // namespace cutlass::flash_attention::kernel
