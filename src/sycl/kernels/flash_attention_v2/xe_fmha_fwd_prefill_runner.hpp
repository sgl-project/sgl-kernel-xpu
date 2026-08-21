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
#pragma once

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdio>
#include <cstdlib>
#include <cute/tensor.hpp>
#include <random>

#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "sycl/Utils.h"
#include "sycl/comm/common.h"
#include "sycl/kernels/flash_attention_v2/collective/fmha_fusion.hpp"
#include "sycl/kernels/flash_attention_v2/kernel/xe_fmha_fwd_kernel.hpp"
#include "sycl/kernels/flash_attention_v2/kernel/xe_tile_scheduler.hpp"
#include "sycl/kernels/flash_attention_v2/relative_attention.hpp"

using namespace cute;
namespace prefill {
inline constexpr int kRelBiasQTile = flash_attention_v2::relative_attention::kQTile;
inline constexpr int kRelBiasKTile = flash_attention_v2::relative_attention::kKTile;

inline constexpr int rel_bias_padded_cols(int rel_extent) {
  return flash_attention_v2::relative_attention::padded_cols(rel_extent);
}

struct Arguments {
  // The QKV matrices.
  void* __restrict__ q_ptr;
  void* __restrict__ k_ptr;
  void* __restrict__ v_ptr;

  // The stride between rows of the Q, K and V matrices.
  int64_t q_batch_stride;
  int64_t k_batch_stride;
  int64_t v_batch_stride;
  int64_t q_row_stride;
  int64_t k_row_stride;
  int64_t v_row_stride;
  int64_t q_head_stride;
  int64_t k_head_stride;
  int64_t v_head_stride;
  int64_t v_dim_stride;

  // The number of heads.
  int h, h_k;
  int q_group_size = 1;

  // The O matrix (output).
  void* __restrict__ o_ptr;
  void* __restrict__ oaccum_ptr;

  // Sheared relative logits in bf16: [total_q, h, rel_bias_padded_cols(extent)].
  // This is device-produced and consumed directly without host-side staging.
  void* __restrict__ rel_bias_ptr = nullptr;
  int64_t rel_bias_token_stride = 0;
  int64_t rel_bias_head_stride = 0;
  int rel_bias_extent = 0;

  // The stride between rows of O.
  int64_t o_batch_stride;
  int64_t o_row_stride;
  int64_t o_head_stride;

  // The pointer to the softmax sum.
  void* __restrict__ softmax_lse_ptr;
  void* __restrict__ softmax_lseaccum_ptr;

  // The dimensions.
  int b, seqlen_q, seqlen_k, seqlen_knew, d, d_rounded, rotary_dim;
  int total_q, total_k;
  int total_knew = 0;
  int b_k;             // When having KV cache and with cache_batch_idx, K & V might have larger batch size than Q
  int dv, dv_rounded;  // For the case where V headdim is different from Q/K headdim

  // The scaling factors for the kernel.
  float softmax_scale;
  void* softmax_sink_ptr;
  float softcap;

  // FP8 KV cache per-tensor descale. The single scalar lives on-device; the
  // kernel dereferences these pointers so no host-side D2H sync (.item()) is
  // needed. Null => no fp8 dequant (scale = 1.0f).
  const float* k_scale_ptr = nullptr;
  const float* v_scale_ptr = nullptr;

  // array of length b+1 holding starting offset of each sequence.
  int* __restrict__ cu_seqlens_q;
  int* __restrict__ cu_seqlens_k;
  int* __restrict__ cu_seqlens_knew;
  int* __restrict__ leftpad_k;

  // If provided, the actual length of each q/k sequence.
  int* __restrict__ seqused_q;
  int* __restrict__ seqused_k;

  // The stride between rows of Oaccum.
  int64_t oaccum_split_stride;
  int64_t oaccum_batch_stride;
  int64_t oaccum_row_stride;
  int64_t oaccum_head_stride;

  // The stride between rows of LSEaccum.
  int64_t lseaccum_split_stride;
  int64_t lseaccum_batch_stride;
  int64_t lseaccum_head_stride;

  // The K_new and V_new matrices.
  void* __restrict__ knew_ptr;
  void* __restrict__ vnew_ptr;

  // The stride between rows of the Q, K and V matrices.
  int64_t knew_batch_stride;
  int64_t vnew_batch_stride;
  int64_t knew_row_stride;
  int64_t vnew_row_stride;
  int64_t knew_head_stride;
  int64_t vnew_head_stride;

  void* __restrict__ qv_ptr;
  int64_t qv_batch_stride;
  int64_t qv_row_stride;
  int64_t qv_head_stride;

  // The cos and sin matrices for rotary embedding.
  void* __restrict__ rotary_cos_ptr;
  void* __restrict__ rotary_sin_ptr;
  int* __restrict__ seqlens_rotary;

  // The indices to index into the KV cache.
  int* __restrict__ kv_batch_idx;

  // Paged KV cache
  int* __restrict__ page_table;
  int max_num_pages_per_seq;
  int64_t page_table_batch_stride;
  int page_size;
  int num_pages;
  bool pagedkv_tma;

  // The dropout probability (probability of keeping an activation).
  float p_dropout;
  uint8_t p_dropout_in_uint8_t;

  // Scale factor of 1 / (1 - p_dropout).
  float rp_dropout;

  // Local window size
  int window_size_left, window_size_right;

  // Pointer to the RNG seed (idx 0) and offset (idx 1).
  uint64_t* rng_state;

  bool is_bf16;
  bool is_fp16 = false;
  bool is_fp32;
  bool is_e4m3 = false;
  bool is_e5m2 = false;
  bool is_causal;
  bool is_local;
  // When false, the epilogue skips writing softmax_lse. Threaded as a template
  // constexpr via FMHAConfig.
  bool return_softmax_lse = false;

  bool is_rotary_interleaved;

  // Per-batch skip mask for two-kernel mix-batch dispatch
  // (see https://github.com/vllm-project/vllm-xpu-kernels/pull/218).
  // If non-null, the kernel skips batches where mask[idx_b] is true.
  void* skip_batch_mask_ptr = nullptr;

  torch::TensorOptions tensor_opts;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// 3 input matrices: Keys, Queries and Values.
using LayoutQ = cutlass::layout::RowMajor;
using LayoutK = cutlass::layout::ColumnMajor;
using LayoutV = cutlass::layout::RowMajor;
using LayoutO = cutlass::layout::RowMajor;

template <class FMHAPrefillKernel, bool isVarLen = false>
struct PrefillRunner {
  static constexpr int DefaultScoreWorkspaceCapMiB = 1024;

  using StrideQ = typename FMHAPrefillKernel::StrideQ;
  using StrideK = typename FMHAPrefillKernel::StrideK;
  using StrideV = typename FMHAPrefillKernel::StrideV;
  using StrideO = typename FMHAPrefillKernel::StrideO;

  using ElementQ = typename FMHAPrefillKernel::ElementQ;
  using ElementK = typename FMHAPrefillKernel::ElementK;
  using ElementV = typename FMHAPrefillKernel::ElementV;
  using ElementO = typename FMHAPrefillKernel::ElementO;

  using CollectiveMainloop = typename FMHAPrefillKernel::CollectiveMainloop;
  using ElementS = typename CollectiveMainloop::ElementS;

  using ProblemShapeType = cutlass::fmha::kernel::FMHAProblemShape<isVarLen>;

  //
  // Data members
  //

  /// Initialization
  StrideQ stride_Q;
  StrideK stride_K;
  StrideV stride_V;
  StrideK stride_K_cache;
  StrideV stride_V_cache;
  StrideO stride_O;

  //
  // Methods
  //

  template <class ProblemShape>
  auto initialize_varlen(const Arguments& params, const ProblemShape& problem_size) {
    ProblemShape problem_size_for_init = problem_size;
    get<0>(problem_size_for_init) = 1;  // concentrated batch
    get<1>(problem_size_for_init) = params.h;
    get<3>(problem_size_for_init) = params.total_q;
    get<4>(problem_size_for_init) = params.total_knew;
    get<5>(problem_size_for_init) = params.total_k;

    ProblemShapeType problem_size_for_launch{
        .batch = get<0>(problem_size),
        .num_heads_q = get<1>(problem_size),
        .num_heads_kv = get<2>(problem_size),
        .seq_len_qo = {params.seqlen_q, params.total_q, nullptr},
        .seq_len_kv = {params.seqlen_knew, params.total_knew},
        .seq_len_kv_cache = {params.seqlen_k, params.total_k},
        .head_size_qk = get<6>(problem_size),
        .head_size_vo = get<7>(problem_size),
    };

    return cute::make_tuple(problem_size_for_init, problem_size_for_launch);
  }

  /// Initialize operands to be used in the GEMM and reference GEMM
  ProblemShapeType initialize(const Arguments& params) {
    auto problem_shape_in = cute::make_tuple(
        params.b, params.h, params.h_k, params.seqlen_q, params.seqlen_knew, params.seqlen_k, params.d, params.dv);
    ProblemShapeType shape;

    decltype(problem_shape_in) problem_size;

    if constexpr (isVarLen) {
      auto [problem_shape_init, problem_shape_launch] = initialize_varlen(params, problem_shape_in);
      problem_size = problem_shape_init;
      shape = problem_shape_launch;
    } else {
      problem_size = problem_shape_in;
      shape = problem_shape_in;
    }

    auto [batch, num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv, seq_len_kv_cache, head_size_qk, head_size_vo] =
        problem_size;
    // NHD format
    stride_Q = cutlass::make_stride(
        num_heads_q * head_size_qk, Int<1>{}, head_size_qk, head_size_qk * num_heads_q * seq_len_qo);
    stride_K = cutlass::make_stride(
        num_heads_kv * head_size_qk, Int<1>{}, head_size_qk, head_size_qk * num_heads_kv * seq_len_kv);
    stride_V = cutlass::make_stride(
        Int<1>{}, num_heads_kv * head_size_vo, head_size_vo, head_size_vo * num_heads_kv * seq_len_kv);
    stride_K_cache = cutlass::make_stride(
        num_heads_kv * head_size_qk, Int<1>{}, head_size_qk, head_size_qk * num_heads_kv * seq_len_kv_cache);
    stride_V_cache = cutlass::make_stride(
        Int<1>{}, num_heads_kv * head_size_vo, head_size_vo, head_size_vo * num_heads_kv * seq_len_kv_cache);
    stride_O = cutlass::make_stride(
        num_heads_q * head_size_vo, Int<1>{}, head_size_vo, head_size_vo * num_heads_q * seq_len_qo);

    if constexpr (isVarLen) {
      shape.seq_len_qo.cumulative_length = params.cu_seqlens_q;
      shape.seq_len_kv.cumulative_length = params.cu_seqlens_knew;
      shape.seq_len_kv_cache.cumulative_length = params.cu_seqlens_k;
    }

    return shape;
  }

  // Rebase one kernel launch onto [batch_lo, batch_lo + batch_len). Paged
  // prefill uses absolute offsets in its cumulative-length arrays, so the data
  // pointers stay unchanged while the per-batch metadata advances.
  template <class Kernel>
  typename Kernel::Arguments slice_arguments(typename Kernel::Arguments args, int batch_lo, int batch_len) const {
    args.kernel.shape.batch = batch_len;
    if constexpr (isVarLen) {
      if (args.kernel.shape.seq_len_qo.cumulative_length != nullptr) {
        args.kernel.shape.seq_len_qo.cumulative_length += batch_lo;
      }
      if (args.kernel.shape.seq_len_kv.cumulative_length != nullptr) {
        args.kernel.shape.seq_len_kv.cumulative_length += batch_lo;
      }
      if (args.kernel.shape.seq_len_kv_cache.cumulative_length != nullptr) {
        args.kernel.shape.seq_len_kv_cache.cumulative_length += batch_lo;
      }
    }
    if (args.mainloop.ptr_page_table != nullptr) {
      args.mainloop.ptr_page_table += size_t(batch_lo) * size_t(args.mainloop.max_num_pages_per_seq);
    }
    if (args.kernel.skip_batch_mask != nullptr) {
      args.kernel.skip_batch_mask += batch_lo;
    }
    return args;
  }

  // ScoreBlock2D slices contiguous query heads while preserving their mapping
  // to KV heads. The original NHD strides remain in the arguments because the
  // source tensors still contain all heads.
  template <class Kernel>
  typename Kernel::Arguments
  slice_query_head_arguments(typename Kernel::Arguments args, int query_head, int query_head_count) const {
    const int head_group_q = args.kernel.shape.num_heads_q / args.kernel.shape.num_heads_kv;
    const int kv_head = query_head / head_group_q;
    const int head_offset_in_group = query_head % head_group_q;
    const int kv_head_count =
        query_head_count <= head_group_q - head_offset_in_group ? 1 : query_head_count / head_group_q;
    const int head_size_qk = args.kernel.shape.head_size_qk;
    const int head_size_vo = args.kernel.shape.head_size_vo;

    args.kernel.shape.num_heads_q = query_head_count;
    args.kernel.shape.num_heads_kv = kv_head_count;
    args.kernel.Q += size_t(query_head) * size_t(head_size_qk);
    args.kernel.O += size_t(query_head) * size_t(head_size_vo);
    args.kernel.K_cache += size_t(kv_head) * size_t(head_size_qk);
    args.kernel.V_cache += size_t(kv_head) * size_t(head_size_vo);
    if (args.kernel.softmax_lse != nullptr) {
      args.kernel.softmax_lse += size_t(query_head) * size_t(args.kernel.lse_head_stride);
    }
    return args;
  }

  static int get_query_head_slice_size(int query_head, int query_heads, int head_group_q, int requested_heads) {
    const int heads_remaining = query_heads - query_head;
    const int head_offset_in_group = query_head % head_group_q;
    if (head_offset_in_group != 0 || requested_heads < head_group_q) {
      return cute::min(requested_heads, cute::min(heads_remaining, head_group_q - head_offset_in_group));
    }
    // A KV-head-group-aligned slice may include multiple complete GQA groups.
    return cute::min(heads_remaining, (requested_heads / head_group_q) * head_group_q);
  }

  // ScoreBlock2D reuses the same workspace for each contiguous Q-tile chunk.
  // The scheduler receives the chunk's absolute tile start while workspace
  // indexing uses q_tile_count as its local tile extent.
  template <class Kernel>
  typename Kernel::Arguments
  slice_query_tile_arguments(typename Kernel::Arguments args, int query_tile_start, int query_tile_count) const {
    args.kernel.q_tile_start = query_tile_start;
    args.kernel.q_tile_count = query_tile_count;
    return args;
  }

  cutlass::Status run(const Arguments& params, const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShapeType shape = initialize(params);

    typename FMHAPrefillKernel::Arguments arguments{
        {
            shape,
            static_cast<const ElementQ*>(params.q_ptr),
            stride_Q,
            nullptr,
            stride_K,
            nullptr,
            stride_V,
            static_cast<ElementO*>(params.o_ptr),
            stride_O,
            static_cast<const ElementK*>(params.k_ptr),
            stride_K_cache,
            static_cast<const ElementV*>(params.v_ptr),
            stride_V_cache,
            static_cast<const typename FMHAPrefillKernel::ElementSink*>(params.softmax_sink_ptr),
            static_cast<const bool*>(params.skip_batch_mask_ptr),
            params.k_scale_ptr,
            params.v_scale_ptr,
            static_cast<float*>(params.softmax_lse_ptr),
            static_cast<int64_t>(params.total_q),
        },
        {
            params.softmax_scale,
            params.page_table,
            params.page_size,
            params.max_num_pages_per_seq,
            params.window_size_left,
            params.window_size_right,
            static_cast<const ElementQ*>(params.rel_bias_ptr),
            params.rel_bias_token_stride,
            params.rel_bias_head_stride,
            params.rel_bias_extent,
        },
        {},
        hw_info};

    const int batch_total = params.b;
    int batch_slice = batch_total;
    int query_head_slice = params.h;
    int query_tile_slice = -1;
    int q_tiles = 0;
    [[maybe_unused]] int score_workspace_cap_mb = 0;
    if constexpr (CollectiveMainloop::ScoreBlock2D) {
      TORCH_CHECK(
          batch_total > 0 && params.h > 0 && params.seqlen_q > 0 && params.seqlen_k > 0,
          "ScoreBlock2D requires positive batch, query heads, and Q/K sequence lengths");
      q_tiles = cute::ceil_div(params.seqlen_q, int(get<0>(typename FMHAPrefillKernel::TileShapeQK{})));
      query_tile_slice = q_tiles;
      static const int cap_mb = [] {
        if (const char* env = std::getenv("FMHA_SCORE_WS_CAP_MB")) {
          return std::atoi(env);
        }
        return DefaultScoreWorkspaceCapMiB;
      }();
      score_workspace_cap_mb = cap_mb;
      if (cap_mb > 0) {
        const size_t cap_bytes = size_t(cap_mb) << 20;
        const int head_group_q = params.h / params.h_k;
        const auto score_workspace_size = [&](int batches, int query_heads, int query_tiles) {
          auto batch_args = slice_arguments<FMHAPrefillKernel>(arguments, 0, batches);
          auto head_args = slice_query_head_arguments<FMHAPrefillKernel>(batch_args, 0, query_heads);
          auto tile_args = slice_query_tile_arguments<FMHAPrefillKernel>(head_args, 0, query_tiles);
          return FMHAPrefillKernel::get_workspace_size(tile_args);
        };
        const size_t full_batch_workspace = score_workspace_size(1, params.h, q_tiles);
        if (full_batch_workspace <= cap_bytes) {
          // This is the PR342 launch shape: preserve every Q head and tile in
          // one ScoreStore/ScoreLoad pair, reducing only the batch extent.
          batch_slice = int(cute::max(size_t(1), cap_bytes / full_batch_workspace));
          batch_slice = cute::min(batch_total, batch_slice);
        } else {
          // Preserve all Q heads where possible. A Q-tile slice keeps the
          // large multi-head grid from PR342 and needs far fewer launches than
          // slicing into individual GQA heads.
          size_t one_q_tile_workspace = score_workspace_size(1, params.h, 1);
          if (one_q_tile_workspace > cap_bytes) {
            // An entire Q-head set cannot fit even for one Q tile. Reduce
            // heads only as far as required by the workspace cap.
            for (int requested_heads = params.h; requested_heads > 0;) {
              const int candidate_heads = get_query_head_slice_size(0, params.h, head_group_q, requested_heads);
              if (score_workspace_size(1, candidate_heads, 1) <= cap_bytes) {
                query_head_slice = candidate_heads;
                break;
              }
              requested_heads = candidate_heads - 1;
            }
            TORCH_CHECK(
                query_head_slice > 0 && score_workspace_size(1, query_head_slice, 1) <= cap_bytes,
                "ScoreBlock2D one-Q-tile workspace exceeds FMHA_SCORE_WS_CAP_MB=",
                cap_mb,
                "; increase the cap");
            one_q_tile_workspace = score_workspace_size(1, query_head_slice, 1);
          }

          query_tile_slice = cute::min(q_tiles, int(cute::max(size_t(1), cap_bytes / one_q_tile_workspace)));
          const size_t per_batch = score_workspace_size(1, query_head_slice, query_tile_slice);
          batch_slice = int(cute::max(size_t(1), cap_bytes / per_batch));
          batch_slice = cute::min(batch_total, batch_slice);
        }
      }
    }
    const int num_slices = cute::ceil_div(batch_total, batch_slice);

    const int head_group_q = params.h / params.h_k;
    const int first_head_slice = get_query_head_slice_size(0, params.h, head_group_q, query_head_slice);
    if (!FMHAPrefillKernel::can_implement(arguments)) {
      return cutlass::Status::kErrorInvalidProblem;
    }

    // Every batch/head slice reuses the score buffer.
    const auto workspace_shape = [&] {
      auto head_shape = slice_query_head_arguments<FMHAPrefillKernel>(
          num_slices > 1 ? slice_arguments<FMHAPrefillKernel>(arguments, 0, batch_slice) : arguments,
          0,
          first_head_slice);
      if constexpr (CollectiveMainloop::ScoreBlock2D) {
        return slice_query_tile_arguments<FMHAPrefillKernel>(head_shape, 0, query_tile_slice);
      }
      return head_shape;
    }();
    const size_t workspace_size = FMHAPrefillKernel::get_workspace_size(workspace_shape);
    if constexpr (CollectiveMainloop::ScoreBlock2D) {
      if (std::getenv("FMHA_SCORE_WS_VERBOSE") != nullptr) {
        std::fprintf(
            stderr,
            "[fmha] score workspace: batch=%d batch_slice=%d slices=%d query_head_slice=%d query_tile_slice=%d "
            "q_tile_rows=%d bytes=%zu "
            "(%.1f MiB)\n",
            batch_total,
            batch_slice,
            num_slices,
            query_head_slice,
            query_tile_slice,
            int(get<0>(typename FMHAPrefillKernel::TileShapeQK{})),
            workspace_size,
            double(workspace_size) / (1024.0 * 1024.0));
      }
    }
    // Only the HD512 ScoreBlock2D specializations own a score workspace. Other
    // head dimensions pass nullptr, so they never create even a zero-byte XPU
    // allocation here. `workspace` remains alive through all asynchronous
    // ScoreStore/ScoreLoad launches and is released when this runner returns.
    torch::Tensor workspace;
    void* workspace_ptr = nullptr;
    if constexpr (CollectiveMainloop::ScoreBlock2D) {
      TORCH_CHECK(
          score_workspace_cap_mb <= 0 || workspace_size <= (size_t(score_workspace_cap_mb) << 20),
          "ScoreBlock2D minimum workspace (",
          workspace_size,
          " bytes) exceeds FMHA_SCORE_WS_CAP_MB=",
          score_workspace_cap_mb,
          "; increase the cap or reduce sequence length");
      // get_workspace_size() is expressed in bytes. Keep the score workspace
      // byte-addressed so BF16/FP16 tensor options do not allocate two bytes
      // for every requested byte.
      workspace = torch::empty({static_cast<int64_t>(workspace_size)}, torch::device(torch::kXPU).dtype(torch::kByte));
      workspace_ptr = workspace.data_ptr();
    }

    // Initialize the workspace
    FMHAPrefillKernel::initialize_workspace(arguments, workspace_ptr);

    // Run
    if constexpr (CollectiveMainloop::ScoreBlock2D) {
      using ScoreStoreKernel = typename FMHAPrefillKernel::template WithStaticScoreMode<0>;
      using ScoreLoadKernel = typename FMHAPrefillKernel::template WithStaticScoreMode<1>;

      for (int batch_lo = 0; batch_lo < batch_total; batch_lo += batch_slice) {
        const int batch_len = cute::min(batch_slice, batch_total - batch_lo);
        auto batch_args = slice_arguments<FMHAPrefillKernel>(arguments, batch_lo, batch_len);
        for (int query_head = 0; query_head < params.h;) {
          const int head_count = get_query_head_slice_size(query_head, params.h, head_group_q, query_head_slice);
          auto head_args = slice_query_head_arguments<FMHAPrefillKernel>(batch_args, query_head, head_count);
          for (int query_tile = 0; query_tile < q_tiles; query_tile += query_tile_slice) {
            const int query_tile_count = cute::min(query_tile_slice, q_tiles - query_tile);
            auto slice = slice_query_tile_arguments<FMHAPrefillKernel>(head_args, query_tile, query_tile_count);
            launch<ScoreStoreKernel>(ScoreStoreKernel::to_underlying_arguments(slice, workspace_ptr));
            launch<ScoreLoadKernel>(ScoreLoadKernel::to_underlying_arguments(slice, workspace_ptr));
          }
          query_head += head_count;
        }
      }
    } else {
      launch<FMHAPrefillKernel>(FMHAPrefillKernel::to_underlying_arguments(arguments, workspace_ptr));
    }
    return cutlass::Status::kSuccess;
  }
};
template <
    bool Causal,
    bool LocalMask,
    bool Sink,
    bool LSE,
    typename TileShapeQK,
    typename TileShapePV,
    typename TileShapeOutput,
    typename SubgroupLayoutQK,
    typename SubgroupLayoutPV_ = void, /* void -> default */
    bool HasRelBias = false,
    int PipelineStages = 2,            // TODO: This is hard-coded as 1 in kernel.
    bool persistent = false,
    typename ElementQ = bfloat16_t,
    typename ElementK = bfloat16_t,
    typename ElementV = bfloat16_t,
    typename ElementO = bfloat16_t,
    typename MMAOperation_ = void, /* void -> default */
    typename StrideQ = Stride<int, _1, int, int>,
    typename StrideK = Stride<int, _1, int, int>,
    typename StrideV = Stride<_1, int, int, int>,
    typename StrideO = Stride<int, _1, int, int>,
    typename GmemTiledCopyQ = void, /* void -> default block 2D */
    typename GmemTiledCopyK = void,
    typename GmemTiledCopyV = void,
    typename GmemTiledCopyO = void>
struct FMHAConfig {
  static constexpr int SGTileQ = get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})))();
  using MMAOperation = cute::conditional_t<
      is_void_v<MMAOperation_>,
      typename cute::conditional_t<
          cute::is_same_v<ElementQ, cutlass::float_e5m2_t> || cute::is_same_v<ElementQ, cutlass::float_e4m3_t>,
          XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, half_t>,
          XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, ElementQ>>,
      MMAOperation_>;
  using SubgroupLayoutPV = cute::conditional_t<
      is_void_v<SubgroupLayoutPV_>,
      decltype(cutlass::fmha::collective::get_sg_layout_pv(SubgroupLayoutQK{})),
      SubgroupLayoutPV_>;

  template <bool isVarLen, bool CachedKV, bool PagedKV, class Scheduler>
  static int run(const Arguments& params) {
    // The KernelHardwareInfo struct holds the number of EUs on the GPU with a given device ID. This
    // information is used by the underlying kernel.
    cutlass::KernelHardwareInfo hw_info;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    using ProblemShapeType = cutlass::fmha::kernel::FMHAProblemShape<isVarLen>;

    using TiledMMAQK = typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<TileShapeQK>, SubgroupLayoutQK>::TiledMMA;
    using TiledMMAPV = typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<TileShapePV>, SubgroupLayoutPV>::TiledMMA;

    static_assert(
        get<0>(TileShapeOutput{}) == get<0>(TileShapePV{}),
        "Output tile and P*V tile have different sizes in Q dimension");
    constexpr int VTiles = get<1>(TileShapeOutput{}) / get<1>(TileShapePV{});

    auto make_dummy_tensor = [&](auto val, auto stride) {
      return make_tensor(make_gmem_ptr(&val), make_layout(repeat<rank_v<decltype(stride)>>(1), stride));
    };

    using TensorQ = decltype(make_dummy_tensor(ElementQ{}, StrideQ{}));
    using TensorK = decltype(make_dummy_tensor(ElementK{}, StrideK{}));
    using TensorV = decltype(make_dummy_tensor(ElementV{}, StrideV{}));
    using TensorO = decltype(make_dummy_tensor(ElementO{}, StrideO{}));
    using TensorK_cache = TensorK;
    using TensorV_cache = TensorV;
    using GmemTiledCopyK_cache = GmemTiledCopyK;
    using GmemTiledCopyV_cache = GmemTiledCopyV;

    // Mainloop
    using MainloopDispatchPolicy = cutlass::fmha::XeDefault<PipelineStages>;
    using CollectiveMainloop = cutlass::fmha::collective::FMHAFwdMainloop<
        MainloopDispatchPolicy,
        Causal,
        CachedKV,
        PagedKV,
        TiledMMAQK,
        TiledMMAPV,
        VTiles,
        TensorQ,
        TensorK,
        TensorV,
        TensorK_cache,
        TensorV_cache,
        GmemTiledCopyQ,
        GmemTiledCopyK,
        GmemTiledCopyV,
        GmemTiledCopyK_cache,
        GmemTiledCopyV_cache,
        LocalMask,
        false,  // PackGQA is decode-only; relative attention always uses prefill.
        HasRelBias>;

    // Epilogue
    using CollectiveEpilogue = cutlass::fmha::collective::
        FMHAFwdEpilogue<CollectiveMainloop, TileShapeOutput, TensorO, GmemTiledCopyO, Sink, /*PackGQA*/ false, LSE>;

    static_assert(!(persistent & Causal), "persistent SDPA kernel not support Causal yet");
    using FMHAPrefillKernel = conditional_t<
        is_same_v<Scheduler, cutlass::fmha::kernel::XeFHMAIndividualPersistentTileScheduler>,
        cutlass::fmha::kernel::
            XeFMHAFwdDynamicSplitKernel<ProblemShapeType, CollectiveMainloop, CollectiveEpilogue, Scheduler>,
        cutlass::fmha::kernel::XeFMHAFwdKernel<
            ProblemShapeType,
            CollectiveMainloop,
            CollectiveEpilogue,
            Scheduler,
            Step<_2, _0, _1, _3>,
            Step<_2, _0, _1, _3>,
            Step<_0, _2, _1, _3>,
            Step<_2, _0, _1, _3>>>;

    PrefillRunner<FMHAPrefillKernel, isVarLen> kernel;

    kernel.run(params, hw_info);
    return 0;
  }

  // Paged KV cache: the page table encodes absolute KV positions.
  static int run_paged(const Arguments& params) {
    // template <bool isVarLen, bool CachedKV, bool PagedKV, class Scheduler>
    return run<true, true, true, cutlass::fmha::kernel::XeFHMAIndividualTileScheduler>(params);
  }

  // Non-paged (contiguous ragged) KV cache: addressed via cu_seqlens_k offsets.
  static int run_nopaged(const Arguments& params) {
    // template <bool isVarLen, bool CachedKV, bool PagedKV, class Scheduler>
    return run<true, true, false, cutlass::fmha::kernel::XeFHMAIndividualTileScheduler>(params);
  }

  static int run(const Arguments& params) {
    return run_paged(params);
  }
};

// Struct functor for prefill kernel dispatch.
// operator() is declared here; each specialization's body is defined in a
// generated .cpp file (from xe_fmha_fwd_prefill_kernel.cpp.in) so the compiler
// only emits code for the combinations that are actually needed.

template <int HEAD_DIM, class Element = cutlass::bfloat16_t>
struct FmhaPrefillRunner {
  void operator()(const Arguments& params) const;
};

// Non-paged (no_page) prefill is split into its own runner type so its kernel
// instantiations are compiled in translation units separate from the paged
// prefill path, producing independent shared libraries and lowering peak
// compiler memory. Non-paged prefill supports 16-bit (bf16/fp16) queries only (no fp8).
template <int HEAD_DIM, class Element = cutlass::bfloat16_t>
struct FmhaPrefillNpRunner {
  void operator()(const Arguments& params) const;
};

// FP8 KV-cache prefill path is split into its own runner type so that the
// (heavy) e4m3/e5m2 kernel instantiations — which also fan out over
// is_local x is_causal — are compiled in a separate translation unit from the
// 16-bit paged prefill path. This keeps the peak compiler memory of any
// single prefill TU low (avoids OOM during AOT build). The dispatch forwards to
// this when params.is_e4m3 || is_e5m2. The trailing Element is the QUERY dtype
// (bf16 or fp16); K/V stay fp8.
template <int HEAD_DIM, class Element = cutlass::bfloat16_t>
struct FmhaPrefillFp8Runner {
  void operator()(const Arguments& params) const;
};

}  // namespace prefill
