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

#include <cute/tensor.hpp>
#include <random>

#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "sycl/Utils.h"
#include "sycl/comm/common.h"
#include "sycl/kernels/flash_attention_v2/collective/fmha_fusion.hpp"
#include "sycl/kernels/flash_attention_v2/relative_attention.hpp"
#include "sycl/kernels/flash_attention_v2/kernel/xe_fmha_fwd_kernel.hpp"
#include "sycl/kernels/flash_attention_v2/kernel/xe_reduce_split_k.hpp"
#include "sycl/kernels/flash_attention_v2/kernel/xe_tile_scheduler.hpp"
using namespace cute;
namespace decode {

// Device shared-local-memory budget in bytes, keyed to the target Xe arch.
#if SYCL_INTEL_TARGET == 20
constexpr int slm_byte_budget = 128 * 1024;  // Xe20: 128 KiB SLM
#elif SYCL_INTEL_TARGET == 35
constexpr int slm_byte_budget = 384 * 1024;  // Xe35: 384 KiB SLM
#else
#error "Unsupported SYCL_INTEL_TARGET for decode SLM budget"
#endif

// Number of subgroups the KV (page) tile is split across, capped so the epilogue
// cross-subgroup reduction buffer fits in shared local memory.
//
// The reduction path (FMHAFwdEpilogue::SharedStorageReduceK) allocates
//   num_sg * (qg * head_dim_rounded + 2 * aligned_qg) floats
// of SLM (each of the num_sg subgroups stores its full partial O tile plus a
// padded row of running sum/max). For large head dims combined with large GQA
// query groups and large page sizes this exceeds the device SLM budget (128 KiB),
// which hangs the device instead of failing cleanly (e.g. HEAD_DIM=512, qg=8,
// PAGE_SIZE=128 -> 132 KiB). Reducing the subgroup count keeps the buffer within
// budget; it only lowers KV parallelism and does not affect correctness.
//
// This mirrors the epilogue's SLM layout by hand; the DecodeRunner/
// SplitDecodeKernelRunner run() paths add a static_assert on the fully
// instantiated kernel's real SharedStorageSize as a compile-time backstop in
// case that layout ever diverges from this formula.
//
// full_kv_subgroups (= PAGE_SIZE / 16) is always a power of two, so the returned
// value is the largest power-of-two divisor that fits the budget (>= 1).
constexpr int kv_subgroups_for_slm(int full_kv_subgroups, int qg, int head_dim_rounded) {
  constexpr int slm_float_budget = slm_byte_budget / static_cast<int>(sizeof(float));
  const int aligned_qg = ((qg + intel::sg_size - 1) / intel::sg_size) * intel::sg_size;
  const int floats_per_subgroup = qg * head_dim_rounded + 2 * aligned_qg;
  const int max_subgroups = slm_float_budget / floats_per_subgroup;
  int sg = full_kv_subgroups;
  while (sg > 1 && sg > max_subgroups) {
    sg /= 2;
  }
  return sg < 1 ? 1 : sg;
}

struct Arguments {
  // The QKV matrices.
  void* __restrict__ q_ptr;
  void* __restrict__ k_ptr;
  void* __restrict__ v_ptr;

  // FP8 KV cache per-tensor descale. The single scalar lives on-device; the
  // kernel dereferences these pointers so no host-side D2H sync (.item()) is
  // needed. Null => no fp8 dequant (scale = 1.0f).
  const float* k_scale_ptr = nullptr;
  const float* v_scale_ptr = nullptr;

  void* __restrict__ temp_out_ptr = nullptr;
  void* __restrict__ exp_sums_ptr = nullptr;
  void* __restrict__ max_logits_ptr = nullptr;
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

  int64_t k_stride_page = 0;
  int64_t k_stride_seq = 0;
  int64_t k_stride_heads = 0;
  int64_t v_stride_page = 0;
  int64_t v_stride_seq = 0;
  int64_t v_stride_heads = 0;

  // The number of heads.
  int h, h_k;
  int q_group_size = 1;
  int num_kv_splits = -1;  // For split-KV version
  bool use_split_kv = false;
  bool use_sink = false;
  bool is_causal = false;
  bool is_local = false;

  // The O matrix (output).
  void* __restrict__ o_ptr;
  void* __restrict__ oaccum_ptr;

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

  // PagedKV KV cache
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

  // LocalMask window size
  int window_size_left = -1;
  int window_size_right = -1;

  // Sheared relative-attention bias. The decode producer uses a zero M-drift
  // surface [total_q, h, rel_bias_head_stride], unlike prefill's Q-tile drift.
  void* __restrict__ rel_bias_ptr = nullptr;
  int64_t rel_bias_token_stride = 0;
  int64_t rel_bias_head_stride = 0;
  int rel_bias_extent = 0;

  // Pointer to the RNG seed (idx 0) and offset (idx 1).
  uint64_t* rng_state;

  bool is_bf16;
  bool is_fp16 = false;
  bool is_fp32;
  bool is_e4m3 = false;
  bool is_e5m2 = false;

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

template <class FMHADecodeKernel, bool isVarLen = false>
struct DecodeRunner {
  using StrideQ = typename FMHADecodeKernel::StrideQ;
  using StrideK = typename FMHADecodeKernel::StrideK;
  using StrideV = typename FMHADecodeKernel::StrideV;
  using StrideO = typename FMHADecodeKernel::StrideO;

  using ElementQ = typename FMHADecodeKernel::ElementQ;
  using ElementK = typename FMHADecodeKernel::ElementK;
  using ElementV = typename FMHADecodeKernel::ElementV;
  using ElementO = typename FMHADecodeKernel::ElementO;

  using CollectiveMainloop = typename FMHADecodeKernel::CollectiveMainloop;
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

  cutlass::Status run(const Arguments& params, const cutlass::KernelHardwareInfo& hw_info) {
    static_assert(
        FMHADecodeKernel::SharedStorageSize <= slm_byte_budget,
        "Decode kernel SharedStorage exceeds the device SLM budget; lower the KV "
        "subgroup split (see kv_subgroups_for_slm).");
    ProblemShapeType shape = initialize(params);

    typename FMHADecodeKernel::Arguments arguments{
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
            static_cast<const typename FMHADecodeKernel::ElementSink*>(params.softmax_sink_ptr),
            static_cast<const bool*>(params.skip_batch_mask_ptr),
            params.k_scale_ptr,
            params.v_scale_ptr,
        },
        {params.softmax_scale,
         params.page_table,
         params.page_size,
         params.max_num_pages_per_seq,
         params.window_size_left,
         params.window_size_right,
         static_cast<const ElementQ*>(params.rel_bias_ptr),
         params.rel_bias_token_stride,
         params.rel_bias_head_stride,
         params.rel_bias_extent},
        {},
        hw_info};

    // Define device-global scratch memory
    size_t workspace_size = FMHADecodeKernel::get_workspace_size(arguments);
    auto workspace = torch::empty(workspace_size, params.tensor_opts);

    if (!FMHADecodeKernel::can_implement(arguments)) {
      return cutlass::Status::kErrorInvalidProblem;
    }

    // Initialize the workspace
    FMHADecodeKernel::initialize_workspace(arguments, workspace.data_ptr());

    // Convert host-side arguments to device-side arguments to be passed to the kernel
    auto kernel_params = FMHADecodeKernel::to_underlying_arguments(arguments, workspace.data_ptr());

    // Run
    launch<FMHADecodeKernel>(kernel_params);
    return cutlass::Status::kSuccess;
  }
};

template <class FMHAKernel, class ReductionSplitKernel, bool isVarLen>
struct SplitDecodeKernelRunner {
  using StrideQ = typename FMHAKernel::StrideQ;
  using StrideK = typename FMHAKernel::StrideK;
  using StrideV = typename FMHAKernel::StrideV;
  using StrideO = typename FMHAKernel::StrideO;

  using ElementQ = typename FMHAKernel::ElementQ;
  using ElementK = typename FMHAKernel::ElementK;
  using ElementV = typename FMHAKernel::ElementV;
  using ElementO = typename FMHAKernel::ElementO;
  using ElementLSE = typename FMHAKernel::ElementLSE;

  using CollectiveMainloop = typename FMHAKernel::CollectiveMainloop;
  using ElementS = typename CollectiveMainloop::ElementS;

  using ProblemShapeType = cutlass::fmha::kernel::FMHAProblemShape<isVarLen>;
  using ProblemShapeTypeInit = cutlass::fmha::kernel::FMHAProblemShape<false>;

  /// Initialization
  StrideQ stride_Q;
  StrideK stride_K;
  StrideV stride_V;
  StrideO stride_O;
  StrideO stride_Oaccum;
  StrideO stride_exp_sums;
  StrideO stride_max_logits;

  int num_kv_splits;

  ProblemShapeType initialize(const Arguments& params) {
    ProblemShapeType shape;
    ProblemShapeTypeInit shape_init;
    auto batch = shape.batch = shape_init.batch = params.b;
    auto num_heads_q = shape.num_heads_q = shape_init.num_heads_q = params.h;
    auto num_heads_kv = shape.num_heads_kv = shape_init.num_heads_kv = params.h_k;
    auto head_size_qk = shape.head_size_qk = shape_init.head_size_qk = params.d;
    auto head_size_vo = shape.head_size_vo = shape_init.head_size_vo = params.d;

    if constexpr (isVarLen) {
      batch = shape_init.batch = 1;
      shape_init.seq_len_qo = params.total_q;
      shape_init.seq_len_kv = params.total_k;

      shape.seq_len_qo = cutlass::fmha::collective::VariableLength{
          params.seqlen_q, params.total_q, reinterpret_cast<int*>(params.cu_seqlens_q)};
      shape.seq_len_kv = cutlass::fmha::collective::VariableLength{
          params.seqlen_k, params.total_k, reinterpret_cast<int*>(params.cu_seqlens_k)};
      shape.seq_len_kv_cache = cutlass::fmha::collective::VariableLength{
          params.seqlen_k, params.total_k, reinterpret_cast<int*>(params.cu_seqlens_k)};
    } else {
      shape.seq_len_qo = shape_init.seq_len_qo = params.seqlen_q;
      shape.seq_len_kv = shape_init.seq_len_kv = params.seqlen_k;
      shape.seq_len_kv_cache = shape_init.seq_len_kv_cache = params.seqlen_k;
    }

    auto seq_len_qo = shape_init.seq_len_qo;
    auto seq_len_kv = shape_init.seq_len_kv;

    num_kv_splits = params.num_kv_splits;

    stride_Q =
        cutlass::make_cute_packed_stride(StrideQ{}, cute::make_shape(seq_len_qo, head_size_qk, num_heads_q, batch));
    if (params.k_stride_seq > 0) {
      // Use actual strides from KV cache tensors (supports non-contiguous
      // layouts such as MLA combined KV cache)
      constexpr int64_t kIntMax = static_cast<int64_t>(std::numeric_limits<int>::max());
      TORCH_CHECK(
          params.k_stride_seq <= kIntMax && params.k_stride_heads <= kIntMax && params.k_stride_page <= kIntMax &&
              params.v_stride_seq <= kIntMax && params.v_stride_heads <= kIntMax && params.v_stride_page <= kIntMax,
          "KV cache stride exceeds int32 max (",
          kIntMax,
          "): k_stride_seq=",
          params.k_stride_seq,
          " k_stride_heads=",
          params.k_stride_heads,
          " k_stride_page=",
          params.k_stride_page,
          " v_stride_seq=",
          params.v_stride_seq,
          " v_stride_heads=",
          params.v_stride_heads,
          " v_stride_page=",
          params.v_stride_page);
      stride_K = StrideK{
          static_cast<int>(params.k_stride_seq),
          _1{},
          static_cast<int>(params.k_stride_heads),
          static_cast<int>(params.k_stride_page)};
      stride_V = StrideV{
          _1{},
          static_cast<int>(params.v_stride_seq),
          static_cast<int>(params.v_stride_heads),
          static_cast<int>(params.v_stride_page)};
    } else {
      stride_K =
          cutlass::make_cute_packed_stride(StrideK{}, cute::make_shape(seq_len_kv, head_size_qk, num_heads_kv, batch));
      stride_V =
          cutlass::make_cute_packed_stride(StrideV{}, cute::make_shape(head_size_vo, seq_len_kv, num_heads_kv, batch));
    }
    stride_O =
        cutlass::make_cute_packed_stride(StrideO{}, cute::make_shape(seq_len_qo, head_size_vo, num_heads_q, batch));
    stride_Oaccum = cutlass::make_cute_packed_stride(
        StrideO{}, cute::make_shape(seq_len_qo, head_size_vo, num_heads_q * num_kv_splits, batch));

    stride_exp_sums =
        cutlass::make_cute_packed_stride(StrideO{}, cute::make_shape(seq_len_qo, num_kv_splits, num_heads_q, batch));

    stride_max_logits =
        cutlass::make_cute_packed_stride(StrideO{}, cute::make_shape(seq_len_qo, num_kv_splits, num_heads_q, batch));

    return shape;
  }

  cutlass::Status run(const Arguments& params, const cutlass::KernelHardwareInfo& hw_info) {
    static_assert(
        FMHAKernel::SharedStorageSize <= slm_byte_budget && ReductionSplitKernel::SharedStorageSize <= slm_byte_budget,
        "Split-decode kernel SharedStorage exceeds the device SLM budget; lower the "
        "KV subgroup split (see kv_subgroups_for_slm).");
    ProblemShapeType shape = initialize(params);

    typename FMHAKernel::Arguments arguments{
        {
            shape,
            reinterpret_cast<ElementQ*>(params.q_ptr),
            stride_Q,
            reinterpret_cast<ElementK*>(params.k_ptr),
            stride_K,
            reinterpret_cast<ElementV*>(params.v_ptr),
            stride_V,
            reinterpret_cast<ElementO*>(params.temp_out_ptr),
            stride_Oaccum,
            reinterpret_cast<ElementLSE*>(params.exp_sums_ptr),
            stride_exp_sums,
            reinterpret_cast<ElementLSE*>(params.max_logits_ptr),
            stride_max_logits,
            reinterpret_cast<ElementQ*>(params.softmax_sink_ptr),
            static_cast<const bool*>(params.skip_batch_mask_ptr),
            params.k_scale_ptr,
            params.v_scale_ptr,
        },
        {params.softmax_scale,
         static_cast<int*>(params.page_table),
         params.page_size,
         params.max_num_pages_per_seq,
         params.total_k,
         params.window_size_left,
         params.window_size_right,
         static_cast<const ElementQ*>(params.rel_bias_ptr),
         params.rel_bias_head_stride,
         params.rel_bias_extent},
        {},
        hw_info,
        params.num_kv_splits};

    typename ReductionSplitKernel::Arguments reduce_arg{
        {shape,
         reinterpret_cast<ElementO*>(params.o_ptr),
         stride_O,
         reinterpret_cast<ElementO*>(params.temp_out_ptr),
         stride_Oaccum,
         reinterpret_cast<ElementLSE*>(params.exp_sums_ptr),
         stride_exp_sums,
         reinterpret_cast<ElementLSE*>(params.max_logits_ptr),
         stride_max_logits,
         params.window_size_left,
         static_cast<const bool*>(params.skip_batch_mask_ptr)},
        hw_info,
        params.num_kv_splits};

    // Define device-global scratch memory
    size_t workspace_size = FMHAKernel::get_workspace_size(arguments);
    size_t reduce_workspace_size = ReductionSplitKernel::get_workspace_size(reduce_arg);
    torch::Tensor workspace = torch::empty(
        {static_cast<int64_t>(workspace_size + reduce_workspace_size)}, torch::device(torch::kXPU).dtype(torch::kByte));
    uint8_t* workspace_ptr = static_cast<uint8_t*>(workspace.data_ptr());

    TORCH_CHECK(
        params.num_kv_splits <= FMHAKernel::max_num_kv_splits,
        "num_splits (",
        params.num_kv_splits,
        ") exceeds the maximum the split-KV decode kernel supports for page_size ",
        params.page_size,
        " (",
        int(FMHAKernel::max_num_kv_splits),
        ")");

    if (!FMHAKernel::can_implement(arguments)) {
      // std::cout << "Invalid Problem Size: " << params.batch_size << 'x'
      //           << params.num_heads_q << 'x' << params.max_queries << 'x'
      //           << params.max_keys << 'x' << params.head_size << 'x'
      //           << params.head_size << std::endl;
      return cutlass::Status::kErrorInvalidProblem;
    }

    // Initialize the workspace
    FMHAKernel::initialize_workspace(arguments, workspace_ptr);

    // Convert host-side arguments to device-side arguments to be passed to the
    // kernel
    auto kernel_params = FMHAKernel::to_underlying_arguments(arguments, workspace_ptr);
    auto reduce_params = ReductionSplitKernel::to_underlying_arguments(reduce_arg, workspace_ptr + workspace_size);

    ReductionSplitKernel::initialize_workspace(reduce_arg, workspace_ptr + workspace_size);
    run(kernel_params, reduce_params, params.num_kv_splits > 1);

    return cutlass::Status::kSuccess;
  }

  static void
  run(typename FMHAKernel::Params params, typename ReductionSplitKernel::Params reduce_params, bool need_reduce) {
    launch<FMHAKernel>(params);

    if (need_reduce) {
      launch<ReductionSplitKernel>(reduce_params);
    }
  }
};

template <
    bool Causal,
    bool LocalMask,
    bool Sink,
    typename TileShapeQK,
    typename TileShapePV,
    typename TileShapeOutput,
    typename SubgroupLayoutQK,
    typename SubgroupLayoutPV_ = void, /* void -> default */
    int PipelineStages = 1,            // TODO: This is hard-coded as 1 in kernel.
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
    typename GmemTiledCopyO = void,
    bool HasRelBias = false>
struct DecodeConfig {
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

    // Pack the GQA query group into the M dimension for decode. Decode always
    // has seq_len_qo == 1, so the M tile is free to hold the head_group_q query
    // heads that share a KV head; the mainloop/epilogue handle the packed
    // local-mask (fixed decode row) and per-row sink. Decode is always
    // non-causal (a single query token cannot be masked by a causal rule), so
    // packing is always enabled here. Prefill keeps the default false on all
    // three components and is unaffected.
    // SGL_DISABLE_PACKGQA: benchmark/debug escape hatch to force the unpacked
    // (per-head launch) decode path for A/B perf comparison.
#ifdef SGL_DISABLE_PACKGQA
    constexpr bool PackGQA = false;
#else
    constexpr bool PackGQA = true;
#endif

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
        PackGQA,
        HasRelBias>;

    // Epilogue
    using CollectiveEpilogue = cutlass::fmha::collective::
        FMHAFwdEpilogue<CollectiveMainloop, TileShapeOutput, TensorO, GmemTiledCopyO, Sink, PackGQA>;

    static_assert(!(persistent & Causal), "persistent SDPA kernel not support Causal yet");
    using FMHADecodeKernel = conditional_t<
        is_same_v<Scheduler, cutlass::fmha::kernel::XeFHMAIndividualPersistentTileScheduler>,
        cutlass::fmha::kernel::
            XeFMHAFwdDynamicSplitKernel<ProblemShapeType, CollectiveMainloop, CollectiveEpilogue, Scheduler>,
        cutlass::fmha::kernel::XeFMHAFwdKernel<
            ProblemShapeType,
            CollectiveMainloop,
            CollectiveEpilogue,
            Scheduler,
            Step<_1, _0, _2, _3>,
            Step<_2, _0, _1, _3>,
            Step<_0, _2, _1, _3>,
            Step<_1, _0, _2, _3>,
            PackGQA>>;

    DecodeRunner<FMHADecodeKernel, isVarLen> kernel;

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

template <
    bool Causal,
    bool LocalMask,
    bool Sink,
    typename TileShapeQK,
    typename TileShapePV,
    typename TileShapeOutput,
    typename SubgroupLayoutQK,
    typename SubgroupLayoutPV_ = void /* void -> default */,
    int PipelineStages = 1,
    typename ElementQ = bfloat16_t,
    typename ElementK = bfloat16_t,
    typename ElementV = bfloat16_t,
    typename ElementO = bfloat16_t,
    typename MMAOperation_ = void, /* void -> default */
    typename StrideQ = Stride<int, _1, int, int>,
    typename StrideK = Stride<int, _1, int, int>,
    typename StrideV = Stride<_1, int, int, int>,
    typename StrideO = Stride<int, _1, int, int>,
    typename StrideOaccum = Stride<int, _1, int, int>,
    typename GmemTiledCopyQ = void, /* void -> default block 2D */
    typename GmemTiledCopyK = void,
    typename GmemTiledCopyV = void,
    typename GmemTiledCopyO = void,
    bool HasRelBias = false>
struct SplitDecodeConfig {
  static constexpr int SGTileQ = get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})))();
  using MMAOperation =
      cute::conditional_t<is_void_v<MMAOperation_>, XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, ElementQ>, MMAOperation_>;
  using SubgroupLayoutPV = cute::conditional_t<
      is_void_v<SubgroupLayoutPV_>,
      decltype(cutlass::fmha::collective::get_sg_layout_pv(SubgroupLayoutQK{})),
      SubgroupLayoutPV_>;

  template <bool isVarLen, bool CachedKV, bool PagedKV, class Scheduler>
  static void run(const Arguments& params) {
    // constexpr bool isVarLen = true;
    // constexpr bool PagedKV = true;
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
    using TensorO = decltype(make_dummy_tensor(ElementO{}, StrideOaccum{}));
    using TensorLSE = decltype(make_dummy_tensor(float{}, StrideO{}));

    // Mainloop
    using MainloopDispatchPolicy = cutlass::fmha::XeDefault<PipelineStages>;
    using CollectiveMainloop = cutlass::fmha::collective::DecodeFwdMainloop<
        MainloopDispatchPolicy,
        PagedKV,
        Causal,
        TiledMMAQK,
        TiledMMAPV,
        VTiles,
        TensorQ,
        TensorK,
        TensorV,
        GmemTiledCopyQ,
        GmemTiledCopyK,
        GmemTiledCopyV,
        LocalMask,
        HasRelBias>;

    // Epilogue
    using CollectiveEpilogue = cutlass::fmha::collective::
        DecodeFwdEpilogue<CollectiveMainloop, TileShapeOutput, TensorO, TensorLSE, void, Sink>;

    using FMHAKernel = cutlass::fmha::kernel::
        XeFMHAFwdSplitKVKernel<ProblemShapeType, CollectiveMainloop, CollectiveEpilogue, Scheduler>;

    using ReduceSplitKernel = cutlass::reduction::kernel::
        ReduceSplitK<ProblemShapeType, cutlass::fmha::kernel::XeReduceSplitKTileScheduler, FMHAKernel>;

    SplitDecodeKernelRunner<FMHAKernel, ReduceSplitKernel, isVarLen> launcher;

    launcher.run(params, hw_info);
  }

  static void run(const Arguments& params) {
    return run<true, true, true, cutlass::fmha::kernel::XeFHMAIndividualTileScheduler>(params);
  }
};

// Struct functors for decode kernel dispatch.
// operator() is declared here; each specialization's body is defined in a
// generated .cpp file (from xe_fmha_fwd_decode_kernel.cpp.in /
// xe_fmha_fwd_split_decode_kernel.cpp.in) so the compiler only emits code
// for the combinations that are actually needed.

template <int QG_SZ, int HEAD_DIM, int PAGE_SIZE, class Element = cutlass::bfloat16_t>
struct FmhaDecodeRunner {
  void operator()(const Arguments& params) const;
};

// Non-paged (no_page) decode is split into its own runner type (no PAGE_SIZE
// template parameter) so its kernel instantiations are compiled in translation
// units separate from the paged decode path, producing independent shared
// libraries and lowering peak compiler memory. Non-paged decode supports 16-bit
// (bf16/fp16) queries only (no fp8 KV cache, no split-KV).
template <int QG_SZ, int HEAD_DIM, class Element = cutlass::bfloat16_t>
struct FmhaDecodeNpRunner {
  void operator()(const Arguments& params) const;
};

template <int QG_SZ, int HEAD_DIM, int PAGE_SIZE, class Element = cutlass::bfloat16_t>
struct FmhaSplitDecodeRunner {
  void operator()(const Arguments& params) const;
};

// FP8 KV-cache decode paths are split into their own runner types so that the
// (heavy) fp8 e4m3/e5m2 kernel instantiations are compiled in a separate
// translation unit from the 16-bit paged decode path. This keeps the peak
// compiler memory of any single decode TU low (avoids OOM during AOT build).
// The trailing Element is the QUERY dtype (bf16 or fp16); K/V stay fp8.
template <int QG_SZ, int HEAD_DIM, int PAGE_SIZE, class Element = cutlass::bfloat16_t>
struct FmhaDecodeFp8Runner {
  void operator()(const Arguments& params) const;
};

template <int QG_SZ, int HEAD_DIM, int PAGE_SIZE, class Element = cutlass::bfloat16_t>
struct FmhaSplitDecodeFp8Runner {
  void operator()(const Arguments& params) const;
};

}  // namespace decode
