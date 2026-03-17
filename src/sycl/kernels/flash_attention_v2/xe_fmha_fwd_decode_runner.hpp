/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
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
#include "sycl/kernels/flash_attention_v2/kernel/xe_fhma_fwd_kernel.hpp"
#include "sycl/kernels/flash_attention_v2/kernel/xe_reduce_split_k.hpp"
#include "sycl/kernels/flash_attention_v2/kernel/xe_tile_scheduler.hpp"
using namespace cute;
namespace decode {
struct Arguments {
  using index_t = int64_t;

  // The QKV matrices.
  void* __restrict__ q_ptr;
  void* __restrict__ k_ptr;
  void* __restrict__ v_ptr;

  void* __restrict__ k_scale_ptr = nullptr;
  void* __restrict__ v_scale_ptr = nullptr;

  void* __restrict__ temp_out_ptr = nullptr;
  void* __restrict__ exp_sums_ptr = nullptr;
  void* __restrict__ max_logits_ptr = nullptr;
  // The stride between rows of the Q, K and V matrices.
  index_t q_batch_stride;
  index_t k_batch_stride;
  index_t v_batch_stride;
  index_t q_row_stride;
  index_t k_row_stride;
  index_t v_row_stride;
  index_t q_head_stride;
  index_t k_head_stride;
  index_t v_head_stride;
  index_t v_dim_stride;

  // The number of heads.
  int h, h_k;
  int q_group_size = 1;
  bool use_split_kv_decode = false;

  // The O matrix (output).
  void* __restrict__ o_ptr;
  void* __restrict__ oaccum_ptr;

  // The stride between rows of O.
  index_t o_batch_stride;
  index_t o_row_stride;
  index_t o_head_stride;

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
  index_t oaccum_split_stride;
  index_t oaccum_batch_stride;
  index_t oaccum_row_stride;
  index_t oaccum_head_stride;

  // The stride between rows of LSEaccum.
  index_t lseaccum_split_stride;
  index_t lseaccum_batch_stride;
  index_t lseaccum_head_stride;

  // The K_new and V_new matrices.
  void* __restrict__ knew_ptr;
  void* __restrict__ vnew_ptr;

  // The stride between rows of the Q, K and V matrices.
  index_t knew_batch_stride;
  index_t vnew_batch_stride;
  index_t knew_row_stride;
  index_t vnew_row_stride;
  index_t knew_head_stride;
  index_t vnew_head_stride;

  void* __restrict__ qv_ptr;
  index_t qv_batch_stride;
  index_t qv_row_stride;
  index_t qv_head_stride;

  // The cos and sin matrices for rotary embedding.
  void* __restrict__ rotary_cos_ptr;
  void* __restrict__ rotary_sin_ptr;
  int* __restrict__ seqlens_rotary;

  // The indices to index into the KV cache.
  int* __restrict__ kv_batch_idx;

  // PagedKV KV cache
  int* __restrict__ page_table;
  int max_num_pages_per_seq;
  index_t page_table_batch_stride;
  int page_size;
  int num_pages;
  bool pagedkv_tma;

  // The dropout probability (probability of keeping an activation).
  float p_dropout;
  // uint32_t p_dropout_in_uint;
  // uint16_t p_dropout_in_uint16_t;
  uint8_t p_dropout_in_uint8_t;

  // Scale factor of 1 / (1 - p_dropout).
  float rp_dropout;

  // LocalMask window size
  int window_size_left = -1;
  int window_size_right = -1;

  // Pointer to the RNG seed (idx 0) and offset (idx 1).
  uint64_t* rng_state;
  int num_kv_splits;  // For split-KV version

  bool is_bf16;
  bool is_fp32;
  bool is_e4m3;
  bool is_causal;
  bool is_local;

  bool is_rotary_interleaved;

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
    get<1>(problem_size_for_init) = params.use_split_kv_decode ? params.h : params.h_k;
    get<3>(problem_size_for_init) = params.use_split_kv_decode ? params.total_q : params.total_q * params.q_group_size;
    get<4>(problem_size_for_init) = params.total_knew;
    get<5>(problem_size_for_init) = params.total_k;

    ProblemShapeType problem_size_for_launch{
        .batch = get<0>(problem_size),
        .num_heads_q = params.use_split_kv_decode ? get<1>(problem_size) : get<2>(problem_size),
        .num_heads_kv = get<2>(problem_size),
        .seq_len_qo =
            {params.use_split_kv_decode ? params.seqlen_q * params.q_group_size : params.seqlen_q,
             params.use_split_kv_decode ? params.total_q : params.total_q * params.q_group_size,
             nullptr,
             params.use_split_kv_decode ? 1 : params.q_group_size},

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
            static_cast<const ElementV*>(params.k_ptr),
            stride_K_cache,
            static_cast<const ElementV*>(params.v_ptr),
            stride_V_cache,
        },
        {params.softmax_scale, params.page_table, params.page_size, params.max_num_pages_per_seq},
        {},
        hw_info};

    // Define device-global scratch memory
    size_t workspace_size = FMHADecodeKernel::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    if (!FMHADecodeKernel::can_implement(arguments)) {
      return cutlass::Status::kErrorInvalidProblem;
    }

    // Initialize the workspace
    FMHADecodeKernel::initialize_workspace(arguments, workspace.get());

    // Convert host-side arguments to device-side arguments to be passed to the kernel
    auto kernel_params = FMHADecodeKernel::to_underlying_arguments(arguments, workspace.get());

    // Run
    // run(kernel_params);
    launch<FMHADecodeKernel>(kernel_params);
    return cutlass::Status::kSuccess;
  }
};

template <class FMHAKernel, class ReductionSplitKernel, bool isVarLen>
struct DecodeKernelLauncher {
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

      shape.seq_len_qo = cutlass::fmha::collective::VariableLength{params.seqlen_q};
      shape.seq_len_qo.cumulative_length = reinterpret_cast<int*>(params.cu_seqlens_q);
      shape.seq_len_kv = cutlass::fmha::collective::VariableLength{params.seqlen_k};
      shape.seq_len_kv.cumulative_length = reinterpret_cast<int*>(params.cu_seqlens_k);
    } else {
      shape.seq_len_qo = shape_init.seq_len_qo = params.seqlen_q;
      shape.seq_len_kv = shape_init.seq_len_kv = params.seqlen_k;
    }

    auto seq_len_qo = shape_init.seq_len_qo;
    auto seq_len_kv = shape_init.seq_len_kv;

    num_kv_splits = params.num_kv_splits;

    stride_Q =
        cutlass::make_cute_packed_stride(StrideQ{}, cute::make_shape(seq_len_qo, head_size_qk, num_heads_q, batch));
    stride_K =
        cutlass::make_cute_packed_stride(StrideK{}, cute::make_shape(seq_len_kv, head_size_qk, num_heads_kv, batch));
    stride_V =
        cutlass::make_cute_packed_stride(StrideV{}, cute::make_shape(head_size_vo, seq_len_kv, num_heads_kv, batch));
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
        },
        {params.softmax_scale,
         params.k_scale_ptr,
         params.v_scale_ptr,
         static_cast<int*>(params.page_table),
         params.page_size,
         params.max_num_pages_per_seq,
         params.total_k,
         params.window_size_left,
         params.window_size_right},
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
         params.window_size_left},
        hw_info,
        params.num_kv_splits};

    // Define device-global scratch memory
    size_t workspace_size = FMHAKernel::get_workspace_size(arguments);
    size_t reduce_workspace_size = ReductionSplitKernel::get_workspace_size(reduce_arg);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size + reduce_workspace_size);

    if (!FMHAKernel::can_implement(arguments)) {
      // std::cout << "Invalid Problem Size: " << params.batch_size << 'x'
      //           << params.num_heads_q << 'x' << params.max_queries << 'x'
      //           << params.max_keys << 'x' << params.head_size << 'x'
      //           << params.head_size << std::endl;
      return cutlass::Status::kErrorInvalidProblem;
    }

    // Initialize the workspace
    FMHAKernel::initialize_workspace(arguments, workspace.get());

    // Convert host-side arguments to device-side arguments to be passed to the
    // kernel
    auto kernel_params = FMHAKernel::to_underlying_arguments(arguments, workspace.get());
    auto reduce_params = ReductionSplitKernel::to_underlying_arguments(reduce_arg, workspace.get() + workspace_size);

    ReductionSplitKernel::initialize_workspace(reduce_arg, workspace.get() + workspace_size);
    run(kernel_params, reduce_params, params.num_kv_splits > 1);

    return cutlass::Status::kSuccess;
  }

  static void
  run(typename FMHAKernel::Params params, typename ReductionSplitKernel::Params reduce_params, bool need_reduce) {
    auto stream = at::xpu::getCurrentXPUStream();
    auto q = stream.queue();

    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;

    dim3 const block = FMHAKernel::get_block_shape();
    dim3 const grid = FMHAKernel::get_grid_shape(params);

    // cute::print("Launching FMHAKernel with grid: "); cute::print("%d x %d x
    // %d ", grid.x, grid.y, grid.z); cute::print("and block: ");
    // cute::print("%d x %d x %d\n", block.x, block.y, block.z);

    // configure smem size and carveout
    int smem_size = FMHAKernel::SharedStorageSize;

    const auto sycl_block = compat::dim3(block.x, block.y, block.z);
    const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);

    // Launch parameters depend on whether SYCL compiler supports work-group
    // scratch memory extension
    compat::experimental::launch_properties launch_props{
        syclex::work_group_scratch_size(smem_size),
    };
    compat::experimental::kernel_properties kernel_props{
        syclex::sub_group_size<cute::intel::sg_size>, intelex::grf_size<256>};
    compat::experimental::launch_policy policy{sycl_grid, sycl_block, launch_props, kernel_props};

    sycl::ext::oneapi::experimental::launch_config config(policy.get_range(), policy.get_launch_properties());
    auto cgf = [&](::sycl::handler& cgh) {
      auto KernelFunctor =
          compat::experimental::detail::build_kernel_functor<cutlass::device_kernel<FMHAKernel>>(cgh, policy, params);
      sycl::ext::oneapi::experimental::detail::
          LaunchConfigAccess<sycl::nd_range<3>, decltype(policy.get_launch_properties())>
              ConfigAccess(config);
      cgh.parallel_for<KernelCur<FMHAKernel>>(ConfigAccess.getRange(), ConfigAccess.getProperties(), KernelFunctor);
    };

    q.submit(cgf);
    // auto event =
    //     compat::experimental::launch<cutlass::device_kernel<FMHAKernel>>(
    //         policy, queue, params);
    // EventManager::getInstance().addEvent(event);

    // event.wait();

    if (need_reduce) {
      dim3 const reduce_grid = ReductionSplitKernel::get_grid_shape(reduce_params);
      int reduce_smem_size = ReductionSplitKernel::SharedStorageSize;
      const auto reduce_sycl_block = compat::dim3(block.x, block.y, block.z);
      const auto reduce_sycl_grid = compat::dim3(reduce_grid.x, reduce_grid.y, reduce_grid.z);
      compat::experimental::launch_properties launch_props_reduce{
          syclex::work_group_scratch_size(reduce_smem_size),
      };
      compat::experimental::launch_policy reduce_policy{
          reduce_sycl_grid, reduce_sycl_block, launch_props_reduce, kernel_props};

      sycl::ext::oneapi::experimental::launch_config reduce_config(
          reduce_policy.get_range(), reduce_policy.get_launch_properties());
      auto cgf = [&](::sycl::handler& cgh) {
        auto KernelFunctor =
            compat::experimental::detail::build_kernel_functor<cutlass::device_kernel<ReductionSplitKernel>>(
                cgh, reduce_policy, reduce_params);
        sycl::ext::oneapi::experimental::detail::
            LaunchConfigAccess<sycl::nd_range<3>, decltype(reduce_policy.get_launch_properties())>
                ConfigAccess(reduce_config);
        cgh.parallel_for<KernelCur<ReductionSplitKernel>>(
            ConfigAccess.getRange(), ConfigAccess.getProperties(), KernelFunctor);
      };
      q.submit(cgf);

      // auto reduce_event = compat::experimental::launch<
      //     cutlass::device_kernel<ReductionSplitKernel>>(
      //     reduce_policy, queue, reduce_params);

      // // reduce_event.wait();

      // EventManager::getInstance().addEvent(reduce_event);
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
    int PipelineStages = 1,
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
        GmemTiledCopyV_cache>;

    // Epilogue
    using CollectiveEpilogue =
        cutlass::fmha::collective::FMHAFwdEpilogue<CollectiveMainloop, TileShapeOutput, TensorO, GmemTiledCopyO>;

    static_assert(!(persistent & Causal), "persistent SDPA kernel not support Causal yet");
    using FMHADecodeKernel = conditional_t<
        is_same_v<Scheduler, cutlass::fmha::kernel::XeFHMAIndividualPersistentTileScheduler>,
        cutlass::fmha::kernel::
            XeFMHAFwdDynamicSplitKernel<ProblemShapeType, CollectiveMainloop, CollectiveEpilogue, Scheduler>,
        cutlass::fmha::kernel::XeFMHAFwdKernel<ProblemShapeType, CollectiveMainloop, CollectiveEpilogue, Scheduler>>;

    DecodeRunner<FMHADecodeKernel, isVarLen> kernel;

    kernel.run(params, hw_info);
    return 0;
  }

  static int run(const Arguments& params) {
    if (params.page_table == nullptr || params.cu_seqlens_k == nullptr) {
      TORCH_CHECK(false, "Only support varlen with paged KV or varlen with cached KV now");
    } else {
      // template <bool isVarLen, bool CachedKV, bool PagedKV, class Scheduler>
      return run<true, true, true, cutlass::fmha::kernel::XeFHMAIndividualTileScheduler>(params);
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
    typename GmemTiledCopyO = void>
struct SplitDeodeConfig {
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
        LocalMask>;

    // Epilogue
    using CollectiveEpilogue = cutlass::fmha::collective::
        DecodeFwdEpilogue<CollectiveMainloop, TileShapeOutput, TensorO, TensorLSE, void, Sink>;

    using FMHAKernel = cutlass::fmha::kernel::
        XeFMHAFwdSplitKVKernel<ProblemShapeType, CollectiveMainloop, CollectiveEpilogue, Scheduler>;

    using ReduceSplitKernel = cutlass::reduction::kernel::
        ReduceSplitK<ProblemShapeType, cutlass::fmha::kernel::XeReduceSplitKTileScheduler, FMHAKernel>;

    DecodeKernelLauncher<FMHAKernel, ReduceSplitKernel, isVarLen> launcher;

    launcher.run(params, hw_info);
  }

  static void kernel_dispatch(const Arguments& params) {
    return run<true, true, true, cutlass::fmha::kernel::DecodeTileScheduler>(params);
  }
};

std::vector<at::Tensor> mha_fwd(
    const at::Tensor& q,  // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
    const at::Tensor& k,  // (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size,
                          // h_k, d) if there is page_table.
    const at::Tensor& v,  // (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages,
                          // page_size, h_k, dv) if there is page_table.
    std::optional<const at::Tensor>& q_v_,  // (b, s_q, h, dv) or (total_q_new, h, dv) if there is cu_seqlens_q
    const at::Tensor& cu_seqlens_q,         // b+1
    const at::Tensor& cu_seqlens_k,         // b+1
    int max_seqlen_q,
    int max_seqlen_k,
    std::optional<const at::Tensor>& page_table,       // (b_k, max_num_pages_per_seq)
    std::optional<const at::Tensor>& kv_batch_idx_,    // b. indices to index into the KV cache
    std::optional<const at::Tensor>& leftpad_k_,       // b
    std::optional<const at::Tensor>& rotary_cos_,      // seqlen_ro x (rotary_dim / 2)
    std::optional<const at::Tensor>& rotary_sin_,      // seqlen_ro x (rotary_dim / 2)
    std::optional<const at::Tensor>& seqlens_rotary_,  // b
    std::optional<at::Tensor>& q_descale_,             // (b, h_k), not (b, h)
    std::optional<at::Tensor>& k_descale_,             // (b, h_k)
    std::optional<at::Tensor>& v_descale_,             // (b, h_k)
    const float softmax_scale_,
    std::optional<const at::Tensor>& sinks_,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    float const softcap,
    bool const is_rotary_interleaved,  // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
    std::optional<at::Tensor>& scheduler_metadata_,  // (b + 1)
    // int num_kv_splits,
    std::optional<bool> pack_gqa_,
    int const sm_margin) {
  auto q_type = q.scalar_type();
  TORCH_CHECK(
      q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16,
      "mha_fwd only supports Half and BFloat16, got",
      q_type);

  TORCH_CHECK(k.scalar_type() == q_type, "query and key must have the same dtype");
  TORCH_CHECK(v.scalar_type() == q_type, "query and value must have the same dtype");
  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_INPUT(v);
  TORCH_CHECK(
      q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

  TORCH_CHECK(page_table.value().dtype() == torch::kInt32, "page_table must have dtype torch.int32");
  TORCH_CHECK(page_table.value().stride(-1) == 1, "page_table must have contiguous last dimension");

  TORCH_CHECK(q.dim() == 3, "query must be in ragged format");
  CHECK_INPUT(cu_seqlens_q);
  TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype torch.int32");

  CHECK_INPUT(cu_seqlens_k);
  TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype torch.int32");

  auto const sizes = q.sizes();
  const int batch_size = cu_seqlens_q.size(0) - 1;
  int seqlen_q = max_seqlen_q;
  int total_q = q.size(0);
  int num_heads = q.size(-2);
  int const head_size = q.size(-1);
  int const head_size_v = v.size(-1);
  int const max_num_pages_per_seq = page_table.value().size(1);
  int const num_pages = k.size(0);
  int const page_size = k.size(1);
  int const seqlen_k = max_num_pages_per_seq * page_size;
  int const total_k = num_pages * page_size;
  int const num_heads_k = k.size(-2);
  int q_group_size = num_heads / num_heads_k;

  int const batch_size_k = page_table.value().size(0);
  float softmax_scale = softmax_scale_;

  if (!kv_batch_idx_.has_value()) {
    TORCH_CHECK(batch_size == batch_size_k, "batch_size must be equal to batch_size_k");
  }

  // Currently only support head dims <= 256
  static constexpr int max_headdim = 256;
  TORCH_CHECK(head_size <= max_headdim, "FlashAttention forward only supports head dimension at most ", max_headdim);
  TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

  // This needs to go before kBlockM & kBlockN since we rely on the correct window_size and is_causal to set kBlockM
  // TODO: check this

  if (window_size_left >= seqlen_k - 1) {
    window_size_left = -1;
  }
  window_size_right = min(window_size_right, seqlen_q);
  // causal=true is the same as causal=false in this case
  if (is_causal) {
    window_size_right = 0;
  }

  CHECK_SHAPE(k, num_pages, page_size, num_heads_k, head_size);
  CHECK_SHAPE(v, num_pages, page_size, num_heads_k, head_size_v);
  CHECK_SHAPE(page_table.value(), batch_size_k, max_num_pages_per_seq);

  if (leftpad_k_.has_value()) {
    auto leftpad_k = leftpad_k_.value();
    TORCH_CHECK(leftpad_k.dtype() == torch::kInt32, "leftpad_k must have dtype int32");
    CHECK_INPUT(leftpad_k);
    CHECK_SHAPE(leftpad_k, batch_size);
  }

  static constexpr int alignment = 8;
  TORCH_CHECK(head_size % alignment == 0, "head_size should be a multiple of " + std::to_string(alignment));
  TORCH_CHECK(head_size_v % alignment == 0, "head_size_v should be a multiple of " + std::to_string(alignment));

  auto opts = q.options();
  auto device_opts = opts.device(q.device());
  at::Tensor out;
  at::Tensor temp_out;    // [batch, num_kv_splits, num_head_q, seq_q, head_size]
  at::Tensor exp_sums;    // [batch, num_head_q, seq_q, num_kv_splits]
  at::Tensor max_logits;  // [batch, num_head_q, seq_q, num_kv_splits]
  int num_kv_splits = 1;
  out = torch::empty({total_q, num_heads, head_size_v}, opts);
  Arguments params;
  params.use_split_kv_decode = true;
  if (params.use_split_kv_decode) {
    //
    auto get_num_splits = [](int batch_size, int num_heads_kv, int max_seqlen_k, int block_size) {
      auto stream = at::xpu::getCurrentXPUStream();
      auto queue = stream.queue();
      auto device = queue.get_device();
      int num_xe_cores = device.get_info<sycl::ext::intel::info::device::gpu_slices>() *
                         device.get_info<sycl::ext::intel::info::device::gpu_subslices_per_slice>();

      int parallel_ = num_xe_cores;
      int parallel_2 = num_xe_cores * 2;
      int cur_parallel_d = batch_size * num_heads_kv;

      int num_splits = (parallel_ + cur_parallel_d - 1) / cur_parallel_d;

      if (cur_parallel_d * num_splits > parallel_ && num_splits > 1) {
        num_splits = std::ceil(parallel_2 / static_cast<float>(cur_parallel_d)) - 1;
      }

      int max_splits = (max_seqlen_k + block_size - 1) / block_size;
      max_splits = std::min(max_splits, parallel_);
      return std::min(num_splits, max_splits);
    };
    //
    num_kv_splits = 1;  // get_num_splits(batch_size, num_heads_k, max_seqlen_k, page_size);
    temp_out = num_kv_splits == 1
                   ? out
                   : torch::empty({total_q, num_kv_splits * num_heads, head_size_v}, q.options().device(q.device()));
    // auto float_options = opts.dtype(at::kFloat).device(q.device());

    exp_sums = torch::empty({total_q, num_heads, num_kv_splits}, q.options().dtype(at::kFloat).device(q.device()));
    max_logits = torch::empty({total_q, num_heads, num_kv_splits}, q.options().dtype(at::kFloat).device(q.device()));
    params.temp_out_ptr = temp_out.data_ptr();
    params.exp_sums_ptr = exp_sums.data_ptr();
    params.max_logits_ptr = max_logits.data_ptr();
  }
  int const head_size_rounded = round_up_headdim(head_size);
  int const head_size_v_rounded = head_size_v == head_size ? head_size_rounded : round_up_headdim(head_size_v);

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  c10::DeviceGuard device_guard(q.device());

  at::Tensor softmax_lse;
  softmax_lse = torch::empty({num_heads, total_q}, opts.dtype(at::kFloat));

  // align with FA3

  params.is_bf16 = q.dtype() == torch::kBFloat16;

  // Set the pointers and strides.
  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  // All stride are in elements, not bytes.
  params.q_row_stride = q.stride(-3);
  params.k_row_stride = k.stride(-3);
  params.v_row_stride = v.stride(-3);
  params.q_head_stride = q.stride(-2);
  params.k_head_stride = k.stride(-2);
  params.v_head_stride = v.stride(-2);
  params.v_dim_stride = v.stride(-1);
  params.o_ptr = out.data_ptr();
  params.o_row_stride = out.stride(-3);
  params.o_head_stride = out.stride(-2);

  params.cu_seqlens_q = cu_seqlens_q.data_ptr<int>();
  params.cu_seqlens_k = cu_seqlens_k.data_ptr<int>();
  params.num_kv_splits = num_kv_splits;
  // Softmax sum
  params.softmax_lse_ptr = softmax_lse.data_ptr();

  // Set the dimensions.
  params.b = batch_size;
  params.h = num_heads;
  params.h_k = num_heads_k;
  params.q_group_size = num_heads / num_heads_k;
  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;
  params.d = head_size;
  params.d_rounded = head_size_rounded;

  // Set the different scale values.
  params.softmax_scale = softmax_scale;
  bool use_sink = sinks_.has_value();
  params.softmax_sink_ptr = use_sink ? sinks_.value().data_ptr() : nullptr;

  params.softcap = softcap;

  // Set this to probability of keeping an element to simplify things.
  params.p_dropout = 1.f;

  // Causal is the special case where window_size_right == 0 and window_size_left < 0.
  // LocalMask is the more general case where window_size_right >= 0 or window_size_left >= 0.
  params.is_causal = window_size_left < 0 && window_size_right == 0;
  params.is_local = (window_size_left >= 0 || window_size_right >= 0) && !params.is_causal;

  // TODO: check this
  if (window_size_left < 0) {
    window_size_left = seqlen_k - 1;
  }
  if (window_size_right < 0) {
    window_size_right = seqlen_q - 1;
  }
  params.window_size_left = window_size_left;
  params.window_size_right = window_size_right;
  params.total_q = total_q;
  params.total_k = total_k;
  params.b_k = batch_size_k;
  params.dv = head_size_v;
  params.page_table = page_table.value().data_ptr<int>();
  params.page_table_batch_stride = page_table.value().stride(0);
  params.max_num_pages_per_seq = max_num_pages_per_seq;
  params.page_size = page_size;
  params.num_pages = num_pages;

  if (q_v_.has_value()) {
    TORCH_CHECK(head_size <= 64, "q_v is only supported for head_size <= 64");
    TORCH_CHECK(
        q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16,
        "q_v is only supported for fp16 and bf16 data type");
    TORCH_CHECK(false, "q_v is not supported yet");
    at::Tensor q_v = q_v_.value();
    TORCH_CHECK(q_v.dtype() == q_type, "q_v must have the same dtype as query");
    TORCH_CHECK(q_v.stride(-1) == 1, "q_v tensor must have contiguous last dimension");
    CHECK_SHAPE(q_v, total_q, num_heads, head_size_v);
    params.qv_ptr = q_v.data_ptr();
    // All stride are in elements, not bytes.
    params.qv_row_stride = q_v.stride(-3);
    params.qv_head_stride = q_v.stride(-2);
  }

  if (rotary_cos_.has_value()) {
    auto rotary_cos = rotary_cos_.value();
    CHECK_INPUT(rotary_cos);
    params.rotary_dim = rotary_cos.size(1) * 2;
    TORCH_CHECK(params.rotary_dim <= head_size, "rotary_dim must be <= headdim");
    TORCH_CHECK(params.rotary_dim % 16 == 0, "Only rotary dimensions divisible by 16 are currently supported");
    const int seqlen_ro = rotary_cos.size(0);
    TORCH_CHECK(seqlen_ro >= seqlen_k, "cos/sin seqlen must be at least the seqlen of KV cache");
    CHECK_SHAPE(rotary_cos, seqlen_ro, params.rotary_dim / 2);
    TORCH_CHECK(rotary_cos.scalar_type() == q_type, "rotary_cos must have the same dtype as query");

    TORCH_CHECK(rotary_sin_.has_value(), "If rotary cos is provided, rotary sin must also be provided");
    auto rotary_sin = rotary_sin_.value();
    CHECK_INPUT(rotary_sin);
    CHECK_SHAPE(rotary_sin, seqlen_ro, params.rotary_dim / 2);
    TORCH_CHECK(rotary_sin.scalar_type() == q_type, "rotary_cos must have the same dtype as query");
    params.rotary_cos_ptr = rotary_cos.data_ptr();
    params.rotary_sin_ptr = rotary_sin.data_ptr();
    params.is_rotary_interleaved = is_rotary_interleaved;
    if (seqlens_rotary_.has_value()) {
      at::Tensor seqlens_rotary = seqlens_rotary_.value();
      CHECK_INPUT(seqlens_rotary);
      TORCH_CHECK(seqlens_rotary.dtype() == torch::kInt32, "seqlens_rotary must have dtype torch.int32");
      CHECK_SHAPE(seqlens_rotary, batch_size);
      params.seqlens_rotary = seqlens_rotary.data_ptr<int>();
    }
  } else {
    params.rotary_dim = 0;
  }

  params.tensor_opts = torch::TensorOptions().dtype(torch::kUInt8).device(q.device());

  at::Tensor out_accum, softmax_lse_accum;
  auto outaccum_type = at::ScalarType::Float;

  constexpr bool Causal = false;  // The decode kernel does not support causal mode. It must be set to false.
  using ShapeQK = Shape<_8, _64, _64>;
  using ShapePV = Shape<_8, _32, _64>;
  using ShapeOut = Shape<_8, _64>;
  using SubgroupLayoutQK = Layout<Shape<_1, _4, _1>>;
  // SplitDeodeConfig<Causal, LocalMask, Sink, TileShapeQK, TileShapePV, TileShapeOutput,
  // SubgroupLayoutQK>::run(params);
  SplitDeodeConfig<false, false, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK>::kernel_dispatch(params);

  // auto launch_kernel = [&](auto _QG_SZ, auto _HEAD_DIM, auto _PAGE_SIZE, auto _NUM_SG) {
  //   using TileShapeQK = cute::Shape<decltype(_QG_SZ), decltype(_PAGE_SIZE), _64>;
  //   using TileShapePV = cute::Shape<decltype(_QG_SZ), _32, decltype(_PAGE_SIZE)>;
  //   using TileShapeOutput = cute::Shape<decltype(_QG_SZ), decltype(_HEAD_DIM)>;
  //   using SubgroupLayoutQK = cute::Layout<cute::Shape<_1, decltype(_NUM_SG), _1>>;

  //   AT_DISPATCH_BOOL_NO_RETURN(use_sink, Sink, {
  //     AT_DISPATCH_BOOL_NO_RETURN(params.is_local, LocalMask, {
  //       DecodeConfig<Causal, LocalMask, Sink, TileShapeQK, TileShapePV, TileShapeOutput,
  //       SubgroupLayoutQK>::run(params);
  //     });
  //   });
  // };

  // auto dispatch_page_size = [&](auto _QG_SZ, auto _HEAD_DIM) {
  //   switch (params.page_size) {
  //     case 32:
  //       launch_kernel(_QG_SZ, _HEAD_DIM, _32{}, _2{});
  //       break;
  //     case 64:
  //       launch_kernel(_QG_SZ, _HEAD_DIM, _64{}, _4{});
  //       break;
  //     case 128:
  //       launch_kernel(_QG_SZ, _HEAD_DIM, _128{}, _8{});
  //       break;
  //     default:
  //       TORCH_CHECK(false, "Unsupported page size for decode attention: ", params.page_size);
  //   }
  // };

  // auto dispatch_q_group = [&](auto _HEAD_DIM) {
  //   switch (nextPowerOf2(max_seqlen_q)) {
  //     case 1:
  //       dispatch_page_size(_1{}, _HEAD_DIM);
  //       break;
  //     case 2:
  //       dispatch_page_size(_2{}, _HEAD_DIM);
  //       break;
  //     case 4:
  //       dispatch_page_size(_4{}, _HEAD_DIM);
  //       break;
  //     case 8:
  //       dispatch_page_size(_8{}, _HEAD_DIM);
  //       break;
  //     case 16:
  //       dispatch_page_size(_16{}, _HEAD_DIM);
  //       break;
  //     case 32:
  //       dispatch_page_size(_32{}, _HEAD_DIM);
  //       break;
  //     default:
  //       TORCH_CHECK(false, "Unsupported qgroup_size for decode attention: ", max_seqlen_q);
  //   }
  // };

  // switch (params.d) {
  //   case 64:
  //     dispatch_q_group(_64{});
  //     break;
  //   case 96:
  //     dispatch_q_group(_96{});
  //     break;
  //   case 128:
  //     dispatch_q_group(_128{});
  //     break;
  //   case 192:
  //     dispatch_q_group(_192{});
  //     break;
  //   default:
  //     TORCH_CHECK(false, "Unsupported head size for decode attention: ", params.d);
  // }
  return {out, softmax_lse, out_accum, softmax_lse_accum};
}
}  // namespace decode
