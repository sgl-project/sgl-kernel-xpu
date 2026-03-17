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
}  // namespace decode
