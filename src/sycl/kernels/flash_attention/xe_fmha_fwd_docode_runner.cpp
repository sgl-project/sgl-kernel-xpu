/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
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
#pragma once

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/packed_stride.hpp"
#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include "flash_attention_v2/kernel/xe_fhma_fwd_kernel.hpp"
#include "flash_attention_v2/kernel/xe_tile_scheduler.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/sycl_event_manager.hpp"
#include <cute/tensor.hpp>
#include <random>

#include "helper.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "sycl_common.hpp"

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

using namespace cute;

struct Arguments {
  using index_t = int64_t;

  // The QKV matrices.
  void* __restrict__ q_ptr;
  void* __restrict__ k_ptr;
  void* __restrict__ v_ptr;

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
  int b, seqlen_q, seqlen_k, seqlen_knew, d, seqlen_q_rounded, seqlen_k_rounded, d_rounded, rotary_dim;
  int total_q, total_k;
  int total_knew = 0;
  int b_k;             // When having KV cache and with cache_batch_idx, K & V might have larger batch size than Q
  int dv, dv_rounded;  // For the case where V headdim is different from Q/K headdim

  // The scaling factors for the kernel.
  float scale_softmax;
  void* sink_softmax;
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

  // Paged KV cache
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

  // Local window size
  int window_size_left, window_size_right;

  // Pointer to the RNG seed (idx 0) and offset (idx 1).
  uint64_t* rng_state;

  bool is_bf16;
  bool is_fp32;
  bool is_e4m3;
  bool is_causal;
  bool is_local;

  bool is_rotary_interleaved;

  int num_splits;  // For split-KV version
  bool pack_gqa;

  int* __restrict__ tile_count_semaphore;
  // int * __restrict__ num_m_blocks_ptr;
  // int * __restrict__ num_n_blocks_ptr;
  int* __restrict__ num_splits_dynamic_ptr;
  bool skip_scheduler_metadata_computation;

  torch::TensorOptions tensor_opts;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// 3 input matrices: (K)eys, (Q)ueries and (V)alues.
using LayoutQ = cutlass::layout::RowMajor;
using LayoutK = cutlass::layout::ColumnMajor;
using LayoutV = cutlass::layout::RowMajor;
using LayoutO = cutlass::layout::RowMajor;

template <class FMHAKernel, bool isVarLen = false>
struct KernelRunner {

  using StrideQ = typename FMHAKernel::StrideQ;
  using StrideK = typename FMHAKernel::StrideK;
  using StrideV = typename FMHAKernel::StrideV;
  using StrideO = typename FMHAKernel::StrideO;

  using ElementQ = typename FMHAKernel::ElementQ;
  using ElementK = typename FMHAKernel::ElementK;
  using ElementV = typename FMHAKernel::ElementV;
  using ElementO = typename FMHAKernel::ElementO;

  using CollectiveMainloop = typename FMHAKernel::CollectiveMainloop;
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

  // cutlass::DeviceAllocation<ElementQ> block_Q;
  // cutlass::DeviceAllocation<ElementK> block_K;
  // cutlass::DeviceAllocation<ElementV> block_V;
  // cutlass::DeviceAllocation<ElementK> block_K_cache;
  // cutlass::DeviceAllocation<ElementV> block_V_cache;
  // cutlass::DeviceAllocation<ElementO> block_O;
  // cutlass::DeviceAllocation<ElementO> block_ref_O;

  // std::vector<int> cumulative_seqlen_q;
  // std::vector<int> cumulative_seqlen_kv;
  // std::vector<int> cumulative_seqlen_kv_cache;
  // cutlass::DeviceAllocation<int> device_cumulative_seqlen_q;
  // cutlass::DeviceAllocation<int> device_cumulative_seqlen_kv;
  // cutlass::DeviceAllocation<int> device_cumulative_seqlen_kv_cache;

  // struct PagedKVParams {
  //   cutlass::DeviceAllocation<int> page_table;
  //   int page_size = 0;
  //   cutlass::DeviceAllocation<int> num_pages_per_seq;
  // };
  // PagedKVParams paged_kv_cache;

  //
  // Methods
  //

  template<class ProblemShape>
  auto initialize_varlen(const ProblemShape& problem_size) {
    int num_batches = get<0>(problem_size);

    ProblemShape problem_size_for_init = problem_size;
    get<0>(problem_size_for_init) = 1;
    get<3>(problem_size_for_init) = params.total_q;
    get<4>(problem_size_for_init) = params.total_knew;
    get<5>(problem_size_for_init) = params.total_k;

    ProblemShapeType problem_size_for_launch;
    problem_size_for_launch.batch = get<0>(problem_size);
    problem_size_for_launch.num_heads_q = get<1>(problem_size);
    problem_size_for_launch.num_heads_kv = get<2>(problem_size);
    problem_size_for_launch.seq_len_qo = cutlass::fmha::collective::VariableLength{max_seqlen_q};
    problem_size_for_launch.seq_len_kv = cutlass::fmha::collective::VariableLength{max_seqlen_kv};
    problem_size_for_launch.seq_len_kv_cache = cutlass::fmha::collective::VariableLength{max_seqlen_kv_cache};
    problem_size_for_launch.head_size_qk = get<6>(problem_size);
    problem_size_for_launch.head_size_vo = get<7>(problem_size);

    return cute::make_tuple(problem_size_for_init, problem_size_for_launch);
  }

  /// Initialize operands to be used in the GEMM and reference GEMM
  ProblemShapeType initialize(const Arguments &params) {
    int q_group_size = params.h / params.h_k;
    auto problem_shape_in = cute::make_tuple(
        params.b,
        params.h / q_group_size,
        params.h_k,
        params.seqlen_q * q_group_size,
        params.seqlen_knew,
        params.seqlen_k,
        params.d,
        params.dv);
    ProblemShapeType shape;

    decltype(problem_shape_in) problem_size;

    if constexpr (isVarLen) {
      auto [problem_shape_init, problem_shape_launch] = initialize_varlen(problem_shape_in);
      problem_size = problem_shape_init;
      shape = problem_shape_launch;
    } else {
      // problem_size = problem_shape_in;
      // shape.batch        = params.batch;
      // shape.num_heads_q  = params.num_heads_q;
      // shape.num_heads_kv = params.num_heads_kv;
      // shape.seq_len_qo   = params.seq_len_qo;
      // shape.seq_len_kv   = params.seq_len_kv;
      // shape.seq_len_kv_cache = params.seq_len_kv_cache;
      // shape.head_size_qk = params.head_size_qk;
      // shape.head_size_vo = params.head_size_vo;
    }

    auto [batch, num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv, seq_len_kv_cache, head_size_qk, head_size_vo] = problem_size;
    auto shape_Q = cute::make_shape(seq_len_qo, head_size_qk, num_heads_q,  batch);
    auto shape_K = cute::make_shape(seq_len_kv, head_size_qk, num_heads_kv, batch);
    auto shape_V = cute::make_shape(head_size_vo, seq_len_kv, num_heads_kv, batch);
    auto shape_K_cache = cute::make_shape(seq_len_kv_cache, head_size_qk, num_heads_kv, batch);
    auto shape_V_cache = cute::make_shape(head_size_vo, seq_len_kv_cache, num_heads_kv, batch);
    auto shape_O = cute::make_shape(seq_len_qo, head_size_vo, num_heads_q,  batch);

    stride_Q = cutlass::make_cute_packed_stride(StrideQ{}, shape_Q);
    stride_K = cutlass::make_cute_packed_stride(StrideK{}, shape_K);
    stride_V = cutlass::make_cute_packed_stride(StrideV{}, shape_V);
    stride_K_cache = cutlass::make_cute_packed_stride(StrideK{}, shape_K_cache);
    stride_V_cache = cutlass::make_cute_packed_stride(StrideV{}, shape_V_cache);
    stride_O = cutlass::make_cute_packed_stride(StrideO{}, shape_O);

    block_Q.reset(static_cast<std::size_t>(batch) * num_heads_q * seq_len_qo * head_size_qk);
    block_K.reset(static_cast<std::size_t>(batch) * num_heads_kv * seq_len_kv * head_size_qk);
    block_V.reset(static_cast<std::size_t>(batch) * num_heads_kv * seq_len_kv * head_size_vo);
    block_K_cache.reset(static_cast<std::size_t>(batch) * num_heads_kv * seq_len_kv_cache * head_size_qk);
    block_V_cache.reset(static_cast<std::size_t>(batch) * num_heads_kv * seq_len_kv_cache * head_size_vo);
    block_O.reset(static_cast<std::size_t>(batch) * num_heads_q * seq_len_qo * head_size_vo);
    block_ref_O.reset(static_cast<std::size_t>(batch) * num_heads_q * seq_len_qo * head_size_vo);
    // Zero-initialize output buffer for the kernel result
    // block_ref_O is fully written in verify() before being read, so no initialization needed
    compat::memset(block_O.get(), 0, block_O.size() * sizeof(ElementO));
    if (params.use_paged_kv) {
      paged_kv_cache.page_size = params.page_size;
      std::vector<int> num_pages_per_seq{0};
      int num_pages = 0;
      for(int b = 0; b < shape.batch; b++) {
        int seq_len_cache = isVarLen ? cumulative_seqlen_kv_cache[b + 1] - cumulative_seqlen_kv_cache[b] : seq_len_kv_cache;
        int pages_per_seq = ceil_div(seq_len_cache, paged_kv_cache.page_size);
        num_pages_per_seq.push_back(num_pages_per_seq.back() + pages_per_seq);
        num_pages += pages_per_seq;
      }
      paged_kv_cache.page_table.reset(num_pages);

      // initialize block table with random mapping for non-contiguous layout
      std::vector<int> page_mapping(num_pages);
      for (int b = 0; b < shape.batch; ++b) {
        std::vector<int> physical_pages(num_pages_per_seq[b + 1] - num_pages_per_seq[b]);
        std::iota(physical_pages.begin(), physical_pages.end(), 0);
        // shuffle physical pages
        std::shuffle(physical_pages.begin(), physical_pages.end(), std::mt19937{ std::random_device{}() });
        for (int blk = 0; blk < physical_pages.size(); ++blk) {
          int logical_idx = num_pages_per_seq[b] + blk;
          page_mapping[logical_idx] = physical_pages[blk];
        }
      }
      compat::memcpy(paged_kv_cache.page_table.get(), page_mapping.data(), page_mapping.size() * sizeof(int));

      paged_kv_cache.num_pages_per_seq.reset(num_pages_per_seq.size());
      compat::memcpy(paged_kv_cache.num_pages_per_seq.get(), num_pages_per_seq.data(), num_pages_per_seq.size() * sizeof(int));
    }

    initialize_block(block_Q, seed + 2023);
    initialize_block(block_K, seed + 2022);
    initialize_block(block_V, seed + 2021);
    initialize_block(block_K_cache, seed + 2024);
    initialize_block(block_V_cache, seed + 2025);

    if (!cumulative_seqlen_q.empty()) {
      device_cumulative_seqlen_q.reset(cumulative_seqlen_q.size());
      device_cumulative_seqlen_q.copy_from_host(cumulative_seqlen_q.data(), cumulative_seqlen_q.size());
    }

    if (!cumulative_seqlen_kv.empty()) {
      device_cumulative_seqlen_kv.reset(cumulative_seqlen_kv.size());
      device_cumulative_seqlen_kv.copy_from_host(cumulative_seqlen_kv.data(), cumulative_seqlen_kv.size());
    }

    if (!cumulative_seqlen_kv_cache.empty()) {
      device_cumulative_seqlen_kv_cache.reset(cumulative_seqlen_kv_cache.size());
      device_cumulative_seqlen_kv_cache.copy_from_host(cumulative_seqlen_kv_cache.data(), cumulative_seqlen_kv_cache.size());
    }

    if constexpr (isVarLen) {
      shape.seq_len_qo.cumulative_length = device_cumulative_seqlen_q.get();
      shape.seq_len_kv.cumulative_length = device_cumulative_seqlen_kv.get();
      shape.seq_len_kv_cache.cumulative_length = device_cumulative_seqlen_kv_cache.get();
    }
    return shape;
  }

  // Note that the GemmUniversalAdapter currently doesn't support flash attention, which is why this
  // secondary `run` function is required to launch the kernel.
  static void run(typename FMHAKernel::Params params)
  {
    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;

    dim3 const block = FMHAKernel::get_block_shape();
    dim3 const grid = FMHAKernel::get_grid_shape(params);

    // configure smem size and carveout
    int smem_size = FMHAKernel::SharedStorageSize;

    const auto sycl_block = compat::dim3(block.x, block.y, block.z);
    const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);

    // Launch parameters depend on whether SYCL compiler supports work-group scratch memory extension
    compat::experimental::launch_properties launch_props {
      syclex::work_group_scratch_size(smem_size),
    };
    compat::experimental::kernel_properties kernel_props{
      syclex::sub_group_size<cute::intel::sg_size>,
      intelex::grf_size<256>
    };
    compat::experimental::launch_policy policy{sycl_grid, sycl_block, launch_props, kernel_props};
    auto event = compat::experimental::launch<cutlass::device_kernel<FMHAKernel>, FMHAKernel>(policy, params);

    EventManager::getInstance().addEvent(event);
  }

  cutlass::Status run(const Arguments &params, const cutlass::KernelHardwareInfo &hw_info) {

    ProblemShapeType shape = initialize(params);

    typename FMHAKernel::Arguments arguments{
      {
        shape,
        block_Q.get(), stride_Q,
        block_K.get(), stride_K,
        block_V.get(), stride_V,
        block_O.get(), stride_O,
        block_K_cache.get(), stride_K_cache,
        block_V_cache.get(), stride_V_cache,
      },
      {
        params.softmax_scale,
        params.use_paged_kv ? paged_kv_cache.page_table.get() : nullptr,
        params.use_paged_kv ? paged_kv_cache.page_size : 0,
        params.use_paged_kv ? paged_kv_cache.num_pages_per_seq.get() : nullptr
      },
      {},
      hw_info
    };

    // Define device-global scratch memory
    size_t workspace_size = FMHAKernel::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    if (!FMHAKernel::can_implement(arguments)) {
      std::cout << "Invalid Problem Size: " << params.batch << 'x' << params.num_heads_q << 'x' <<
        params.seq_len_qo << 'x' << params.seq_len_kv << 'x' << params.head_size_qk << 'x'  << params.head_size_vo
        << (params.is_causal ? "xCausal" : "xNonCausal") << std::endl;
      return cutlass::Status::kErrorInvalidProblem;
    }

    // Initialize the workspace
    CUTLASS_CHECK(FMHAKernel::initialize_workspace(arguments, workspace.get()));

    // Convert host-side arguments to device-side arguments to be passed to the kernel
    auto params = FMHAKernel::to_underlying_arguments(arguments, workspace.get());

    // Run
    run(params);
    return cutlass::Status::kSuccess;
  }
};

template <bool Causal,
          typename TileShapeQK,
          typename TileShapePV,
          typename TileShapeOutput,
          typename SubgroupLayoutQK,
          typename SubgroupLayoutPV_,      /* void -> default */
          int PipelineStages,
          bool persistent,
          typename ElementQ = bfloat16_t,
          typename ElementK = bfloat16_t,
          typename ElementV = bfloat16_t,
          typename ElementO = float,
          typename MMAOperation_ = void,    /* void -> default */
          typename StrideQ = Stride<int, _1, int, int>,
          typename StrideK = Stride<int, _1, int, int>,
          typename StrideV = Stride<_1, int, int, int>,
          typename StrideO = Stride<int, _1, int, int>,
          typename GmemTiledCopyQ = void,   /* void -> default block 2D */
          typename GmemTiledCopyK = void,
          typename GmemTiledCopyV = void,
          typename GmemTiledCopyO = void>
struct FMHAConfig {

  static constexpr int SGTileQ = get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})))();
  using MMAOperation = cute::conditional_t<is_void_v<MMAOperation_>,
                                           typename cute::conditional_t<
                                               cute::is_same_v<ElementQ, cutlass::float_e5m2_t> || cute::is_same_v<ElementQ, cutlass::float_e4m3_t>,
                                               XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, half_t>,
                                               XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, ElementQ>
                                           >,
                                           MMAOperation_>;
  using SubgroupLayoutPV = cute::conditional_t<is_void_v<SubgroupLayoutPV_>,
                                               decltype(cutlass::fmha::collective::get_sg_layout_pv(SubgroupLayoutQK{})),
                                               SubgroupLayoutPV_>;

  template <bool isVarLen, bool CachedKV, bool PagedKV, class Scheduler>
  static int run(const Arguments &params) {
    //
    // Run examples
    //

    // The KernelHardwareInfo struct holds the number of EUs on the GPU with a given device ID. This
    // information is used by the underlying kernel.
    cutlass::KernelHardwareInfo hw_info;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    using ProblemShapeType = cutlass::fmha::kernel::FMHAProblemShape<isVarLen>;

    using TiledMMAQK = typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<TileShapeQK>, SubgroupLayoutQK>::TiledMMA;
    using TiledMMAPV = typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<TileShapePV>, SubgroupLayoutPV>::TiledMMA;

    static_assert(get<0>(TileShapeOutput{}) == get<0>(TileShapePV{}),
        "Output tile and P*V tile have different sizes in Q dimension");
    constexpr int VTiles = get<1>(TileShapeOutput{}) / get<1>(TileShapePV{});

    auto make_dummy_tensor = [&](auto val, auto stride) {
      return make_tensor(make_gmem_ptr(&val),
                         make_layout(repeat<rank_v<decltype(stride)>>(1), stride));
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
        MainloopDispatchPolicy, Causal, CachedKV, PagedKV,
        TiledMMAQK, TiledMMAPV, VTiles,
        TensorQ, TensorK, TensorV,
        TensorK_cache, TensorV_cache,
        GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV,
        GmemTiledCopyK_cache, GmemTiledCopyV_cache
    >;

    // Epilogue
    using CollectiveEpilogue = cutlass::fmha::collective::FMHAFwdEpilogue<
        CollectiveMainloop,
        TileShapeOutput,
        TensorO,
        GmemTiledCopyO
    >;

    static_assert(!(persistent & Causal), "persistent SDPA kernel not support Causal yet");
    using FMHAKernel = conditional_t<is_same_v<Scheduler, cutlass::fmha::kernel::XeFHMAIndividualPersistentTileScheduler>,
      cutlass::fmha::kernel::XeFMHAFwdDynamicSplitKernel<
        ProblemShapeType, CollectiveMainloop, CollectiveEpilogue, Scheduler>,
        cutlass::fmha::kernel::XeFMHAFwdKernel<
        ProblemShapeType, CollectiveMainloop, CollectiveEpilogue, Scheduler>
        >;

    KernelRunner<FMHAKernel, isVarLen> kernel;

    CUTLASS_CHECK(kernel.run(params, hw_info));
    return 0;
  }

  static int run(const Arguments &params) {
    if (params.page_table != nullptr && params.cu_seqlens_k != nullptr) {
        return run<true, true, false, cutlass::fmha::kernel::XeFHMAIndividualTileScheduler>(params);
    } else {
      return 0;
    }
  }
};
