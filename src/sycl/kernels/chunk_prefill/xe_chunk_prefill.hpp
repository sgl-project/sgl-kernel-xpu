/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,this
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once
#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "xe_flash_attn_chunk_prefill_mma.hpp"
#define THREAD_ID 0
#define BLOCK_ID 0

namespace cutlass::flash_attention::kernel {

template <
    class ProblemShape_,
    class CollectiveMainloop_,
    class CollectiveSoftmaxEpilogue_,
    class CollectiveEpilogue_,
    class TileScheduler_ = void>
class FMHAPrefillChunk;
///////////////////////////////////////////////////////////////////////////////
template <
    class ProblemShape_,
    class CollectiveMainloop_,
    class CollectiveSoftmaxEpilogue_,
    class CollectiveEpilogue_,
    class TileScheduler_>
class FMHAPrefillChunk {
 public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;

  // ProblemShape: <batch, num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv,
  // head_size_qk, head_size_vo>
  static_assert(
      rank(ProblemShape{}) == 8,
      "ProblemShape{} should be <batch, num_heads_q, num_heads_kv, seq_len_qo, "
      "seq_len_kv, seq_len_kv_cache, head_size_qk, head_size_vo>");
  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShapeQK = typename CollectiveMainloop::TileShapeQK;
  using TileShapePV = typename CollectiveMainloop::TileShapePV;
  using TiledMmaQK = typename CollectiveMainloop::TiledMmaQK;
  using TiledMmaPV = typename CollectiveMainloop::TiledMmaPV;
  using ArchTag = typename CollectiveMainloop::ArchTag;
  using ElementQ = typename CollectiveMainloop::ElementQ;
  using StrideQ = typename CollectiveMainloop::StrideQ;
  using ElementK = typename CollectiveMainloop::ElementK;
  using StrideK = typename CollectiveMainloop::StrideK;
  using ElementV = typename CollectiveMainloop::ElementV;
  using StrideV = typename CollectiveMainloop::StrideV;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;
  using traits_load_Q = typename CollectiveMainloop::traits_load_Q;
  using traits_load_K = typename CollectiveMainloop::traits_load_K;

  using CollectiveSoftmaxEpilogue = CollectiveSoftmaxEpilogue_;
  using SoftmaxArguments = typename CollectiveSoftmaxEpilogue::Arguments;
  using SoftmaxParams = typename CollectiveSoftmaxEpilogue::Params;

  static_assert(
      cute::is_void_v<TileScheduler_> or cute::is_same_v<TileScheduler_, PersistentScheduler> or
          cute::is_same_v<TileScheduler_, IndividualScheduler>,
      "Unsupported TileScheduler for Intel Xe.");
  using TileSchedulerTag = TileScheduler_;
  using TileScheduler = typename detail::TileSchedulerSelector<TileScheduler_, ArchTag>::Scheduler;
  using TileSchedulerParams = typename TileScheduler::Params;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementO = typename CollectiveEpilogue::ElementO;
  using StrideO = typename CollectiveEpilogue::StrideO;
  using ElementLSE = typename CollectiveEpilogue::ElementLSE;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;
  using TileShapeOutput = typename CollectiveEpilogue::TileShapeOutput;
  using TiledMmaOutput = typename CollectiveEpilogue::TiledMmaOutput;

  static_assert(
      cute::is_same_v<ElementAccumulator, typename CollectiveEpilogue::ElementAccumulator>,
      "Mainloop and epilogue do not agree on accumulator value type.");
  // MSVC requires the cast to fix a warning-as-error.
  static constexpr int SharedStorageSize = 0;

  static constexpr bool CausalMask = CollectiveMainloop::CausalMask;
  static constexpr bool LocalMask = CollectiveMainloop::LocalMask;

  static_assert(!(CausalMask && LocalMask), "Cannot be both causal and local");
  static constexpr bool PagedKV = CollectiveMainloop::PagedKV;

  static constexpr int SubgroupSize = CollectiveMainloop::SubgroupSize;  // sub_group size
  static constexpr uint32_t MaxThreadsPerBlock = CollectiveMainloop::MaxThreadsPerBlock;
  using MmaAtomShape = typename CollectiveMainloop::MmaAtomShape;  // 8,16,16

  static constexpr int QK_BLK_M = CollectiveMainloop::QK_BLK_M;
  static constexpr int QK_BLK_N = CollectiveMainloop::QK_BLK_N;
  static constexpr int QK_BLK_K = CollectiveMainloop::QK_BLK_K;

  static constexpr int QK_ATOM_N = CollectiveMainloop::QK_ATOM_N;
  static constexpr int QK_ATOM_K = CollectiveMainloop::QK_ATOM_K;

  static constexpr int QK_SG_M = CollectiveMainloop::QK_SG_M;

  static constexpr int Epilogue_BLK_N = get<1>(TileShapeOutput{});
  static constexpr int Epilogue_BLK_K = get<2>(TileShapeOutput{});

  static constexpr int PV_ATOM_M = CollectiveMainloop::PV_ATOM_M;
  static constexpr int PV_ATOM_N = CollectiveMainloop::PV_ATOM_N;
  static constexpr int PV_ATOM_K = CollectiveMainloop::PV_ATOM_K;

  static constexpr auto Num_SGs = PV_ATOM_N * PV_ATOM_M * PV_ATOM_K;
  static constexpr int Vec = CollectiveMainloop::Vec;
  static constexpr int FragsM = CollectiveMainloop::FragsM;
  // The FragsN here used for Creation of S matrix so we use the FragsN for S
  // shape
  static constexpr int FragsN = CollectiveMainloop::FragsNS;

  static constexpr int VSlicer =
      get<1>(TileShapeOutput{}) / (get<1>(TileShapePV{}) * PV_ATOM_N);  // ceil_div(FragsNOut,FragsNS);
  using AccumeShape =
      decltype(make_shape(Int<Vec>{}, Int<FragsM>{}, get<1>(TileShapePV{}) / get<1>(MmaAtomShape()), Int<VSlicer>{}));

  static constexpr bool is_var_len = CollectiveMainloop::is_var_len;
  static constexpr bool rope_enabled = CollectiveMainloop::rope_enabled;
  
  template <typename T>
  static constexpr bool is_fp8_v = cute::is_same_v<T,float_e4m3_t> || cute::is_same_v<T,float_e5m2_t>;
  
  // Kernel level shared memory storage
  struct SharedStorage {
    using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;
    EpilogueTensorStorage epilogue;
  };

  // Device side arguments
  struct Arguments {
    gemm::GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    SoftmaxArguments softmax{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
  };

  // Kernel entry point API
  struct Params {
    gemm::GemmUniversalMode mode;
    ProblemShape problem_shape;
    MainloopParams mainloop;
    SoftmaxParams softmax;
    EpilogueParams epilogue;
    TileSchedulerParams scheduler;
  };

  //
  // Methods
  //

  // Convert to underlying arguments. In this case, a simple copy for the
  // aliased type.
  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    (void)workspace;
    return {
        args.mode,
        args.problem_shape,
        CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, workspace),
        CollectiveSoftmaxEpilogue::to_underlying_arguments(args.softmax),
        CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, workspace),
        TileScheduler::to_underlying_arguments(args.problem_shape, args.hw_info, TileShapeOutput{})};
  }

  static bool can_implement(Arguments const& args) {
    bool mode_implementable = args.mode == gemm::GemmUniversalMode::kGemm or
                              (args.mode == gemm::GemmUniversalMode::kBatched && rank(ProblemShape{}) == 4);
    return mode_implementable;
  }

  static int get_workspace_size(Arguments const& args) {
    return 0;
  }

  static cutlass::Status initialize_workspace(
      Arguments const& args,
      void* workspace = nullptr,
      cudaStream_t stream = nullptr,
      CudaHostAdapter* cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::template get_grid_shape<Num_SGs>(params.scheduler);
  }

  static dim3 get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  CUTLASS_DEVICE
  Shape<int, int> get_sequence_length_shape(ProblemShape const& problem_shape, int const& batch) {
    if constexpr (is_var_len) {
      return cutlass::fmha::collective::apply_variable_length(select<3, 5>(problem_shape), batch);
    } else {
      return select<3, 5>(problem_shape);
    }
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    // Preconditions
    CUTE_STATIC_ASSERT(is_static<TileShapeQK>::value);
    CUTE_STATIC_ASSERT(is_static<TileShapePV>::value);
    // Separate out problem shape for convenience

    // "ProblemShape{} should be <batch, num_heads_q, num_heads_kv, seq_len_qo,
    // seq_len_kv, head_size_qk, head_size_vo>");
    auto batch = get<0>(params.problem_shape);
    auto num_heads_q = get<1>(params.problem_shape);
    auto num_heads_kv = get<2>(params.problem_shape);
    auto seq_len_kv = get<4>(params.problem_shape);

    auto& head_size_qk = get<6>(params.problem_shape);
    auto& head_size_vo = get<7>(params.problem_shape);
    // Preconditions
    static_assert(
        cute::rank(StrideQ{}) == 3,
        "StrideQ must be rank-3: [seq_len_qo, head_size_qk, batch * "
        "num_heads_q].");
    static_assert(
        cute::rank(StrideK{}) == 3,
        "StrideK must be rank-3: [head_size_qk, seq_len_kv, batch * "
        "num_heads_kv].");
    static_assert(
        cute::rank(StrideV{}) == 3,
        "StrideV must be rank-3: [seq_len_kv, head_size_vo, batch * "
        "num_heads_kv].");

    int thread_idx = int(ThreadIdxX());
    // int sub_group_id = thread_idx / SubgroupSize;
    auto sub_group_id = get_sub_group_id();
    auto local_id = get_sub_group_local_id();

    TileScheduler tile_scheduler{params.scheduler};
    CUTLASS_PRAGMA_NO_UNROLL
    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      auto blk_coord = tile_scheduler.get_block_coord();  // head_size_blk_idx, seq_len_blk_idx,
                                                          // batch_blk_idx, num_heads_blk_idx

      auto blk_m_coord = get<0>(blk_coord);   // seq_len_blk_idx
      auto blk_n_coord = 0;                   // nums_head_blk_idx
      auto q_head_coord = get<1>(blk_coord);  // q_heads_idx
      auto batch_coord = get<2>(blk_coord);   // batch_blk_idx

      // For variable sequence length case, batch is considered to be 1 (same
      // as group gemm). For fixed sequence length case, the l_coord is the
      // weighted sum of both batch_coord and num_heads_coord. Flash Attention
      // implementation combines batch and num_heads to calculate the total
      // batch_size. iff is_var_len: batch_size = num_heads (as each batch
      // would have it's own seq_len_qo and seq_len_kv) iff !is_var_len:
      // batch_size = batch * num_heads
      // auto blk_l_coord = q_head_coord;

      // Get problem shape for the current batch_blk_idx. For variable
      // sequence length, it loads the sequence length from Global memory for
      // the given batch_blk_idx and returns the appropriate problem_shape.
      // For fixed sequence length, sequence_length_shape == select<3, 4,
      // 5>(params.problem_shape). sequence_length_shape = [batch,
      // num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv, seq_len_kv_cache,
      // head_size_qk, head_size_vo]
      auto sequence_length_shape = get_sequence_length_shape(params.problem_shape, batch_coord);

      auto [seq_len_qo, seq_len_kv_cache] = sequence_length_shape;
      // int seq_len_kv_total = seq_len_kv_cache + seq_len_kv;
      // For variable sequence length case, batch is considered to be 1 (same
      // as group gemm). For fixed sequence length case, the l_coord is the
      // weighted sum of both batch_coord and num_heads_coord. Flash Attention
      // implementation combines batch and num_heads to calculate the total
      // batch_size. iff is_var_len: batch_size = num_heads (as each batch
      // would have it's own seq_len_qo and seq_len_kv) iff !is_var_len:
      // batch_size = batch * num_heads

      // Calculate the seq_len_idx (blk_m_coord * get<0>(TileShapeOutput{}))
      // and check if it is still within bounds of the actual seq_len_qo
      // (get<0>(sequence_length_shape)).
      if (blk_m_coord * get<0>(TileShapeOutput{}) >= seq_len_qo) {
        continue;
      }

      const int seq_coord =
          cute::min(seq_len_qo, (blk_m_coord * QK_BLK_M + (sub_group_id / PV_ATOM_N) * QK_SG_M) % seq_len_qo);
      // auto offset = cute::min(seq_len_qo, seq_len_kv);  //(2048, 1024)
      // auto discard_seq_coord = seq_len_qo - offset;     // 1024
      // auto full_tile_offset = seq_len_kv - offset;      // 0

      // const int seq_len = seq_len_kv;
      // CausalMask
      //     ? full_tile_offset +
      //           cute::min(seq_len_kv, seq_coord - discard_seq_coord) +
      //           QK_SG_M
      //     : seq_len_kv;

      const int kv_splits_cache = cute::ceil_div(seq_len_kv_cache, QK_BLK_N);
      const int kv_splits = kv_splits_cache;

      int tiles_per_page = params.mainloop.page_size / QK_BLK_N;

      Tensor mQ_mkl = cute::get_xe_tensor(make_shape(seq_len_qo, head_size_qk, 1));  //(m,k,l)
      Tensor mK_nkl = cute::get_xe_tensor(make_shape(seq_len_kv, head_size_qk, 1)); //(n,k,l)

      Tensor mK_cache_nkl = cute::get_xe_tensor(make_shape(seq_len_kv_cache, head_size_qk, 1));  // (n_cache,k,l)
      Tensor mV_cache_nkl = cute::get_xe_tensor(make_shape(head_size_vo, seq_len_kv_cache, 1));  // (n_cache,k,l)

      Tensor mCosQ_mkl = cute::get_xe_tensor(
          make_shape(seq_len_qo, head_size_qk, 1));  // (m, k, l)
      Tensor mSinQ_mkl = cute::get_xe_tensor(
          make_shape(seq_len_qo, head_size_qk, 1));  // (m, k, l)
      Tensor mCosK_nkl = cute::get_xe_tensor(
          make_shape(seq_len_kv, head_size_qk, 1));  // (n, k, l)
      Tensor mSinK_nkl = cute::get_xe_tensor(
          make_shape(seq_len_kv, head_size_qk, 1));  // (n, k, l)

      // block_size and head_size are the same size. So no coord is needed.
      Tensor mQ_mk = mQ_mkl(_, _, 0);
      Tensor mK_nk = mK_nkl(_, _, 0); // (n,k)

      Tensor mK_cache_nk = mK_cache_nkl(_, _, 0);  // (n_cache, k)
      Tensor mV_cache_nk = mV_cache_nkl(_, _, 0);  // (n_cache, k)

      Tensor mCosQ_mk = mCosQ_mkl(_, _, 0);                                                // (m,k)
      Tensor mSinQ_mk = mSinQ_mkl(_, _, 0);                                                // (m,k)
      Tensor mCosK_nk = mCosK_nkl(_, _, 0);                                                // (n,k)
      Tensor mSinK_nk = mSinK_nkl(_, _, 0);

      auto gQ = local_tile(mQ_mk, TileShapeQK{}, make_coord(blk_m_coord, _, _), Step<_1, X, _1>{});
      auto gK = local_tile(mK_nk, TileShapeQK{}, make_coord(_, _, _), Step<X, _1, _1>{});

      auto gK_cache = local_tile(mK_cache_nk, TileShapeQK{}, make_coord(_, _, _), Step<X, _1, _1>{});
      auto gV_cache = local_tile(mV_cache_nk, TileShapeOutput{}, make_coord(_, blk_n_coord, _), Step<X, _1, _1>{});

      auto gCosQ = local_tile(mCosQ_mk, TileShapeQK{}, 
                              make_coord(blk_m_coord, _, _), Step<_1,  X, _1>{});
      auto gSinQ = local_tile(mSinQ_mk, TileShapeQK{}, 
                              make_coord(blk_m_coord, _, _), Step<_1,  X, _1>{});
      auto gCosK = local_tile(mCosK_nk, TileShapeQK{},
                              make_coord(_, _ , _), Step<X, _1, _1>{});
      auto gSinK = local_tile(mSinK_nk, TileShapeQK{}, 
                              make_coord(_, _ , _), Step<X, _1, _1>{});

      auto mainloop_params = CollectiveMainloop::get_updated_copies(
          params.mainloop, params.problem_shape, sequence_length_shape, batch_coord, q_head_coord);

      // currently RoPE is not supported for fp8.
    if constexpr (rope_enabled && !is_fp8_v<ElementQ>) {
      if(cute::thread(THREAD_ID,BLOCK_ID)){
        print("inside rope in kernel\n");
      }
      int block_idx = static_cast<int>(BlockIdxX());
      int block_idy = static_cast<int>(BlockIdxY());
      int block_idz = static_cast<int>(BlockIdxZ());
      int block_dimx = static_cast<int>(BlockDimX());
      int block_dimy = static_cast<int>(BlockDimY());
      int block_dimz = static_cast<int>(BlockDimZ());
      int thread_idx = static_cast<int>(ThreadIdxX());
      int thread_idy = static_cast<int>(ThreadIdxY());
      int thread_idz = static_cast<int>(ThreadIdxZ());
      int grid_dimx = static_cast<int>(GridDimX());
      int grid_dimy = static_cast<int>(GridDimY());
      int grid_dimz = static_cast<int>(GridDimZ());
      int block_id = block_idx + block_idy * grid_dimx + block_idz * grid_dimx * grid_dimy;
      int thread_id = block_id * block_dimx * block_dimy * block_dimz + thread_idz * block_dimx * block_dimy + thread_idy * block_dimx + thread_idx;


      // calculate the base_ptr and offset for Q, K.
      // also calculate the layout for Q, K.
      // then apply RoPE on Q, K accordingly
      auto [batch, num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv, seq_len_kv_cache, head_size_qk, head_size_vo] = params.problem_shape;

      int offset_q = num_heads_q * head_size_qk * seq_len_qo * batch_coord +  // Jump to the correct batch
                 q_head_coord * head_size_qk +  // Jump to the correct head
                 (blk_m_coord*QK_BLK_M*head_size_qk); // Jump to the correct seq_len_qo block

      auto q_group_size = num_heads_q / num_heads_kv;
      auto kv_head_coord = q_head_coord / q_group_size;
      int offset_k = num_heads_kv * head_size_qk * seq_len_kv * batch_coord +
                 kv_head_coord * head_size_qk;

      // calculate Q/cosQ/sinQ ptr
      auto q_traits = static_cast<traits_load_Q const&>(mainloop_params.gmem_tiled_copy_q);
      ElementQ* base_ptr_q = (ElementQ*)q_traits.base_ptr;

      auto q_traits_cos = static_cast<traits_load_Q const&>(mainloop_params.gmem_tiled_copy_q_cos);
      ElementQ* base_ptr_q_cos = (ElementQ*)q_traits_cos.base_ptr;

      auto q_traits_sin = static_cast<traits_load_Q const&>(mainloop_params.gmem_tiled_copy_q_sin);
      ElementQ* base_ptr_q_sin = (ElementQ*)q_traits_sin.base_ptr;

      auto static_shape_q = make_shape(size<0>(gQ), size<1>(gQ)*size<2>(gQ));
      int s = head_size_qk * num_heads_q;
      auto stride_q = make_stride(s, Int<1>{});
      auto layout_q = make_layout(static_shape_q, stride_q);

      // calculate K/cosK/sinK ptr
      auto k_traits = static_cast<traits_load_K const&>(mainloop_params.gmem_tiled_copy_k);
      ElementK* base_ptr_k = (ElementK*)k_traits.base_ptr;

      auto k_traits_cos = static_cast<traits_load_K const&>(mainloop_params.gmem_tiled_copy_k_cos);
      ElementK* base_ptr_k_cos = (ElementK*)k_traits_cos.base_ptr;

      auto k_traits_sin = static_cast<traits_load_K const&>(mainloop_params.gmem_tiled_copy_k_sin);
      ElementK* base_ptr_k_sin = (ElementK*)k_traits_sin.base_ptr;

      auto static_shape_k = make_shape(size<0>(gK), size<1>(gK)*size<3>(gK));
      auto layout_k = make_layout(static_shape_k, LayoutRight{});
      auto gK_dim3 = size<3>(gK);

      // calculating rope for Q
      auto tensorQ = make_tensor(make_gmem_ptr(base_ptr_q+offset_q), layout_q);
      auto tensorCosQ = make_tensor(make_gmem_ptr(base_ptr_q_cos+offset_q), layout_q);
      auto tensorSinQ = make_tensor(make_gmem_ptr(base_ptr_q_sin+offset_q), layout_q);
      cutlass::flash_attention::collective::apply_rope_interleaved_gmem(thread_idx, tensorQ, tensorCosQ, tensorSinQ, tensorQ);

      //calculating rope for K
      // need to consider the case when there are multiple blocks in y direction
      // each block in y direction will handle a different set of K
      // so need to adjust the base pointer of K accordingly.
      if(grid_dimx == 4){
        if (block_id%4==1){
          offset_k += QK_BLK_N*QK_BLK_K*gK_dim3;
        } else if (block_id%4==2){
          offset_k += 2*QK_BLK_N*QK_BLK_K*gK_dim3;
        } else if (block_id%4==3){
          offset_k += 3*QK_BLK_N*QK_BLK_K*gK_dim3;
        }

        auto new_offset_k = offset_k;
        for (int i =0 ;i< size<2>(gK); i+=4){
          auto tensorK = make_tensor(make_gmem_ptr(base_ptr_k+new_offset_k), layout_k);
          auto tensorCosK = make_tensor(make_gmem_ptr(base_ptr_k_cos+new_offset_k), layout_k);
          auto tensorSinK = make_tensor(make_gmem_ptr(base_ptr_k_sin+new_offset_k), layout_k);
          // fix next
          // cutlass::flash_attention::collective::apply_rope_interleaved_gmem(thread_idx, tensorK, tensorCosK, tensorSinK, tensorK); 
          new_offset_k += 4*QK_BLK_N*QK_BLK_K*gK_dim3;
        }
      } else if (grid_dimx ==2){
        if (block_id%2==1){
          offset_k += QK_BLK_N*QK_BLK_K*gK_dim3;
        }
        auto new_offset_k = offset_k;
        for (int i =0 ;i< size<2>(gK); i+=2){
          auto tensorK = make_tensor(make_gmem_ptr(base_ptr_k+new_offset_k), layout_k);
          auto tensorCosK = make_tensor(make_gmem_ptr(base_ptr_k_cos+new_offset_k), layout_k);
          auto tensorSinK = make_tensor(make_gmem_ptr(base_ptr_k_sin+new_offset_k), layout_k);
          // fix next
          // cutlass::flash_attention::collective::apply_rope_interleaved_gmem(thread_idx, tensorK, tensorCosK, tensorSinK, tensorK);
          new_offset_k += 2*QK_BLK_N*QK_BLK_K*gK_dim3;
        }
      }

      if(cute::thread(THREAD_ID,BLOCK_ID)){
        print("after rope\n");
      }
    }

      // we limit the horizontal size to two subgroup, the empirical results
      // show that reading the two cacheline side by side in gives better
      // performance and anything after that does not have an effect on
      // performance. // (64 here for float b float when possible and loop over
      // to cover all the data needed)
      auto tiled_prefetch_q =
          cute::prefetch_selector<Shape<Int<QK_BLK_M>, Int<cute::max(cute::gcd(QK_BLK_K, 64), 32)>>, Num_SGs>(
              mainloop_params.gmem_tiled_copy_q);

      auto tiled_prefetch_k_cache =
          cute::prefetch_selector<Shape<Int<QK_BLK_N>, Int<cute::max(cute::gcd(QK_BLK_K, 64), 32)>>, Num_SGs>(
              mainloop_params.gmem_tiled_copy_k_cache);
      auto tiled_prefetch_v_cache = cute::
          prefetch_selector<Shape<Int<cute::max(cute::gcd(Epilogue_BLK_N, 64), 32)>, Int<Epilogue_BLK_K>>, Num_SGs>(
              mainloop_params.gmem_tiled_copy_v_cache);
      auto thr_prefetch_Q = tiled_prefetch_q.get_slice(thread_idx);
      auto thr_prefetch_K_cache = tiled_prefetch_k_cache.get_slice(thread_idx);
      auto thr_prefetch_V_cache = tiled_prefetch_v_cache.get_slice(thread_idx);
      auto pQgQ = thr_prefetch_Q.partition_S(gQ);

      // assuming the copy function is the same otherwise this need to have its
      // own tile_prefetch
      auto pKgK_cache = thr_prefetch_K_cache.partition_S(gK_cache);
      auto pVgV_cache = thr_prefetch_V_cache.partition_S(gV_cache);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size<3>(pQgQ); i++) {
        prefetch(tiled_prefetch_q, pQgQ(_, _, _, i));
      }
      auto& prefetch_K = tiled_prefetch_k_cache;
      auto& pKgK1_ = pKgK_cache;

      int cached_nblock = 0;
      if constexpr (PagedKV) {
        // int curr_batch_pages = ceil_div(seq_len_kv_cache, mainloop_params.page_size);// max_page_size_per_seq
        // int batch_offset = is_var_len ? mainloop_params.num_pages_per_seq[batch_coord] : batch_coord *
        // curr_batch_pages;
        int batch_offset = batch_coord * mainloop_params.max_num_pages_per_seq;
        cached_nblock = mainloop_params.ptr_page_table[batch_offset  // page table for this batch
        ] * tiles_per_page;                                          // base block idx of physical page
      }
      // The headsize for both cached and non-cached version is the same
      for (int j = 0; j < size<4>(pKgK1_); j++) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = cached_nblock; i < cached_nblock + DispatchPolicy::Stages; i++) {
          prefetch(prefetch_K, pKgK1_(_, _, _, i, j));
        }
      }

      // Allocate the tiled_mma and the accumulators for the (M,N)
      // workgroup_shape
      Tensor out_reg = make_tensor<ElementAccumulator>(AccumeShape{});

      // There are 16 workitem and 16 max per subgroup, each worktime contains 1
      // max and cumulatively, they calculate the max per subgroup
      ElementAccumulator max_reg{-INFINITY};
      // The sum reg each contains a 2d tensor for 8 x 2 This is number of
      // sequence length process per subgroup
      Tensor sum_reg = make_tensor<ElementAccumulator>(Shape<Int<Vec>, Int<FragsM>>{});

      clear(sum_reg);
      clear(out_reg);
      // Perform the collective scoped MMA
      CollectiveMainloop collective_mma;
      // when causal mask is true. It is not possible to set the scope
      // of the barrier to workgroup level as the number n block is
      // different for each subgroup due to triangular nature of causal based
      // operation
      static constexpr int barrier_scope = CausalMask ? 3 : 2;

      int q_start_coord = blk_m_coord * QK_BLK_M;
      int q_end_coord = cute::min(q_start_coord + QK_BLK_M, seq_len_qo);
      int seq_diff = seq_len_kv_cache - seq_len_qo;

      CUTLASS_PRAGMA_UNROLL
      for (int split = 0; split < kv_splits; split++) {
        barrier_arrive(barrier_scope);

        int kv_start_coord = split * QK_BLK_N;

        if constexpr (CausalMask) {
          if (kv_start_coord >= q_end_coord + seq_diff) break;
        }

        //                                                               // = 0, all KV is kv_cache
        // 1) Load KV (performed inside mmaQK)
        auto gK_ = gK_cache(_, _, cached_nblock, _);
        auto gV_ = gV_cache(_, _, cached_nblock);
        // 2) Create Tensor S
        Tensor tSr = make_tensor<ElementAccumulator>(Shape<Int<Vec>, Int<FragsM>, Int<FragsN>>{});
        clear(tSr);
        // 3) Perform GEMM S = Q*K
        // Then modify layout to LayoutQ = ((seq_leq_q, group_head_q),
        // head_size_qk, batch* num_heads_q / group_head_q), which can be merged
        // into one gemm for (int i = 0; i < q_group_size; ++i) {
        collective_mma.mmaQK(tSr, gQ, gK_, tSr, ceil_div(head_size_qk, QK_BLK_K), mainloop_params);

        if constexpr (LocalMask) {
          // Sliding windows
          // mask the elements of each tile where j - left > i || j + right < i
          const int item_id = thread_idx % SubgroupSize;
          int col_idx = item_id + split * cute::min(QK_BLK_N, seq_len_kv_cache);

          CUTLASS_PRAGMA_UNROLL
          for (int n = 0; n < FragsN; n++, col_idx += get<1>(MmaAtomShape())) {  // 4
            CUTLASS_PRAGMA_UNROLL
            for (int m = 0; m < FragsM; m++) {  // 2
              int row_idx = m * Vec + seq_coord;
              int col_ref = seq_len_kv_cache - seq_len_qo;
              // int col_ref = seq_len_kv_cache + seq_len_kv - seq_len_qo;
              CUTLASS_PRAGMA_UNROLL
              for (int row = 0; row < Vec; row++) {  // 8
                bool left_mask = col_idx < cute::max(0, row + row_idx + col_ref - mainloop_params.window_left);
                bool right_mask =
                    col_idx > cute::min(seq_len_kv_cache, row + row_idx + col_ref + mainloop_params.window_right);
                if (left_mask || right_mask) {
                  tSr(row, m, n) = ElementAccumulator{-INFINITY};
                }
              }
            }
          }
        }

        if constexpr (PagedKV) {
          // // if constexpr(!(CausalMask || LocalMask) && PagedKV) {
          // // Processing Not divisible, mask padding
          //   const int item_id = thread_idx % SubgroupSize;
          //   int col_idx = item_id + split * cute::min(QK_BLK_N,
          //   seq_len_kv_cache + seq_len_kv);
          //     CUTLASS_PRAGMA_UNROLL
          //     for (int n = 0; n < FragsN; n++, col_idx +=
          //     get<1>(MmaAtomShape())) { // 4
          //       CUTLASS_PRAGMA_UNROLL
          //       for (int m = 0; m < FragsM; m++) { // 2
          //         int row_idx = m * Vec + seq_coord;
          //         CUTLASS_PRAGMA_UNROLL
          //         for (int row = 0; row < Vec; row++) { // 8
          //           if (col_idx >= seq_len_kv_cache + seq_len_kv || row_idx +
          //           row >= seq_len_qo) {
          //             tSr(row, m, n) = ElementAccumulator{-INFINITY};
          //         }
          //       }
          //     }
          //   }

          int col_start = local_id + kv_start_coord;
          int col_end = col_start + (FragsN - 1) * get<1>(MmaAtomShape());
          if (col_end >= seq_len_kv_cache) {
            int col_idx = col_start;
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < FragsN; n++, col_idx += get<1>(MmaAtomShape())) {  // 4
              if (col_idx >= seq_len_kv_cache) {
                CUTLASS_PRAGMA_UNROLL
                for (int m = 0; m < FragsM; m++) {  // 2
                  CUTLASS_PRAGMA_UNROLL
                  for (int row = 0; row < Vec; row++) {  // 8
                    tSr(row, m, n) = ElementAccumulator{-INFINITY};
                  }
                }
              }
            }
          }
          if constexpr (CausalMask) {
            int row_start = q_start_coord + sub_group_id * QK_SG_M;
            if (row_start + seq_diff < col_end) {
              int col_idx = col_start;
              CUTLASS_PRAGMA_UNROLL
              for (int n = 0; n < FragsN; n++, col_idx += get<1>(MmaAtomShape())) {  // 4
                if (col_idx > row_start + seq_diff) {
                  CUTLASS_PRAGMA_UNROLL
                  for (int m = 0; m < FragsM; m++) {  // 2
                    CUTLASS_PRAGMA_UNROLL
                    for (int row = 0; row < Vec; row++) {  // 8
                      int row_idx = row_start + m * Vec + row;
                      if (row_idx + seq_diff < col_idx) tSr(row, m, n) = ElementAccumulator{-INFINITY};
                    }
                  }
                }
              }
            }
          }
        }
        auto& tiled_prefetch_v_ = tiled_prefetch_v_cache;
        auto& pVgV_ = pVgV_cache;
        int v_prefetch_idx = cached_nblock;
        for (int i = 0; i < size<1>(pVgV_); i++) {
          prefetch(tiled_prefetch_v_, pVgV_(_, i, _, v_prefetch_idx));
        }
        int next_cached_nblock = split + 1;
        if constexpr (PagedKV) {
          // int curr_batch_pages = ceil_div(seq_len_kv_cache, mainloop_params.page_size);
          // int batch_offset =
          //     is_var_len ? mainloop_params.num_pages_per_seq[batch_coord] : batch_coord * curr_batch_pages;
          int curr_batch_pages = mainloop_params.max_num_pages_per_seq;  // max_page_size_per_seq
          int batch_offset = batch_coord * curr_batch_pages;
          int next_page_logical_idx = next_cached_nblock * QK_BLK_N / params.mainloop.page_size;
          bool valid_page = next_page_logical_idx < curr_batch_pages;
          // get physical page idx from page table
          if (valid_page) {
            next_cached_nblock = params.mainloop.ptr_page_table
                                         [batch_offset +               // page table for this batch
                                          next_page_logical_idx        // split (tile idx) to logical
                                                                       // page idx
            ] * tiles_per_page +                                       // base block idx of physical page
                                 next_cached_nblock % tiles_per_page;  // offset within page
          } else {
            next_cached_nblock = curr_batch_pages * tiles_per_page;  // push idx out of bounds to respect the
                                                                     // boundary between batches
          }
        }

        // 4) Fused softmax
        CollectiveSoftmaxEpilogue softmax(params.softmax);
        softmax(split == 0, tSr, max_reg, sum_reg, out_reg);

        // 5) Perform GEMM O = S*V
        collective_mma.template mmaPV<VSlicer>(out_reg, tSr, gV_, out_reg, mainloop_params);
        // ... prefetch next tile ...
        // Prefetch the next Q tile
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<3>(pQgQ); i++) {
          prefetch(tiled_prefetch_q, pQgQ(_, _, _, i));
        }

        cached_nblock = next_cached_nblock;
        // Prefetch the next K tile
        // there is no need to guard it with if statement as prefetch will
        // ignore out of bound reading
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < size<4>(pKgK_cache); j++) {
          prefetch(tiled_prefetch_k_cache, pKgK_cache(_, _, _, cached_nblock, j));
        }
        barrier_wait(barrier_scope);
      }

      // Epilogue
      auto epilogue_params = CollectiveEpilogue::template get_updated_copies<is_var_len>(
          params.epilogue, params.problem_shape, sequence_length_shape, batch_coord, q_head_coord);
      CollectiveEpilogue epilogue{epilogue_params, shared_storage.epilogue};
      auto blk_coord_mnkl = make_coord(blk_m_coord, blk_n_coord, _, 0);
      epilogue(params.problem_shape, sequence_length_shape, blk_coord_mnkl, out_reg, max_reg, sum_reg);
    }
  }
};

///////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::flash_attention::kernel
