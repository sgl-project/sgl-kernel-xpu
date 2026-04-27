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
/*! \file
    \brief MLA Split-KV Reduction Kernel
*/
#pragma once

#include <cute/tensor.hpp>
#include <limits>
#include <sycl/sycl.hpp>

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "sycl/kernels/mla/kernel/mla_tile_scheduler.hpp"

namespace cutlass::flash_attention::kernel {

using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////
struct DummyReductionKernel {
  struct Arguments {};
  struct Params {};
  static bool can_implement(Arguments const&) {
    return true;
  }
  static int get_workspace_size(Arguments const&) {
    return 0;
  }
  static Params to_underlying_arguments(Arguments const&, void*) {
    return {};
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
template <class ProblemShape_, class TileScheduler_, class MlaKernel_>
class XeMlaReduceSplitKV {
 public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;
  using TileScheduler = TileScheduler_;
  using TileSchedulerParams = typename TileScheduler::Params;

  // Derive types from the main MLA split-KV kernel
  using ElementO = typename MlaKernel_::ElementO;
  using StrideO = typename MlaKernel_::StrideO;
  using TileShapeO = typename MlaKernel_::TileShapeO;
  using TileShapeQK = typename MlaKernel_::TileShapeQK;
  using SGPerWG = typename MlaKernel_::SGPerWG;

  using ElementLSE = float;  // exp_sums/max_logits accumulator type

  // Number of output values processed by each thread
  static constexpr int num_vals_per_thread = int(get<1>(TileShapeO{}) / (SGPerWG::value * intel::sg_size));

  //
  // KernelArguments
  //
  struct KernelArguments {
    ProblemShape shape;
    // Final output
    ElementO* O = nullptr;
    StrideO dO{};
    // Partial outputs from split kernel
    const ElementO* O_accum = nullptr;
    StrideO dO_accum{};
    // Softmax statistics per split
    const ElementLSE* exp_sums = nullptr;
    const ElementLSE* max_logits = nullptr;
    StrideO dLSE{};
  };
  //
  // KernelParams
  //
  using KernelParams = KernelArguments;

  //
  // Arguments
  //
  struct Arguments {
    KernelArguments kernel{};
    KernelHardwareInfo hw_info{};
    int num_kv_splits = -1;
  };

  //
  // Params
  //
  struct Params {
    KernelParams kernel;
    TileSchedulerParams scheduler;
  };

  struct SharedStorage {
    cutlass::Array<ElementLSE, MlaKernel_::kvMaxSplits> max_logits_slm;
    cutlass::Array<ElementLSE, MlaKernel_::kvMaxSplits> exp_sums_slm;
  };
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

 public:
  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    return {
        args.kernel,
        TileScheduler::to_underlying_arguments(args.kernel.shape, args.hw_info, TileShapeO{}, args.num_kv_splits)};
  }

  static bool can_implement(Arguments const& args) {
    if (args.kernel.shape.batch <= 0) return false;
    if (args.num_kv_splits <= 1) return false;
    if (args.num_kv_splits > MlaKernel_::kvMaxSplits) return false;
    return true;
  }

  static int get_workspace_size(Arguments const& /*args*/) {
    return 0;
  }

  static cutlass::Status initialize_workspace(Arguments const& /*args*/, void* /*workspace*/ = nullptr) {
    return Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::template get_grid_shape<SGPerWG::value>(params.scheduler);
  }

  static dim3 get_block_shape() {
    return dim3(SGPerWG::value * intel::sg_size, 1, 1);
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    using namespace sycl::ext::oneapi::this_work_item;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    auto& p = params.kernel;
    ProblemShape const& s = p.shape;

    int thr_id = int(ThreadIdxX());
    constexpr int WG_SIZE = SGPerWG::value * intel::sg_size;

    TileScheduler tile_scheduler{params.scheduler};
    int num_kv_splits = params.scheduler.num_kv_splits;

    auto seq_len_qo = s.seq_len_qo;
    auto num_heads_q = s.num_heads_q;
    auto head_size_o = s.head_size_o;
    auto batch = s.batch;
    auto shape_O = make_shape(seq_len_qo, head_size_o, num_heads_q, batch);
    auto shape_Oaccum = make_shape(seq_len_qo, head_size_o, num_heads_q * num_kv_splits, batch);
    auto shape_exp_sums = make_shape(seq_len_qo, num_kv_splits, num_heads_q, batch);
    auto shape_max_logits = make_shape(seq_len_qo, num_kv_splits, num_heads_q, batch);

    Tensor O = make_tensor(make_gmem_ptr(p.O), make_layout(shape_O, p.dO));
    Tensor Oaccum = make_tensor(make_gmem_ptr(const_cast<ElementO*>(p.O_accum)), make_layout(shape_Oaccum, p.dO_accum));
    Tensor exp_sums =
        make_tensor(make_gmem_ptr(const_cast<ElementLSE*>(p.exp_sums)), make_layout(shape_exp_sums, p.dLSE));
    Tensor max_logits =
        make_tensor(make_gmem_ptr(const_cast<ElementLSE*>(p.max_logits)), make_layout(shape_max_logits, p.dLSE));

    CUTLASS_PRAGMA_NO_UNROLL
    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      auto [seq_idx, head_q, idx_b] = tile_scheduler.get_block_coord();

      // Step 1: Load per-split statistics into SLM.
      // Strided loop handles num_kv_splits > WG_SIZE.
      // Each thread also tracks the max of its assigned splits for group reduction.
      ElementLSE global_max{cutlass::platform::numeric_limits<ElementLSE>::lowest()};

      for (int s_idx = thr_id; s_idx < num_kv_splits; s_idx += WG_SIZE) {
        ElementLSE cur_max_logit = max_logits(seq_idx, s_idx, head_q, idx_b);
        global_max = sycl::max(global_max, cur_max_logit);
        shared_storage.max_logits_slm[s_idx] = cur_max_logit;

        shared_storage.exp_sums_slm[s_idx] = exp_sums(seq_idx, s_idx, head_q, idx_b);
      }

      // Barrier: ensure SLM writes are visible to all threads
      sycl::group_barrier(get_work_group<3>());

      // Step 2: Find global max across all splits via HW group reduction
      global_max = reduce_over_group(get_work_group<1>(), global_max, sycl::maximum<>());
      global_max = sycl::group_broadcast(get_work_group<1>(), global_max, 0);

      // Step 3: Cooperatively reduce output elements
      // O_accum is unnormalized (numerator only),
      // so acc += O_accum * rescale, and global_exp_sums += exp_sum * rescale.
      for (int j = thr_id; j < head_size_o; j += WG_SIZE) {
        ElementLSE acc = ElementLSE(0);
        ElementLSE global_exp_sums = ElementLSE(0);
        for (int k = 0; k < num_kv_splits; k++) {
          ElementLSE local_exp_sum = shared_storage.exp_sums_slm[k];
          // Skip empty splits (exp_sums=0, max_logits=-inf sentinel)
          if (local_exp_sum <= ElementLSE(0)) continue;

          ElementLSE local_max = shared_storage.max_logits_slm[k];
          ElementLSE rescale = sycl::native::exp2(local_max - global_max);

          ElementLSE o_val = static_cast<ElementLSE>(Oaccum(seq_idx, j, head_q * num_kv_splits + k, idx_b));

          // O_accum is unnormalized (not divided by exp_sum in epilogue),
          // so multiply by rescale only
          acc += o_val * rescale;
          global_exp_sums += local_exp_sum * rescale;
        }

        ElementLSE inv_global_exp_sums = ElementLSE(1) / global_exp_sums;
        acc *= inv_global_exp_sums;

        O(seq_idx, j, head_q, idx_b) = static_cast<ElementO>(acc);
      }
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::flash_attention::kernel
