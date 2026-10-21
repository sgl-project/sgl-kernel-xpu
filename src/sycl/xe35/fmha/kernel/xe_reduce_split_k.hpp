#pragma once

#include "cute/util/type_traits.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "sycl/xe35/fmha/kernel/xe_tile_scheduler.hpp"

namespace cutlass::reduction_xe35::kernel {

using namespace cute;

template <class ProblemShape_, class TileScheduler_, class FMHAKernel_>
class ReduceSplitK_xe35 {
 public:
  using ProblemShape = ProblemShape_;
  using TileScheduler = TileScheduler_;
  using VariableLength_xe35 = cutlass::fmha_xe35::collective::VariableLength_xe35;
  static constexpr bool is_var_len = cutlass::fmha_xe35::collective::is_variable_length_v<typename ProblemShape::SeqLenType>;
  static_assert(is_same_v<TileScheduler, cutlass::fmha_xe35::kernel::XeReduceSplitKTileScheduler_xe35>);

  using ElementO = typename FMHAKernel_::ElementO;
  using ElementLSE = float;
  using StrideO = typename FMHAKernel_::StrideO;
  using TileShapeQK = typename FMHAKernel_::TileShapeQK;
  using SGPerWG = typename FMHAKernel_::SGPerWG;
  static constexpr int SharedStorageSize = 0;

  struct KernelArguments {
    ProblemShape shape;
    ElementO* O;
    StrideO dO;
    const ElementO* Oaccum;
    StrideO dOaccum;
    const ElementLSE* exp_sums;
    StrideO dExp_sums;
    const ElementLSE* max_logits;
    StrideO dMax_logits;
    int window_size_left = -1;
    float scale_v = 1.0f;
  };
  using KernelParams = KernelArguments;

  struct Arguments {
    KernelArguments kernel{};
    KernelHardwareInfo hw_info{};
    int num_kv_splits = -1;
  };

  struct Params {
    KernelParams kernel;
    typename TileScheduler::Params scheduler;
  };

  static Params to_underlying_arguments(Arguments const& args, void*) {
    return {
        args.kernel,
        TileScheduler::to_underlying_arguments(
            args.kernel.shape, args.hw_info, typename FMHAKernel_::TileShapeO{}, args.num_kv_splits)};
  }

  static bool can_implement(Arguments const& args) {
    return args.num_kv_splits >= 1 && args.num_kv_splits <= FMHAKernel_::max_num_kv_splits;
  }

  static int get_workspace_size(Arguments const&) {
    return 0;
  }

  static cutlass::Status
  initialize_workspace(Arguments const&, void* = nullptr, cudaStream_t = nullptr, CudaHostAdapter* = nullptr) {
    return Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::template get_grid_shape<SGPerWG::value>(params.scheduler);
  }
  static dim3 get_block_shape() {
    return dim3(SGPerWG::value * intel::sg_size, 1, 1);
  }

  CUTLASS_DEVICE Shape<int, int> get_sequence_length_shape(ProblemShape const& shape, int batch) {
    if constexpr (is_var_len) {
      auto q_len = cutlass::fmha_xe35::collective::apply_variable_length_xe35(Shape<VariableLength_xe35>{shape.seq_len_qo}, batch);
      if (shape.seq_len_kv_cache.cumulative_length != nullptr) {
        return {get<0>(q_len), shape.seq_len_kv_cache.cumulative_length[batch]};
      } else if (shape.seq_len_kv.cumulative_length != nullptr) {
        int kv_len = shape.seq_len_kv.cumulative_length[batch + 1] - shape.seq_len_kv.cumulative_length[batch];
        return {get<0>(q_len), kv_len};
      } else {
        return {get<0>(q_len), 0};
      }
    } else {
      int kv_len = shape.seq_len_kv_cache > 0 ? shape.seq_len_kv_cache : shape.seq_len_kv;
      return {shape.seq_len_qo, kv_len};
    }
  }

  CUTLASS_DEVICE void operator()(Params const& params, char*) {
    auto& p = params.kernel;
    auto splits = params.scheduler.num_kv_splits;
    auto scheduler = TileScheduler{params.scheduler};
    const int batch_dim = is_var_len ? 1 : p.shape.batch;

    for (; scheduler.is_valid(); ++scheduler) {
      auto [seq_idx, head_q, idx_b] = scheduler.get_block_coord();
      auto [seq_len_q, seq_len_kv] = get_sequence_length_shape(p.shape, idx_b);
      if (seq_idx >= seq_len_q) continue;

      const int head_group_q = p.shape.num_heads_q / p.shape.num_heads_kv;
      const int head_kv = head_q / head_group_q;
      const int q_in_group = head_q % head_group_q;

      int offset_q = 0;
      int offset_o = 0;
      int offset_accum = 0;
      int offset_lse = 0;
      if constexpr (is_var_len) {
        auto q_offsets = p.shape.seq_len_qo.cumulative_length;
        offset_o = p.shape.num_heads_q * p.shape.head_size_vo * q_offsets[idx_b];
        offset_accum = p.shape.num_heads_q * p.shape.head_size_vo * splits * q_offsets[idx_b];
        offset_lse = p.shape.num_heads_q * splits * q_offsets[idx_b] + head_q * splits;
      } else {
        offset_lse = idx_b * p.shape.num_heads_q * splits + head_q * splits;
      }

      auto shape_o = make_shape(seq_len_q, p.shape.head_size_vo, p.shape.num_heads_q, batch_dim);
      auto shape_accum = make_shape(head_group_q, p.shape.head_size_vo, p.shape.num_heads_kv, splits, batch_dim);
      auto layout_o = make_ordered_layout(shape_o, Step<_1, _0, _2, _3>{});
      auto layout_accum = make_ordered_layout(shape_accum, Step<_1, _0, _2, _3, _4>{});

      Tensor O = make_tensor(make_gmem_ptr(p.O + offset_o), layout_o);
      Tensor Oaccum = make_tensor(make_gmem_ptr(const_cast<ElementO*>(p.Oaccum) + offset_accum), layout_accum);
      Tensor exp_sums = make_tensor(
          make_gmem_ptr(const_cast<ElementLSE*>(p.exp_sums) + offset_lse),
          make_layout(make_shape(splits), make_stride(_1{})));
      Tensor max_logits = make_tensor(
          make_gmem_ptr(const_cast<ElementLSE*>(p.max_logits) + offset_lse),
          make_layout(make_shape(splits), make_stride(_1{})));
      int l_coord = is_var_len ? 0 : idx_b;

      ElementLSE local_max_logits[FMHAKernel_::max_num_kv_splits];
      ElementLSE local_rescale[FMHAKernel_::max_num_kv_splits];
      ElementLSE global_max = cutlass::platform::numeric_limits<ElementLSE>::lowest();
      ElementLSE global_sum = 0;
      int active_splits = 0;
      if (seq_len_kv > 0) {
        // The producer partitions only the visible SWA tiles. A split can be
        // active even when its index exceeds a split count estimated from the
        // full KV length. Every producer slot writes a nonnegative exp sum.
        for (int split = 0; split < splits; ++split) {
          local_rescale[split] = 0;
          ElementLSE sum = exp_sums(split);
          if (sum <= ElementLSE(0)) continue;
          local_max_logits[split] = max_logits(split);
          local_rescale[split] = sum;
          global_max = sycl::max(global_max, local_max_logits[split]);
          active_splits = split + 1;
        }
      }
      for (int split = 0; split < active_splits; ++split) {
        if (local_rescale[split] <= ElementLSE(0)) continue;
        ElementLSE s = sycl::native::exp2(local_max_logits[split] - global_max);
        global_sum += local_rescale[split] * s;
        local_rescale[split] = s;
      }
      ElementLSE inv_global_sum = global_sum > 0 ? (ElementLSE(p.scale_v) / global_sum) : ElementLSE(0);
      for (int split = 0; split < active_splits; ++split) {
        local_rescale[split] *= inv_global_sum;
      }

      for (int value = int(ThreadIdxX()); value < p.shape.head_size_vo; value += SGPerWG::value * intel::sg_size) {
        ElementLSE value_sum = 0;
        for (int split = 0; split < active_splits; ++split) {
          if (local_rescale[split] == ElementLSE(0)) continue;
          value_sum +=
              static_cast<ElementLSE>(Oaccum(q_in_group, value, head_kv, split, l_coord)) * local_rescale[split];
        }
        O(seq_idx, value, head_q, l_coord) = static_cast<ElementO>(value_sum);
      }
    }
  }
};

}  // namespace cutlass::reduction_xe35::kernel
