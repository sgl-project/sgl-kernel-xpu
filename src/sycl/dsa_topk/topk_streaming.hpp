// Adapted from vllm-xpu-kernels csrc/dsa_topk (5bc6a53a).
#pragma once

// TopKStreaming: rows too long for the register path -- two vectorized passes
// over global memory.

#include <cstdint>
#include <sycl/sycl.hpp>

#include "topk_config.hpp"
#include "topk_radix_base.hpp"
#include "topk_warp.hpp"

namespace sgl_kernel {
namespace dsa_topk {

struct TopKStreaming : TopKRadixBase<12> {
  static void forward(const sycl::nd_item<1>& item, TopKProblem problem, void* _smem) {
    auto group = item.get_group();
    const uint32_t tx = item.get_local_linear_id();
    Smem* smem = static_cast<Smem*>(_smem);

    init_histogram(item, smem->histogram);
    if (tx == 0) {
      smem->count_eq = 0;
      smem->count_gt = 0;
    }
    sycl::group_barrier(group);

    // Phase 1: load and build histogram.
    for_each_input(item, problem.in, problem.seq_len, [&](float val, uint32_t) {
      slm_atomic_add(smem->histogram[extract_coarse_bin<kHistBits>(val)], 1);
    });
    const uint32_t num_padding = problem.input_start;
    if (tx == 0 && num_padding != 0) {
      slm_atomic_sub(smem->histogram[extract_coarse_bin<kHistBits>(padding_value())], num_padding);
      slm_atomic_add(smem->histogram[0], num_padding);
    }
    sycl::group_barrier(group);

    // Phase 2: find the threshold bin.
    find_threshold(item, problem.topk, problem.seq_len, smem, [&](uint32_t threshold_bin) {
      const auto b = coarse_bin_bounds<kHistBits>(threshold_bin);
      smem->v_lo = b[0];
      smem->v_hi = b[1];
      if constexpr (kExact) smem->thr_count = smem->histogram[threshold_bin];
    });

    // Phase 3: collect by the bin's two fp32 edges.
    const float v_hi = smem->v_hi;
    const float v_lo = smem->v_lo;
    const uint32_t topk = problem.topk;

    for_each_input(item, problem.in, problem.seq_len, [&](float val, uint32_t idx) {
      if (val >= v_hi) {
        const uint32_t pos = slm_atomic_add(smem->count_gt, 1);
        if (pos < topk) problem.emit(pos, idx);
      } else if (val >= v_lo) {
        const uint32_t pos = slm_atomic_add(smem->count_eq, 1);
        if (pos < kMaxNumTie) smem->tie_values[pos] = {val, idx};
      }
    });

    // Phase 4: handle ties.
    sycl::group_barrier(group);
    handle_staged_ties(item, problem, smem);
  }
};

}  // namespace dsa_topk
}  // namespace sgl_kernel
