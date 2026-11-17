// Adapted from vllm-xpu-kernels csrc/dsa_topk (5bc6a53a).
#pragma once

// TopKRegister: register-resident top-k. Scores stay in registers across both
// passes (read once).

#include <cstdint>
#include <sycl/sycl.hpp>

#include "topk_config.hpp"
#include "topk_radix_base.hpp"
#include "topk_warp.hpp"

namespace sgl_kernel {
namespace dsa_topk {

inline uint32_t div_ceil_u(uint32_t a, uint32_t b) {
  return (a + b - 1) / b;
}

template <uint32_t kLocalVecs_>
struct TopKRegister : TopKRadixBase<12> {
  static constexpr uint32_t kLocalVecs = kLocalVecs_;
  static constexpr uint32_t kMaxSeqLen = kBlockSize * kVecSize * kLocalVecs;

  // Input may be float or bf16, which widens on load. `in` overrides problem.in.
  template <typename InT>
  static void forward_in(const sycl::nd_item<1>& item, const TopKProblem& problem, const InT* in, void* _smem) {
    auto group = item.get_group();
    const uint32_t tx = item.get_local_linear_id();
    Smem* smem = static_cast<Smem*>(_smem);

    init_histogram(item, smem->histogram);
    if (tx == 0) {
      smem->count_eq = 0;
      smem->count_gt = 0;
    }
    sycl::group_barrier(group);

    const uint32_t num_full = div_ceil_u(problem.seq_len, kVecSize);
    const uint32_t tail_start = (problem.seq_len - 1) % kVecSize + 1;

    // Phase 1: load full vectors + build histogram. The final vector may be
    // partial (seq_len not a multiple of kVecSize); load only its valid lanes
    // so the 16 B read never runs past the row, padding the rest.
    vec_t local_vecs[kLocalVecs];
#pragma unroll
    for (uint32_t i = 0; i < kLocalVecs; ++i) {
      const uint32_t vi = tx + kBlockSize * i;
      if (vi >= num_full) break;
      if (vi == num_full - 1)
        local_vecs[i].load_n(in, vi, tail_start, padding_value());
      else
        local_vecs[i].load(in, vi);
    }

#pragma unroll
    for (uint32_t i = 0; i < kLocalVecs; ++i) {
      const uint32_t vi = tx + kBlockSize * i;
      if (vi >= num_full) break;
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j)
        slm_atomic_add(smem->histogram[extract_coarse_bin<kHistBits>(local_vecs[i][j])], 1);
    }
    const uint32_t num_padding = kVecSize - tail_start + problem.input_start;
    if (tx == 0 && num_padding > 0) {
      slm_atomic_sub(smem->histogram[extract_coarse_bin<kHistBits>(padding_value())], num_padding);
      slm_atomic_add(smem->histogram[0], num_padding);
    }
    sycl::group_barrier(group);

    // Phase 2: find the threshold bin.
    find_threshold(item, problem.topk, num_full * kVecSize, smem, [&](uint32_t threshold_bin) {
      const auto b = coarse_bin_bounds<kHistBits>(threshold_bin);
      smem->v_lo = b[0];
      smem->v_hi = b[1];
      if constexpr (kExact) smem->thr_count = smem->histogram[threshold_bin];
    });

    // Phase 3: collect by two fp32 boundaries.
    const uint32_t topk = problem.topk;
    const float v_hi = smem->v_hi;
    const float v_lo = smem->v_lo;

#pragma unroll
    for (uint32_t i = 0; i < kLocalVecs; ++i) {
      const uint32_t vi = tx + kBlockSize * i;
      const uint32_t base = vi * kVecSize;
      if (vi >= num_full) break;
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        const uint32_t idx = base + j;
        const float val = local_vecs[i][j];
        if (val >= v_hi) {
          const uint32_t pos = slm_atomic_add(smem->count_gt, 1);
          if (pos < topk) problem.emit(pos, idx);
        } else if (val >= v_lo) {
          const uint32_t pos = slm_atomic_add(smem->count_eq, 1);
          if (pos < kMaxNumTie) smem->tie_values[pos] = {val, idx};
        }
      }
    }

    // Phase 4: handle ties.
    sycl::group_barrier(group);
    handle_staged_ties(item, problem, smem);
  }

  // fp32 entry: reads problem.in.
  static void forward(const sycl::nd_item<1>& item, const TopKProblem& problem, void* _smem) {
    forward_in<float>(item, problem, problem.in, _smem);
  }
};

}  // namespace dsa_topk
}  // namespace sgl_kernel
