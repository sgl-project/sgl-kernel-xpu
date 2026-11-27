// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "launch.hpp"

namespace sgl_kernel::minicpm5::xe20::MINICPM5_MONO_NAMESPACE {

template <int TP, int Rows>
void launch_rows(sycl::queue& queue, Weights weights, Workspace work, Step step, bool dense) {
  if (dense) {
    launch<TP, Rows, true>(queue, weights, work, step);
    return;
  }
#if defined(MINICPM5_MONO_FP16) && !defined(MINICPM5_MONO_B60)
  if constexpr (Rows == 8) {
    launch<TP, Rows, false, true, false, true>(queue, weights, work, step);
    return;
  }
#endif
#if !defined(MINICPM5_MONO_FP16) && !defined(MINICPM5_MONO_B60)
  if constexpr (Rows == MaxRows) {
    // A long single-request verify block has a common 64-token page table.
    if (step.block_table_stride == 0 && step.block_size == 64 &&
        step.block_table_cols * 64 * Rows >= DynamicAttentionTokens) {
#if !defined(MINICPM5_MONO_SPLIT)
      if constexpr (TP == 1 || TP == 2) {
        launch<TP, Rows, false, false, false, false, true, false, LayerInput::VerifyAttentionFront>(
            queue, weights, work, step);
        launch<TP, Rows, false, false, false, false, true, false, LayerInput::VerifyAttentionTail>(
            queue, weights, work, step);
        return;
      }
#endif
      launch<TP, Rows, false, false, false, false, true>(queue, weights, work, step);
      return;
    }
  }
#endif
#if !defined(MINICPM5_MONO_FP16) && !defined(MINICPM5_MONO_B60) && !defined(MINICPM5_MONO_SPLIT)
  if constexpr (TP == 2 && (Rows == 4 || Rows == 8)) {
    if (step.block_table_stride != 0 && step.block_size == 64) {
      launch<TP, Rows, false, false, false, false, false, false, LayerInput::OrdinaryAttentionFront>(
          queue, weights, work, step);
      launch<TP, Rows, false, false, false, false, false, false, LayerInput::OrdinaryAttentionTail>(
          queue, weights, work, step);
      return;
    }
  }
#endif
  launch<TP, Rows, false>(queue, weights, work, step);
}

template <int TP>
void dispatch_rows(sycl::queue& queue, Weights weights, Workspace work, Step step, int rows, bool dense) {
  switch (rows) {
    case 1:
      return launch_rows<TP, 1>(queue, weights, work, step, dense);
    case 2:
      return launch_rows<TP, 2>(queue, weights, work, step, dense);
    case 3:
      return launch_rows<TP, 3>(queue, weights, work, step, dense);
    case 4:
      return launch_rows<TP, 4>(queue, weights, work, step, dense);
    case 5:
      return launch_rows<TP, 5>(queue, weights, work, step, dense);
    case 6:
      return launch_rows<TP, 6>(queue, weights, work, step, dense);
    case 7:
      return launch_rows<TP, 7>(queue, weights, work, step, dense);
    case 8:
      return launch_rows<TP, 8>(queue, weights, work, step, dense);
    default:
      throw std::invalid_argument("MiniCPM5 mono width must be between 1 and 8");
  }
}

#pragma GCC visibility push(default)
void dispatch(sycl::queue& queue, Weights weights, Workspace work, Step step, int rows, int tp, bool dense) {
  if (tp == 1) {
    dispatch_rows<1>(queue, weights, work, step, rows, dense);
  } else {
    dispatch_rows<2>(queue, weights, work, step, rows, dense);
  }
}
#pragma GCC visibility pop

}  // namespace sgl_kernel::minicpm5::xe20::MINICPM5_MONO_NAMESPACE
