/* Copyright 2025 SGLang Team. All Rights Reserved.
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *
 * This file adapts the Inkling BMG HMLP fold_timespace_to_depth helper from
 * /data2/syk/cutlass-sycl/examples/22_bmg_hmlp for the sgl-kernel XPU
 * extension ABI.
 */

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include "Utils.h"

namespace {

constexpr int kBlockSize = 256;
constexpr int kPackBytes = 16;
constexpr int kMediumLaneBytes = 64;
constexpr int kLargeLaneBytes = 128;
constexpr int kEsimdCopyWords = 64;
constexpr int kLargeEsimdCopyWords = 128;
constexpr int64_t kLargeLaneElementsThreshold = 1 << 20;

inline int64_t ceil_div_i64(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

inline int64_t round_up_i64(int64_t x, int64_t multiple) {
  return ceil_div_i64(x, multiple) * multiple;
}

template <typename scalar_t>
struct FoldParams {
  scalar_t const* x = nullptr;
  scalar_t* out = nullptr;
  int B = 0;
  int T = 0;
  int H = 0;
  int W = 0;
  int C = 0;
  int t_fold = 1;
  int hw_fold = 1;
  int t_new = 0;
  int h_new = 0;
  int w_new = 0;
  int fold_count = 1;
  int64_t total_elements = 0;
  int64_t lanes_per_segment = 0;
  int64_t vec_count = 0;
};

template <typename scalar_t>
bool is_contiguous_reinterpret(FoldParams<scalar_t> const& params) {
  if (params.t_fold == 1 && params.hw_fold == 1) {
    return true;
  }
  if (params.h_new == 1 && params.w_new == 1) {
    return true;
  }
  if (params.t_fold == 1 && params.w_new == 1) {
    return true;
  }
  return false;
}

template <typename scalar_t>
int64_t segment_count(FoldParams<scalar_t> const& params) {
  return static_cast<int64_t>(params.B) * params.t_new * params.h_new * params.w_new * params.fold_count;
}

template <typename scalar_t, int LaneElems>
struct FoldTimespaceToDepthKernel {
  FoldParams<scalar_t> params;
  int total_lanes = 0;
  int lanes_per_segment = 0;

  void operator()(sycl::nd_item<1> item) const {
    constexpr int kPackElems = kPackBytes / static_cast<int>(sizeof(scalar_t));
    constexpr int kPackWords = kPackBytes / static_cast<int>(sizeof(uint32_t));
    constexpr int kPacksPerLane = LaneElems / kPackElems;

    const int idx = static_cast<int>(item.get_global_id(0));
    if (idx >= total_lanes) {
      return;
    }

    const int segment = idx / lanes_per_segment;
    const int lane = idx - segment * lanes_per_segment;

    int fold = segment % params.fold_count;
    int outer = segment / params.fold_count;
    const int w_out = outer % params.w_new;
    outer /= params.w_new;
    const int h_out = outer % params.h_new;
    outer /= params.h_new;
    const int t_out = outer % params.t_new;
    const int b = outer / params.t_new;

    const int wf = fold % params.hw_fold;
    fold /= params.hw_fold;
    const int hf = fold % params.hw_fold;
    const int tf = fold / params.hw_fold;

    int c = 0;
    if (lane < params.vec_count) {
      c = lane * LaneElems;
    } else {
      c = static_cast<int>(params.vec_count) * LaneElems + (lane - static_cast<int>(params.vec_count));
    }

    const int64_t src = (((static_cast<int64_t>(b) * params.T + t_out * params.t_fold + tf) * params.H +
                          h_out * params.hw_fold + hf) *
                             params.W +
                         w_out * params.hw_fold + wf) *
                            params.C +
        c;
    const int64_t dst = static_cast<int64_t>(segment) * params.C + c;

    if (lane < params.vec_count) {
      using pack_t = sycl::vec<uint32_t, kPackWords>;
#pragma unroll
      for (int pack = 0; pack < kPacksPerLane; ++pack) {
        pack_t value;
        value.load(0, reinterpret_cast<uint32_t const*>(params.x + src + pack * kPackElems));
        value.store(0, reinterpret_cast<uint32_t*>(params.out + dst + pack * kPackElems));
      }
    } else {
      params.out[dst] = params.x[src];
    }
  }
};

template <typename scalar_t, int LaneElems>
struct FoldTimespaceToDepthRowSliceKernel {
  FoldParams<scalar_t> params;
  int total_lanes = 0;
  int lanes_per_segment = 0;

  void operator()(sycl::nd_item<1> item) const {
    constexpr int kPackElems = kPackBytes / static_cast<int>(sizeof(scalar_t));
    constexpr int kPackWords = kPackBytes / static_cast<int>(sizeof(uint32_t));
    constexpr int kPacksPerLane = LaneElems / kPackElems;

    const int idx = static_cast<int>(item.get_global_id(0));
    if (idx >= total_lanes) {
      return;
    }

    int slice = idx / lanes_per_segment;
    const int lane = idx - slice * lanes_per_segment;

    const int hf = slice % params.hw_fold;
    slice /= params.hw_fold;
    const int tf = slice % params.t_fold;
    int outer = slice / params.t_fold;
    const int w_out = outer % params.w_new;
    outer /= params.w_new;
    const int h_out = outer % params.h_new;
    outer /= params.h_new;
    const int t_out = outer % params.t_new;
    const int b = outer / params.t_new;

    int c = 0;
    if (lane < params.vec_count) {
      c = lane * LaneElems;
    } else {
      c = static_cast<int>(params.vec_count) * LaneElems + (lane - static_cast<int>(params.vec_count));
    }

    const int64_t outer_cell =
        (((static_cast<int64_t>(b) * params.t_new + t_out) * params.h_new + h_out) * params.w_new + w_out);
    const int64_t dst =
        (outer_cell * params.fold_count + (static_cast<int64_t>(tf) * params.hw_fold + hf) * params.hw_fold) *
            params.C +
        c;
    const int64_t src = (((static_cast<int64_t>(b) * params.T + t_out * params.t_fold + tf) * params.H +
                          h_out * params.hw_fold + hf) *
                             params.W +
                         w_out * params.hw_fold) *
                            params.C +
        c;

    if (lane < params.vec_count) {
      using pack_t = sycl::vec<uint32_t, kPackWords>;
#pragma unroll
      for (int pack = 0; pack < kPacksPerLane; ++pack) {
        pack_t value;
        value.load(0, reinterpret_cast<uint32_t const*>(params.x + src + pack * kPackElems));
        value.store(0, reinterpret_cast<uint32_t*>(params.out + dst + pack * kPackElems));
      }
    } else {
      params.out[dst] = params.x[src];
    }
  }
};

template <typename scalar_t, int CopyWords>
struct FoldTimespaceToDepthSegmentEsimdKernel {
  FoldParams<scalar_t> params;
  int chunks_per_segment = 0;

  void operator()(sycl::item<1> item) const SYCL_ESIMD_KERNEL {
    int linear = static_cast<int>(item.get_linear_id());
    int segment = linear / chunks_per_segment;
    int chunk = linear - segment * chunks_per_segment;

    int fold = segment % params.fold_count;
    int outer = segment / params.fold_count;
    int w_out = outer % params.w_new;
    outer /= params.w_new;
    int h_out = outer % params.h_new;
    outer /= params.h_new;
    int t_out = outer % params.t_new;
    int b = outer / params.t_new;

    int wf = fold % params.hw_fold;
    int fold_tmp = fold / params.hw_fold;
    int hf = fold_tmp % params.hw_fold;
    int tf = fold_tmp / params.hw_fold;

    int64_t src = (((static_cast<int64_t>(b) * params.T + t_out * params.t_fold + tf) * params.H +
                    h_out * params.hw_fold + hf) *
                       params.W +
                   w_out * params.hw_fold + wf) *
        params.C;
    int64_t dst = static_cast<int64_t>(segment) * params.C;

    auto value = sycl::ext::intel::esimd::block_load<uint32_t, CopyWords>(
        reinterpret_cast<uint32_t const*>(params.x + src) + chunk * CopyWords);
    sycl::ext::intel::esimd::block_store<uint32_t, CopyWords>(
        reinterpret_cast<uint32_t*>(params.out + dst) + chunk * CopyWords, value);
  }
};

template <typename scalar_t, int CopyWords>
struct FoldTimespaceToDepthRowSliceEsimdKernel {
  FoldParams<scalar_t> params;
  int chunks_per_slice = 0;

  void operator()(sycl::item<1> item) const SYCL_ESIMD_KERNEL {
    int linear = static_cast<int>(item.get_linear_id());
    int slice = linear / chunks_per_slice;
    int chunk = linear - slice * chunks_per_slice;

    int hf = slice % params.hw_fold;
    slice /= params.hw_fold;
    int tf = slice % params.t_fold;
    int outer = slice / params.t_fold;
    int w_out = outer % params.w_new;
    outer /= params.w_new;
    int h_out = outer % params.h_new;
    outer /= params.h_new;
    int t_out = outer % params.t_new;
    int b = outer / params.t_new;

    int64_t outer_cell =
        (((static_cast<int64_t>(b) * params.t_new + t_out) * params.h_new + h_out) * params.w_new + w_out);
    int64_t dst =
        (outer_cell * params.fold_count + (static_cast<int64_t>(tf) * params.hw_fold + hf) * params.hw_fold) *
        params.C;
    int64_t src = (((static_cast<int64_t>(b) * params.T + t_out * params.t_fold + tf) * params.H +
                    h_out * params.hw_fold + hf) *
                       params.W +
                   w_out * params.hw_fold) *
        params.C;

    auto value = sycl::ext::intel::esimd::block_load<uint32_t, CopyWords>(
        reinterpret_cast<uint32_t const*>(params.x + src) + chunk * CopyWords);
    sycl::ext::intel::esimd::block_store<uint32_t, CopyWords>(
        reinterpret_cast<uint32_t*>(params.out + dst) + chunk * CopyWords, value);
  }
};

template <typename scalar_t, int CopyWords>
struct FoldTimespaceToDepthHwf2PairRowsEsimdKernel {
  FoldParams<scalar_t> params;
  int chunks_per_slice = 0;

  void operator()(sycl::item<1> item) const SYCL_ESIMD_KERNEL {
    int linear = static_cast<int>(item.get_linear_id());
    int tf_cell = linear / chunks_per_slice;
    int chunk = linear - tf_cell * chunks_per_slice;

    int tf = tf_cell % params.t_fold;
    int outer = tf_cell / params.t_fold;
    int w_out = outer % params.w_new;
    outer /= params.w_new;
    int h_out = outer % params.h_new;
    outer /= params.h_new;
    int t_out = outer % params.t_new;
    int b = outer / params.t_new;

    int64_t src0 = (((static_cast<int64_t>(b) * params.T + t_out * params.t_fold + tf) * params.H +
                     h_out * 2) *
                        params.W +
                    w_out * 2) *
        params.C;
    int64_t src1 = src0 + static_cast<int64_t>(params.W) * params.C;
    int64_t outer_cell =
        (((static_cast<int64_t>(b) * params.t_new + t_out) * params.h_new + h_out) * params.w_new + w_out);
    int64_t dst0 = (outer_cell * params.fold_count + static_cast<int64_t>(tf) * 4) * params.C;
    int64_t dst1 = dst0 + 2 * params.C;

    auto row0 = sycl::ext::intel::esimd::block_load<uint32_t, CopyWords>(
        reinterpret_cast<uint32_t const*>(params.x + src0) + chunk * CopyWords);
    auto row1 = sycl::ext::intel::esimd::block_load<uint32_t, CopyWords>(
        reinterpret_cast<uint32_t const*>(params.x + src1) + chunk * CopyWords);
    sycl::ext::intel::esimd::block_store<uint32_t, CopyWords>(
        reinterpret_cast<uint32_t*>(params.out + dst0) + chunk * CopyWords, row0);
    sycl::ext::intel::esimd::block_store<uint32_t, CopyWords>(
        reinterpret_cast<uint32_t*>(params.out + dst1) + chunk * CopyWords, row1);
  }
};

template <typename scalar_t>
bool pointer_aligned(scalar_t const* ptr, int bytes) {
  return reinterpret_cast<std::uintptr_t>(ptr) % static_cast<std::uintptr_t>(bytes) == 0;
}

template <typename scalar_t>
bool pointer_aligned(scalar_t* ptr, int bytes) {
  return reinterpret_cast<std::uintptr_t>(ptr) % static_cast<std::uintptr_t>(bytes) == 0;
}

template <int CopyWords, typename scalar_t>
bool can_use_esimd_segment_copy(FoldParams<scalar_t> const& params, int& chunks_per_segment) {
  constexpr int kCopyBytes = CopyWords * static_cast<int>(sizeof(uint32_t));
  int segment_bytes = params.C * static_cast<int>(sizeof(scalar_t));
  if (segment_bytes % kCopyBytes != 0 || !pointer_aligned(params.x, kPackBytes) ||
      !pointer_aligned(params.out, kPackBytes)) {
    return false;
  }
  chunks_per_segment = segment_bytes / kCopyBytes;
  return chunks_per_segment > 0;
}

template <int CopyWords, typename scalar_t>
bool can_use_esimd_row_slice_copy(FoldParams<scalar_t> const& params, int& chunks_per_slice) {
  constexpr int kCopyBytes = CopyWords * static_cast<int>(sizeof(uint32_t));
  int slice_bytes = params.hw_fold * params.C * static_cast<int>(sizeof(scalar_t));
  if (slice_bytes % kCopyBytes != 0 || !pointer_aligned(params.x, kPackBytes) ||
      !pointer_aligned(params.out, kPackBytes)) {
    return false;
  }
  chunks_per_slice = slice_bytes / kCopyBytes;
  return chunks_per_slice > 0;
}

template <int CopyWords, typename scalar_t>
void launch_fold_segment_esimd(sycl::queue& queue, FoldParams<scalar_t> params, int chunks_per_segment) {
  const int64_t total_chunks = segment_count(params) * static_cast<int64_t>(chunks_per_segment);
  TORCH_CHECK(total_chunks <= std::numeric_limits<int>::max(), "fold_timespace_to_depth launch grid is too large");
  FoldTimespaceToDepthSegmentEsimdKernel<scalar_t, CopyWords> kernel{params, chunks_per_segment};
  queue.parallel_for<FoldTimespaceToDepthSegmentEsimdKernel<scalar_t, CopyWords>>(
      sycl::range<1>(static_cast<std::size_t>(total_chunks)), kernel);
}

template <int CopyWords, typename scalar_t>
void launch_fold_row_slice_esimd(sycl::queue& queue, FoldParams<scalar_t> params, int chunks_per_slice) {
  const int64_t slices =
      static_cast<int64_t>(params.B) * params.t_new * params.h_new * params.w_new * params.t_fold * params.hw_fold;
  const int64_t total_chunks = slices * static_cast<int64_t>(chunks_per_slice);
  TORCH_CHECK(total_chunks <= std::numeric_limits<int>::max(), "fold_timespace_to_depth launch grid is too large");
  FoldTimespaceToDepthRowSliceEsimdKernel<scalar_t, CopyWords> kernel{params, chunks_per_slice};
  queue.parallel_for<FoldTimespaceToDepthRowSliceEsimdKernel<scalar_t, CopyWords>>(
      sycl::range<1>(static_cast<std::size_t>(total_chunks)), kernel);
}

template <int CopyWords, typename scalar_t>
void launch_fold_hwf2_pair_rows_esimd(sycl::queue& queue, FoldParams<scalar_t> params, int chunks_per_slice) {
  const int64_t tf_cells =
      static_cast<int64_t>(params.B) * params.t_new * params.h_new * params.w_new * params.t_fold;
  const int64_t total_chunks = tf_cells * static_cast<int64_t>(chunks_per_slice);
  TORCH_CHECK(total_chunks <= std::numeric_limits<int>::max(), "fold_timespace_to_depth launch grid is too large");
  FoldTimespaceToDepthHwf2PairRowsEsimdKernel<scalar_t, CopyWords> kernel{params, chunks_per_slice};
  queue.parallel_for<FoldTimespaceToDepthHwf2PairRowsEsimdKernel<scalar_t, CopyWords>>(
      sycl::range<1>(static_cast<std::size_t>(total_chunks)), kernel);
}

template <typename scalar_t, int LaneElems>
void launch_fold_kernel_static(sycl::queue& queue, FoldParams<scalar_t> params) {
  constexpr int kPackElems = kPackBytes / static_cast<int>(sizeof(scalar_t));
  static_assert(LaneElems % kPackElems == 0, "lane must contain whole 16B packs");

  const bool aligned = (params.C % kPackElems == 0) &&
      (reinterpret_cast<std::uintptr_t>(params.x) % kPackBytes == 0) &&
      (reinterpret_cast<std::uintptr_t>(params.out) % kPackBytes == 0);
  params.vec_count = aligned ? params.C / LaneElems : 0;
  const int64_t scalar_tail = params.C - params.vec_count * LaneElems;
  params.lanes_per_segment = params.vec_count + scalar_tail;

  const int64_t total_lanes = segment_count(params) * params.lanes_per_segment;
  TORCH_CHECK(total_lanes <= std::numeric_limits<int>::max(), "fold_timespace_to_depth launch grid is too large");
  const int64_t global = round_up_i64(total_lanes, kBlockSize);
  FoldTimespaceToDepthKernel<scalar_t, LaneElems> kernel{
      params, static_cast<int>(total_lanes), static_cast<int>(params.lanes_per_segment)};
  queue.parallel_for<FoldTimespaceToDepthKernel<scalar_t, LaneElems>>(
      sycl::nd_range<1>(
          sycl::range<1>(static_cast<std::size_t>(global)),
          sycl::range<1>(static_cast<std::size_t>(kBlockSize))),
      kernel);
}

template <typename scalar_t, int LaneElems>
void launch_fold_row_slice_kernel_static(sycl::queue& queue, FoldParams<scalar_t> params) {
  constexpr int kPackElems = kPackBytes / static_cast<int>(sizeof(scalar_t));
  static_assert(LaneElems % kPackElems == 0, "lane must contain whole 16B packs");

  const int slice_elems = params.hw_fold * params.C;
  const bool aligned = (params.C % kPackElems == 0) &&
      (reinterpret_cast<std::uintptr_t>(params.x) % kPackBytes == 0) &&
      (reinterpret_cast<std::uintptr_t>(params.out) % kPackBytes == 0);
  params.vec_count = aligned ? slice_elems / LaneElems : 0;
  const int64_t scalar_tail = slice_elems - params.vec_count * LaneElems;
  params.lanes_per_segment = params.vec_count + scalar_tail;

  const int64_t slices =
      static_cast<int64_t>(params.B) * params.t_new * params.h_new * params.w_new * params.t_fold * params.hw_fold;
  const int64_t total_lanes = slices * params.lanes_per_segment;
  TORCH_CHECK(total_lanes <= std::numeric_limits<int>::max(), "fold_timespace_to_depth launch grid is too large");
  const int64_t global = round_up_i64(total_lanes, kBlockSize);
  FoldTimespaceToDepthRowSliceKernel<scalar_t, LaneElems> kernel{
      params, static_cast<int>(total_lanes), static_cast<int>(params.lanes_per_segment)};
  queue.parallel_for<FoldTimespaceToDepthRowSliceKernel<scalar_t, LaneElems>>(
      sycl::nd_range<1>(
          sycl::range<1>(static_cast<std::size_t>(global)),
          sycl::range<1>(static_cast<std::size_t>(kBlockSize))),
      kernel);
}

template <typename scalar_t>
void launch_fold_timespace_to_depth(sycl::queue& queue, FoldParams<scalar_t> params) {
  if (params.total_elements == 0) {
    return;
  }

  if (is_contiguous_reinterpret(params)) {
    queue.memcpy(params.out, params.x, static_cast<std::size_t>(params.total_elements * sizeof(scalar_t)));
    return;
  }

  constexpr int kSmallLaneElems = kPackBytes / static_cast<int>(sizeof(scalar_t));
  constexpr int kMediumLaneElems = kMediumLaneBytes / static_cast<int>(sizeof(scalar_t));
  constexpr int kLargeLaneElems = kLargeLaneBytes / static_cast<int>(sizeof(scalar_t));

  if (params.hw_fold > 1) {
    const int slice_elems = params.hw_fold * params.C;
    int chunks_per_slice = 0;
    if constexpr (sizeof(scalar_t) == 2) {
      if (params.hw_fold == 2 && can_use_esimd_row_slice_copy<kLargeEsimdCopyWords>(params, chunks_per_slice)) {
        launch_fold_hwf2_pair_rows_esimd<kLargeEsimdCopyWords>(queue, params, chunks_per_slice);
        return;
      }
      if (params.hw_fold == 2 && can_use_esimd_row_slice_copy<kEsimdCopyWords>(params, chunks_per_slice)) {
        launch_fold_hwf2_pair_rows_esimd<kEsimdCopyWords>(queue, params, chunks_per_slice);
        return;
      }
    }
    if (can_use_esimd_row_slice_copy<kLargeEsimdCopyWords>(params, chunks_per_slice)) {
      launch_fold_row_slice_esimd<kLargeEsimdCopyWords>(queue, params, chunks_per_slice);
      return;
    }
    if (can_use_esimd_row_slice_copy<kEsimdCopyWords>(params, chunks_per_slice)) {
      launch_fold_row_slice_esimd<kEsimdCopyWords>(queue, params, chunks_per_slice);
      return;
    }
    if (params.total_elements >= kLargeLaneElementsThreshold &&
        slice_elems >= kLargeLaneElems &&
        slice_elems % kLargeLaneElems == 0 &&
        params.C % (kPackBytes / static_cast<int>(sizeof(scalar_t))) == 0) {
      launch_fold_row_slice_kernel_static<scalar_t, kLargeLaneElems>(queue, params);
      return;
    }
    if (params.total_elements >= kLargeLaneElementsThreshold && slice_elems >= kMediumLaneElems) {
      launch_fold_row_slice_kernel_static<scalar_t, kMediumLaneElems>(queue, params);
      return;
    }
    launch_fold_row_slice_kernel_static<scalar_t, kSmallLaneElems>(queue, params);
    return;
  }

  int chunks_per_segment = 0;
  if (can_use_esimd_segment_copy<kLargeEsimdCopyWords>(params, chunks_per_segment)) {
    launch_fold_segment_esimd<kLargeEsimdCopyWords>(queue, params, chunks_per_segment);
    return;
  }
  if (can_use_esimd_segment_copy<kEsimdCopyWords>(params, chunks_per_segment)) {
    launch_fold_segment_esimd<kEsimdCopyWords>(queue, params, chunks_per_segment);
    return;
  }
  if (params.total_elements >= kLargeLaneElementsThreshold &&
      params.C >= kLargeLaneElems &&
      params.C % kLargeLaneElems == 0) {
    launch_fold_kernel_static<scalar_t, kLargeLaneElems>(queue, params);
    return;
  }
  if (params.total_elements >= kLargeLaneElementsThreshold && params.C >= kMediumLaneElems) {
    launch_fold_kernel_static<scalar_t, kMediumLaneElems>(queue, params);
    return;
  }
  launch_fold_kernel_static<scalar_t, kSmallLaneElems>(queue, params);
}

int checked_int64_to_int(int64_t value, char const* name) {
  TORCH_CHECK(value > 0 && value <= std::numeric_limits<int>::max(), name, " must be positive and fit in int32");
  return static_cast<int>(value);
}

}  // namespace

at::Tensor inkling_hmlp_fold_timespace_to_depth(
    const at::Tensor& x,
    int64_t t_fold,
    int64_t hw_fold) {
  CHECK_INPUT(x);
  TORCH_CHECK(
      x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half ||
          x.scalar_type() == at::ScalarType::BFloat16,
      "x must be float32, float16, or bfloat16");
  TORCH_CHECK(x.dim() == 5, "x must be 5D [B, T, H, W, C]");
  const int B = checked_int64_to_int(x.size(0), "B");
  const int T = checked_int64_to_int(x.size(1), "T");
  const int H = checked_int64_to_int(x.size(2), "H");
  const int W = checked_int64_to_int(x.size(3), "W");
  const int C = checked_int64_to_int(x.size(4), "C");
  const int tf = checked_int64_to_int(t_fold, "t_fold");
  const int hwf = checked_int64_to_int(hw_fold, "hw_fold");

  TORCH_CHECK(T % tf == 0, "T must be divisible by t_fold");
  TORCH_CHECK(H % hwf == 0, "H must be divisible by hw_fold");
  TORCH_CHECK(W % hwf == 0, "W must be divisible by hw_fold");
  TORCH_CHECK(
      static_cast<int64_t>(tf) * hwf * hwf <= std::numeric_limits<int>::max(),
      "fold_count must fit in int32");

  const int t_new = T / tf;
  const int h_new = H / hwf;
  const int w_new = W / hwf;
  const int fold_count = tf * hwf * hwf;
  const int64_t out_c = static_cast<int64_t>(fold_count) * C;
  at::Tensor out = at::empty({B, t_new, h_new, w_new, out_c}, x.options());

  sycl::queue& queue = dpcppGetCurrentQueue();
  DISPATCH_FLOAT_TYPES(x.scalar_type(), "inkling_hmlp_fold_timespace_to_depth", [&] {
    FoldParams<scalar_t> params;
    params.x = x.data_ptr<scalar_t>();
    params.out = out.data_ptr<scalar_t>();
    params.B = B;
    params.T = T;
    params.H = H;
    params.W = W;
    params.C = C;
    params.t_fold = tf;
    params.hw_fold = hwf;
    params.t_new = t_new;
    params.h_new = h_new;
    params.w_new = w_new;
    params.fold_count = fold_count;
    params.total_elements = x.numel();
    launch_fold_timespace_to_depth(queue, params);
  });
  return out;
}
