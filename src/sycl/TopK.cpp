#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdint>
#include <optional>
#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"

namespace {
constexpr int kTopK = 2048;
constexpr int kThreadsPerBlock = 1024;
constexpr int kRadix = 256;
constexpr int kHistPad = 128;  // padding so suffix-sum can read [tx + j] safely
constexpr int kHistStride = kRadix + kHistPad;
constexpr int kSmemInputBytes = 32 * 1024;  // 32 KB for ping-pong index buffer
constexpr int kSmemInputSize = kSmemInputBytes / (2 * sizeof(int32_t));

// Scalars layout in local memory
//   [0] s_counter           -> running number of outputs already emitted
//   [1] s_threshold_bin_id  -> threshold bin selected for the current round
//   [2] s_num_input[0]      -> stash count for bank 0
//   [3] s_num_input[1]      -> stash count for bank 1
//   [4] s_last_remain       -> remaining slots in the final round
constexpr int kNumScalars = 5;

inline uint8_t convert_to_uint8(float x) {
  sycl::half h = static_cast<sycl::half>(x);
  uint16_t bits = sycl::bit_cast<uint16_t>(h);
  uint16_t key = (bits & 0x8000u) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000u);
  return static_cast<uint8_t>(key >> 8);
}

inline uint32_t convert_to_uint32(float x) {
  uint32_t bits = sycl::bit_cast<uint32_t>(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

using LocalAtomic = sycl::atomic_ref<
    int32_t,
    sycl::memory_order::relaxed,
    sycl::memory_scope::work_group,
    sycl::access::address_space::local_space>;

inline int32_t lm_atomic_add(int32_t& v, int32_t delta) {
  return LocalAtomic(v).fetch_add(delta);
}

inline int wg_reserve(sycl::nd_item<1>& item, int32_t& counter, int want) {
  auto g = item.get_group();
  const int my_off = sycl::exclusive_scan_over_group(g, want, sycl::plus<int>());
  const int wg_total = sycl::reduce_over_group(g, want, sycl::plus<int>());
  int wg_base = 0;
  if (item.get_local_id(0) == 0 && wg_total > 0) {
    wg_base = LocalAtomic(counter).fetch_add(wg_total);
  }
  wg_base = sycl::group_broadcast(g, wg_base, 0);
  return wg_base + my_off;
}

inline int wg_reserve_dec(sycl::nd_item<1>& item, int32_t& counter, int want) {
  auto g = item.get_group();
  const int my_off = sycl::exclusive_scan_over_group(g, want, sycl::plus<int>());
  const int wg_total = sycl::reduce_over_group(g, want, sycl::plus<int>());
  int wg_base = 0;
  if (item.get_local_id(0) == 0 && wg_total > 0) {
    wg_base = LocalAtomic(counter).fetch_add(-wg_total);
  }
  wg_base = sycl::group_broadcast(g, wg_base, 0);
  return wg_base - my_off;
}

inline void run_cumsum(sycl::nd_item<1>& item, int32_t* s_histogram_buf /* [2][kHistStride] */) {
  const int tx = static_cast<int>(item.get_local_id(0));
#pragma unroll 8
  for (int i = 0; i < 8; ++i) {
    const int j = 1 << i;
    const int k = i & 1;
    int32_t value = 0;
    if (tx < kRadix) {
      value = s_histogram_buf[k * kHistStride + tx];
      if (tx < kRadix - j) {
        value += s_histogram_buf[k * kHistStride + tx + j];
      }
    }
    item.barrier(sycl::access::fence_space::local_space);
    if (tx < kRadix) {
      s_histogram_buf[(k ^ 1) * kHistStride + tx] = value;
    }
    item.barrier(sycl::access::fence_space::local_space);
  }
}

inline void fast_topk_radix(
    sycl::nd_item<1>& item,
    const float* input,
    int32_t* index,
    int row_start,
    int length,
    int32_t* s_histogram_buf,
    int32_t* s_scalars,
    int32_t* s_input_idx) {
  int topk = kTopK;
  const int tx = static_cast<int>(item.get_local_id(0));

  if (tx < kRadix + 1) {
    s_histogram_buf[tx] = 0;
    s_histogram_buf[kHistStride + tx] = 0;
  }
  if (tx < kNumScalars) {
    s_scalars[tx] = 0;
  }
  item.barrier(sycl::access::fence_space::local_space);

  for (int idx = tx; idx < length; idx += kThreadsPerBlock) {
    const int bin = static_cast<int>(convert_to_uint8(input[idx + row_start]));
    LocalAtomic(s_histogram_buf[bin]).fetch_add(1);
  }
  item.barrier(sycl::access::fence_space::local_space);

  run_cumsum(item, s_histogram_buf);

  if (tx < kRadix && s_histogram_buf[tx] > topk && s_histogram_buf[tx + 1] <= topk) {
    s_scalars[1] = tx;
    s_scalars[2] = 0;
    s_scalars[0] = 0;
  }
  item.barrier(sycl::access::fence_space::local_space);

  int threshold_bin = s_scalars[1];
  topk -= s_histogram_buf[threshold_bin + 1];

  const int n_iters_full = (length + kThreadsPerBlock - 1) / kThreadsPerBlock;

  if (topk == 0) {
    for (int it = 0; it < n_iters_full; ++it) {
      const int idx = it * kThreadsPerBlock + tx;
      const bool valid = (idx < length);
      int bin = -1;
      if (valid) {
        bin = static_cast<int>(convert_to_uint8(input[idx + row_start]));
      }
      const int want = (valid && bin > threshold_bin) ? 1 : 0;
      const int pos = wg_reserve(item, s_scalars[0], want);
      if (want) {
        index[pos] = idx;
      }
    }
    item.barrier(sycl::access::fence_space::local_space);
    return;
  }

  if (tx < kRadix + 1) {
    s_histogram_buf[tx] = 0;
  }
  item.barrier(sycl::access::fence_space::local_space);

  for (int it = 0; it < n_iters_full; ++it) {
    const int idx = it * kThreadsPerBlock + tx;
    const bool valid = (idx < length);
    float raw = 0.0f;
    int bin = -1;
    if (valid) {
      raw = input[idx + row_start];
      bin = static_cast<int>(convert_to_uint8(raw));
    }
    const int want_out = (valid && bin > threshold_bin) ? 1 : 0;
    const int want_stash = (valid && bin == threshold_bin) ? 1 : 0;

    const int out_pos = wg_reserve(item, s_scalars[0], want_out);
    if (want_out) {
      index[out_pos] = idx;
    }

    const int stash_pos = wg_reserve(item, s_scalars[2], want_stash);
    if (want_stash && stash_pos < kSmemInputSize) {
      s_input_idx[0 * kSmemInputSize + stash_pos] = idx;
      const uint32_t b32 = convert_to_uint32(raw);
      const int sub_bin = static_cast<int>((b32 >> 24) & 0xFFu);
      LocalAtomic(s_histogram_buf[sub_bin]).fetch_add(1);
    }
  }
  item.barrier(sycl::access::fence_space::local_space);

#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    const int r_idx = round & 1;

    const int raw_num = s_scalars[2 + r_idx];
    const int num_input = (raw_num < kSmemInputSize) ? raw_num : kSmemInputSize;

    run_cumsum(item, s_histogram_buf);

    if (tx < kRadix && s_histogram_buf[tx] > topk && s_histogram_buf[tx + 1] <= topk) {
      s_scalars[1] = tx;
      s_scalars[2 + (r_idx ^ 1)] = 0;
      s_scalars[4] = topk - s_histogram_buf[tx + 1];
    }
    item.barrier(sycl::access::fence_space::local_space);

    threshold_bin = s_scalars[1];
    topk -= s_histogram_buf[threshold_bin + 1];
    const int offset = 24 - round * 8;

    const int n_iters_stash = (num_input + kThreadsPerBlock - 1) / kThreadsPerBlock;

    if (topk == 0) {
      for (int it = 0; it < n_iters_stash; ++it) {
        const int i = it * kThreadsPerBlock + tx;
        const bool valid = (i < num_input);
        int idx = 0;
        int bin = -1;
        if (valid) {
          idx = s_input_idx[r_idx * kSmemInputSize + i];
          bin = static_cast<int>((convert_to_uint32(input[idx + row_start]) >> offset) & 0xFFu);
        }
        const int want = (valid && bin > threshold_bin) ? 1 : 0;
        const int pos = wg_reserve(item, s_scalars[0], want);
        if (want) {
          index[pos] = idx;
        }
      }
      item.barrier(sycl::access::fence_space::local_space);
      break;
    }

    if (tx < kRadix + 1) {
      s_histogram_buf[tx] = 0;
    }
    item.barrier(sycl::access::fence_space::local_space);

    for (int it = 0; it < n_iters_stash; ++it) {
      const int i = it * kThreadsPerBlock + tx;
      const bool valid = (i < num_input);
      int idx = 0;
      float raw = 0.0f;
      int bin = -1;
      if (valid) {
        idx = s_input_idx[r_idx * kSmemInputSize + i];
        raw = input[idx + row_start];
        bin = static_cast<int>((convert_to_uint32(raw) >> offset) & 0xFFu);
      }
      const int want_out = (valid && bin > threshold_bin) ? 1 : 0;
      const int out_pos = wg_reserve(item, s_scalars[0], want_out);
      if (want_out) {
        index[out_pos] = idx;
      }

      if (round == 3) {
        const int want_last = (valid && bin == threshold_bin) ? 1 : 0;
        const int last_pos = wg_reserve_dec(item, s_scalars[4], want_last);
        if (want_last && last_pos > 0) {
          index[kTopK - last_pos] = idx;
        }
      } else {
        const int want_stash = (valid && bin == threshold_bin) ? 1 : 0;
        const int stash_pos = wg_reserve(item, s_scalars[2 + (r_idx ^ 1)], want_stash);
        if (want_stash && stash_pos < kSmemInputSize) {
          s_input_idx[(r_idx ^ 1) * kSmemInputSize + stash_pos] = idx;
          const uint32_t b32 = convert_to_uint32(raw);
          const int sub_bin = static_cast<int>((b32 >> (offset - 8)) & 0xFFu);
          LocalAtomic(s_histogram_buf[sub_bin]).fetch_add(1);
        }
      }
    }
    item.barrier(sycl::access::fence_space::local_space);
  }
}

inline void naive_topk(sycl::nd_item<1>& item, int32_t* indices, int length) {
  const int tx = static_cast<int>(item.get_local_id(0));
  for (int i = tx; i < kTopK; i += kThreadsPerBlock) {
    indices[i] = (i < length) ? i : -1;
  }
}

inline void
naive_topk_transform(sycl::nd_item<1>& item, int length, int32_t* dst_page_entry, const int32_t* src_page_entry) {
  const int tx = static_cast<int>(item.get_local_id(0));
  for (int i = tx; i < kTopK; i += kThreadsPerBlock) {
    dst_page_entry[i] = (i < length) ? src_page_entry[i] : -1;
  }
}

inline void
naive_topk_transform_ragged(sycl::nd_item<1>& item, int length, int32_t* dst_indices_entry, int32_t offset) {
  const int tx = static_cast<int>(item.get_local_id(0));
  for (int i = tx; i < kTopK; i += kThreadsPerBlock) {
    dst_indices_entry[i] = (i < length) ? (i + offset) : -1;
  }
}

struct FastTopKParams {
  const float* input;
  const int32_t* row_starts;
  int32_t* indices;
  const int32_t* lengths;
  int64_t input_stride;
};

FastTopKParams get_params(
    const at::Tensor& score,
    const at::Tensor& lengths,
    std::optional<at::Tensor> row_starts_opt = std::nullopt,
    std::optional<at::Tensor> indices_opt = std::nullopt) {
  const auto B = score.size(0);
  TORCH_CHECK(score.dim() == 2 && score.stride(1) == 1, "score must be 2D with contiguous last dim");
  TORCH_CHECK(score.scalar_type() == at::kFloat, "score must be float32");
  if (row_starts_opt.has_value()) {
    const auto& row_starts = row_starts_opt.value();
    TORCH_CHECK(row_starts.dim() == 1 && row_starts.is_contiguous() && row_starts.size(0) == B);
    TORCH_CHECK(row_starts.scalar_type() == at::kInt, "row_starts must be int32");
  }
  TORCH_CHECK(lengths.dim() == 1 && lengths.is_contiguous() && lengths.size(0) == B);
  TORCH_CHECK(lengths.scalar_type() == at::kInt, "lengths must be int32");

  int32_t* indices_ptr = nullptr;
  if (indices_opt.has_value()) {
    const auto& indices = indices_opt.value();
    TORCH_CHECK(indices.dim() == 2 && indices.is_contiguous());
    TORCH_CHECK(indices.size(0) == B && indices.size(1) == kTopK);
    TORCH_CHECK(indices.scalar_type() == at::kInt, "indices must be int32");
    indices_ptr = indices.data_ptr<int32_t>();
  }

  FastTopKParams params{};
  params.input = score.data_ptr<float>();
  params.row_starts = row_starts_opt.has_value() ? row_starts_opt->data_ptr<int32_t>() : nullptr;
  params.indices = indices_ptr;
  params.lengths = lengths.data_ptr<int32_t>();
  params.input_stride = score.stride(0);
  return params;
}

struct FastTopKKernel : public __SYCL_KER_CONFIG_CONVENTION__ {
  FastTopKParams params;

  sycl::local_accessor<int32_t, 1> s_histogram_;
  sycl::local_accessor<int32_t, 1> s_scalars_;
  sycl::local_accessor<int32_t, 1> s_input_idx_;

  explicit FastTopKKernel(const FastTopKParams& p) : params(p) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {
    s_histogram_ = sycl::local_accessor<int32_t, 1>(2 * kHistStride, cgh);
    s_scalars_ = sycl::local_accessor<int32_t, 1>(kNumScalars, cgh);
    s_input_idx_ = sycl::local_accessor<int32_t, 1>(2 * kSmemInputSize, cgh);
  }

  [[sycl::reqd_sub_group_size(32)]] void operator()(sycl::nd_item<1> item) const {
    const int64_t bid = static_cast<int64_t>(item.get_group(0));
    const int row_start = params.row_starts ? params.row_starts[bid] : 0;
    const int length = params.lengths[bid];
    int32_t* indice = params.indices + bid * kTopK;
    const float* score = params.input + bid * params.input_stride;

    int32_t* hist = s_histogram_.get_multi_ptr<sycl::access::decorated::no>().get();
    int32_t* scalars = s_scalars_.get_multi_ptr<sycl::access::decorated::no>().get();
    int32_t* in_idx = s_input_idx_.get_multi_ptr<sycl::access::decorated::no>().get();

    if (length <= kTopK) {
      naive_topk(item, indice, length);
    } else {
      fast_topk_radix(item, score, indice, row_start, length, hist, scalars, in_idx);
    }
  }
};

struct FastTopKTransformFusedDecodeKernel : public __SYCL_KER_CONFIG_CONVENTION__ {
  FastTopKParams params;
  int32_t* dst_page_table;
  const int32_t* src_page_table;
  int64_t src_stride;

  sycl::local_accessor<int32_t, 1> s_histogram_;
  sycl::local_accessor<int32_t, 1> s_scalars_;
  sycl::local_accessor<int32_t, 1> s_input_idx_;
  sycl::local_accessor<int32_t, 1> s_indices_;

  FastTopKTransformFusedDecodeKernel(const FastTopKParams& p, int32_t* dst, const int32_t* src, int64_t stride)
      : params(p), dst_page_table(dst), src_page_table(src), src_stride(stride) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {
    s_histogram_ = sycl::local_accessor<int32_t, 1>(2 * kHistStride, cgh);
    s_scalars_ = sycl::local_accessor<int32_t, 1>(kNumScalars, cgh);
    s_input_idx_ = sycl::local_accessor<int32_t, 1>(2 * kSmemInputSize, cgh);
    s_indices_ = sycl::local_accessor<int32_t, 1>(kTopK, cgh);
  }

  [[sycl::reqd_sub_group_size(32)]] void operator()(sycl::nd_item<1> item) const {
    const int64_t bid = static_cast<int64_t>(item.get_group(0));
    const int tid = static_cast<int>(item.get_local_id(0));
    const int length = params.lengths[bid];
    const int32_t* src_entry = src_page_table + bid * src_stride;
    int32_t* dst_entry = dst_page_table + bid * kTopK;
    const float* score = params.input + bid * params.input_stride;

    if (length <= kTopK) {
      naive_topk_transform(item, length, dst_entry, src_entry);
      return;
    }

    int32_t* hist = s_histogram_.get_multi_ptr<sycl::access::decorated::no>().get();
    int32_t* scalars = s_scalars_.get_multi_ptr<sycl::access::decorated::no>().get();
    int32_t* in_idx = s_input_idx_.get_multi_ptr<sycl::access::decorated::no>().get();
    int32_t* s_idx = s_indices_.get_multi_ptr<sycl::access::decorated::no>().get();

    fast_topk_radix(item, score, s_idx, /*row_start=*/0, length, hist, scalars, in_idx);

    static_assert(kTopK == 2 * kThreadsPerBlock, "kTopK must be 2 * kThreadsPerBlock");
    const int i0 = tid;
    const int i1 = tid + kThreadsPerBlock;
    const int p0 = s_idx[i0];
    const int p1 = s_idx[i1];
    dst_entry[i0] = src_entry[p0];
    dst_entry[i1] = src_entry[p1];
  }
};

struct FastTopKTransformFusedPrefillKernel : public __SYCL_KER_CONFIG_CONVENTION__ {
  FastTopKParams params;
  int32_t* dst_page_table;
  const int32_t* src_page_table;
  int64_t src_stride;
  const int32_t* cu_seqlens_q;
  int64_t prefill_bs;

  sycl::local_accessor<int32_t, 1> s_histogram_;
  sycl::local_accessor<int32_t, 1> s_scalars_;
  sycl::local_accessor<int32_t, 1> s_input_idx_;
  sycl::local_accessor<int32_t, 1> s_indices_;
  sycl::local_accessor<int32_t, 1> s_src_row_;  // [1] selected src row id

  FastTopKTransformFusedPrefillKernel(
      const FastTopKParams& p, int32_t* dst, const int32_t* src, int64_t stride, const int32_t* cu_q, int64_t pbs)
      : params(p), dst_page_table(dst), src_page_table(src), src_stride(stride), cu_seqlens_q(cu_q), prefill_bs(pbs) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {
    s_histogram_ = sycl::local_accessor<int32_t, 1>(2 * kHistStride, cgh);
    s_scalars_ = sycl::local_accessor<int32_t, 1>(kNumScalars, cgh);
    s_input_idx_ = sycl::local_accessor<int32_t, 1>(2 * kSmemInputSize, cgh);
    s_indices_ = sycl::local_accessor<int32_t, 1>(kTopK, cgh);
    s_src_row_ = sycl::local_accessor<int32_t, 1>(1, cgh);
  }

  [[sycl::reqd_sub_group_size(32)]] void operator()(sycl::nd_item<1> item) const {
    const int64_t bid = static_cast<int64_t>(item.get_group(0));
    const int tid = static_cast<int>(item.get_local_id(0));
    const int length = params.lengths[bid];
    const int row_start = params.row_starts ? params.row_starts[bid] : 0;
    int32_t* dst_entry = dst_page_table + bid * kTopK;
    const float* score = params.input + bid * params.input_stride;

    int32_t* s_src_row = s_src_row_.get_multi_ptr<sycl::access::decorated::no>().get();

    if (tid == 0) {
      s_src_row[0] = 0;
    }
    item.barrier(sycl::access::fence_space::local_space);

    if (prefill_bs <= kThreadsPerBlock) {
      if (tid < prefill_bs) {
        if (bid >= cu_seqlens_q[tid] && bid < cu_seqlens_q[tid + 1]) {
          s_src_row[0] = tid;
        }
      }
    } else {
      for (int64_t i = tid; i < prefill_bs; i += kThreadsPerBlock) {
        if (bid >= cu_seqlens_q[i] && bid < cu_seqlens_q[i + 1]) {
          s_src_row[0] = static_cast<int32_t>(i);
        }
      }
    }
    item.barrier(sycl::access::fence_space::local_space);

    const int32_t* src_entry = src_page_table + s_src_row[0] * src_stride;

    if (length <= kTopK) {
      naive_topk_transform(item, length, dst_entry, src_entry);
      return;
    }

    int32_t* hist = s_histogram_.get_multi_ptr<sycl::access::decorated::no>().get();
    int32_t* scalars = s_scalars_.get_multi_ptr<sycl::access::decorated::no>().get();
    int32_t* in_idx = s_input_idx_.get_multi_ptr<sycl::access::decorated::no>().get();
    int32_t* s_idx = s_indices_.get_multi_ptr<sycl::access::decorated::no>().get();

    fast_topk_radix(item, score, s_idx, row_start, length, hist, scalars, in_idx);

    static_assert(kTopK == 2 * kThreadsPerBlock, "kTopK must be 2 * kThreadsPerBlock");
    const int i0 = tid;
    const int i1 = tid + kThreadsPerBlock;
    const int p0 = s_idx[i0];
    const int p1 = s_idx[i1];
    dst_entry[i0] = src_entry[p0];
    dst_entry[i1] = src_entry[p1];
  }
};

struct FastTopKTransformRaggedFusedKernel : public __SYCL_KER_CONFIG_CONVENTION__ {
  FastTopKParams params;
  int32_t* topk_indices_ragged;
  const int32_t* topk_indices_offset;

  sycl::local_accessor<int32_t, 1> s_histogram_;
  sycl::local_accessor<int32_t, 1> s_scalars_;
  sycl::local_accessor<int32_t, 1> s_input_idx_;
  sycl::local_accessor<int32_t, 1> s_indices_;

  FastTopKTransformRaggedFusedKernel(const FastTopKParams& p, int32_t* out, const int32_t* off)
      : params(p), topk_indices_ragged(out), topk_indices_offset(off) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {
    s_histogram_ = sycl::local_accessor<int32_t, 1>(2 * kHistStride, cgh);
    s_scalars_ = sycl::local_accessor<int32_t, 1>(kNumScalars, cgh);
    s_input_idx_ = sycl::local_accessor<int32_t, 1>(2 * kSmemInputSize, cgh);
    s_indices_ = sycl::local_accessor<int32_t, 1>(kTopK, cgh);
  }

  [[sycl::reqd_sub_group_size(32)]] void operator()(sycl::nd_item<1> item) const {
    const int64_t bid = static_cast<int64_t>(item.get_group(0));
    const int tid = static_cast<int>(item.get_local_id(0));
    const int row_start = params.row_starts ? params.row_starts[bid] : 0;
    const int length = params.lengths[bid];
    int32_t* dst_entry = topk_indices_ragged + bid * kTopK;
    const float* score = params.input + bid * params.input_stride;
    const int32_t offset = topk_indices_offset[bid];

    if (length <= kTopK) {
      naive_topk_transform_ragged(item, length, dst_entry, offset);
      return;
    }

    int32_t* hist = s_histogram_.get_multi_ptr<sycl::access::decorated::no>().get();
    int32_t* scalars = s_scalars_.get_multi_ptr<sycl::access::decorated::no>().get();
    int32_t* in_idx = s_input_idx_.get_multi_ptr<sycl::access::decorated::no>().get();
    int32_t* s_idx = s_indices_.get_multi_ptr<sycl::access::decorated::no>().get();

    fast_topk_radix(item, score, s_idx, row_start, length, hist, scalars, in_idx);

    static_assert(kTopK == 2 * kThreadsPerBlock, "kTopK must be 2 * kThreadsPerBlock");
    const int i0 = tid;
    const int i1 = tid + kThreadsPerBlock;
    dst_entry[i0] = s_idx[i0] + offset;
    dst_entry[i1] = s_idx[i1] + offset;
  }
};

}  // namespace

void fast_topk_interface(
    const at::Tensor& score, at::Tensor& indices, const at::Tensor& lengths, std::optional<at::Tensor> row_starts_opt) {
  CHECK_INPUT(score);
  CHECK_DEVICE(indices);
  CHECK_DEVICE(lengths);
  if (row_starts_opt.has_value()) {
    CHECK_DEVICE(row_starts_opt.value());
  }

  const auto params = get_params(score, lengths, row_starts_opt, indices);
  const int64_t B = score.size(0);

  auto stream = at::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();

  FastTopKKernel kernel(params);
  sycl_kernel_submit(
      sycl::range<1>(static_cast<size_t>(B) * kThreadsPerBlock), sycl::range<1>(kThreadsPerBlock), queue, kernel);
}

void fast_topk_transform_interface(
    const at::Tensor& score,
    const at::Tensor& lengths,
    at::Tensor& dst_page_table,
    const at::Tensor& src_page_table,
    const at::Tensor& cu_seqlens_q,
    std::optional<at::Tensor> row_starts_opt) {
  CHECK_INPUT(score);
  CHECK_DEVICE(lengths);
  CHECK_DEVICE(dst_page_table);
  CHECK_DEVICE(src_page_table);
  CHECK_DEVICE(cu_seqlens_q);
  if (row_starts_opt.has_value()) {
    CHECK_DEVICE(row_starts_opt.value());
  }

  const auto params = get_params(score, lengths, row_starts_opt);
  const int64_t B = score.size(0);

  TORCH_CHECK(dst_page_table.dim() == 2 && dst_page_table.is_contiguous());
  TORCH_CHECK(src_page_table.dim() == 2 && src_page_table.stride(1) == 1);
  TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_q.is_contiguous());
  TORCH_CHECK(dst_page_table.scalar_type() == at::kInt);
  TORCH_CHECK(src_page_table.scalar_type() == at::kInt);
  TORCH_CHECK(cu_seqlens_q.scalar_type() == at::kInt);

  const int64_t prefill_bs = cu_seqlens_q.size(0) - 1;
  TORCH_CHECK(dst_page_table.size(0) == B);
  TORCH_CHECK(dst_page_table.size(1) == kTopK);
  TORCH_CHECK(src_page_table.size(0) == prefill_bs);
  TORCH_CHECK(prefill_bs <= B);

  auto stream = at::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();
  const int64_t src_stride = src_page_table.stride(0);

  const bool is_decode = !row_starts_opt.has_value() && prefill_bs == B;
  if (is_decode) {
    FastTopKTransformFusedDecodeKernel kernel(
        params, dst_page_table.data_ptr<int32_t>(), src_page_table.data_ptr<int32_t>(), src_stride);
    sycl_kernel_submit(
        sycl::range<1>(static_cast<size_t>(B) * kThreadsPerBlock), sycl::range<1>(kThreadsPerBlock), queue, kernel);
  } else {
    FastTopKTransformFusedPrefillKernel kernel(
        params,
        dst_page_table.data_ptr<int32_t>(),
        src_page_table.data_ptr<int32_t>(),
        src_stride,
        cu_seqlens_q.data_ptr<int32_t>(),
        prefill_bs);
    sycl_kernel_submit(
        sycl::range<1>(static_cast<size_t>(B) * kThreadsPerBlock), sycl::range<1>(kThreadsPerBlock), queue, kernel);
  }
}

void fast_topk_transform_ragged_interface(
    const at::Tensor& score,
    const at::Tensor& lengths,
    at::Tensor& topk_indices_ragged,
    const at::Tensor& topk_indices_offset,
    std::optional<at::Tensor> row_starts_opt) {
  CHECK_INPUT(score);
  CHECK_DEVICE(lengths);
  CHECK_DEVICE(topk_indices_ragged);
  CHECK_DEVICE(topk_indices_offset);
  if (row_starts_opt.has_value()) {
    CHECK_DEVICE(row_starts_opt.value());
  }

  const auto params = get_params(score, lengths, row_starts_opt);
  const int64_t B = score.size(0);
  TORCH_CHECK(topk_indices_ragged.dim() == 2 && topk_indices_ragged.is_contiguous());
  TORCH_CHECK(topk_indices_offset.dim() == 1 && topk_indices_offset.is_contiguous());
  TORCH_CHECK(topk_indices_ragged.size(0) == B);
  TORCH_CHECK(topk_indices_ragged.size(1) == kTopK);
  TORCH_CHECK(topk_indices_offset.size(0) == B);
  TORCH_CHECK(topk_indices_ragged.scalar_type() == at::kInt);
  TORCH_CHECK(topk_indices_offset.scalar_type() == at::kInt);

  auto stream = at::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();

  FastTopKTransformRaggedFusedKernel kernel(
      params, topk_indices_ragged.data_ptr<int32_t>(), topk_indices_offset.data_ptr<int32_t>());
  sycl_kernel_submit(
      sycl::range<1>(static_cast<size_t>(B) * kThreadsPerBlock), sycl::range<1>(kThreadsPerBlock), queue, kernel);
}
