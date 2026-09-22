/* Copyright 2026 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <atomic>
#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"
#include "sgl_kernel_export.h"

namespace {

constexpr int32_t kNoNode = -1;

constexpr int64_t kLaneTarget = 128;

constexpr int32_t kSubGroupWidth = 16;

inline int64_t xe_core_count(const sycl::device& dev) {
  static std::atomic<int64_t> cached{0};
  int64_t cores = cached.load(std::memory_order_relaxed);
  if (cores == 0) {
    cores = std::max<int64_t>(
        static_cast<int64_t>(dev.get_info<sycl::ext::intel::info::device::gpu_slices>()) *
            dev.get_info<sycl::ext::intel::info::device::gpu_subslices_per_slice>(),
        1);
    cached.store(cores, std::memory_order_relaxed);
  }
  return cores;
}

static_assert(sizeof(bool) == 1, "tree_mask packing assumes 1-byte bool");

inline uint32_t compress_mask_nibble(uint32_t bytes) {
  return ((bytes * 0x00204081u) >> 21) & 0xFu;
}

inline uint64_t compress_mask_byte8(uint64_t bytes) {
  return (bytes * 0x0102040810204080ull) >> 56;
}

inline uint64_t splat_byte(int32_t v) {
  return static_cast<uint64_t>(static_cast<uint8_t>(v)) * 0x0101010101010101ull;
}

inline uint64_t byte_eq_mask(uint64_t v, uint64_t splat) {
  constexpr uint64_t kLow7 = 0x7f7f7f7f7f7f7f7full;
  const uint64_t x = v ^ splat;
  const uint64_t zero_bytes = ~((((x & kLow7) + kLow7) | x) | kLow7);
  return compress_mask_byte8(zero_bytes >> 7);
}

inline uint64_t pack_mask_bytes(const bool* src, int32_t count) {
  uint64_t bits = 0;
  int32_t c = 0;
#pragma unroll 2
  for (; c + 4 <= count; c += 4) {
    const uint32_t chunk = static_cast<uint32_t>(static_cast<unsigned char>(src[c])) |
                           (static_cast<uint32_t>(static_cast<unsigned char>(src[c + 1])) << 8) |
                           (static_cast<uint32_t>(static_cast<unsigned char>(src[c + 2])) << 16) |
                           (static_cast<uint32_t>(static_cast<unsigned char>(src[c + 3])) << 24);
    bits |= static_cast<uint64_t>(compress_mask_nibble(chunk)) << c;
  }
  for (; c < count; ++c) {
    if (src[c]) {
      bits |= uint64_t{1} << c;
    }
  }
  return bits;
}

inline uint64_t pack_mask_row(const bool* src, int32_t count, bool wide) {
  if (!wide) {
    return pack_mask_bytes(src, count);
  }
  const uint64_t* words = reinterpret_cast<const uint64_t*>(src);
  uint64_t bits = 0;
#pragma unroll 2
  for (int32_t c = 0; c < count; c += 8) {
    bits |= compress_mask_byte8(words[c >> 3]) << c;
  }
  return bits;
}

template <typename seq_t>
struct ReconstructTreeKernel : public __SYCL_KER_CONFIG_CONVENTION__ {
  using word_t = uint64_t;
  static constexpr int32_t kWordBits = 64;

  ReconstructTreeKernel(
      const bool* tree_mask,
      const seq_t* verified_seq_len,
      int64_t* positions,
      int64_t* retrive_index,
      int64_t* retrive_next_token,
      int64_t* retrive_next_sibling,
      int32_t batch_size,
      int32_t num_nodes,
      int32_t words_per_row,
      int32_t lanes_per_request,
      int32_t requests_per_group)
      : tree_mask_(tree_mask),
        verified_seq_len_(verified_seq_len),
        positions_(positions),
        retrive_index_(retrive_index),
        retrive_next_token_(retrive_next_token),
        retrive_next_sibling_(retrive_next_sibling),
        batch_size_(batch_size),
        num_nodes_(num_nodes),
        words_per_row_(words_per_row),
        lanes_per_request_(lanes_per_request),
        requests_per_group_(requests_per_group) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {
    const size_t slots = static_cast<size_t>(requests_per_group_);
    rows_ = sycl::local_accessor<word_t, 1>(sycl::range<1>(slots * num_nodes_ * words_per_row_), cgh);
    parent_ = sycl::local_accessor<int32_t, 1>(sycl::range<1>(slots * num_nodes_), cgh);
  }

  inline word_t strict_word(const word_t* row, int32_t i, int32_t w) const {
    const int32_t lo = w * kWordBits;
    if (i <= lo) {
      return 0;
    }
    const int32_t keep = i - lo;
    const word_t m = keep >= kWordBits ? ~word_t{0} : static_cast<word_t>((word_t{1} << keep) - 1);
    return row[w] & m;
  }

  inline int32_t immediate_parent(const word_t* row, int32_t i) const {
#pragma unroll 2
    for (int32_t w = words_per_row_ - 1; w >= 0; --w) {
      const word_t bits = strict_word(row, i, w);
      if (bits != 0) {
        return w * kWordBits + (63 - static_cast<int32_t>(sycl::clz(bits)));
      }
    }
    return kNoNode;
  }

  void operator()(sycl::nd_item<1> item) const {
    const int32_t n = num_nodes_;
    const int32_t nw = words_per_row_;
    const int32_t lpr = lanes_per_request_;
    const int32_t local_id = static_cast<int32_t>(item.get_local_id(0));
    const int32_t slot = local_id / lpr;
    const int32_t lane = local_id - slot * lpr;
    const int64_t bid = static_cast<int64_t>(item.get_group(0)) * requests_per_group_ + slot;
    const bool active = bid < batch_size_;

    word_t* rows =
        rows_.template get_multi_ptr<sycl::access::decorated::no>().get() + static_cast<int64_t>(slot) * n * nw;
    int32_t* parent = parent_.template get_multi_ptr<sycl::access::decorated::no>().get() + slot * n;

    if (active) {
      const bool* block = tree_mask_ + bid * static_cast<int64_t>(n) * n;
#pragma unroll 2
      for (int32_t i = lane; i < n; i += lpr) {
        const bool* src = block + static_cast<int64_t>(i) * n;
#pragma unroll 2
        for (int32_t w = 0; w < nw; ++w) {
          const int32_t lo = w * kWordBits;
          rows[static_cast<int64_t>(i) * nw + w] = pack_mask_bytes(src + lo, sycl::min(kWordBits, n - lo));
        }
      }
    }
    sycl::group_barrier(item.get_group());

    const int64_t out_base = bid * n;
    if (active) {
      const int64_t seq_len = static_cast<int64_t>(verified_seq_len_[bid]);
#pragma unroll 2
      for (int32_t i = lane; i < n; i += lpr) {
        const word_t* row = rows + static_cast<int64_t>(i) * nw;
        int32_t depth = 0;
#pragma unroll 2
        for (int32_t w = 0; w < nw; ++w) {
          depth += static_cast<int32_t>(sycl::popcount(strict_word(row, i, w)));
        }
        positions_[out_base + i] = seq_len + depth;
        retrive_index_[out_base + i] = out_base + i;
        parent[i] = immediate_parent(row, i);
      }
    }
    sycl::group_barrier(item.get_group());

    if (active) {
#pragma unroll 2
      for (int32_t i = lane; i < n; i += lpr) {
        const int32_t word = i / kWordBits;
        const word_t bit = word_t{1} << (i % kWordBits);

        int32_t next_token = kNoNode;
#pragma unroll 2
        for (int32_t k = i + 1; k < n; ++k) {
          if (rows[static_cast<int64_t>(k) * nw + word] & bit) {
            next_token = k;
            break;
          }
        }
        retrive_next_token_[out_base + i] = next_token;

        int32_t next_sibling = kNoNode;
        if (parent[i] != kNoNode) {
#pragma unroll 2
          for (int32_t k = i + 1; k < n; ++k) {
            if (parent[k] == parent[i]) {
              next_sibling = k;
              break;
            }
          }
        }
        retrive_next_sibling_[out_base + i] = next_sibling;
      }
    }
  }

  const bool* tree_mask_;
  const seq_t* verified_seq_len_;
  int64_t* positions_;
  int64_t* retrive_index_;
  int64_t* retrive_next_token_;
  int64_t* retrive_next_sibling_;
  int32_t batch_size_;
  int32_t num_nodes_;
  int32_t words_per_row_;
  int32_t lanes_per_request_;
  int32_t requests_per_group_;

  sycl::local_accessor<word_t, 1> rows_;
  sycl::local_accessor<int32_t, 1> parent_;
};

constexpr int64_t kFastPathMaxNodes = 64;

inline bool request_in_one_sub_group(int32_t n, int32_t lrange, int32_t max_width) {
  return lrange <= max_width || max_width % n == 0;
}

inline void sync_request_scope(sycl::nd_item<1> item, int32_t n, int32_t lrange, int32_t max_width) {
  if (request_in_one_sub_group(n, lrange, max_width)) {
    sycl::group_barrier(item.get_sub_group());
  } else {
    sycl::group_barrier(item.get_group());
  }
}

template <typename seq_t>
struct ReconstructTreeSmallKernel : public __SYCL_KER_CONFIG_CONVENTION__ {
  ReconstructTreeSmallKernel(
      const bool* tree_mask,
      const seq_t* verified_seq_len,
      int64_t* positions,
      int64_t* retrive_index,
      int64_t* retrive_next_token,
      int64_t* retrive_next_sibling,
      int32_t batch_size,
      int32_t num_nodes,
      int32_t requests_per_group)
      : tree_mask_(tree_mask),
        verified_seq_len_(verified_seq_len),
        positions_(positions),
        retrive_index_(retrive_index),
        retrive_next_token_(retrive_next_token),
        retrive_next_sibling_(retrive_next_sibling),
        batch_size_(batch_size),
        num_nodes_(num_nodes),
        requests_per_group_(requests_per_group) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {
    const size_t words_per_request = static_cast<size_t>((num_nodes_ + 7) / 8);
    parents_ = sycl::local_accessor<uint64_t, 1>(
        sycl::range<1>(static_cast<size_t>(requests_per_group_) * words_per_request), cgh);
  }

  void operator()(sycl::nd_item<1> item) const {
    const int32_t n = num_nodes_;
    const int32_t max_width = static_cast<int32_t>(item.get_sub_group().get_max_local_range()[0]);
    const int32_t lrange = static_cast<int32_t>(item.get_local_range(0));
    const int32_t local_id = static_cast<int32_t>(item.get_local_id(0));
    const int32_t slot = local_id / n;
    const int32_t i = local_id - slot * n;
    const int64_t bid = static_cast<int64_t>(item.get_group(0)) * requests_per_group_ + slot;
    const bool active = bid < batch_size_;

    const bool wide = (n & 7) == 0;
    const int64_t out = bid * n + i;

    int32_t parent = kNoNode;
    if (active) {
      const int64_t seq_base = static_cast<int64_t>(verified_seq_len_[bid]);
      const uint64_t row = pack_mask_row(tree_mask_ + out * static_cast<int64_t>(n), n, wide);
      const uint64_t strict = row & ((uint64_t{1} << i) - 1);

      positions_[out] = seq_base + static_cast<int64_t>(sycl::popcount(strict));
      retrive_index_[out] = out;

      if (strict != 0) {
        parent = 63 - static_cast<int32_t>(sycl::clz(strict));
      }
    }

    const int32_t words_per_request = (n + 7) >> 3;
    uint64_t* slot_words = parents_.template get_multi_ptr<sycl::access::decorated::no>().get() +
                           static_cast<int64_t>(slot) * words_per_request;

    reinterpret_cast<uint8_t*>(slot_words)[i] = static_cast<uint8_t>(parent);
    sync_request_scope(item, n, lrange, max_width);

    const uint64_t want_kids = splat_byte(i);
    const uint64_t want_sibs = splat_byte(parent);
    uint64_t kids = 0;
    uint64_t sibs = 0;
#pragma unroll 2
    for (int32_t w = 0; w < words_per_request; ++w) {
      const uint64_t v = slot_words[w];
      kids |= byte_eq_mask(v, want_kids) << (w << 3);
      sibs |= byte_eq_mask(v, want_sibs) << (w << 3);
    }
    if (n < 64) {
      const uint64_t valid = (uint64_t{1} << n) - 1;
      kids &= valid;
      sibs &= valid;
    }

    if (active) {
      retrive_next_token_[out] = kids ? static_cast<int64_t>(sycl::ctz(kids)) : kNoNode;

      int64_t next_sibling = kNoNode;
      if (parent != kNoNode) {
        const uint64_t above = (i >= 63) ? 0ull : (sibs & ~((uint64_t{1} << (i + 1)) - 1));
        if (above) {
          next_sibling = static_cast<int64_t>(sycl::ctz(above));
        }
      }
      retrive_next_sibling_[out] = next_sibling;
    }
  }

  const bool* tree_mask_;
  const seq_t* verified_seq_len_;
  int64_t* positions_;
  int64_t* retrive_index_;
  int64_t* retrive_next_token_;
  int64_t* retrive_next_sibling_;
  int32_t batch_size_;
  int32_t num_nodes_;
  int32_t requests_per_group_;

  sycl::local_accessor<uint64_t, 1> parents_;
};

}  // namespace

SGL_KERNEL_EXPORT void reconstruct_indices_from_tree_mask(
    const at::Tensor& tree_mask,
    const at::Tensor& verified_seq_len,
    at::Tensor& positions,
    at::Tensor& retrive_index,
    at::Tensor& retrive_next_token,
    at::Tensor& retrive_next_sibling,
    int64_t batch_size,
    int64_t draft_token_num) {
  CHECK_INPUT(tree_mask);
  CHECK_INPUT(verified_seq_len);
  CHECK_INPUT(positions);
  CHECK_INPUT(retrive_index);
  CHECK_INPUT(retrive_next_token);
  CHECK_INPUT(retrive_next_sibling);

  TORCH_CHECK(batch_size >= 0, "reconstruct_indices_from_tree_mask: batch_size must be non-negative, got ", batch_size);
  TORCH_CHECK(
      draft_token_num > 0,
      "reconstruct_indices_from_tree_mask: draft_token_num must be positive, got ",
      draft_token_num);

  TORCH_CHECK(
      tree_mask.scalar_type() == at::kBool,
      "reconstruct_indices_from_tree_mask: tree_mask must be bool, got ",
      tree_mask.scalar_type());
  for (const at::Tensor* out : {&positions, &retrive_index, &retrive_next_token, &retrive_next_sibling}) {
    TORCH_CHECK(
        out->scalar_type() == at::kLong,
        "reconstruct_indices_from_tree_mask: outputs must be int64, got ",
        out->scalar_type());
    TORCH_CHECK(
        out->numel() >= batch_size * draft_token_num,
        "reconstruct_indices_from_tree_mask: outputs must hold batch_size * draft_token_num = ",
        batch_size * draft_token_num,
        " elements, got ",
        out->numel());
  }
  TORCH_CHECK(
      verified_seq_len.numel() >= batch_size,
      "reconstruct_indices_from_tree_mask: verified_seq_len must hold at least ",
      batch_size,
      " elements, got ",
      verified_seq_len.numel());
  TORCH_CHECK(
      tree_mask.numel() >= batch_size * draft_token_num * draft_token_num,
      "reconstruct_indices_from_tree_mask: tree_mask needs at least ",
      batch_size * draft_token_num * draft_token_num,
      " elements, got ",
      tree_mask.numel());

  if (batch_size == 0) {
    return;
  }

  auto& queue = dpcppGetCurrentQueue();
  const int64_t bs = batch_size;
  const int64_t n = draft_token_num;
  const int64_t max_wg = dpcppMaxWorkGroupSize();

  const int64_t lanes_per_request = std::min<int64_t>(n, max_wg);

  const int64_t num_subslices = xe_core_count(queue.get_device());
  int64_t requests_per_group = std::max<int64_t>(bs / num_subslices, 1);
  requests_per_group = std::min(requests_per_group, std::max<int64_t>(kLaneTarget / lanes_per_request, 1));
  requests_per_group = std::min(requests_per_group, std::max<int64_t>(max_wg / lanes_per_request, 1));

  const bool use_fast_path = n <= kFastPathMaxNodes && lanes_per_request == n;

  if (use_fast_path && n < kSubGroupWidth && kSubGroupWidth % n == 0) {
    const int64_t per_sub_group = kSubGroupWidth / n;
    if (requests_per_group >= per_sub_group) {
      requests_per_group -= requests_per_group % per_sub_group;
    } else {
      requests_per_group = std::min<int64_t>(per_sub_group, bs);
    }
  }

  const int64_t local_range = requests_per_group * lanes_per_request;
  const int64_t groups = (bs + requests_per_group - 1) / requests_per_group;
  const int64_t words_per_row = (n + 63) / 64;

  AT_DISPATCH_INDEX_TYPES(verified_seq_len.scalar_type(), "reconstruct_indices_from_tree_mask", [&] {
    if (use_fast_path) {
      ReconstructTreeSmallKernel<index_t> kernel(
          tree_mask.data_ptr<bool>(),
          verified_seq_len.data_ptr<index_t>(),
          positions.data_ptr<int64_t>(),
          retrive_index.data_ptr<int64_t>(),
          retrive_next_token.data_ptr<int64_t>(),
          retrive_next_sibling.data_ptr<int64_t>(),
          static_cast<int32_t>(bs),
          static_cast<int32_t>(n),
          static_cast<int32_t>(requests_per_group));
      sycl_kernel_submit(groups * local_range, local_range, queue, kernel);
      return;
    }
    ReconstructTreeKernel<index_t> kernel(
        tree_mask.data_ptr<bool>(),
        verified_seq_len.data_ptr<index_t>(),
        positions.data_ptr<int64_t>(),
        retrive_index.data_ptr<int64_t>(),
        retrive_next_token.data_ptr<int64_t>(),
        retrive_next_sibling.data_ptr<int64_t>(),
        static_cast<int32_t>(bs),
        static_cast<int32_t>(n),
        static_cast<int32_t>(words_per_row),
        static_cast<int32_t>(lanes_per_request),
        static_cast<int32_t>(requests_per_group));
    sycl_kernel_submit(groups * local_range, local_range, queue, kernel);
  });
}
