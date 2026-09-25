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

#include <sycl/sycl.hpp>
#include <type_traits>

#include "SYCLHelpers.h"
#include "Utils.h"
#include "sgl_kernel_export.h"

namespace {

constexpr int32_t kNoNode = -1;

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

template <int32_t N>
inline uint64_t pack_mask_row_static(const bool* src) {
  if constexpr ((N & 7) != 0) {
    return pack_mask_bytes(src, N);
  } else {
    const uint64_t* words = reinterpret_cast<const uint64_t*>(src);
    uint64_t bits = 0;
#pragma unroll
    for (int32_t w = 0; w < N / 8; ++w) {
      bits |= compress_mask_byte8(words[w]) << (w << 3);
    }
    return bits;
  }
}

inline void
accumulate_byte_matches(uint64_t v, int32_t w, uint64_t want_kids, uint64_t want_sibs, uint64_t& kids, uint64_t& sibs) {
  kids |= byte_eq_mask(v, want_kids) << (w << 3);
  sibs |= byte_eq_mask(v, want_sibs) << (w << 3);
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
      int32_t num_nodes,
      int32_t words_per_row,
      int32_t lanes_per_request)
      : tree_mask_(tree_mask),
        verified_seq_len_(verified_seq_len),
        positions_(positions),
        retrive_index_(retrive_index),
        retrive_next_token_(retrive_next_token),
        retrive_next_sibling_(retrive_next_sibling),
        num_nodes_(num_nodes),
        words_per_row_(words_per_row),
        lanes_per_request_(lanes_per_request) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {
    rows_ = sycl::local_accessor<word_t, 1>(sycl::range<1>(static_cast<size_t>(num_nodes_) * words_per_row_), cgh);
    parent_ = sycl::local_accessor<int32_t, 1>(sycl::range<1>(static_cast<size_t>(num_nodes_)), cgh);
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
    const int32_t lane = static_cast<int32_t>(item.get_local_id(0));
    const int64_t bid = static_cast<int64_t>(item.get_group(0));

    word_t* rows = rows_.template get_multi_ptr<sycl::access::decorated::no>().get();
    int32_t* parent = parent_.template get_multi_ptr<sycl::access::decorated::no>().get();

    // Phase 1: bit-pack this request's [n, n] byte mask into SLM, one qword per 64
    // nodes, so the two link phases below read 8x fewer bytes.
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
    sycl::group_barrier(item.get_group());

    // Phase 2: depth is the population count of row i below the diagonal, and the
    // immediate parent is that row's highest set bit below the diagonal.
    const int64_t out_base = bid * n;
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
    sycl::group_barrier(item.get_group());

    // Phase 3: invert. First child is the lowest k > i whose row has bit i set;
    // next sibling is the lowest k > i sharing i's parent.
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

  const bool* tree_mask_;
  const seq_t* verified_seq_len_;
  int64_t* positions_;
  int64_t* retrive_index_;
  int64_t* retrive_next_token_;
  int64_t* retrive_next_sibling_;
  int32_t num_nodes_;
  int32_t words_per_row_;
  int32_t lanes_per_request_;

  sycl::local_accessor<word_t, 1> rows_;
  sycl::local_accessor<int32_t, 1> parent_;
};

constexpr int64_t kFastPathMaxNodes = 64;

inline void sync_request_scope(sycl::nd_item<1> item, int32_t n, int32_t max_width) {
  if (n <= max_width) {
    sycl::group_barrier(item.get_sub_group());
  } else {
    sycl::group_barrier(item.get_group());
  }
}

template <typename seq_t, int32_t N_CONST>
struct ReconstructTreeSmallKernel : public __SYCL_KER_CONFIG_CONVENTION__ {
  static constexpr bool kStaticN = N_CONST > 0;
  static constexpr int32_t kStaticWords = kStaticN ? (N_CONST + 7) >> 3 : 0;

  ReconstructTreeSmallKernel(
      const bool* tree_mask,
      const seq_t* verified_seq_len,
      int64_t* positions,
      int64_t* retrive_index,
      int64_t* retrive_next_token,
      int64_t* retrive_next_sibling,
      int32_t num_nodes)
      : tree_mask_(tree_mask),
        verified_seq_len_(verified_seq_len),
        positions_(positions),
        retrive_index_(retrive_index),
        retrive_next_token_(retrive_next_token),
        retrive_next_sibling_(retrive_next_sibling),
        num_nodes_(num_nodes) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {
    const size_t words = static_cast<size_t>(kStaticN ? kStaticWords : (num_nodes_ + 7) / 8);
    parents_ = sycl::local_accessor<uint64_t, 1>(sycl::range<1>(words), cgh);
  }

  void operator()(sycl::nd_item<1> item) const {
    const int32_t n = kStaticN ? N_CONST : num_nodes_;
    const int32_t max_width = static_cast<int32_t>(item.get_sub_group().get_max_local_range()[0]);
    const int32_t i = static_cast<int32_t>(item.get_local_id(0));
    const int64_t bid = static_cast<int64_t>(item.get_group(0));
    const int64_t out = bid * n + i;

    const int64_t seq_base = static_cast<int64_t>(verified_seq_len_[bid]);
    const bool* src = tree_mask_ + out * static_cast<int64_t>(n);
    uint64_t row;
    if constexpr (kStaticN) {
      row = pack_mask_row_static<N_CONST>(src);
    } else {
      row = pack_mask_row(src, n, (n & 7) == 0);
    }
    const uint64_t strict = row & ((uint64_t{1} << i) - 1);

    positions_[out] = seq_base + static_cast<int64_t>(sycl::popcount(strict));
    retrive_index_[out] = out;

    int32_t parent = kNoNode;
    if (strict != 0) {
      parent = 63 - static_cast<int32_t>(sycl::clz(strict));
    }

    const int32_t words = kStaticN ? kStaticWords : (n + 7) >> 3;
    uint64_t* parent_words = parents_.template get_multi_ptr<sycl::access::decorated::no>().get();

    reinterpret_cast<uint8_t*>(parent_words)[i] = static_cast<uint8_t>(parent);
    sync_request_scope(item, n, max_width);

    const uint64_t want_kids = splat_byte(i);
    const uint64_t want_sibs = splat_byte(parent);
    uint64_t kids = 0;
    uint64_t sibs = 0;
    if constexpr (kStaticN) {
#pragma unroll
      for (int32_t w = 0; w < kStaticWords; ++w) {
        accumulate_byte_matches(parent_words[w], w, want_kids, want_sibs, kids, sibs);
      }
    } else {
#pragma unroll 2
      for (int32_t w = 0; w < words; ++w) {
        accumulate_byte_matches(parent_words[w], w, want_kids, want_sibs, kids, sibs);
      }
    }

    if (n < 64) {
      const uint64_t valid = (uint64_t{1} << n) - 1;
      kids &= valid;
      sibs &= valid;
    }

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

  const bool* tree_mask_;
  const seq_t* verified_seq_len_;
  int64_t* positions_;
  int64_t* retrive_index_;
  int64_t* retrive_next_token_;
  int64_t* retrive_next_sibling_;
  int32_t num_nodes_;

  sycl::local_accessor<uint64_t, 1> parents_;
};

template <typename seq_t, int32_t N_CONST>
inline void submit_small_kernel(
    sycl::queue& queue,
    const at::Tensor& tree_mask,
    const at::Tensor& verified_seq_len,
    at::Tensor& positions,
    at::Tensor& retrive_index,
    at::Tensor& retrive_next_token,
    at::Tensor& retrive_next_sibling,
    int32_t n,
    int64_t global_range,
    int64_t local_range) {
  ReconstructTreeSmallKernel<seq_t, N_CONST> kernel(
      tree_mask.data_ptr<bool>(),
      verified_seq_len.data_ptr<seq_t>(),
      positions.data_ptr<int64_t>(),
      retrive_index.data_ptr<int64_t>(),
      retrive_next_token.data_ptr<int64_t>(),
      retrive_next_sibling.data_ptr<int64_t>(),
      n);
  sycl_kernel_submit(global_range, local_range, queue, kernel);
}

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

  // One work-group per request, one work-item per draft token (strided when the
  // token count exceeds the device work-group limit).
  const int64_t local_range = std::min<int64_t>(n, max_wg);
  const int64_t global_range = bs * local_range;
  const bool use_fast_path = n <= kFastPathMaxNodes && local_range == n;
  const int64_t words_per_row = (n + 63) / 64;

  AT_DISPATCH_INDEX_TYPES(verified_seq_len.scalar_type(), "reconstruct_indices_from_tree_mask", [&] {
    if (use_fast_path) {
      auto submit_small = [&](auto n_const) {
        submit_small_kernel<index_t, decltype(n_const)::value>(
            queue,
            tree_mask,
            verified_seq_len,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            static_cast<int32_t>(n),
            global_range,
            local_range);
      };
      // Specialize the node counts speculative decoding actually runs at; every
      // other count falls through to the runtime-n instantiation. Adding a width
      // is one more case, at the cost of one more AOT-compiled kernel.
      switch (n) {
        case 8:
          submit_small(std::integral_constant<int32_t, 8>{});
          break;
        case 16:
          submit_small(std::integral_constant<int32_t, 16>{});
          break;
        case 32:
          submit_small(std::integral_constant<int32_t, 32>{});
          break;
        default:
          submit_small(std::integral_constant<int32_t, 0>{});
          break;
      }
      return;
    }
    ReconstructTreeKernel<index_t> kernel(
        tree_mask.data_ptr<bool>(),
        verified_seq_len.data_ptr<index_t>(),
        positions.data_ptr<int64_t>(),
        retrive_index.data_ptr<int64_t>(),
        retrive_next_token.data_ptr<int64_t>(),
        retrive_next_sibling.data_ptr<int64_t>(),
        static_cast<int32_t>(n),
        static_cast<int32_t>(words_per_row),
        static_cast<int32_t>(local_range));
    sycl_kernel_submit(global_range, local_range, queue, kernel);
  });
}
