/* Copyright 2025 SGLang Team. All Rights Reserved.

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

// reconstruct_indices_from_tree_mask: the inverse of build_tree_kernel_efficient.
// build_tree consumes parent_list/selected_index and *emits* tree_mask plus the
// four verify-metadata tensors; this op consumes tree_mask and emits only those
// four.  NGRAM speculative decoding produces the mask directly, so it needs the
// inverse direction.
//
// Contract (per request b, per node tid, n = draft_token_num).  tree_mask is the
// transitive-closure ancestor matrix with the diagonal set, so
//   anc[i, j] = tree_mask[b, i, j] && j < i        (strict ancestors)
//   positions[b, tid]             = |anc[tid, :]| + verified_seq_len[b]
//   retrive_index[b, tid]         = b * n + tid
//   parent[tid]                   = max{ j < tid : anc[tid, j] }  else -1
//   retrive_next_token[b, tid]    = min{ k > tid : anc[k, tid] }  else -1
//   retrive_next_sibling[b, tid]  = min{ k > tid : parent[k] == parent[tid] }
//                                   else -1, and -1 when parent[tid] < 0.
//
// Because the mask is a transitive closure over topologically ordered nodes, the
// smallest k > tid having tid as an ancestor is necessarily tid's *first child*:
// if k's parent p were not tid then tid < p < k and p would be a smaller such k.
// So the first-descendant column scan below is exactly build_tree's first-child
// link, without needing to materialize a child bitmask.
//
// Once a mask row is bit-packed the whole problem is popcount/clz/bit-test:
//   strict = row[i] with bits >= i cleared
//   depth  = popcount(strict)     (build_tree needs a serial parent-chain walk)
//   parent = highest set bit of strict, else -1
//
// Shape of the launch.  One lane per node, `requests_per_group` requests packed
// per work-group, rows bit-packed once into SLM.  Packing requests matters on the
// store side: the mask is n*n bytes (torch bool is 1 byte per element) but the
// outputs are 4 * n int64 = 32n bytes, so for n < 32 this kernel *writes* more
// than it reads, and one request alone covers only 8n bytes of each output -- a
// partial cache line at the n = 8..16 that NGRAM actually uses.  Several requests
// per work-group turn those into full-line stores.
//
// Rows live in SLM rather than in per-lane registers on purpose.  A private
// word_t row[n] would be indexed by a runtime k in the column scans, which forces
// either indirect register addressing or a spill to scratch; SLM is the addressing
// mode built for that.  At realistic shapes it costs nothing measurable anyway --
// bs=256, n=8 moves ~80 KB, about 0.2 us of traffic against a kernel-launch floor
// of several microseconds.

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"
#include "sgl_kernel_export.h"

namespace {

constexpr int32_t kNoNode = -1;

// Target work-group width in lanes; requests_per_group fills up to this.
constexpr int64_t kLaneTarget = 128;

static_assert(sizeof(bool) == 1, "tree_mask packing assumes 1-byte bool");

// Inverse of build_tree's expand_mask_nibble(), same magic constant: gather the
// low bit of each of 4 bool bytes into the low nibble of the result.  Byte k sits
// at bit 8k and the multiply moves it to bit 8k + (21 - 7k) = 21 + k.  Every
// partial-product position 21 + 8i - 7j is distinct (8(i-i') == 7(j-j') forces
// i == i' and j == j' over i, j in [0, 4)), so nothing carries, and only the i == j
// terms land in [21, 24].  Products above bit 31 are dropped harmlessly.
inline uint32_t compress_mask_nibble(uint32_t bytes) {
  return ((bytes * 0x00204081u) >> 21) & 0xFu;
}

// Pack `count` (<= 64) bool bytes at `src` into the low bits of a 64-bit word.
inline uint64_t pack_mask_bytes(const bool* src, int32_t count) {
  uint64_t bits = 0;
  int32_t c = 0;
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

  // Bits of word `w` of node `i`'s row that belong to strict ancestors (j < i).
  inline word_t strict_word(const word_t* row, int32_t i, int32_t w) const {
    const int32_t lo = w * kWordBits;
    if (i <= lo) {
      return 0;
    }
    const int32_t keep = i - lo;
    const word_t m = keep >= kWordBits ? ~word_t{0} : static_cast<word_t>((word_t{1} << keep) - 1);
    return row[w] & m;
  }

  // Immediate parent of `i`: the highest strict ancestor.  The mask is a
  // transitive closure over topologically ordered nodes, so the greatest j < i
  // that is an ancestor of i is exactly i's parent.
  inline int32_t immediate_parent(const word_t* row, int32_t i) const {
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
    // Trailing slots of the last work-group have no request, but they must still
    // reach every barrier, so gate the work and not the synchronization.
    const bool active = bid < batch_size_;

    word_t* rows = rows_.template get_multi_ptr<sycl::access::decorated::no>().get() +
                   static_cast<int64_t>(slot) * n * nw;
    int32_t* parent = parent_.template get_multi_ptr<sycl::access::decorated::no>().get() + slot * n;

    // Phase 1: bit-pack this request's mask into SLM, one row per lane.
    if (active) {
      const bool* block = tree_mask_ + bid * static_cast<int64_t>(n) * n;
      for (int32_t i = lane; i < n; i += lpr) {
        const bool* src = block + static_cast<int64_t>(i) * n;
        for (int32_t w = 0; w < nw; ++w) {
          const int32_t lo = w * kWordBits;
          rows[static_cast<int64_t>(i) * nw + w] = pack_mask_bytes(src + lo, sycl::min(kWordBits, n - lo));
        }
      }
    }
    sycl::group_barrier(item.get_group());

    // Phase 2: positions / retrive_index / parent -- all row-local.
    const int64_t out_base = bid * n;
    if (active) {
      const int64_t seq_len = static_cast<int64_t>(verified_seq_len_[bid]);
      for (int32_t i = lane; i < n; i += lpr) {
        const word_t* row = rows + static_cast<int64_t>(i) * nw;
        int32_t depth = 0;
        for (int32_t w = 0; w < nw; ++w) {
          depth += static_cast<int32_t>(sycl::popcount(strict_word(row, i, w)));
        }
        positions_[out_base + i] = seq_len + depth;
        retrive_index_[out_base + i] = out_base + i;
        parent[i] = immediate_parent(row, i);
      }
    }
    sycl::group_barrier(item.get_group());

    // Phase 3: the two tree links, both column scans over the packed rows.
    if (active) {
      for (int32_t i = lane; i < n; i += lpr) {
        const int32_t word = i / kWordBits;
        const word_t bit = word_t{1} << (i % kWordBits);

        int32_t next_token = kNoNode;
        for (int32_t k = i + 1; k < n; ++k) {
          if (rows[static_cast<int64_t>(k) * nw + word] & bit) {
            next_token = k;
            break;
          }
        }
        retrive_next_token_[out_base + i] = next_token;

        // Roots never link to each other: parent[i] < 0 short-circuits, and a
        // root k would fail parent[k] == parent[i] anyway.
        int32_t next_sibling = kNoNode;
        if (parent[i] != kNoNode) {
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

  TORCH_CHECK(
      batch_size >= 0, "reconstruct_indices_from_tree_mask: batch_size must be non-negative, got ", batch_size);
  TORCH_CHECK(
      draft_token_num > 0,
      "reconstruct_indices_from_tree_mask: draft_token_num must be positive, got ",
      draft_token_num);

  TORCH_CHECK(
      tree_mask.scalar_type() == at::kBool,
      "reconstruct_indices_from_tree_mask: tree_mask must be bool, got ",
      tree_mask.scalar_type());
  for (const auto& out : {positions, retrive_index, retrive_next_token, retrive_next_sibling}) {
    TORCH_CHECK(
        out.scalar_type() == at::kLong,
        "reconstruct_indices_from_tree_mask: outputs must be int64, got ",
        out.scalar_type());
    TORCH_CHECK(
        out.numel() >= batch_size * draft_token_num,
        "reconstruct_indices_from_tree_mask: outputs must hold batch_size * draft_token_num = ",
        batch_size * draft_token_num,
        " elements, got ",
        out.numel());
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

  // One lane per node.  Only a tree wider than a work-group strip-mines.
  const int64_t lanes_per_request = std::min<int64_t>(n, max_wg);

  // Requests per work-group: fill the work-group up to kLaneTarget lanes, but
  // never so much that we drop below one work-group per Xe-core while the batch
  // could still have supplied one.  The division floors deliberately -- rounding
  // up packs too early and starves cores (bs = 64, n = 8 would give 16
  // work-groups for 20 cores instead of 22).
  const int64_t num_subslices = std::max<int64_t>(
      queue.get_device().get_info<sycl::ext::intel::info::device::gpu_slices>() *
          queue.get_device().get_info<sycl::ext::intel::info::device::gpu_subslices_per_slice>(),
      1);
  int64_t requests_per_group = std::max<int64_t>(bs / num_subslices, 1);
  requests_per_group = std::min(requests_per_group, std::max<int64_t>(kLaneTarget / lanes_per_request, 1));
  requests_per_group = std::min(requests_per_group, std::max<int64_t>(max_wg / lanes_per_request, 1));

  const int64_t local_range = requests_per_group * lanes_per_request;
  const int64_t groups = (bs + requests_per_group - 1) / requests_per_group;
  const int64_t words_per_row = (n + 63) / 64;

  AT_DISPATCH_INDEX_TYPES(verified_seq_len.scalar_type(), "reconstruct_indices_from_tree_mask", [&] {
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
