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

#include "SYCLHelpers.h"
#include "Utils.h"
#include "sgl_kernel_export.h"

namespace {

constexpr int32_t kNoNode = -1;

static_assert(sizeof(bool) == 1, "tree_mask staging assumes 1-byte bool");

// Copy one [n] mask row verbatim. Rows start 8-byte aligned whenever n % 8 == 0
// (tree_mask is torch-allocated, and both the block offset bid*n*n and the row
// offset i*n are then multiples of 8), so the common case moves qwords.
inline void copy_mask_row(uint8_t* dst, const uint8_t* src, int32_t n, bool wide) {
  if (wide) {
    uint64_t* d = reinterpret_cast<uint64_t*>(dst);
    const uint64_t* s = reinterpret_cast<const uint64_t*>(src);
    const int32_t words = n >> 3;
#pragma unroll 2
    for (int32_t w = 0; w < words; ++w) {
      d[w] = s[w];
    }
    return;
  }
#pragma unroll 2
  for (int32_t c = 0; c < n; ++c) {
    dst[c] = src[c];
  }
}

inline int64_t tile_words(int64_t n) {
  return (n * n + 7) / 8;
}

// One request per work-group, so a request's lanes are all in one sub-group
// whenever the work-group fits in one.
inline void sync_lanes(sycl::nd_item<1> item) {
  const auto sg = item.get_sub_group();
  if (item.get_local_range(0) <= sg.get_max_local_range()[0]) {
    sycl::group_barrier(sg);
  } else {
    sycl::group_barrier(item.get_group());
  }
}

/* Dense masked reductions over the [n, n] mask tile -- the same formulation as
   the Triton reference in benchmark/bench_reconstruct_tree_mask.py.

   tree_mask is the transitive-closure ancestor matrix with the diagonal set, so
   the whole contract is a set of reductions over the block:
     ancestor[i, j]  = mask[i, j] && j < i        (strict ancestors)
     positions[i]    = |ancestor[i, :]| + verified_seq_len
     parent[i]       = max{ j : ancestor[i, j] }, else -1
     next_token[i]   = min{ k > i : ancestor[k, i] }, else -1
     next_sibling[i] = min{ k > i : parent[k] == parent[i] }, -1 if parent[i] < 0

   Geometry mirrors Triton's grid = (batch_size,): one work-group per request,
   one lane per node. Every reduction runs the full n and predicates instead of
   breaking early, so trip counts are uniform across lanes and n stands in for
   "none found" exactly as Triton's sentinel does.

   The first two are row-local -- lane i reduces its own row along axis 1. The
   last two are column reductions along axis 0, so lane i needs column i, which
   lives in every other lane's row; the tile is staged in SLM once and reduced
   both ways from there. kStageTile == false is the fallback for an n whose tile
   exceeds SLM, and reads the columns straight from global memory instead. */
template <typename seq_t, bool kStageTile>
struct ReconstructTreeDenseKernel : public __SYCL_KER_CONFIG_CONVENTION__ {
  ReconstructTreeDenseKernel(
      const bool* tree_mask,
      const seq_t* verified_seq_len,
      int64_t* positions,
      int64_t* retrive_index,
      int64_t* retrive_next_token,
      int64_t* retrive_next_sibling,
      int32_t num_nodes,
      int32_t lanes_per_request)
      : tree_mask_(tree_mask),
        verified_seq_len_(verified_seq_len),
        positions_(positions),
        retrive_index_(retrive_index),
        retrive_next_token_(retrive_next_token),
        retrive_next_sibling_(retrive_next_sibling),
        num_nodes_(num_nodes),
        lanes_per_request_(lanes_per_request) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {
    const size_t words = kStageTile ? static_cast<size_t>(tile_words(num_nodes_)) : 1;
    tile_ = sycl::local_accessor<uint64_t, 1>(sycl::range<1>(words), cgh);
    parents_ = sycl::local_accessor<int32_t, 1>(sycl::range<1>(static_cast<size_t>(num_nodes_)), cgh);
  }

  void operator()(sycl::nd_item<1> item) const {
    const int32_t n = num_nodes_;
    const int32_t lpr = lanes_per_request_;
    const int32_t lane = static_cast<int32_t>(item.get_local_id(0));
    const int64_t bid = static_cast<int64_t>(item.get_group(0));
    const bool wide = (n & 7) == 0;

    const uint8_t* block = reinterpret_cast<const uint8_t*>(tree_mask_) + bid * static_cast<int64_t>(n) * n;
    int32_t* parents = parents_.template get_multi_ptr<sycl::access::decorated::no>().get();
    uint8_t* tile = nullptr;
    if constexpr (kStageTile) {
      tile = reinterpret_cast<uint8_t*>(tile_.template get_multi_ptr<sycl::access::decorated::no>().get());
    }

    const int64_t out_base = bid * n;
    // Read ahead of the row load: the two are independent cold trips, and
    // folding this into the positions_ store put it downstream of the row.
    const int64_t seq_len = static_cast<int64_t>(verified_seq_len_[bid]);

    // Axis-1 reductions. Lane i owns row i, so no communication is needed --
    // but it does stage the row for the column pass below.
#pragma unroll 2
    for (int32_t i = lane; i < n; i += lpr) {
      const uint8_t* row = block + static_cast<int64_t>(i) * n;
      if constexpr (kStageTile) {
        uint8_t* dst = tile + static_cast<int64_t>(i) * n;
        copy_mask_row(dst, row, n, wide);
        row = dst;
      }

      int32_t depth = 0;
      int32_t parent = kNoNode;
#pragma unroll 2
      for (int32_t j = 0; j < n; ++j) {
        const bool ancestor = (row[j] != 0) && (j < i);
        depth += ancestor ? 1 : 0;
        parent = ancestor ? j : parent;  // ascending j, so this is the max
      }

      positions_[out_base + i] = seq_len + depth;
      retrive_index_[out_base + i] = out_base + i;
      parents[i] = parent;
    }

    sync_lanes(item);

    // Axis-0 reductions. Both are a min over rows k of a predicate on column i,
    // with n as the "none found" sentinel.
#pragma unroll 2
    for (int32_t i = lane; i < n; i += lpr) {
      const int32_t parent_i = parents[i];
      int32_t first_child = n;
      int32_t first_sibling = n;
#pragma unroll 2
      for (int32_t k = 0; k < n; ++k) {
        const int64_t cell = static_cast<int64_t>(k) * n + i;
        uint8_t ancestor_ki;
        if constexpr (kStageTile) {
          ancestor_ki = tile[cell];
        } else {
          ancestor_ki = block[cell];
        }
        const bool later = k > i;
        first_child = sycl::min(first_child, (ancestor_ki != 0 && later) ? k : n);
        // Roots never link to each other: parent_i < 0 rules the column out.
        const bool sibling = later && parent_i != kNoNode && parents[k] == parent_i;
        first_sibling = sycl::min(first_sibling, sibling ? k : n);
      }

      retrive_next_token_[out_base + i] = first_child < n ? first_child : kNoNode;
      retrive_next_sibling_[out_base + i] = first_sibling < n ? first_sibling : kNoNode;
    }
  }

  const bool* tree_mask_;
  const seq_t* verified_seq_len_;
  int64_t* positions_;
  int64_t* retrive_index_;
  int64_t* retrive_next_token_;
  int64_t* retrive_next_sibling_;
  int32_t num_nodes_;
  int32_t lanes_per_request_;

  sycl::local_accessor<uint64_t, 1> tile_;
  sycl::local_accessor<int32_t, 1> parents_;
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

  // One work-group per request, matching Triton's grid = (batch_size,). A lane
  // per node, and a lane walks n/lpr rows when n exceeds the work-group limit.
  const int64_t lanes_per_request = std::min<int64_t>(n, max_wg);
  const int64_t groups = bs;

  const int64_t local_mem = static_cast<int64_t>(queue.get_device().get_info<sycl::info::device::local_mem_size>());
  const int64_t parent_bytes = n * static_cast<int64_t>(sizeof(int32_t));
  TORCH_CHECK(
      parent_bytes <= local_mem,
      "reconstruct_indices_from_tree_mask: draft_token_num = ",
      n,
      " needs ",
      parent_bytes,
      " bytes of local memory for the parent vector, device has ",
      local_mem);
  const bool stage_tile = tile_words(n) * 8 + parent_bytes <= local_mem;

  AT_DISPATCH_INDEX_TYPES(verified_seq_len.scalar_type(), "reconstruct_indices_from_tree_mask", [&] {
    if (stage_tile) {
      ReconstructTreeDenseKernel<index_t, true> kernel(
          tree_mask.data_ptr<bool>(),
          verified_seq_len.data_ptr<index_t>(),
          positions.data_ptr<int64_t>(),
          retrive_index.data_ptr<int64_t>(),
          retrive_next_token.data_ptr<int64_t>(),
          retrive_next_sibling.data_ptr<int64_t>(),
          static_cast<int32_t>(n),
          static_cast<int32_t>(lanes_per_request));
      sycl_kernel_submit(groups * lanes_per_request, lanes_per_request, queue, kernel);
      return;
    }
    ReconstructTreeDenseKernel<index_t, false> kernel(
        tree_mask.data_ptr<bool>(),
        verified_seq_len.data_ptr<index_t>(),
        positions.data_ptr<int64_t>(),
        retrive_index.data_ptr<int64_t>(),
        retrive_next_token.data_ptr<int64_t>(),
        retrive_next_sibling.data_ptr<int64_t>(),
        static_cast<int32_t>(n),
        static_cast<int32_t>(lanes_per_request));
    sycl_kernel_submit(groups * lanes_per_request, lanes_per_request, queue, kernel);
  });
}
