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

// EAGLE speculative-decoding tree builder (build_tree_kernel_efficient).
//
// Semantics are those of the upstream Triton kernel
// (sglang/kernels/ops/speculative/spec_tree.py:sgl_build_tree_kernel_efficient_triton),
// but the work is reorganized for XPU:
//
//   * Triton runs one *serial* program per batch item. Here a whole work-group
//     owns one batch item and every draft token is a work-item, so the per-node
//     work (ancestor walk, mask row, sibling links) runs in parallel.
//   * The parent of each node is resolved once into `parent_pos` in shared local
//     memory. Triton instead re-runs the O(draft_token_num) linear search over
//     `selected_index` at every step of every ancestor walk, i.e. it does
//     O(N^2 * depth) global loads per batch item where this does O(N^2) local
//     ones (N = draft_token_num).
//   * The sibling links are computed directly rather than by the serial
//     head-insertion Triton uses. Head-inserting nodes N-1..1 leaves, for each
//     parent p with ascending children i1<i2<...<ik:
//         next_token[p] = i1,  next_sibling[i_j] = i_{j+1},  next_sibling[i_k] = -1
//     which is a per-node scan and needs no cross-node ordering.
//
// Node numbering: 0 is the root (the bonus token); node i >= 1 corresponds to
// slot i-1 of `selected_index`. `parent_pos[i]` is the *node* index of i's
// parent, 0 when the parent is the root, and kParentNotFound when the parent
// token is absent from `selected_index` (Triton reads out of bounds / stops the
// walk in that case; we stop the walk and drop the link, which is well defined).

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"
#include "sgl_kernel_export.h"

namespace {

// Mirrors sglang.srt.speculative.eagle_utils.TreeMaskMode.
enum class TreeMaskMode : int64_t {
  FULL_MASK = 0,
  QLEN_ONLY = 1,
  QLEN_ONLY_BITPACKING = 2,
};

constexpr int32_t kParentNotFound = -1;

template <typename seq_t>
struct BuildTreeKernel : public __SYCL_KER_CONFIG_CONVENTION__ {
  BuildTreeKernel(
      const int64_t* parent_list,
      const int64_t* selected_index,
      const seq_t* verified_seq_len,
      bool* tree_mask,
      int64_t* positions,
      int64_t* retrieve_index,
      int64_t* retrieve_next_token,
      int64_t* retrieve_next_sibling,
      int32_t topk,
      int32_t depth,
      int32_t draft_token_num,
      int32_t parent_list_width,
      int64_t parent_list_stride,
      int64_t selected_index_stride,
      bool full_mask)
      : parent_list_(parent_list),
        selected_index_(selected_index),
        verified_seq_len_(verified_seq_len),
        tree_mask_(tree_mask),
        positions_(positions),
        retrieve_index_(retrieve_index),
        retrieve_next_token_(retrieve_next_token),
        retrieve_next_sibling_(retrieve_next_sibling),
        topk_(topk),
        depth_(depth),
        draft_token_num_(draft_token_num),
        parent_list_width_(parent_list_width),
        parent_list_stride_(parent_list_stride),
        selected_index_stride_(selected_index_stride),
        full_mask_(full_mask) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {
    parent_pos_ = sycl::local_accessor<int32_t, 1>(sycl::range<1>(draft_token_num_), cgh);
  }

  // Node index of `node`'s parent, or kParentNotFound.
  inline int32_t resolve_parent(int64_t bid, int32_t node) const {
    const int64_t* sel = selected_index_ + bid * selected_index_stride_;
    const int32_t parent_tb_idx = static_cast<int32_t>(sel[node - 1] / topk_);
    if (parent_tb_idx == 0) {
      return 0;  // child of the root
    }
    // Triton indexes parent_list unconditionally; an out-of-range table index
    // would read past the end, so treat it as "no parent" instead.
    if (parent_tb_idx < 0 || parent_tb_idx >= parent_list_width_) {
      return kParentNotFound;
    }
    const int64_t parent_token_idx = parent_list_[bid * parent_list_stride_ + parent_tb_idx];
    for (int32_t pp = 0; pp < draft_token_num_ - 1; ++pp) {
      if (sel[pp] == parent_token_idx) {
        return pp + 1;
      }
    }
    return kParentNotFound;
  }

  void operator()(sycl::nd_item<1> item) const {
    const int64_t bid = item.get_group(0);
    const int32_t tid = static_cast<int32_t>(item.get_local_id(0));
    const int32_t lrange = static_cast<int32_t>(item.get_local_range(0));
    const int32_t num_nodes = draft_token_num_;
    const int64_t seq_len = static_cast<int64_t>(verified_seq_len_[bid]);

    int32_t* parent_pos = parent_pos_.get_multi_ptr<sycl::access::decorated::no>().get();

    // FULL_MASK rows are (seq_len + draft_token_num) wide and packed per batch
    // item, so the row base needs sum(seq_len[0:bid]). One work-group owns one
    // batch item, so reduce the prefix across the group instead of paying for a
    // separate cumsum launch.
    int64_t mask_base = 0;
    int64_t row_stride = num_nodes;
    int64_t col_offset = 0;
    if (full_mask_) {
      int64_t partial = 0;
      for (int64_t b = tid; b < bid; b += lrange) {
        partial += static_cast<int64_t>(verified_seq_len_[b]);
      }
      const int64_t seq_len_prefix_sum = sycl::reduce_over_group(item.get_group(), partial, sycl::plus<int64_t>());
      mask_base = static_cast<int64_t>(num_nodes) * num_nodes * bid + seq_len_prefix_sum * num_nodes;
      row_stride = seq_len + num_nodes;
      col_offset = seq_len;
    } else {
      mask_base = static_cast<int64_t>(num_nodes) * num_nodes * bid;
    }

    // Phase 1: resolve every node's parent into SLM.
    for (int32_t i = tid; i < num_nodes; i += lrange) {
      parent_pos[i] = (i == 0) ? kParentNotFound : resolve_parent(bid, i);
    }
    sycl::group_barrier(item.get_group());

    const int64_t out_base = bid * num_nodes;
    for (int32_t i = tid; i < num_nodes; i += lrange) {
      // Phase 2: retrieve_index is the identity over the flattened draft tokens.
      retrieve_index_[out_base + i] = out_base + i;

      // Phase 3: first child of node i, and i's next sibling.
      int64_t next_token = -1;
      for (int32_t j = 1; j < num_nodes; ++j) {
        if (parent_pos[j] == i) {
          next_token = j;
          break;
        }
      }
      retrieve_next_token_[out_base + i] = next_token;

      int64_t next_sibling = -1;
      if (i > 0 && parent_pos[i] != kParentNotFound) {
        for (int32_t j = i + 1; j < num_nodes; ++j) {
          if (parent_pos[j] == parent_pos[i]) {
            next_sibling = j;
            break;
          }
        }
      }
      retrieve_next_sibling_[out_base + i] = next_sibling;

      // Phase 4: mask row for node i, plus its position (= depth in the tree).
      // Every tree cell is written here, so the caller's prefix fill (which only
      // supplies the [0, seq_len) columns) never affects this block.
      bool* row = tree_mask_ + mask_base + row_stride * i + col_offset;
      for (int32_t c = 0; c < num_nodes; ++c) {
        row[c] = false;
      }
      row[0] = true;  // every draft token attends the root

      if (i == 0) {
        positions_[out_base] = seq_len;
        continue;
      }

      int32_t node_depth = 0;
      int32_t cur = i;
      for (int32_t d = 0; d < depth_; ++d) {
        ++node_depth;
        row[cur] = true;
        const int32_t parent = parent_pos[cur];
        if (parent <= 0) {
          break;  // 0 -> reached the root (already marked); -1 -> unresolved
        }
        cur = parent;
      }
      positions_[out_base + i] = seq_len + node_depth;
    }
  }

  const int64_t* parent_list_;
  const int64_t* selected_index_;
  const seq_t* verified_seq_len_;
  bool* tree_mask_;
  int64_t* positions_;
  int64_t* retrieve_index_;
  int64_t* retrieve_next_token_;
  int64_t* retrieve_next_sibling_;
  int32_t topk_;
  int32_t depth_;
  int32_t draft_token_num_;
  int32_t parent_list_width_;
  int64_t parent_list_stride_;
  int64_t selected_index_stride_;
  bool full_mask_;

  sycl::local_accessor<int32_t, 1> parent_pos_;
};

}  // namespace

SGL_KERNEL_EXPORT void build_tree_kernel_efficient(
    at::Tensor parent_list,
    at::Tensor selected_index,
    at::Tensor verified_seq_len,
    at::Tensor tree_mask,
    at::Tensor positions,
    at::Tensor retrive_index,
    at::Tensor retrive_next_token,
    at::Tensor retrive_next_sibling,
    int64_t topk,
    int64_t depth,
    int64_t draft_token_num,
    int64_t tree_mask_mode) {
  CHECK_INPUT(parent_list);
  CHECK_INPUT(selected_index);
  CHECK_INPUT(verified_seq_len);
  CHECK_INPUT(tree_mask);
  CHECK_INPUT(positions);
  CHECK_INPUT(retrive_index);
  CHECK_INPUT(retrive_next_token);
  CHECK_INPUT(retrive_next_sibling);

  const auto mode = static_cast<TreeMaskMode>(tree_mask_mode);
  TORCH_CHECK(
      mode == TreeMaskMode::FULL_MASK || mode == TreeMaskMode::QLEN_ONLY,
      "build_tree_kernel_efficient: unsupported tree_mask_mode ",
      tree_mask_mode,
      " (QLEN_ONLY_BITPACKING is not implemented on XPU)");

  TORCH_CHECK(topk > 0, "build_tree_kernel_efficient: topk must be positive, got ", topk);
  TORCH_CHECK(depth > 0, "build_tree_kernel_efficient: depth must be positive, got ", depth);
  TORCH_CHECK(
      draft_token_num > 0, "build_tree_kernel_efficient: draft_token_num must be positive, got ", draft_token_num);

  TORCH_CHECK(parent_list.scalar_type() == at::kLong, "parent_list must be int64");
  TORCH_CHECK(selected_index.scalar_type() == at::kLong, "selected_index must be int64");
  TORCH_CHECK(tree_mask.scalar_type() == at::kBool, "tree_mask must be bool");
  TORCH_CHECK(positions.scalar_type() == at::kLong, "positions must be int64");
  TORCH_CHECK(retrive_index.scalar_type() == at::kLong, "retrive_index must be int64");
  TORCH_CHECK(retrive_next_token.scalar_type() == at::kLong, "retrive_next_token must be int64");
  TORCH_CHECK(retrive_next_sibling.scalar_type() == at::kLong, "retrive_next_sibling must be int64");

  const int64_t bs = verified_seq_len.numel();
  TORCH_CHECK(
      selected_index.dim() == 2 && selected_index.size(0) == bs,
      "selected_index must be (batch_size, *), got ",
      selected_index.sizes(),
      " for batch_size ",
      bs);
  TORCH_CHECK(
      selected_index.size(1) >= draft_token_num - 1,
      "selected_index must hold at least draft_token_num - 1 = ",
      draft_token_num - 1,
      " entries per request, got ",
      selected_index.size(1));

  for (const auto& out : {positions, retrive_index, retrive_next_token, retrive_next_sibling}) {
    TORCH_CHECK(
        out.numel() == bs * draft_token_num,
        "build_tree_kernel_efficient: output tensors must hold batch_size * draft_token_num = ",
        bs * draft_token_num,
        " elements, got ",
        out.numel());
  }

  // parent_list is (bs, width); organize_draft_results emits (bs, 0) when there
  // are no non-root parents (single-step MTP), which must stay legal.
  int64_t parent_list_stride = 0;
  int64_t parent_list_width = 0;
  if (parent_list.dim() > 1) {
    TORCH_CHECK(
        parent_list.size(0) == bs, "parent_list must be (batch_size, *), got ", parent_list.sizes(), " for bs ", bs);
    parent_list_stride = parent_list.stride(0);
    parent_list_width = parent_list.size(1);
  } else {
    parent_list_stride = parent_list.numel();
    parent_list_width = parent_list.numel();
  }

  const bool full_mask = (mode == TreeMaskMode::FULL_MASK);
  const int64_t expected_mask_numel =
      full_mask ? verified_seq_len.sum().item<int64_t>() * draft_token_num + draft_token_num * draft_token_num * bs
                : bs * draft_token_num * draft_token_num;
  TORCH_CHECK(
      tree_mask.numel() >= expected_mask_numel,
      "build_tree_kernel_efficient: tree_mask needs at least ",
      expected_mask_numel,
      " elements for tree_mask_mode ",
      tree_mask_mode,
      ", got ",
      tree_mask.numel());

  if (bs == 0) {
    return;
  }

  auto& queue = dpcppGetCurrentQueue();
  // One work-group per request; one work-item per draft token (nodes beyond the
  // work-group size are handled by the strided loops in the kernel).
  const int64_t max_wg = dpcppMaxWorkGroupSize();
  const int64_t local_range = std::min<int64_t>(std::max<int64_t>((draft_token_num + 31) / 32 * 32, 32), max_wg);

  AT_DISPATCH_INDEX_TYPES(verified_seq_len.scalar_type(), "build_tree_kernel_efficient", [&] {
    using Kernel = BuildTreeKernel<index_t>;
    Kernel kernel(
        parent_list.data_ptr<int64_t>(),
        selected_index.data_ptr<int64_t>(),
        verified_seq_len.data_ptr<index_t>(),
        tree_mask.data_ptr<bool>(),
        positions.data_ptr<int64_t>(),
        retrive_index.data_ptr<int64_t>(),
        retrive_next_token.data_ptr<int64_t>(),
        retrive_next_sibling.data_ptr<int64_t>(),
        static_cast<int32_t>(topk),
        static_cast<int32_t>(depth),
        static_cast<int32_t>(draft_token_num),
        static_cast<int32_t>(parent_list_width),
        parent_list_stride,
        selected_index.stride(0),
        full_mask);
    sycl_kernel_submit(bs * local_range, local_range, queue, kernel);
  });
}
