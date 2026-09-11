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

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>
#include <type_traits>

#include "SYCLHelpers.h"
#include "Utils.h"
#include "sgl_kernel_export.h"

namespace {

constexpr int kSubGroupSize = 32;
constexpr int kSubGroupShift = 5;
constexpr int kMaxNodesPerLane = 4;
constexpr int64_t kMaxShuffleNodes = kSubGroupSize * kMaxNodesPerLane;

template <typename in_t, typename out_t, int kNodesPerLane>
struct VerifyTreeGreedySubGroupKernel {
  VerifyTreeGreedySubGroupKernel(
      out_t* predicts,
      out_t* accept_index,
      out_t* accept_token_num,
      const in_t* candidates,
      const in_t* retrive_index,
      const in_t* retrive_next_token,
      const in_t* retrive_next_sibling,
      const in_t* target_predict,
      int32_t num_draft_tokens,
      int32_t num_spec_steps)
      : predicts_(predicts),
        accept_index_(accept_index),
        accept_token_num_(accept_token_num),
        candidates_(candidates),
        retrive_index_(retrive_index),
        retrive_next_token_(retrive_next_token),
        retrive_next_sibling_(retrive_next_sibling),
        target_predict_(target_predict),
        num_draft_tokens_(num_draft_tokens),
        num_spec_steps_(num_spec_steps) {}

  template <typename T>
  static inline T fetch(const sycl::sub_group& sg, const T (&reg)[kNodesPerLane], int32_t node) {
    const sycl::id<1> lane(node & (kSubGroupSize - 1));
    T value = sycl::select_from_group(sg, reg[0], lane);
    if constexpr (kNodesPerLane > 1) {
      const int32_t chunk = node >> kSubGroupShift;
#pragma unroll
      for (int c = 1; c < kNodesPerLane; ++c) {
        const T other = sycl::select_from_group(sg, reg[c], lane);
        if (c == chunk) {
          value = other;
        }
      }
    }
    return value;
  }

  [[sycl::reqd_sub_group_size(kSubGroupSize)]] void operator()(sycl::nd_item<1> item) const {
    const sycl::sub_group sg = item.get_sub_group();
    const int64_t bid = item.get_group(0);
    const int32_t lane = static_cast<int32_t>(sg.get_local_linear_id());
    const int32_t num_nodes = num_draft_tokens_;
    const int64_t row_base = bid * num_nodes;

    in_t cand[kNodesPerLane];
    in_t retr[kNodesPerLane];
    int32_t next_token[kNodesPerLane];
    int32_t next_sibling[kNodesPerLane];

#pragma unroll
    for (int c = 0; c < kNodesPerLane; ++c) {
      const int32_t node = lane + c * kSubGroupSize;
      const int64_t off = row_base + (node < num_nodes ? node : 0);
      cand[c] = candidates_[off];
      retr[c] = retrive_index_[off];
      next_token[c] = static_cast<int32_t>(retrive_next_token_[off]);
      next_sibling[c] = static_cast<int32_t>(retrive_next_sibling_[off]);
    }

    int32_t num_accepted = 0;
    int64_t last_accept_flat = static_cast<int64_t>(fetch(sg, retr, 0));
    if (lane == 0) {
      accept_index_[bid * num_spec_steps_] = static_cast<out_t>(last_accept_flat);
    }

    int32_t cur = 0;
    for (int32_t level = 1; level < num_spec_steps_; ++level) {
      cur = fetch(sg, next_token, cur);
      const in_t target = target_predict_[last_accept_flat];

      int32_t matched_node = -1;
      while (cur >= 0 && cur < num_nodes) {
        if (fetch(sg, cand, cur) == target) {
          matched_node = cur;
          break;
        }
        cur = fetch(sg, next_sibling, cur);
      }
      if (matched_node < 0) {
        break;
      }

      if (lane == 0) {
        predicts_[last_accept_flat] = static_cast<out_t>(target);
      }
      last_accept_flat = static_cast<int64_t>(fetch(sg, retr, matched_node));
      ++num_accepted;
      if (lane == 0) {
        accept_index_[bid * num_spec_steps_ + num_accepted] = static_cast<out_t>(last_accept_flat);
      }
    }

    if (lane == 0) {
      accept_token_num_[bid] = static_cast<out_t>(num_accepted);
      predicts_[last_accept_flat] = static_cast<out_t>(target_predict_[last_accept_flat]);
    }
  }

  out_t* predicts_;
  out_t* accept_index_;
  out_t* accept_token_num_;
  const in_t* candidates_;
  const in_t* retrive_index_;
  const in_t* retrive_next_token_;
  const in_t* retrive_next_sibling_;
  const in_t* target_predict_;
  int32_t num_draft_tokens_;
  int32_t num_spec_steps_;
};

// SLM staging, for trees too large for the register path.
template <typename in_t, typename out_t>
struct VerifyTreeGreedySlmKernel : public __SYCL_KER_CONFIG_CONVENTION__ {
  VerifyTreeGreedySlmKernel(
      out_t* predicts,
      out_t* accept_index,
      out_t* accept_token_num,
      const in_t* candidates,
      const in_t* retrive_index,
      const in_t* retrive_next_token,
      const in_t* retrive_next_sibling,
      const in_t* target_predict,
      int32_t num_draft_tokens,
      int32_t num_spec_steps)
      : predicts_(predicts),
        accept_index_(accept_index),
        accept_token_num_(accept_token_num),
        candidates_(candidates),
        retrive_index_(retrive_index),
        retrive_next_token_(retrive_next_token),
        retrive_next_sibling_(retrive_next_sibling),
        target_predict_(target_predict),
        num_draft_tokens_(num_draft_tokens),
        num_spec_steps_(num_spec_steps) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {
    const auto nodes = sycl::range<1>(num_draft_tokens_);
    cand_ = sycl::local_accessor<in_t, 1>(nodes, cgh);
    retr_ = sycl::local_accessor<in_t, 1>(nodes, cgh);
    next_token_ = sycl::local_accessor<int32_t, 1>(nodes, cgh);
    next_sibling_ = sycl::local_accessor<int32_t, 1>(nodes, cgh);
  }

  void operator()(sycl::nd_item<1> item) const {
    const int64_t bid = item.get_group(0);
    const int32_t tid = static_cast<int32_t>(item.get_local_id(0));
    const int32_t lrange = static_cast<int32_t>(item.get_local_range(0));
    const int32_t num_nodes = num_draft_tokens_;
    const int64_t row_base = bid * num_nodes;

    in_t* cand = cand_.template get_multi_ptr<sycl::access::decorated::no>().get();
    in_t* retr = retr_.template get_multi_ptr<sycl::access::decorated::no>().get();
    int32_t* next_token = next_token_.get_multi_ptr<sycl::access::decorated::no>().get();
    int32_t* next_sibling = next_sibling_.get_multi_ptr<sycl::access::decorated::no>().get();

#pragma unroll 2
    for (int32_t i = tid; i < num_nodes; i += lrange) {
      cand[i] = candidates_[row_base + i];
      retr[i] = retrive_index_[row_base + i];
      next_token[i] = static_cast<int32_t>(retrive_next_token_[row_base + i]);
      next_sibling[i] = static_cast<int32_t>(retrive_next_sibling_[row_base + i]);
    }
    sycl::group_barrier(item.get_group());

    if (tid != 0) {
      return;
    }

    int32_t num_accepted = 0;
    int64_t last_accept_flat = static_cast<int64_t>(retr[0]);
    accept_index_[bid * num_spec_steps_] = static_cast<out_t>(last_accept_flat);

    int32_t cur = 0;
    for (int32_t level = 1; level < num_spec_steps_; ++level) {
      cur = next_token[cur];
      const in_t target = target_predict_[last_accept_flat];

      bool matched = false;
      while (cur >= 0 && cur < num_nodes) {
        if (cand[cur] == target) {
          predicts_[last_accept_flat] = static_cast<out_t>(target);
          last_accept_flat = static_cast<int64_t>(retr[cur]);
          ++num_accepted;
          accept_index_[bid * num_spec_steps_ + num_accepted] = static_cast<out_t>(last_accept_flat);
          matched = true;
          break;
        }
        cur = next_sibling[cur];
      }
      if (!matched) {
        break;
      }
    }

    accept_token_num_[bid] = static_cast<out_t>(num_accepted);
    predicts_[last_accept_flat] = static_cast<out_t>(target_predict_[last_accept_flat]);
  }

  out_t* predicts_;
  out_t* accept_index_;
  out_t* accept_token_num_;
  const in_t* candidates_;
  const in_t* retrive_index_;
  const in_t* retrive_next_token_;
  const in_t* retrive_next_sibling_;
  const in_t* target_predict_;
  int32_t num_draft_tokens_;
  int32_t num_spec_steps_;

  sycl::local_accessor<in_t, 1> cand_;
  sycl::local_accessor<in_t, 1> retr_;
  sycl::local_accessor<int32_t, 1> next_token_;
  sycl::local_accessor<int32_t, 1> next_sibling_;
};

template <typename in_t, typename out_t>
void launch_verify_tree_greedy(
    out_t* predicts,
    out_t* accept_index,
    out_t* accept_token_num,
    const in_t* candidates,
    const in_t* retrive_index,
    const in_t* retrive_next_token,
    const in_t* retrive_next_sibling,
    const in_t* target_predict,
    int64_t bs,
    int64_t num_draft_tokens,
    int64_t num_spec_steps) {
  auto& queue = dpcppGetCurrentQueue();
  const int32_t nodes = static_cast<int32_t>(num_draft_tokens);
  const int32_t steps = static_cast<int32_t>(num_spec_steps);

  if (num_draft_tokens <= kMaxShuffleNodes) {
    // One sub-group per request, so the group is exactly one sub-group wide and
    auto submit = [&](auto nodes_per_lane_tag) {
      constexpr int kNodesPerLane = decltype(nodes_per_lane_tag)::value;
      VerifyTreeGreedySubGroupKernel<in_t, out_t, kNodesPerLane> kernel(
          predicts,
          accept_index,
          accept_token_num,
          candidates,
          retrive_index,
          retrive_next_token,
          retrive_next_sibling,
          target_predict,
          nodes,
          steps);
      sycl_kernel_submit(bs * kSubGroupSize, static_cast<int64_t>(kSubGroupSize), queue, kernel);
    };
    switch (static_cast<int>((num_draft_tokens + kSubGroupSize - 1) / kSubGroupSize)) {
      case 1:
        submit(std::integral_constant<int, 1>{});
        break;
      case 2:
        submit(std::integral_constant<int, 2>{});
        break;
      case 3:
        submit(std::integral_constant<int, 3>{});
        break;
      default:
        submit(std::integral_constant<int, 4>{});
        break;
    }
    return;
  }

  const int64_t max_wg = dpcppMaxWorkGroupSize();
  const int64_t local_range = std::min<int64_t>(
      std::max<int64_t>((num_draft_tokens + kSubGroupSize - 1) / kSubGroupSize * kSubGroupSize, kSubGroupSize), max_wg);

  VerifyTreeGreedySlmKernel<in_t, out_t> kernel(
      predicts,
      accept_index,
      accept_token_num,
      candidates,
      retrive_index,
      retrive_next_token,
      retrive_next_sibling,
      target_predict,
      nodes,
      steps);
  sycl_kernel_submit(bs * local_range, local_range, queue, kernel);
}

}  // namespace

SGL_KERNEL_EXPORT void verify_tree_greedy(
    at::Tensor& predicts,
    at::Tensor& accept_index,
    at::Tensor& accept_token_num,
    const at::Tensor& candidates,
    const at::Tensor& retrive_index,
    const at::Tensor& retrive_next_token,
    const at::Tensor& retrive_next_sibling,
    const at::Tensor& target_predict) {
  CHECK_INPUT(predicts);
  CHECK_INPUT(accept_index);
  CHECK_INPUT(accept_token_num);
  CHECK_INPUT(candidates);
  CHECK_INPUT(retrive_index);
  CHECK_INPUT(retrive_next_token);
  CHECK_INPUT(retrive_next_sibling);
  CHECK_INPUT(target_predict);

  TORCH_CHECK(
      candidates.dim() == 2,
      "verify_tree_greedy: candidates must be (batch_size, num_draft_tokens), got ",
      candidates.sizes());
  TORCH_CHECK(
      accept_index.dim() == 2,
      "verify_tree_greedy: accept_index must be (batch_size, num_spec_steps), got ",
      accept_index.sizes());

  const int64_t bs = candidates.size(0);
  const int64_t num_draft_tokens = candidates.size(1);
  const int64_t num_spec_steps = accept_index.size(1);

  TORCH_CHECK(num_draft_tokens > 0, "verify_tree_greedy: num_draft_tokens must be positive");
  TORCH_CHECK(num_spec_steps > 0, "verify_tree_greedy: num_spec_steps must be positive");
  TORCH_CHECK(
      accept_index.size(0) == bs,
      "verify_tree_greedy: accept_index batch ",
      accept_index.size(0),
      " does not match candidates batch ",
      bs);
  TORCH_CHECK(
      accept_token_num.numel() == bs,
      "verify_tree_greedy: accept_token_num must hold ",
      bs,
      " elements, got ",
      accept_token_num.numel());

  for (const auto& t : {retrive_index, retrive_next_token, retrive_next_sibling, target_predict}) {
    TORCH_CHECK(
        t.dim() == 2 && t.size(0) == bs && t.size(1) == num_draft_tokens,
        "verify_tree_greedy: tree inputs must all be (",
        bs,
        ", ",
        num_draft_tokens,
        "), got ",
        t.sizes());
  }
  TORCH_CHECK(
      predicts.numel() >= bs * num_draft_tokens,
      "verify_tree_greedy: predicts must hold at least ",
      bs * num_draft_tokens,
      " elements, got ",
      predicts.numel());

  // The traversal compares candidates against target_predict and indexes with
  // retrive_*, so the tree side must share one integer type; the three outputs
  // share another.
  const auto in_type = candidates.scalar_type();
  const auto out_type = predicts.scalar_type();
  TORCH_CHECK(
      in_type == at::kInt || in_type == at::kLong,
      "verify_tree_greedy: candidates must be int32 or int64, got ",
      in_type);
  TORCH_CHECK(
      out_type == at::kInt || out_type == at::kLong,
      "verify_tree_greedy: predicts must be int32 or int64, got ",
      out_type);
  for (const auto& t : {retrive_index, retrive_next_token, retrive_next_sibling, target_predict}) {
    TORCH_CHECK(
        t.scalar_type() == in_type,
        "verify_tree_greedy: all tree inputs must share candidates' dtype ",
        in_type,
        ", got ",
        t.scalar_type());
  }
  for (const auto& t : {accept_index, accept_token_num}) {
    TORCH_CHECK(
        t.scalar_type() == out_type,
        "verify_tree_greedy: all outputs must share predicts' dtype ",
        out_type,
        ", got ",
        t.scalar_type());
  }

  if (bs == 0) {
    return;
  }

  auto launch = [&](auto in_tag, auto out_tag) {
    using in_t = decltype(in_tag);
    using out_t = decltype(out_tag);
    launch_verify_tree_greedy<in_t, out_t>(
        predicts.data_ptr<out_t>(),
        accept_index.data_ptr<out_t>(),
        accept_token_num.data_ptr<out_t>(),
        candidates.data_ptr<in_t>(),
        retrive_index.data_ptr<in_t>(),
        retrive_next_token.data_ptr<in_t>(),
        retrive_next_sibling.data_ptr<in_t>(),
        target_predict.data_ptr<in_t>(),
        bs,
        num_draft_tokens,
        num_spec_steps);
  };

  if (in_type == at::kLong) {
    if (out_type == at::kLong) {
      launch(int64_t{}, int64_t{});
    } else {
      launch(int64_t{}, int32_t{});
    }
  } else {
    if (out_type == at::kLong) {
      launch(int32_t{}, int64_t{});
    } else {
      launch(int32_t{}, int32_t{});
    }
  }
}
