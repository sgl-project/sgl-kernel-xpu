#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"

namespace at::native::xpu {

namespace TopKSoftmaxImpl {

template <typename T>
struct FusedTopkSoftmax {
  static constexpr int sub_group_size = 32;
  static constexpr int max_group_size = 1024;
  static constexpr int malloc_per_item = 8;
  static constexpr float kNegInfinity = -std::numeric_limits<float>::infinity();

  FusedTopkSoftmax(
      float* topk_weights,
      int* topk_ids,
      const T* gating_output,
      const bool renormalize,
      const int tokens,
      const int experts,
      const int top_k)
      : topk_weights(topk_weights),
        topk_ids(topk_ids),
        gating_output(gating_output),
        renormalize(renormalize),
        tokens(tokens),
        experts(experts),
        top_k(top_k) {}

  static inline sycl::nd_range<3> get_nd_range(const int tokens, const int experts) {
    int calc_per_item = div_up(experts, sub_group_size);
    int group_size = div_up(experts, calc_per_item);
    group_size = group_size < sub_group_size ? sub_group_size : group_size;
    group_size = group_size < max_group_size ? group_size : max_group_size;
    int sub_groups_per_group = div_up(group_size, sub_group_size);
    group_size = sub_groups_per_group * sub_group_size;
    int global_size = div_up(tokens, sub_groups_per_group);

    sycl::range<3> local(1, 1, group_size);
    sycl::range<3> global(1, 1, global_size);
    return sycl::nd_range<3>(global * local, local);
  }

  static inline T Sigmoid(T x) {
    float sycl_x = static_cast<float>(x);
    float result = 1.0f / (1.0f + sycl::exp(-sycl_x));
    return static_cast<T>(result);
  }

  [[sycl::reqd_sub_group_size(sub_group_size)]] void operator()(sycl::nd_item<3> item) const {
    int group_id = item.get_group_linear_id();
    int local_range = item.get_local_range(2);
    int sub_groups_per_group = local_range / sub_group_size;
    int calc_per_item = (experts + sub_group_size - 1) / sub_group_size;

    sycl::sub_group sg = item.get_sub_group();
    int sg_id = sg.get_group_id();
    int sg_local_id = sg.get_local_id();

    int tid = group_id * sub_groups_per_group + sg_id;

    if (tid >= tokens) {
      return;  // Out of bounds
    }

    T local_elems[malloc_per_item];
    int local_idx[malloc_per_item];

    int start_offset = sg_local_id * calc_per_item;
    int local_num = calc_per_item;

    if (start_offset + local_num >= experts) {
      local_num = experts - start_offset;
      if (local_num < 0) {
        local_num = 0;  // No elements to process
      }
    }

    for (int e = 0; e < calc_per_item; ++e) {
      local_elems[e] = kNegInfinity;
      local_idx[e] = -1;
    }

    for (int e = 0; e < local_num; ++e) {
      local_elems[e] = gating_output[tid * experts + start_offset + e];
      local_idx[e] = start_offset + e;
    }

    // Perform top-k selection
    T topk_weights_local[malloc_per_item];
    int topk_ids_local[malloc_per_item];

    for (int k = 0; k < top_k; ++k) {
      T k_max = kNegInfinity;
      int k_max_idx = -1;
      int remove_ix = -1;
      for (int e = 0; e < calc_per_item; ++e) {
        T my_val = local_elems[e];
        int my_idx = local_idx[e];
        for (int offset = sub_group_size / 2; offset > 0; offset /= 2) {
          T other_val = sycl::permute_group_by_xor(sg, my_val, offset);
          int other_idx = sycl::permute_group_by_xor(sg, my_idx, offset);
          if (other_val > my_val || (other_val == my_val && other_idx < my_idx)) {
            my_val = other_val;
            my_idx = other_idx;
          }
        }
        if (my_val > k_max || (my_val == k_max && my_idx < k_max_idx)) {
          k_max = my_val;
          k_max_idx = my_idx;

          if (k_max_idx == local_idx[e]) {
            remove_ix = e;  // Mark this index for removal
          } else
            remove_ix = -1;
        }
      }
      topk_weights_local[k] = k_max;
      topk_ids_local[k] = k_max_idx;
      if (remove_ix != -1) {
        // Reset the score to avoid re-selection
        local_elems[remove_ix] = kNegInfinity;
        local_idx[remove_ix] = -1;
        remove_ix = -1;
      }
    }

    float max_score = topk_weights_local[0];
    float sum_exp = 0;

    for (int i = 0; i < top_k; ++i) {
      float score = topk_weights_local[i];
      sum_exp += sycl::exp(score - max_score);
    }

    for (int e = 0; e < calc_per_item; ++e) {
      float score = local_elems[e];
      float my_val = sycl::exp(score - max_score);
      for (int offset = sub_group_size / 2; offset > 0; offset /= 2) {
        float other_val = sycl::permute_group_by_xor(sg, my_val, offset);
        my_val += other_val;
      }
      sum_exp += my_val;
    }

    for (int i = 0; i < top_k; ++i) {
      float score = topk_weights_local[i];
      topk_weights_local[i] = sycl::exp(score - max_score) / sum_exp;
    }

    if (renormalize) {
      // Renormalize the top-k weights
      float sum = 0;
      for (int i = 0; i < top_k; ++i) {
        sum += topk_weights_local[i];
      }
      if (sum > 0) {
        for (int i = 0; i < top_k; ++i) {
          topk_weights_local[i] /= sum;
        }
      }
    }

    if (sg_local_id == 0) {
      int offset = tid * top_k;
      for (int i = 0; i < top_k; ++i) {
        topk_weights[offset + i] = topk_weights_local[i];
        if (topk_ids_local[i] < 0 || topk_ids_local[i] >= experts) {
          // Ensure valid index
          topk_ids[offset + i] = 0;
          continue;
        }
        topk_ids[offset + i] = topk_ids_local[i];
      }
    }
  }
  float* topk_weights;
  int* topk_ids;
  const T* gating_output;
  const bool renormalize;
  const int tokens;
  const int experts;
  const int top_k;
};

template <typename T>
void launch_fused_topk_softmax(
    sycl::queue& queue,
    const T* gating_output,
    float* topk_weights,
    int* topk_indices,
    const bool renormalize,
    const int top_k,
    const int num_tokens,
    const int num_experts) {
  using Kernel = FusedTopkSoftmax<T>;
  auto range = Kernel::get_nd_range(num_tokens, num_experts);

  auto global_range = range.get_global_range();
  auto local_range = range.get_local_range();

  Kernel task(topk_weights, topk_indices, gating_output, renormalize, num_tokens, num_experts, top_k);

  sycl_kernel_submit(global_range, local_range, queue, task);
  return;
}

template <typename T>
void fused_topk_softmax(
    const T* gating_output,
    float* topk_weights,
    int* topk_indices,
    const bool renormalize,
    const int num_tokens,
    const int num_experts,
    const int topk) {
  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  launch_fused_topk_softmax(
      queue, gating_output, topk_weights, topk_indices, renormalize, topk, num_tokens, num_experts);
};
};  // namespace TopKSoftmaxImpl

/**
 * @brief Perform topk after softmax on gating_output.
 * @param topk_weights The topk_weights tensor of shape [n_tokens, n_topk].
 * @param topk_indices The topk_indices tensor of shape [n_tokens, n_topk].
 * @param gating_output The gating output tensor of shape [n_tokens, n_experts].
 * @param renormalize The renormalize bool whether the topk_weights needs to be renormalized.
 * @return void.
 */
void topk_softmax(at::Tensor& topk_weights, at::Tensor& topk_indices, at::Tensor& gating_output, bool renormalize) {
  auto shape = gating_output.sizes().vec();
  TORCH_CHECK(shape.size() == 2, "gating_output must be 2D tensor, but got ", shape.size(), "D");
  int64_t n_tokens = shape[0];
  int64_t n_experts = shape[1];

  TORCH_CHECK(n_experts <= 128, "n_experts only support up to 128, but got ", n_experts);

  TORCH_CHECK(topk_weights.scalar_type() == at::kFloat, "topk_weights should be Float");
  TORCH_CHECK(topk_indices.scalar_type() == at::kInt, "topk_indices should be Int");

  constexpr int64_t alignment = 8;
  int64_t n_experts_aligned = div_up(n_experts, alignment) * alignment;  // align to 8

  int64_t n_topk = topk_weights.size(1);

  AT_DISPATCH_REDUCED_FLOATING_TYPES(gating_output.scalar_type(), "fused_topk_softmax_kernel", [&]() {
    TopKSoftmaxImpl::fused_topk_softmax<scalar_t>(
        gating_output.data_ptr<scalar_t>(),
        topk_weights.data_ptr<float>(),
        topk_indices.data_ptr<int>(),
        renormalize,
        n_tokens,
        n_experts,
        n_topk);
  });
}
}  // namespace at::native::xpu
