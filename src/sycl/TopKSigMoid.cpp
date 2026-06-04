#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <limits>
#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"

namespace at::native::xpu {

namespace TopKSigmoidImpl {

template <typename T>
struct FusedTopkSigmoid {
  static constexpr int sub_group_size = 32;
  static constexpr int max_group_size = 1024;
  static constexpr int malloc_per_item = 8;
  static constexpr float kNegInfinity = -std::numeric_limits<float>::infinity();

  FusedTopkSigmoid(
      float* topk_weights,
      int* topk_ids,
      const T* gating_output,
      const float* correction_bias,
      const bool renormalize,
      const int tokens,
      const int experts,
      const int top_k)
      : topk_weights(topk_weights),
        topk_ids(topk_ids),
        gating_output(gating_output),
        correction_bias(correction_bias),
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

  [[sycl::reqd_sub_group_size(sub_group_size)]]
  void operator()(sycl::nd_item<3> item) const {
    int group_id = item.get_group_linear_id();
    int local_range = item.get_local_range(2);
    int sub_groups_per_group = local_range / sub_group_size;
    int calc_per_item = (experts + sub_group_size - 1) / sub_group_size;

    sycl::sub_group sg = item.get_sub_group();
    int sg_id = sg.get_group_id();
    int sg_local_id = sg.get_local_id();

    int tid = group_id * sub_groups_per_group + sg_id;
    if (tid >= tokens) return;

    // Keep ranking scores in float so bf16/fp16 inputs do not perturb top-k order.
    float local_elems[malloc_per_item];
    int local_idx[malloc_per_item];

    int start_offset = sg_local_id * calc_per_item;
    int local_num = calc_per_item;
    if (start_offset + local_num >= experts) local_num = experts - start_offset;
    if (local_num < 0) local_num = 0;

    for (int e = 0; e < calc_per_item; ++e) {
      local_elems[e] = kNegInfinity;
      local_idx[e] = -1;
    }

    for (int e = 0; e < local_num; ++e) {
      int expert_id = start_offset + e;
      float logit = static_cast<float>(gating_output[tid * experts + expert_id]);

      // sigmoid activation
      float sig = 1.0f / (1.0f + sycl::exp(-logit));

      local_idx[e] = expert_id;

      // add bias for ranking purposes only
      if (correction_bias != nullptr) sig += correction_bias[expert_id];

      local_elems[e] = sig;
    }

    // Top-K selection
    float topk_weights_local[malloc_per_item];
    int topk_ids_local[malloc_per_item];

    for (int k = 0; k < top_k; ++k) {
      float k_max = kNegInfinity;
      int k_max_idx = -1;
      int remove_ix = -1;

      for (int e = 0; e < calc_per_item; ++e) {
        float my_val = local_elems[e];
        int my_idx = local_idx[e];

        // butterfly argmax across sub-group — same as softmax
        for (int offset = sub_group_size / 2; offset > 0; offset /= 2) {
          float other_val = sycl::permute_group_by_xor(sg, my_val, offset);
          int other_idx = sycl::permute_group_by_xor(sg, my_idx, offset);
          if (other_val > my_val || (other_val == my_val && other_idx < my_idx)) {
            my_val = other_val;
            my_idx = other_idx;
          }
        }

        if (my_val > k_max || (my_val == k_max && my_idx < k_max_idx)) {
          k_max = my_val;
          k_max_idx = my_idx;
          remove_ix = (k_max_idx == local_idx[e]) ? e : -1;
        }
      }
      topk_weights_local[k] = k_max;
      topk_ids_local[k] = k_max_idx;

      // owner lane clears winning slot to prevent re-selection
      if (remove_ix != -1) {
        local_elems[remove_ix] = kNegInfinity;
        local_idx[remove_ix] = -1;
      }
    }

    // Reuse the selected ranking scores. When bias participates in ranking,
    // subtract it back out so outputs stay as raw sigmoid weights.
    if (correction_bias != nullptr) {
      for (int i = 0; i < top_k; ++i) {
        int id = topk_ids_local[i];
        topk_weights_local[i] -= correction_bias[id];
      }
    }

    // Optional renormalize on unbiased sigmoid values.
    if (renormalize) {
      float sum = 0.0f;
      for (int i = 0; i < top_k; ++i)
        sum += topk_weights_local[i];
      if (sum > 0.0f)
        for (int i = 0; i < top_k; ++i)
          topk_weights_local[i] /= sum;
    }

    // Write output
    if (sg_local_id == 0) {
      int offset = tid * top_k;
      for (int i = 0; i < top_k; ++i) {
        topk_weights[offset + i] = topk_weights_local[i];
        topk_ids[offset + i] = (topk_ids_local[i] >= 0) ? topk_ids_local[i] : 0;
      }
    }
  }

  float* topk_weights;
  int* topk_ids;
  const T* gating_output;
  const float* correction_bias;
  const bool renormalize;
  const int tokens;
  const int experts;
  const int top_k;
};

template <typename T>
void launch_fused_topk_sigmoid(
    sycl::queue& queue,
    const T* gating_output,
    float* topk_weights,
    int* topk_indices,
    const float* correction_bias,
    const bool renormalize,
    const int top_k,
    const int num_tokens,
    const int num_experts) {
  using Kernel = FusedTopkSigmoid<T>;
  auto range = Kernel::get_nd_range(num_tokens, num_experts);
  Kernel task(topk_weights, topk_indices, gating_output, correction_bias, renormalize, num_tokens, num_experts, top_k);
  sycl_kernel_submit(range.get_global_range(), range.get_local_range(), queue, task);
}

template <typename T>
void fused_topk_sigmoid(
    const T* gating_output,
    float* topk_weights,
    int* topk_indices,
    const float* correction_bias,
    const bool renormalize,
    const int num_tokens,
    const int num_experts,
    const int topk) {
  auto queue = at::xpu::getCurrentXPUStream().queue();
  launch_fused_topk_sigmoid(
      queue, gating_output, topk_weights, topk_indices, correction_bias, renormalize, topk, num_tokens, num_experts);
}
};  // namespace TopKSigmoidImpl

/**
 * @brief Perform topk after sigmoid on gating_output.
 * @param topk_weights The topk_weights tensor of shape [n_tokens, n_topk].
 * @param topk_indices The topk_indices tensor of shape [n_tokens, n_topk].
 * @param gating_output The gating output tensor of shape [n_tokens, n_experts].
 * @param renormalize The renormalize bool whether the topk_weights needs to be renormalized.
 * @param correction_bias The correction_bias tensor of shape [num_experts].
 * @return void.
 */
void topk_sigmoid(
    at::Tensor& topk_weights,
    at::Tensor& topk_indices,
    at::Tensor& gating_output,
    bool renormalize,
    const c10::optional<at::Tensor>& correction_bias) {
  auto shape = gating_output.sizes().vec();
  TORCH_CHECK(shape.size() == 2, "gating_output must be 2D");
  int64_t n_tokens = shape[0];
  int64_t n_experts = shape[1];
  TORCH_CHECK(n_experts <= 256, "n_experts only support up to 256, got ", n_experts);
  TORCH_CHECK(topk_weights.scalar_type() == at::kFloat, "topk_weights should be Float");
  TORCH_CHECK(topk_indices.scalar_type() == at::kInt, "topk_indices should be Int");

  int64_t n_topk = topk_weights.size(1);
  // The max topk value is 8, which is constrained by 'malloc_per_item'.
  auto max_topk = n_experts < 8 ? n_experts : 8;
  TORCH_CHECK(0 < n_topk && n_topk <= max_topk, "topk must be less than or equal to num_experts and 8");

  const float* bias_ptr = nullptr;
  if (correction_bias.has_value()) {
    const auto& bias = correction_bias.value();
    TORCH_CHECK(bias.dim() == 1 && bias.size(0) == n_experts, "correction_bias must be 1D [num_experts]");
    TORCH_CHECK(bias.scalar_type() == at::kFloat, "correction_bias must be float32");
    bias_ptr = bias.data_ptr<float>();
  }

  // Cover Float in addition to the reduced float types (Half, BFloat16): some
  // models (e.g. Nemotron-3-Nano MoE) emit fp32 router gating logits. The kernel
  // already upcasts each element to float internally, so the fp32 path is a
  // no-op cast and adds no numerical change.
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, gating_output.scalar_type(), "fused_topk_sigmoid_kernel", [&]() {
        TopKSigmoidImpl::fused_topk_sigmoid<scalar_t>(
            gating_output.data_ptr<scalar_t>(),
            topk_weights.data_ptr<float>(),
            topk_indices.data_ptr<int>(),
            bias_ptr,
            renormalize,
            n_tokens,
            n_experts,
            n_topk);
      });
}
}  // namespace at::native::xpu
