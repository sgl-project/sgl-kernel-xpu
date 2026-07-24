
#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cfloat>
#include <cstdint>
#include <limits>
#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"

namespace at::native::xpu {

namespace {

constexpr uint32_t kWarpSize = 32;
constexpr uint32_t kWarpsPerCTA = 6;
constexpr uint32_t kSmallTokenThreshold = 512;
constexpr uint32_t kMaxExperts = 512;
constexpr uint32_t kMaxTopK = 16;

enum class ScoringFunc : uint32_t {
  kSigmoid = 0,
  kSqrtSoftplus = 1,
};

template <ScoringFunc kScoringFunc>
static inline float compute_score(float x) {
  if constexpr (kScoringFunc == ScoringFunc::kSigmoid) {
    return 1.0f / (1.0f + sycl::native::exp(-x));
  } else {
    return sycl::native::sqrt(sycl::log1p(sycl::native::exp(x)));
  }
}

template <typename T, ScoringFunc kScoringFunc>
struct BiasedTopkKernel : public __SYCL_KER_CONFIG_CONVENTION__ {
  const T* __restrict__ input_;
  const float* __restrict__ bias_;
  float* __restrict__ output_;
  int32_t* __restrict__ indices_;
  uint32_t num_rows_;
  uint32_t num_experts_;
  uint32_t topk_;
  uint32_t num_fused_shared_experts_;
  bool renormalize_;
  float routed_scaling_factor_;
  bool apply_routed_scaling_factor_on_output_;
  uint32_t warps_per_cta_;

  sycl::local_accessor<float, 1> scores_;
  sycl::local_accessor<float, 1> original_scores_;
  sycl::local_accessor<int, 1> selected_experts_;

  BiasedTopkKernel(
      const T* input,
      const float* bias,
      float* output,
      int32_t* indices,
      uint32_t num_rows,
      uint32_t num_experts,
      uint32_t topk,
      uint32_t num_fused_shared_experts,
      bool renormalize,
      float routed_scaling_factor,
      bool apply_routed_scaling_factor_on_output,
      uint32_t warps_per_cta)
      : input_(input),
        bias_(bias),
        output_(output),
        indices_(indices),
        num_rows_(num_rows),
        num_experts_(num_experts),
        topk_(topk),
        num_fused_shared_experts_(num_fused_shared_experts),
        renormalize_(renormalize),
        routed_scaling_factor_(routed_scaling_factor),
        apply_routed_scaling_factor_on_output_(apply_routed_scaling_factor_on_output),
        warps_per_cta_(warps_per_cta) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {
    scores_ = sycl::local_accessor<float, 1>(sycl::range<1>(warps_per_cta_ * num_experts_), cgh);
    original_scores_ = sycl::local_accessor<float, 1>(sycl::range<1>(warps_per_cta_ * num_experts_), cgh);
    selected_experts_ = sycl::local_accessor<int, 1>(sycl::range<1>(warps_per_cta_ * kMaxTopK), cgh);
  }

  [[sycl::reqd_sub_group_size(kWarpSize)]] void operator()(sycl::nd_item<3> item) const {
    const uint32_t lane_id = item.get_local_id(2);
    const uint32_t warp_id = item.get_local_id(1);
    const uint32_t row_idx = item.get_group(1) * warps_per_cta_ + warp_id;

    if (row_idx >= num_rows_) {
      return;
    }

    auto sg = item.get_sub_group();
    float* shared_scores = scores_.get_multi_ptr<sycl::access::decorated::no>().get() + warp_id * num_experts_;
    float* shared_original_scores =
        original_scores_.get_multi_ptr<sycl::access::decorated::no>().get() + warp_id * num_experts_;
    int* warp_selected_experts =
        selected_experts_.get_multi_ptr<sycl::access::decorated::no>().get() + warp_id * kMaxTopK;

    for (uint32_t e = lane_id; e < num_experts_; e += kWarpSize) {
      const float input_val = static_cast<float>(input_[static_cast<int64_t>(row_idx) * num_experts_ + e]);
      float bias_val = bias_[e];
      float score_val = compute_score<kScoringFunc>(input_val);
      shared_scores[e] = score_val + bias_val;
      shared_original_scores[e] = score_val;
    }

    const uint32_t topk_routed = topk_ - num_fused_shared_experts_;
    for (uint32_t k = 0; k < topk_routed; ++k) {
      float max_val = -FLT_MAX;
      int max_expert = -1;

      for (uint32_t e = lane_id; e < num_experts_; e += kWarpSize) {
        float val = shared_scores[e];
        if (val > max_val) {
          max_val = val;
          max_expert = static_cast<int>(e);
        }
      }

#pragma unroll
      for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        float other_val = sycl::permute_group_by_xor(sg, max_val, offset);
        int other_expert = sycl::permute_group_by_xor(sg, max_expert, offset);
        if (other_val > max_val || (other_val == max_val && other_expert < max_expert)) {
          max_val = other_val;
          max_expert = other_expert;
        }
      }

      if (lane_id == 0) {
        warp_selected_experts[k] = max_expert;
        if (max_expert >= 0) {
          shared_scores[max_expert] = -FLT_MAX;
        }
      }
      sycl::group_barrier(sg, sycl::memory_scope::sub_group);
    }

    float routed_weight = 0.0f;
    int32_t selected_expert = 0;
    if (lane_id < topk_routed) {
      int expert_id = warp_selected_experts[lane_id];
      if (expert_id >= 0 && expert_id < static_cast<int>(num_experts_)) {
        routed_weight = shared_original_scores[expert_id];
        selected_expert = static_cast<int32_t>(expert_id);
      }
    }

    const float routed_sum = sycl::reduce_over_group(sg, routed_weight, sycl::plus<float>());
    if (lane_id < topk_) {
      const bool is_shared = lane_id >= topk_routed;
      const int64_t output_idx = static_cast<int64_t>(row_idx) * topk_ + lane_id;

      const float weight = is_shared ? (routed_sum / routed_scaling_factor_) : routed_weight;
      const int32_t expert_id =
          is_shared ? static_cast<int32_t>(num_experts_ + lane_id - topk_routed) : selected_expert;

      const float scale = apply_routed_scaling_factor_on_output_ ? routed_scaling_factor_ : 1.0f;
      const float norm = renormalize_ && routed_sum > 0.0f ? routed_sum : 1.0f;

      output_[output_idx] = (weight / norm) * scale;
      indices_[output_idx] = expert_id;
    }
  }
};

template <typename T, ScoringFunc kScoringFunc, uint32_t kWarpsPerToken>
struct BiasedTopkSmallTokenKernel : public __SYCL_KER_CONFIG_CONVENTION__ {
  const T* __restrict__ input_;
  const float* __restrict__ bias_;
  float* __restrict__ output_;
  int32_t* __restrict__ indices_;
  uint32_t num_rows_;
  uint32_t num_experts_;
  uint32_t topk_;
  uint32_t num_fused_shared_experts_;
  bool renormalize_;
  float routed_scaling_factor_;
  bool apply_routed_scaling_factor_on_output_;

  sycl::local_accessor<float, 1> scores_;
  sycl::local_accessor<float, 1> original_scores_;
  sycl::local_accessor<float, 1> warp_maxs_;
  sycl::local_accessor<int, 1> warp_experts_;
  sycl::local_accessor<int, 1> selected_experts_;

  BiasedTopkSmallTokenKernel(
      const T* input,
      const float* bias,
      float* output,
      int32_t* indices,
      uint32_t num_rows,
      uint32_t num_experts,
      uint32_t topk,
      uint32_t num_fused_shared_experts,
      bool renormalize,
      float routed_scaling_factor,
      bool apply_routed_scaling_factor_on_output)
      : input_(input),
        bias_(bias),
        output_(output),
        indices_(indices),
        num_rows_(num_rows),
        num_experts_(num_experts),
        topk_(topk),
        num_fused_shared_experts_(num_fused_shared_experts),
        renormalize_(renormalize),
        routed_scaling_factor_(routed_scaling_factor),
        apply_routed_scaling_factor_on_output_(apply_routed_scaling_factor_on_output) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {
    scores_ = sycl::local_accessor<float, 1>(sycl::range<1>(num_experts_), cgh);
    original_scores_ = sycl::local_accessor<float, 1>(sycl::range<1>(num_experts_), cgh);
    warp_maxs_ = sycl::local_accessor<float, 1>(sycl::range<1>(kWarpsPerToken), cgh);
    warp_experts_ = sycl::local_accessor<int, 1>(sycl::range<1>(kWarpsPerToken), cgh);
    selected_experts_ = sycl::local_accessor<int, 1>(sycl::range<1>(kMaxTopK), cgh);
  }

  [[sycl::reqd_sub_group_size(kWarpSize)]] void operator()(sycl::nd_item<3> item) const {
    const uint32_t lane_id = item.get_local_id(2);
    const uint32_t warp_id = item.get_local_id(1);
    const uint32_t tid = warp_id * kWarpSize + lane_id;
    const uint32_t threads_per_block = kWarpsPerToken * kWarpSize;
    const uint32_t row_idx = item.get_group(1);

    if (row_idx >= num_rows_) {
      return;
    }

    auto sg = item.get_sub_group();
    float* shared_scores = scores_.get_multi_ptr<sycl::access::decorated::no>().get();
    float* shared_original_scores = original_scores_.get_multi_ptr<sycl::access::decorated::no>().get();
    float* warp_maxs = warp_maxs_.get_multi_ptr<sycl::access::decorated::no>().get();
    int* warp_experts = warp_experts_.get_multi_ptr<sycl::access::decorated::no>().get();
    int* selected_experts = selected_experts_.get_multi_ptr<sycl::access::decorated::no>().get();

    for (uint32_t e = tid; e < num_experts_; e += threads_per_block) {
      const float input_val = static_cast<float>(input_[static_cast<int64_t>(row_idx) * num_experts_ + e]);
      float bias_val = bias_[e];
      float score_val = compute_score<kScoringFunc>(input_val);
      shared_scores[e] = score_val + bias_val;
      shared_original_scores[e] = score_val;
    }
    sycl::group_barrier(item.get_group(), sycl::memory_scope::work_group);

    const uint32_t topk_routed = topk_ - num_fused_shared_experts_;
    for (uint32_t k = 0; k < topk_routed; ++k) {
      float max_val = -FLT_MAX;
      int max_expert = -1;
      for (uint32_t e = tid; e < num_experts_; e += threads_per_block) {
        float val = shared_scores[e];
        if (val > max_val) {
          max_val = val;
          max_expert = static_cast<int>(e);
        }
      }

#pragma unroll
      for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        float other_val = sycl::permute_group_by_xor(sg, max_val, offset);
        int other_expert = sycl::permute_group_by_xor(sg, max_expert, offset);
        if (other_val > max_val || (other_val == max_val && other_expert < max_expert)) {
          max_val = other_val;
          max_expert = other_expert;
        }
      }

      if (lane_id == 0) {
        warp_maxs[warp_id] = max_val;
        warp_experts[warp_id] = max_expert;
      }
      sycl::group_barrier(item.get_group(), sycl::memory_scope::work_group);

      if (warp_id == 0) {
        float block_max = lane_id < kWarpsPerToken ? warp_maxs[lane_id] : -FLT_MAX;
        int block_expert = lane_id < kWarpsPerToken ? warp_experts[lane_id] : -1;

#pragma unroll
        for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
          float other_val = sycl::permute_group_by_xor(sg, block_max, offset);
          int other_expert = sycl::permute_group_by_xor(sg, block_expert, offset);
          if (other_val > block_max || (other_val == block_max && other_expert < block_expert)) {
            block_max = other_val;
            block_expert = other_expert;
          }
        }

        if (lane_id == 0) {
          selected_experts[k] = block_expert;
          if (block_expert >= 0) {
            shared_scores[block_expert] = -FLT_MAX;
          }
        }
      }
      sycl::group_barrier(item.get_group(), sycl::memory_scope::work_group);
    }

    float routed_weight = 0.0f;
    int32_t selected_expert = 0;
    if (tid < topk_routed) {
      int expert_id = selected_experts[tid];
      if (expert_id >= 0 && expert_id < static_cast<int>(num_experts_)) {
        routed_weight = shared_original_scores[expert_id];
        selected_expert = static_cast<int32_t>(expert_id);
      }
    }

    const float routed_sum = sycl::reduce_over_group(item.get_group(), routed_weight, sycl::plus<float>());
    if (tid < topk_) {
      const bool is_shared = tid >= topk_routed;
      const int64_t output_idx = static_cast<int64_t>(row_idx) * topk_ + tid;

      const float weight = is_shared ? (routed_sum / routed_scaling_factor_) : routed_weight;
      const int32_t expert_id = is_shared ? static_cast<int32_t>(num_experts_ + tid - topk_routed) : selected_expert;

      const float scale = apply_routed_scaling_factor_on_output_ ? routed_scaling_factor_ : 1.0f;
      const float norm = renormalize_ && routed_sum > 0.0f ? routed_sum : 1.0f;

      output_[output_idx] = (weight / norm) * scale;
      indices_[output_idx] = expert_id;
    }
  }
};

template <typename T, ScoringFunc kScoringFunc, uint32_t kWarpsPerToken>
void launch_biased_topk_small_token_kernel(
    const at::Tensor& input,
    const at::Tensor& bias,
    at::Tensor& output,
    at::Tensor& indices,
    uint32_t topk,
    uint32_t num_fused_shared_experts,
    bool renormalize,
    float routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output,
    sycl::queue& queue) {
  const uint32_t num_rows = static_cast<uint32_t>(input.size(0));
  const uint32_t num_experts = static_cast<uint32_t>(input.size(1));

  sycl::range<3> local_range{1, kWarpsPerToken, kWarpSize};
  sycl::range<3> global_range{1, num_rows * kWarpsPerToken, kWarpSize};

  auto* input_ptr = reinterpret_cast<const T*>(input.data_ptr());
  auto* bias_ptr = reinterpret_cast<const float*>(bias.data_ptr());
  auto* output_ptr = reinterpret_cast<float*>(output.data_ptr());
  auto* indices_ptr = reinterpret_cast<int32_t*>(indices.data_ptr());

  BiasedTopkSmallTokenKernel<T, kScoringFunc, kWarpsPerToken> task(
      input_ptr,
      bias_ptr,
      output_ptr,
      indices_ptr,
      num_rows,
      num_experts,
      topk,
      num_fused_shared_experts,
      renormalize,
      routed_scaling_factor,
      apply_routed_scaling_factor_on_output);

  sycl_kernel_submit(global_range, local_range, queue, task);
}

template <typename T, ScoringFunc kScoringFunc>
void launch_biased_topk(
    const at::Tensor& input,
    const at::Tensor& bias,
    at::Tensor& output,
    at::Tensor& indices,
    uint32_t topk,
    uint32_t num_fused_shared_experts,
    bool renormalize,
    float routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output,
    sycl::queue& queue) {
  const uint32_t num_rows = static_cast<uint32_t>(input.size(0));
  const uint32_t num_experts = static_cast<uint32_t>(input.size(1));

  const uint32_t warps_per_token = std::min<uint32_t>((num_experts + kWarpSize - 1) / kWarpSize, 16u);
  const bool use_small_token_kernel = num_rows <= kSmallTokenThreshold;
  if (use_small_token_kernel) {
    if (warps_per_token <= 8u) {
      launch_biased_topk_small_token_kernel<T, kScoringFunc, 8>(
          input,
          bias,
          output,
          indices,
          topk,
          num_fused_shared_experts,
          renormalize,
          routed_scaling_factor,
          apply_routed_scaling_factor_on_output,
          queue);
    } else if (warps_per_token <= 12u) {
      launch_biased_topk_small_token_kernel<T, kScoringFunc, 12>(
          input,
          bias,
          output,
          indices,
          topk,
          num_fused_shared_experts,
          renormalize,
          routed_scaling_factor,
          apply_routed_scaling_factor_on_output,
          queue);
    } else {
      launch_biased_topk_small_token_kernel<T, kScoringFunc, 16>(
          input,
          bias,
          output,
          indices,
          topk,
          num_fused_shared_experts,
          renormalize,
          routed_scaling_factor,
          apply_routed_scaling_factor_on_output,
          queue);
    }
    return;
  }

  const uint32_t warps_per_cta = kWarpsPerCTA;

  const uint32_t num_blocks = (num_rows + warps_per_cta - 1) / warps_per_cta;
  sycl::range<3> local_range{1, warps_per_cta, kWarpSize};
  sycl::range<3> global_range{1, num_blocks * warps_per_cta, kWarpSize};

  auto* input_ptr = reinterpret_cast<const T*>(input.data_ptr());
  auto* bias_ptr = reinterpret_cast<const float*>(bias.data_ptr());
  auto* output_ptr = reinterpret_cast<float*>(output.data_ptr());
  auto* indices_ptr = reinterpret_cast<int32_t*>(indices.data_ptr());

  BiasedTopkKernel<T, kScoringFunc> task(
      input_ptr,
      bias_ptr,
      output_ptr,
      indices_ptr,
      num_rows,
      num_experts,
      topk,
      num_fused_shared_experts,
      renormalize,
      routed_scaling_factor,
      apply_routed_scaling_factor_on_output,
      warps_per_cta);

  sycl_kernel_submit(global_range, local_range, queue, task);
}
}  // namespace

void biased_topk(
    const at::Tensor& input,
    const at::Tensor& bias,
    at::Tensor& output,
    at::Tensor& indices,
    int64_t topk,
    int64_t scoring_func,
    int64_t num_fused_shared_experts,
    bool renormalize,
    double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
  TORCH_CHECK(input.dim() == 2, "input must be 2D, got ", input.dim(), "D");
  TORCH_CHECK(bias.dim() == 1, "bias must be 1D, got ", bias.dim(), "D");
  TORCH_CHECK(input.size(1) == bias.size(0), "input.size(1) must match bias.size(0)");
  TORCH_CHECK(
      input.scalar_type() == torch::kFloat32 || input.scalar_type() == torch::kFloat16 ||
          input.scalar_type() == torch::kBFloat16,
      "input must be float32, float16, or bfloat16");
  TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "bias must be float32");
  TORCH_CHECK(input.size(1) <= kMaxExperts, "num_experts exceeds maximum supported value: ", kMaxExperts);
  TORCH_CHECK(topk > num_fused_shared_experts, "topk must be greater than num_fused_shared_experts");
  TORCH_CHECK(topk <= kMaxTopK, "topk exceeds maximum supported value: ", kMaxTopK);
  TORCH_CHECK(scoring_func == 0 || scoring_func == 1, "scoring_func must be 0 (sigmoid) or 1 (sqrtsoftplus)");
  TORCH_CHECK(output.scalar_type() == torch::kFloat32, "output must be float32");
  TORCH_CHECK(indices.scalar_type() == torch::kInt32, "indices must be int32");
  TORCH_CHECK(output.dim() == 2, "output must be 2D, got ", output.dim(), "D");
  TORCH_CHECK(indices.dim() == 2, "indices must be 2D, got ", indices.dim(), "D");
  TORCH_CHECK(output.size(0) == input.size(0) && output.size(1) == topk, "output shape mismatch");
  TORCH_CHECK(indices.size(0) == input.size(0) && indices.size(1) == topk, "indices shape mismatch");

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  if (scoring_func == static_cast<int64_t>(ScoringFunc::kSigmoid)) {
    DISPATCH_FLOAT_TYPES(input.scalar_type(), "biased_topk_kernel", [&] {
      launch_biased_topk<scalar_t, ScoringFunc::kSigmoid>(
          input,
          bias,
          output,
          indices,
          static_cast<uint32_t>(topk),
          static_cast<uint32_t>(num_fused_shared_experts),
          renormalize,
          static_cast<float>(routed_scaling_factor),
          apply_routed_scaling_factor_on_output,
          queue);
    });
  } else {
    DISPATCH_FLOAT_TYPES(input.scalar_type(), "biased_topk_kernel", [&] {
      launch_biased_topk<scalar_t, ScoringFunc::kSqrtSoftplus>(
          input,
          bias,
          output,
          indices,
          static_cast<uint32_t>(topk),
          static_cast<uint32_t>(num_fused_shared_experts),
          renormalize,
          static_cast<float>(routed_scaling_factor),
          apply_routed_scaling_factor_on_output,
          queue);
    });
  }
}

}  // namespace at::native::xpu
