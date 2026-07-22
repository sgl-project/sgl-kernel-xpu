#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cmath>
#include <cstdint>
#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"

namespace at::native::xpu {

namespace HashTopKImpl {

static constexpr int kWarpSize = 32;
static constexpr int kBlockSize = 128;

static inline float act_sqrt_softplus(float x) {
  const float softplus = sycl::fmax(x, 0.0f) + sycl::log1p(sycl::exp(-sycl::fabs(x)));
  return sycl::sqrt(softplus);
}

template <typename T>
struct HashTopKKernel {
  HashTopKKernel(
      const T* router_logits,
      const int64_t* input_id,
      const int32_t* tid2eid,
      int32_t* topk_ids,
      float* topk_weights,
      uint32_t num_tokens,
      uint32_t num_routed_experts,
      uint32_t tid2eid_rows,
      uint32_t topk_routed,
      uint32_t topk_fused,
      float routed_scaling_factor)
      : router_logits(router_logits),
        input_id(input_id),
        tid2eid(tid2eid),
        topk_ids(topk_ids),
        topk_weights(topk_weights),
        num_tokens(num_tokens),
        num_routed_experts(num_routed_experts),
        tid2eid_rows(tid2eid_rows),
        topk_routed(topk_routed),
        topk_fused(topk_fused),
        routed_scaling_factor(routed_scaling_factor) {}

  [[sycl::reqd_sub_group_size(kWarpSize)]] void operator()(sycl::nd_item<1> item) const {
    const uint32_t local_id = static_cast<uint32_t>(item.get_local_id(0));
    const uint32_t warp_in_block = local_id / static_cast<uint32_t>(kWarpSize);
    const uint32_t lane_id = local_id % static_cast<uint32_t>(kWarpSize);
    const uint32_t warps_per_block = static_cast<uint32_t>(kBlockSize / kWarpSize);
    const uint32_t warp_id = static_cast<uint32_t>(item.get_group(0)) * warps_per_block + warp_in_block;
    if (warp_id >= num_tokens) {
      return;
    }

    int32_t expert_id = 0;
    float routed_weight = 0.0f;
    if (lane_id < topk_routed) {
      const int64_t token_id = input_id[warp_id];
      if (token_id >= 0) {
        const uint32_t token_id_u32 = static_cast<uint32_t>(token_id);
        if (token_id_u32 < tid2eid_rows) {
          const int32_t mapped = tid2eid[token_id_u32 * topk_routed + lane_id];
          if (mapped >= 0 && static_cast<uint32_t>(mapped) < num_routed_experts) {
            expert_id = mapped;
            const float logit = static_cast<float>(router_logits[warp_id * num_routed_experts + expert_id]);
            routed_weight = act_sqrt_softplus(logit);
          }
        }
      }
    }

    const auto sg = item.get_sub_group();
    const float routed_sum = sycl::reduce_over_group(sg, routed_weight, sycl::plus<float>());
    const float inv_routed_sum = (routed_sum > 0.0f) ? (1.0f / routed_sum) : 0.0f;

    if (lane_id < topk_fused) {
      const bool is_shared = lane_id >= topk_routed;
      const uint32_t output_offset = warp_id * topk_fused + lane_id;
      if (is_shared) {
        topk_ids[output_offset] = static_cast<int32_t>(num_routed_experts + lane_id - topk_routed);
        topk_weights[output_offset] = 1.0f / routed_scaling_factor;
      } else {
        topk_ids[output_offset] = expert_id;
        topk_weights[output_offset] = routed_weight * inv_routed_sum;
      }
    }
  }

  const T* router_logits;
  const int64_t* input_id;
  const int32_t* tid2eid;
  int32_t* topk_ids;
  float* topk_weights;
  const uint32_t num_tokens;
  const uint32_t num_routed_experts;
  const uint32_t tid2eid_rows;
  const uint32_t topk_routed;
  const uint32_t topk_fused;
  const float routed_scaling_factor;
};

template <typename T>
void launch_hash_topk(
    sycl::queue& queue,
    const T* router_logits,
    const int64_t* input_id,
    const int32_t* tid2eid,
    int32_t* topk_ids,
    float* topk_weights,
    uint32_t num_tokens,
    uint32_t num_routed_experts,
    uint32_t tid2eid_rows,
    uint32_t topk_routed,
    uint32_t topk_fused,
    float routed_scaling_factor) {
  const uint32_t warps_per_block = static_cast<uint32_t>(kBlockSize / kWarpSize);
  const uint32_t num_blocks = div_up<uint32_t>(num_tokens, warps_per_block);
  const int64_t global_size = static_cast<int64_t>(num_blocks) * kBlockSize;

  HashTopKKernel<T> task(
      router_logits,
      input_id,
      tid2eid,
      topk_ids,
      topk_weights,
      num_tokens,
      num_routed_experts,
      tid2eid_rows,
      topk_routed,
      topk_fused,
      routed_scaling_factor);
  sycl_kernel_submit(global_size, kBlockSize, queue, task);
}

}  // namespace HashTopKImpl

void hash_topk(
    at::Tensor& router_logits,
    at::Tensor& input_id,
    at::Tensor& tid2eid,
    at::Tensor& topk_weights,
    at::Tensor& topk_ids,
    double routed_scaling_factor) {
  TORCH_CHECK(router_logits.dim() == 2, "router_logits must be 2D [num_tokens, num_routed_experts]");
  TORCH_CHECK(input_id.dim() == 1, "input_id must be 1D [num_tokens]");
  TORCH_CHECK(tid2eid.dim() == 2, "tid2eid must be 2D [vocab_size, topk_routed]");
  TORCH_CHECK(topk_weights.dim() == 2, "topk_weights must be 2D [num_tokens, topk_fused]");
  TORCH_CHECK(topk_ids.dim() == 2, "topk_ids must be 2D [num_tokens, topk_fused]");

  TORCH_CHECK(
      router_logits.scalar_type() == at::kFloat || router_logits.scalar_type() == at::kHalf ||
          router_logits.scalar_type() == at::kBFloat16,
      "router_logits must be float32/float16/bfloat16");
  TORCH_CHECK(input_id.scalar_type() == at::kLong, "input_id must be int64");
  TORCH_CHECK(tid2eid.scalar_type() == at::kInt, "tid2eid must be int32");
  TORCH_CHECK(topk_weights.scalar_type() == at::kFloat, "topk_weights must be float32");
  TORCH_CHECK(topk_ids.scalar_type() == at::kInt, "topk_ids must be int32");

  const int64_t num_tokens = router_logits.size(0);
  const int64_t num_routed_experts = router_logits.size(1);
  const int64_t topk_routed = tid2eid.size(1);
  const int64_t topk_fused = topk_weights.size(1);

  if (num_tokens == 0) {
    return;
  }

  auto& queue = dpcppGetCurrentQueue();
  DISPATCH_FLOAT_TYPES(router_logits.scalar_type(), "hash_topk_xpu", [&] {
    HashTopKImpl::launch_hash_topk<scalar_t>(
        queue,
        router_logits.data_ptr<scalar_t>(),
        input_id.data_ptr<int64_t>(),
        tid2eid.data_ptr<int32_t>(),
        topk_ids.data_ptr<int32_t>(),
        topk_weights.data_ptr<float>(),
        static_cast<uint32_t>(num_tokens),
        static_cast<uint32_t>(num_routed_experts),
        static_cast<uint32_t>(tid2eid.size(0)),
        static_cast<uint32_t>(topk_routed),
        static_cast<uint32_t>(topk_fused),
        static_cast<float>(routed_scaling_factor));
  });
}

}  // namespace at::native::xpu
