#include "sgl_kernel/xe35/routing_quant.h"

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdint>
#include <limits>
#include <sycl/ext/oneapi/sub_group_mask.hpp>
#include <sycl/sycl.hpp>

#include "sycl/SYCLHelpers.h"
#include "sycl/Utils.h"

#include "sgl_kernel_ops.h"

namespace at::native::xpu {
namespace TopKSigmoidImpl_xe35 {
template <typename T>
struct LogitBits_xe35 {
  using Raw = uint16_t;
  static float to_float(Raw bits) {
    return static_cast<float>(sycl::bit_cast<T>(bits));
  }
};

template <>
struct LogitBits_xe35<float> {
  using Raw = float;
  static float to_float(Raw value) {
    return value;
  }
};

// Runtime routing parameters shared by every supported expert count and top-k.
struct RoutingArguments_xe35 {
  float* weights;
  int* ids;
  float const* correction_bias;     // null for no bias
  int const* local_expert_mapping;  // null when a rank owns every expert
  int const* num_valid_tokens;      // device scalar; null when every row is real
  int tokens;
  int experts;
  int topk;
  float router_multiplier;
  float route_scale;
  float division_eps;
  bool renormalize;
};

// Global -> local expert id, for the ranks that own a slice of the experts.
inline int map_expert_xe35(RoutingArguments_xe35 const& args, int expert) {
  return args.local_expert_mapping == nullptr ? expert : args.local_expert_mapping[expert];
}

// A row past the real token count is padding: emit the sentinel id -1, which
// matches no expert, so the MoE backend skips it.
inline bool row_is_padding_xe35(RoutingArguments_xe35 const& args, int row) {
  return args.num_valid_tokens != nullptr && row >= *args.num_valid_tokens;
}

// A full 32-lane tile can load its logits and bias as vectors. This selection
// algorithm applies to every supported full tile width when top-k is eight;
// the tail-safe kernel below handles all other expert counts and top-k values.
template <typename T, int kValuesPerLane>
struct FullTileTopk8_xe35 {
  static constexpr int kSubgroup = 32;
  static constexpr int kTopK = 8;
  static constexpr int kSubgroupsPerGroup = 8;
  static constexpr int kExperts = kSubgroup * kValuesPerLane;
  using Raw = typename LogitBits_xe35<T>::Raw;
  using LogitVec = sycl::vec<Raw, kValuesPerLane>;
  using BiasVec = sycl::vec<float, kValuesPerLane>;
  static_assert(sizeof(T) == sizeof(Raw));

  T const* input;
  RoutingArguments_xe35 args;

  [[sycl::reqd_sub_group_size(kSubgroup)]]
  void operator()(sycl::nd_item<1> item) const {
    constexpr float kInfinity = std::numeric_limits<float>::infinity();
    sycl::sub_group subgroup = item.get_sub_group();
    int lane = static_cast<int>(subgroup.get_local_linear_id());
    int row =
        static_cast<int>(item.get_group(0)) * kSubgroupsPerGroup + static_cast<int>(subgroup.get_group_linear_id());
    if (row >= args.tokens) return;
    int out = row * kTopK + lane;
    if (row_is_padding_xe35(args, row)) {
      if (lane < kTopK) {
        args.ids[out] = -1;
        args.weights[out] = 0.0f;
      }
      return;
    }

    auto const* row_ptr = reinterpret_cast<Raw const*>(input) + static_cast<size_t>(row) * kExperts;
    LogitVec raw = *reinterpret_cast<LogitVec const*>(row_ptr + lane * kValuesPerLane);
    float bias[kValuesPerLane];
    if (args.correction_bias != nullptr) {
      BiasVec bias_vec = *reinterpret_cast<BiasVec const*>(args.correction_bias + lane * kValuesPerLane);
#pragma unroll
      for (int i = 0; i < kValuesPerLane; ++i) {
        bias[i] = bias_vec[i];
      }
    } else {
#pragma unroll
      for (int i = 0; i < kValuesPerLane; ++i) {
        bias[i] = 0.0f;
      }
    }

    float values[kValuesPerLane];
#pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i) {
      float logit = LogitBits_xe35<T>::to_float(raw[i]);
      float scaled = static_cast<float>(static_cast<T>(logit * args.router_multiplier));
      values[i] = 1.0f / (1.0f + sycl::exp(-scaled)) + bias[i];
    }

    int selected_id = -1;
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      float lane_value = -kInfinity;
      int lane_slot = 0;
#pragma unroll
      for (int i = 0; i < kValuesPerLane; ++i) {
        bool greater = values[i] > lane_value;
        lane_value = greater ? values[i] : lane_value;
        lane_slot = greater ? i : lane_slot;
      }
      float group_value = sycl::reduce_over_group(subgroup, lane_value, sycl::maximum<float>());
      std::uint32_t holders = 0;
      sycl::ext::oneapi::group_ballot(subgroup, lane_value == group_value).extract_bits(holders);
      int winner_lane = __builtin_ctz(holders);
      int winner_slot = sycl::select_from_group(subgroup, lane_slot, winner_lane);
      if (lane == k) selected_id = winner_lane * kValuesPerLane + winner_slot;
      if (lane == winner_lane) {
#pragma unroll
        for (int i = 0; i < kValuesPerLane; ++i) {
          if (i == winner_slot) values[i] = -kInfinity;
        }
      }
    }

    float weight = 0.0f;
    if (lane < kTopK) {
      float logit = LogitBits_xe35<T>::to_float(row_ptr[selected_id]);
      float scaled = static_cast<float>(static_cast<T>(logit * args.router_multiplier));
      weight = 1.0f / (1.0f + sycl::exp(-scaled));
    }
    float sum = sycl::reduce_over_group(subgroup, weight, sycl::plus<float>());
    if (lane < kTopK) {
      args.ids[out] = map_expert_xe35(args, selected_id);
      args.weights[out] = args.renormalize ? weight * (args.route_scale / (sum + args.division_eps)) : weight;
    }
  }
};

// Each lane owns a fixed-capacity slice of the experts. Four vector widths
// cover every expert count up to 256; partially filled slices use masked
// scalar loads. The tile width is a subgroup memory-layout choice, independent
// of routing top-k or a model's expert count.
template <typename T, int kValuesPerLane, bool kFullTile>
struct MuseSparkRouting_xe35 {
  static constexpr int kSubgroup = 32;
  static constexpr int kSubgroupsPerGroup = 8;
  static constexpr int kMaxTopK = 8;
  using Raw = typename LogitBits_xe35<T>::Raw;
  using LogitVec = sycl::vec<Raw, kValuesPerLane>;
  using BiasVec = sycl::vec<float, kValuesPerLane>;
  static_assert(sizeof(T) == sizeof(Raw));

  T const* input;
  RoutingArguments_xe35 args;

  [[sycl::reqd_sub_group_size(kSubgroup)]]
  void operator()(sycl::nd_item<1> item) const {
    constexpr float kInfinity = std::numeric_limits<float>::infinity();
    sycl::sub_group subgroup = item.get_sub_group();
    int row =
        static_cast<int>(item.get_group(0)) * kSubgroupsPerGroup + static_cast<int>(subgroup.get_group_linear_id());
    int lane = static_cast<int>(subgroup.get_local_linear_id());
    if (row >= args.tokens) {
      return;
    }
    if (row_is_padding_xe35(args, row)) {
      if (lane < args.topk) {
        args.ids[row * args.topk + lane] = -1;
        args.weights[row * args.topk + lane] = 0.0f;
      }
      return;
    }

    const size_t row_start = static_cast<size_t>(row) * (kFullTile ? kValuesPerLane * kSubgroup : args.experts);
    LogitVec logits{};
    BiasVec biases{};
    if constexpr (kFullTile) {
      auto const* bits = reinterpret_cast<Raw const*>(input) + row_start + lane * kValuesPerLane;
      logits = *reinterpret_cast<LogitVec const*>(bits);
      if (args.correction_bias != nullptr) {
        biases = *reinterpret_cast<BiasVec const*>(args.correction_bias + lane * kValuesPerLane);
      }
    }
    float rank_values[kValuesPerLane];

#pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i) {
      int expert = lane * kValuesPerLane + i;
      if (kFullTile || expert < args.experts) {
        float logit = kFullTile ? LogitBits_xe35<T>::to_float(logits[i]) : static_cast<float>(input[row_start + expert]);
        float scaled = static_cast<float>(static_cast<T>(logit * args.router_multiplier));
        float score = 1.0f / (1.0f + sycl::exp(-scaled));
        rank_values[i] =
            args.correction_bias == nullptr ? score : score + (kFullTile ? biases[i] : args.correction_bias[expert]);
      } else {
        rank_values[i] = -kInfinity;
      }
    }

    // Result lane k owns selected expert k. Keep selection shared across the
    // subgroup, then compute the unbiased weights in parallel.
    int selected_id = -1;

#pragma unroll
    for (int k = 0; k < kMaxTopK; ++k) {
      if (k >= args.topk) {
        break;
      }
      float lane_value = -kInfinity;
      int lane_slot = -1;
#pragma unroll
      for (int i = 0; i < kValuesPerLane; ++i) {
        const int expert = lane * kValuesPerLane + i;
        if ((kFullTile || expert < args.experts) && (lane_slot < 0 || rank_values[i] > lane_value)) {
          lane_value = rank_values[i];
          lane_slot = i;
        }
      }

      // Select the maximum with one subgroup reduction and a ballot. The
      // local search visits increasing expert IDs and keeps the first equal
      // score. The lowest holder lane therefore has the lowest expert ID
      // across lanes. This also works for partially filled lanes.
      const float group_value = sycl::reduce_over_group(subgroup, lane_value, sycl::maximum<float>());
      std::uint32_t holders = 0;
      sycl::ext::oneapi::group_ballot(subgroup, lane_slot >= 0 && lane_value == group_value).extract_bits(holders);
      const int winner_lane = __builtin_ctz(holders);
      const int winner_slot = sycl::select_from_group(subgroup, lane_slot, winner_lane);
      const int winner_id = winner_lane * kValuesPerLane + winner_slot;
      if (lane == winner_lane && lane_slot >= 0) {
#pragma unroll
        for (int i = 0; i < kValuesPerLane; ++i) {
          if (i == lane_slot) rank_values[i] = -kInfinity;
        }
      }

      if (lane == k) selected_id = winner_id;
    }

    float weight = 0.0f;
    if (lane < args.topk) {
      float logit = static_cast<float>(input[static_cast<size_t>(row) * args.experts + selected_id]);
      float scaled = static_cast<float>(static_cast<T>(logit * args.router_multiplier));
      weight = 1.0f / (1.0f + sycl::exp(-scaled));
    }
    const float sum = sycl::reduce_over_group(subgroup, weight, sycl::plus<float>());
    if (lane < args.topk) {
      const float factor = args.renormalize ? args.route_scale / (sum + args.division_eps) : 1.0f;
      args.ids[row * args.topk + lane] = map_expert_xe35(args, selected_id);
      args.weights[row * args.topk + lane] = weight * factor;
    }
  }
};

template <typename T, int kValuesPerLane, bool kFullTile>
void launch_muse_spark_routing_tile_xe35(sycl::queue& queue, T const* input, RoutingArguments_xe35 const& args) {
  using Kernel = MuseSparkRouting_xe35<T, kValuesPerLane, kFullTile>;
  sycl::range<1> local(Kernel::kSubgroupsPerGroup * Kernel::kSubgroup);
  sycl::range<1> global(static_cast<size_t>(div_up(args.tokens, Kernel::kSubgroupsPerGroup)) * local[0]);
  sycl_kernel_submit(global, local, queue, Kernel{input, args});
}

template <typename T, int kValuesPerLane>
void launch_full_tile_topk8_xe35(sycl::queue& queue, T const* input, RoutingArguments_xe35 const& args) {
  using Kernel = FullTileTopk8_xe35<T, kValuesPerLane>;
  sycl::range<1> local(Kernel::kSubgroupsPerGroup * Kernel::kSubgroup);
  sycl::range<1> global(static_cast<size_t>(div_up(args.tokens, Kernel::kSubgroupsPerGroup)) * local[0]);
  sycl_kernel_submit(global, local, queue, Kernel{input, args});
}

template <typename T, int kValuesPerLane>
void launch_muse_spark_routing_bucket_xe35(sycl::queue& queue, T const* input, RoutingArguments_xe35 const& args) {
  if (args.experts == kValuesPerLane * 32) {
    if (args.topk == 8) {
      launch_full_tile_topk8_xe35<T, kValuesPerLane>(queue, input, args);
    } else {
      launch_muse_spark_routing_tile_xe35<T, kValuesPerLane, true>(queue, input, args);
    }
  } else {
    launch_muse_spark_routing_tile_xe35<T, kValuesPerLane, false>(queue, input, args);
  }
}

template <typename T>
void launch_muse_spark_routing_xe35(sycl::queue& queue, T const* input, RoutingArguments_xe35 const& args) {
  const int values_per_lane = div_up(args.experts, 32);
  if (values_per_lane <= 1) {
    launch_muse_spark_routing_bucket_xe35<T, 1>(queue, input, args);
  } else if (values_per_lane <= 2) {
    launch_muse_spark_routing_bucket_xe35<T, 2>(queue, input, args);
  } else if (values_per_lane <= 4) {
    launch_muse_spark_routing_bucket_xe35<T, 4>(queue, input, args);
  } else {
    launch_muse_spark_routing_bucket_xe35<T, 8>(queue, input, args);
  }
}

}

void topk_sigmoid_xe35(
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

  const int values_per_lane = div_up<int64_t>(n_experts, 32);
  const bool full_tile = n_experts == values_per_lane * 32;
  const bool vector_width = values_per_lane == 1 || values_per_lane == 2 || values_per_lane == 4 || values_per_lane == 8;
  if ((gating_output.scalar_type() == at::kHalf || gating_output.scalar_type() == at::kBFloat16) && n_topk == 8 && full_tile && vector_width) {
    auto queue = at::xpu::getCurrentXPUStream().queue();
    DISPATCH_FLOAT_TYPES(gating_output.scalar_type(), "topk_sigmoid_xe35", [&] {
      TopKSigmoidImpl_xe35::RoutingArguments_xe35 args{
          topk_weights.data_ptr<float>(), topk_indices.data_ptr<int>(), bias_ptr,
          nullptr, nullptr, static_cast<int>(n_tokens), static_cast<int>(n_experts),
          static_cast<int>(n_topk), 1.0f, 1.0f, 0.0f, renormalize};
      TopKSigmoidImpl_xe35::launch_muse_spark_routing_xe35(queue, gating_output.data_ptr<scalar_t>(), args);
    });
    return;
  }
  topk_sigmoid(topk_weights, topk_indices, gating_output, renormalize, correction_bias);
}

void muse_spark_moe_routing_xe35(
    at::Tensor& topk_weights,
    at::Tensor& topk_ids,
    at::Tensor& gating_output,
    const c10::optional<at::Tensor>& correction_bias,
    double router_output_multiplier,
    double route_scale,
    double division_eps,
    const c10::optional<at::Tensor>& local_expert_mapping,
    const c10::optional<at::Tensor>& num_valid_tokens) {
  TORCH_CHECK(gating_output.dim() == 2, "gating_output must be 2D");
  TORCH_CHECK(gating_output.is_contiguous(), "gating_output must be contiguous");
  int64_t n_tokens = gating_output.size(0);
  int64_t n_experts = gating_output.size(1);
  TORCH_CHECK(n_experts <= 256, "n_experts only support up to 256, got ", n_experts);
  TORCH_CHECK(topk_weights.scalar_type() == at::kFloat, "topk_weights should be Float");
  TORCH_CHECK(topk_ids.scalar_type() == at::kInt, "topk_ids should be Int");
  TORCH_CHECK(topk_weights.dim() == 2 && topk_weights.size(0) == n_tokens, "topk_weights must be [n_tokens, topk]");
  TORCH_CHECK(topk_ids.sizes() == topk_weights.sizes(), "topk_ids and topk_weights must have the same shape");
  TORCH_CHECK(topk_weights.is_contiguous() && topk_ids.is_contiguous(), "outputs must be contiguous");

  int64_t n_topk = topk_weights.size(1);
  auto max_topk = n_experts < 8 ? n_experts : 8;
  TORCH_CHECK(0 < n_topk && n_topk <= max_topk, "topk must be less than or equal to num_experts and 8");

  const float* bias_ptr = nullptr;
  if (correction_bias.has_value()) {
    const auto& bias = correction_bias.value();
    TORCH_CHECK(bias.dim() == 1 && bias.size(0) == n_experts, "correction_bias must be 1D [num_experts]");
    TORCH_CHECK(bias.scalar_type() == at::kFloat, "correction_bias must be float32");
    TORCH_CHECK(bias.is_contiguous(), "correction_bias must be contiguous");
    bias_ptr = bias.data_ptr<float>();
  }

  const int* mapping_ptr = nullptr;
  if (local_expert_mapping.has_value()) {
    const auto& mapping = local_expert_mapping.value();
    TORCH_CHECK(mapping.dim() == 1 && mapping.size(0) == n_experts, "local_expert_mapping must be 1D [num_experts]");
    TORCH_CHECK(mapping.scalar_type() == at::kInt, "local_expert_mapping must be int32");
    TORCH_CHECK(mapping.is_contiguous(), "local_expert_mapping must be contiguous");
    mapping_ptr = mapping.data_ptr<int>();
  }

  const int* valid_ptr = nullptr;
  if (num_valid_tokens.has_value()) {
    const auto& valid = num_valid_tokens.value();
    TORCH_CHECK(valid.numel() == 1, "num_valid_tokens must be a single element tensor");
    TORCH_CHECK(valid.scalar_type() == at::kInt, "num_valid_tokens must be int32");
    valid_ptr = valid.data_ptr<int>();
  }

  if (n_tokens == 0) {
    return;
  }

  auto queue = at::xpu::getCurrentXPUStream().queue();
  DISPATCH_FLOAT_TYPES(gating_output.scalar_type(), "muse_spark_moe_routing_kernel", [&] {
    TopKSigmoidImpl_xe35::RoutingArguments_xe35 args{
        topk_weights.data_ptr<float>(),
        topk_ids.data_ptr<int>(),
        bias_ptr,
        mapping_ptr,
        valid_ptr,
        static_cast<int>(n_tokens),
        static_cast<int>(n_experts),
        static_cast<int>(n_topk),
        static_cast<float>(router_output_multiplier),
        static_cast<float>(route_scale),
        static_cast<float>(division_eps),
        /*renormalize=*/true};
    TopKSigmoidImpl_xe35::launch_muse_spark_routing_xe35<scalar_t>(queue, gating_output.data_ptr<scalar_t>(), args);
  });
}
}
