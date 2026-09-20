#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdint>
#include <sycl/sycl.hpp>

#include "QuantUtils.h"
#include "Utils.h"
#include "sgl_kernel_export.h"

namespace {

constexpr uint32_t kSubGroupSize = 32;
constexpr uint32_t kBlockSize = 64;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kRopeDim = 64;
constexpr int64_t kGroupSize = 32;
constexpr int64_t kNumGroups = kHeadDim / kGroupSize;

inline uint8_t quant_fp4_e2m1(float value) {
  const float absolute_value = sycl::fmin(sycl::fabs(value), 6.0f);
  uint8_t code = 0;
  code += absolute_value > 0.25f;
  code += absolute_value > 0.75f;
  code += absolute_value > 1.25f;
  code += absolute_value > 1.75f;
  code += absolute_value > 2.5f;
  code += absolute_value > 3.5f;
  code += absolute_value > 5.0f;
  if (value < 0.0f && code != 0) code |= 0x8;
  return code;
}

inline float subgroup_reduce_max(sycl::sub_group subgroup, float value) {
  value = sycl::fmax(value, sycl::permute_group_by_xor(subgroup, value, 8));
  value = sycl::fmax(value, sycl::permute_group_by_xor(subgroup, value, 4));
  value = sycl::fmax(value, sycl::permute_group_by_xor(subgroup, value, 2));
  value = sycl::fmax(value, sycl::permute_group_by_xor(subgroup, value, 1));
  value = sycl::fmax(value, sycl::permute_group_by_xor(subgroup, value, 16));
  return value;
}

inline void reduce_max_64(
    float* reduction,
    sycl::nd_item<1> item,
    sycl::sub_group subgroup,
    uint32_t local_id) {
  const uint32_t lane_id = local_id & (kSubGroupSize - 1u);
  const uint32_t subgroup_id = local_id / kSubGroupSize;
  const float subgroup_max = subgroup_reduce_max(subgroup, reduction[local_id]);
  if (lane_id == 0) reduction[subgroup_id] = subgroup_max;
  item.barrier(sycl::access::fence_space::local_space);
  if (subgroup_id == 0) {
    const float block_max = subgroup_reduce_max(
        subgroup, lane_id < kBlockSize / kSubGroupSize ? reduction[lane_id] : 0.0f);
    if (lane_id == 0) reduction[0] = block_max;
  }
  item.barrier(sycl::access::fence_space::local_space);
}

template <typename input_t, typename position_t>
struct FusedQIndexerRopeHadamardFp4QuantKernel {
  [[sycl::reqd_sub_group_size(kSubGroupSize)]]
  void operator()(sycl::nd_item<1> item) const {
    const uint32_t row_id = static_cast<uint32_t>(item.get_group(0));
    const uint32_t local_id = static_cast<uint32_t>(item.get_local_id(0));
    if (row_id >= num_rows) return;

    float* data = local_data.template get_multi_ptr<sycl::access::decorated::no>().get();
    float* reduction = local_reduction.template get_multi_ptr<sycl::access::decorated::no>().get();
    float* group_inverse = local_group_inverse.template get_multi_ptr<sycl::access::decorated::no>().get();
    uint8_t* group_scale = local_group_scale.template get_multi_ptr<sycl::access::decorated::no>().get();
    const int64_t dim0 = static_cast<int64_t>(local_id);
    const int64_t dim1 = dim0 + kBlockSize;
    const uint32_t batch_id = row_id / num_heads;
    auto subgroup = item.get_sub_group();

    const input_t* input_row = q_input + static_cast<int64_t>(row_id) * kHeadDim;
    data[dim0] = static_cast<float>(input_row[dim0]);
    data[dim1] = static_cast<float>(input_row[dim1]);
    item.barrier(sycl::access::fence_space::local_space);

    if (local_id < kRopeDim / 2) {
      const int64_t rope_offset = kHeadDim - kRopeDim + 2 * static_cast<int64_t>(local_id);
      const float* frequency = rope_cache + static_cast<int64_t>(positions[batch_id]) * kRopeDim;
      const float real = data[rope_offset];
      const float imaginary = data[rope_offset + 1];
      const float cosine = frequency[2 * local_id];
      const float sine = frequency[2 * local_id + 1];
      data[rope_offset] = real * cosine - imaginary * sine;
      data[rope_offset + 1] = real * sine + imaginary * cosine;
    }
    item.barrier(sycl::access::fence_space::local_space);

    float first = data[dim0] + data[dim1];
    float second = data[dim0] - data[dim1];
    const uint32_t lane_id = local_id & (kSubGroupSize - 1u);
    for (uint32_t mask = 1; mask <= 16; mask <<= 1) {
      const float first_other = sycl::permute_group_by_xor(subgroup, first, mask);
      const float second_other = sycl::permute_group_by_xor(subgroup, second, mask);
      first = (lane_id & mask) ? (first_other - first) : (first + first_other);
      second = (lane_id & mask) ? (second_other - second) : (second + second_other);
    }
    data[dim0] = first;
    data[dim1] = second;
    item.barrier(sycl::access::fence_space::local_space);
    for (uint32_t mask = kSubGroupSize; mask < kBlockSize; mask <<= 1) {
      const uint32_t peer = local_id ^ mask;
      const float first_other = data[peer];
      const float second_other = data[peer + kBlockSize];
      first = (local_id & mask) ? (first_other - first) : (first + first_other);
      second = (local_id & mask) ? (second_other - second) : (second + second_other);
      item.barrier(sycl::access::fence_space::local_space);
      data[dim0] = first;
      data[dim1] = second;
      item.barrier(sycl::access::fence_space::local_space);
    }
    constexpr float kHadamardScale = 0.08838834764831845f;
    data[dim0] = first * kHadamardScale;
    data[dim1] = second * kHadamardScale;
    item.barrier(sycl::access::fence_space::local_space);

    for (int64_t group = 0; group < kNumGroups; ++group) {
      const int64_t element = group * kGroupSize + static_cast<int64_t>(local_id);
      reduction[local_id] = local_id < kGroupSize ? sycl::fabs(data[element]) : 0.0f;
      reduce_max_64(reduction, item, subgroup, local_id);
      if (local_id == 0) {
        const uint8_t scale = at::native::xpu::castToUE8M0(
            sycl::fmax(1.0e-4f, reduction[0]) / 6.0f);
        group_scale[group] = scale;
        group_inverse[group] = at::native::xpu::invScaleUE8M0(scale);
      }
      item.barrier(sycl::access::fence_space::local_space);
    }

    const int64_t packed_dim0 = 2 * static_cast<int64_t>(local_id);
    const int64_t packed_dim1 = packed_dim0 + 1;
    const uint8_t first_code = quant_fp4_e2m1(
      data[packed_dim0] * group_inverse[packed_dim0 / kGroupSize]);
    const uint8_t second_code = quant_fp4_e2m1(
      data[packed_dim1] * group_inverse[packed_dim1 / kGroupSize]);
    q_fp4[static_cast<int64_t>(row_id) * kBlockSize + local_id] =
        static_cast<uint8_t>((first_code & 0x0F) | ((second_code & 0x0F) << 4));
    if (local_id < kNumGroups) q_scale[static_cast<int64_t>(row_id) * kNumGroups + local_id] = group_scale[local_id];
    if (local_id == 0) weights_out[row_id] = static_cast<float>(weight[row_id]) * weight_scale;
  }

  const input_t* q_input;
  uint8_t* q_fp4;
  uint8_t* q_scale;
  const input_t* weight;
  float* weights_out;
  const float* rope_cache;
  const position_t* positions;
  uint32_t num_rows;
  uint32_t num_heads;
  float weight_scale;
  sycl::local_accessor<float, 1> local_data;
  sycl::local_accessor<float, 1> local_reduction;
  sycl::local_accessor<float, 1> local_group_inverse;
  sycl::local_accessor<uint8_t, 1> local_group_scale;
};

}  // namespace

SGL_KERNEL_EXPORT void fused_q_indexer_rope_hadamard_fp4_quant(
    const at::Tensor& q_input,
    at::Tensor& q_fp4,
    at::Tensor& q_scale,
    const at::Tensor& weight,
    at::Tensor& weights_out,
    double weight_scale,
    const at::Tensor& rope_cache,
    const at::Tensor& positions) {
  CHECK_INPUT(q_input);
  CHECK_INPUT(q_fp4);
  CHECK_INPUT(q_scale);
  CHECK_INPUT(weight);
  CHECK_INPUT(weights_out);
  CHECK_INPUT(rope_cache);
  CHECK_INPUT(positions);
  CHECK_DIM(3, q_input);
  CHECK_DIM(3, q_fp4);
  CHECK_DIM(3, q_scale);
  CHECK_DIM(2, weight);
  CHECK_DIM(3, weights_out);
  CHECK_DIM(2, rope_cache);
  CHECK_DIM(1, positions);
  TORCH_CHECK(q_input.size(2) == kHeadDim, "q_input head_dim must be 128");
  TORCH_CHECK(q_fp4.size(0) == q_input.size(0) && q_fp4.size(1) == q_input.size(1) && q_fp4.size(2) == kBlockSize, "q_fp4 must be [B, H, 64]");
  TORCH_CHECK(q_scale.size(0) == q_input.size(0) && q_scale.size(1) == q_input.size(1) && q_scale.size(2) == kNumGroups, "q_scale must be [B, H, 4]");
  TORCH_CHECK(weight.size(0) == q_input.size(0) && weight.size(1) == q_input.size(1), "weight must be [B, H]");
  TORCH_CHECK(weights_out.size(0) == q_input.size(0) && weights_out.size(1) == q_input.size(1) && weights_out.size(2) == 1, "weights_out must be [B, H, 1]");
  TORCH_CHECK(rope_cache.size(1) == kRopeDim && positions.size(0) == q_input.size(0), "invalid rope_cache or positions shape");
  TORCH_CHECK(q_input.scalar_type() == at::kBFloat16 || q_input.scalar_type() == at::kHalf, "q_input must be bf16 or fp16");
  TORCH_CHECK(weight.scalar_type() == q_input.scalar_type(), "weight dtype must match q_input");
  TORCH_CHECK(q_fp4.scalar_type() == at::kByte && q_scale.scalar_type() == at::kByte, "q_fp4 and q_scale must be uint8");
  TORCH_CHECK(weights_out.scalar_type() == at::kFloat && rope_cache.scalar_type() == at::kFloat, "weights_out and rope_cache must be float32");
  TORCH_CHECK(positions.scalar_type() == at::kInt || positions.scalar_type() == at::kLong, "positions must be int32 or int64");
  TORCH_CHECK(q_input.is_contiguous() && q_fp4.is_contiguous() && q_scale.is_contiguous() && weight.is_contiguous() && weights_out.is_contiguous() && rope_cache.is_contiguous() && positions.is_contiguous(), "inputs and outputs must be contiguous");
  if (q_input.numel() == 0) return;

  const uint32_t num_rows = static_cast<uint32_t>(q_input.size(0) * q_input.size(1));
  auto queue = c10::xpu::getCurrentXPUStream().queue();
  SYCL_DISPATCH_FLOATING_TYPES(at::kHalf, at::kBFloat16, q_input.scalar_type(), "fused_q_indexer_rope_hadamard_fp4_quant", [&]() {
    using input_t = scalar_t;
    AT_DISPATCH_INDEX_TYPES(positions.scalar_type(), "fused_q_indexer_rope_hadamard_fp4_quant", [&]() {
      using position_t = index_t;
      queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> local_data(sycl::range<1>(kHeadDim), cgh);
        sycl::local_accessor<float, 1> local_reduction(sycl::range<1>(kBlockSize), cgh);
        sycl::local_accessor<float, 1> local_group_inverse(sycl::range<1>(kNumGroups), cgh);
        sycl::local_accessor<uint8_t, 1> local_group_scale(sycl::range<1>(kNumGroups), cgh);
        FusedQIndexerRopeHadamardFp4QuantKernel<input_t, position_t> kernel{
            q_input.data_ptr<input_t>(), q_fp4.data_ptr<uint8_t>(), q_scale.data_ptr<uint8_t>(),
            weight.data_ptr<input_t>(), weights_out.data_ptr<float>(), rope_cache.data_ptr<float>(),
            positions.data_ptr<position_t>(), num_rows, static_cast<uint32_t>(q_input.size(1)),
            static_cast<float>(weight_scale), local_data, local_reduction, local_group_inverse, local_group_scale};
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(num_rows * kBlockSize), sycl::range<1>(kBlockSize)), kernel);
      });
    });
  });
}