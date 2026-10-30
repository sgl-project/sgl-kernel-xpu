#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdint>
#include <sycl/sycl.hpp>

#include "Fp4Utils.h"
#include "QuantUtils.h"
#include "SYCLHelpers.h"
#include "Utils.h"
#include "sgl_kernel_export.h"

namespace {

namespace fp4_utils = at::native::xpu::fp4;

constexpr uint32_t kSubGroupSize = 32;
constexpr uint32_t kBlockSize = 64;
constexpr uint32_t kWorkgroupSize = 256;
constexpr uint32_t kRowsPerWorkgroup = kWorkgroupSize / kSubGroupSize;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kRopeDim = 64;
constexpr int64_t kGroupSize = 32;
constexpr int64_t kNumGroups = kHeadDim / kGroupSize;

inline float subgroup_reduce_max(sycl::sub_group subgroup, float value) {
  value = sycl::fmax(value, sycl::permute_group_by_xor(subgroup, value, 8));
  value = sycl::fmax(value, sycl::permute_group_by_xor(subgroup, value, 4));
  value = sycl::fmax(value, sycl::permute_group_by_xor(subgroup, value, 2));
  value = sycl::fmax(value, sycl::permute_group_by_xor(subgroup, value, 1));
  value = sycl::fmax(value, sycl::permute_group_by_xor(subgroup, value, 16));
  return value;
}

template <typename input_t, typename position_t>
struct FusedQIndexerRopeHadamardFp4QuantKernel {
  [[sycl::reqd_sub_group_size(kSubGroupSize)]]
  void operator()(sycl::nd_item<1> item) const {
    const uint32_t local_id = static_cast<uint32_t>(item.get_local_id(0));
    const uint32_t subgroup_id = local_id / kSubGroupSize;
    const uint32_t lane_id = local_id % kSubGroupSize;
    const uint32_t row_id = static_cast<uint32_t>(item.get_group(0)) * kRowsPerWorkgroup + subgroup_id;
    if (row_id >= num_rows) return;

    const auto subgroup = item.get_sub_group();
    const uint32_t batch_id = row_id / num_heads;
    const input_t* input_row = q_input + static_cast<int64_t>(row_id) * kHeadDim;
    float x0 = static_cast<float>(input_row[lane_id]);
    float x1 = static_cast<float>(input_row[lane_id + kSubGroupSize]);
    float x2 = static_cast<float>(input_row[lane_id + 2 * kSubGroupSize]);
    float x3 = static_cast<float>(input_row[lane_id + 3 * kSubGroupSize]);

    const float* frequency = rope_cache + static_cast<int64_t>(positions[batch_id]) * kRopeDim;
    const bool is_even_lane = (lane_id & 1u) == 0;
    const uint32_t rope_pair = lane_id / 2;
    const float x2_peer = sycl::permute_group_by_xor(subgroup, x2, 1);
    const float x3_peer = sycl::permute_group_by_xor(subgroup, x3, 1);
    const float x2_real = is_even_lane ? x2 : x2_peer;
    const float x2_imag = is_even_lane ? x2_peer : x2;
    const float x3_real = is_even_lane ? x3 : x3_peer;
    const float x3_imag = is_even_lane ? x3_peer : x3;
    const float x2_cos = frequency[2 * rope_pair];
    const float x2_sin = frequency[2 * rope_pair + 1];
    const float x3_cos = frequency[kRopeDim / 2 + 2 * rope_pair];
    const float x3_sin = frequency[kRopeDim / 2 + 2 * rope_pair + 1];
    const float x2_rot_real = static_cast<float>(static_cast<input_t>(x2_real * x2_cos - x2_imag * x2_sin));
    const float x2_rot_imag = static_cast<float>(static_cast<input_t>(x2_real * x2_sin + x2_imag * x2_cos));
    const float x3_rot_real = static_cast<float>(static_cast<input_t>(x3_real * x3_cos - x3_imag * x3_sin));
    const float x3_rot_imag = static_cast<float>(static_cast<input_t>(x3_real * x3_sin + x3_imag * x3_cos));
    x2 = is_even_lane ? x2_rot_real : x2_rot_imag;
    x3 = is_even_lane ? x3_rot_real : x3_rot_imag;

    float sum0 = x0 + x2;
    float sum1 = x1 + x3;
    float diff0 = x0 - x2;
    float diff1 = x1 - x3;
    for (uint32_t mask = 1; mask <= 16; mask <<= 1) {
      const float sum0_other = sycl::permute_group_by_xor(subgroup, sum0, mask);
      const float sum1_other = sycl::permute_group_by_xor(subgroup, sum1, mask);
      const float diff0_other = sycl::permute_group_by_xor(subgroup, diff0, mask);
      const float diff1_other = sycl::permute_group_by_xor(subgroup, diff1, mask);
      sum0 = (lane_id & mask) ? (sum0_other - sum0) : (sum0 + sum0_other);
      sum1 = (lane_id & mask) ? (sum1_other - sum1) : (sum1 + sum1_other);
      diff0 = (lane_id & mask) ? (diff0_other - diff0) : (diff0 + diff0_other);
      diff1 = (lane_id & mask) ? (diff1_other - diff1) : (diff1 + diff1_other);
    }
    x0 = sum0 + sum1;
    x1 = sum0 - sum1;
    x2 = diff0 + diff1;
    x3 = diff0 - diff1;

    constexpr float kHadamardScale = 0.08838834764831845f;
    x0 *= kHadamardScale;
    x1 *= kHadamardScale;
    x2 *= kHadamardScale;
    x3 *= kHadamardScale;
    x0 = static_cast<float>(static_cast<input_t>(x0));
    x1 = static_cast<float>(static_cast<input_t>(x1));
    x2 = static_cast<float>(static_cast<input_t>(x2));
    x3 = static_cast<float>(static_cast<input_t>(x3));

    const uint8_t scale0 =
        at::native::xpu::castToUE8M0(sycl::fmax(1.0e-4f, subgroup_reduce_max(subgroup, sycl::fabs(x0))) / 6.0f);
    const uint8_t scale1 =
        at::native::xpu::castToUE8M0(sycl::fmax(1.0e-4f, subgroup_reduce_max(subgroup, sycl::fabs(x1))) / 6.0f);
    const uint8_t scale2 =
        at::native::xpu::castToUE8M0(sycl::fmax(1.0e-4f, subgroup_reduce_max(subgroup, sycl::fabs(x2))) / 6.0f);
    const uint8_t scale3 =
        at::native::xpu::castToUE8M0(sycl::fmax(1.0e-4f, subgroup_reduce_max(subgroup, sycl::fabs(x3))) / 6.0f);
    const float inv0 = at::native::xpu::invScaleUE8M0(scale0);
    const float inv1 = at::native::xpu::invScaleUE8M0(scale1);
    const float inv2 = at::native::xpu::invScaleUE8M0(scale2);
    const float inv3 = at::native::xpu::invScaleUE8M0(scale3);
    const uint8_t code0 = fp4_utils::encode_e2m1_ties_to_zero(x0 * inv0);
    const uint8_t code1 = fp4_utils::encode_e2m1_ties_to_zero(x1 * inv1);
    const uint8_t code2 = fp4_utils::encode_e2m1_ties_to_zero(x2 * inv2);
    const uint8_t code3 = fp4_utils::encode_e2m1_ties_to_zero(x3 * inv3);
    const uint8_t code0_peer = sycl::permute_group_by_xor(subgroup, code0, 1);
    const uint8_t code1_peer = sycl::permute_group_by_xor(subgroup, code1, 1);
    const uint8_t code2_peer = sycl::permute_group_by_xor(subgroup, code2, 1);
    const uint8_t code3_peer = sycl::permute_group_by_xor(subgroup, code3, 1);

    if (lane_id == 0) {
      q_scale[static_cast<int64_t>(row_id) * kNumGroups] = scale0;
      q_scale[static_cast<int64_t>(row_id) * kNumGroups + 1] = scale1;
      q_scale[static_cast<int64_t>(row_id) * kNumGroups + 2] = scale2;
      q_scale[static_cast<int64_t>(row_id) * kNumGroups + 3] = scale3;
      weights_out[row_id] = static_cast<float>(weight[row_id]) * weight_scale;
    }
    if (is_even_lane) {
      const int64_t row_offset = static_cast<int64_t>(row_id) * kBlockSize + lane_id / 2;
      q_fp4[row_offset] = static_cast<int8_t>((code0 & 0x0F) | ((code0_peer & 0x0F) << 4));
      q_fp4[row_offset + 16] = static_cast<int8_t>((code1 & 0x0F) | ((code1_peer & 0x0F) << 4));
      q_fp4[row_offset + 32] = static_cast<int8_t>((code2 & 0x0F) | ((code2_peer & 0x0F) << 4));
      q_fp4[row_offset + 48] = static_cast<int8_t>((code3 & 0x0F) | ((code3_peer & 0x0F) << 4));
    }
  }

  const input_t* q_input;
  int8_t* q_fp4;
  uint8_t* q_scale;
  const input_t* weight;
  float* weights_out;
  const float* rope_cache;
  const position_t* positions;
  uint32_t num_rows;
  uint32_t num_heads;
  float weight_scale;
};

}  // namespace

SGL_KERNEL_EXPORT std::tuple<at::Tensor, at::Tensor, at::Tensor> fused_q_indexer_rope_hadamard_fp4_quant(
    const at::Tensor& q_input,
    const at::Tensor& weight,
    double weight_scale,
    const at::Tensor& freqs_cis,
    const at::Tensor& positions) {
  // Same interface as the CUDA op in sglang.kernels.ops.attention.dsv4:
  // outputs are allocated here, freqs_cis may be the complex64 cache
  // [max_pos, 32] or its real float32 view [max_pos, 64], and the four
  // ue8m0 scale bytes per head are returned packed as one int32.
  const at::Tensor rope_cache =
      freqs_cis.is_complex() ? at::view_as_real(freqs_cis).flatten(-2).contiguous() : freqs_cis;
  CHECK_INPUT(q_input);
  CHECK_INPUT(weight);
  CHECK_INPUT(rope_cache);
  CHECK_INPUT(positions);
  CHECK_DIM(3, q_input);
  CHECK_DIM(2, weight);
  CHECK_DIM(2, rope_cache);
  CHECK_DIM(1, positions);
  TORCH_CHECK(q_input.size(2) == kHeadDim, "q_input head_dim must be 128");
  TORCH_CHECK(weight.size(0) == q_input.size(0) && weight.size(1) == q_input.size(1), "weight must be [B, H]");
  TORCH_CHECK(
      rope_cache.size(1) == kRopeDim && positions.size(0) == q_input.size(0), "invalid rope_cache or positions shape");
  TORCH_CHECK(
      q_input.scalar_type() == at::kBFloat16 || q_input.scalar_type() == at::kHalf, "q_input must be bf16 or fp16");
  TORCH_CHECK(weight.scalar_type() == q_input.scalar_type(), "weight dtype must match q_input");
  TORCH_CHECK(rope_cache.scalar_type() == at::kFloat, "rope_cache must be float32 (or complex64 freqs_cis)");
  TORCH_CHECK(
      positions.scalar_type() == at::kInt || positions.scalar_type() == at::kLong, "positions must be int32 or int64");

  const int64_t batch = q_input.size(0);
  const int64_t heads = q_input.size(1);
  at::Tensor q_fp4 = at::empty({batch, heads, kBlockSize}, q_input.options().dtype(at::kChar));
  // One packed int32 per head; the kernel writes its four ue8m0 exponent
  // bytes through this zero-copy uint8 view.
  at::Tensor q_sf = at::empty({batch, heads}, q_input.options().dtype(at::kInt));
  at::Tensor q_scale = q_sf.view(at::kByte).view({batch, heads, kNumGroups});
  at::Tensor weights_out = at::empty({batch, heads, 1}, q_input.options().dtype(at::kFloat));
  if (q_input.numel() == 0) return {q_fp4, q_sf, weights_out};

  const uint32_t num_rows = static_cast<uint32_t>(batch * heads);
  auto queue = c10::xpu::getCurrentXPUStream().queue();
  // The dispatch macros `return` out of the enclosing callable, which must
  // therefore be void; the tuple is returned outside it.
  [&]() -> void {
    SYCL_DISPATCH_FLOATING_TYPES(
      at::kHalf, at::kBFloat16, q_input.scalar_type(), "fused_q_indexer_rope_hadamard_fp4_quant", [&]() {
        using input_t = scalar_t;
        AT_DISPATCH_INDEX_TYPES(positions.scalar_type(), "fused_q_indexer_rope_hadamard_fp4_quant", [&]() {
          using position_t = index_t;
          FusedQIndexerRopeHadamardFp4QuantKernel<input_t, position_t> kernel{
              q_input.data_ptr<input_t>(),
              q_fp4.data_ptr<int8_t>(),
              q_scale.data_ptr<uint8_t>(),
              weight.data_ptr<input_t>(),
              weights_out.data_ptr<float>(),
              rope_cache.data_ptr<float>(),
              positions.data_ptr<position_t>(),
              num_rows,
              static_cast<uint32_t>(q_input.size(1)),
              static_cast<float>(weight_scale)};
          const sycl::range<1> global_range(((num_rows + kRowsPerWorkgroup - 1) / kRowsPerWorkgroup) * kWorkgroupSize);
          const sycl::range<1> local_range(kWorkgroupSize);
          sycl_kernel_submit(global_range, local_range, queue, kernel);
        });
      });
  }();
  return {q_fp4, q_sf, weights_out};
}
