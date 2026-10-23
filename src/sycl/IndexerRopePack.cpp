#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdint>
#include <sycl/sycl.hpp>
#include <type_traits>

#include "Utils.h"
#include "sgl_kernel_export.h"

namespace {

using bf16_t = sycl::ext::oneapi::bfloat16;

constexpr int64_t kHeadDim = 128;
constexpr int64_t kRopeDim = 64;
constexpr int64_t kGroupSize = 32;
constexpr int64_t kNumGroups = kHeadDim / kGroupSize;
constexpr int64_t kPackedDim = kHeadDim / 2;
constexpr int64_t kSlotBytes = kPackedDim + kNumGroups;
constexpr uint32_t kWorkGroupSize = kHeadDim;
constexpr uint32_t kQSubGroupSize = 16;
// Avoid subgroup launch overhead for the small batches measured on XPU.
constexpr uint32_t kQSubgroupThreshold = 64;
constexpr float kFakeQuantAmaxFloor = 6.0f * 0x1p-126f;

inline uint32_t ceil_ue8m0_exponent(float value) {
  const uint32_t bits = sycl::bit_cast<uint32_t>(value);
  const uint32_t exponent = (bits >> 23) & 0xffu;
  const uint32_t mantissa = bits & 0x7fffffu;
  return sycl::clamp(exponent + static_cast<uint32_t>(mantissa != 0), 1u, 254u);
}

inline float ue8m0_scale(uint32_t exponent) {
  return sycl::bit_cast<float>(exponent << 23);
}

inline uint8_t fp4_e2m1_rne(float value) {
  const float magnitude = sycl::fmin(sycl::fabs(value), 6.0f);
  uint8_t code = 0;
  code += magnitude >= 0.25f;
  code += magnitude >= 0.75f;
  code += magnitude >= 1.25f;
  code += magnitude >= 1.75f;
  code += magnitude >= 2.5f;
  code += magnitude >= 3.5f;
  code += magnitude >= 5.0f;
  const bool boundary = magnitude == 0.25f || magnitude == 0.75f || magnitude == 1.25f || magnitude == 1.75f ||
                        magnitude == 2.5f || magnitude == 3.5f || magnitude == 5.0f;
  if (boundary && (code & 1u)) --code;
  if (value < 0.0f && code != 0) code |= 0x8u;
  return code;
}

inline float fp4_e2m1_value(uint8_t code) {
  constexpr float kValues[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float value = kValues[code & 7u];
  return (code & 8u) ? -value : value;
}

inline void fake_quant_stage(sycl::nd_item<1> item, uint32_t local_id, float* values, float* reduction, float* scales) {
  const uint32_t group = local_id / static_cast<uint32_t>(kGroupSize);
  reduction[local_id] = sycl::fabs(values[local_id]);
  item.barrier(sycl::access::fence_space::local_space);

  if ((local_id % kGroupSize) == 0) {
    float amax = 0.0f;
    const uint32_t group_start = group * static_cast<uint32_t>(kGroupSize);
    for (uint32_t i = 0; i < kGroupSize; ++i) {
      amax = sycl::fmax(amax, reduction[group_start + i]);
    }
    const float scale_raw = sycl::fmax(amax, kFakeQuantAmaxFloor) * (1.0f / 6.0f);
    const uint32_t exponent = ceil_ue8m0_exponent(scale_raw);
    scales[group] = ue8m0_scale(exponent);
  }
  item.barrier(sycl::access::fence_space::local_space);

  const float normalized = sycl::clamp(values[local_id] / scales[group], -6.0f, 6.0f);
  const uint8_t code = fp4_e2m1_rne(normalized);
  values[local_id] = static_cast<float>(static_cast<bf16_t>(fp4_e2m1_value(code) * scales[group]));
  item.barrier(sycl::access::fence_space::local_space);
}

inline uint32_t pack_group_exponents(const uint32_t* exponents) {
  return exponents[0] | (exponents[1] << 8) | (exponents[2] << 16) | (exponents[3] << 24);
}

inline float quantize_fake_fp4(float value, float scale) {
  const float normalized = sycl::clamp(value / scale, -6.0f, 6.0f);
  const uint8_t code = fp4_e2m1_rne(normalized);
  return static_cast<float>(static_cast<bf16_t>(fp4_e2m1_value(code) * scale));
}

template <typename position_t>
struct IndexQLegacyKernel {
  void operator()(sycl::nd_item<1> item) const {
    const uint32_t row_id = static_cast<uint32_t>(item.get_group(0));
    const uint32_t local_id = static_cast<uint32_t>(item.get_local_id(0));
    if (row_id >= num_rows) return;

    float* values = local_values.get_multi_ptr<sycl::access::decorated::no>().get();
    float* reduction = local_reduction.get_multi_ptr<sycl::access::decorated::no>().get();
    float* fake_scales = local_fake_scales.get_multi_ptr<sycl::access::decorated::no>().get();
    uint32_t* exponents = local_exponents.get_multi_ptr<sycl::access::decorated::no>().get();
    const uint32_t token_id = row_id / num_heads;
    const int64_t position = static_cast<int64_t>(positions[token_id]);

    values[local_id] = static_cast<float>(input[row_id * kHeadDim + local_id]);
    item.barrier(sycl::access::fence_space::local_space);

    if (local_id < kRopeDim / 2) {
      const int64_t offset = kHeadDim - kRopeDim + 2 * static_cast<int64_t>(local_id);
      const float* freq = freqs_cis + position * kRopeDim;
      const float real = values[offset];
      const float imag = values[offset + 1];
      const float cosine = freq[2 * local_id];
      const float sine = freq[2 * local_id + 1];
      values[offset] = static_cast<float>(static_cast<bf16_t>(real * cosine - imag * sine));
      values[offset + 1] = static_cast<float>(static_cast<bf16_t>(real * sine + imag * cosine));
    }
    item.barrier(sycl::access::fence_space::local_space);

    fake_quant_stage(item, local_id, values, reduction, fake_scales);

    reduction[local_id] = sycl::fabs(values[local_id]);
    item.barrier(sycl::access::fence_space::local_space);
    const uint32_t group = local_id / static_cast<uint32_t>(kGroupSize);
    if ((local_id % kGroupSize) == 0) {
      float amax = 0.0f;
      const uint32_t group_start = group * static_cast<uint32_t>(kGroupSize);
      for (uint32_t i = 0; i < kGroupSize; ++i) {
        amax = sycl::fmax(amax, reduction[group_start + i]);
      }
      exponents[group] = ceil_ue8m0_exponent(sycl::fmax(amax / 6.0f, 1.0e-4f));
    }
    item.barrier(sycl::access::fence_space::local_space);

    if (local_id < kPackedDim) {
      const uint32_t dim0 = 2 * local_id;
      const uint8_t code0 = fp4_e2m1_rne(values[dim0] / ue8m0_scale(exponents[dim0 / kGroupSize]));
      const uint8_t code1 = fp4_e2m1_rne(values[dim0 + 1] / ue8m0_scale(exponents[(dim0 + 1) / kGroupSize]));
      payload[static_cast<int64_t>(row_id) * kPackedDim + local_id] = static_cast<int8_t>(code0 | (code1 << 4));
    }
    if (local_id == 0) {
      scale[row_id] = static_cast<int32_t>(pack_group_exponents(exponents));
      const float scaled_weight = static_cast<float>(head_weights[row_id]) * weight_scale;
      weights_out[row_id] = static_cast<float>(static_cast<bf16_t>(scaled_weight));
    }
  }

  const bf16_t* input;
  const float* freqs_cis;
  const position_t* positions;
  const bf16_t* head_weights;
  int8_t* payload;
  int32_t* scale;
  float* weights_out;
  uint32_t num_rows;
  uint32_t num_heads;
  float weight_scale;
  sycl::local_accessor<float, 1> local_values;
  sycl::local_accessor<float, 1> local_reduction;
  sycl::local_accessor<float, 1> local_fake_scales;
  sycl::local_accessor<uint32_t, 1> local_exponents;
};

template <typename position_t>
struct IndexQSubgroupKernel {
  [[sycl::reqd_sub_group_size(kQSubGroupSize)]]
  void operator()(sycl::nd_item<2> item) const {
    const uint32_t token_id = static_cast<uint32_t>(item.get_group(0));
    const uint32_t head = static_cast<uint32_t>(item.get_group(1));
    const uint32_t lane = static_cast<uint32_t>(item.get_local_id(1));
    if (token_id >= num_tokens || head >= num_heads) return;

    const auto subgroup = item.get_sub_group();
    const int64_t position = static_cast<int64_t>(positions[token_id]);
    const float* freq = freqs_cis + position * kRopeDim;
    const uint32_t* head_words =
        reinterpret_cast<const uint32_t*>(input + (static_cast<int64_t>(token_id) * num_heads + head) * kHeadDim);

    float x0[kNumGroups];
    float x1[kNumGroups];
    float fake_scales[kNumGroups];
    uint32_t exponents[kNumGroups];
    uint8_t packed[kNumGroups];
    for (uint32_t block = 0; block < kNumGroups; ++block) {
      const uint32_t word = head_words[block * kQSubGroupSize + lane];
      x0[block] = static_cast<float>(sycl::bit_cast<bf16_t>(static_cast<uint16_t>(word)));
      x1[block] = static_cast<float>(sycl::bit_cast<bf16_t>(static_cast<uint16_t>(word >> 16)));
    }

    const uint32_t rope_word0 = head_words[2 * kQSubGroupSize + lane];
    const uint32_t rope_word1 = head_words[3 * kQSubGroupSize + lane];
    const float r0 = static_cast<float>(sycl::bit_cast<bf16_t>(static_cast<uint16_t>(rope_word0)));
    const float i0 = static_cast<float>(sycl::bit_cast<bf16_t>(static_cast<uint16_t>(rope_word0 >> 16)));
    const float r1 = static_cast<float>(sycl::bit_cast<bf16_t>(static_cast<uint16_t>(rope_word1)));
    const float i1 = static_cast<float>(sycl::bit_cast<bf16_t>(static_cast<uint16_t>(rope_word1 >> 16)));
    const float cosine0 = freq[2 * lane];
    const float sine0 = freq[2 * lane + 1];
    const float cosine1 = freq[2 * (lane + kQSubGroupSize)];
    const float sine1 = freq[2 * (lane + kQSubGroupSize) + 1];
    x0[2] = static_cast<float>(static_cast<bf16_t>(r0 * cosine0 - i0 * sine0));
    x1[2] = static_cast<float>(static_cast<bf16_t>(r0 * sine0 + i0 * cosine0));
    x0[3] = static_cast<float>(static_cast<bf16_t>(r1 * cosine1 - i1 * sine1));
    x1[3] = static_cast<float>(static_cast<bf16_t>(r1 * sine1 + i1 * cosine1));

    for (uint32_t block = 0; block < kNumGroups; ++block) {
      const float local_amax = sycl::fmax(sycl::fabs(x0[block]), sycl::fabs(x1[block]));
      const float amax = sycl::reduce_over_group(subgroup, local_amax, sycl::maximum<float>());
      const float fake_scale_raw = sycl::fmax(amax, kFakeQuantAmaxFloor) * (1.0f / 6.0f);
      exponents[block] = ceil_ue8m0_exponent(fake_scale_raw);
      fake_scales[block] = ue8m0_scale(exponents[block]);
      x0[block] = quantize_fake_fp4(x0[block], fake_scales[block]);
      x1[block] = quantize_fake_fp4(x1[block], fake_scales[block]);
    }

    for (uint32_t block = 0; block < kNumGroups; ++block) {
      const float local_amax = sycl::fmax(sycl::fabs(x0[block]), sycl::fabs(x1[block]));
      const float amax = sycl::reduce_over_group(subgroup, local_amax, sycl::maximum<float>());
      exponents[block] = ceil_ue8m0_exponent(sycl::fmax(amax / 6.0f, 1.0e-4f));
      const float pack_scale = ue8m0_scale(exponents[block]);
      const uint8_t code0 = fp4_e2m1_rne(x0[block] / pack_scale);
      const uint8_t code1 = fp4_e2m1_rne(x1[block] / pack_scale);
      packed[block] = static_cast<uint8_t>(code0 | (code1 << 4));
    }

    const int64_t row_id = static_cast<int64_t>(token_id) * num_heads + head;
    for (uint32_t block = 0; block < kNumGroups; ++block) {
      payload[row_id * kPackedDim + block * kQSubGroupSize + lane] = static_cast<int8_t>(packed[block]);
    }
    if (lane == 0) {
      scale[row_id] = static_cast<int32_t>(pack_group_exponents(exponents));
      const float scaled_weight = static_cast<float>(head_weights[row_id]) * weight_scale;
      weights_out[row_id] = static_cast<float>(static_cast<bf16_t>(scaled_weight));
    }
  }

  const bf16_t* input;
  const float* freqs_cis;
  const position_t* positions;
  const bf16_t* head_weights;
  int8_t* payload;
  int32_t* scale;
  float* weights_out;
  uint32_t num_tokens;
  uint32_t num_heads;
  float weight_scale;
};

template <typename norm_weight_t, typename position_t>
struct IndexKernelStore {
  void operator()(sycl::nd_item<1> item) const {
    const uint32_t row_id = static_cast<uint32_t>(item.get_group(0));
    const uint32_t local_id = static_cast<uint32_t>(item.get_local_id(0));
    if (row_id >= num_rows) return;

    const int64_t slot_id = loc[row_id];
    if (slot_id <= 0) return;

    float* values = local_values.get_multi_ptr<sycl::access::decorated::no>().get();
    float* reduction = local_reduction.get_multi_ptr<sycl::access::decorated::no>().get();
    float* fake_scales = local_fake_scales.get_multi_ptr<sycl::access::decorated::no>().get();
    uint32_t* exponents = local_exponents.get_multi_ptr<sycl::access::decorated::no>().get();

    const float input_value = static_cast<float>(input[row_id * kHeadDim + local_id]);
    values[local_id] = input_value * input_value;
    item.barrier(sycl::access::fence_space::local_space);
    for (uint32_t stride = kWorkGroupSize / 2; stride > 0; stride >>= 1) {
      if (local_id < stride) values[local_id] += values[local_id + stride];
      item.barrier(sycl::access::fence_space::local_space);
    }

    const float inv_rms = sycl::rsqrt(values[0] / static_cast<float>(kHeadDim) + eps);
    const float norm_value = static_cast<float>(norm_weight[local_id]) * (input_value * inv_rms);
    values[local_id] = static_cast<float>(static_cast<bf16_t>(norm_value));
    item.barrier(sycl::access::fence_space::local_space);

    if (local_id < kRopeDim / 2) {
      const int64_t position = static_cast<int64_t>(positions[row_id]) & ~static_cast<int64_t>(ratio - 1);
      const int64_t offset = kHeadDim - kRopeDim + 2 * static_cast<int64_t>(local_id);
      const float* freq = freqs_cis + position * kRopeDim;
      const float real = values[offset];
      const float imag = values[offset + 1];
      const float cosine = freq[2 * local_id];
      const float sine = freq[2 * local_id + 1];
      values[offset] = static_cast<float>(static_cast<bf16_t>(real * cosine - imag * sine));
      values[offset + 1] = static_cast<float>(static_cast<bf16_t>(real * sine + imag * cosine));
    }
    item.barrier(sycl::access::fence_space::local_space);

    fake_quant_stage(item, local_id, values, reduction, fake_scales);

    reduction[local_id] = sycl::fabs(values[local_id]);
    item.barrier(sycl::access::fence_space::local_space);
    const uint32_t group = local_id / static_cast<uint32_t>(kGroupSize);
    if ((local_id % kGroupSize) == 0) {
      float amax = 0.0f;
      const uint32_t group_start = group * static_cast<uint32_t>(kGroupSize);
      for (uint32_t i = 0; i < kGroupSize; ++i) {
        amax = sycl::fmax(amax, reduction[group_start + i]);
      }
      exponents[group] = ceil_ue8m0_exponent(sycl::fmax(amax / 6.0f, 1.0e-4f));
    }
    item.barrier(sycl::access::fence_space::local_space);

    if (local_id < kPackedDim) {
      const uint32_t dim0 = 2 * local_id;
      const uint8_t code0 = fp4_e2m1_rne(values[dim0] / ue8m0_scale(exponents[dim0 / kGroupSize]));
      const uint8_t code1 = fp4_e2m1_rne(values[dim0 + 1] / ue8m0_scale(exponents[(dim0 + 1) / kGroupSize]));
      const int64_t page = slot_id / page_size;
      const int64_t offset = slot_id % page_size;
      const int64_t page_base = page * cache_stride;
      cache[page_base + offset * kPackedDim + local_id] = static_cast<uint8_t>(code0 | (code1 << 4));
    }
    if (local_id < kNumGroups) {
      const int64_t page = slot_id / page_size;
      const int64_t offset = slot_id % page_size;
      const int64_t scale_base = page * cache_stride + page_size * kPackedDim + offset * kNumGroups;
      cache[scale_base + local_id] = static_cast<uint8_t>(exponents[local_id]);
    }
  }

  const bf16_t* input;
  const norm_weight_t* norm_weight;
  const float* freqs_cis;
  const position_t* positions;
  const int64_t* loc;
  uint8_t* cache;
  uint32_t num_rows;
  int64_t page_size;
  int64_t cache_stride;
  int64_t ratio;
  float eps;
  sycl::local_accessor<float, 1> local_values;
  sycl::local_accessor<float, 1> local_reduction;
  sycl::local_accessor<float, 1> local_fake_scales;
  sycl::local_accessor<uint32_t, 1> local_exponents;
};

template <typename position_t>
void launch_index_q(
    const at::Tensor& input,
    const at::Tensor& freqs_cis,
    const at::Tensor& positions,
    const at::Tensor& head_weights,
    at::Tensor& payload,
    at::Tensor& scale,
    at::Tensor& weights_out,
    float weight_scale) {
  const uint32_t num_rows = static_cast<uint32_t>(input.size(0) * input.size(1));
  if (num_rows == 0) return;

  auto queue = c10::xpu::getCurrentXPUStream().queue();
  if (input.size(0) >= kQSubgroupThreshold) {
    queue.submit([&](sycl::handler& cgh) {
      IndexQSubgroupKernel<position_t> kernel{
          reinterpret_cast<const bf16_t*>(input.data_ptr<at::BFloat16>()),
          freqs_cis.data_ptr<float>(),
          positions.data_ptr<position_t>(),
          reinterpret_cast<const bf16_t*>(head_weights.data_ptr<at::BFloat16>()),
          payload.data_ptr<int8_t>(),
          scale.data_ptr<int32_t>(),
          weights_out.data_ptr<float>(),
          static_cast<uint32_t>(input.size(0)),
          static_cast<uint32_t>(input.size(1)),
          weight_scale};
      cgh.parallel_for(
          sycl::nd_range<2>(
              sycl::range<2>(input.size(0), input.size(1) * kQSubGroupSize), sycl::range<2>(1, kQSubGroupSize)),
          kernel);
    });
    return;
  }

  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> values(sycl::range<1>(kHeadDim), cgh);
    sycl::local_accessor<float, 1> reduction(sycl::range<1>(kHeadDim), cgh);
    sycl::local_accessor<float, 1> fake_scales(sycl::range<1>(kNumGroups), cgh);
    sycl::local_accessor<uint32_t, 1> exponents(sycl::range<1>(kNumGroups), cgh);
    IndexQLegacyKernel<position_t> kernel{
        reinterpret_cast<const bf16_t*>(input.data_ptr<at::BFloat16>()),
        freqs_cis.data_ptr<float>(),
        positions.data_ptr<position_t>(),
        reinterpret_cast<const bf16_t*>(head_weights.data_ptr<at::BFloat16>()),
        payload.data_ptr<int8_t>(),
        scale.data_ptr<int32_t>(),
        weights_out.data_ptr<float>(),
        num_rows,
        static_cast<uint32_t>(input.size(1)),
        weight_scale,
        values,
        reduction,
        fake_scales,
        exponents};
    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(num_rows * kWorkGroupSize), sycl::range<1>(kWorkGroupSize)), kernel);
  });
}

template <typename norm_weight_t, typename position_t>
void launch_index_k(
    const at::Tensor& input,
    const at::Tensor& norm_weight,
    const at::Tensor& freqs_cis,
    const at::Tensor& positions,
    const at::Tensor& loc,
    at::Tensor& cache,
    int64_t ratio,
    int64_t page_size,
    float eps) {
  const uint32_t num_rows = static_cast<uint32_t>(input.size(0));
  if (num_rows == 0) return;

  auto queue = c10::xpu::getCurrentXPUStream().queue();
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> values(sycl::range<1>(kHeadDim), cgh);
    sycl::local_accessor<float, 1> reduction(sycl::range<1>(kHeadDim), cgh);
    sycl::local_accessor<float, 1> fake_scales(sycl::range<1>(kNumGroups), cgh);
    sycl::local_accessor<uint32_t, 1> exponents(sycl::range<1>(kNumGroups), cgh);
    const norm_weight_t* norm_weight_ptr;
    if constexpr (std::is_same_v<norm_weight_t, bf16_t>) {
      norm_weight_ptr = reinterpret_cast<const bf16_t*>(norm_weight.data_ptr<at::BFloat16>());
    } else {
      norm_weight_ptr = norm_weight.data_ptr<float>();
    }
    IndexKernelStore<norm_weight_t, position_t> kernel{
        reinterpret_cast<const bf16_t*>(input.data_ptr<at::BFloat16>()),
        norm_weight_ptr,
        freqs_cis.data_ptr<float>(),
        positions.data_ptr<position_t>(),
        loc.data_ptr<int64_t>(),
        cache.data_ptr<uint8_t>(),
        num_rows,
        page_size,
        cache.stride(0),
        ratio,
        eps,
        values,
        reduction,
        fake_scales,
        exponents};
    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(num_rows * kWorkGroupSize), sycl::range<1>(kWorkGroupSize)), kernel);
  });
}

}  // namespace

SGL_KERNEL_EXPORT void index_q_rope_pack_weights(
    const at::Tensor& input,
    const at::Tensor& freqs_cis,
    const at::Tensor& positions,
    const at::Tensor& head_weights,
    at::Tensor& payload,
    at::Tensor& scale,
    at::Tensor& weights_out,
    double weight_scale) {
  CHECK_INPUT(input);
  CHECK_INPUT(freqs_cis);
  CHECK_INPUT(positions);
  CHECK_INPUT(head_weights);
  CHECK_INPUT(payload);
  CHECK_INPUT(scale);
  CHECK_INPUT(weights_out);
  TORCH_CHECK(input.dim() == 3 && input.size(2) == kHeadDim, "input must be [tokens, heads, 128]");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16, "input must be bfloat16");
  TORCH_CHECK(freqs_cis.dim() == 2 && freqs_cis.size(1) == kRopeDim, "freqs_cis must be [max_pos, 64]");
  TORCH_CHECK(freqs_cis.scalar_type() == at::kFloat, "freqs_cis must be float32");
  TORCH_CHECK(positions.dim() == 1 && positions.size(0) == input.size(0), "positions must be [tokens]");
  TORCH_CHECK(
      positions.scalar_type() == at::kInt || positions.scalar_type() == at::kLong, "positions must be int32 or int64");
  TORCH_CHECK(
      head_weights.dim() == 2 && head_weights.size(0) == input.size(0) && head_weights.size(1) == input.size(1),
      "head_weights must be [tokens, heads]");
  TORCH_CHECK(head_weights.scalar_type() == at::kBFloat16, "head_weights must be bfloat16");
  TORCH_CHECK(
      payload.dim() == 2 && payload.size(0) == input.size(0) * input.size(1) && payload.size(1) == kPackedDim &&
          payload.scalar_type() == at::kChar,
      "payload must be int8 [tokens * heads, 64]");
  TORCH_CHECK(
      scale.dim() == 1 && scale.size(0) == input.size(0) * input.size(1) && scale.scalar_type() == at::kInt,
      "scale must be int32 [tokens * heads]");
  TORCH_CHECK(
      weights_out.dim() == 2 && weights_out.size(0) == head_weights.size(0) &&
          weights_out.size(1) == head_weights.size(1) && weights_out.scalar_type() == at::kFloat,
      "weights_out must be float32 [tokens, heads]");

  const float scale_value = static_cast<float>(weight_scale);
  if (positions.scalar_type() == at::kInt) {
    launch_index_q<int32_t>(input, freqs_cis, positions, head_weights, payload, scale, weights_out, scale_value);
  } else {
    launch_index_q<int64_t>(input, freqs_cis, positions, head_weights, payload, scale, weights_out, scale_value);
  }
}

SGL_KERNEL_EXPORT void index_k_norm_rope_pack_store(
    const at::Tensor& input,
    const at::Tensor& norm_weight,
    double eps,
    const at::Tensor& freqs_cis,
    const at::Tensor& positions,
    const at::Tensor& loc,
    at::Tensor& cache,
    int64_t ratio) {
  CHECK_INPUT(input);
  CHECK_INPUT(norm_weight);
  CHECK_INPUT(freqs_cis);
  CHECK_INPUT(positions);
  CHECK_INPUT(loc);
  CHECK_INPUT(cache);
  TORCH_CHECK(input.dim() == 2 && input.size(1) == kHeadDim, "input must be [tokens, 128]");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16, "input must be bfloat16");
  TORCH_CHECK(norm_weight.dim() == 1 && norm_weight.size(0) == kHeadDim, "norm_weight must be [128]");
  TORCH_CHECK(
      norm_weight.scalar_type() == at::kBFloat16 || norm_weight.scalar_type() == at::kFloat,
      "norm_weight must be bfloat16 or float32");
  TORCH_CHECK(freqs_cis.dim() == 2 && freqs_cis.size(1) == kRopeDim, "freqs_cis must be [max_pos, 64]");
  TORCH_CHECK(freqs_cis.scalar_type() == at::kFloat, "freqs_cis must be float32");
  TORCH_CHECK(positions.dim() == 1 && positions.size(0) == input.size(0), "positions must be [tokens]");
  TORCH_CHECK(
      positions.scalar_type() == at::kInt || positions.scalar_type() == at::kLong, "positions must be int32 or int64");
  TORCH_CHECK(loc.dim() == 1 && loc.size(0) == input.size(0), "loc must be [tokens]");
  TORCH_CHECK(loc.scalar_type() == at::kLong, "loc must be int64");
  TORCH_CHECK(cache.dim() == 2 && cache.scalar_type() == at::kByte, "cache must be 2-D uint8");
  TORCH_CHECK(cache.size(1) % kSlotBytes == 0, "cache row size must be divisible by 68");
  TORCH_CHECK(ratio > 0 && (ratio & (ratio - 1)) == 0, "ratio must be a positive power of two");
  TORCH_CHECK(eps > 0.0, "eps must be positive");
  const int64_t page_size = cache.size(1) / kSlotBytes;
  TORCH_CHECK(page_size > 0, "cache page size must be positive");

  if (positions.scalar_type() == at::kInt) {
    if (norm_weight.scalar_type() == at::kBFloat16) {
      launch_index_k<bf16_t, int32_t>(
          input, norm_weight, freqs_cis, positions, loc, cache, ratio, page_size, static_cast<float>(eps));
    } else {
      launch_index_k<float, int32_t>(
          input, norm_weight, freqs_cis, positions, loc, cache, ratio, page_size, static_cast<float>(eps));
    }
  } else if (norm_weight.scalar_type() == at::kBFloat16) {
    launch_index_k<bf16_t, int64_t>(
        input, norm_weight, freqs_cis, positions, loc, cache, ratio, page_size, static_cast<float>(eps));
  } else {
    launch_index_k<float, int64_t>(
        input, norm_weight, freqs_cis, positions, loc, cache, ratio, page_size, static_cast<float>(eps));
  }
}
