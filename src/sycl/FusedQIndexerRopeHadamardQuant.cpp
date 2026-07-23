#include <ATen/ATen.h>
#include <cutlass/float8.h>

#include <cstdint>
#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"

using cutlass::float_e4m3_t;

namespace {

constexpr int kSubGroupSize = 32;
constexpr int kBlockSize = 128;
constexpr int kWarpsPerBlock = kBlockSize / kSubGroupSize;
constexpr int kHeadDim = 128;
constexpr int kRopeDim = 64;
constexpr int kVecSize = 4;
constexpr int kRopeSize = kRopeDim / kVecSize;
constexpr float kFp8E4M3Max = 448.0f;
constexpr float kHadamardScale = 0.08838834764831845f;  // 1/sqrt(128)

template <typename T>
inline float to_float(T v) {
  return static_cast<float>(v);
}

template <typename DType, typename PosT>
struct FusedQIndexerRopeHadamardQuantKernel {
  const DType* q_input;
  float_e4m3_t* q_fp8;
  const DType* weight;
  float* weights_out;
  const float* rope_cache;
  const PosT* positions;
  int64_t weight_stride_batch;
  uint32_t batch_size;
  uint32_t num_heads;
  float weight_scale;

  [[sycl::reqd_sub_group_size(kSubGroupSize)]] void operator()(sycl::nd_item<1> item) const {
    const auto warp_id = item.get_local_id(0) / kSubGroupSize;
    const auto lane_id = item.get_local_id(0) % kSubGroupSize;
    const uint32_t work_id = static_cast<uint32_t>(item.get_group(0)) * kWarpsPerBlock + warp_id;
    const uint32_t total_works = batch_size * num_heads;

    if (work_id >= total_works) {
      return;
    }

    const uint32_t batch_id = work_id / num_heads;
    const uint32_t head_id = work_id - batch_id * num_heads;

    const DType* input_ptr = q_input + static_cast<int64_t>(work_id) * kHeadDim;
    float_e4m3_t* output_ptr = q_fp8 + static_cast<int64_t>(work_id) * kHeadDim;
    const auto position = static_cast<int32_t>(positions[batch_id]);
    const float* freq_row = rope_cache + static_cast<int64_t>(position) * kRopeDim;
    const float weight_val = to_float(weight[static_cast<int64_t>(batch_id) * weight_stride_batch + head_id]);

    float data[kVecSize];
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      data[i] = to_float(input_ptr[lane_id * kVecSize + i]);
    }

    const bool is_rope_lane = lane_id >= (kSubGroupSize - kRopeSize);
    if (is_rope_lane) {
      const int rope_lane = lane_id - (kSubGroupSize - kRopeSize);
      const float* f = freq_row + rope_lane * kVecSize;

      const float x_real = data[0];
      const float x_imag = data[1];
      const float y_real = data[2];
      const float y_imag = data[3];
      const float fxr = f[0];
      const float fxi = f[1];
      const float fyr = f[2];
      const float fyi = f[3];

      data[0] = x_real * fxr - x_imag * fxi;
      data[1] = x_real * fxi + x_imag * fxr;
      data[2] = y_real * fyr - y_imag * fyi;
      data[3] = y_real * fyi + y_imag * fyr;
    }

    {
      const float a0 = data[0];
      const float a1 = data[1];
      const float a2 = data[2];
      const float a3 = data[3];
      data[0] = a0 + a1;
      data[1] = a0 - a1;
      data[2] = a2 + a3;
      data[3] = a2 - a3;
    }
    {
      const float a0 = data[0];
      const float a1 = data[1];
      const float a2 = data[2];
      const float a3 = data[3];
      data[0] = a0 + a2;
      data[1] = a1 + a3;
      data[2] = a0 - a2;
      data[3] = a1 - a3;
    }

    const auto sg = item.get_sub_group();
#pragma unroll
    for (int mask = 1; mask < kSubGroupSize; mask <<= 1) {
#pragma unroll
      for (int i = 0; i < kVecSize; ++i) {
        const float other = sycl::permute_group_by_xor(sg, data[i], mask);
        data[i] = (lane_id & mask) ? (other - data[i]) : (data[i] + other);
      }
    }

#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      data[i] *= kHadamardScale;
    }

    float local_max = sycl::fabs(data[0]);
#pragma unroll
    for (int i = 1; i < kVecSize; ++i) {
      local_max = sycl::fmax(local_max, sycl::fabs(data[i]));
    }

    const float abs_max = sycl::reduce_over_group(sg, local_max, sycl::maximum<float>());
    const float scale = sycl::fmax(1e-4f, abs_max) / kFp8E4M3Max;
    const float inv_scale = 1.0f / scale;

#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const float q = sycl::fmax(sycl::fmin(data[i] * inv_scale, kFp8E4M3Max), -kFp8E4M3Max);
      output_ptr[lane_id * kVecSize + i] = static_cast<float_e4m3_t>(q);
    }

    if (lane_id == 0) {
      weights_out[work_id] = weight_val * weight_scale * scale;
    }
  }
};

template <typename T>
struct SyclInputType;

template <>
struct SyclInputType<at::Half> {
  using type = sycl::half;
};

template <>
struct SyclInputType<at::BFloat16> {
  using type = sycl::ext::oneapi::bfloat16;
};

template <>
struct SyclInputType<float> {
  using type = float;
};

template <typename DType, typename PosT>
void launch_fused_q_indexer_rope_hadamard_quant(
    const at::Tensor& q_input,
    at::Tensor& q_fp8,
    const at::Tensor& weight,
    at::Tensor& weights_out,
    double weight_scale,
    const at::Tensor& rope_cache,
    const at::Tensor& positions) {
  const auto batch_size = static_cast<uint32_t>(q_input.size(0));
  const auto num_heads = static_cast<uint32_t>(q_input.size(1));
  if (batch_size == 0) {
    return;
  }

  const uint32_t total_works = batch_size * num_heads;
  const uint32_t num_blocks = (total_works + kWarpsPerBlock - 1) / kWarpsPerBlock;

  auto& queue = dpcppGetCurrentQueue();
  auto kernel = FusedQIndexerRopeHadamardQuantKernel<DType, PosT>{
      static_cast<const DType*>(q_input.data_ptr()),
      static_cast<float_e4m3_t*>(q_fp8.data_ptr()),
      static_cast<const DType*>(weight.data_ptr()),
      static_cast<float*>(weights_out.data_ptr()),
      static_cast<const float*>(rope_cache.data_ptr()),
      static_cast<const PosT*>(positions.data_ptr()),
      weight.stride(0),
      batch_size,
      num_heads,
      static_cast<float>(weight_scale)};

  sycl_kernel_submit(sycl::range<1>(num_blocks * kBlockSize), sycl::range<1>(kBlockSize), queue, kernel);
}

}  // namespace

void fused_q_indexer_rope_hadamard_quant(
    const at::Tensor& q_input,
    at::Tensor& q_fp8,
    const at::Tensor& weight,
    at::Tensor& weights_out,
    double weight_scale,
    const at::Tensor& rope_cache,
    const at::Tensor& positions) {
  CHECK_DEVICE(q_input);
  CHECK_DEVICE(q_fp8);
  CHECK_DEVICE(weight);
  CHECK_DEVICE(weights_out);
  CHECK_DEVICE(rope_cache);
  CHECK_DEVICE(positions);

  TORCH_CHECK(q_input.dim() == 3, "q_input must be [B, H, 128]");
  TORCH_CHECK(q_fp8.dim() == 3, "q_fp8 must be [B, H, 128]");
  TORCH_CHECK(weight.dim() == 2, "weight must be [B, H]");
  TORCH_CHECK(weights_out.dim() == 3, "weights_out must be [B, H, 1]");
  TORCH_CHECK(rope_cache.dim() == 2, "rope_cache must be [max_pos, 64]");
  TORCH_CHECK(positions.dim() == 1, "positions must be [B]");

  TORCH_CHECK(q_input.size(2) == kHeadDim, "q_input head_dim must be 128, got ", q_input.size(2));
  TORCH_CHECK(q_fp8.size(2) == kHeadDim, "q_fp8 head_dim must be 128, got ", q_fp8.size(2));
  TORCH_CHECK(rope_cache.size(1) == kRopeDim, "rope_cache last dim must be 64, got ", rope_cache.size(1));
  TORCH_CHECK(weights_out.size(2) == 1, "weights_out last dim must be 1");

  TORCH_CHECK(q_input.size(0) == q_fp8.size(0) && q_input.size(1) == q_fp8.size(1), "q_input/q_fp8 shape mismatch");
  TORCH_CHECK(q_input.size(0) == weight.size(0) && q_input.size(1) == weight.size(1), "q_input/weight shape mismatch");
  TORCH_CHECK(
      q_input.size(0) == weights_out.size(0) && q_input.size(1) == weights_out.size(1),
      "q_input/weights_out shape mismatch");
  TORCH_CHECK(q_input.size(0) == positions.size(0), "q_input and positions batch mismatch");

  TORCH_CHECK(
      q_input.scalar_type() == at::kHalf || q_input.scalar_type() == at::kBFloat16 ||
          q_input.scalar_type() == at::kFloat,
      "q_input must be fp16/bf16/fp32");
  TORCH_CHECK(weight.scalar_type() == q_input.scalar_type(), "weight dtype must match q_input dtype");
  TORCH_CHECK(q_fp8.scalar_type() == at::kFloat8_e4m3fn, "q_fp8 must be Float8_e4m3fn");
  TORCH_CHECK(weights_out.scalar_type() == at::kFloat, "weights_out must be float32");
  TORCH_CHECK(rope_cache.scalar_type() == at::kFloat, "rope_cache must be float32");
  TORCH_CHECK(
      positions.scalar_type() == at::kInt || positions.scalar_type() == at::kLong, "positions must be int32 or int64");

  TORCH_CHECK(
      q_input.stride(2) == 1 && q_input.stride(1) == kHeadDim, "q_input must be contiguous in [B,H,128] layout");
  TORCH_CHECK(q_fp8.stride(2) == 1 && q_fp8.stride(1) == kHeadDim, "q_fp8 must be contiguous in [B,H,128] layout");
  TORCH_CHECK(weights_out.stride(2) == 1, "weights_out last dim must be contiguous");
  TORCH_CHECK(weight.stride(1) == 1, "weight last dim must be contiguous");
  TORCH_CHECK(rope_cache.stride(1) == 1, "rope_cache last dim must be contiguous");
  TORCH_CHECK(positions.is_contiguous(), "positions must be contiguous");

  const int64_t expected_q_stride0 = q_input.size(1) * kHeadDim;
  TORCH_CHECK(
      q_input.stride(0) == expected_q_stride0,
      "q_input must be contiguous (B,H,128); got stride[0]=",
      q_input.stride(0));
  TORCH_CHECK(
      q_fp8.stride(0) == expected_q_stride0, "q_fp8 must be contiguous (B,H,128); got stride[0]=", q_fp8.stride(0));

  DISPATCH_FLOAT_TYPES(q_input.scalar_type(), "fused_q_indexer_rope_hadamard_quant", [&]() {
    using sycl_in_t = typename SyclInputType<scalar_t>::type;
    if (positions.scalar_type() == at::kInt) {
      launch_fused_q_indexer_rope_hadamard_quant<sycl_in_t, int32_t>(
          q_input, q_fp8, weight, weights_out, weight_scale, rope_cache, positions);
    } else {
      launch_fused_q_indexer_rope_hadamard_quant<sycl_in_t, int64_t>(
          q_input, q_fp8, weight, weights_out, weight_scale, rope_cache, positions);
    }
  });
}
