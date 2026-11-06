#include <ATen/ATen.h>
#include <cutlass/float8.h>

#include <cstdint>
#include <sycl/sycl.hpp>

#include "SGLKernelPerf.h"
#include "SYCLHelpers.h"
#include "Utils.h"
#include "sgl_kernel_export.h"

using cutlass::float_e4m3_t;

namespace {

constexpr int kSubGroupSize = 32;  // one warp on XPU/SYCL
constexpr int kBlockSize = 128;    // 4 warps per block; matches the 128-dim head layout
constexpr int kWarpsPerBlock = kBlockSize / kSubGroupSize;
constexpr int kHeadDim = 128;  // fused Q indexer contract: [B, H, 128]
constexpr int kRopeDim = 64;   // RoPE uses the last 64 dims of the 128-dim head
constexpr int kVecSize = 4;    // per-lane vectorized load/store width
constexpr int kRopeSize = kRopeDim / kVecSize;
constexpr float kFp8E4M3Max = 448.0f;                   // max finite value for FP8 E4M3
constexpr float kHadamardScale = 0.08838834764831845f;  // 1/sqrt(128) for FWHT normalization

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
      const float x0 = data[0] + data[1];
      const float x1 = data[0] - data[1];
      const float x2 = data[2] + data[3];
      const float x3 = data[2] - data[3];
      data[0] = x0 + x2;
      data[1] = x1 + x3;
      data[2] = x0 - x2;
      data[3] = x1 - x3;
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

SGL_KERNEL_EXPORT void fused_q_indexer_rope_hadamard_quant(
    const at::Tensor& q_input,
    at::Tensor& q_fp8,
    const at::Tensor& weight,
    at::Tensor& weights_out,
    double weight_scale,
    const at::Tensor& rope_cache,
    const at::Tensor& positions) {
  CHECK_INPUT(3, q_input);
  CHECK_DTYPES(q_input, at::kFloat, at::kHalf, at::kBFloat16);
  CHECK_INPUT(3, at::kFloat8_e4m3fn, q_fp8);
  CHECK_INPUT(2, q_input.scalar_type(), weight);
  CHECK_INPUT(3, at::kFloat, weights_out);
  CHECK_INPUT(2, at::kFloat, rope_cache);
  CHECK_INPUT(1, positions);
  CHECK_DTYPES(positions, at::kInt, at::kLong);

  CHECK_SAME_SHAPE(q_input, q_fp8);
  TORCH_CHECK(
      q_input.size(0) == weight.size(0) && q_input.size(1) == weight.size(1), "q_input and weight batch/head mismatch");
  TORCH_CHECK(
      weights_out.size(0) == q_input.size(0) && weights_out.size(1) == q_input.size(1) && weights_out.size(2) == 1,
      "weights_out must be [B, H, 1]");
  TORCH_CHECK(q_input.size(0) == positions.size(0), "q_input and positions batch mismatch");

  TORCH_CHECK(q_input.size(2) == kHeadDim, "q_input head_dim must be 128, got ", q_input.size(2));
  TORCH_CHECK(rope_cache.size(1) == kRopeDim, "rope_cache last dim must be 64, got ", rope_cache.size(1));

  TORCH_CHECK(
      q_input.stride(2) == 1 && q_input.stride(1) == kHeadDim, "q_input must be contiguous in [B,H,128] layout");
  TORCH_CHECK(q_fp8.stride(2) == 1 && q_fp8.stride(1) == kHeadDim, "q_fp8 must be contiguous in [B,H,128] layout");

  TORCH_CHECK(weights_out.stride(2) == 1, "weights_out last dim must be contiguous");
  TORCH_CHECK(weights_out.stride(1) == 1, "weights_out H dim must be contiguous");
  TORCH_CHECK(weights_out.stride(0) == weights_out.size(1), "weights_out must be contiguous in [B,H,1] layout");
  TORCH_CHECK(weight.stride(1) == 1, "weight last dim must be contiguous");

  TORCH_CHECK(rope_cache.stride(1) == 1, "rope_cache last dim must be contiguous");
  TORCH_CHECK(rope_cache.stride(0) == kRopeDim, "rope_cache must be contiguous in [max_pos,64] layout");

  const int64_t expected_q_stride0 = q_input.size(1) * kHeadDim;
  TORCH_CHECK(
      q_input.stride(0) == expected_q_stride0,
      "q_input must be contiguous (B,H,128); got stride[0]=",
      q_input.stride(0));
  TORCH_CHECK(
      q_fp8.stride(0) == expected_q_stride0, "q_fp8 must be contiguous (B,H,128); got stride[0]=", q_fp8.stride(0));

#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
  const int64_t B = q_input.size(0);
  const int64_t H = q_input.size(1);
  // rope (~4*rope_dim per B*H) + hadamard (~2*head_dim*log2(head_dim)/head_dim*head_dim ≈ 2*head_dim*log2 flops)
  // + fp8 quant (~2 per element). Simplify: hadamard = 2*log2(128) = 14 flops/elem => 14*head_dim per B*H.
  const double flops = static_cast<double>(B) * static_cast<double>(H) *
                       (4.0 * static_cast<double>(kRopeDim) + 14.0 * static_cast<double>(kHeadDim) +
                        2.0 * static_cast<double>(kHeadDim));
  const double q_elem = static_cast<double>(q_input.element_size());
  // Read q_input, read weight, read rope_cache, write q_fp8 (1 byte/elem), write weights_out (fp32).
  const double bytes =
      static_cast<double>(B) * static_cast<double>(H) * static_cast<double>(kHeadDim) * q_elem +
      static_cast<double>(B) * static_cast<double>(H) * static_cast<double>(weight.element_size()) +
      static_cast<double>(B) * static_cast<double>(kRopeDim) * static_cast<double>(rope_cache.element_size()) +
      static_cast<double>(B) * static_cast<double>(H) * static_cast<double>(kHeadDim) * 1.0 +
      static_cast<double>(B) * static_cast<double>(H) * 4.0;
  auto profiling_queue = at::xpu::getCurrentXPUStream().queue();
  SGL_KERNEL_PERF_SCOPE("fused_q_indexer_rope_hadamard_quant", profiling_queue, bytes, flops);
#endif

  SYCL_DISPATCH_FLOATING_TYPES_V2(q_input.scalar_type(), "fused_q_indexer_rope_hadamard_quant", [&]() {
    AT_DISPATCH_INDEX_TYPES(positions.scalar_type(), "fused_q_indexer_rope_hadamard_quant", [&]() {
      launch_fused_q_indexer_rope_hadamard_quant<scalar_t, index_t>(
          q_input, q_fp8, weight, weights_out, weight_scale, rope_cache, positions);
    });
  });
}
