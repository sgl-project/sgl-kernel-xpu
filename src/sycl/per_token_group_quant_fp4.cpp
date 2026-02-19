// SPDX-License-Identifier: Apache-2.0
/*
 * SYCL kernel for per-token group quantization to MXFP4 (E2M1) format.
 *
 * MXFP4 follows the OpenCompute MX (Microscaling) format specification:
 * - Data type: E2M1 (4-bit float with 2-bit exponent, 1-bit mantissa)
 * - Block size: 32 elements per scale factor
 * - Scale format: UE8M0 (unsigned 8-bit exponent-only, no mantissa)
 *
 * E2M1 representable values (magnitude): 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
 * With sign bit, we have 16 total values.
 *
 * Bit layout of E2M1:
 *   Bit 3: Sign (0 = positive, 1 = negative)
 *   Bits 0-2: Magnitude index (0-7)
 *
 * Two FP4 values are packed into a single uint8_t:
 *   - Lower nibble (bits 0-3): First value
 *   - Upper nibble (bits 4-7): Second value
 */

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cmath>
#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"

namespace at::native::xpu {

constexpr float FLOAT4_E2M1_MAX = 6.0f;

template <typename T>
inline T QuantGroupReduceMaxFP4(T val, sycl::nd_item<1> item, int lane_id_in_quant_group) {
  auto sg = item.get_sub_group();

  val = sycl::fmax(val, sycl::permute_group_by_xor(sg, val, 8));
  val = sycl::fmax(val, sycl::permute_group_by_xor(sg, val, 4));
  val = sycl::fmax(val, sycl::permute_group_by_xor(sg, val, 2));
  val = sycl::fmax(val, sycl::permute_group_by_xor(sg, val, 1));

  return val;
}

// E2M1 format (4-bit float): 1 sign bit, 2 exponent bits, 1 mantissa bit
// Encoding: exp=00 (subnormal), exp=01/10/11 (normal with bias=1)
// Result: bits[3]=sign, bits[2:1]=exponent, bits[0]=mantissa
inline uint8_t quantize_to_e2m1(float val) {
  uint8_t sign = (val < 0.0f) ? 1 : 0;
  float abs_val = sycl::fabs(val);

  uint8_t code;
  if (abs_val <= 0.25f) {
    code = 0b000;  // 0.0 (subnormal: exp=00, m=0)
  } else if (abs_val <= 0.75f) {
    code = 0b001;  // 0.5 (subnormal: exp=00, m=1)
  } else if (abs_val <= 1.25f) {
    code = 0b010;  // 1.0 (exp=01, m=0)
  } else if (abs_val <= 1.75f) {
    code = 0b011;  // 1.5 (exp=01, m=1)
  } else if (abs_val <= 2.5f) {
    code = 0b100;  // 2.0 (exp=10, m=0)
  } else if (abs_val <= 3.5f) {
    code = 0b101;  // 3.0 (exp=10, m=1)
  } else if (abs_val <= 5.0f) {
    code = 0b110;  // 4.0 (exp=11, m=0)
  } else {
    code = 0b111;  // 6.0 (exp=11, m=1)
  }

  return (sign << 3) | code;
}

// Use SYCL native vector type for efficient loading
template <typename T, uint32_t N>
using vec_t = sycl::vec<T, N>;

// Compile-time constants for group sizes
template <int GROUP_SIZE>
struct FP4GroupSizeTraits {
  static constexpr int THREADS_PER_GROUP = 16;
  static constexpr int SUB_GROUP_SIZE = 32;
};

template <typename T, int GROUP_SIZE = 32>
struct PerTokenGroupQuantFP4Kernel : public __SYCL_KER_CONFIG_CONVENTION__ {
  static constexpr uint32_t VEC_SIZE = 16 / sizeof(T);
  static constexpr int32_t NUM_VEC_ELEMS = GROUP_SIZE / VEC_SIZE;
  static constexpr int32_t THREADS_PER_GROUP = FP4GroupSizeTraits<GROUP_SIZE>::THREADS_PER_GROUP;
  static constexpr int32_t VECS_PER_THREAD = (NUM_VEC_ELEMS + THREADS_PER_GROUP - 1) / THREADS_PER_GROUP;

  PerTokenGroupQuantFP4Kernel(
      const T* input, uint8_t* output_q, uint8_t* output_s, int num_groups, int groups_per_block, float eps)
      : input(input),
        output_q(output_q),
        output_s(output_s),
        num_groups(num_groups),
        groups_per_block(groups_per_block),
        eps(eps) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {}

  [[sycl::reqd_sub_group_size(32)]] void operator()(sycl::nd_item<1> item) const {
    const int64_t local_group_id = item.get_local_id(0) / THREADS_PER_GROUP;
    const int lane_id = item.get_local_id(0) % THREADS_PER_GROUP;

    const int64_t block_group_id = item.get_group(0) * groups_per_block;
    const int64_t global_group_id = block_group_id + local_group_id;

    if (global_group_id >= num_groups) return;

    const int64_t block_group_offset = global_group_id * GROUP_SIZE;

    float local_absmax = eps;

    const T* group_input = input + block_group_offset;
    // Output is packed FP4 (2 values per byte), so offset is halved
    uint8_t* group_output = output_q + (block_group_offset / 2);

    // Calculate scale output position (row-major layout)
    // Each row has num_groups_per_row scales, stored contiguously
    uint8_t* scale_output = output_s + global_group_id;

    using vec_type = vec_t<T, VEC_SIZE>;
    using float_vec_type = vec_t<float, VEC_SIZE>;

    vec_type input_vecs[VECS_PER_THREAD];
    float_vec_type input_vals[VECS_PER_THREAD];

#pragma unroll
    for (int32_t v = 0; v < VECS_PER_THREAD; ++v) {
      const int32_t i = lane_id + v * THREADS_PER_GROUP;
      if (i < NUM_VEC_ELEMS) {
        input_vecs[v].load(
            0, sycl::multi_ptr<const T, sycl::access::address_space::global_space>(group_input + i * VEC_SIZE));

#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
          float val = static_cast<float>(input_vecs[v][j]);
          input_vals[v][j] = val;
          local_absmax = sycl::fmax(local_absmax, sycl::fabs(val));
        }
      }
    }

    // Reduce across the threads in the quantization group to find the maximum
    local_absmax = QuantGroupReduceMaxFP4(local_absmax, item, lane_id);

    float log2_scale = sycl::ceil(sycl::log2(sycl::fmax(local_absmax / FLOAT4_E2M1_MAX, 1e-10f)));
    int clamped_exponent = sycl::clamp(static_cast<int>(log2_scale), -127, 127);
    float scale_value = sycl::exp2(static_cast<float>(clamped_exponent));

    // Store scale as UE8M0: exponent + 127 bias
    uint8_t scale_ue8m0 = static_cast<uint8_t>(clamped_exponent + 127);

    if (lane_id == 0) {
      *scale_output = scale_ue8m0;
    }

    const float inv_scale = 1.0f / scale_value;

    // Second pass: quantize and pack values
    // Each thread processes VEC_SIZE elements at a time
    // Two FP4 values are packed into one byte
#pragma unroll
    for (int32_t v = 0; v < VECS_PER_THREAD; ++v) {
      const int32_t i = lane_id + v * THREADS_PER_GROUP;
      if (i < NUM_VEC_ELEMS) {
        // Process VEC_SIZE elements, packing pairs into bytes
        uint8_t packed_output[VEC_SIZE / 2];

#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; j += 2) {
          float val0 = input_vals[v][j] * inv_scale;
          float val1 = input_vals[v][j + 1] * inv_scale;

          uint8_t q0 = quantize_to_e2m1(val0);
          uint8_t q1 = quantize_to_e2m1(val1);

          // Pack: first value in lower nibble, second in upper nibble
          packed_output[j / 2] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
        }

        // Store packed output
        // Each vec of VEC_SIZE elements becomes VEC_SIZE/2 packed bytes
        uint8_t* out_ptr = group_output + i * (VEC_SIZE / 2);
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE / 2; ++j) {
          out_ptr[j] = packed_output[j];
        }
      }
    }
  }

 private:
  const T* input;
  uint8_t* output_q;
  uint8_t* output_s;
  int num_groups;
  int groups_per_block;
  float eps;
};

void sgl_per_token_group_quant_fp4(
    torch::Tensor input, torch::Tensor output_q, torch::Tensor output_s, int64_t group_size, double eps) {
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(output_q);

  TORCH_CHECK(group_size == 32, "sgl_per_token_group_quant_fp4: group_size must be 32 for MXFP4", group_size);

  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Half || input.scalar_type() == at::ScalarType::BFloat16 ||
          input.scalar_type() == at::ScalarType::Float,
      "sgl_per_token_group_quant_fp4: input dtype must be Float16, BFloat16, or Float32, got ",
      input.scalar_type());

  TORCH_CHECK(output_q.scalar_type() == at::ScalarType::Byte, "output_q must be uint8 (packed FP4)");
  TORCH_CHECK(output_s.scalar_type() == at::ScalarType::Byte, "output_s must be uint8 (UE8M0 scales)");

  const int num_groups = input.numel() / group_size;
  CHECK_EQ(input.numel() % group_size, 0);

  // Output should be half the size (2 FP4 values per byte)
  CHECK_EQ(output_q.numel(), input.numel() / 2);

  auto queue = dpcppGetCurrentQueue();

  constexpr int THREADS_PER_GROUP = 16;

  int groups_per_block = 1;

  if (num_groups % 16 == 0) {
    groups_per_block = 16;
  } else if (num_groups % 8 == 0) {
    groups_per_block = 8;
  } else if (num_groups % 4 == 0) {
    groups_per_block = 4;
  } else if (num_groups % 2 == 0) {
    groups_per_block = 2;
  }

  const int num_blocks = num_groups / groups_per_block;
  const int num_threads = groups_per_block * THREADS_PER_GROUP;

  sycl::range<1> global_range(num_blocks * num_threads);
  sycl::range<1> local_range(num_threads);

#define LAUNCH_FP4_KERNEL_WITH_GROUP_SIZE(T, GS)                  \
  do {                                                            \
    auto kernel = PerTokenGroupQuantFP4Kernel<T, GS>(             \
        static_cast<const T*>(input.data_ptr()),                  \
        static_cast<uint8_t*>(output_q.data_ptr()),               \
        static_cast<uint8_t*>(output_s.data_ptr()),               \
        num_groups,                                               \
        groups_per_block,                                         \
        static_cast<float>(eps));                                 \
    sycl_kernel_submit(global_range, local_range, queue, kernel); \
  } while (0)

#define LAUNCH_FP4_KERNEL(T)                                        \
  do {                                                              \
    switch (group_size) {                                           \
      case 32:                                                      \
        LAUNCH_FP4_KERNEL_WITH_GROUP_SIZE(T, 32);                   \
        break;                                                      \
      case 64:                                                      \
        LAUNCH_FP4_KERNEL_WITH_GROUP_SIZE(T, 64);                   \
        break;                                                      \
      case 128:                                                     \
        LAUNCH_FP4_KERNEL_WITH_GROUP_SIZE(T, 128);                  \
        break;                                                      \
      default:                                                      \
        TORCH_CHECK(false, "Unsupported group_size: ", group_size); \
    }                                                               \
  } while (0)

  // Dispatch based on input type
  if (input.scalar_type() == at::ScalarType::Half) {
    LAUNCH_FP4_KERNEL(sycl::half);
  } else if (input.scalar_type() == at::ScalarType::BFloat16) {
    LAUNCH_FP4_KERNEL(sycl::ext::oneapi::bfloat16);
  } else if (input.scalar_type() == at::ScalarType::Float) {
    LAUNCH_FP4_KERNEL(float);
  }

#undef LAUNCH_FP4_KERNEL
#undef LAUNCH_FP4_KERNEL_WITH_GROUP_SIZE
}

}  // namespace at::native::xpu
