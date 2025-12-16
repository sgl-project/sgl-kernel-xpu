#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cmath>
#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"
#include "cutlass/float8.h"

namespace at::native::xpu {

// SYCL helper for group reduce max using sub-groups
// Emulates CUDA's __shfl_xor_sync with masking for 16-thread groups
template <typename T>
inline T QuantGroupReduceMax(T val, sycl::nd_item<1> item, int tid) {
  auto sg = item.get_sub_group();
  uint32_t lane_id = sg.get_local_id()[0];

  // Perform butterfly reduction with XOR pattern
  // XOR with 8, 4, 2, 1 to reduce within 16 threads
  val = sycl::fmax(val, sycl::permute_group_by_xor(sg, val, 8));
  val = sycl::fmax(val, sycl::permute_group_by_xor(sg, val, 4));
  val = sycl::fmax(val, sycl::permute_group_by_xor(sg, val, 2));
  val = sycl::fmax(val, sycl::permute_group_by_xor(sg, val, 1));

  return val;
}

// Use SYCL native vector type for efficient loading
template <typename T, uint32_t N>
using vec_t = sycl::vec<T, N>;

template <
    typename T,
    typename DST_DTYPE,
    bool IS_COLUMN_MAJOR = false,
    bool SCALE_UE8M0 = false,
    typename scale_packed_t = std::conditional_t<SCALE_UE8M0, uint32_t, float>>
struct PerTokenGroupQuant8bitKernel : public __SYCL_KER_CONFIG_CONVENTION__ {
  PerTokenGroupQuant8bitKernel(
      const T* input,
      void* output_q,
      scale_packed_t* output_s,
      int group_size,
      int num_groups,
      int groups_per_block,
      float eps,
      float min_8bit,
      float max_8bit,
      int num_groups_per_row = 0,
      int scale_stride = 0,
      const int threads_per_group = 16)
      : input(input),
        output_q(output_q),
        output_s(output_s),
        group_size(group_size),
        num_groups(num_groups),
        groups_per_block(groups_per_block),
        eps(eps),
        min_8bit(min_8bit),
        max_8bit(max_8bit),
        num_groups_per_row(num_groups_per_row),
        scale_stride(scale_stride),
        threads_per_group(threads_per_group) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {}

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> item) const {
    // all the variable names refer to CUDA nomenclature for easier mapping
    // so 'groups' in the kernel refer to tensor groups for quantization rather than
    // SYCL work-groups/sub-groups
    const int64_t local_group_id = item.get_local_id(0) / threads_per_group;
    const int lane_id = item.get_local_id(0) % threads_per_group;

    const int64_t block_group_id = item.get_group(0) * groups_per_block;
    const int64_t global_group_id = block_group_id + local_group_id;
    const int64_t block_group_offset = global_group_id * group_size;

    float local_absmax = eps;

    using scale_element_t = std::conditional_t<SCALE_UE8M0, uint8_t, float>;
    static_assert(sizeof(scale_packed_t) % sizeof(scale_element_t) == 0);

    const T* group_input = input + block_group_offset;
    DST_DTYPE* group_output = static_cast<DST_DTYPE*>(output_q) + block_group_offset;
    scale_element_t* scale_output;

    if constexpr (IS_COLUMN_MAJOR) {
      const int num_elems_per_pack = static_cast<int>(sizeof(scale_packed_t) / sizeof(scale_element_t));
      const int row_idx = global_group_id / num_groups_per_row;
      const int col_idx_unpacked = global_group_id % num_groups_per_row;
      const int col_idx = col_idx_unpacked / num_elems_per_pack;
      const int pack_idx = col_idx_unpacked % num_elems_per_pack;
      scale_output = reinterpret_cast<scale_element_t*>(output_s) +
                     (col_idx * scale_stride * num_elems_per_pack + row_idx * num_elems_per_pack + pack_idx);
    } else {
      static_assert(!SCALE_UE8M0);
      scale_output = output_s + global_group_id;
    }

    constexpr uint32_t vec_size = 16 / sizeof(T);
    using vec_type = vec_t<T, vec_size>;

    // TODO: Handle case where group_size is not divisible by vec_size
    const int32_t num_vec_elems = group_size / vec_size;

    // First pass: find local_absmax
    for (int32_t i = lane_id; i < num_vec_elems; i += 16) {
      // Load vector using SYCL's native load operation
      vec_type input_vec;
      input_vec.load(
          0, sycl::multi_ptr<const T, sycl::access::address_space::global_space>(group_input + i * vec_size));

#pragma unroll
      for (uint32_t j = 0; j < vec_size; ++j) {
        float val = static_cast<float>(input_vec[j]);
        float abs_val = sycl::fabs(val);
        local_absmax = sycl::fmax(local_absmax, abs_val);
      }
    }

    // Reduce across the 16 threads in the group to find the maximum
    local_absmax = QuantGroupReduceMax(local_absmax, item, lane_id);

    // Calculate scale factor
    float y_s = local_absmax / max_8bit;
    scale_element_t y_s_quant;

    // Quantize the scale factor for UE8M0 format if needed
    if constexpr (SCALE_UE8M0) {
      float exp_s = sycl::ceil(sycl::log2(sycl::fmax(y_s, 1e-10f)));
      y_s = sycl::exp2(exp_s);
      // represent quantized scale as power of 2 exponent + 127 bias
      y_s_quant = static_cast<scale_element_t>(static_cast<int>(exp_s) + 127);
    } else {
      y_s_quant = y_s;
    }

    if (lane_id == 0) {
      *scale_output = y_s_quant;
    }

    // Second pass: quantize
    for (int32_t i = lane_id; i < num_vec_elems; i += 16) {
      // Load vector using SYCL's native load operation
      vec_type input_vec;
      // TODO: Optimize out the duplicate loads we do here and above for local_absmax
      input_vec.load(
          0, sycl::multi_ptr<const T, sycl::access::address_space::global_space>(group_input + i * vec_size));

#pragma unroll
      for (uint32_t j = 0; j < vec_size; ++j) {
        float val = static_cast<float>(input_vec[j]);
        float q_val = sycl::fmin(sycl::fmax(val / y_s, min_8bit), max_8bit);

        // Special handling for FP8 types using CUTLASS
        if constexpr (std::is_same_v<DST_DTYPE, cutlass::float_e4m3_t>) {
          // TODO: Remove CUTLASS emulation of float e4m3_t and use native SYCL FP8 when available
          group_output[i * vec_size + j] = cutlass::float_e4m3_t::from_float(q_val);
        } else {
          group_output[i * vec_size + j] = static_cast<DST_DTYPE>(q_val);
        }
      }
    }
  }

 private:
  const T* input;
  void* output_q;
  scale_packed_t* output_s;
  int group_size;
  int num_groups;
  int groups_per_block;
  float eps;
  float min_8bit;
  float max_8bit;
  int num_groups_per_row;
  int scale_stride;
  const int threads_per_group;
};

void sgl_per_token_group_quant_8bit(
    torch::Tensor input,
    torch::Tensor output_q,
    torch::Tensor output_s,
    int64_t group_size,
    double eps,
    double min_8bit,
    double max_8bit,
    bool scale_ue8m0) {
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(output_q);

  const int num_groups = input.numel() / group_size;

  CHECK_EQ(input.numel() % group_size, 0);
  CHECK_EQ(output_s.dim(), 2);

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

  auto dst_type = output_q.scalar_type();
  const int num_blocks = num_groups / groups_per_block;
  const int num_threads = groups_per_block * THREADS_PER_GROUP;

  const bool is_column_major = output_s.stride(0) < output_s.stride(1);
  const int hidden_dim = input.size(input.dim() - 1);
  const int num_groups_per_row = hidden_dim / group_size;
  const int scale_stride = output_s.stride(1);

  // Check for supported dtypes to avoid silent dispatch failure
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Half || input.scalar_type() == at::ScalarType::BFloat16 ||
          input.scalar_type() == at::ScalarType::Float,
      "sgl_per_token_group_quant_8bit: input dtype must be Float16, BFloat16, or Float32, got ",
      input.scalar_type());

  TORCH_CHECK(
      dst_type == at::ScalarType::Char || dst_type == at::ScalarType::Float8_e4m3fn,
      "sgl_per_token_group_quant_8bit: output_q dtype must be Int8 or Float8_e4m3fn, got ",
      dst_type);

  sycl::range<1> global_range(num_blocks * num_threads);
  sycl::range<1> local_range(num_threads);

#define LAUNCH_KERNEL(T, DST_DTYPE)                                            \
  do {                                                                         \
    if (is_column_major) {                                                     \
      if (scale_ue8m0) {                                                       \
        auto kernel = PerTokenGroupQuant8bitKernel<T, DST_DTYPE, true, true>(  \
            static_cast<const T*>(input.data_ptr()),                           \
            output_q.data_ptr(),                                               \
            static_cast<uint32_t*>(output_s.data_ptr()),                       \
            group_size,                                                        \
            num_groups,                                                        \
            groups_per_block,                                                  \
            static_cast<float>(eps),                                           \
            static_cast<float>(min_8bit),                                      \
            static_cast<float>(max_8bit),                                      \
            num_groups_per_row,                                                \
            scale_stride,                                                      \
            THREADS_PER_GROUP);                                                \
        sycl_kernel_submit(global_range, local_range, queue, kernel);          \
      } else {                                                                 \
        auto kernel = PerTokenGroupQuant8bitKernel<T, DST_DTYPE, true, false>( \
            static_cast<const T*>(input.data_ptr()),                           \
            output_q.data_ptr(),                                               \
            static_cast<float*>(output_s.data_ptr()),                          \
            group_size,                                                        \
            num_groups,                                                        \
            groups_per_block,                                                  \
            static_cast<float>(eps),                                           \
            static_cast<float>(min_8bit),                                      \
            static_cast<float>(max_8bit),                                      \
            num_groups_per_row,                                                \
            scale_stride,                                                      \
            THREADS_PER_GROUP);                                                \
        sycl_kernel_submit(global_range, local_range, queue, kernel);          \
      }                                                                        \
    } else {                                                                   \
      assert(!scale_ue8m0);                                                    \
      auto kernel = PerTokenGroupQuant8bitKernel<T, DST_DTYPE, false>(         \
          static_cast<const T*>(input.data_ptr()),                             \
          output_q.data_ptr(),                                                 \
          static_cast<float*>(output_s.data_ptr()),                            \
          group_size,                                                          \
          num_groups,                                                          \
          groups_per_block,                                                    \
          static_cast<float>(eps),                                             \
          static_cast<float>(min_8bit),                                        \
          static_cast<float>(max_8bit));                                       \
      sycl_kernel_submit(global_range, local_range, queue, kernel);            \
    }                                                                          \
  } while (0)

  // Dispatch based on input and output types
  if (input.scalar_type() == at::ScalarType::Half) {
    if (dst_type == at::ScalarType::Char) {
      LAUNCH_KERNEL(sycl::half, int8_t);
    } else if (dst_type == at::ScalarType::Float8_e4m3fn) {
      // Use CUTLASS float_e4m3_t for FP8 E4M3 support
      LAUNCH_KERNEL(sycl::half, cutlass::float_e4m3_t);
    }
  } else if (input.scalar_type() == at::ScalarType::BFloat16) {
    if (dst_type == at::ScalarType::Char) {
      LAUNCH_KERNEL(sycl::ext::oneapi::bfloat16, int8_t);
    } else if (dst_type == at::ScalarType::Float8_e4m3fn) {
      // Use CUTLASS float_e4m3_t for FP8 E4M3 support
      LAUNCH_KERNEL(sycl::ext::oneapi::bfloat16, cutlass::float_e4m3_t);
    }
  } else if (input.scalar_type() == at::ScalarType::Float) {
    if (dst_type == at::ScalarType::Char) {
      LAUNCH_KERNEL(float, int8_t);
    } else if (dst_type == at::ScalarType::Float8_e4m3fn) {
      // Use CUTLASS float_e4m3_t for FP8 E4M3 support with proper conversion
      LAUNCH_KERNEL(float, cutlass::float_e4m3_t);
    }
  }

#undef LAUNCH_KERNEL
}

}  // namespace at::native::xpu
