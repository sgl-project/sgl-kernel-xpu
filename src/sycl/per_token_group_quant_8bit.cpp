#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cmath>
#include <sycl/sycl.hpp>

#include "cutlass/float8.h"
#include "SYCLHelpers.h"
#include "Utils.h"

namespace at::native::xpu {

// SYCL helper for group reduce max using sub-groups
// Emulates CUDA's __shfl_xor_sync with masking for 16-thread groups
template <typename T>
inline T GroupReduceMax(T val, sycl::nd_item<1> item, int tid) {
  auto sg = item.get_sub_group();
  uint32_t lane_id = sg.get_local_id()[0];

  // CUDA uses masks to operate on half-warps (16 threads)
  // Upper half: lanes 16-31, Lower half: lanes 0-15
  bool is_upper_half = (tid % 32) >= 16;

  // Perform butterfly reduction with XOR pattern
  // XOR with 8, 4, 2, 1 to reduce within 16 threads
  val = sycl::fmax(val, sycl::permute_group_by_xor(sg, val, 8));
  val = sycl::fmax(val, sycl::permute_group_by_xor(sg, val, 4));
  val = sycl::fmax(val, sycl::permute_group_by_xor(sg, val, 2));
  val = sycl::fmax(val, sycl::permute_group_by_xor(sg, val, 1));

  return val;
}

// Simple vector type for loading data
template <typename T, uint32_t N>
struct vec_t {
  T data[N];

  T& operator[](uint32_t i) { return data[i]; }
  const T& operator[](uint32_t i) const { return data[i]; }

  void cast_load(const T* ptr) {
    #pragma unroll
    for (uint32_t i = 0; i < N; ++i) {
      data[i] = ptr[i];
    }
  }
};

template <
    typename T,
    typename DST_DTYPE,
    bool IS_COLUMN_MAJOR = false>
struct PerTokenGroupQuant8bitKernel : public __SYCL_KER_CONFIG_CONVENTION__ {

  PerTokenGroupQuant8bitKernel(
      const T* input,
      void* output_q,
      float* output_s,
      int group_size,
      int num_groups,
      int groups_per_block,
      float eps,
      float min_8bit,
      float max_8bit,
      int num_groups_per_row = 0,
      int scale_stride = 0)
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
        scale_stride(scale_stride) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {}

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> item) const {
    const int threads_per_group = 16;
    const int64_t local_group_id = item.get_local_id(0) / threads_per_group;
    const int lane_id = item.get_local_id(0) % threads_per_group;

    const int64_t block_group_id = item.get_group(0) * groups_per_block;
    const int64_t global_group_id = block_group_id + local_group_id;
    const int64_t block_group_offset = global_group_id * group_size;

    float local_absmax = eps;

    const T* group_input = input + block_group_offset;
    DST_DTYPE* group_output = static_cast<DST_DTYPE*>(output_q) + block_group_offset;
    float* scale_output;

    if constexpr (IS_COLUMN_MAJOR) {
      const int row_idx = global_group_id / num_groups_per_row;
      const int col_idx = global_group_id % num_groups_per_row;
      scale_output = output_s + (col_idx * scale_stride + row_idx);
    } else {
      scale_output = output_s + global_group_id;
    }

    constexpr uint32_t vec_size = 16 / sizeof(T);
    using vec_type = vec_t<T, vec_size>;

    const int32_t num_vec_elems = group_size / vec_size;

    // First pass: find local_absmax
    for (int32_t i = lane_id; i < num_vec_elems; i += 16) {
      vec_type input_vec;
      input_vec.cast_load(group_input + i * vec_size);

      #pragma unroll
      for (uint32_t j = 0; j < vec_size; ++j) {
        float val = static_cast<float>(input_vec[j]);
        float abs_val = sycl::fabs(val);
        local_absmax = sycl::fmax(local_absmax, abs_val);
      }
    }

    // Reduce across the 16 threads in the group to find the maximum
    local_absmax = GroupReduceMax(local_absmax, item, lane_id);

    // Calculate scale factor
    float y_s = local_absmax / max_8bit;

    if (lane_id == 0) {
      *scale_output = y_s;
    }

    // Second pass: quantize
    for (int32_t i = lane_id; i < num_vec_elems; i += 16) {
      vec_type input_vec;
      input_vec.cast_load(group_input + i * vec_size);

      #pragma unroll
      for (uint32_t j = 0; j < vec_size; ++j) {
        float val = static_cast<float>(input_vec[j]);
        float q_val = sycl::fmin(sycl::fmax(val / y_s, min_8bit), max_8bit);

        // Special handling for FP8 types using CUTLASS
        group_output[i * vec_size + j] = cutlass::float_e4m3_t::from_float(q_val);
      }
    }
  }

private:
  const T* input;
  void* output_q;
  float* output_s;
  int group_size;
  int num_groups;
  int groups_per_block;
  float eps;
  float min_8bit;
  float max_8bit;
  int num_groups_per_row;
  int scale_stride;
};

void sgl_per_token_group_quant_8bit(
    torch::Tensor input,
    torch::Tensor output_q,
    torch::Tensor output_s,
    int64_t group_size,
    double eps,
    double min_8bit,
    double max_8bit) {

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

  sycl::range<1> global_range(num_blocks * num_threads);
  sycl::range<1> local_range(num_threads);

#define LAUNCH_KERNEL(T, DST_DTYPE)                                                                     \
  do {                                                                                                  \
    if (is_column_major) {                                                                              \
      auto kernel = PerTokenGroupQuant8bitKernel<T, DST_DTYPE, true>(                                  \
          static_cast<const T*>(input.data_ptr()),                                                      \
          output_q.data_ptr(),                                                                          \
          static_cast<float*>(output_s.data_ptr()),                                                     \
          group_size,                                                                                   \
          num_groups,                                                                                   \
          groups_per_block,                                                                             \
          static_cast<float>(eps),                                                                      \
          static_cast<float>(min_8bit),                                                                 \
          static_cast<float>(max_8bit),                                                                 \
          num_groups_per_row,                                                                           \
          scale_stride);                                                                                \
      sycl_kernel_submit(global_range, local_range, queue, kernel);                                    \
    } else {                                                                                            \
      auto kernel = PerTokenGroupQuant8bitKernel<T, DST_DTYPE, false>(                                 \
          static_cast<const T*>(input.data_ptr()),                                                      \
          output_q.data_ptr(),                                                                          \
          static_cast<float*>(output_s.data_ptr()),                                                     \
          group_size,                                                                                   \
          num_groups,                                                                                   \
          groups_per_block,                                                                             \
          static_cast<float>(eps),                                                                      \
          static_cast<float>(min_8bit),                                                                 \
          static_cast<float>(max_8bit));                                                                \
      sycl_kernel_submit(global_range, local_range, queue, kernel);                                    \
    }                                                                                                   \
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
