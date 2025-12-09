/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#include <ATen/ATen.h>
#include <cutlass/float8.h>

#include <algorithm>
#include <sycl/sycl.hpp>

#include "MemoryAccess.h"
#include "SYCLHelpers.h"
#include "Utils.h"

// TODO: Remove this when sycl float8 is supported
using cutlass::float_e4m3_t;

constexpr float FP8_E4M3_MAX = 448.0f;
constexpr float eps = 1e-8f;

template <typename T, typename DST_DTYPE, int VEC_SIZE>
class PerTensorQuantFP8Kernel {
 private:
  const T* input_;
  DST_DTYPE* output_;
  const float* scale_;
  int64_t num_elements_;

 public:
  PerTensorQuantFP8Kernel(const T* input, DST_DTYPE* output, const float* scale, int64_t num_elements)
      : input_(input), output_(output), scale_(scale), num_elements_(num_elements) {}

  void operator()(sycl::nd_item<1> item) const {
    const int gid = item.get_global_id(0);
    const int grid_size = item.get_global_range(0);

    const int64_t num_vec_elems = num_elements_ / VEC_SIZE;

    float scale_val = 1.0f / (*scale_ + eps);  // eps to avoid div by zero
    using vec_type_in = vec_t<T, VEC_SIZE>;

    // Realize fp8 as uint8_t storage as sycl does not have native fp8 type currently
    // TODO: Remove this when sycl float8 is supported
    using output_storage_t = uint8_t;
    using vec_type_out = vec_t<output_storage_t, VEC_SIZE>;

    for (int64_t i = gid; i < num_vec_elems; i += grid_size) {
      vec_type_in input_vec;
      int64_t base_idx = i * VEC_SIZE;
      input_vec.load(0, sycl::multi_ptr<const T, sycl::access::address_space::global_space>(input_ + base_idx));
      vec_type_out output_vec;
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        float val = static_cast<float>(input_vec[j]) * scale_val;
        val = sycl::fmax(-FP8_E4M3_MAX, sycl::fmin(val, FP8_E4M3_MAX));
        DST_DTYPE fp8_val = static_cast<DST_DTYPE>(val);
        output_vec[j] = *reinterpret_cast<output_storage_t*>(&fp8_val);
      }
      // TODO: output_storage_t needs to be changed back to DST_DTYPE during store when fp8 type is supported in sycl
      output_vec.store(
          0,
          sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(
              reinterpret_cast<output_storage_t*>(output_ + base_idx)));
    }

    const int64_t remaining_start = num_vec_elems * VEC_SIZE;
    for (int64_t idx = remaining_start + gid; idx < num_elements_; idx += grid_size) {
      float val = static_cast<float>(input_[idx]) * scale_val;
      val = sycl::fmax(-FP8_E4M3_MAX, sycl::fmin(val, FP8_E4M3_MAX));
      output_[idx] = static_cast<DST_DTYPE>(val);
    }
  }
};

template <typename T, int VEC_SIZE>
class PerTensorAbsMaxKernel {
 private:
  const T* input_;
  float* output_;
  int64_t num_elements_;

 public:
  PerTensorAbsMaxKernel(const T* input, float* output, int64_t num_elements)
      : input_(input), output_(output), num_elements_(num_elements) {}

  void operator()(sycl::nd_item<1> item) const {
    const int gid = item.get_global_id(0);
    const int grid_size = item.get_global_range(0);
    const int tid = item.get_local_id(0);
    const int local_size = item.get_local_range(0);

    float max_value = 0.0f;

    const int64_t num_vec_elems = num_elements_ / VEC_SIZE;
    using vec_type = vec_t<T, VEC_SIZE>;

    for (int64_t i = gid; i < num_vec_elems; i += grid_size) {
      int64_t base_idx = i * VEC_SIZE;

      vec_type input_vec;
      input_vec.load(0, sycl::multi_ptr<const T, sycl::access::address_space::global_space>(input_ + base_idx));

#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        T input_val = *reinterpret_cast<const T*>(&input_vec[j]);
        float val = sycl::fabs(static_cast<float>(input_val));
        max_value = sycl::fmax(max_value, val);
      }
    }

    const int64_t remaining_start = num_vec_elems * VEC_SIZE;
    for (int64_t idx = remaining_start + gid; idx < num_elements_; idx += grid_size) {
      float val = sycl::fabs(static_cast<float>(input_[idx]));
      max_value = sycl::fmax(max_value, val);
    }

    // Block-level max
    auto work_group = item.get_group();
    max_value = sycl::reduce_over_group(work_group, max_value, sycl::maximum<>());

    // update global maximum scale
    if (tid == 0) {
      float local_scale = max_value / FP8_E4M3_MAX;
      sycl::atomic_ref<
          float,
          sycl::memory_order::relaxed,
          sycl::memory_scope::device,
          sycl::access::address_space::global_space>
          atomic_scale(*output_);

      float global_scale = atomic_scale.load();
      while (global_scale < local_scale && !atomic_scale.compare_exchange_strong(global_scale, local_scale))
        ;
    }
  }
};

void sgl_per_tensor_quant_fp8(at::Tensor input, at::Tensor output_q, at::Tensor output_s, bool is_static) {
  CHECK_INPUT(input);
  CHECK_INPUT(output_q);
  CHECK_INPUT(output_s);
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::BFloat16 || input.scalar_type() == at::ScalarType::Half,
      "input must be BFloat16/Half tensor");
  TORCH_CHECK(output_q.scalar_type() == at::ScalarType::Float8_e4m3fn, "output must be Float8_e4m3fn tensor");
  TORCH_CHECK(output_s.scalar_type() == at::ScalarType::Float, "scale must be Float tensor");

  const int block_size = 256;
  const int64_t num_elements = input.numel();
  const int num_blocks = std::min((num_elements + block_size - 1) / block_size, (int64_t)1024);

  auto& Q = dpcppGetCurrentQueue();
  sycl::range<1> global_range(num_blocks * block_size);
  sycl::range<1> local_range(block_size);

#define LAUNCH_KERNEL(T, DST_DTYPE, VEC_SIZE)                                           \
  do {                                                                                  \
    if (!is_static) {                                                                   \
      auto kernel = PerTensorAbsMaxKernel<T, VEC_SIZE>(                                 \
          static_cast<T*>(input.data_ptr()), output_s.data_ptr<float>(), num_elements); \
      sycl_kernel_submit(global_range, local_range, Q, kernel);                         \
      Q.wait_and_throw();                                                               \
    }                                                                                   \
    auto kernel = PerTensorQuantFP8Kernel<T, DST_DTYPE, VEC_SIZE>(                      \
        static_cast<T*>(input.data_ptr()),                                              \
        static_cast<DST_DTYPE*>(output_q.data_ptr()),                                   \
        output_s.data_ptr<float>(),                                                     \
        num_elements);                                                                  \
    sycl_kernel_submit(global_range, local_range, Q, kernel);                           \
    Q.wait_and_throw();                                                                 \
  } while (0)

#define DISPATCH_VEC_SIZE(T, DST_DTYPE, vec_size)                                       \
  switch (vec_size) {                                                                   \
    case 1:                                                                             \
      LAUNCH_KERNEL(T, DST_DTYPE, 1);                                                   \
      break;                                                                            \
    case 2:                                                                             \
      LAUNCH_KERNEL(T, DST_DTYPE, 2);                                                   \
      break;                                                                            \
    case 4:                                                                             \
      LAUNCH_KERNEL(T, DST_DTYPE, 4);                                                   \
      break;                                                                            \
    case 8:                                                                             \
      LAUNCH_KERNEL(T, DST_DTYPE, 8);                                                   \
      break;                                                                            \
    case 16:                                                                            \
      LAUNCH_KERNEL(T, DST_DTYPE, 16);                                                  \
      break;                                                                            \
    default:                                                                            \
      throw std::runtime_error("Unsupported vector size: " + std::to_string(vec_size)); \
  }

  // Dispatch based on input and output types
  if (input.scalar_type() == at::ScalarType::Half) {
    if (output_q.scalar_type() == at::ScalarType::Float8_e4m3fn) {
      int vec_size = preferred_vector_width(dpcppGetDeviceIdOfCurrentQueue(), sizeof(sycl::half));
      DISPATCH_VEC_SIZE(sycl::half, cutlass::float_e4m3_t, vec_size);
    }
  } else if (input.scalar_type() == at::ScalarType::BFloat16) {
    if (output_q.scalar_type() == at::ScalarType::Float8_e4m3fn) {
      int vec_size = preferred_vector_width(dpcppGetDeviceIdOfCurrentQueue(), sizeof(sycl::ext::oneapi::bfloat16));
      DISPATCH_VEC_SIZE(sycl::ext::oneapi::bfloat16, cutlass::float_e4m3_t, vec_size);
    }
  } else {
    throw std::runtime_error("Unsupported data type for per-tensor FP8 quantization");
  }
#undef DISPATCH_VEC_SIZE
#undef LAUNCH_KERNEL
}