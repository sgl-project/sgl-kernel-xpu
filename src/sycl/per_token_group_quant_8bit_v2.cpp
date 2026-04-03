/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
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
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cmath>
#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"
#include "cutlass/float8.h"

namespace at::native::xpu {

constexpr float LOCAL_ABSMAX_ABS = 1e-10;
constexpr uint32_t INPUT_PRIMARY_VEC_NUM_BYTES = 32;

// Use SYCL native vector type for efficient loading
template <typename T, uint32_t N>
using vec_t = sycl::vec<T, N>;

template <typename T>
struct DtypeInfo;

template <>
struct DtypeInfo<int8_t> {
  static constexpr float MIN = -128;
  static constexpr float MAX = 127;
};

template <>
struct DtypeInfo<c10::Float8_e4m3fn> {
  static constexpr float MIN = -448;
  static constexpr float MAX = 448;
};

struct dim3 {
  int x, y, z;
};

template <
    int GROUP_SIZE,
    int THREADS_PER_SUBWARP,
    typename T,
    typename DST_DTYPE,
    bool IS_COLUMN_MAJOR,
    bool SCALE_UE8M0,
    bool FUSE_SILU_AND_MUL,
    typename scale_packed_t>
struct MainKernel {
  MainKernel(
      const T* input,
      DST_DTYPE* output_q,
      scale_packed_t* output_s,
      const int32_t* masked_m,
      float eps,
      float min_8bit,
      float max_8bit,
      const int subwarps_per_block,
      const int hidden_dim_num_groups,
      const int scale_expert_stride,
      const int scale_hidden_stride,
      const int num_tokens_per_expert)
      : input(input),
        output_q(output_q),
        output_s(output_s),
        masked_m(masked_m),
        eps(eps),
        min_8bit(min_8bit),
        max_8bit(max_8bit),
        subwarps_per_block(subwarps_per_block),
        hidden_dim_num_groups(hidden_dim_num_groups),
        scale_expert_stride(scale_expert_stride),
        scale_hidden_stride(scale_hidden_stride),
        num_tokens_per_expert(num_tokens_per_expert) {}

  using float2 = sycl::vec<float, 2>;
  using dst_dtype_info = DtypeInfo<DST_DTYPE>;
  using scale_element_t = std::conditional_t<SCALE_UE8M0, uint8_t, float>;
  static_assert(sizeof(scale_packed_t) % sizeof(scale_element_t) == 0);
  using fp8x2_storage_t = uint16_t;

  inline float silu(const float& val) const {
    float half = 0.5f * val;
    float t = sycl::tanh(half);
    return half * (1.0f + t);
  }

  inline float2 fmul2_rn(float2 a, float2 b) const {
    return a * b;
  }

  // Copied and modified from DeepEP
  inline float fast_pow2(int x) const {
    // We can ensure `-126 <= x and x <= 127`
    uint32_t bits_x = (x + 127) << 23;
    return *reinterpret_cast<float*>(&bits_x);
  }

  // Copied and modified from DeepEP
  inline int fast_log2_ceil(float x) const {
    auto bits_x = *reinterpret_cast<uint32_t*>(&x);
    auto exp_x = (bits_x >> 23) & 0xff;
    auto man_bits = bits_x & ((1 << 23) - 1);
    return exp_x - 127 + (man_bits != 0);
  }

  // Copied and modified from DeepEP
  inline void calculate_fp8_scales(float amax, float& scale, float& scale_inv) const {
    constexpr float MAX_8BIT_INV = 1.0f / dst_dtype_info::MAX;
    if constexpr (SCALE_UE8M0) {
      auto exp_scale_inv = fast_log2_ceil(amax * MAX_8BIT_INV);
      scale = fast_pow2(-exp_scale_inv);
      scale_inv = fast_pow2(exp_scale_inv);
    } else {
      scale_inv = amax * MAX_8BIT_INV;
      scale = dst_dtype_info::MAX / amax;
    }
  }

  // Copied and modified from DeepEP
  inline scale_element_t extract_required_scale_format(float value) const {
    if constexpr (SCALE_UE8M0) {
      return static_cast<scale_element_t>((*reinterpret_cast<uint32_t*>(&value)) >> 23);
    } else {
      return value;
    }
  }

  void compute(
      const int expert_idx,
      const int token_idx,
      const int hidden_dim_group_idx,
      const int lane_id,
      const int input_group_start_offset,
      sycl::sub_group sg) const {
    constexpr uint32_t INPUT_PRIMARY_VEC_SIZE = INPUT_PRIMARY_VEC_NUM_BYTES / sizeof(T);
    constexpr uint32_t INPUT_PRIMARY_INT4_SIZE = INPUT_PRIMARY_VEC_NUM_BYTES / (4 * sizeof(int));

    const int offset_num_groups = expert_idx * num_tokens_per_expert * hidden_dim_num_groups +
                                  token_idx * hidden_dim_num_groups + hidden_dim_group_idx;

    using int4 = sycl::vec<int, 4>;
    int4 input_primary_int4[INPUT_PRIMARY_INT4_SIZE];
    T* input_primary_vec = reinterpret_cast<T*>(input_primary_int4);
    static_assert(sizeof(input_primary_vec[0]) * INPUT_PRIMARY_VEC_SIZE == sizeof(input_primary_int4));

    int4 input_secondary_int4[INPUT_PRIMARY_INT4_SIZE];
    T* input_secondary_vec = reinterpret_cast<T*>(input_secondary_int4);
    static_assert(sizeof(input_secondary_vec[0]) * INPUT_PRIMARY_VEC_SIZE == sizeof(input_secondary_int4));

    auto primary_base_ptr =
        reinterpret_cast<const int4*>(input + input_group_start_offset + lane_id * INPUT_PRIMARY_VEC_SIZE);
#pragma unroll
    for (uint32_t j = 0; j < INPUT_PRIMARY_INT4_SIZE; ++j) {
      input_primary_int4[j] = primary_base_ptr[j];
    }
    if constexpr (FUSE_SILU_AND_MUL) {
      const int secondary_offset = hidden_dim_num_groups * GROUP_SIZE;
      auto secondary_base_ptr = reinterpret_cast<const int4*>(
          input + input_group_start_offset + lane_id * INPUT_PRIMARY_VEC_SIZE + secondary_offset);
#pragma unroll
      for (uint32_t j = 0; j < INPUT_PRIMARY_INT4_SIZE; ++j) {
        input_secondary_int4[j] = secondary_base_ptr[j];
      }
    }

    constexpr int num_elems_per_pack = static_cast<int>(sizeof(scale_packed_t) / sizeof(scale_element_t));
    scale_element_t* scale_output;
    if constexpr (IS_COLUMN_MAJOR) {
      constexpr int scale_token_stride = 1;

      const int hidden_idx_packed = hidden_dim_group_idx / num_elems_per_pack;
      const int pack_idx = hidden_dim_group_idx % num_elems_per_pack;
      scale_output = reinterpret_cast<scale_element_t*>(output_s) +
                     (expert_idx * scale_expert_stride * num_elems_per_pack +
                      hidden_idx_packed * scale_hidden_stride * num_elems_per_pack +
                      token_idx * scale_token_stride * num_elems_per_pack + pack_idx);
    } else {
      static_assert(!SCALE_UE8M0);
      scale_output = output_s + offset_num_groups;
    }

    // can speed up if too slow
    if constexpr (IS_COLUMN_MAJOR and SCALE_UE8M0) {
      const int remainder_num_groups = hidden_dim_num_groups % num_elems_per_pack;
      if ((remainder_num_groups != 0) and (hidden_dim_group_idx == hidden_dim_num_groups - 1) and
          (lane_id < num_elems_per_pack - remainder_num_groups)) {
        const int shift = 1 + lane_id;
        *(scale_output + shift) = 0;
      }
    }

    float local_absmax = LOCAL_ABSMAX_ABS;

#pragma unroll
    for (uint32_t j = 0; j < INPUT_PRIMARY_VEC_SIZE; ++j) {
      float val;
      if constexpr (FUSE_SILU_AND_MUL) {
        // TODO maybe vectorize
        T val_lowprec = static_cast<T>(silu(static_cast<float>(input_primary_vec[j]))) * input_secondary_vec[j];
        val = static_cast<float>(val_lowprec);
        input_primary_vec[j] = val_lowprec;
      } else {
        val = static_cast<float>(input_primary_vec[j]);
      }

      float abs_val = fabsf(val);
      local_absmax = fmaxf(local_absmax, abs_val);
    }

    uint32_t lane = sg.get_local_id()[0];  // 0..15
    // Logical subgroup of size 16/8/4/...
    uint32_t logical_lane = lane & (THREADS_PER_SUBWARP - 1);  // lane % 8
    uint32_t group_base = lane & ~(THREADS_PER_SUBWARP - 1);   // (lane / 8) * 8

// argmax reduce
#pragma unroll
    for (int mask = THREADS_PER_SUBWARP / 2; mask > 0; mask >>= 1) {
      uint32_t target_logical = logical_lane ^ mask;
      uint32_t target_lane = group_base + target_logical;

      // Convert absolute lane → xor distance
      uint32_t xor_mask = lane ^ target_lane;

      T other_max = sycl::permute_group_by_xor(sg, local_absmax, xor_mask);
      local_absmax = fmaxf(local_absmax, other_max);
    }

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

    const float inv_y_s = 1.0f / y_s;

    using output_storage_t = uint8_t;
    using output_vec_type = vec_t<output_storage_t, INPUT_PRIMARY_VEC_SIZE>;
    output_vec_type output_vec;

    for (uint32_t j = 0; j < INPUT_PRIMARY_VEC_SIZE; j++) {
      float val = input_primary_vec[j];
      float q_val = sycl::fmin(sycl::fmax(val * inv_y_s, min_8bit), max_8bit);

      // Special handling for FP8 types using CUTLASS
      if constexpr (std::is_same_v<DST_DTYPE, c10::Float8_e4m3fn>) {
        // TODO: Remove CUTLASS emulation of float e4m3_t and use native SYCL FP8 when available
        DST_DTYPE fp8_val = static_cast<DST_DTYPE>(q_val);
        output_vec[j] = sycl::bit_cast<output_storage_t>(fp8_val);
      } else {
        output_vec[j] = static_cast<DST_DTYPE>(q_val);
      }
    }

    // Vectorized store
    output_vec.store(
        0,
        sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(
            reinterpret_cast<output_storage_t*>(
                output_q + offset_num_groups * GROUP_SIZE + lane_id * INPUT_PRIMARY_VEC_SIZE)));
  }

  const T* input;
  DST_DTYPE* output_q;
  scale_packed_t* output_s;
  const int32_t* masked_m;
  float eps;
  float min_8bit;
  float max_8bit;
  const int subwarps_per_block;
  const int hidden_dim_num_groups;
  const int scale_expert_stride;
  const int scale_hidden_stride;
  const int num_tokens_per_expert;
};

template <
    typename SCHEDULER,
    int GROUP_SIZE,
    int THREADS_PER_SUBWARP,
    typename T,
    typename DST_DTYPE,
    bool IS_COLUMN_MAJOR,
    bool SCALE_UE8M0,
    bool FUSE_SILU_AND_MUL,
    typename scale_packed_t>
struct NaiveKernel : MainKernel<
                         GROUP_SIZE,
                         THREADS_PER_SUBWARP,
                         T,
                         DST_DTYPE,
                         IS_COLUMN_MAJOR,
                         SCALE_UE8M0,
                         FUSE_SILU_AND_MUL,
                         scale_packed_t> {
  NaiveKernel(
      const T* input,
      DST_DTYPE* output_q,
      scale_packed_t* output_s,
      const int32_t* masked_m,
      float eps,
      float min_8bit,
      float max_8bit,
      const int subwarps_per_block,
      const int hidden_dim_num_groups,
      const int scale_expert_stride,
      const int scale_hidden_stride,
      const int num_tokens_per_expert)
      : MainKernel<
            GROUP_SIZE,
            THREADS_PER_SUBWARP,
            T,
            DST_DTYPE,
            IS_COLUMN_MAJOR,
            SCALE_UE8M0,
            FUSE_SILU_AND_MUL,
            scale_packed_t>(
            input,
            output_q,
            output_s,
            masked_m,
            eps,
            min_8bit,
            max_8bit,
            subwarps_per_block,
            hidden_dim_num_groups,
            scale_expert_stride,
            scale_hidden_stride,
            num_tokens_per_expert),  // base initialization
        subwarps_per_block(subwarps_per_block),
        hidden_dim_num_groups(hidden_dim_num_groups),
        scale_expert_stride(scale_expert_stride),
        scale_hidden_stride(scale_hidden_stride),
        num_tokens_per_expert(num_tokens_per_expert) {}

  inline int compute_input_group_start_offset(
      int expert_idx,
      int token_idx,
      int hidden_dim_group_idx,
      int hidden_size,
      int num_tokens_per_expert,
      int group_size) const {
    return expert_idx * num_tokens_per_expert * hidden_size * (FUSE_SILU_AND_MUL ? 2 : 1) +
           token_idx * hidden_size * (FUSE_SILU_AND_MUL ? 2 : 1) + hidden_dim_group_idx * group_size;
  }

  [[sycl::reqd_sub_group_size(32)]]
  void operator()(sycl::nd_item<3> item) const {
    constexpr int expert_idx = 0;

    int threadIdx_x = item.get_local_linear_id();
    int blockIdx_x = item.get_group().get_group_linear_id();

    const int64_t subwarp_id = threadIdx_x / THREADS_PER_SUBWARP;
    const int lane_id = threadIdx_x % THREADS_PER_SUBWARP;

    const int64_t block_group_id = blockIdx_x * subwarps_per_block;
    const int64_t group_id = block_group_id + subwarp_id;

    int64_t input_group_start_offset;
    if constexpr (!FUSE_SILU_AND_MUL) {
      input_group_start_offset = group_id * GROUP_SIZE;
    }

    const int token_idx = group_id / hidden_dim_num_groups;
    // At the hidden_size dimension, we are handling idx-th group
    const int hidden_dim_group_idx = group_id % hidden_dim_num_groups;

    if constexpr (FUSE_SILU_AND_MUL) {
      const int hidden_size = hidden_dim_num_groups * GROUP_SIZE;
      input_group_start_offset = compute_input_group_start_offset(
          expert_idx, token_idx, hidden_dim_group_idx, hidden_size, num_tokens_per_expert, GROUP_SIZE);
    }

    this->compute(expert_idx, token_idx, hidden_dim_group_idx, lane_id, input_group_start_offset, item.get_sub_group());
  }

  const int subwarps_per_block;
  const int hidden_dim_num_groups;
  const int scale_expert_stride;
  const int scale_hidden_stride;
  const int num_tokens_per_expert;
};

template <
    typename SCHEDULER,
    int GROUP_SIZE,
    int THREADS_PER_SUBWARP,
    typename T,
    typename DST_DTYPE,
    bool IS_COLUMN_MAJOR,
    bool SCALE_UE8M0,
    bool FUSE_SILU_AND_MUL,
    typename scale_packed_t>
struct MaskedKernel : MainKernel<
                          GROUP_SIZE,
                          THREADS_PER_SUBWARP,
                          T,
                          DST_DTYPE,
                          IS_COLUMN_MAJOR,
                          SCALE_UE8M0,
                          FUSE_SILU_AND_MUL,
                          scale_packed_t> {
  MaskedKernel(
      const T* input,
      DST_DTYPE* output_q,
      scale_packed_t* output_s,
      const int32_t* masked_m,
      float eps,
      float min_8bit,
      float max_8bit,
      const int subwarps_per_block,
      const int hidden_dim_num_groups,
      const int scale_expert_stride,
      const int scale_hidden_stride,
      const int num_tokens_per_expert)
      : MainKernel<
            GROUP_SIZE,
            THREADS_PER_SUBWARP,
            T,
            DST_DTYPE,
            IS_COLUMN_MAJOR,
            SCALE_UE8M0,
            FUSE_SILU_AND_MUL,
            scale_packed_t>(
            input,
            output_q,
            output_s,
            masked_m,
            eps,
            min_8bit,
            max_8bit,
            subwarps_per_block,
            hidden_dim_num_groups,
            scale_expert_stride,
            scale_hidden_stride,
            num_tokens_per_expert),  // base initialization
        subwarps_per_block(subwarps_per_block),
        hidden_dim_num_groups(hidden_dim_num_groups),
        scale_expert_stride(scale_expert_stride),
        scale_hidden_stride(scale_hidden_stride),
        num_tokens_per_expert(num_tokens_per_expert),
        masked_m(masked_m) {}

  inline int compute_input_group_start_offset(
      int expert_idx,
      int token_idx,
      int hidden_dim_group_idx,
      int hidden_size,
      int num_tokens_per_expert,
      int group_size) const {
    return expert_idx * num_tokens_per_expert * hidden_size * (FUSE_SILU_AND_MUL ? 2 : 1) +
           token_idx * hidden_size * (FUSE_SILU_AND_MUL ? 2 : 1) + hidden_dim_group_idx * group_size;
  }

  [[sycl::reqd_sub_group_size(32)]]
  void operator()(sycl::nd_item<3> item) const {
    int threadIdx_x = item.get_local_linear_id();
    const int64_t subwarp_id = threadIdx_x / THREADS_PER_SUBWARP;
    const int lane_id = threadIdx_x % THREADS_PER_SUBWARP;

    auto group = item.get_group();

    const int expert_idx = group.get_group_id(0);
    const int token_idx_start = group.get_group_id(1);
    const int chunk_id = group.get_group_id(2);

    const int64_t hidden_dim_group_idx = chunk_id * SCHEDULER::SUBWARPS_PER_BLOCK + subwarp_id;

    // only valid tokens are handled
    const int curr_expert_token_num = masked_m[expert_idx];

    // skip maksed tokens
    for (int token_idx = token_idx_start; token_idx < curr_expert_token_num;
         token_idx += SCHEDULER::TOKEN_DIM_BLOCK_NUM_PER_EXPERT) {
      const int hidden_size = hidden_dim_num_groups * GROUP_SIZE;

      // chunk offset
      const int64_t input_group_start_offset = compute_input_group_start_offset(
          expert_idx, token_idx, hidden_dim_group_idx, hidden_size, num_tokens_per_expert, GROUP_SIZE);

      // invoke only unmasked token
      this->compute(
          expert_idx, token_idx, hidden_dim_group_idx, lane_id, input_group_start_offset, item.get_sub_group());
    }
  }

  const int subwarps_per_block;
  const int hidden_dim_num_groups;
  const int scale_expert_stride;
  const int scale_hidden_stride;
  const int num_tokens_per_expert;
  const int32_t* masked_m;
};

struct NaiveScheduler {
  static void compute_exec_config(
      int threads_per_subwarp,
      int num_local_experts,
      int hidden_dim_num_groups,
      int num_groups,
      int& subwarps_per_block,
      dim3& grid,
      dim3& block) {
    subwarps_per_block = ([=]() -> int {
      if (num_groups % 16 == 0) {
        return 16;
      } else if (num_groups % 8 == 0) {
        return 8;
      } else if (num_groups % 4 == 0) {
        return 4;
      } else if (num_groups % 2 == 0) {
        return 2;
      }
      return 1;
    })();
    grid = dim3{num_groups / subwarps_per_block, 1, 1};
    block = dim3{subwarps_per_block * threads_per_subwarp, 1, 1};
  }
};

struct MaskedLayoutScheduler {
  // TODO can be dynamically determined (which may be good when num rank is small)
  static constexpr int TOKEN_DIM_BLOCK_NUM_PER_EXPERT = 1024;
  static constexpr int SUBWARPS_PER_BLOCK = 16;

  static void compute_exec_config(
      int threads_per_subwarp,
      int num_local_experts,
      int hidden_dim_num_groups,
      int num_groups,
      int& subwarps_per_block,
      dim3& grid,
      dim3& block) {
    subwarps_per_block = SUBWARPS_PER_BLOCK;
    TORCH_CHECK(hidden_dim_num_groups % subwarps_per_block == 0);
    grid = dim3{hidden_dim_num_groups / subwarps_per_block, TOKEN_DIM_BLOCK_NUM_PER_EXPERT, num_local_experts};
    block = dim3{subwarps_per_block * threads_per_subwarp};
  }
};

template <
    typename SCHEDULER,
    int GROUP_SIZE,
    int THREADS_PER_SUBWARP,
    typename T,
    typename DST_DTYPE,
    bool IS_COLUMN_MAJOR = false,
    bool SCALE_UE8M0 = false,
    bool FUSE_SILU_AND_MUL = false,
    typename scale_packed_t = std::conditional_t<SCALE_UE8M0, uint32_t, float>>
void per_token_group_quant_8bit_kernel_impl(
    const T* input,
    DST_DTYPE* output_q,
    scale_packed_t* output_s,
    const int32_t* masked_m,
    float eps,
    float min_8bit,
    float max_8bit,
    const int subwarps_per_block,
    const dim3 grid,
    const dim3 blocks,
    const int hidden_dim_num_groups,
    const int scale_expert_stride,
    const int scale_hidden_stride,
    const int num_tokens_per_expert) {
  // config kernel params
  if constexpr (std::is_same_v<SCHEDULER, NaiveScheduler>) {
    using Kernel = NaiveKernel<
        SCHEDULER,
        GROUP_SIZE,
        THREADS_PER_SUBWARP,
        T,
        DST_DTYPE,
        IS_COLUMN_MAJOR,
        SCALE_UE8M0,
        FUSE_SILU_AND_MUL,
        scale_packed_t>;

    auto stream = at::xpu::getCurrentXPUStream();
    auto queue = stream.queue();

    unsigned long num_grids = grid.x;
    unsigned long num_blocks = blocks.x;
    sycl::range<3> global_range{num_grids * num_blocks, 1, 1};
    sycl::range<3> local_range{num_blocks, 1, 1};

    Kernel task(
        input,
        output_q,
        output_s,
        masked_m,
        eps,
        min_8bit,
        max_8bit,
        subwarps_per_block,
        hidden_dim_num_groups,
        scale_expert_stride,
        scale_hidden_stride,
        num_tokens_per_expert);

    sycl_kernel_submit(global_range, local_range, queue, task);

  } else if constexpr (std::is_same_v<SCHEDULER, MaskedLayoutScheduler>) {
    using Kernel = MaskedKernel<
        SCHEDULER,
        GROUP_SIZE,
        THREADS_PER_SUBWARP,
        T,
        DST_DTYPE,
        IS_COLUMN_MAJOR,
        SCALE_UE8M0,
        FUSE_SILU_AND_MUL,
        scale_packed_t>;

    auto stream = at::xpu::getCurrentXPUStream();
    auto queue = stream.queue();

    unsigned long num_experts = grid.z;
    unsigned long token_dims_per_expert = grid.y;
    unsigned long sub_warp_chunks = grid.x;
    unsigned long num_block_threads = blocks.x;

    sycl::range<3> global_range{num_experts, token_dims_per_expert, sub_warp_chunks};
    sycl::range<3> local_range{1, 1, num_block_threads};

    Kernel task(
        input,
        output_q,
        output_s,
        masked_m,
        eps,
        min_8bit,
        max_8bit,
        subwarps_per_block,
        hidden_dim_num_groups,
        scale_expert_stride,
        scale_hidden_stride,
        num_tokens_per_expert);

    sycl_kernel_submit(global_range, local_range, queue, task);
  }
}

void sgl_per_token_group_quant_8bit_v2(
    // vanilla: (num_tokens, hidden_size)
    // fuse_silu_and_mul: (num_tokens, hidden_size * 2)
    // fuse_silu_and_mul + masked_layout: (num_experts, num_tokens-with-padding, hidden_size * 2)
    torch::Tensor input,
    torch::Tensor output_q,
    torch::Tensor output_s,
    int64_t group_size,
    double eps,
    double min_8bit,
    double max_8bit,
    bool scale_ue8m0,
    bool fuse_silu_and_mul,
    const std::optional<torch::Tensor>& masked_m) {
  CHECK_INPUT(input);
  CHECK_INPUT(output_q);
  TORCH_CHECK(input.numel() > 0);

  TORCH_CHECK(std::abs(LOCAL_ABSMAX_ABS - eps) < 1e-13);

  CHECK_EQ(input.numel() % group_size, 0);
  const int num_groups = static_cast<int>(input.numel()) / group_size / (fuse_silu_and_mul ? 2 : 1);

  const bool masked_layout = masked_m.has_value();
  TORCH_CHECK(output_s.dim() == (masked_layout ? 3 : 2));

  const int num_local_experts = masked_layout ? input.size(0) : 1;

  auto dst_type = output_q.scalar_type();

  const bool is_column_major = output_s.stride(-2) < output_s.stride(-1);
  const int hidden_dim_num_groups = static_cast<int>(output_q.size(-1)) / group_size;
  const int num_tokens_per_expert = static_cast<int>(output_q.size(-2));
  const int scale_expert_stride = masked_layout ? static_cast<int>(output_s.stride(0)) : 0;
  const int scale_hidden_stride = static_cast<int>(output_s.stride(-1));

#define LAUNCH_KERNEL_INNER(SCHEDULER, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, output_s_dtype, ...)           \
  do {                                                                                                               \
    int subwarps_per_block;                                                                                          \
    dim3 grid, block;                                                                                                \
    SCHEDULER::compute_exec_config(                                                                                  \
        THREADS_PER_SUBWARP, num_local_experts, hidden_dim_num_groups, num_groups, subwarps_per_block, grid, block); \
                                                                                                                     \
    per_token_group_quant_8bit_kernel_impl<SCHEDULER, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, __VA_ARGS__>(   \
        static_cast<T*>(input.data_ptr()),                                                                           \
        static_cast<DST_DTYPE*>(output_q.data_ptr()),                                                                \
        static_cast<output_s_dtype*>(output_s.data_ptr()),                                                           \
        static_cast<int32_t*>(masked_m.has_value() ? masked_m->data_ptr() : 0),                                      \
        static_cast<float>(eps),                                                                                     \
        static_cast<float>(min_8bit),                                                                                \
        static_cast<float>(max_8bit),                                                                                \
        subwarps_per_block,                                                                                          \
        grid,                                                                                                        \
        block,                                                                                                       \
        hidden_dim_num_groups,                                                                                       \
        scale_expert_stride,                                                                                         \
        scale_hidden_stride,                                                                                         \
        num_tokens_per_expert);                                                                                      \
  } while (0)

#define LAUNCH_KERNEL(GROUP_SIZE, T, DST_DTYPE)                                                                     \
  do {                                                                                                              \
    constexpr int THREADS_PER_SUBWARP = GROUP_SIZE / 16;                                                            \
    TORCH_CHECK(THREADS_PER_SUBWARP* INPUT_PRIMARY_VEC_NUM_BYTES == group_size * sizeof(T));                        \
                                                                                                                    \
    using dst_dtype_info = DtypeInfo<DST_DTYPE>;                                                                    \
    CHECK_EQ(dst_dtype_info::MIN, min_8bit);                                                                        \
    CHECK_EQ(dst_dtype_info::MAX, max_8bit);                                                                        \
                                                                                                                    \
    if (is_column_major) {                                                                                          \
      if (scale_ue8m0) {                                                                                            \
        if (fuse_silu_and_mul) {                                                                                    \
          if (masked_layout) {                                                                                      \
            LAUNCH_KERNEL_INNER(                                                                                    \
                MaskedLayoutScheduler, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, uint32_t, true, true, true);  \
          } else {                                                                                                  \
            LAUNCH_KERNEL_INNER(                                                                                    \
                NaiveScheduler, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, uint32_t, true, true, true);         \
          }                                                                                                         \
        } else {                                                                                                    \
          LAUNCH_KERNEL_INNER(NaiveScheduler, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, uint32_t, true, true); \
        }                                                                                                           \
      } else {                                                                                                      \
        LAUNCH_KERNEL_INNER(NaiveScheduler, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, float, true);            \
      }                                                                                                             \
    } else {                                                                                                        \
      LAUNCH_KERNEL_INNER(NaiveScheduler, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, float, false);             \
    }                                                                                                               \
  } while (0)

#define LAUNCH_KERNEL_OUTER(...)                    \
  switch (group_size) {                             \
    case 16:                                        \
      LAUNCH_KERNEL(16, __VA_ARGS__);               \
      break;                                        \
    case 32:                                        \
      LAUNCH_KERNEL(32, __VA_ARGS__);               \
      break;                                        \
    case 64:                                        \
      LAUNCH_KERNEL(64, __VA_ARGS__);               \
      break;                                        \
    case 128:                                       \
      LAUNCH_KERNEL(128, __VA_ARGS__);              \
      break;                                        \
    default:                                        \
      TORCH_CHECK(false, "Unsupported group_size"); \
  }                                                 \
  while (0)

  SYCL_DISPATCH_ONLY_FLOATING16_TYPES(
      at::ScalarType::BFloat16, at::ScalarType::Half, input.scalar_type(), "sgl_per_token_group_quant_8bit_v2", [&]() {
        if (dst_type == at::ScalarType::Char) {
          LAUNCH_KERNEL_OUTER(scalar_t, int8_t);
          return true;
        } else if (dst_type == at::ScalarType::Float8_e4m3fn) {
          LAUNCH_KERNEL_OUTER(scalar_t, c10::Float8_e4m3fn);
          return true;
        }
        return false;
      });

#undef LAUNCH_KERNEL
#undef LAUNCH_KERNEL_INNER
}

}  // namespace at::native::xpu
