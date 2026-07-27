/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
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
/*! \file
    \brief Shared device-side pointer-array grouped-GEMM metadata build for LoRA.

    Kernel-agnostic helpers that construct the per-segment problem sizes, strides,
    byte offsets, and absolute device pointer arrays that a CUTLASS pointer-array
    grouped GEMM consumes. Everything is built on device (one SYCL thread per
    segment, no host round-trip), so this header is reused across the LoRA
    grouped-GEMM launchers (A-fwd, B-fwd, ...).
*/

#pragma once

#include <ATen/ATen.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>

#include "sycl/SYCLHelpers.h"

namespace at::native::xpu {

//----------------- Shared device-side grouped-GEMM metadata ------------------//
//
// Per-segment problem sizes, element byte-offsets, and strides that the CUTLASS
// pointer-array grouped GEMM consumes. Built on device by a single SYCL kernel
// (one thread per segment)
//
// For each segment s in [0, num_segments):
//   M_s = seg_indptr[s+1] - seg_indptr[s]
//
//   a_off[s] = seg_indptr[s]        * K * elem_bytes    (into input_x)
//   b_off[s] = weight_indices[s] * N * K * elem_bytes   (into weights)
//   d_off[s] = seg_indptr[s]        * N * elem_bytes    (into output)

struct GroupedGemmMeta {
  torch::Tensor problem_sizes;  // int32 [num_segments, 3]  (M_s, N, K), on device
  torch::Tensor stride_A;       // int64 [num_segments]     leading dim of A = K
  torch::Tensor stride_B;       // int64 [num_segments]     leading dim of B = K
  torch::Tensor stride_D;       // int64 [num_segments]     leading dim of D = N
  torch::Tensor a_off;          // int64 [num_segments]     byte offset into A per segment (device)
  torch::Tensor b_off;          // int64 [num_segments]     byte offset into B per segment (device)
  torch::Tensor d_off;          // int64 [num_segments]     byte offset into D per segment (device)
};

// One thread per segment: derive M_s / lora_id from the index tensors and write
// problem sizes, constant strides, and byte offsets straight into device memory.
struct BuildGroupedGemmMetaKernel {
  const int32_t* seg_indptr;      // [num_segments + 1]
  const int32_t* weight_indices;  // [num_segments]
  int32_t* problem_sizes;         // [num_segments * 3]
  int64_t* stride_A;              // [num_segments]
  int64_t* stride_B;              // [num_segments]
  int64_t* stride_D;              // [num_segments]
  int64_t* a_off;                 // [num_segments]
  int64_t* b_off;                 // [num_segments]
  int64_t* d_off;                 // [num_segments]
  int N;
  int K;
  int64_t elem_bytes;
  int num_segments;

  void operator()(sycl::nd_item<1> item) const {
    const int s = static_cast<int>(item.get_global_linear_id());
    if (s >= num_segments) {
      return;
    }
    const int32_t row_start = seg_indptr[s];
    const int32_t M_s = seg_indptr[s + 1] - row_start;
    const int32_t lora_id = weight_indices[s];

    problem_sizes[3 * s + 0] = M_s;
    problem_sizes[3 * s + 1] = N;
    problem_sizes[3 * s + 2] = K;

    // Strides in elements (leading dim of A/B = K, D = N).
    stride_A[s] = static_cast<int64_t>(K);
    stride_B[s] = static_cast<int64_t>(K);
    stride_D[s] = static_cast<int64_t>(N);

    a_off[s] = static_cast<int64_t>(row_start) * K * elem_bytes;
    b_off[s] = static_cast<int64_t>(lora_id) * static_cast<int64_t>(N) * K * elem_bytes;
    d_off[s] = static_cast<int64_t>(row_start) * N * elem_bytes;
  }
};

// One thread per segment: turn a base address + per-segment byte offset into an
// absolute device pointer for the pointer-array grouped GEMM.
struct MakeDevicePtrsKernel {
  int64_t base_addr;
  const int64_t* off_bytes;  // [num_segments]
  int64_t* ptrs;             // [num_segments]
  int num_segments;

  void operator()(sycl::nd_item<1> item) const {
    const int s = static_cast<int>(item.get_global_linear_id());
    if (s >= num_segments) {
      return;
    }
    ptrs[s] = base_addr + off_bytes[s];
  }
};

// Round num_segments up to a whole number of work-groups of `wg` threads.
template <typename Kernel>
inline void submit_per_segment(sycl::queue& queue, int num_segments, Kernel kernel) {
  constexpr int wg = 256;
  const int64_t global = (static_cast<int64_t>(num_segments) + wg - 1) / wg * wg;
  sycl_kernel_submit(sycl::range<1>(global), sycl::range<1>(wg), queue, kernel);
}

inline GroupedGemmMeta build_grouped_gemm_meta(
    const torch::Tensor& seg_indptr_i32,      // int32 [num_segments + 1]
    const torch::Tensor& weight_indices_i32,  // int32 [num_segments]
    const int N,
    const int K,
    const int num_segments,
    const int64_t elem_bytes,
    const at::Device device,
    sycl::queue& queue) {
  auto opt_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
  auto opt_i64 = torch::TensorOptions().dtype(torch::kInt64).device(device);

  GroupedGemmMeta meta;
  meta.problem_sizes = torch::empty({num_segments, 3}, opt_i32);
  meta.stride_A = torch::empty({num_segments}, opt_i64);
  meta.stride_B = torch::empty({num_segments}, opt_i64);
  meta.stride_D = torch::empty({num_segments}, opt_i64);
  meta.a_off = torch::empty({num_segments}, opt_i64);
  meta.b_off = torch::empty({num_segments}, opt_i64);
  meta.d_off = torch::empty({num_segments}, opt_i64);

  BuildGroupedGemmMetaKernel kernel{
      seg_indptr_i32.data_ptr<int32_t>(),
      weight_indices_i32.data_ptr<int32_t>(),
      meta.problem_sizes.data_ptr<int32_t>(),
      meta.stride_A.data_ptr<int64_t>(),
      meta.stride_B.data_ptr<int64_t>(),
      meta.stride_D.data_ptr<int64_t>(),
      meta.a_off.data_ptr<int64_t>(),
      meta.b_off.data_ptr<int64_t>(),
      meta.d_off.data_ptr<int64_t>(),
      N,
      K,
      elem_bytes,
      num_segments};
  submit_per_segment(queue, num_segments, kernel);
  return meta;
}

// Turn a base tensor + device byte-offsets into a device int64 pointer array
// (one absolute device address per segment) for the pointer-array grouped GEMM.
inline torch::Tensor make_device_ptrs(const torch::Tensor& base, const torch::Tensor& off_bytes, sycl::queue& queue) {
  const int64_t base_addr = reinterpret_cast<int64_t>(base.data_ptr());
  const int num_segments = static_cast<int>(off_bytes.numel());
  auto ptrs = torch::empty({num_segments}, off_bytes.options());

  MakeDevicePtrsKernel kernel{base_addr, off_bytes.data_ptr<int64_t>(), ptrs.data_ptr<int64_t>(), num_segments};
  submit_per_segment(queue, num_segments, kernel);
  return ptrs;
}

}  // namespace at::native::xpu
