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

    Kernel-agnostic helpers that construct the per-group problem sizes, strides,
    byte offsets, and absolute device pointer arrays that a CUTLASS pointer-array
    grouped GEMM consumes. Everything is built on device (one SYCL thread per
    group, no host round-trip), so this header is reused across the LoRA
    grouped-GEMM launchers (A-fwd, B-fwd, fused QKV / gate-up, ...).

    The build is *sliced*: each input segment s emits n_slices groups, one per
    projection p in [0, n_slices). Group g = n_slices * s + p. The per-group N and
    the A/D column bands come from an optional output_offset table, and the A
    operand's leading dim / per-slice column advance are passed in explicitly --
    so one kernel covers every fused shape:

      * plain SGEMM LoRA-A / LoRA-B: n_slices = 1, no output_offset (N_p = N_total,
        col_start = 0), A un-sliced (a_row_stride = K, a_slice_stride = 0).
      * fused QKV / gate-up LoRA-B: n_slices in {2, 3}, output_offset gives the
        variable per-projection N, A is the packed [num_tokens, n_slices*K] input
        (a_row_stride = n_slices*K, a_slice_stride = K).
      * fused QKV / gate-up LoRA-A: A is the *shared* [num_tokens, in_dim] input
        (a_row_stride = K = in_dim, a_slice_stride = 0), output_offset gives the
        uniform per-projection rank band.
*/

#pragma once

#include <ATen/ATen.h>
#include <torch/all.h>

#include <optional>
#include <sycl/sycl.hpp>

#include "sycl/SYCLHelpers.h"

namespace at::native::xpu {

//----------------- Shared device-side grouped-GEMM metadata ------------------//
//
// Per-group problem sizes, element byte-offsets, and strides that the CUTLASS
// pointer-array grouped GEMM consumes. Built on device by a single SYCL kernel
// (one thread per group). num_groups == n_slices * num_segments; the plain
// (unsliced) build is just n_slices == 1, so num_groups == num_segments.
//
// For group g = n_slices * s + p (segment s, projection p), with
// row_start = seg_indptr[s], and the projection column band [col_start, col_start
// + N_p) taken from output_offset (or [0, N_total) when no output_offset):
//   M_g = seg_indptr[s+1] - row_start
//
//   a_off[g] = (row_start * a_row_stride + p * a_slice_stride)  * elem_bytes  (into A)
//   b_off[g] = (lora_id * N_total * K + col_start * K)          * elem_bytes  (into B)
//   d_off[g] = (row_start * N_total + col_start)                * elem_bytes  (into D)

struct GroupedGemmMeta {
  torch::Tensor problem_sizes;  // int32 [num_groups, 3]  (M_g, N_p, K), on device
  torch::Tensor stride_A;       // int64 [num_groups]     leading dim of A = a_row_stride
  torch::Tensor stride_B;       // int64 [num_groups]     leading dim of B = K
  torch::Tensor stride_D;       // int64 [num_groups]     leading dim of D = N_total
  torch::Tensor a_off;          // int64 [num_groups]     byte offset into A per group (device)
  torch::Tensor b_off;          // int64 [num_groups]     byte offset into B per group (device)
  torch::Tensor d_off;          // int64 [num_groups]     byte offset into D per group (device)
  // Per-group epilogue alpha: a *contiguous* fp32 value buffer (one alpha per
  // group). Callers may build an alpha_ptr_array (one pointer per group into
  // this buffer) for the grouped epilogue. Undefined when no scalings were
  // supplied (A-fwd), in which case the caller falls back to a single broadcast alpha.
  torch::Tensor alpha;  // float32 [num_groups], or undefined
};

// One thread per group g in [0, n_slices * num_segments): decode (s, p) =
// (g / n_slices, g % n_slices), derive M_s / lora_id from the index tensors and
// the per-projection column band from output_offset, then write the per-group
// problem size, strides, byte offsets, and alpha straight into device memory.
struct BuildGroupedGemmMetaKernel {
  const int32_t* seg_indptr;      // [num_segments + 1]
  const int32_t* weight_indices;  // [num_segments]
  const int32_t* output_offset;   // [n_slices + 1] projection column boundaries, or nullptr (single slice)
  const float* scalings;          // [num_loras], or nullptr (A-fwd)
  int32_t* problem_sizes;         // [num_groups * 3]
  int64_t* stride_A;              // [num_groups]
  int64_t* stride_B;              // [num_groups]
  int64_t* stride_D;              // [num_groups]
  int64_t* a_off;                 // [num_groups]
  int64_t* b_off;                 // [num_groups]
  int64_t* d_off;                 // [num_groups]
  float* alpha;                   // [num_groups] contiguous alpha values, or nullptr
  int N_total;                    // total output columns (== output_offset[n_slices] when sliced)
  int K;                          // reduction dim
  int64_t elem_bytes;
  int num_segments;
  int n_slices;              // projections packed per segment (1 = plain)
  int64_t a_row_stride;      // leading dim of A (K when unsliced, n_slices*K for packed B-fwd)
  int64_t a_slice_stride;    // per-slice column advance in A (0 when A is shared, K for packed B-fwd)

  void operator()(sycl::nd_item<1> item) const {
    const int g = static_cast<int>(item.get_global_linear_id());
    const int num_groups = n_slices * num_segments;
    if (g >= num_groups) {
      return;
    }
    const int s = g / n_slices;  // segment
    const int p = g % n_slices;  // projection

    const int32_t row_start = seg_indptr[s];
    const int32_t M_s = seg_indptr[s + 1] - row_start;
    const int32_t lora_id = weight_indices[s];

    // Projection column band. Without an output_offset the single slice spans the
    // full N_total (the plain SGEMM case).
    const int32_t col_start = output_offset ? output_offset[p] : 0;
    const int32_t N_p = output_offset ? (output_offset[p + 1] - col_start) : N_total;

    problem_sizes[3 * g + 0] = M_s;
    problem_sizes[3 * g + 1] = N_p;
    problem_sizes[3 * g + 2] = K;

    // Leading dims (in elements): A spans a_row_stride, B rows are the contiguous
    // K reduction band, D spans the full N_total output row.
    stride_A[g] = a_row_stride;
    stride_B[g] = static_cast<int64_t>(K);
    stride_D[g] = static_cast<int64_t>(N_total);

    a_off[g] = (static_cast<int64_t>(row_start) * a_row_stride + static_cast<int64_t>(p) * a_slice_stride) * elem_bytes;
    b_off[g] = (static_cast<int64_t>(lora_id) * N_total * K + static_cast<int64_t>(col_start) * K) * elem_bytes;
    d_off[g] = (static_cast<int64_t>(row_start) * N_total + static_cast<int64_t>(col_start)) * elem_bytes;

    if (scalings) {
      alpha[g] = scalings[lora_id];
    }
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

// Build the pointer-array grouped-GEMM metadata (n_slices * num_segments groups)
// on device. The trailing (output_offset_i32, n_slices, a_row_stride,
// a_slice_stride) parameters default to the plain single-slice build, so plain
// SGEMM A-/B-fwd callers pass none of them; fused QKV / gate-up callers supply
// the projection column boundaries (output_offset) and the packed-A strides.
inline GroupedGemmMeta build_grouped_gemm_meta(
    const torch::Tensor& seg_indptr_i32,      // int32 [num_segments + 1]
    const torch::Tensor& weight_indices_i32,  // int32 [num_segments]
    const int N_total,
    const int K,
    const int num_segments,
    const int64_t elem_bytes,
    const at::Device device,
    sycl::queue& queue,
    const std::optional<torch::Tensor>& scalings = std::nullopt,        // float32 [num_loras], or nullopt (A-fwd)
    const std::optional<torch::Tensor>& output_offset_i32 = std::nullopt,  // int32 [n_slices + 1], or nullopt (1 slice)
    const int n_slices = 1,
    const int64_t a_row_stride = 0,   // leading dim of A; 0 -> default K (single-slice A)
    const int64_t a_slice_stride = 0  // per-slice column advance in A; 0 -> A shared / single slice
) {
  const int num_groups = n_slices * num_segments;
  // Default the A leading dim to K (the plain single-slice case).
  const int64_t a_ld = a_row_stride > 0 ? a_row_stride : static_cast<int64_t>(K);

  auto opt_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
  auto opt_i64 = torch::TensorOptions().dtype(torch::kInt64).device(device);
  auto opt_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);

  GroupedGemmMeta meta;
  meta.problem_sizes = torch::empty({num_groups, 3}, opt_i32);
  meta.stride_A = torch::empty({num_groups}, opt_i64);
  meta.stride_B = torch::empty({num_groups}, opt_i64);
  meta.stride_D = torch::empty({num_groups}, opt_i64);
  meta.a_off = torch::empty({num_groups}, opt_i64);
  meta.b_off = torch::empty({num_groups}, opt_i64);
  meta.d_off = torch::empty({num_groups}, opt_i64);
  // Only materialize the per-group alpha buffer when scalings are supplied;
  // otherwise leave it undefined so the caller uses a single broadcast alpha.
  if (scalings.has_value()) {
    meta.alpha = torch::empty({num_groups}, opt_f32);
  }

  BuildGroupedGemmMetaKernel kernel{
      seg_indptr_i32.data_ptr<int32_t>(),
      weight_indices_i32.data_ptr<int32_t>(),
      output_offset_i32.has_value() ? output_offset_i32->data_ptr<int32_t>() : nullptr,
      scalings.has_value() ? scalings->data_ptr<float>() : nullptr,
      meta.problem_sizes.data_ptr<int32_t>(),
      meta.stride_A.data_ptr<int64_t>(),
      meta.stride_B.data_ptr<int64_t>(),
      meta.stride_D.data_ptr<int64_t>(),
      meta.a_off.data_ptr<int64_t>(),
      meta.b_off.data_ptr<int64_t>(),
      meta.d_off.data_ptr<int64_t>(),
      meta.alpha.defined() ? meta.alpha.data_ptr<float>() : nullptr,
      N_total,
      K,
      elem_bytes,
      num_segments,
      n_slices,
      a_ld,
      a_slice_stride};
  // submit_per_segment rounds the global range up to a whole work-group; pass the
  // group count so all n_slices * num_segments groups get a thread.
  submit_per_segment(queue, num_groups, kernel);
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
