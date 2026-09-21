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
/*!
  \file
  \brief Shared LoRA gather / scatter row-permute device kernel + launcher.

  This is the physical<->logical row reordering used by the chunked-SGMV LoRA
  shrink path: the decode batch interleaves LoRA adapters ("zigzag"), so tokens
  are permuted to group rows by adapter before the segmented grouped GEMM, then
  permuted back afterwards.

  `permutation[logical] -> physical` maps a logical (adapter-grouped) row index
  to the physical (original token) row index; it must be a bijection over
  [0, num_rows).

    gather : output[i, :]              = input[permutation[i], :]
    scatter: output[permutation[i], :] = input[i, :]

  Reused by both the standalone lora_gather_rows / lora_scatter_rows ops
  (LoraGatherScatter.cpp) and the fused chunked_sgmv_lora_shrink_forward
  entrypoint (ChunkedSgmvLoraShrink.cpp), so the copy kernel lives in exactly
  one place. Callers own validation + the int64 cast of `permutation`.
*/

#pragma once

#include <ATen/ATen.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>

#include "sycl/SYCLHelpers.h"

namespace lora_permute_rows_impl {

//----------------- torch dtype -> SYCL element type --------------------//

template <typename T>
struct ToSyclElementType {
  using type = T;
};

template <>
struct ToSyclElementType<at::Half> {
  using type = sycl::half;
};

template <>
struct ToSyclElementType<at::BFloat16> {
  using type = sycl::ext::oneapi::bfloat16;
};

//----------------- Kernel definition --------------------//

// GATHER == true : output[row]       = input[perm[row]]
// GATHER == false: output[perm[row]] = input[row]
template <typename ElemType, bool GATHER>
struct LoraPermuteRows : public __SYCL_KER_CONFIG_CONVENTION__ {
  const ElemType* src;
  ElemType* dst;
  const int64_t* perm;
  int64_t num_rows;
  int64_t width;

  static constexpr int WG = 256;
  static constexpr int sub_group_size = 16;

  LoraPermuteRows(const ElemType* src, ElemType* dst, const int64_t* perm, int64_t num_rows, int64_t width)
      : src(src), dst(dst), perm(perm), num_rows(num_rows), width(width) {}

  [[sycl::reqd_sub_group_size(sub_group_size)]]
  void operator()(sycl::nd_item<3> item) const {
    const int64_t row = static_cast<int64_t>(item.get_group(0));
    if (row >= num_rows) {
      return;
    }

    const int64_t p = perm[row];
    // gather reads the permuted (physical) row and writes the logical row;
    // scatter reads the logical row and writes the permuted (physical) row.
    const int64_t src_row = GATHER ? p : row;
    const int64_t dst_row = GATHER ? row : p;

    const ElemType* s = src + src_row * width;
    ElemType* d = dst + dst_row * width;

    for (int64_t c = static_cast<int64_t>(item.get_local_id(2)); c < width; c += WG) {
      d[c] = s[c];
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) const {}
};

//----------------- Kernel launch --------------------//

template <typename TensorDType, bool GATHER>
void launch_permute_rows(
    const torch::Tensor& src,
    torch::Tensor& dst,
    const torch::Tensor& perm_i64,
    const int64_t num_rows,
    const int64_t width,
    sycl::queue& queue) {
  using ElemType = typename ToSyclElementType<TensorDType>::type;
  constexpr int WG = LoraPermuteRows<ElemType, GATHER>::WG;

  sycl::range<3> local(1, 1, WG);
  sycl::range<3> global(static_cast<size_t>(num_rows), 1, static_cast<size_t>(WG));

  auto kernel = LoraPermuteRows<ElemType, GATHER>(
      reinterpret_cast<const ElemType*>(src.data_ptr<TensorDType>()),
      reinterpret_cast<ElemType*>(dst.data_ptr<TensorDType>()),
      perm_i64.data_ptr<int64_t>(),
      num_rows,
      width);
  sycl_kernel_submit(global, local, queue, kernel);
}

// Dtype-dispatched row permute. `perm_i64` must already be int64 and validated
// (numel == num_rows, values in [0, num_rows)). No-op when num_rows or width
// is 0. Supports fp32 / fp16 / bf16.
template <bool GATHER>
void permute_rows_dispatch(
    const torch::Tensor& src, torch::Tensor& dst, const torch::Tensor& perm_i64, sycl::queue& queue) {
  const int64_t num_rows = src.size(0);
  const int64_t width = src.size(1);
  if (num_rows == 0 || width == 0) {
    return;
  }
  switch (src.scalar_type()) {
    case torch::kFloat32:
      launch_permute_rows<float, GATHER>(src, dst, perm_i64, num_rows, width, queue);
      break;
    case torch::kHalf:
      launch_permute_rows<at::Half, GATHER>(src, dst, perm_i64, num_rows, width, queue);
      break;
    case torch::kBFloat16:
      launch_permute_rows<at::BFloat16, GATHER>(src, dst, perm_i64, num_rows, width, queue);
      break;
    default:
      TORCH_CHECK(false, "Unsupported dtype for lora gather/scatter: ", src.scalar_type());
  }
}

}  // namespace lora_permute_rows_impl
