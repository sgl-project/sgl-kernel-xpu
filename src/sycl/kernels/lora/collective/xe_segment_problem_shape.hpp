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
  \brief Grouped-GEMM problem shape of the chunked-SGMV LoRA kernels, read on the device.

  The problem shape of sycl-tla's grouped GemmUniversal, which calls only get_problem_shape(g) on
  it; the tile scheduler is the specialization in xe_segment_tile_scheduler.hpp. A group is one
  (segment, slice) with problem (segment rows, slice width, K), computed from seg_indptr /
  slice_offsets on the device, so no per-group shape array has to be built and no host sync is
  needed.
*/

#pragma once

#include <cstdint>
#include <cute/tensor.hpp>

#include "cutlass/cutlass.h"

namespace cutlass::lora::collective {

struct SegmentSliceProblemShape {
  using UnderlyingProblemShape = cute::Shape<int, int, int>;

  int32_t num_segments = 0;
  const int32_t* seg_indptr = nullptr;      // [num_segments + 1] over the logical row order
  const int32_t* weight_indices = nullptr;  // [num_segments] adapter of each segment
  const int32_t* slice_offsets = nullptr;   // [n_slices + 1] output-column bands; nullptr: one slice [0, N)
  int n_slices = 1;
  int N = 0;  // output columns of all slices
  int K = 0;
  int64_t num_rows = 0;  // host-known token count; sizes the scheduler's grid

  // Group id = segment | slice << kSliceShift: decoded with a shift and a mask, since Xe has no
  // integer divide and the id is decoded on the critical path of every tile.
  static constexpr int kSliceShift = 24;
  static constexpr int kMaxSegments = 1 << kSliceShift;
  CUTLASS_HOST_DEVICE static int32_t group_id(int segment, int slice) {
    return segment | (slice << kSliceShift);
  }
  CUTLASS_HOST_DEVICE int32_t groups() const {
    return num_segments * n_slices;
  }
  CUTLASS_HOST_DEVICE static int segment(int g) {
    return g & (kMaxSegments - 1);
  }
  CUTLASS_HOST_DEVICE static int slice(int g) {
    return g >> kSliceShift;
  }
  CUTLASS_HOST_DEVICE int row0(int g) const {
    return seg_indptr[segment(g)];
  }
  CUTLASS_HOST_DEVICE int rows(int g) const {
    return seg_indptr[segment(g) + 1] - seg_indptr[segment(g)];
  }
  CUTLASS_HOST_DEVICE int64_t adapter(int g) const {
    return weight_indices[segment(g)];
  }
  CUTLASS_HOST_DEVICE int n_lo(int g) const {
    return slice_offsets ? slice_offsets[slice(g)] : 0;
  }
  CUTLASS_HOST_DEVICE int n_hi(int g) const {
    return slice_offsets ? slice_offsets[slice(g) + 1] : N;
  }

  CUTLASS_HOST_DEVICE UnderlyingProblemShape const get_problem_shape(int32_t g) const {
    return {rows(g), n_hi(g) - n_lo(g), K};
  }
  // Deliberately no get_host_problem_shape() / is_host_problem_shape_available(): sycl-tla's stock
  // group scheduler needs them, so it cannot compile with this shape. Only the specialization in
  // xe_segment_tile_scheduler.hpp (which reads the shapes on the device) can.
};

}  // namespace cutlass::lora::collective
