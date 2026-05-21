/***************************************************************************************************
 * Copyright 2025 SGLang Team. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 **************************************************************************************************/
/*! \file
    \brief Header-only SYCL helpers for building CUTLASS grouped-GEMM pointer tables on device.

    Provides two inline launcher functions and their named SYCL kernel class tags:

      BuildGroupGemmPointers          — fill {5, E} int64 ptr_table from uniform 3D tensors
      TransposeScalesAndBuildPointers — combined A-scale transpose + pointer-build for the
                                        FP4 2D-flat adapter (cutlass_fp4_group_mm)
*/

#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

// ---------------------------------------------------------------------------
// Named kernel class tags (must be at TU scope for SYCL named-kernel rules)
// ---------------------------------------------------------------------------

/// Tag for the uniform 3D pointer-table fill kernel.
class BuildGroupGemmPointers;

/// Tag for the FP4 2D-flat adapter scatter + pointer-build kernel.
class TransposeScalesAndBuildPointers;

// ---------------------------------------------------------------------------
// Helper 1: fill {5, num_experts} int64 ptr_table from 3D uniform tensors.
//
// One work-item per expert.
// Row layout:  0 = a_ptrs,  1 = b_ptrs,  2 = out_ptrs,
//              3 = scales_a_ptrs,  4 = scales_b_ptrs.
// Each entry   = base_bytes + expert_id * stride_bytes.
// ---------------------------------------------------------------------------
inline void build_group_gemm_pointers(
    sycl::queue& queue,
    int num_experts,
    int64_t* ptr_table,
    int64_t a_base,
    int64_t a_stride,
    int64_t b_base,
    int64_t b_stride,
    int64_t out_base,
    int64_t out_stride,
    int64_t sa_base,
    int64_t sa_stride,
    int64_t sb_base,
    int64_t sb_stride) {
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<BuildGroupGemmPointers>(sycl::range<1>(static_cast<size_t>(num_experts)), [=](sycl::id<1> id) {
      const int e = static_cast<int>(id[0]);
      const int64_t ei = static_cast<int64_t>(e);
      ptr_table[0 * num_experts + e] = a_base + ei * a_stride;
      ptr_table[1 * num_experts + e] = b_base + ei * b_stride;
      ptr_table[2 * num_experts + e] = out_base + ei * out_stride;
      ptr_table[3 * num_experts + e] = sa_base + ei * sa_stride;
      ptr_table[4 * num_experts + e] = sb_base + ei * sb_stride;
    });
  });
}

// ---------------------------------------------------------------------------
// Helper 2: combined A-scale transpose + pointer-build for the FP4 adapter.
//
// Work-range = {num_experts, max_m}.
// work-item(e, 0) writes 5 pointer-table entries for expert e.
// work-item(e, r) transposes one row of A-scales for expert e:
//   a_scales_3d[e, s, r] = a_scales_flat[scale_offsets[e] + r, s]
// ---------------------------------------------------------------------------
inline void launch_fp4_transpose_scales_and_build_pointers(
    sycl::queue& queue,
    int num_experts,
    int max_m,
    int packed_k,
    int n,
    int scale_cols,
    const int32_t* problem_sizes_ptr,
    const int32_t* expert_offsets_ptr,
    const int32_t* scale_offsets_ptr,
    const uint8_t* a_flat_ptr,  // base pointer into flat A (unused in body; a_flat_base encodes it)
    const uint8_t* a_scales_flat_ptr,
    uint8_t* a_scales_3d_ptr,
    int64_t* ptr_table_ptr,
    int64_t a_flat_base,
    int64_t b_base,
    int64_t output_3d_base,
    int64_t a_scales_3d_base,
    int64_t b_scales_3d_base,
    int64_t b_expert_stride,
    int64_t output_expert_stride,
    int64_t a_scales_expert_stride,
    int64_t b_scales_expert_stride) {
  (void)a_flat_ptr;  // a_flat_base already encodes this address
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<TransposeScalesAndBuildPointers>(
        sycl::range<2>(static_cast<size_t>(num_experts), static_cast<size_t>(max_m)), [=](sycl::id<2> id) {
          const int expert_idx = static_cast<int>(id[0]);
          const int row = static_cast<int>(id[1]);

          // First work-item for each expert fills the 5 pointer-table entries
          if (row == 0) {
            const int64_t ei = static_cast<int64_t>(expert_idx);
            const int row_off = expert_offsets_ptr[expert_idx];
            // Row 0: point directly into flat A at this expert's row offset
            ptr_table_ptr[0 * num_experts + expert_idx] = a_flat_base + static_cast<int64_t>(row_off) * packed_k;
            ptr_table_ptr[1 * num_experts + expert_idx] = b_base + ei * b_expert_stride;
            ptr_table_ptr[2 * num_experts + expert_idx] = output_3d_base + ei * output_expert_stride;
            ptr_table_ptr[3 * num_experts + expert_idx] = a_scales_3d_base + ei * a_scales_expert_stride;
            ptr_table_ptr[4 * num_experts + expert_idx] = b_scales_3d_base + ei * b_scales_expert_stride;
          }

          // Skip padding rows beyond this expert's actual token count
          const int expert_m = problem_sizes_ptr[expert_idx * 3];
          if (row >= expert_m) return;

          // Transpose A-scales: a_scales_3d[expert, s, row] = a_scales_flat[offset + row, s]
          const int scale_offset = scale_offsets_ptr[expert_idx];
          const int64_t src_row = static_cast<int64_t>(scale_offset) + row;
          for (int s = 0; s < scale_cols; ++s) {
            a_scales_3d_ptr
                [static_cast<int64_t>(expert_idx) * a_scales_expert_stride + static_cast<int64_t>(s) * max_m + row] =
                    a_scales_flat_ptr[src_row * scale_cols + s];
          }
        });
  });
}
