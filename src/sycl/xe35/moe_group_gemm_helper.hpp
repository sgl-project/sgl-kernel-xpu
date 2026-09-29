/***************************************************************************************************
 * Copyright 2025 SGLang Team. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 **************************************************************************************************/
/*! \file
    \brief Header-only SYCL helpers for building CUTLASS grouped-GEMM pointer tables on device.

    Provides inline launcher functions and their named SYCL kernel class tags:

      BuildGroupGemmPointers          — fill {5, E} int64 ptr_table from uniform 3D tensors
      TransposeScalesAndBuildPointers — combined A-scale transpose + pointer-build for the
                                        NVFP4 2D-flat adapter (cutlass_fp4_group_mm)
      BuildPointersAndTransposeScalesMxFp8Flat — fp32-A-scale transpose + ptr-build for the
                                        MXFP8/FP8 flat-2D path with ragged m_i
      BuildPointersAndTransposeScalesU8Flat — u8-A-scale transpose + ptr-build for
                                        flat-2D u8-scale paths (MXFP4, MXFP8).
      TransposeScalesU8B              — u8 B-scale transpose (E, N, cols) ->
                                        (E, cols, N). Used by MXFP4 and MXFP8.
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

/// Tag for the MXFP8 flat-layout combined ptr-build + A-scale transpose kernel
/// (A / scales_a / output are flat 2D; per-expert offsets come from
/// expert_offsets[]).
class BuildPointersAndTransposeScalesMxFp8Flat;

/// u8-A-scale ptr-build + A-scale transpose (MXFP4 and MXFP8).
class BuildPointersAndTransposeScalesU8Flat;

/// u8 B-scale transpose (MXFP4 and MXFP8).
class TransposeScalesU8B;

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

// ---------------------------------------------------------------------------
// Helper 3: combined ptr-build + A-scale transpose for MXFP8 / FP8 blockwise
//           grouped GEMM with **flat 2D** A / scales_a / output. Per-expert
//           pointers come from expert_offsets[]; B / scales_b are still 3D.
//
// Inputs:
//   - a_scales_in:  fp32, shape (sum_m_i, scale_cols), row-major flat.
//                   Per-expert sub-slice: rows expert_offsets[e]..[e]+m_i.
//   - a_scales_out: fp32, padded (E, scale_cols, max_m).
//                   Each expert's slice is packed with stride max_m between
//                   scale columns (matches StrideScaleA override = max_m).
//   - problem_sizes_ptr: per-expert (M_i, N, K). Used to bound the transpose
//                        loop so we never read past the expert's actual rows.
//   - expert_offsets_ptr: cumulative per-expert row index (in tokens). Expert
//                         e's flat-A rows start at expert_offsets[e].
//
// Work-range = {num_experts, max_m}.
//   work-item(e, 0)            writes 5 pointer-table entries for expert e.
//   work-item(e, r), r < m_e   transposes one row of A-scales for expert e.
//
// Pointer-table layout (matches the other helpers):
//   row 0 = a_ptrs (a_base + expert_offsets[e] * K * a_elem),
//   row 1 = b_ptrs (b_base + e * b_stride),
//   row 2 = out_ptrs (out_base + expert_offsets[e] * N * out_elem),
//   row 3 = a_scales_ptrs -> transposed buffer slot (a_scales_out_base + e * stride),
//   row 4 = b_scales_ptrs (b_scales_base + e * b_scales_stride).
// ---------------------------------------------------------------------------
inline void launch_mxfp8_build_pointers_and_transpose_scales_flat(
    sycl::queue& queue,
    int num_experts,
    int max_m,
    int scale_cols,
    int n,               // for output per-expert offset
    int k,               // for A per-expert offset
    int a_elem_bytes,    // sizeof(fp8_e4m3) = 1
    int out_elem_bytes,  // sizeof(float) = 4
    const int32_t* problem_sizes_ptr,
    const int32_t* expert_offsets_ptr,
    const float* a_scales_in_ptr,
    float* a_scales_out_ptr,
    int64_t* ptr_table_ptr,
    int64_t a_base,
    int64_t b_base,
    int64_t b_stride,
    int64_t out_base,
    int64_t a_scales_out_base,
    int64_t a_scales_out_stride,
    int64_t b_scales_base,
    int64_t b_scales_stride,
    int64_t a_scales_in_row_stride_elems,  // typically scale_cols (row-major)
    int64_t a_scales_out_stride_elems) {
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<BuildPointersAndTransposeScalesMxFp8Flat>(
        sycl::range<2>(static_cast<size_t>(num_experts), static_cast<size_t>(max_m)), [=](sycl::id<2> id) {
          const int expert_idx = static_cast<int>(id[0]);
          const int row = static_cast<int>(id[1]);

          if (row == 0) {
            const int64_t ei = static_cast<int64_t>(expert_idx);
            const int64_t row_off = static_cast<int64_t>(expert_offsets_ptr[expert_idx]);
            ptr_table_ptr[0 * num_experts + expert_idx] = a_base + row_off * static_cast<int64_t>(k) * a_elem_bytes;
            ptr_table_ptr[1 * num_experts + expert_idx] = b_base + ei * b_stride;
            ptr_table_ptr[2 * num_experts + expert_idx] = out_base + row_off * static_cast<int64_t>(n) * out_elem_bytes;
            ptr_table_ptr[3 * num_experts + expert_idx] = a_scales_out_base + ei * a_scales_out_stride;
            ptr_table_ptr[4 * num_experts + expert_idx] = b_scales_base + ei * b_scales_stride;
          }

          // Bound the transpose to this expert's actual row count.
          const int expert_m = problem_sizes_ptr[expert_idx * 3];
          if (row >= expert_m) return;

          // Source row in flat scales: expert_offsets[e] + row.
          const int64_t row_off = static_cast<int64_t>(expert_offsets_ptr[expert_idx]);
          const int64_t in_row_base = (row_off + row) * a_scales_in_row_stride_elems;
          const int64_t out_expert_base = static_cast<int64_t>(expert_idx) * a_scales_out_stride_elems;
          for (int s = 0; s < scale_cols; ++s) {
            a_scales_out_ptr[out_expert_base + static_cast<int64_t>(s) * max_m + row] =
                a_scales_in_ptr[in_row_base + s];
          }
        });
  });
}

// u8-A / u8-A-scale variant of helper 3 (MXFP4, MXFP8). a_row_stride_bytes
// is a.size(1) in bytes (packed_k for MXFP4, k for MXFP8; A is uint8 in both).
inline void launch_u8_scale_build_pointers_and_transpose_scales_flat(
    sycl::queue& queue,
    int num_experts,
    int max_m,
    int scale_cols,
    int n,
    int a_row_stride_bytes,
    int out_elem_bytes,
    const int32_t* problem_sizes_ptr,
    const int32_t* expert_offsets_ptr,
    const uint8_t* a_scales_in_ptr,
    uint8_t* a_scales_out_ptr,
    int64_t* ptr_table_ptr,
    int64_t a_base,
    int64_t b_base,
    int64_t b_stride,
    int64_t out_base,
    int64_t a_scales_out_base,
    int64_t a_scales_out_stride,
    int64_t b_scales_base,
    int64_t b_scales_stride,
    int64_t a_scales_in_row_stride_elems,
    int64_t a_scales_out_stride_elems) {
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<BuildPointersAndTransposeScalesU8Flat>(
        sycl::range<2>(static_cast<size_t>(num_experts), static_cast<size_t>(max_m)), [=](sycl::id<2> id) {
          const int expert_idx = static_cast<int>(id[0]);
          const int row = static_cast<int>(id[1]);

          if (row == 0) {
            const int64_t ei = static_cast<int64_t>(expert_idx);
            const int64_t row_off = static_cast<int64_t>(expert_offsets_ptr[expert_idx]);
            // A is uint8 (1 byte/elem); row stride is a_row_stride_bytes (=
            // packed_k for MXFP4, k for MXFP8).
            ptr_table_ptr[0 * num_experts + expert_idx] = a_base + row_off * static_cast<int64_t>(a_row_stride_bytes);
            ptr_table_ptr[1 * num_experts + expert_idx] = b_base + ei * b_stride;
            ptr_table_ptr[2 * num_experts + expert_idx] = out_base + row_off * static_cast<int64_t>(n) * out_elem_bytes;
            ptr_table_ptr[3 * num_experts + expert_idx] = a_scales_out_base + ei * a_scales_out_stride;
            ptr_table_ptr[4 * num_experts + expert_idx] = b_scales_base + ei * b_scales_stride;
          }

          const int expert_m = problem_sizes_ptr[expert_idx * 3];
          if (row >= expert_m) return;

          const int64_t row_off = static_cast<int64_t>(expert_offsets_ptr[expert_idx]);
          const int64_t in_row_base = (row_off + row) * a_scales_in_row_stride_elems;
          const int64_t out_expert_base = static_cast<int64_t>(expert_idx) * a_scales_out_stride_elems;
          for (int s = 0; s < scale_cols; ++s) {
            a_scales_out_ptr[out_expert_base + static_cast<int64_t>(s) * max_m + row] =
                a_scales_in_ptr[in_row_base + s];
          }
        });
  });
}

// u8 B-scale transpose: (E, N, cols) -> (E, cols, N). Required by MN-major
// StrideScaleB. One-time per-call cost since B is uniform per expert.
inline void launch_u8_transpose_b_scales(
    sycl::queue& queue,
    int num_experts,
    int n,
    int scale_cols,
    const uint8_t* b_scales_in_ptr,
    uint8_t* b_scales_out_ptr) {
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<TransposeScalesU8B>(
        sycl::range<3>(static_cast<size_t>(num_experts), static_cast<size_t>(n), static_cast<size_t>(scale_cols)),
        [=](sycl::id<3> id) {
          const int e = static_cast<int>(id[0]);
          const int row = static_cast<int>(id[1]);
          const int s = static_cast<int>(id[2]);
          const int64_t ei = static_cast<int64_t>(e);
          const int64_t in_idx = ei * static_cast<int64_t>(n) * scale_cols + static_cast<int64_t>(row) * scale_cols + s;
          const int64_t out_idx = ei * static_cast<int64_t>(scale_cols) * n + static_cast<int64_t>(s) * n + row;
          b_scales_out_ptr[out_idx] = b_scales_in_ptr[in_idx];
        });
  });
}
