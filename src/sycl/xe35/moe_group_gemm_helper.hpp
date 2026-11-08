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

// Namespace scope, not function scope: as locals these get captured, and under
// -fsycl-host-compiler g++ and the device compiler disagree on the lambda size.
constexpr int kBScaleTransposeTile = 64;  // one cache line either way
constexpr int kBScaleTransposeWG = 256;
// +4 B so the strided pass steps 17 dwords per lane, keeping a subgroup off one bank.
constexpr int kBScaleTransposePitch = kBScaleTransposeTile + 4;

// ---------------------------------------------------------------------------
// Helper 1: fill {5, num_experts} int64 ptr_table from 3D uniform tensors.
// Rows are a / b / out / scales_a / scales_b.
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
// Helper 2: A-scale transpose + pointer-build for the FP4 adapter. scales_a is 3D and
// indexed by scale_offsets[], not expert_offsets[].
// ---------------------------------------------------------------------------
inline void launch_fp4_transpose_scales_and_build_pointers(
    sycl::queue& queue,
    int num_experts,
    int max_m,
    int packed_k,
    [[maybe_unused]] int n,
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
// Helper 3: ptr-build + A-scale transpose for MXFP8 / FP8. A, scales_a and output are
// flat 2D indexed by expert_offsets[]; B and scales_b stay 3D. a_scales_out is padded
// (E, scale_cols, padded_max_m) so the stride between scale columns matches the
// StrideScaleA override rather than the expert's own m_i.
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

// u8 B-scale transpose: (E, N, cols) -> (E, cols, N), needed because StrideScaleB is
// MN-major. Runs on every call. One work-item per element dirtied a cache line per
// byte written, ~1 GB/layer at Qwen3 sizes; staging a tile through shared local memory
// lets both the reads and the writes use whole lines.
inline void launch_u8_transpose_b_scales(
    sycl::queue& queue,
    int num_experts,
    int n,
    int scale_cols,
    const uint8_t* b_scales_in_ptr,
    uint8_t* b_scales_out_ptr) {
  const int n_tiles = (n + kBScaleTransposeTile - 1) / kBScaleTransposeTile;
  const int s_tiles = (scale_cols + kBScaleTransposeTile - 1) / kBScaleTransposeTile;
  const size_t groups = static_cast<size_t>(num_experts) * n_tiles * s_tiles;

  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<uint8_t, 1> tile(static_cast<size_t>(kBScaleTransposeTile) * kBScaleTransposePitch, cgh);
    cgh.parallel_for<TransposeScalesU8B>(
        sycl::nd_range<1>(groups * kBScaleTransposeWG, kBScaleTransposeWG), [=](sycl::nd_item<1> item) {
          const int g = static_cast<int>(item.get_group(0));
          const int lid = static_cast<int>(item.get_local_id(0));

          const int s_tile = g % s_tiles;
          const int n_tile = (g / s_tiles) % n_tiles;
          const int e = g / (s_tiles * n_tiles);

          const int n0 = n_tile * kBScaleTransposeTile;
          const int s0 = s_tile * kBScaleTransposeTile;
          const int ne = (n - n0) < kBScaleTransposeTile ? (n - n0) : kBScaleTransposeTile;
          const int se = (scale_cols - s0) < kBScaleTransposeTile ? (scale_cols - s0) : kBScaleTransposeTile;
          const int count = ne * se;

          const int64_t in_base = static_cast<int64_t>(e) * n * scale_cols;
          const int64_t out_base = static_cast<int64_t>(e) * scale_cols * n;

          for (int idx = lid; idx < count; idx += kBScaleTransposeWG) {
            const int s_off = idx % se;
            const int n_off = idx / se;
            tile[s_off * kBScaleTransposePitch + n_off] =
                b_scales_in_ptr[in_base + static_cast<int64_t>(n0 + n_off) * scale_cols + s0 + s_off];
          }

          sycl::group_barrier(item.get_group());

          for (int idx = lid; idx < count; idx += kBScaleTransposeWG) {
            const int n_off = idx % ne;
            const int s_off = idx / ne;
            b_scales_out_ptr[out_base + static_cast<int64_t>(s0 + s_off) * n + n0 + n_off] =
                tile[s_off * kBScaleTransposePitch + n_off];
          }
        });
  });
}
