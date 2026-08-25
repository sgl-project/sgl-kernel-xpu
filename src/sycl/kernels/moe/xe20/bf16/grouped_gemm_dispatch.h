// Single source of truth for the bf16 MoE grouped-GEMM tile dispatch, shared by
// the AOT dispatcher (src/sycl/GroupGemmXe20.cpp) and the runtime-JIT wrapper
// (src/jit/moe_jit.cpp). Kept dependency-free (no torch, no SYCL, no cute) so it
// can be included both by the torch/SYCL AOT translation unit and by the
// torch-free `sgl_jit` static library.
//
// The tile-selection decision (grouped_gemm_select_tile) and the per-tile
// (Shape, Layout) tokens live here once: AOT pastes the tokens into cute
// Shape<>/Layout<> template args; the JIT wrapper stringifies them into template
// substitutions. Keep the SGL_MOE_GG_SHAPE_n / SGL_MOE_GG_LAYOUT_n rows and the
// select function's return ids in sync -- they now share one definition, so
// retuning the tiles touches only this file.
#pragma once

#include <cstdint>

namespace sgl {
namespace moe {

// BF16 grouped-GEMM tile-selection cutoff. Python mirrors this value in
// python/sgl_kernel/moe.py (documented there); update both when retuning.
inline constexpr int64_t kGroupedGemmSmallWeightThreshold = int64_t(4096) * 4096;

// Per-tile (Shape, Layout) tokens. Row id == the value returned by
// grouped_gemm_select_tile below.
#define SGL_MOE_GG_SHAPE_0 Shape<_8, _64, _32>
#define SGL_MOE_GG_SHAPE_1 Shape<_16, _64, _32>
#define SGL_MOE_GG_SHAPE_2 Shape<_32, _64, _32>
#define SGL_MOE_GG_SHAPE_3 Shape<_128, _64, _32>
#define SGL_MOE_GG_SHAPE_4 Shape<_128, _128, _32>
#define SGL_MOE_GG_SHAPE_5 Shape<_256, _64, _32>
#define SGL_MOE_GG_SHAPE_6 Shape<_256, _256, _32>

#define SGL_MOE_GG_LAYOUT_0 Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>
#define SGL_MOE_GG_LAYOUT_1 Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>
#define SGL_MOE_GG_LAYOUT_2 Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>
#define SGL_MOE_GG_LAYOUT_3 Layout<Shape<_4, _2, _1>, Stride<_2, _1, _0>>
#define SGL_MOE_GG_LAYOUT_4 Layout<Shape<_4, _2, _1>, Stride<_2, _1, _0>>
#define SGL_MOE_GG_LAYOUT_5 Layout<Shape<_8, _2, _1>, Stride<_2, _1, _0>>
#define SGL_MOE_GG_LAYOUT_6 Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>

// Number of tiles in the table above.
inline constexpr int kGroupedGemmNumTiles = 7;

// Map a runtime problem shape to a tile id (index into the table above). The
// AOT dispatcher and the JIT wrapper both call this, so the selection logic is
// defined exactly once. `fuse_act` selects the fused-activation code path.
inline int grouped_gemm_select_tile(int avg_m, int gemm_k, int gemm_n, bool fuse_act) {
  const bool small_weight = static_cast<int64_t>(gemm_k) * gemm_n <= kGroupedGemmSmallWeightThreshold;
  const bool narrow_k = gemm_k <= 256;
  const bool narrow_n_fused = fuse_act && (gemm_n <= 512);

  if (avg_m <= 8) return 0;
  if (avg_m <= 16 && small_weight) return 1;
  if (avg_m <= 32 && small_weight) return 2;
  if (avg_m <= 128 && small_weight) return fuse_act ? 3 : 4;
  if (narrow_k) return fuse_act ? 3 : 4;
  if (narrow_n_fused) return 3;
  return fuse_act ? 5 : 6;
}

}  // namespace moe
}  // namespace sgl
