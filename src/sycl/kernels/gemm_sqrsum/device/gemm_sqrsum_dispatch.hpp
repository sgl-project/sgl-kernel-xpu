/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
/*!
  \file
  \brief Forward declarations for generated GEMM+SqrSum kernel launch functions
*/

#pragma once

#include <ATen/ATen.h>
#include <sycl/sycl.hpp>

namespace gemm_sqrsum {

// Each function is defined in a separate generated .cpp file from
// gemm_sqrsum_kernel.cpp.in, compiled as its own library.
//
// Naming: launch_gemm_sqrsum_<ELEM_TAG>_<TILE_M>x<TILE_N>x<TILE_K>
// Parameters:
//   ELEM_TAG  in {half, bf16, float}
//   TILE_M, TILE_N, TILE_K: tile dimensions

#define DECLARE_GEMM_SQRSUM_LAUNCH(ELEM, TM, TN, TK) \
  void launch_gemm_sqrsum_##ELEM##_##TM##x##TN##x##TK( \
      at::Tensor& C,                                   \
      at::Tensor& sqrsum,                              \
      const at::Tensor& A,                             \
      const at::Tensor& B);

#define DECLARE_GEMM_SQRSUM_ALL_TILES(ELEM) \
  DECLARE_GEMM_SQRSUM_LAUNCH(ELEM, 256, 256, 16)

// Only half and bf16 supported (DPAS limitation - no float x float)
DECLARE_GEMM_SQRSUM_ALL_TILES(half)
DECLARE_GEMM_SQRSUM_ALL_TILES(bf16)

#undef DECLARE_GEMM_SQRSUM_LAUNCH
#undef DECLARE_GEMM_SQRSUM_ALL_TILES

}  // namespace gemm_sqrsum
