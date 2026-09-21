/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
/*!
  \file
  \brief Compile-time skip flags for the two-stage sparse MLA kernels, for timing only.

    Every flag defaults to 0: the kernel runs as it is. Set one to 1 and that section is
    compiled out -- results are WRONG, but the time difference against the default build
    is roughly that section's cost. Deltas do not sum to the total (memory and DPAS
    overlap), and hiding a producer can fold its consumers away too.

    Pass them via the MLA_SPARSE_ABLATE build variable (see src/CMakeLists.txt), bare
    NAME=VALUE without the -D; several at once, separated by commas or spaces:

      MLA_SPARSE_ABLATE="SGLANG_MLA_SPARSE_SKIP_QK=1, SGLANG_MLA_SPARSE_SKIP_PV=1" \
        pip install -e . --no-build-isolation

    Coarse on purpose: find the section that dominates first, then add finer guards inside
    it.

    Retired 2026-09-03: SGLANG_MLA_SPARSE_EXACT_DEQUANT and SGLANG_MLA_SPARSE_HOIST_SCALES
    briefly lived in kernel/xe_mla_sparse_2stage_gather_kernel.hpp and were not of this
    kind -- both settings of each were numerically correct, and they existed to price one
    optimization each. Both questions are answered (the integer dequant was 0.7-5.6%
    slower and is gone; the hoisted scale load was neutral and is now unconditional), so
    the flags and the code they selected are removed. See section 4.6 of
    docs/MLA_SPARSE_STAGE1_GATHER_OPTIMIZATION.md, which keeps the measurements and the
    removed algorithm.
*/

#pragma once

// Stage 1 -- sparse gather (kernel/xe_mla_sparse_2stage_gather_kernel.hpp)
#ifndef SGLANG_MLA_SPARSE_SKIP_G1_INDEX
#define SGLANG_MLA_SPARSE_SKIP_G1_INDEX 0  // topk index read + paged address resolve
#endif
#ifndef SGLANG_MLA_SPARSE_SKIP_G1_COPY
#define SGLANG_MLA_SPARSE_SKIP_G1_COPY 0  // KV read + dequant + store into the gathered tile
#endif

// Stage 2 -- mainloop (collective/xe_mla_sparse_2stage_mainloop.hpp)
#ifndef SGLANG_MLA_SPARSE_SKIP_QK
#define SGLANG_MLA_SPARSE_SKIP_QK 0  // Q/K loads, reorders, QK DPAS
#endif
#ifndef SGLANG_MLA_SPARSE_SKIP_MASK
#define SGLANG_MLA_SPARSE_SKIP_MASK 0  // valid-mask scan writing -inf into invalid columns
#endif
#ifndef SGLANG_MLA_SPARSE_SKIP_SOFTMAX
#define SGLANG_MLA_SPARSE_SKIP_SOFTMAX 0  // running max/exp2/sum, exp-sum and O rescale
#endif
#ifndef SGLANG_MLA_SPARSE_SKIP_PV
#define SGLANG_MLA_SPARSE_SKIP_PV 0  // V loads, reorder, PV DPAS
#endif

// Stage 2 -- epilogue (collective/xe_mla_sparse_2stage_epilogue.hpp)
#ifndef SGLANG_MLA_SPARSE_SKIP_EPI_REDUCE
#define SGLANG_MLA_SPARSE_SKIP_EPI_REDUCE 0  // cross-subgroup SLM reduction (ReduceK > 1 only)
#endif
#ifndef SGLANG_MLA_SPARSE_SKIP_EPI_FINISH
#define SGLANG_MLA_SPARSE_SKIP_EPI_FINISH 0  // LSE store, attn_sink merge, softmax normalization
#endif
#ifndef SGLANG_MLA_SPARSE_SKIP_EPI_STORE_O
#define SGLANG_MLA_SPARSE_SKIP_EPI_STORE_O 0  // store-order reorder + block-2D store of O
#endif
