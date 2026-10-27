/***************************************************************************************************
 * Copyright 2026 Intel corporation. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
/*! \file
    \brief Architecture probe for the multi-architecture ("fat") wheel.

           Built once, independently of SYCL_INTEL_TARGET, and installed at the
           top level of the package. python/sgl_kernel/_arch.py loads it with
           ctypes to decide whether to import the _xe20 or _xe35 backend -- the
           real `query_device` op cannot be used for this because it lives
           inside the backend we are trying to choose.

           Deliberately a C entry point with out-parameters: `query_device`
           returns std::tuple<int64_t,int64_t>, which is returned via a hidden
           sret pointer and is not safely callable through ctypes.
*/

#include <cstdint>

#include "sgl_arch_query.h"
#include "sgl_kernel_export.h"

extern "C" SGL_KERNEL_EXPORT int sgl_probe_device_capability(int64_t device_index, int64_t* major, int64_t* minor) {
  if (major == nullptr || minor == nullptr) {
    return -1;
  }
  try {
    auto [maj, min] = sgl::device_capability(device_index);
    *major = maj;
    *minor = min;
    return 0;
  } catch (...) {
    return -2;
  }
}
