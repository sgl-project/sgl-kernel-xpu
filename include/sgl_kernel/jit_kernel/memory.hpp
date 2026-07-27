/**
 * SYCL Memory Utilities for SGLang JIT Kernels
 *
 * This file provides the aligned_vector type for vectorized memory access.
 */

#pragma once

#include <sycl/sycl.hpp>

namespace sgl {
namespace sycl {

// ============================================================================
// Aligned Vector Types (for vectorized memory access)
// ============================================================================

/**
 * aligned_vector: Properly aligned array for vectorized loads/stores
 *
 * This struct ensures that arrays of T elements are properly aligned for
 * vectorized memory operations. The alignas specifier guarantees that the
 * struct is aligned to N * sizeof(T) bytes, enabling efficient SIMD loads.
 *
 * Usage:
 *   using Vec = aligned_vector<float, 4>;  // 4-element float vector
 *   Vec* ptr = reinterpret_cast<Vec*>(data);
 *   Vec v = ptr[i];  // Vectorized load
 *
 * Note: This is a simpler alternative to sycl::vec for JIT kernels where
 * explicit control over memory layout is needed.
 */
template <typename T, int N>
struct alignas(N * sizeof(T)) aligned_vector {
  T data[N];

  aligned_vector() = default;

  T& operator[](int i) {
    return data[i];
  }
  const T& operator[](int i) const {
    return data[i];
  }
};

}  // namespace sycl
}  // namespace sgl
