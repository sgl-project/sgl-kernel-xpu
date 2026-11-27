#pragma once

#include <ATen/core/Tensor.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>

// Static GPU-architecture limit: per-work-item vector load/store width in bytes,
// keyed by SYCL_INTEL_TARGET (see CMakeLists.txt).
#if defined(SYCL_INTEL_TARGET) && SYCL_INTEL_TARGET == 20
inline constexpr int kArchMaxVecBytes = 16;
#elif defined(SYCL_INTEL_TARGET) && SYCL_INTEL_TARGET == 35
inline constexpr int kArchMaxVecBytes = 16;
#else
inline constexpr int kArchMaxVecBytes = 16;
#endif

inline constexpr int max_vec_size_for_bytes(size_t elem_size) {
  return static_cast<int>(kArchMaxVecBytes / elem_size);
}

template <typename T>
inline constexpr int max_vec_size() {
  return max_vec_size_for_bytes(sizeof(T));
}

// Helper for std::initializer_list to accept both `at::Tensor` and `std::optional<at::Tensor>`.
struct OptTensorRef {
  const at::Tensor* ptr = nullptr;
  OptTensorRef() = default;
  OptTensorRef(const at::Tensor& t) : ptr(t.defined() ? &t : nullptr) {}
  OptTensorRef(const std::optional<at::Tensor>& t) : ptr(t && t->defined() ? &*t : nullptr) {}
};

// Largest power-of-two V <= max_vec such that for every defined tensor:
//   1. data_ptr is aligned to V * element_size bytes;
//   2. all outer strides (dim < ndim - 1 with size(dim) > 1) are multiples of V;
//   3. when has_tail is false, innermost size(-1) is a multiple of V (covering 1D and batch == 1).
// Falls back to 1 when no wider width is valid.
inline int get_vec_size(int max_vec, std::initializer_list<OptTensorRef> tensors, bool has_tail = false) {
  int v = max_vec;
  for (const OptTensorRef& ref : tensors) {
    if (v <= 1) return 1;
    const at::Tensor* t = ref.ptr;
    if (t == nullptr) continue;

    const uintptr_t addr = reinterpret_cast<uintptr_t>(t->data_ptr());
    const size_t elem_size = t->element_size();
    while (v > 1 && addr % (v * elem_size) != 0)
      v >>= 1;

    const int64_t ndim = t->dim();
    uint64_t mask = (!has_tail && ndim > 0) ? static_cast<uint64_t>(t->size(ndim - 1)) : 0;
    for (int64_t d = 0; d < ndim - 1; ++d) {
      if (t->size(d) > 1) mask |= static_cast<uint64_t>(t->stride(d));
    }
    while (v > 1 && (mask & (v - 1)) != 0)
      v >>= 1;
  }
  return v > 1 ? v : 1;
}

template <typename T>
inline int get_vec_size(std::initializer_list<OptTensorRef> tensors, bool has_tail = false) {
  return get_vec_size(max_vec_size<T>(), tensors, has_tail);
}
