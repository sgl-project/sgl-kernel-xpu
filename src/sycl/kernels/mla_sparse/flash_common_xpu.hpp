/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once
#include <limits>

// cute/util/print.hpp defines `printf` as a function-like macro
// (sycl::ext::oneapi::experimental::printf). If it is active when the torch /
// Python headers are parsed, it corrupts declarations such as
// `Py_GCC_ATTRIBUTE((format(printf, 2, 3)))`. Shield the torch includes.
#pragma push_macro("printf")
#undef printf

// Include these 2 headers instead of torch/extension.h since we don't need all of the torch headers.
#include <torch/nn/functional.h>
#include <torch/python.h>

#pragma pop_macro("printf")

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_xpu(), #x " must be on XPU")
#define CHECK_SHAPE(x, ...) \
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DTYPE(x, dtype) TORCH_CHECK(x.dtype() == dtype, #x " must have dtype " #dtype)

#define CHECK_OPTIONAL_DEVICE(x)                           \
  if (x.has_value()) {                                     \
    TORCH_CHECK(x.value().is_xpu(), #x " must be on XPU"); \
  }
#define CHECK_OPTIONAL_SHAPE(x, ...)                                                                               \
  if (x.has_value()) {                                                                                             \
    TORCH_CHECK(x.value().sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")"); \
  }
#define CHECK_OPTIONAL_CONTIGUOUS(x)                                  \
  if (x.has_value()) {                                                \
    TORCH_CHECK(x.value().is_contiguous(), #x " must be contiguous"); \
  }
#define CHECK_OPTIONAL_DTYPE(x, dt)                                   \
  if (x.has_value()) {                                                \
    TORCH_CHECK(x.value().dtype() == dt, #x " must have dtype " #dt); \
  }

//
// Convert int64_t stride to int32_t, with overflow check.
inline int int64_stride_to_int(int64_t orig_stride) {
  if (orig_stride > std::numeric_limits<int>::max()) {
    TORCH_CHECK(false, "Stride exceeds int32 limit: ", orig_stride);
  }
  return static_cast<int>(orig_stride);
}

// leverage from
// https://github.com/deepseek-ai/FlashMLA/blob/main/csrc/kerutils/include/kerutils/supplemental/torch_tensors.h#L30

// Get the pointer of the given tensor
// Return (PtrT*)tensor.data_ptr() if the tensor has a backend storage, nullptr otherwise
template <typename PtrT>
static inline PtrT* get_tensor_ptr(const at::Tensor& tensor) {
  if (tensor.has_storage()) {
    return (PtrT*)tensor.data_ptr();
  } else {
    return nullptr;
  }
}

// Get the pointer of the given tensor or optional tensor
// Return (PtrT*)tensor.data_ptr() if tensor_or_opt has value and points to a valid tensor, return nullptr otherwise
template <typename PtrT, typename T>
static inline PtrT* get_optional_tensor_ptr(const T& tensor_or_opt) {
  if constexpr (std::is_same<T, at::Tensor>::value) {
    return get_tensor_ptr<PtrT>(tensor_or_opt);
  } else {
    if (tensor_or_opt.has_value()) {
      return get_tensor_ptr<PtrT>(*tensor_or_opt);
    } else {
      return nullptr;
    }
  }
}
