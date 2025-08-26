#pragma once

#include <ATen/ATen.h>
#include <ATen/core/Array.h>
#include <ATen/detail/FunctionTraits.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/TypeCast.h>

#include <cstdint>
#include <type_traits>

#include "Utils.h"

static inline int preferred_vector_width(at::DeviceIndex dev_id, int elem_sz) {
#define PRIVATE_CASE_SIZE(SIZE, TYPE)                                       \
  case SIZE: {                                                              \
    static_assert(sizeof(TYPE) == SIZE, "the TYPE size is not SIZE bytes"); \
    ret = dpcppPrefVectorWidth<TYPE>(dev_id);                               \
    break;                                                                  \
  }
  size_t ret;
  switch (elem_sz) {
    PRIVATE_CASE_SIZE(1, char);
    PRIVATE_CASE_SIZE(2, short);
    PRIVATE_CASE_SIZE(4, int);
    PRIVATE_CASE_SIZE(8, int64_t);
    default:
      // no vectorize
      ret = 1;
  }
  return ret;
}

// aligned vector generates vectorized load/store on XPU
template <int N_BYTES>
struct aligned_element {};
template <>
struct aligned_element<1> {
  using element_type = uint8_t;
};

template <>
struct aligned_element<2> {
  using element_type = uint16_t;
};

template <>
struct aligned_element<4> {
  using element_type = uint32_t;
};

template <>
struct aligned_element<8> {
  using element_type = uint64_t;
};

struct alignas(16) uint128_t {
  char data[16];
};

template <>
struct aligned_element<16> {
  using element_type = uint128_t;
};

template <typename scalar_t, int vec_size>
struct aligned_vector {
  using element_type = typename aligned_element<sizeof(scalar_t)>::element_type;
  using type = sycl::vec<element_type, vec_size>;
};

template <typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector_loop {
  scalar_t val[vec_size];

  scalar_t& operator[](int index) {
    return val[index];
  }

  scalar_t const& operator[](int index) const {
    return val[index];
  }
};

template <typename scalar_t>
inline int can_vectorize_up_to(at::DeviceIndex dev_id, char* pointer) {
  int elem_size = sizeof(scalar_t);
  int preferred_width = preferred_vector_width(dev_id, elem_size);
  uint64_t address = reinterpret_cast<uint64_t>(pointer);
  constexpr int vec2_alignment = std::alignment_of<typename aligned_vector<scalar_t, 2>::type>::value;
  constexpr int vec4_alignment = std::alignment_of<typename aligned_vector<scalar_t, 4>::type>::value;
  constexpr int vec8_alignment = std::alignment_of<typename aligned_vector<scalar_t, 8>::type>::value;
  constexpr int vec16_alignment = std::alignment_of<typename aligned_vector<scalar_t, 16>::type>::value;
  if (address % vec16_alignment == 0) {
    return std::min<int>(preferred_width, 16);
  } else if (address % vec8_alignment == 0) {
    return std::min<int>(preferred_width, 8);
  } else if (address % vec4_alignment == 0) {
    return std::min<int>(preferred_width, 4);
  } else if (address % vec2_alignment == 0) {
    return std::min<int>(preferred_width, 2);
  }
  return 1;
}

template <typename... Args>
int get_min_vec_size(int vec_size, Args*... args) {
  auto limit_func = [](int vec_size, auto* data) {
    if (!data) return vec_size;
    return can_vectorize_up_to<std::remove_pointer_t<decltype(data)>>(
        dpcppGetDeviceIdOfCurrentQueue(), reinterpret_cast<char*>(data));
  };
  return get_min(limit_func, vec_size, args...);
}