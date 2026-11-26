#pragma once

template <typename T, int N>
struct alignas(sizeof(T) * N) aligned_vec {
  T val[N];

  T& operator[](int index) {
    return val[index];
  }

  const T& operator[](int index) const {
    return val[index];
  }
};
