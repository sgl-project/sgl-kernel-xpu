// SPDX-License-Identifier: Apache-2.0
//
// Xe2 DPAS and 2D block-load primitives of the decode stages.
#pragma once
#include "sgl_kernel/stage/xe20/common.hpp"

namespace sgl_kernel::stage::xe20 {

// ---- Xe2 DPAS + 2D block loads (inline vISA / IGC builtins, as the sycl-tla
// arch primitives): joint_matrix element access (get_wi_data) does not build
// with IGC 2.38.
#if defined(__clang__)
typedef float v8f __attribute__((ext_vector_type(8)));
typedef unsigned int v8u __attribute__((ext_vector_type(8)));
typedef unsigned int v4u __attribute__((ext_vector_type(4)));
typedef int v2i __attribute__((ext_vector_type(2)));
#else
// the host compiler pass (g++) parses the kernels too: a stand-in with the
// operations they use
template <typename T, int N>
struct hvec {
  T v[N];
  hvec() = default;
  explicit hvec(T x) {
    for (int i = 0; i < N; ++i)
      v[i] = x;
  }
  T& operator[](int i) {
    return v[i];
  }
  const T& operator[](int i) const {
    return v[i];
  }
  hvec& operator+=(const hvec& o) {
    for (int i = 0; i < N; ++i)
      v[i] += o.v[i];
    return *this;
  }
  hvec operator*(T s) const {
    hvec r;
    for (int i = 0; i < N; ++i)
      r.v[i] = v[i] * s;
    return r;
  }
};
using v8f = hvec<float, 8>;
using v8u = hvec<unsigned int, 8>;
using v4u = hvec<unsigned int, 4>;
using v2i = hvec<int, 2>;
#endif
#ifdef __SYCL_DEVICE_ONLY__
extern "C" SYCL_EXTERNAL v8u __builtin_IB_subgroup_block_read_flat_transpose_u32_k8(
    intptr_t base, int width_m1, int height_m1, int pitch_m1, v2i xy);
extern "C" SYCL_EXTERNAL v8u __builtin_IB_subgroup_block_read_flat_transform_u16_k16(
    intptr_t base, int width_m1, int height_m1, int pitch_m1, v2i xy);
#endif
// d[8 rows x 16 cols] (lane = column, element i = row i) += a[8 x 16 half] .
// b[16 x 16 half]
//   a: dpas A layout, lane l element j = (A[2j + l/8][2(l%8)], A[2j +
//   l/8][2(l%8) + 1]) b: VNNI, lane n element p = (B[2p][n], B[2p + 1][n])
inline void dpas_hf(v8f& d, const v4u& a, const v8u& b) {
#ifdef __SYCL_DEVICE_ONLY__
  asm("{\n"
      ".decl DST     v_type=G type=f num_elts=128 alias=<%0,0>\n"
      ".decl SRC1_UD v_type=G type=UD num_elts=128 alias=<%2,0>\n"
      ".decl SRC2_UD v_type=G type=UD num_elts=64 alias=<%1,0>\n"
      "dpas.hf.hf.8.8 (M1, 16) DST.0 DST.0 SRC1_UD.0 SRC2_UD(0,0)\n"
      "}\n"
      : "+rw"(d)
      : "rw"(a), "rw"(b));
#else
  (void)d;
  (void)a;
  (void)b;
#endif
}
// the same for DT operands (attention: q, P, K and V in the KV cache dtype)
template <typename DT>
inline void dpas16(v8f& d, const v4u& a, const v8u& b) {
  if constexpr (std::is_same_v<DT, half>) {
    dpas_hf(d, a, b);
  } else {
#ifdef __SYCL_DEVICE_ONLY__
    asm("{\n"
        ".decl DST     v_type=G type=f num_elts=128 alias=<%0,0>\n"
        ".decl SRC1_UD v_type=G type=UD num_elts=128 alias=<%2,0>\n"
        ".decl SRC2_UD v_type=G type=UD num_elts=64 alias=<%1,0>\n"
        "dpas.bf.bf.8.8 (M1, 16) DST.0 DST.0 SRC1_UD.0 SRC2_UD(0,0)\n"
        "}\n"
        : "+rw"(d)
        : "rw"(a), "rw"(b));
#else
    (void)d;
    (void)a;
    (void)b;
#endif
  }
}

}  // namespace sgl_kernel::stage::xe20
