// SPDX-License-Identifier: Apache-2.0
//
// Xe2 GEMV pieces of the decode stages: block-FP8 (e4m3, 128x128 block
// scales) and 16-bit weight rows dotted with fp16 activations.
#pragma once
#include "sgl_kernel/stage/xe20/common.hpp"

namespace sgl_kernel::stage::xe20 {

inline uint32_t fp8x2_lo(uint32_t w) {  // bytes 0,2 -> two fp16 = e4m3 / 256
  uint32_t t = w & 0x00ff00ffu;
  return ((t & 0x00800080u) << 8) | ((t & 0x007f007fu) << 7);
}
inline uint32_t fp8x2_hi(uint32_t w) {  // bytes 1,3
  uint32_t t = (w >> 8) & 0x00ff00ffu;
  return ((t & 0x00800080u) << 8) | ((t & 0x007f007fu) << 7);
}
inline h2 as_h2(uint32_t u) {
  return sycl::bit_cast<h2>(u);
}

// 16 fp8 weights . 16 activations stored permuted (x0,x2,x1,x3 per 4) in SLM.
inline float dot16_fp8(sycl::uint4 w, const uint32_t* xp) {
  sycl::uint4 xa = *reinterpret_cast<const sycl::uint4*>(xp);
  sycl::uint4 xb = *reinterpret_cast<const sycl::uint4*>(xp + 4);
  h2 acc = as_h2(fp8x2_lo(w.x())) * as_h2(xa.x());
  acc += as_h2(fp8x2_hi(w.x())) * as_h2(xa.y());
  acc += as_h2(fp8x2_lo(w.y())) * as_h2(xa.z());
  acc += as_h2(fp8x2_hi(w.y())) * as_h2(xa.w());
  acc += as_h2(fp8x2_lo(w.z())) * as_h2(xb.x());
  acc += as_h2(fp8x2_hi(w.z())) * as_h2(xb.y());
  acc += as_h2(fp8x2_lo(w.w())) * as_h2(xb.z());
  acc += as_h2(fp8x2_hi(w.w())) * as_h2(xb.w());
  return (float)acc[0] + (float)acc[1];
}
// 16 fp8 weights . 16 activations in natural order (registers)
inline float dot16_fp8_nat(sycl::uint4 w, const sycl::vec<half, 16>& x) {
  h2 acc = as_h2(fp8x2_lo(w.x())) * h2(x[0], x[2]);
  acc += as_h2(fp8x2_hi(w.x())) * h2(x[1], x[3]);
  acc += as_h2(fp8x2_lo(w.y())) * h2(x[4], x[6]);
  acc += as_h2(fp8x2_hi(w.y())) * h2(x[5], x[7]);
  acc += as_h2(fp8x2_lo(w.z())) * h2(x[8], x[10]);
  acc += as_h2(fp8x2_hi(w.z())) * h2(x[9], x[11]);
  acc += as_h2(fp8x2_lo(w.w())) * h2(x[12], x[14]);
  acc += as_h2(fp8x2_hi(w.w())) * h2(x[13], x[15]);
  return (float)acc[0] + (float)acc[1];
}
// 8 16-bit (DT) weights . 8 activations in natural order (fp16 in SLM), fp32
template <typename DT>
inline float dot8_w16(sycl::uint4 w, const half* x) {
  const sycl::vec<uint16_t, 8> wh = sycl::bit_cast<sycl::vec<uint16_t, 8>>(w);
  const sycl::vec<half, 8> xv = *reinterpret_cast<const sycl::vec<half, 8>*>(x);
  float a = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i)
    a += f32<DT>(wh[i]) * (float)xv[i];
  return a;
}
template <int KCH>
inline void load_row(sycl::uint4 (&w)[KCH], const uint8_t* row, int lane) {
#pragma unroll
  for (int j = 0; j < KCH; ++j)
    w[j] = *reinterpret_cast<const sycl::uint4*>(row + j * 256 + lane * 16);
}
// The block scales of this lane's chunks of a K = KCH * 256 fp8 row: chunk j
// covers K bytes j * 256 + lane * 16 .. + 16, i.e. K block 2 j + lane / 8.
// srow: the row's [K / 128] scales.
template <int KCH>
inline void load_scales(float (&sc)[KCH], const float* srow, int lane) {
#pragma unroll
  for (int j = 0; j < KCH; ++j)
    sc[j] = srow[2 * j + (lane >> 3)];
}
// sum over a K = KCH * 256 fp8 row (registers) . x (permuted SLM), the lanes'
// chunks scaled by their K block, reduced over the sub-group: * 256 = output
template <int KCH>
inline float
dot_row_regs(sycl::sub_group sg, int lane, const sycl::uint4 (&w)[KCH], const uint32_t* xp, const float (&sc)[KCH]) {
  float sacc = 0.f;
#pragma unroll
  for (int j = 0; j < KCH; ++j)
    sacc += dot16_fp8(w[j], xp + (j * 256 + lane * 16) / 2) * sc[j];
  return sycl::reduce_over_group(sg, sacc, sycl::plus<float>());
}
// x[k] -> permuted slot (x0,x2,x1,x3 per 4): swap bits 0 and 1 of k.
inline void store_perm(half* xp, int k, half v) {
  xp[(k & ~3) | ((k & 1) << 1) | ((k >> 1) & 1)] = v;
}

}  // namespace sgl_kernel::stage::xe20
