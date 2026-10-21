#pragma once

#include "sycl/xe35/fmha/xe_fmha_fwd_decode_runner.hpp"

SGL_KERNEL_EXPORT Xe35DispatchFn_xe35 dispatch_xe35_decode_64_bf16_xe35;
SGL_KERNEL_EXPORT Xe35DispatchFn_xe35 dispatch_xe35_decode_64_fp8_e4m3_xe35;
SGL_KERNEL_EXPORT Xe35DispatchFn_xe35 dispatch_xe35_decode_64_fp8_e5m2_xe35;
SGL_KERNEL_EXPORT Xe35DispatchFn_xe35 dispatch_xe35_decode_128_bf16_xe35;
SGL_KERNEL_EXPORT Xe35DispatchFn_xe35 dispatch_xe35_decode_128_fp8_e4m3_xe35;
SGL_KERNEL_EXPORT Xe35DispatchFn_xe35 dispatch_xe35_decode_128_fp8_e5m2_xe35;
SGL_KERNEL_EXPORT Xe35DispatchFn_xe35 dispatch_xe35_decode_256_bf16_xe35;
SGL_KERNEL_EXPORT Xe35DispatchFn_xe35 dispatch_xe35_decode_256_fp8_e4m3_xe35;
SGL_KERNEL_EXPORT Xe35DispatchFn_xe35 dispatch_xe35_decode_256_fp8_e5m2_xe35;

SGL_KERNEL_EXPORT Xe35DispatchFn_xe35 dispatch_xe35_decode_nopage_64_xe35;
SGL_KERNEL_EXPORT Xe35DispatchFn_xe35 dispatch_xe35_decode_nopage_128_xe35;

inline Xe35DispatchFn_xe35* get_xe35_decode_dispatch_xe35(int head_dim, at::ScalarType dtype) {
  if (head_dim == 64) {
    if (dtype == at::ScalarType::Float8_e4m3fn) return &dispatch_xe35_decode_64_fp8_e4m3_xe35;
    if (dtype == at::ScalarType::Float8_e5m2) return &dispatch_xe35_decode_64_fp8_e5m2_xe35;
    return &dispatch_xe35_decode_64_bf16_xe35;
  }
  if (head_dim == 128) {
    if (dtype == at::ScalarType::Float8_e4m3fn) return &dispatch_xe35_decode_128_fp8_e4m3_xe35;
    if (dtype == at::ScalarType::Float8_e5m2) return &dispatch_xe35_decode_128_fp8_e5m2_xe35;
    return &dispatch_xe35_decode_128_bf16_xe35;
  }
  if (dtype == at::ScalarType::Float8_e4m3fn) return &dispatch_xe35_decode_256_fp8_e4m3_xe35;
  if (dtype == at::ScalarType::Float8_e5m2) return &dispatch_xe35_decode_256_fp8_e5m2_xe35;
  return &dispatch_xe35_decode_256_bf16_xe35;
}
