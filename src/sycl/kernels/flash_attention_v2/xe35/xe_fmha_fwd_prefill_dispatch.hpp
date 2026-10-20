#pragma once

#include "sycl/kernels/flash_attention_v2/xe35/xe_fmha_fwd_prefill_runner.hpp"

Xe35DispatchFn dispatch_xe35_prefill_64_bf16;
Xe35DispatchFn dispatch_xe35_prefill_64_fp8_e4m3;
Xe35DispatchFn dispatch_xe35_prefill_64_fp8_e5m2;
Xe35DispatchFn dispatch_xe35_prefill_128_bf16;
Xe35DispatchFn dispatch_xe35_prefill_128_fp8_e4m3;
Xe35DispatchFn dispatch_xe35_prefill_128_fp8_e5m2;
Xe35DispatchFn dispatch_xe35_prefill_256_bf16;
Xe35DispatchFn dispatch_xe35_prefill_256_fp8_e4m3;
Xe35DispatchFn dispatch_xe35_prefill_256_fp8_e5m2;

Xe35DispatchFn dispatch_xe35_prefill_nopage_64;
Xe35DispatchFn dispatch_xe35_prefill_nopage_128;

inline Xe35DispatchFn* get_xe35_prefill_dispatch(int head_dim, at::ScalarType dtype) {
  if (head_dim == 64) {
    if (dtype == at::ScalarType::Float8_e4m3fn) return &dispatch_xe35_prefill_64_fp8_e4m3;
    if (dtype == at::ScalarType::Float8_e5m2) return &dispatch_xe35_prefill_64_fp8_e5m2;
    return &dispatch_xe35_prefill_64_bf16;
  }
  if (head_dim == 128) {
    if (dtype == at::ScalarType::Float8_e4m3fn) return &dispatch_xe35_prefill_128_fp8_e4m3;
    if (dtype == at::ScalarType::Float8_e5m2) return &dispatch_xe35_prefill_128_fp8_e5m2;
    return &dispatch_xe35_prefill_128_bf16;
  }
  if (dtype == at::ScalarType::Float8_e4m3fn) return &dispatch_xe35_prefill_256_fp8_e4m3;
  if (dtype == at::ScalarType::Float8_e5m2) return &dispatch_xe35_prefill_256_fp8_e5m2;
  return &dispatch_xe35_prefill_256_bf16;
}
