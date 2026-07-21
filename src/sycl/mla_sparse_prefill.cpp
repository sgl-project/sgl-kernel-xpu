// DeepSeek V4 Sparse MLA prefill (dense gather + fused attention).
#ifndef SYCL_INTEL_TARGET
#define SYCL_INTEL_TARGET 20
#endif

#include <ATen/xpu/XPUContext.h>
#include <torch/all.h>

// Include torch/python.h (via flash_common_xpu.hpp) before any cute header, which
// defines a `printf` macro that otherwise breaks torch's pythoncapi_compat.h.
#include "cutlass/bfloat16.h"
#include "cutlass/kernel_hardware_info.h"
#include "kernels/mla_sparse/flash.h"
#include "kernels/mla_sparse/flash_common_xpu.hpp"
#include "kernels/mla_sparse/namespace_config.h"
#include "kernels/mla_sparse/utils.h"

namespace FLASH_NAMESPACE {

std::vector<at::Tensor> flash_mla_sparse_prefill_impl(
    const at::Tensor& q,        // [s_q, h_q, d_qk]
    const at::Tensor& kv,       // [s_kv, h_kv, d_qk]
    const at::Tensor& indices,  // [s_q, h_kv, topk]
    double sm_scale,
    int64_t d_v,
    const std::optional<at::Tensor>& attn_sink,    // [h_q]
    const std::optional<at::Tensor>& topk_length,  // [s_q]
    bool return_softmax_lse) {
  int s_q = q.size(0);
  int s_kv = kv.size(0);
  int h_q = q.size(1);
  int h_kv = kv.size(1);
  int d_qk = q.size(2);
  int topk = indices.size(2);
  bool have_topk_length = topk_length.has_value();

  TORCH_CHECK(q.dim() == 3, "q must be 3D");
  TORCH_CHECK(kv.dim() == 3, "kv must be 3D");
  TORCH_CHECK(indices.dim() == 3, "indices must be 3D");
  if (attn_sink.has_value()) {
    TORCH_CHECK(attn_sink.value().dim() == 1, "attn_sink must be 1D");
  }
  if (topk_length.has_value()) {
    TORCH_CHECK(topk_length.value().dim() == 1, "topk_length must be 1D");
  }

  TORCH_CHECK(d_qk == 576 || d_qk == 512, "d_qk must be 576 or 512");
  TORCH_CHECK(d_v == 512, "d_v must be 512");

  TORCH_CHECK(q.dtype() == torch::kBFloat16, "q must be bfloat16");
  TORCH_CHECK(kv.dtype() == torch::kBFloat16, "kv must be bfloat16");
  TORCH_CHECK(indices.dtype() == torch::kInt32, "indices must be int32");
  CHECK_DEVICE(q);
  CHECK_DEVICE(kv);
  CHECK_DEVICE(indices);
  if (attn_sink.has_value()) {
    TORCH_CHECK(attn_sink.value().dtype() == torch::kFloat32, "attn_sink must be float32");
    CHECK_DEVICE(attn_sink.value());
  }
  if (topk_length.has_value()) {
    TORCH_CHECK(topk_length.value().dtype() == torch::kInt32, "topk_length must be int32");
    CHECK_DEVICE(topk_length.value());
  }

  // Allocate results and buffers
  const c10::DeviceGuard device_guard(q.device());
  auto opts = q.options();

  at::Tensor out = torch::empty({s_q, h_q, static_cast<int64_t>(d_v)}, opts);
  at::Tensor lse = torch::empty({s_q, h_q}, opts.dtype(torch::kFloat));
  at::Tensor max_logits = torch::empty({s_q, h_q}, opts.dtype(torch::kFloat));

  // Chunk gathered_k along s_q to bound peak device memory.
  // gathered_k is [chunk, topk, d_qk] in bf16; cap the workspace at ~256 MB.
  TORCH_CHECK(topk > 0, "topk must be > 0");
  constexpr int64_t PREFILL_GATHERED_K_MAX_BYTES = 256LL * 1024 * 1024;
  const int64_t per_seq_gathered_bytes = static_cast<int64_t>(topk) * d_qk * 2;  // bf16 = 2 bytes
  int chunk_size = std::max(1, static_cast<int>(PREFILL_GATHERED_K_MAX_BYTES / per_seq_gathered_bytes));
  chunk_size = std::min(chunk_size, s_q);

  at::Tensor gathered_k = torch::empty({chunk_size, topk, d_qk}, opts);
  at::Tensor gathered_valid_mask = torch::empty({chunk_size, topk}, opts.dtype(torch::kInt));

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = q.device().index();
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
  TORCH_CHECK(hw_info.sm_count > 0, "Failed to query device multiprocessor count");

  XPUSparseAttnFwdParams params{
      s_q,
      s_kv,
      h_q,
      h_kv,
      d_qk,
      static_cast<int>(d_v),
      topk,
      static_cast<float>(sm_scale),
      static_cast<float>(sm_scale) * LOG_2_E,

      (cutlass::bfloat16_t*)q.data_ptr(),
      (cutlass::bfloat16_t*)kv.data_ptr(),
      (int*)indices.data_ptr(),

      get_optional_tensor_ptr<float>(attn_sink),
      get_optional_tensor_ptr<int>(topk_length),

      int64_stride_to_int(q.stride(0)),
      int64_stride_to_int(q.stride(1)),
      int64_stride_to_int(kv.stride(0)),
      int64_stride_to_int(kv.stride(1)),
      int64_stride_to_int(indices.stride(0)),
      int64_stride_to_int(indices.stride(1)),

      (cutlass::bfloat16_t*)gathered_k.data_ptr(),
      (int*)gathered_valid_mask.data_ptr(),

      int64_stride_to_int(gathered_k.stride(0)),
      int64_stride_to_int(gathered_k.stride(1)),
      int64_stride_to_int(gathered_valid_mask.stride(0)),

      (cutlass::bfloat16_t*)out.data_ptr(),
      (float*)max_logits.data_ptr(),
      (float*)lse.data_ptr(),
      true,

      hw_info.sm_count,

      chunk_size,
      at::xpu::getCurrentXPUStream().queue()};

  DISPATCH_HEAD_DIM(params.d_qk, HEAD_DIM_QK, [&] {
    DISPATCH_BOOLEAN_FLAG(have_topk_length, HAVE_TOPK_LENGTH, [&] {
      DISPATCH_BOOLEAN_FLAG(attn_sink.has_value(), HAS_ATTN_SINK, [&] {
        launch_sparse_mla_prefill_fwd_kernel<HEAD_DIM_QK, HAVE_TOPK_LENGTH, HAS_ATTN_SINK>(params);
      });
    });
  });

  if (return_softmax_lse) {
    return std::vector<at::Tensor>{out, max_logits, lse};
  }
  return std::vector<at::Tensor>{out};
}

}  // namespace FLASH_NAMESPACE

std::vector<at::Tensor> flash_mla_sparse_prefill(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    double sm_scale,
    int64_t d_v,
    const std::optional<at::Tensor>& attn_sink,
    const std::optional<at::Tensor>& topk_length,
    bool return_softmax_lse) {
  return FLASH_NAMESPACE::flash_mla_sparse_prefill_impl(
      q, kv, indices, sm_scale, d_v, attn_sink, topk_length, return_softmax_lse);
}
