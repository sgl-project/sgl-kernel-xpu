#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdint>
#include <optional>
#include <sycl/sycl.hpp>
#include <type_traits>

#include "sgl_kernel_export.h"

namespace at::native::xpu {

namespace {

constexpr size_t kLocalSize = 256;

template <typename ReqIndexT, typename StartLocT, typename SeqLenT>
struct DSV4ExpandPrefillKernel {
  [[sycl::reqd_sub_group_size(32)]] void operator()(sycl::nd_item<1> item) const {
    const int64_t row = static_cast<int64_t>(item.get_group(0));
    const int64_t lane = static_cast<int64_t>(item.get_local_id(0));
    int64_t start = 0;
    if (extend_start_loc != nullptr) {
      start = static_cast<int64_t>(extend_start_loc[row]);
    } else {
      int64_t partial = 0;
      for (int64_t i = lane; i < row; i += static_cast<int64_t>(kLocalSize)) {
        partial += static_cast<int64_t>(extend_seq_lens[i]);
      }
      start = sycl::reduce_over_group(item.get_group(), partial, sycl::plus<int64_t>());
    }

    const int64_t extend_len = sycl::max(static_cast<int64_t>(extend_seq_lens[row]), int64_t{0});
    const int64_t begin = sycl::min(sycl::max(start, int64_t{0}), num_tokens);
    const int64_t end = sycl::min(sycl::max(start + extend_len, int64_t{0}), num_tokens);
    const int64_t causal_begin = static_cast<int64_t>(seq_lens[row]) - static_cast<int64_t>(extend_seq_lens[row]) + 1;
    const ReqIndexT req = req_pool_indices[row];

    for (int64_t token = begin + lane; token < end; token += static_cast<int64_t>(kLocalSize)) {
      seq_lens_causal[token] = static_cast<int32_t>(causal_begin + token - start);
      req_pool_indices_repeated[token] = req;
    }

    if (row == batch_size - 1) {
      for (int64_t token = num_tokens + lane; token < padded_num_tokens; token += static_cast<int64_t>(kLocalSize)) {
        seq_lens_causal[token] = 1;
        req_pool_indices_repeated[token] = req;
      }
    }
  }

  const ReqIndexT* req_pool_indices;
  const SeqLenT* seq_lens;
  const int32_t* extend_seq_lens;
  const StartLocT* extend_start_loc;
  int32_t* seq_lens_causal;
  ReqIndexT* req_pool_indices_repeated;
  int64_t batch_size;
  int64_t num_tokens;
  int64_t padded_num_tokens;
};

template <typename ReqIndexT, typename StartLocT, typename SeqLenT>
void launch_dsv4_expand_prefill(
    sycl::queue& queue,
    const ReqIndexT* req_pool_indices,
    const SeqLenT* seq_lens,
    const int32_t* extend_seq_lens,
    const StartLocT* extend_start_loc,
    int32_t* seq_lens_causal,
    ReqIndexT* req_pool_indices_repeated,
    int64_t batch_size,
    int64_t num_tokens,
    int64_t padded_num_tokens) {
  DSV4ExpandPrefillKernel<ReqIndexT, StartLocT, SeqLenT> kernel{
      req_pool_indices,
      seq_lens,
      extend_seq_lens,
      extend_start_loc,
      seq_lens_causal,
      req_pool_indices_repeated,
      batch_size,
      num_tokens,
      padded_num_tokens};
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(batch_size) * kLocalSize), sycl::range<1>(kLocalSize)),
        kernel);
  });
}

}  // namespace

SGL_KERNEL_EXPORT void dsv4_expand_prefill_causally_out(
    torch::Tensor req_pool_indices,
    torch::Tensor seq_lens,
    torch::Tensor extend_seq_lens,
    std::optional<torch::Tensor> extend_start_loc,
    torch::Tensor seq_lens_causal,
    torch::Tensor req_pool_indices_repeated,
    int64_t num_tokens,
    int64_t padded_num_tokens) {
  TORCH_CHECK(
      req_pool_indices.is_xpu() &&
          (req_pool_indices.scalar_type() == torch::kInt32 || req_pool_indices.scalar_type() == torch::kInt64) &&
          req_pool_indices.dim() == 1 && req_pool_indices.is_contiguous(),
      "req_pool_indices must be a contiguous 1D int32 or int64 XPU tensor");
  TORCH_CHECK(
      seq_lens.is_xpu() && (seq_lens.scalar_type() == torch::kInt32 || seq_lens.scalar_type() == torch::kInt64) &&
          seq_lens.dim() == 1 && seq_lens.is_contiguous(),
      "seq_lens must be a contiguous 1D int32 or int64 XPU tensor");
  TORCH_CHECK(
      extend_seq_lens.is_xpu() && extend_seq_lens.scalar_type() == torch::kInt32 && extend_seq_lens.dim() == 1 &&
          extend_seq_lens.is_contiguous(),
      "extend_seq_lens must be a contiguous 1D int32 XPU tensor");
  TORCH_CHECK(
      seq_lens_causal.is_xpu() && seq_lens_causal.scalar_type() == torch::kInt32 && seq_lens_causal.dim() == 1 &&
          seq_lens_causal.is_contiguous(),
      "seq_lens_causal must be a contiguous 1D int32 XPU tensor");
  TORCH_CHECK(
      req_pool_indices_repeated.is_xpu() && req_pool_indices_repeated.scalar_type() == req_pool_indices.scalar_type() &&
          req_pool_indices_repeated.dim() == 1 && req_pool_indices_repeated.is_contiguous(),
      "req_pool_indices_repeated must be a contiguous 1D XPU tensor matching req_pool_indices dtype");

  const auto device = req_pool_indices.device();
  const int64_t batch_size = req_pool_indices.size(0);
  TORCH_CHECK(seq_lens.device() == device, "seq_lens must be on the same device as req_pool_indices");
  TORCH_CHECK(extend_seq_lens.device() == device, "extend_seq_lens must be on the same device as req_pool_indices");
  TORCH_CHECK(seq_lens_causal.device() == device, "seq_lens_causal must be on the same device as req_pool_indices");
  TORCH_CHECK(
      req_pool_indices_repeated.device() == device,
      "req_pool_indices_repeated must be on the same device as req_pool_indices");
  TORCH_CHECK(seq_lens.size(0) == batch_size, "seq_lens must have one entry per request");
  TORCH_CHECK(extend_seq_lens.size(0) == batch_size, "extend_seq_lens must have one entry per request");

  if (extend_start_loc.has_value()) {
    const auto& tensor = *extend_start_loc;
    TORCH_CHECK(
        tensor.is_xpu() && (tensor.scalar_type() == torch::kInt32 || tensor.scalar_type() == torch::kInt64) &&
            tensor.dim() == 1 && tensor.is_contiguous(),
        "extend_start_loc must be a contiguous 1D int32 or int64 XPU tensor");
    TORCH_CHECK(tensor.device() == device, "extend_start_loc must be on the same device as req_pool_indices");
    TORCH_CHECK(tensor.size(0) == batch_size, "extend_start_loc must have one entry per request");
  }

  TORCH_CHECK(num_tokens >= 0, "num_tokens must be non-negative");
  TORCH_CHECK(padded_num_tokens >= num_tokens, "padded_num_tokens must be at least num_tokens");
  TORCH_CHECK(seq_lens_causal.numel() == padded_num_tokens, "seq_lens_causal length must equal padded_num_tokens");
  TORCH_CHECK(
      req_pool_indices_repeated.numel() == padded_num_tokens,
      "req_pool_indices_repeated length must equal padded_num_tokens");
  TORCH_CHECK(batch_size > 0 || padded_num_tokens == 0, "a non-empty padded output requires at least one request");

  at::assert_no_internal_overlap(seq_lens_causal);
  at::assert_no_internal_overlap(req_pool_indices_repeated);
  at::assert_no_overlap(seq_lens_causal, req_pool_indices);
  at::assert_no_overlap(seq_lens_causal, seq_lens);
  at::assert_no_overlap(seq_lens_causal, extend_seq_lens);
  at::assert_no_overlap(req_pool_indices_repeated, req_pool_indices);
  at::assert_no_overlap(req_pool_indices_repeated, seq_lens);
  at::assert_no_overlap(req_pool_indices_repeated, extend_seq_lens);
  at::assert_no_overlap(seq_lens_causal, req_pool_indices_repeated);
  if (extend_start_loc.has_value()) {
    at::assert_no_overlap(seq_lens_causal, *extend_start_loc);
    at::assert_no_overlap(req_pool_indices_repeated, *extend_start_loc);
  }

  if (batch_size == 0) return;

  c10::DeviceGuard device_guard(device);
  auto& queue = c10::xpu::getCurrentXPUStream().queue();
  const auto launch = [&](auto req_ptr, auto start_ptr, auto seq_lens_ptr) {
    launch_dsv4_expand_prefill(
        queue,
        req_ptr,
        seq_lens_ptr,
        extend_seq_lens.data_ptr<int32_t>(),
        start_ptr,
        seq_lens_causal.data_ptr<int32_t>(),
        req_pool_indices_repeated.data_ptr<std::remove_pointer_t<decltype(req_ptr)>>(),
        batch_size,
        num_tokens,
        padded_num_tokens);
  };
  const auto dispatch_seq_lens = [&](auto req_ptr, auto start_ptr) {
    if (seq_lens.scalar_type() == torch::kInt32) {
      launch(req_ptr, start_ptr, seq_lens.data_ptr<int32_t>());
    } else {
      launch(req_ptr, start_ptr, seq_lens.data_ptr<int64_t>());
    }
  };
  const auto dispatch_start = [&](auto req_ptr) {
    if (!extend_start_loc.has_value()) {
      dispatch_seq_lens(req_ptr, static_cast<const int32_t*>(nullptr));
    } else if (extend_start_loc->scalar_type() == torch::kInt32) {
      dispatch_seq_lens(req_ptr, extend_start_loc->data_ptr<int32_t>());
    } else {
      dispatch_seq_lens(req_ptr, extend_start_loc->data_ptr<int64_t>());
    }
  };
  if (req_pool_indices.scalar_type() == torch::kInt32) {
    dispatch_start(req_pool_indices.data_ptr<int32_t>());
  } else {
    dispatch_start(req_pool_indices.data_ptr<int64_t>());
  }
}

}  // namespace at::native::xpu
