#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"

namespace at::native::xpu {

namespace CompressPlanImpl {

// Kernel for plan_compress_decode
struct CompressDecodeKernel {
  void operator()(sycl::nd_item<1> item) const {
    int32_t idx = item.get_global_id(0);
    if (idx >= batch_size_) return;

    int64_t rid = req_pool_indices_[idx];
    int64_t seq_len = seq_lens_[idx];
    int32_t pos1 = seq_len - 1;
    int32_t pos0 = sycl::max(pos1 - compress_ratio_, 0);

    int64_t loc1;
    int64_t loc0;
    if (compress_ratio_ == 128) {
      // DeepSeek V4 c128 path uses request-local state indexing.
      loc1 = rid * ring_size_ + (static_cast<int64_t>(pos1) % ring_size_);
      loc0 = rid * ring_size_ + (static_cast<int64_t>(pos0) % ring_size_);
    } else {
      const int32_t* req_to_token_row = req_to_token_ + rid * req_to_token_stride_;

      // Look up full token indices
      int64_t raw1 = static_cast<int64_t>(req_to_token_row[pos1]);
      int64_t raw0 = static_cast<int64_t>(req_to_token_row[pos0]);

      // Convert to swa indices
      int64_t swa1 = full_to_state_[raw1];
      int64_t swa0 = full_to_state_[raw0];

      // Compute ring buffer locations
      loc1 = (swa1 / swa_page_size_) * ring_size_ + (swa1 % ring_size_);
      loc0 = (swa0 / swa_page_size_) * ring_size_ + (swa0 % ring_size_);
    }

    // Pack into output: [seq_len, loc1, loc0/compress_ratio, loc1/compress_ratio]
    int32_t* output = output_i32_ + idx * 4;
    output[0] = seq_len;
    output[1] = static_cast<int32_t>(loc1);
    output[2] = static_cast<int32_t>(loc0 / static_cast<int64_t>(compress_ratio_));
    output[3] = static_cast<int32_t>(loc1 / static_cast<int64_t>(compress_ratio_));
  }

  const int64_t* req_pool_indices_;
  const int64_t* seq_lens_;
  const int32_t* req_to_token_;
  const int64_t* full_to_state_;
  int32_t* output_i32_;
  int64_t swa_page_size_;
  int64_t ring_size_;
  int32_t compress_ratio_;
  int64_t req_to_token_stride_;
  int32_t batch_size_;
};

}  // namespace CompressPlanImpl

// SYCL helper to launch kernels
inline sycl::nd_range<1> get_1d_range(int32_t size) {
  constexpr int32_t local_size = 256;
  return sycl::nd_range<1>(
      sycl::range<1>((size + local_size - 1) / local_size * local_size),
      sycl::range<1>(local_size));
}

// XPU wrapper for plan_compress_decode
torch::Tensor plan_compress_decode(
    torch::Tensor req_pool_indices,
    torch::Tensor req_to_token,
  torch::Tensor full_to_state,
    torch::Tensor seq_lens,
    int64_t compress_ratio,
    int64_t swa_page_size,
    int64_t ring_size) {
  TORCH_CHECK(req_pool_indices.dtype() == torch::kInt64);
  TORCH_CHECK(req_to_token.dtype() == torch::kInt32);
  TORCH_CHECK(full_to_state.dtype() == torch::kInt64);
  TORCH_CHECK(seq_lens.dtype() == torch::kInt64);
  TORCH_CHECK(req_to_token.dim() == 2, "req_to_token must be a 2D tensor");
  TORCH_CHECK(req_to_token.stride(1) == 1, "req_to_token must be contiguous in the last dim");
  TORCH_CHECK(compress_ratio > 0, "compress_ratio must be > 0");

  int32_t batch_size = static_cast<int32_t>(seq_lens.numel());
  if (batch_size == 0) {
    return torch::empty({0, 16}, req_pool_indices.options().dtype(torch::kUInt8));
  }

  int64_t req_to_token_stride = req_to_token.stride(0);
  auto output = torch::empty(
      {batch_size, 16}, req_pool_indices.options().dtype(torch::kUInt8));
  auto output_i32 = reinterpret_cast<int32_t*>(output.data_ptr<uint8_t>());

  auto queue = c10::xpu::getCurrentXPUStream().queue();
  queue.submit([&](sycl::handler& cgh) {
    CompressPlanImpl::CompressDecodeKernel kernel{
        req_pool_indices.data_ptr<int64_t>(),
        seq_lens.data_ptr<int64_t>(),
        req_to_token.data_ptr<int32_t>(),
        full_to_state.data_ptr<int64_t>(),
        output_i32,
        swa_page_size,
        ring_size,
        static_cast<int32_t>(compress_ratio),
        req_to_token_stride,
        batch_size};
    cgh.parallel_for(get_1d_range(batch_size), kernel);
  });

  return output;
}

}  // namespace at::native::xpu
