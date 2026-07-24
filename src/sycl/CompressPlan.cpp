#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"

namespace at::native::xpu {

struct alignas(16) DecodePlan {
  uint32_t seq_len;
  int32_t write_loc;
  int32_t read_page_0;
  int32_t read_page_1;
};
static_assert(sizeof(DecodePlan) == 16, "DecodePlan must be 16 bytes");

namespace CompressPlanImpl {

// Kernel for plan_compress_decode
struct CompressDecodeKernel {
  void operator()(sycl::nd_item<1> item) const {
    uint32_t idx = item.get_global_id(0);
    if (idx >= batch_size) return;

    int64_t rid = rid_ptr[idx];
    int32_t seq_len = static_cast<int32_t>(seq_ptr[idx]);
    int32_t position_1 = seq_len - 1;
    int32_t position_0 = sycl::max(position_1 - compress_ratio, 0);

    const auto compute_loc = [&](int32_t swa_loc) {
      int32_t swa_page = swa_loc / swa_page_size;
      int32_t ring_offset = swa_loc % ring_size;
      return swa_page * ring_size + ring_offset;
    };
    const auto compute_c128_loc = [&](int64_t rid_, int32_t position) {
      return static_cast<int32_t>(rid_ * ring_size + position % ring_size);
    };

    int32_t write_loc;
    int32_t read_page_0;
    int32_t read_page_1;
    if (compress_ratio == 128) {
      write_loc = compute_c128_loc(rid, position_1);
      read_page_0 = compute_c128_loc(rid, position_0) / 128;
      read_page_1 = write_loc / 128;
    } else {
      const int32_t* mapping = r2t_ptr + rid * stride_r2t;

      // Look up full token indices
      const auto raw_loc_0 = mapping[position_0];
      const auto raw_loc_1 = mapping[position_1];

      // Convert to swa indices
      const auto state_loc_0 = f2s_ptr[raw_loc_0];
      const auto state_loc_1 = f2s_ptr[raw_loc_1];
      // Compute ring buffer locations
      write_loc = static_cast<int32_t>(compute_loc(state_loc_1));
      read_page_0 = static_cast<int32_t>(compute_loc(state_loc_0) / compress_ratio);
      read_page_1 = static_cast<int32_t>(write_loc / compress_ratio);
    }

    // Pack into output: [seq_len, write_loc, read_page_0, read_page_1]
    plan_d[idx] = DecodePlan{
        static_cast<uint32_t>(seq_len),
        write_loc,
        read_page_0,
        read_page_1,
    };
  }

  const int64_t* rid_ptr;
  const int64_t* seq_ptr;
  const int32_t* r2t_ptr;
  const int64_t* f2s_ptr;
  DecodePlan* plan_d;
  int32_t swa_page_size;
  int32_t ring_size;
  int32_t compress_ratio;
  int64_t stride_r2t;
  uint32_t batch_size;
};

}  // namespace CompressPlanImpl

// SYCL helper to launch kernels
inline sycl::nd_range<1> get_1d_range(int32_t size) {
  constexpr int32_t local_size = 256;
  return sycl::nd_range<1>(
      sycl::range<1>((size + local_size - 1) / local_size * local_size), sycl::range<1>(local_size));
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
  TORCH_CHECK(
      req_pool_indices.is_xpu() && req_pool_indices.dtype() == torch::kInt64,
      "req_pool_indices must be an int64 XPU tensor");
  TORCH_CHECK(
      req_to_token.is_xpu() && req_to_token.dtype() == torch::kInt32, "req_to_token must be an int32 XPU tensor");
  TORCH_CHECK(
      full_to_state.is_xpu() && full_to_state.dtype() == torch::kInt64, "full_to_state must be an int64 XPU tensor");
  TORCH_CHECK(seq_lens.is_xpu() && seq_lens.dtype() == torch::kInt64, "seq_lens must be an int64 XPU tensor");
  TORCH_CHECK(req_to_token.dim() == 2, "req_to_token must be a 2D tensor");
  TORCH_CHECK(req_to_token.stride(1) == 1, "req_to_token must be contiguous in the last dim");
  TORCH_CHECK(compress_ratio > 0, "compress_ratio must be > 0");

  uint32_t batch_size = static_cast<uint32_t>(seq_lens.numel());
  if (batch_size == 0) {
    return torch::empty({0, static_cast<int64_t>(sizeof(DecodePlan))}, req_pool_indices.options().dtype(torch::kUInt8));
  }

  auto output = torch::empty(
      {static_cast<int64_t>(batch_size), static_cast<int64_t>(sizeof(DecodePlan))},
      req_pool_indices.options().dtype(torch::kUInt8));

  auto queue = c10::xpu::getCurrentXPUStream().queue();
  queue.submit([&](sycl::handler& cgh) {
    CompressPlanImpl::CompressDecodeKernel kernel{
        req_pool_indices.data_ptr<int64_t>(),
        seq_lens.data_ptr<int64_t>(),
        req_to_token.data_ptr<int32_t>(),
        full_to_state.data_ptr<int64_t>(),
        reinterpret_cast<DecodePlan*>(output.data_ptr<uint8_t>()),
        static_cast<int32_t>(swa_page_size),
        static_cast<int32_t>(ring_size),
        static_cast<int32_t>(compress_ratio),
        req_to_token.size(1),
        batch_size};
    cgh.parallel_for(get_1d_range(batch_size), kernel);
  });

  return output;
}

}  // namespace at::native::xpu
